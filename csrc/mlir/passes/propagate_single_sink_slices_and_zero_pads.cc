/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "csrc/mlir/passes/propagate_single_sink_slices_and_zero_pads.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

// Returns a static ranked tensor type if `val` has one, or nullptr otherwise.
[[nodiscard]] mlir::RankedTensorType GetStaticRankedType(mlir::Value val) {
  auto type = llvm::dyn_cast<mlir::RankedTensorType>(val.getType());
  if (!type || !type.hasStaticShape()) {
    return nullptr;
  }
  return type;
}

// Returns true if `val` is a splat zero constant or a broadcast of one.
[[nodiscard]] bool IsZeroSplatTensor(mlir::Value val) {
  if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    return IsZeroSplatTensor(bcast.getOperand());
  }
  if (auto reshape = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    return IsZeroSplatTensor(reshape.getOperand());
  }
  if (auto convert = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    return IsZeroSplatTensor(convert.getOperand());
  }
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return false;
  }
  auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!dense || !dense.isSplat()) {
    return false;
  }
  mlir::Type elem_ty = dense.getElementType();
  if (llvm::isa<mlir::FloatType>(elem_ty)) {
    return dense.getSplatValue<llvm::APFloat>().isZero();
  }
  if (elem_ty.isIntOrIndex()) {
    return dense.getSplatValue<llvm::APInt>().isZero();
  }
  return false;
}

// Returns the scalar integer value of a rank-0 ConstantOp if available.
[[nodiscard]] std::optional<int64_t> GetConstantScalarInt(mlir::Value val) {
  auto type = GetStaticRankedType(val);
  // GENERIC: semantic rank-0 scalar index check
  if (!type || type.getRank() != 0 || !type.getElementType().isIntOrIndex()) {
    return std::nullopt;
  }
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return std::nullopt;
  }
  auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!dense || !dense.isSplat()) {
    return std::nullopt;
  }
  return dense.getSplatValue<llvm::APInt>().getSExtValue();
}

// Returns true if `region` is a single-block scalar addition reduction body.
[[nodiscard]] bool IsSingleInputScalarAddRegion(mlir::Region& region) {
  // GENERIC: semantic single-block 2-arg scalar reduction body
  if (!region.hasOneBlock()) {
    return false;
  }
  mlir::Block& body = region.front();
  if (body.getNumArguments() != 2) {
    return false;
  }
  auto ret_op = llvm::dyn_cast<mlir::stablehlo::ReturnOp>(body.getTerminator());
  if (!ret_op || ret_op.getResults().size() != 1) {
    return false;
  }
  auto add_op = ret_op.getResults()[0].getDefiningOp<mlir::stablehlo::AddOp>();
  if (!add_op) {
    return false;
  }
  mlir::Value a0 = body.getArgument(0);
  mlir::Value a1 = body.getArgument(1);
  return (add_op.getLhs() == a0 && add_op.getRhs() == a1) ||
         (add_op.getLhs() == a1 && add_op.getRhs() == a0);
}

// Creates a rank-0 zero ConstantOp of `elem_type`.
[[nodiscard]] mlir::Value CreateScalarZero(mlir::OpBuilder& builder,
                                           mlir::Location loc,
                                           mlir::Type elem_type) {
  auto scalar_ty = mlir::RankedTensorType::get({}, elem_type);
  mlir::Attribute zero_attr = builder.getZeroAttr(elem_type);
  auto dense_attr = mlir::DenseElementsAttr::get(scalar_ty, zero_attr);
  return mlir::stablehlo::ConstantOp::create(builder, loc, scalar_ty,
                                             dense_attr);
}

// Emits a unit-stride SliceOp along `dim` for `[start, start + length)`.
[[nodiscard]] mlir::Value SliceSingleAxis(mlir::OpBuilder& builder,
                                          mlir::Location loc, mlir::Value input,
                                          int64_t dim, int64_t start,
                                          int64_t length) {
  auto in_type = GetStaticRankedType(input);
  const int64_t rank = in_type.getRank();
  if (start == 0 && length == in_type.getDimSize(dim)) {
    return input;
  }
  llvm::SmallVector<int64_t> starts(rank, 0);
  llvm::SmallVector<int64_t> limits(in_type.getShape().begin(),
                                    in_type.getShape().end());
  // GENERIC: semantic unit-stride slice step
  llvm::SmallVector<int64_t> strides(rank, 1);
  starts[dim] = start;
  limits[dim] = start + length;
  llvm::SmallVector<int64_t> out_shape(in_type.getShape().begin(),
                                       in_type.getShape().end());
  out_shape[dim] = length;
  auto out_type =
      mlir::RankedTensorType::get(out_shape, in_type.getElementType());
  return mlir::stablehlo::SliceOp::create(
      builder, loc, out_type, input, builder.getDenseI64ArrayAttr(starts),
      builder.getDenseI64ArrayAttr(limits),
      builder.getDenseI64ArrayAttr(strides));
}

// Emits a zero-padding PadOp along `dim` placing `raw` at `[start, start+L)`.
[[nodiscard]] mlir::Value EmitZeroPadAlongAxis(mlir::OpBuilder& builder,
                                               mlir::Location loc,
                                               mlir::Value raw, int64_t dim,
                                               int64_t start,
                                               int64_t full_size) {
  auto raw_type = GetStaticRankedType(raw);
  const int64_t rank = raw_type.getRank();
  const int64_t length = raw_type.getDimSize(dim);
  if (start == 0 && length == full_size) {
    return raw;
  }
  llvm::SmallVector<int64_t> low(rank, 0);
  llvm::SmallVector<int64_t> high(rank, 0);
  llvm::SmallVector<int64_t> interior(rank, 0);
  low[dim] = start;
  high[dim] = full_size - start - length;
  llvm::SmallVector<int64_t> out_shape(raw_type.getShape().begin(),
                                       raw_type.getShape().end());
  out_shape[dim] = full_size;
  auto out_type =
      mlir::RankedTensorType::get(out_shape, raw_type.getElementType());
  mlir::Value zero = CreateScalarZero(builder, loc, raw_type.getElementType());
  return mlir::stablehlo::PadOp::create(builder, loc, out_type, raw, zero,
                                        builder.getDenseI64ArrayAttr(low),
                                        builder.getDenseI64ArrayAttr(high),
                                        builder.getDenseI64ArrayAttr(interior));
}

struct SingleAxisSliceSpec {
  int64_t dim = -1;
  int64_t start = 0;
  int64_t limit = 0;

  bool operator==(const SingleAxisSliceSpec& other) const = default;
};

// Matches a unit-stride SliceOp that slices strictly along a single axis.
[[nodiscard]] std::optional<SingleAxisSliceSpec> MatchSingleAxisUnitSlice(
    mlir::stablehlo::SliceOp slice_op) {
  auto in_type = GetStaticRankedType(slice_op.getOperand());
  auto out_type = GetStaticRankedType(slice_op.getResult());
  if (!in_type || !out_type) {
    return std::nullopt;
  }
  const int64_t rank = in_type.getRank();
  llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
  llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();
  llvm::ArrayRef<int64_t> strides = slice_op.getStrides();
  int64_t sliced_dim = -1;
  for (int64_t i = 0; i < rank; ++i) {
    // GENERIC: semantic unit-stride requirement
    if (strides[i] != 1) {
      return std::nullopt;
    }
    if (starts[i] == 0 && limits[i] == in_type.getDimSize(i)) {
      continue;
    }
    if (sliced_dim != -1 || starts[i] < 0 || limits[i] <= starts[i] ||
        limits[i] > in_type.getDimSize(i)) {
      return std::nullopt;
    }
    sliced_dim = i;
  }
  if (sliced_dim == -1) {
    return std::nullopt;
  }
  return SingleAxisSliceSpec{sliced_dim, starts[sliced_dim],
                             limits[sliced_dim]};
}

// Maps an axis `dim` across a ReshapeOp if that axis is preserved without
// splitting or mixing across the sliced extent.
[[nodiscard]] std::optional<int64_t> MapPreservedReshapeDim(
    llvm::ArrayRef<int64_t> src_shape, llvm::ArrayRef<int64_t> dst_shape,
    int64_t src_dim) {
  const int64_t target_len = src_shape[src_dim];
  int64_t prefix_prod = 1;
  for (int64_t i = 0; i < src_dim; ++i) {
    prefix_prod *= src_shape[i];
  }
  int64_t suffix_prod = 1;
  for (int64_t i = src_dim + 1; i < static_cast<int64_t>(src_shape.size());
       ++i) {
    suffix_prod *= src_shape[i];
  }
  int64_t running_prefix = 1;
  for (int64_t j = 0; j < static_cast<int64_t>(dst_shape.size()); ++j) {
    if (running_prefix == prefix_prod && dst_shape[j] == target_len) {
      int64_t rem_prod = 1;
      for (int64_t k = j + 1; k < static_cast<int64_t>(dst_shape.size()); ++k) {
        rem_prod *= dst_shape[k];
      }
      if (rem_prod == suffix_prod) {
        return j;
      }
    }
    running_prefix *= dst_shape[j];
  }
  return std::nullopt;
}

// Canonicalizes ConcatenateOp/SliceOp/ReduceWindowOp/ReshapeOp boundary
// patterns before and after slice hoisting.
bool CanonicalizeSliceBoundaries(mlir::func::FuncOp func) {
  bool changed = false;
  for (mlir::Block& block : func.getBody()) {
    for (mlir::Operation& op : llvm::make_early_inc_range(block)) {
      // Pattern 1: ConcatenateOp of contiguous single-axis unit-stride slices
      // of the same source tensor along concat_dim.
      if (auto concat = llvm::dyn_cast<mlir::stablehlo::ConcatenateOp>(&op)) {
        const uint64_t cdim = concat.getDimension();
        llvm::SmallVector<mlir::Value> new_inputs(concat.getInputs().begin(),
                                                  concat.getInputs().end());
        bool concat_changed = false;
        std::size_t idx = 0;
        while (idx + 1 < new_inputs.size()) {
          auto s0 = new_inputs[idx].getDefiningOp<mlir::stablehlo::SliceOp>();
          if (!s0) {
            ++idx;
            continue;
          }
          auto spec0 = MatchSingleAxisUnitSlice(s0);
          if (!spec0 || spec0->dim != static_cast<int64_t>(cdim)) {
            ++idx;
            continue;
          }
          mlir::Value base_src = s0.getOperand();
          int64_t cur_end = spec0->limit;
          std::size_t end_idx = idx + 1;
          while (end_idx < new_inputs.size()) {
            auto sj =
                new_inputs[end_idx].getDefiningOp<mlir::stablehlo::SliceOp>();
            if (!sj || sj.getOperand() != base_src) {
              break;
            }
            auto specj = MatchSingleAxisUnitSlice(sj);
            if (!specj || specj->dim != static_cast<int64_t>(cdim) ||
                specj->start != cur_end) {
              break;
            }
            cur_end = specj->limit;
            ++end_idx;
          }
          if (end_idx > idx + 1) {
            mlir::OpBuilder builder(concat);
            mlir::Value merged =
                SliceSingleAxis(builder, concat.getLoc(), base_src, spec0->dim,
                                spec0->start, cur_end - spec0->start);
            new_inputs[idx] = merged;
            new_inputs.erase(new_inputs.begin() + idx + 1,
                             new_inputs.begin() + end_idx);
            concat_changed = true;
            continue;
          }
          ++idx;
        }
        if (concat_changed) {
          if (new_inputs.size() == 1) {
            concat.replaceAllUsesWith(new_inputs[0]);
            concat.erase();
          } else {
            mlir::OpBuilder builder(concat);
            auto new_concat = mlir::stablehlo::ConcatenateOp::create(
                builder, concat.getLoc(), concat.getType(), new_inputs, cdim);
            concat.replaceAllUsesWith(new_concat.getResult());
            concat.erase();
          }
          changed = true;
          continue;
        }
      }

      auto slice_op = llvm::dyn_cast<mlir::stablehlo::SliceOp>(&op);
      if (!slice_op) {
        continue;
      }
      auto spec = MatchSingleAxisUnitSlice(slice_op);
      if (!spec) {
        continue;
      }

      // Pattern 2: SliceOp(ConcatenateOp(inputs, d), d, [r0:r1]) where [r0, r1)
      // falls completely inside a single concatenated operand.
      if (auto concat = slice_op.getOperand()
                            .getDefiningOp<mlir::stablehlo::ConcatenateOp>()) {
        if (static_cast<int64_t>(concat.getDimension()) == spec->dim) {
          int64_t offset = 0;
          mlir::Value matched_input = nullptr;
          int64_t local_start = 0;
          for (mlir::Value in_val : concat.getInputs()) {
            auto in_ty = GetStaticRankedType(in_val);
            if (!in_ty) {
              matched_input = nullptr;
              break;
            }
            const int64_t extent = in_ty.getDimSize(spec->dim);
            if (spec->start >= offset && spec->limit <= offset + extent) {
              matched_input = in_val;
              local_start = spec->start - offset;
              break;
            }
            offset += extent;
          }
          if (matched_input != nullptr) {
            mlir::OpBuilder builder(slice_op);
            mlir::Value repl = SliceSingleAxis(
                builder, slice_op.getLoc(), matched_input, spec->dim,
                local_start, spec->limit - spec->start);
            slice_op.replaceAllUsesWith(repl);
            slice_op.erase();
            changed = true;
            continue;
          }
        }
      }

      // Pattern 3: SliceOp(CumSum_ReduceWindowOp(X, d), d, [N-1:N]) ->
      // ReshapeOp(ReduceOp(X, 0, dims={d})).
      if (auto rw = slice_op.getOperand()
                        .getDefiningOp<mlir::stablehlo::ReduceWindowOp>()) {
        if (rw->hasOneUse() && rw.getInputs().size() == 1 &&
            rw.getInitValues().size() == 1 && rw.getNumResults() == 1 &&
            IsZeroSplatTensor(rw.getInitValues()[0]) &&
            IsSingleInputScalarAddRegion(rw.getBody())) {
          auto in_ty = GetStaticRankedType(rw.getInputs()[0]);
          if (in_ty) {
            const int64_t rank = in_ty.getRank();
            const int64_t n_ext = in_ty.getDimSize(spec->dim);
            if (n_ext > 1 && spec->start == n_ext - 1 && spec->limit == n_ext) {
              llvm::ArrayRef<int64_t> win_dims = rw.getWindowDimensions();
              bool win_ok = (static_cast<int64_t>(win_dims.size()) == rank);
              for (int64_t i = 0; i < rank && win_ok; ++i) {
                if (win_dims[i] != (i == spec->dim ? n_ext : 1)) {
                  win_ok = false;
                }
              }
              auto check_all_ones =
                  [](std::optional<llvm::ArrayRef<int64_t>> arr) {
                    if (!arr.has_value()) {
                      return true;
                    }
                    for (int64_t v : *arr) {
                      if (v != 1) {
                        return false;
                      }
                    }
                    return true;
                  };
              win_ok = win_ok && check_all_ones(rw.getWindowStrides()) &&
                       check_all_ones(rw.getBaseDilations()) &&
                       check_all_ones(rw.getWindowDilations());
              if (win_ok) {
                bool pad_ok = false;
                if (auto pad_attr = rw.getPaddingAttr()) {
                  auto vals = pad_attr.getValues<int64_t>();
                  // GENERIC: semantic low/high padding pair per dimension
                  if (static_cast<int64_t>(vals.size()) == 2 * rank) {
                    pad_ok = true;
                    for (int64_t i = 0; i < rank; ++i) {
                      const int64_t low = vals[2 * i];
                      const int64_t high = vals[2 * i + 1];
                      if (i == spec->dim) {
                        if (low != n_ext - 1 || high != 0) {
                          pad_ok = false;
                        }
                      } else if (low != 0 || high != 0) {
                        pad_ok = false;
                      }
                    }
                  }
                }
                if (pad_ok) {
                  mlir::OpBuilder builder(slice_op);
                  llvm::SmallVector<int64_t> red_shape;
                  for (int64_t i = 0; i < rank; ++i) {
                    if (i != spec->dim) {
                      red_shape.push_back(in_ty.getDimSize(i));
                    }
                  }
                  auto red_ty = mlir::RankedTensorType::get(
                      red_shape, in_ty.getElementType());
                  auto red_op = mlir::stablehlo::ReduceOp::create(
                      builder, slice_op.getLoc(), mlir::TypeRange{red_ty},
                      mlir::ValueRange{rw.getInputs()[0]},
                      mlir::ValueRange{rw.getInitValues()[0]},
                      builder.getDenseI64ArrayAttr({spec->dim}));
                  mlir::Type elem_ty = in_ty.getElementType();
                  auto scalar_ty = mlir::RankedTensorType::get({}, elem_ty);
                  mlir::Block* body = builder.createBlock(
                      &red_op.getBody(), {}, {scalar_ty, scalar_ty},
                      {slice_op.getLoc(), slice_op.getLoc()});
                  auto sum_val = mlir::stablehlo::AddOp::create(
                      builder, slice_op.getLoc(), body->getArgument(0),
                      body->getArgument(1));
                  mlir::stablehlo::ReturnOp::create(builder, slice_op.getLoc(),
                                                    mlir::ValueRange{sum_val});
                  builder.setInsertionPointAfter(red_op);
                  auto reshaped = mlir::stablehlo::ReshapeOp::create(
                      builder, slice_op.getLoc(), slice_op.getType(),
                      red_op.getResult(0));
                  slice_op.replaceAllUsesWith(reshaped.getResult());
                  slice_op.erase();
                  rw.erase();
                  changed = true;
                  continue;
                }
              }
            }
          }
        }
      }

      // Pattern 4: SliceOp(ReshapeOp(X)) where ReshapeOp merged an axis M with
      // inner axes of product K (e.g. [A * M, K] -> [A, M * K] sliced at
      // [0:A, r0*K:r1*K]).
      if (auto reshape = slice_op.getOperand()
                             .getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
        if (reshape->hasOneUse()) {
          auto x_ty = GetStaticRankedType(reshape.getOperand());
          auto r_ty = GetStaticRankedType(reshape.getResult());
          // GENERIC: semantic 2D [A*M, K] -> [A, M*K] factorized slice
          if (x_ty && r_ty && x_ty.getRank() == 2 && r_ty.getRank() == 2 &&
              spec->dim == 1) {
            const int64_t a_dim = r_ty.getDimSize(0);
            const int64_t mk_dim = r_ty.getDimSize(1);
            const int64_t am_dim = x_ty.getDimSize(0);
            const int64_t k_dim = x_ty.getDimSize(1);
            if (a_dim > 0 && k_dim > 0 && am_dim % a_dim == 0 &&
                mk_dim % k_dim == 0) {
              const int64_t m_dim = am_dim / a_dim;
              if (m_dim > 1 && m_dim * k_dim == mk_dim &&
                  spec->start % k_dim == 0 && spec->limit % k_dim == 0) {
                const int64_t s0 = spec->start / k_dim;
                const int64_t s1 = spec->limit / k_dim;
                mlir::OpBuilder builder(slice_op);
                auto mid_ty = mlir::RankedTensorType::get(
                    {a_dim, m_dim, k_dim}, x_ty.getElementType());
                auto r3d = mlir::stablehlo::ReshapeOp::create(
                    builder, slice_op.getLoc(), mid_ty, reshape.getOperand());
                mlir::Value sl3d =
                    SliceSingleAxis(builder, slice_op.getLoc(), r3d.getResult(),
                                    1, s0, s1 - s0);
                auto r2d = mlir::stablehlo::ReshapeOp::create(
                    builder, slice_op.getLoc(), slice_op.getType(), sl3d);
                slice_op.replaceAllUsesWith(r2d.getResult());
                slice_op.erase();
                reshape.erase();
                changed = true;
                continue;
              }
            }
          }
        }
      }
    }
  }
  return changed;
}

struct ZeroPadInfo {
  mlir::Value raw;
  int64_t dim = -1;
  int64_t start = 0;
  int64_t length = 0;
  int64_t full_size = 0;
  bool is_root = false;

  bool MatchesFrame(const ZeroPadInfo& other) const {
    return dim == other.dim && start == other.start && length == other.length &&
           full_size == other.full_size;
  }
};

// Matches a base operation that constructs a zero-padded slice along one axis.
[[nodiscard]] std::optional<ZeroPadInfo> MatchRootZeroPad(mlir::Operation* op) {
  if (auto pad = llvm::dyn_cast<mlir::stablehlo::PadOp>(op)) {
    if (!IsZeroSplatTensor(pad.getPaddingValue())) {
      return std::nullopt;
    }
    auto in_ty = GetStaticRankedType(pad.getOperand());
    auto out_ty = GetStaticRankedType(pad.getResult());
    if (!in_ty || !out_ty) {
      return std::nullopt;
    }
    const int64_t rank = in_ty.getRank();
    llvm::ArrayRef<int64_t> low = pad.getEdgePaddingLow();
    llvm::ArrayRef<int64_t> high = pad.getEdgePaddingHigh();
    llvm::ArrayRef<int64_t> interior = pad.getInteriorPadding();
    int64_t pad_dim = -1;
    for (int64_t i = 0; i < rank; ++i) {
      if (interior[i] != 0) {
        return std::nullopt;
      }
      if (low[i] == 0 && high[i] == 0) {
        continue;
      }
      if (pad_dim != -1 || low[i] < 0 || high[i] < 0) {
        return std::nullopt;
      }
      pad_dim = i;
    }
    if (pad_dim == -1) {
      return std::nullopt;
    }
    return ZeroPadInfo{pad.getOperand(),
                       pad_dim,
                       low[pad_dim],
                       in_ty.getDimSize(pad_dim),
                       out_ty.getDimSize(pad_dim),
                       /*is_root=*/true};
  }

  if (auto dus = llvm::dyn_cast<mlir::stablehlo::DynamicUpdateSliceOp>(op)) {
    if (!IsZeroSplatTensor(dus.getOperand())) {
      return std::nullopt;
    }
    auto base_ty = GetStaticRankedType(dus.getOperand());
    auto upd_ty = GetStaticRankedType(dus.getUpdate());
    if (!base_ty || !upd_ty) {
      return std::nullopt;
    }
    const int64_t rank = base_ty.getRank();
    int64_t diff_dim = -1;
    int64_t start_val = 0;
    for (int64_t i = 0; i < rank; ++i) {
      auto idx = GetConstantScalarInt(dus.getStartIndices()[i]);
      if (!idx.has_value()) {
        return std::nullopt;
      }
      if (upd_ty.getDimSize(i) == base_ty.getDimSize(i)) {
        if (*idx != 0) {
          return std::nullopt;
        }
        continue;
      }
      if (diff_dim != -1 || *idx < 0 ||
          *idx + upd_ty.getDimSize(i) > base_ty.getDimSize(i)) {
        return std::nullopt;
      }
      diff_dim = i;
      start_val = *idx;
    }
    if (diff_dim == -1) {
      return std::nullopt;
    }
    return ZeroPadInfo{dus.getUpdate(),
                       diff_dim,
                       start_val,
                       upd_ty.getDimSize(diff_dim),
                       base_ty.getDimSize(diff_dim),
                       /*is_root=*/true};
  }

  if (auto concat = llvm::dyn_cast<mlir::stablehlo::ConcatenateOp>(op)) {
    auto out_ty = GetStaticRankedType(concat.getResult());
    // GENERIC: semantic multi-operand concatenation check
    if (!out_ty || concat.getInputs().size() < 2) {
      return std::nullopt;
    }
    const int64_t cdim = static_cast<int64_t>(concat.getDimension());
    mlir::Value non_zero = nullptr;
    int64_t start_offset = 0;
    int64_t running = 0;
    for (mlir::Value in_v : concat.getInputs()) {
      auto in_ty = GetStaticRankedType(in_v);
      if (!in_ty) {
        return std::nullopt;
      }
      if (IsZeroSplatTensor(in_v)) {
        running += in_ty.getDimSize(cdim);
        continue;
      }
      if (non_zero != nullptr) {
        return std::nullopt;
      }
      non_zero = in_v;
      start_offset = running;
      running += in_ty.getDimSize(cdim);
    }
    if (non_zero == nullptr) {
      return std::nullopt;
    }
    auto nz_ty = GetStaticRankedType(non_zero);
    return ZeroPadInfo{non_zero,
                       cdim,
                       start_offset,
                       nz_ty.getDimSize(cdim),
                       out_ty.getDimSize(cdim),
                       /*is_root=*/true};
  }

  return std::nullopt;
}

// Propagates ZeroPadInfo forward through homogeneous zero-preserving backward
// subgraphs and folds contracting/reducing sinks.
bool PropagateZeroPaddedSlicesForward(mlir::func::FuncOp func) {
  bool changed = false;
  for (mlir::Block& block : func.getBody()) {
    llvm::DenseMap<mlir::Value, ZeroPadInfo> zp_map;
    llvm::SmallVector<mlir::Value> derived_zp_values;

    for (mlir::Operation& op : llvm::make_early_inc_range(block)) {
      if (auto root_info = MatchRootZeroPad(&op)) {
        zp_map[op.getResult(0)] = *root_info;
        continue;
      }

      // Sink 1: SliceOp extracting the exact unpadded slice.
      if (auto slice_op = llvm::dyn_cast<mlir::stablehlo::SliceOp>(&op)) {
        auto it = zp_map.find(slice_op.getOperand());
        if (it != zp_map.end()) {
          auto spec = MatchSingleAxisUnitSlice(slice_op);
          if (spec && spec->dim == it->second.dim &&
              spec->start == it->second.start &&
              spec->limit == it->second.start + it->second.length) {
            slice_op.replaceAllUsesWith(it->second.raw);
            slice_op.erase();
            changed = true;
            continue;
          }
        }
      }

      // Propagate across ConvertOp (float/int types preserve 0 -> 0).
      if (auto conv = llvm::dyn_cast<mlir::stablehlo::ConvertOp>(&op)) {
        auto it = zp_map.find(conv.getOperand());
        auto out_ty = GetStaticRankedType(conv.getResult());
        if (it != zp_map.end() && out_ty &&
            (llvm::isa<mlir::FloatType>(out_ty.getElementType()) ||
             out_ty.getElementType().isIntOrIndex())) {
          mlir::OpBuilder builder(conv);
          auto raw_in_ty = GetStaticRankedType(it->second.raw);
          auto raw_out_ty = mlir::RankedTensorType::get(
              raw_in_ty.getShape(), out_ty.getElementType());
          auto new_raw = mlir::stablehlo::ConvertOp::create(
              builder, conv.getLoc(), raw_out_ty, it->second.raw);
          ZeroPadInfo info = it->second;
          info.raw = new_raw.getResult();
          info.is_root = false;
          zp_map[conv.getResult()] = info;
          derived_zp_values.push_back(conv.getResult());
          continue;
        }
      }

      // Propagate across NegOp (-0 == 0).
      if (auto neg = llvm::dyn_cast<mlir::stablehlo::NegOp>(&op)) {
        auto it = zp_map.find(neg.getOperand());
        if (it != zp_map.end()) {
          mlir::OpBuilder builder(neg);
          auto new_raw = mlir::stablehlo::NegOp::create(builder, neg.getLoc(),
                                                        it->second.raw);
          ZeroPadInfo info = it->second;
          info.raw = new_raw.getResult();
          info.is_root = false;
          zp_map[neg.getResult()] = info;
          derived_zp_values.push_back(neg.getResult());
          continue;
        }
      }

      // Propagate across MulOp (0 * y == 0 and x * 0 == 0).
      if (auto mul = llvm::dyn_cast<mlir::stablehlo::MulOp>(&op)) {
        auto it_l = zp_map.find(mul.getLhs());
        auto it_r = zp_map.find(mul.getRhs());
        if (it_l != zp_map.end() && it_r != zp_map.end() &&
            it_l->second.MatchesFrame(it_r->second)) {
          mlir::OpBuilder builder(mul);
          auto new_raw = mlir::stablehlo::MulOp::create(
              builder, mul.getLoc(), it_l->second.raw, it_r->second.raw);
          ZeroPadInfo info = it_l->second;
          info.raw = new_raw.getResult();
          info.is_root = false;
          zp_map[mul.getResult()] = info;
          derived_zp_values.push_back(mul.getResult());
          continue;
        }
        if (it_l != zp_map.end() || it_r != zp_map.end()) {
          const ZeroPadInfo info =
              (it_l != zp_map.end()) ? it_l->second : it_r->second;
          mlir::Value other =
              (it_l != zp_map.end()) ? mul.getRhs() : mul.getLhs();
          mlir::OpBuilder builder(mul);
          mlir::Value other_sl = SliceSingleAxis(
              builder, mul.getLoc(), other, info.dim, info.start, info.length);
          mlir::Value lhs_raw = (it_l != zp_map.end()) ? info.raw : other_sl;
          mlir::Value rhs_raw = (it_l != zp_map.end()) ? other_sl : info.raw;
          auto new_raw = mlir::stablehlo::MulOp::create(builder, mul.getLoc(),
                                                        lhs_raw, rhs_raw);
          ZeroPadInfo out_info = info;
          out_info.raw = new_raw.getResult();
          out_info.is_root = false;
          zp_map[mul.getResult()] = out_info;
          derived_zp_values.push_back(mul.getResult());
          continue;
        }
      }

      // Propagate across DivOp when numerator is zero-padded (0 / y == 0).
      if (auto div = llvm::dyn_cast<mlir::stablehlo::DivOp>(&op)) {
        auto it_l = zp_map.find(div.getLhs());
        if (it_l != zp_map.end()) {
          const ZeroPadInfo info = it_l->second;
          mlir::OpBuilder builder(div);
          mlir::Value rhs_sl =
              SliceSingleAxis(builder, div.getLoc(), div.getRhs(), info.dim,
                              info.start, info.length);
          auto new_raw = mlir::stablehlo::DivOp::create(builder, div.getLoc(),
                                                        info.raw, rhs_sl);
          ZeroPadInfo out_info = info;
          out_info.raw = new_raw.getResult();
          out_info.is_root = false;
          zp_map[div.getResult()] = out_info;
          derived_zp_values.push_back(div.getResult());
          continue;
        }
      }

      // Propagate across AddOp and SubtractOp when BOTH operands share the
      // same ZeroPadInfo frame, or when one operand is a zero splat.
      if (llvm::isa<mlir::stablehlo::AddOp, mlir::stablehlo::SubtractOp>(&op)) {
        const bool is_add = llvm::isa<mlir::stablehlo::AddOp>(&op);
        mlir::Value lhs = op.getOperand(0);
        mlir::Value rhs = op.getOperand(1);
        auto it_l = zp_map.find(lhs);
        auto it_r = zp_map.find(rhs);
        if (it_l != zp_map.end() && it_r != zp_map.end() &&
            it_l->second.MatchesFrame(it_r->second)) {
          mlir::OpBuilder builder(&op);
          mlir::Value new_raw = is_add ? mlir::stablehlo::AddOp::create(
                                             builder, op.getLoc(),
                                             it_l->second.raw, it_r->second.raw)
                                             .getResult()
                                       : mlir::stablehlo::SubtractOp::create(
                                             builder, op.getLoc(),
                                             it_l->second.raw, it_r->second.raw)
                                             .getResult();
          ZeroPadInfo out_info = it_l->second;
          out_info.raw = new_raw;
          out_info.is_root = false;
          zp_map[op.getResult(0)] = out_info;
          derived_zp_values.push_back(op.getResult(0));
          continue;
        }
        // Also fold AddOp(ZeroPaddedSlice, Base) -> DynamicUpdateSliceOp when
        // Base is not in zp_map and AddOp is a single-sink consumer of root
        // pad.
        if (is_add && ((it_l != zp_map.end() && it_l->second.is_root &&
                        lhs.hasOneUse() && it_r == zp_map.end()) ||
                       (it_r != zp_map.end() && it_r->second.is_root &&
                        rhs.hasOneUse() && it_l == zp_map.end()))) {
          const ZeroPadInfo info =
              (it_l != zp_map.end()) ? it_l->second : it_r->second;
          mlir::Value base = (it_l != zp_map.end()) ? rhs : lhs;
          auto base_ty = GetStaticRankedType(base);
          if (base_ty) {
            mlir::OpBuilder builder(&op);
            mlir::Value base_sl = SliceSingleAxis(
                builder, op.getLoc(), base, info.dim, info.start, info.length);
            auto sum_sl = mlir::stablehlo::AddOp::create(builder, op.getLoc(),
                                                         base_sl, info.raw);
            auto idx_ty = mlir::RankedTensorType::get({}, builder.getI64Type());
            auto zero_idx = mlir::stablehlo::ConstantOp::create(
                builder, op.getLoc(), idx_ty,
                mlir::DenseElementsAttr::get(idx_ty,
                                             builder.getI64IntegerAttr(0)));
            auto start_idx = mlir::stablehlo::ConstantOp::create(
                builder, op.getLoc(), idx_ty,
                mlir::DenseElementsAttr::get(
                    idx_ty, builder.getI64IntegerAttr(info.start)));
            llvm::SmallVector<mlir::Value> indices(base_ty.getRank(), zero_idx);
            indices[info.dim] = start_idx;
            auto dus = mlir::stablehlo::DynamicUpdateSliceOp::create(
                builder, op.getLoc(), base_ty, base, sum_sl.getResult(),
                indices);
            op.getResult(0).replaceAllUsesWith(dus.getResult());
            op.erase();
            changed = true;
            continue;
          }
        }
      }

      // Propagate across SelectOp when both branches are zero outside [s, s+L).
      if (auto sel = llvm::dyn_cast<mlir::stablehlo::SelectOp>(&op)) {
        auto it_t = zp_map.find(sel.getOnTrue());
        auto it_f = zp_map.find(sel.getOnFalse());
        const bool f_zero = IsZeroSplatTensor(sel.getOnFalse());
        const bool t_zero = IsZeroSplatTensor(sel.getOnTrue());
        if ((it_t != zp_map.end() && f_zero) ||
            (it_f != zp_map.end() && t_zero) ||
            (it_t != zp_map.end() && it_f != zp_map.end() &&
             it_t->second.MatchesFrame(it_f->second))) {
          const ZeroPadInfo info =
              (it_t != zp_map.end()) ? it_t->second : it_f->second;
          mlir::OpBuilder builder(sel);
          mlir::Value pred = sel.getPred();
          auto pred_ty = GetStaticRankedType(pred);
          if (pred_ty && pred_ty.getRank() > 0) {
            pred = SliceSingleAxis(builder, sel.getLoc(), pred, info.dim,
                                   info.start, info.length);
          }
          mlir::Value t_raw =
              (it_t != zp_map.end())
                  ? it_t->second.raw
                  : SliceSingleAxis(builder, sel.getLoc(), sel.getOnTrue(),
                                    info.dim, info.start, info.length);
          mlir::Value f_raw =
              (it_f != zp_map.end())
                  ? it_f->second.raw
                  : SliceSingleAxis(builder, sel.getLoc(), sel.getOnFalse(),
                                    info.dim, info.start, info.length);
          auto new_raw = mlir::stablehlo::SelectOp::create(
              builder, sel.getLoc(), pred, t_raw, f_raw);
          ZeroPadInfo out_info = info;
          out_info.raw = new_raw.getResult();
          out_info.is_root = false;
          zp_map[sel.getResult()] = out_info;
          derived_zp_values.push_back(sel.getResult());
          continue;
        }
      }

      // Propagate across BroadcastInDimOp when `info.dim` maps to `d_out` of
      // the same extent `info.full_size`.
      if (auto bcast = llvm::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(&op)) {
        auto it = zp_map.find(bcast.getOperand());
        auto out_ty = GetStaticRankedType(bcast.getResult());
        if (it != zp_map.end() && out_ty) {
          const ZeroPadInfo info = it->second;
          llvm::ArrayRef<int64_t> bdims = bcast.getBroadcastDimensions();
          if (info.dim >= 0 && info.dim < static_cast<int64_t>(bdims.size())) {
            const int64_t d_out = bdims[info.dim];
            if (out_ty.getDimSize(d_out) == info.full_size) {
              mlir::OpBuilder builder(bcast);
              llvm::SmallVector<int64_t> raw_out_shape(
                  out_ty.getShape().begin(), out_ty.getShape().end());
              raw_out_shape[d_out] = info.length;
              auto raw_out_ty = mlir::RankedTensorType::get(
                  raw_out_shape, out_ty.getElementType());
              auto new_raw = mlir::stablehlo::BroadcastInDimOp::create(
                  builder, bcast.getLoc(), raw_out_ty, info.raw,
                  bcast.getBroadcastDimensionsAttr());
              ZeroPadInfo out_info = info;
              out_info.raw = new_raw.getResult();
              out_info.dim = d_out;
              out_info.is_root = false;
              zp_map[bcast.getResult()] = out_info;
              derived_zp_values.push_back(bcast.getResult());
              continue;
            }
          }
        }
      }

      // Propagate across ReshapeOp when `info.dim` is preserved.
      if (auto reshape = llvm::dyn_cast<mlir::stablehlo::ReshapeOp>(&op)) {
        auto it = zp_map.find(reshape.getOperand());
        auto in_ty = GetStaticRankedType(reshape.getOperand());
        auto out_ty = GetStaticRankedType(reshape.getResult());
        if (it != zp_map.end() && in_ty && out_ty) {
          const ZeroPadInfo info = it->second;
          auto mapped_dim = MapPreservedReshapeDim(in_ty.getShape(),
                                                   out_ty.getShape(), info.dim);
          if (mapped_dim.has_value()) {
            mlir::OpBuilder builder(reshape);
            llvm::SmallVector<int64_t> raw_out_shape(out_ty.getShape().begin(),
                                                     out_ty.getShape().end());
            raw_out_shape[*mapped_dim] = info.length;
            auto raw_out_ty = mlir::RankedTensorType::get(
                raw_out_shape, out_ty.getElementType());
            auto new_raw = mlir::stablehlo::ReshapeOp::create(
                builder, reshape.getLoc(), raw_out_ty, info.raw);
            ZeroPadInfo out_info = info;
            out_info.raw = new_raw.getResult();
            out_info.dim = *mapped_dim;
            out_info.is_root = false;
            zp_map[reshape.getResult()] = out_info;
            derived_zp_values.push_back(reshape.getResult());
            continue;
          }
        }
      }

      // Propagate across TransposeOp.
      if (auto tr = llvm::dyn_cast<mlir::stablehlo::TransposeOp>(&op)) {
        auto it = zp_map.find(tr.getOperand());
        auto out_ty = GetStaticRankedType(tr.getResult());
        if (it != zp_map.end() && out_ty) {
          const ZeroPadInfo info = it->second;
          llvm::ArrayRef<int64_t> perm = tr.getPermutation();
          int64_t out_dim = -1;
          for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
            if (perm[i] == info.dim) {
              out_dim = i;
              break;
            }
          }
          if (out_dim != -1) {
            mlir::OpBuilder builder(tr);
            llvm::SmallVector<int64_t> raw_out_shape(out_ty.getShape().begin(),
                                                     out_ty.getShape().end());
            raw_out_shape[out_dim] = info.length;
            auto raw_out_ty = mlir::RankedTensorType::get(
                raw_out_shape, out_ty.getElementType());
            auto new_raw = mlir::stablehlo::TransposeOp::create(
                builder, tr.getLoc(), raw_out_ty, info.raw,
                tr.getPermutationAttr());
            ZeroPadInfo out_info = info;
            out_info.raw = new_raw.getResult();
            out_info.dim = out_dim;
            out_info.is_root = false;
            zp_map[tr.getResult()] = out_info;
            derived_zp_values.push_back(tr.getResult());
            continue;
          }
        }
      }

      // Sink 2 or Propagate: ReduceOp with additive body and zero init.
      if (auto red = llvm::dyn_cast<mlir::stablehlo::ReduceOp>(&op)) {
        if (red.getInputs().size() == 1 && red.getInitValues().size() == 1 &&
            red.getNumResults() == 1 &&
            IsZeroSplatTensor(red.getInitValues()[0]) &&
            IsSingleInputScalarAddRegion(red.getBody())) {
          auto it = zp_map.find(red.getInputs()[0]);
          auto out_ty = GetStaticRankedType(red.getResult(0));
          if (it != zp_map.end() && out_ty) {
            const ZeroPadInfo info = it->second;
            llvm::ArrayRef<int64_t> dims = red.getDimensions();
            mlir::OpBuilder builder(red);
            if (llvm::is_contained(dims, info.dim)) {
              // Reducing along info.dim eliminates the zero padding completely.
              auto new_red = mlir::stablehlo::ReduceOp::create(
                  builder, red.getLoc(), red.getResultTypes(),
                  mlir::ValueRange{info.raw}, red.getInitValues(),
                  red.getDimensionsAttr());
              mlir::IRMapping mapping;
              red.getBody().cloneInto(&new_red.getBody(), mapping);
              red.getResult(0).replaceAllUsesWith(new_red.getResult(0));
              red.erase();
              changed = true;
              continue;
            } else {
              // Orthogonal reduction: output remains zero-padded along shifted
              // axis `out_dim`.
              int64_t num_less = 0;
              for (int64_t d : dims) {
                if (d < info.dim) {
                  ++num_less;
                }
              }
              const int64_t out_dim = info.dim - num_less;
              llvm::SmallVector<int64_t> raw_out_shape(
                  out_ty.getShape().begin(), out_ty.getShape().end());
              raw_out_shape[out_dim] = info.length;
              auto raw_out_ty = mlir::RankedTensorType::get(
                  raw_out_shape, out_ty.getElementType());
              auto new_red = mlir::stablehlo::ReduceOp::create(
                  builder, red.getLoc(), mlir::TypeRange{raw_out_ty},
                  mlir::ValueRange{info.raw}, red.getInitValues(),
                  red.getDimensionsAttr());
              mlir::IRMapping mapping;
              red.getBody().cloneInto(&new_red.getBody(), mapping);
              ZeroPadInfo out_info = info;
              out_info.raw = new_red.getResult(0);
              out_info.dim = out_dim;
              out_info.is_root = false;
              zp_map[red.getResult(0)] = out_info;
              derived_zp_values.push_back(red.getResult(0));
              continue;
            }
          }
        }
      }

      // Sink 3 or Propagate: DotGeneralOp.
      if (auto dot = llvm::dyn_cast<mlir::stablehlo::DotGeneralOp>(&op)) {
        auto it_l = zp_map.find(dot.getLhs());
        auto it_r = zp_map.find(dot.getRhs());
        auto lhs_ty = GetStaticRankedType(dot.getLhs());
        auto rhs_ty = GetStaticRankedType(dot.getRhs());
        auto out_ty = GetStaticRankedType(dot.getResult());
        if (lhs_ty && rhs_ty && out_ty &&
            ((it_l != zp_map.end()) != (it_r != zp_map.end()))) {
          const bool zp_on_lhs = (it_l != zp_map.end());
          const ZeroPadInfo info = zp_on_lhs ? it_l->second : it_r->second;
          auto dnums = dot.getDotDimensionNumbers();
          llvm::ArrayRef<int64_t> b_lhs = dnums.getLhsBatchingDimensions();
          llvm::ArrayRef<int64_t> b_rhs = dnums.getRhsBatchingDimensions();
          llvm::ArrayRef<int64_t> c_lhs = dnums.getLhsContractingDimensions();
          llvm::ArrayRef<int64_t> c_rhs = dnums.getRhsContractingDimensions();
          llvm::ArrayRef<int64_t> b_self = zp_on_lhs ? b_lhs : b_rhs;
          llvm::ArrayRef<int64_t> b_other = zp_on_lhs ? b_rhs : b_lhs;
          llvm::ArrayRef<int64_t> c_self = zp_on_lhs ? c_lhs : c_rhs;
          llvm::ArrayRef<int64_t> c_other = zp_on_lhs ? c_rhs : c_lhs;

          int64_t c_pos = -1;
          for (int64_t i = 0; i < static_cast<int64_t>(c_self.size()); ++i) {
            if (c_self[i] == info.dim) {
              c_pos = i;
              break;
            }
          }
          int64_t b_pos = -1;
          for (int64_t i = 0; i < static_cast<int64_t>(b_self.size()); ++i) {
            if (b_self[i] == info.dim) {
              b_pos = i;
              break;
            }
          }

          mlir::OpBuilder builder(dot);
          if (c_pos != -1) {
            // Contracting dimension: slice the other operand's matching
            // contracting dimension and eliminate the zero padding completely.
            const int64_t other_dim = c_other[c_pos];
            mlir::Value other_val = zp_on_lhs ? dot.getRhs() : dot.getLhs();
            mlir::Value other_sl =
                SliceSingleAxis(builder, dot.getLoc(), other_val, other_dim,
                                info.start, info.length);
            mlir::Value new_lhs = zp_on_lhs ? info.raw : other_sl;
            mlir::Value new_rhs = zp_on_lhs ? other_sl : info.raw;
            auto new_dot = mlir::stablehlo::DotGeneralOp::create(
                builder, dot.getLoc(), out_ty, new_lhs, new_rhs, dnums,
                dot.getPrecisionConfigAttr(), dot.getAlgorithmAttr());
            dot.replaceAllUsesWith(new_dot.getResult());
            dot.erase();
            changed = true;
            continue;
          } else if (b_pos != -1) {
            // Batching dimension: slice the other operand's matching batching
            // dimension and propagate ZeroPadInfo along output batch dim b_pos.
            const int64_t other_dim = b_other[b_pos];
            mlir::Value other_val = zp_on_lhs ? dot.getRhs() : dot.getLhs();
            mlir::Value other_sl =
                SliceSingleAxis(builder, dot.getLoc(), other_val, other_dim,
                                info.start, info.length);
            mlir::Value new_lhs = zp_on_lhs ? info.raw : other_sl;
            mlir::Value new_rhs = zp_on_lhs ? other_sl : info.raw;
            llvm::SmallVector<int64_t> raw_out_shape(out_ty.getShape().begin(),
                                                     out_ty.getShape().end());
            raw_out_shape[b_pos] = info.length;
            auto raw_out_ty = mlir::RankedTensorType::get(
                raw_out_shape, out_ty.getElementType());
            auto new_dot = mlir::stablehlo::DotGeneralOp::create(
                builder, dot.getLoc(), raw_out_ty, new_lhs, new_rhs, dnums,
                dot.getPrecisionConfigAttr(), dot.getAlgorithmAttr());
            ZeroPadInfo out_info = info;
            out_info.raw = new_dot.getResult();
            out_info.dim = b_pos;
            out_info.is_root = false;
            zp_map[dot.getResult()] = out_info;
            derived_zp_values.push_back(dot.getResult());
            continue;
          } else {
            // Free dimension: compute output dimension index and propagate.
            int64_t lhs_free_count = 0;
            int64_t out_dim = -1;
            for (int64_t d = 0; d < lhs_ty.getRank(); ++d) {
              if (!llvm::is_contained(b_lhs, d) &&
                  !llvm::is_contained(c_lhs, d)) {
                if (zp_on_lhs && d == info.dim) {
                  out_dim = static_cast<int64_t>(b_lhs.size()) + lhs_free_count;
                }
                ++lhs_free_count;
              }
            }
            if (!zp_on_lhs) {
              int64_t rhs_free_idx = 0;
              for (int64_t d = 0; d < rhs_ty.getRank(); ++d) {
                if (!llvm::is_contained(b_rhs, d) &&
                    !llvm::is_contained(c_rhs, d)) {
                  if (d == info.dim) {
                    out_dim = static_cast<int64_t>(b_lhs.size()) +
                              lhs_free_count + rhs_free_idx;
                    break;
                  }
                  ++rhs_free_idx;
                }
              }
            }
            if (out_dim != -1) {
              mlir::Value new_lhs = zp_on_lhs ? info.raw : dot.getLhs();
              mlir::Value new_rhs = zp_on_lhs ? dot.getRhs() : info.raw;
              llvm::SmallVector<int64_t> raw_out_shape(
                  out_ty.getShape().begin(), out_ty.getShape().end());
              raw_out_shape[out_dim] = info.length;
              auto raw_out_ty = mlir::RankedTensorType::get(
                  raw_out_shape, out_ty.getElementType());
              auto new_dot = mlir::stablehlo::DotGeneralOp::create(
                  builder, dot.getLoc(), raw_out_ty, new_lhs, new_rhs, dnums,
                  dot.getPrecisionConfigAttr(), dot.getAlgorithmAttr());
              ZeroPadInfo out_info = info;
              out_info.raw = new_dot.getResult();
              out_info.dim = out_dim;
              out_info.is_root = false;
              zp_map[dot.getResult()] = out_info;
              derived_zp_values.push_back(dot.getResult());
              continue;
            }
          }
        }
      }
    }

    // Materialize a single PadOp at the boundary for any derived zp_map value
    // that still has live users outside the rewritten zp subgraph.
    for (mlir::Value val : llvm::reverse(derived_zp_values)) {
      if (!val || val.use_empty()) {
        continue;
      }
      mlir::Operation* def_op = val.getDefiningOp();
      if (!def_op) {
        continue;
      }
      const ZeroPadInfo info = zp_map.lookup(val);
      mlir::OpBuilder builder(def_op);
      mlir::Value padded =
          EmitZeroPadAlongAxis(builder, def_op->getLoc(), info.raw, info.dim,
                               info.start, info.full_size);
      val.replaceAllUsesWith(padded);
      def_op->erase();
      changed = true;
    }
  }
  return changed;
}

// Returns true if `op` is a pure elementwise operation whose output dimensions
// match its non-scalar operands 1-to-1.
[[nodiscard]] bool IsSupportedElementwiseOp(mlir::Operation* op) {
  return llvm::isa<
      mlir::stablehlo::ConvertOp, mlir::stablehlo::BitcastConvertOp,
      mlir::stablehlo::NegOp, mlir::stablehlo::AbsOp, mlir::stablehlo::ExpOp,
      mlir::stablehlo::Expm1Op, mlir::stablehlo::LogOp,
      mlir::stablehlo::Log1pOp, mlir::stablehlo::RsqrtOp,
      mlir::stablehlo::SqrtOp, mlir::stablehlo::TanhOp,
      mlir::stablehlo::LogisticOp, mlir::stablehlo::SineOp,
      mlir::stablehlo::CosineOp, mlir::chlo::ErfOp, mlir::stablehlo::AddOp,
      mlir::stablehlo::SubtractOp, mlir::stablehlo::MulOp,
      mlir::stablehlo::DivOp, mlir::stablehlo::RemOp, mlir::stablehlo::PowOp,
      mlir::stablehlo::MaxOp, mlir::stablehlo::MinOp, mlir::stablehlo::AndOp,
      mlir::stablehlo::OrOp, mlir::stablehlo::XorOp, mlir::stablehlo::CompareOp,
      mlir::stablehlo::SelectOp, mlir::stablehlo::ClampOp>(op);
}

// Checks whether `bcast` originates from a ReduceOp within `max_depth` steps,
// and if so, verifies that every user in the ReduceOp's forward cone either
// reaches `consumer_op` or is already a matching `SliceOp` along the reduced
// tensor's corresponding axis. This guarantees a LayerNorm/RMSNorm/Softmax
// subgraph is never split in half when its ReduceOp has an unsliced user.
[[nodiscard]] bool IsBroadcastFromReduceSafeToHoist(
    mlir::stablehlo::BroadcastInDimOp bcast, mlir::Operation* consumer_op,
    int64_t bcast_slice_dim, int64_t start, int64_t limit) {
  llvm::ArrayRef<int64_t> bdims = bcast.getBroadcastDimensions();
  int64_t src_dim = -1;
  for (int64_t i = 0; i < static_cast<int64_t>(bdims.size()); ++i) {
    if (bdims[i] == bcast_slice_dim) {
      src_dim = i;
      break;
    }
  }
  if (src_dim == -1) {
    return true;
  }

  // Walk upstream from bcast.getOperand() to find any ReduceOp within 8 hops.
  llvm::SmallVector<mlir::Operation*> worklist;
  llvm::DenseSet<mlir::Operation*> visited_up;
  if (mlir::Operation* def = bcast.getOperand().getDefiningOp()) {
    worklist.push_back(def);
    visited_up.insert(def);
  }
  llvm::SmallVector<mlir::stablehlo::ReduceOp> reduce_ops;
  while (!worklist.empty()) {
    mlir::Operation* cur = worklist.pop_back_val();
    if (auto red = llvm::dyn_cast<mlir::stablehlo::ReduceOp>(cur)) {
      reduce_ops.push_back(red);
      continue;
    }
    // GENERIC: capability bounded backward traversal depth
    if (visited_up.size() >= 16) {
      continue;
    }
    for (mlir::Value opnd : cur->getOperands()) {
      if (mlir::Operation* d = opnd.getDefiningOp()) {
        auto d_ty = GetStaticRankedType(opnd);
        if (!d_ty || d_ty.getRank() == 0) {
          continue;
        }
        if (llvm::isa<mlir::stablehlo::ReshapeOp,
                      mlir::stablehlo::BroadcastInDimOp>(cur) &&
            !opnd.hasOneUse()) {
          continue;
        }
        if (visited_up.insert(d).second) {
          worklist.push_back(d);
        }
      }
    }
  }
  if (reduce_ops.empty()) {
    return true;
  }

  // Collect all ancestors of `consumer_op` up to `reduce_ops`.
  llvm::DenseSet<mlir::Operation*> ancestors_of_consumer;
  llvm::SmallVector<mlir::Operation*> anc_worklist{consumer_op};
  ancestors_of_consumer.insert(consumer_op);
  while (!anc_worklist.empty()) {
    mlir::Operation* cur = anc_worklist.pop_back_val();
    if (llvm::isa<mlir::stablehlo::ReduceOp>(cur)) {
      continue;
    }
    for (mlir::Value opnd : cur->getOperands()) {
      if (mlir::Operation* d = opnd.getDefiningOp()) {
        if (visited_up.contains(d) && ancestors_of_consumer.insert(d).second) {
          anc_worklist.push_back(d);
        }
      }
    }
  }

  // Verify that every user of every op in the ReduceOp forward cone is either
  // in `ancestors_of_consumer`, is a matching SliceOp, or only feeds matching
  // SliceOps.
  llvm::DenseSet<mlir::Operation*> cone;
  llvm::SmallVector<mlir::Operation*> fwd_worklist;
  for (auto red : reduce_ops) {
    cone.insert(red.getOperation());
    fwd_worklist.push_back(red.getOperation());
  }
  while (!fwd_worklist.empty()) {
    mlir::Operation* cur = fwd_worklist.pop_back_val();
    for (mlir::Operation* user : cur->getUsers()) {
      if (ancestors_of_consumer.contains(user)) {
        continue;
      }
      if (auto sl = llvm::dyn_cast<mlir::stablehlo::SliceOp>(user)) {
        auto spec = MatchSingleAxisUnitSlice(sl);
        if (spec && spec->dim == src_dim && spec->start == start &&
            spec->limit == limit) {
          continue;
        }
        return false;
      }
      if ((IsSupportedElementwiseOp(user) ||
           llvm::isa<mlir::stablehlo::ReshapeOp,
                     mlir::stablehlo::BroadcastInDimOp>(user)) &&
          // GENERIC: capability bounded forward cone traversal cap
          cone.size() < 24) {
        if (cone.insert(user).second) {
          fwd_worklist.push_back(user);
        }
        continue;
      }
      return false;
    }
  }
  return true;
}

// Hoists unit-stride single-axis SliceOps upstream across closed single-sink
// subgraphs in reverse topological block order.
bool HoistSingleSinkSlices(mlir::func::FuncOp func) {
  bool any_changed = false;
  for (mlir::Block& block : func.getBody()) {
    for (mlir::Operation& op :
         llvm::make_early_inc_range(llvm::reverse(block))) {
      if (op.getNumResults() == 0 || op.use_empty()) {
        continue;
      }
      if (llvm::isa<mlir::stablehlo::SliceOp, mlir::stablehlo::ConstantOp>(
              &op)) {
        continue;
      }

      // Check that EVERY user of EVERY result of `op` is a single-axis
      // unit-stride SliceOp with the exact same (dim, start, limit).
      std::optional<SingleAxisSliceSpec> common_spec;
      llvm::SmallVector<mlir::stablehlo::SliceOp> user_slices;
      bool all_users_match = true;
      for (mlir::Value res : op.getResults()) {
        if (res.use_empty()) {
          all_users_match = false;
          break;
        }
        auto res_ty = GetStaticRankedType(res);
        if (!res_ty) {
          all_users_match = false;
          break;
        }
        for (mlir::Operation* user : res.getUsers()) {
          auto sl = llvm::dyn_cast<mlir::stablehlo::SliceOp>(user);
          if (!sl || sl->getBlock() != &block) {
            all_users_match = false;
            break;
          }
          auto sp = MatchSingleAxisUnitSlice(sl);
          if (!sp) {
            all_users_match = false;
            break;
          }
          if (!common_spec.has_value()) {
            common_spec = *sp;
          } else if (*common_spec != *sp) {
            all_users_match = false;
            break;
          }
          user_slices.push_back(sl);
        }
        if (!all_users_match) {
          break;
        }
      }
      if (!all_users_match || !common_spec.has_value()) {
        continue;
      }

      const int64_t d = common_spec->dim;
      const int64_t r0 = common_spec->start;
      const int64_t r1 = common_spec->limit;
      const int64_t len = r1 - r0;
      mlir::OpBuilder builder(&op);

      // Cache emitted (value, dim, start, len) slices to avoid duplicate
      // SliceOps when multiple operands of `op` slice the same value.
      llvm::DenseMap<std::tuple<mlir::Value, int64_t, int64_t, int64_t>,
                     mlir::Value>
          slice_cache;

      auto get_or_create_slice =
          [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val,
              int64_t dim, int64_t start, int64_t len) -> mlir::Value {
        auto val_ty = GetStaticRankedType(val);
        if (start == 0 && len == val_ty.getDimSize(dim)) {
          return val;
        }
        auto key = std::make_tuple(val, dim, start, len);
        auto it = slice_cache.find(key);
        if (it != slice_cache.end() && it->second) {
          return it->second;
        }
        mlir::OpBuilder::InsertionGuard guard(builder);
        if (mlir::Operation* val_def = val.getDefiningOp()) {
          builder.setInsertionPointAfter(val_def);
        } else {
          builder.setInsertionPointToStart(&block);
        }
        mlir::Value created =
            SliceSingleAxis(builder, loc, val, dim, start, len);
        slice_cache[key] = created;
        return created;
      };

      // Case 1: Pure elementwise op.
      if (IsSupportedElementwiseOp(&op) && op.getNumResults() == 1) {
        auto out_ty = GetStaticRankedType(op.getResult(0));
        // Check normalization safety if any operand is a BroadcastInDimOp.
        bool safe = true;
        for (mlir::Value opnd : op.getOperands()) {
          auto opnd_ty = GetStaticRankedType(opnd);
          if (!opnd_ty || (opnd_ty.getRank() != 0 &&
                           opnd_ty.getRank() != out_ty.getRank())) {
            safe = false;
            break;
          }
          if (auto bcast =
                  opnd.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
            if (!IsBroadcastFromReduceSafeToHoist(bcast, &op, d, r0, r1)) {
              safe = false;
              break;
            }
          }
        }
        if (!safe) {
          continue;
        }
        llvm::SmallVector<mlir::Value> sliced_operands;
        for (mlir::Value opnd : op.getOperands()) {
          auto opnd_ty = GetStaticRankedType(opnd);
          if (opnd_ty.getRank() == 0) {
            sliced_operands.push_back(opnd);
          } else {
            sliced_operands.push_back(
                get_or_create_slice(builder, op.getLoc(), opnd, d, r0, len));
          }
        }
        llvm::SmallVector<int64_t> new_shape(out_ty.getShape().begin(),
                                             out_ty.getShape().end());
        new_shape[d] = len;
        auto new_out_ty =
            mlir::RankedTensorType::get(new_shape, out_ty.getElementType());
        mlir::OperationState state(op.getLoc(), op.getName());
        state.addOperands(sliced_operands);
        state.addTypes(new_out_ty);
        state.addAttributes(op.getAttrs());
        mlir::Operation* new_op =
            builder.insert(mlir::Operation::create(state));
        for (auto sl : user_slices) {
          sl.replaceAllUsesWith(new_op->getResult(0));
          sl.erase();
        }
        op.erase();
        any_changed = true;
        continue;
      }

      // Case 2: BroadcastInDimOp.
      if (auto bcast = llvm::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(&op)) {
        auto out_ty = GetStaticRankedType(bcast.getResult());
        auto in_ty = GetStaticRankedType(bcast.getOperand());
        if (!out_ty || !in_ty) {
          continue;
        }
        llvm::ArrayRef<int64_t> bdims = bcast.getBroadcastDimensions();
        int64_t in_dim = -1;
        for (int64_t i = 0; i < static_cast<int64_t>(bdims.size()); ++i) {
          if (bdims[i] == d) {
            in_dim = i;
            break;
          }
        }
        mlir::Value new_in = bcast.getOperand();
        if (in_dim != -1 && in_ty.getDimSize(in_dim) == out_ty.getDimSize(d)) {
          new_in = get_or_create_slice(builder, bcast.getLoc(), new_in, in_dim,
                                       r0, len);
        }
        llvm::SmallVector<int64_t> new_shape(out_ty.getShape().begin(),
                                             out_ty.getShape().end());
        new_shape[d] = len;
        auto new_out_ty =
            mlir::RankedTensorType::get(new_shape, out_ty.getElementType());
        auto new_bcast = mlir::stablehlo::BroadcastInDimOp::create(
            builder, bcast.getLoc(), new_out_ty, new_in,
            bcast.getBroadcastDimensionsAttr());
        for (auto sl : user_slices) {
          sl.replaceAllUsesWith(new_bcast.getResult());
          sl.erase();
        }
        bcast.erase();
        any_changed = true;
        continue;
      }

      // Case 3: ReshapeOp preserving axis `d`.
      if (auto reshape = llvm::dyn_cast<mlir::stablehlo::ReshapeOp>(&op)) {
        auto in_ty = GetStaticRankedType(reshape.getOperand());
        auto out_ty = GetStaticRankedType(reshape.getResult());
        if (!in_ty || !out_ty) {
          continue;
        }
        auto in_dim =
            MapPreservedReshapeDim(out_ty.getShape(), in_ty.getShape(), d);
        if (!in_dim.has_value()) {
          continue;
        }
        mlir::Value sl_in = get_or_create_slice(
            builder, reshape.getLoc(), reshape.getOperand(), *in_dim, r0, len);
        llvm::SmallVector<int64_t> new_shape(out_ty.getShape().begin(),
                                             out_ty.getShape().end());
        new_shape[d] = len;
        auto new_out_ty =
            mlir::RankedTensorType::get(new_shape, out_ty.getElementType());
        auto new_reshape = mlir::stablehlo::ReshapeOp::create(
            builder, reshape.getLoc(), new_out_ty, sl_in);
        for (auto sl : user_slices) {
          sl.replaceAllUsesWith(new_reshape.getResult());
          sl.erase();
        }
        reshape.erase();
        any_changed = true;
        continue;
      }

      // Case 4: TransposeOp.
      if (auto tr = llvm::dyn_cast<mlir::stablehlo::TransposeOp>(&op)) {
        auto out_ty = GetStaticRankedType(tr.getResult());
        if (!out_ty) {
          continue;
        }
        llvm::ArrayRef<int64_t> perm = tr.getPermutation();
        const int64_t in_dim = perm[d];
        mlir::Value sl_in = get_or_create_slice(
            builder, tr.getLoc(), tr.getOperand(), in_dim, r0, len);
        llvm::SmallVector<int64_t> new_shape(out_ty.getShape().begin(),
                                             out_ty.getShape().end());
        new_shape[d] = len;
        auto new_out_ty =
            mlir::RankedTensorType::get(new_shape, out_ty.getElementType());
        auto new_tr = mlir::stablehlo::TransposeOp::create(
            builder, tr.getLoc(), new_out_ty, sl_in, tr.getPermutationAttr());
        for (auto sl : user_slices) {
          sl.replaceAllUsesWith(new_tr.getResult());
          sl.erase();
        }
        tr.erase();
        any_changed = true;
        continue;
      }

      // Case 5: ReduceOp (orthogonal reduction where `d` is NOT in reduced
      // dimensions; supports both single-input and variadic 2-input LayerNorm).
      if (auto red = llvm::dyn_cast<mlir::stablehlo::ReduceOp>(&op)) {
        auto in0_ty = GetStaticRankedType(red.getInputs()[0]);
        auto out0_ty = GetStaticRankedType(red.getResult(0));
        if (!in0_ty || !out0_ty) {
          continue;
        }
        llvm::ArrayRef<int64_t> red_dims = red.getDimensions();
        // Map output dimension `d` back to input dimension `in_dim`.
        llvm::SmallVector<int64_t> kept_dims;
        for (int64_t i = 0; i < in0_ty.getRank(); ++i) {
          if (!llvm::is_contained(red_dims, i)) {
            kept_dims.push_back(i);
          }
        }
        if (d < 0 || d >= static_cast<int64_t>(kept_dims.size())) {
          continue;
        }
        const int64_t in_dim = kept_dims[d];
        llvm::SmallVector<mlir::Value> sliced_inputs;
        for (mlir::Value in_v : red.getInputs()) {
          sliced_inputs.push_back(get_or_create_slice(builder, red.getLoc(),
                                                      in_v, in_dim, r0, len));
        }
        llvm::SmallVector<mlir::Type> new_res_types;
        for (mlir::Value res : red.getResults()) {
          auto r_ty = GetStaticRankedType(res);
          llvm::SmallVector<int64_t> new_shape(r_ty.getShape().begin(),
                                               r_ty.getShape().end());
          new_shape[d] = len;
          new_res_types.push_back(
              mlir::RankedTensorType::get(new_shape, r_ty.getElementType()));
        }
        auto new_red = mlir::stablehlo::ReduceOp::create(
            builder, red.getLoc(), new_res_types, sliced_inputs,
            red.getInitValues(), red.getDimensionsAttr());
        mlir::IRMapping mapping;
        red.getBody().cloneInto(&new_red.getBody(), mapping);
        for (std::size_t idx = 0; idx < red.getNumResults(); ++idx) {
          mlir::Value old_res = red.getResult(idx);
          mlir::Value new_res = new_red.getResult(idx);
          for (mlir::Operation* user :
               llvm::make_early_inc_range(old_res.getUsers())) {
            user->getResult(0).replaceAllUsesWith(new_res);
            user->erase();
          }
        }
        red.erase();
        any_changed = true;
        continue;
      }

      // Case 6: DotGeneralOp (slicing a batching or free dimension).
      if (auto dot = llvm::dyn_cast<mlir::stablehlo::DotGeneralOp>(&op)) {
        auto lhs_ty = GetStaticRankedType(dot.getLhs());
        auto rhs_ty = GetStaticRankedType(dot.getRhs());
        auto out_ty = GetStaticRankedType(dot.getResult());
        if (!lhs_ty || !rhs_ty || !out_ty) {
          continue;
        }
        auto dnums = dot.getDotDimensionNumbers();
        llvm::ArrayRef<int64_t> b_lhs = dnums.getLhsBatchingDimensions();
        llvm::ArrayRef<int64_t> b_rhs = dnums.getRhsBatchingDimensions();
        llvm::ArrayRef<int64_t> c_lhs = dnums.getLhsContractingDimensions();
        llvm::ArrayRef<int64_t> c_rhs = dnums.getRhsContractingDimensions();

        llvm::SmallVector<int64_t> lhs_free;
        for (int64_t i = 0; i < lhs_ty.getRank(); ++i) {
          if (!llvm::is_contained(b_lhs, i) && !llvm::is_contained(c_lhs, i)) {
            lhs_free.push_back(i);
          }
        }
        llvm::SmallVector<int64_t> rhs_free;
        for (int64_t i = 0; i < rhs_ty.getRank(); ++i) {
          if (!llvm::is_contained(b_rhs, i) && !llvm::is_contained(c_rhs, i)) {
            rhs_free.push_back(i);
          }
        }

        const int64_t num_batch = b_lhs.size();
        const int64_t num_lhs_free = lhs_free.size();
        mlir::Value new_lhs = dot.getLhs();
        mlir::Value new_rhs = dot.getRhs();
        if (d < num_batch) {
          new_lhs = get_or_create_slice(builder, dot.getLoc(), new_lhs,
                                        b_lhs[d], r0, len);
          new_rhs = get_or_create_slice(builder, dot.getLoc(), new_rhs,
                                        b_rhs[d], r0, len);
        } else if (d < num_batch + num_lhs_free) {
          const int64_t l_dim = lhs_free[d - num_batch];
          new_lhs = get_or_create_slice(builder, dot.getLoc(), new_lhs, l_dim,
                                        r0, len);
        } else {
          const int64_t r_dim = rhs_free[d - num_batch - num_lhs_free];
          new_rhs = get_or_create_slice(builder, dot.getLoc(), new_rhs, r_dim,
                                        r0, len);
        }
        llvm::SmallVector<int64_t> new_shape(out_ty.getShape().begin(),
                                             out_ty.getShape().end());
        new_shape[d] = len;
        auto new_out_ty =
            mlir::RankedTensorType::get(new_shape, out_ty.getElementType());
        auto new_dot = mlir::stablehlo::DotGeneralOp::create(
            builder, dot.getLoc(), new_out_ty, new_lhs, new_rhs, dnums,
            dot.getPrecisionConfigAttr(), dot.getAlgorithmAttr());
        for (auto sl : user_slices) {
          sl.replaceAllUsesWith(new_dot.getResult());
          sl.erase();
        }
        dot.erase();
        any_changed = true;
        continue;
      }

      // Case 7: GatherOp (slicing a batch/start_indices dimension).
      if (auto gather = llvm::dyn_cast<mlir::stablehlo::GatherOp>(&op)) {
        auto idx_ty = GetStaticRankedType(gather.getStartIndices());
        auto out_ty = GetStaticRankedType(gather.getResult());
        if (!idx_ty || !out_ty) {
          continue;
        }
        auto dnums = gather.getDimensionNumbers();
        llvm::ArrayRef<int64_t> offset_dims = dnums.getOffsetDims();
        if (llvm::is_contained(offset_dims, d)) {
          continue;
        }
        int64_t batch_ord = 0;
        for (int64_t i = 0; i < d; ++i) {
          if (!llvm::is_contained(offset_dims, i)) {
            ++batch_ord;
          }
        }
        const int64_t iv_dim = dnums.getIndexVectorDim();
        const int64_t idx_dim =
            (iv_dim < idx_ty.getRank() && batch_ord >= iv_dim) ? batch_ord + 1
                                                               : batch_ord;
        if (idx_dim < 0 || idx_dim >= idx_ty.getRank() ||
            idx_ty.getDimSize(idx_dim) != out_ty.getDimSize(d)) {
          continue;
        }
        mlir::Value new_indices =
            get_or_create_slice(builder, gather.getLoc(),
                                gather.getStartIndices(), idx_dim, r0, len);
        llvm::SmallVector<int64_t> new_shape(out_ty.getShape().begin(),
                                             out_ty.getShape().end());
        new_shape[d] = len;
        auto new_out_ty =
            mlir::RankedTensorType::get(new_shape, out_ty.getElementType());
        auto new_gather = mlir::stablehlo::GatherOp::create(
            builder, gather.getLoc(), new_out_ty, gather.getOperand(),
            new_indices, dnums, gather.getSliceSizesAttr(),
            gather.getIndicesAreSortedAttr());
        for (auto sl : user_slices) {
          sl.replaceAllUsesWith(new_gather.getResult());
          sl.erase();
        }
        gather.erase();
        any_changed = true;
        continue;
      }
    }
  }
  return any_changed;
}

// Erases trivially dead side-effect-free operations in reverse order.
void EraseDeadOps(mlir::func::FuncOp func) {
  bool erased = true;
  while (erased) {
    erased = false;
    for (mlir::Block& block : func.getBody()) {
      for (mlir::Operation& op :
           llvm::make_early_inc_range(llvm::reverse(block))) {
        if (!op.hasTrait<mlir::OpTrait::IsTerminator>() && op.use_empty() &&
            mlir::isMemoryEffectFree(&op)) {
          op.erase();
          erased = true;
        }
      }
    }
  }
}

struct PropagateSingleSinkSlicesAndZeroPadsPass
    : public mlir::PassWrapper<PropagateSingleSinkSlicesAndZeroPadsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      PropagateSingleSinkSlicesAndZeroPadsPass)

  llvm::StringRef getArgument() const final {
    return "torch-tpu-propagate-single-sink-slices-and-zero-pads";
  }

  llvm::StringRef getDescription() const final {
    return "Hoists unit-stride SliceOps across closed single-sink subgraphs "
           "and propagates zero-padded slices across homogeneous backward "
           "subgraphs.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    // GENERIC: capability bounded fixed-point iteration cap
    for (int iter = 0; iter < 8; ++iter) {
      bool c1 = CanonicalizeSliceBoundaries(func);
      bool c2 = PropagateZeroPaddedSlicesForward(func);
      EraseDeadOps(func);
      bool c3 = HoistSingleSinkSlices(func);
      EraseDeadOps(func);
      if (!(c1 || c2 || c3)) {
        break;
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createPropagateSingleSinkSlicesAndZeroPadsPass() {
  return std::make_unique<PropagateSingleSinkSlicesAndZeroPadsPass>();
}

}  // namespace torch_tpu
