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

#include "csrc/mlir/passes/factor_cross_axis_reshape_slices_and_hoist_window_cones.h"

#include <cstdint>
#include <memory>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
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

// Returns a static ranked tensor type if `val` has one, else nullptr.
[[nodiscard]] mlir::RankedTensorType GetStaticRankedType(mlir::Value val) {
  auto type = llvm::dyn_cast<mlir::RankedTensorType>(val.getType());
  if (!type || !type.hasStaticShape()) {
    return nullptr;
  }
  return type;
}

// Returns true if `slice_op` has unit strides across all dimensions.
[[nodiscard]] bool IsUnitStrideSlice(mlir::stablehlo::SliceOp slice_op) {
  auto in_ty = GetStaticRankedType(slice_op.getOperand());
  auto out_ty = GetStaticRankedType(slice_op.getResult());
  if (!in_ty || !out_ty) {
    return false;
  }
  for (int64_t s : slice_op.getStrides()) {
    // GENERIC: semantic unit-stride requirement
    if (s != 1) {
      return false;
    }
  }
  llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
  llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();
  for (int64_t i = 0; i < in_ty.getRank(); ++i) {
    if (starts[i] < 0 || limits[i] <= starts[i] ||
        limits[i] > in_ty.getDimSize(i)) {
      return false;
    }
  }
  return true;
}

// Returns true if `(starts, limits)` covers the entire shape of `ty`.
[[nodiscard]] bool IsFullSlice(mlir::RankedTensorType ty,
                               llvm::ArrayRef<int64_t> starts,
                               llvm::ArrayRef<int64_t> limits) {
  for (int64_t i = 0; i < ty.getRank(); ++i) {
    if (starts[i] != 0 || limits[i] != ty.getDimSize(i)) {
      return false;
    }
  }
  return true;
}

// Computes the product of `(limits[i] - starts[i])` across all axes.
[[nodiscard]] int64_t ComputeSliceNumElements(llvm::ArrayRef<int64_t> starts,
                                              llvm::ArrayRef<int64_t> limits) {
  int64_t prod = 1;
  for (std::size_t i = 0; i < starts.size(); ++i) {
    prod *= (limits[i] - starts[i]);
  }
  return prod;
}

// Returns or creates a canonical unit-stride `SliceOp` on `input`, composing
// nested unit-stride `SliceOp`s.
[[nodiscard]] mlir::Value GetOrCreateUnitSlice(mlir::OpBuilder& builder,
                                               mlir::Location loc,
                                               mlir::Value input,
                                               llvm::ArrayRef<int64_t> starts,
                                               llvm::ArrayRef<int64_t> limits) {
  auto in_ty = GetStaticRankedType(input);
  if (!in_ty || IsFullSlice(in_ty, starts, limits)) {
    return input;
  }

  llvm::SmallVector<int64_t, 6> adj_starts(starts.begin(), starts.end());
  llvm::SmallVector<int64_t, 6> adj_limits(limits.begin(), limits.end());
  while (auto inner = input.getDefiningOp<mlir::stablehlo::SliceOp>()) {
    if (!IsUnitStrideSlice(inner)) {
      break;
    }
    llvm::ArrayRef<int64_t> inner_starts = inner.getStartIndices();
    for (std::size_t i = 0; i < adj_starts.size(); ++i) {
      adj_starts[i] += inner_starts[i];
      adj_limits[i] += inner_starts[i];
    }
    input = inner.getOperand();
    in_ty = GetStaticRankedType(input);
  }
  if (IsFullSlice(in_ty, adj_starts, adj_limits)) {
    return input;
  }

  mlir::Block* target_block = builder.getInsertionBlock();
  mlir::Operation* def_op = input.getDefiningOp();
  mlir::OpBuilder::InsertionGuard guard(builder);
  if (def_op && def_op->getBlock() == target_block) {
    builder.setInsertionPointAfter(def_op);
  } else if (llvm::isa<mlir::BlockArgument>(input) && !target_block->empty()) {
    builder.setInsertionPointToStart(target_block);
  }

  const int64_t rank = in_ty.getRank();
  // GENERIC: semantic unit-stride slice step
  llvm::SmallVector<int64_t, 6> strides(rank, 1);
  llvm::SmallVector<int64_t, 6> out_shape(rank, 0);
  for (int64_t i = 0; i < rank; ++i) {
    out_shape[i] = adj_limits[i] - adj_starts[i];
  }
  auto out_ty = mlir::RankedTensorType::get(out_shape, in_ty.getElementType());
  return mlir::stablehlo::SliceOp::create(
      builder, loc, out_ty, input, builder.getDenseI64ArrayAttr(adj_starts),
      builder.getDenseI64ArrayAttr(adj_limits),
      builder.getDenseI64ArrayAttr(strides));
}

// Simplifies identity unit-stride slices, nested `SliceOp(SliceOp(X))` chains,
// duplicate `SliceOp`s on the same operand, and dead view/elementwise ops.
bool SimplifyUnitSlicesAndDeadOps(mlir::func::FuncOp func) {
  bool changed = false;
  for (mlir::Block& block : func.getBody()) {
    llvm::SmallVector<mlir::stablehlo::SliceOp, 32> slices;
    for (mlir::Operation& op : block) {
      if (auto slice_op = llvm::dyn_cast<mlir::stablehlo::SliceOp>(&op)) {
        slices.push_back(slice_op);
      }
    }
    for (mlir::stablehlo::SliceOp slice_op : slices) {
      if (!slice_op || slice_op->getBlock() == nullptr ||
          !IsUnitStrideSlice(slice_op)) {
        continue;
      }
      auto in_ty = GetStaticRankedType(slice_op.getOperand());
      llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
      llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();
      if (IsFullSlice(in_ty, starts, limits)) {
        slice_op.replaceAllUsesWith(slice_op.getOperand());
        slice_op.erase();
        changed = true;
        continue;
      }
      if (auto inner =
              slice_op.getOperand().getDefiningOp<mlir::stablehlo::SliceOp>();
          inner && inner->hasOneUse() && IsUnitStrideSlice(inner)) {
        mlir::OpBuilder builder(slice_op);
        mlir::Value composed = GetOrCreateUnitSlice(
            builder, slice_op.getLoc(), slice_op.getOperand(), starts, limits);
        if (composed != slice_op.getResult()) {
          slice_op.replaceAllUsesWith(composed);
          slice_op.erase();
          if (inner->use_empty()) {
            inner.erase();
          }
          changed = true;
          continue;
        }
      }
      for (mlir::Operation* user : slice_op.getOperand().getUsers()) {
        auto other = llvm::dyn_cast<mlir::stablehlo::SliceOp>(user);
        if (!other || other == slice_op || other->getBlock() != &block ||
            !other->isBeforeInBlock(slice_op) || !IsUnitStrideSlice(other)) {
          continue;
        }
        if (other.getStartIndices() == starts &&
            other.getLimitIndices() == limits) {
          slice_op.replaceAllUsesWith(other.getResult());
          slice_op.erase();
          changed = true;
          break;
        }
      }
    }

    llvm::SmallVector<mlir::Operation*, 32> rev_ops;
    for (mlir::Operation& op : llvm::reverse(block)) {
      rev_ops.push_back(&op);
    }
    for (mlir::Operation* op : rev_ops) {
      if (op && op->getBlock() != nullptr && op->use_empty() &&
          !op->hasTrait<mlir::OpTrait::IsTerminator>() &&
          !llvm::isa<mlir::stablehlo::OptimizationBarrierOp>(op) &&
          mlir::isMemoryEffectFree(op)) {
        op->erase();
        changed = true;
      }
    }
  }
  return changed;
}

// Factors out shared slice axes when a producer `X` has >= 2 users, all of
// which are unit-stride `SliceOp`s in the same block that slice at least one
// common axis `d` to the exact same sub-interval `[start_d, limit_d)`.
bool FactorMultiSliceSharedIntervals(mlir::func::FuncOp func) {
  bool changed = false;
  for (mlir::Block& block : func.getBody()) {
    llvm::SmallVector<mlir::Operation*, 32> ops;
    for (mlir::Operation& op : block) {
      ops.push_back(&op);
    }
    llvm::DenseSet<mlir::Operation*> erased_ops;
    for (mlir::Operation* op_ptr : ops) {
      if (!op_ptr || erased_ops.contains(op_ptr) ||
          op_ptr->getBlock() == nullptr || op_ptr->getNumResults() != 1) {
        continue;
      }
      mlir::Operation& op = *op_ptr;
      mlir::Value val = op.getResult(0);
      auto val_ty = GetStaticRankedType(val);
      if (!val_ty) {
        continue;
      }

      llvm::SmallVector<mlir::stablehlo::SliceOp, 8> slices;
      bool all_unit_slices = true;
      for (mlir::Operation* user : val.getUsers()) {
        auto s = llvm::dyn_cast<mlir::stablehlo::SliceOp>(user);
        if (!s || s->getBlock() != &block || !IsUnitStrideSlice(s)) {
          all_unit_slices = false;
          break;
        }
        slices.push_back(s);
      }
      // GENERIC: semantic multi-consumer check (>= 2 SliceOp users)
      if (!all_unit_slices || slices.size() < 2) {
        continue;
      }

      const int64_t rank = val_ty.getRank();
      llvm::SmallVector<int64_t, 6> shared_starts(rank, 0);
      llvm::SmallVector<int64_t, 6> shared_limits(val_ty.getShape());
      bool has_shared_subslice = false;

      llvm::ArrayRef<int64_t> s0_starts = slices[0].getStartIndices();
      llvm::ArrayRef<int64_t> s0_limits = slices[0].getLimitIndices();
      for (int64_t d = 0; d < rank; ++d) {
        const int64_t st = s0_starts[d];
        const int64_t lim = s0_limits[d];
        if (lim - st >= val_ty.getDimSize(d)) {
          continue;
        }
        bool same_across_all = true;
        for (std::size_t k = 1; k < slices.size(); ++k) {
          if (slices[k].getStartIndices()[d] != st ||
              slices[k].getLimitIndices()[d] != lim) {
            same_across_all = false;
            break;
          }
        }
        if (same_across_all) {
          shared_starts[d] = st;
          shared_limits[d] = lim;
          has_shared_subslice = true;
        }
      }
      if (!has_shared_subslice ||
          ComputeSliceNumElements(shared_starts, shared_limits) <= 1) {
        continue;
      }

      mlir::OpBuilder builder(&block, ++mlir::Block::iterator(&op));
      llvm::SmallVector<int64_t, 6> strides(rank, 1);
      llvm::SmallVector<int64_t, 6> shared_shape(rank, 0);
      for (int64_t d = 0; d < rank; ++d) {
        shared_shape[d] = shared_limits[d] - shared_starts[d];
      }
      auto shared_ty =
          mlir::RankedTensorType::get(shared_shape, val_ty.getElementType());
      auto shared_slice = mlir::stablehlo::SliceOp::create(
          builder, op.getLoc(), shared_ty, val,
          builder.getDenseI64ArrayAttr(shared_starts),
          builder.getDenseI64ArrayAttr(shared_limits),
          builder.getDenseI64ArrayAttr(strides));

      for (mlir::stablehlo::SliceOp s : slices) {
        llvm::SmallVector<int64_t, 6> rel_starts(rank, 0);
        llvm::SmallVector<int64_t, 6> rel_limits(rank, 0);
        for (int64_t d = 0; d < rank; ++d) {
          rel_starts[d] = s.getStartIndices()[d] - shared_starts[d];
          rel_limits[d] = s.getLimitIndices()[d] - shared_starts[d];
        }
        if (IsFullSlice(shared_ty, rel_starts, rel_limits)) {
          s.replaceAllUsesWith(shared_slice.getResult());
        } else {
          mlir::OpBuilder s_builder(s);
          auto rel_slice = mlir::stablehlo::SliceOp::create(
              s_builder, s.getLoc(), s.getType(), shared_slice.getResult(),
              s_builder.getDenseI64ArrayAttr(rel_starts),
              s_builder.getDenseI64ArrayAttr(rel_limits),
              s_builder.getDenseI64ArrayAttr(strides));
          s.replaceAllUsesWith(rel_slice.getResult());
        }
        erased_ops.insert(s);
        s.erase();
      }
      changed = true;
    }
  }
  return changed;
}

// Rewrites `SliceOp(ReshapeOp(X))` across cross-axis split/merge reshapes
// (`[D0..., M, N] -> [D0'..., M * K, C...]` where `N == K * prod(C...)`)
// whenever the slice along `M * K` falls within a single outer row `m0 = lo/K`
// or is aligned to multiples of `K`.
bool FactorCrossAxisReshapeSlices(mlir::func::FuncOp func) {
  bool changed = false;
  for (mlir::Block& block : func.getBody()) {
    llvm::SmallVector<mlir::stablehlo::SliceOp, 32> slices;
    for (mlir::Operation& op : block) {
      if (auto slice_op = llvm::dyn_cast<mlir::stablehlo::SliceOp>(&op)) {
        slices.push_back(slice_op);
      }
    }
    for (mlir::stablehlo::SliceOp slice_op : slices) {
      if (!slice_op || slice_op->getBlock() == nullptr ||
          !IsUnitStrideSlice(slice_op)) {
        continue;
      }
      auto reshape_op =
          slice_op.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
      if (!reshape_op) {
        continue;
      }
      // Require all users of `reshape_op` to be unit-stride `SliceOp`s so
      // `reshape_op` does not remain materialized alongside the sliced reshape.
      bool all_users_slices = true;
      for (mlir::Operation* u : reshape_op->getUsers()) {
        auto us = llvm::dyn_cast<mlir::stablehlo::SliceOp>(u);
        if (!us || !IsUnitStrideSlice(us)) {
          all_users_slices = false;
          break;
        }
      }
      if (!all_users_slices) {
        continue;
      }

      mlir::Value x_val = reshape_op.getOperand();
      auto x_ty = GetStaticRankedType(x_val);
      auto r_ty = GetStaticRankedType(reshape_op.getResult());
      auto s_ty = GetStaticRankedType(slice_op.getResult());
      if (!x_ty || !r_ty || !s_ty || s_ty.getNumElements() <= 1) {
        continue;
      }

      llvm::ArrayRef<int64_t> a_shape = x_ty.getShape();
      llvm::ArrayRef<int64_t> b_shape = r_ty.getShape();
      llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
      llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();
      const int64_t p_rank = x_ty.getRank();
      const int64_t q_rank = r_ty.getRank();

      // Find the first sliced dimension `b_idx` of `r_ty`.
      int64_t b_idx = -1;
      for (int64_t j = 0; j < q_rank; ++j) {
        if (limits[j] - starts[j] < b_shape[j]) {
          b_idx = j;
          break;
        }
      }
      if (b_idx < 0) {
        continue;
      }

      const int64_t lo = starts[b_idx];
      const int64_t hi = limits[b_idx];
      const int64_t b_len = b_shape[b_idx];
      int64_t out_prefix = 1;
      for (int64_t j = 0; j < b_idx; ++j) {
        out_prefix *= b_shape[j];
      }
      int64_t out_suffix = 1;
      for (int64_t j = b_idx + 1; j < q_rank; ++j) {
        out_suffix *= b_shape[j];
      }

      bool matched = false;
      llvm::SmallVector<int64_t, 6> x_starts(p_rank, 0);
      llvm::SmallVector<int64_t, 6> x_limits(a_shape);

      for (int64_t a_inner = 0; a_inner < p_rank && !matched; ++a_inner) {
        int64_t in_suffix = 1;
        for (int64_t i = a_inner + 1; i < p_rank; ++i) {
          in_suffix *= a_shape[i];
        }
        if (in_suffix <= 0 || out_suffix % in_suffix != 0) {
          continue;
        }
        const int64_t block_stride = out_suffix / in_suffix;
        const int64_t n_ext = a_shape[a_inner];
        if (block_stride <= 0 || n_ext % block_stride != 0) {
          continue;
        }
        const int64_t k_ext = n_ext / block_stride;
        if (k_ext <= 0 || b_len % k_ext != 0) {
          continue;
        }
        const int64_t m_ext = b_len / k_ext;

        for (int64_t a_outer = 0; a_outer <= a_inner && !matched; ++a_outer) {
          if (a_outer == a_inner) {
            if (m_ext != 1) {
              continue;
            }
          } else {
            if (a_shape[a_outer] != m_ext) {
              continue;
            }
            bool unit_between = true;
            for (int64_t mid = a_outer + 1; mid < a_inner; ++mid) {
              if (a_shape[mid] != 1) {
                unit_between = false;
                break;
              }
            }
            if (!unit_between) {
              continue;
            }
          }

          int64_t in_prefix = 1;
          for (int64_t i = 0; i < a_outer; ++i) {
            in_prefix *= a_shape[i];
          }
          if (in_prefix != out_prefix) {
            continue;
          }

          // Skip trivial 1-to-1 preserved-axis reshapes (`M == 1 &&
          // block_stride == 1`), which are handled directly by the unit-dim
          // reshape slice hoister.
          if (m_ext == 1 && block_stride == 1) {
            continue;
          }

          // Case A: `[lo, hi)` falls inside a single outer index `m0 = lo / K`.
          if (lo / k_ext == (hi - 1) / k_ext) {
            const int64_t m0 = lo / k_ext;
            const int64_t k_lo = lo % k_ext;
            const int64_t k_hi = (hi - 1) % k_ext + 1;
            if (m_ext == 1 && (k_hi - k_lo) >= k_ext) {
              continue;
            }
            if (a_outer < a_inner) {
              x_starts[a_outer] = m0;
              x_limits[a_outer] = m0 + 1;
            }
            x_starts[a_inner] = k_lo * block_stride;
            x_limits[a_inner] = k_hi * block_stride;
            if (ComputeSliceNumElements(x_starts, x_limits) > 1) {
              matched = true;
            }
            continue;
          }

          // Case B: `[lo, hi)` is aligned to multiples of `K` across `< M`
          // outer rows.
          if (a_outer < a_inner && lo % k_ext == 0 && hi % k_ext == 0 &&
              (hi - lo) < b_len) {
            x_starts[a_outer] = lo / k_ext;
            x_limits[a_outer] = hi / k_ext;
            if (ComputeSliceNumElements(x_starts, x_limits) > 1) {
              matched = true;
            }
          }
        }
      }

      if (!matched) {
        continue;
      }

      mlir::OpBuilder builder(slice_op);
      mlir::Value x_sl = GetOrCreateUnitSlice(builder, slice_op.getLoc(), x_val,
                                              x_starts, x_limits);
      llvm::SmallVector<int64_t, 6> r_sl_shape(b_shape);
      r_sl_shape[b_idx] = hi - lo;
      auto r_sl_ty =
          mlir::RankedTensorType::get(r_sl_shape, r_ty.getElementType());
      mlir::Value r_sl = mlir::stablehlo::ReshapeOp::create(
          builder, slice_op.getLoc(), r_sl_ty, x_sl);

      llvm::SmallVector<int64_t, 6> rem_starts(starts);
      llvm::SmallVector<int64_t, 6> rem_limits(limits);
      rem_starts[b_idx] = 0;
      rem_limits[b_idx] = hi - lo;
      mlir::Value final_val = GetOrCreateUnitSlice(
          builder, slice_op.getLoc(), r_sl, rem_starts, rem_limits);

      slice_op.replaceAllUsesWith(final_val);
      slice_op.erase();
      if (reshape_op->use_empty()) {
        reshape_op.erase();
      }
      changed = true;
    }
  }
  return changed;
}

// Returns true if `op` is a same-shape pointwise elementwise operation.
[[nodiscard]] bool IsSameShapePointwiseOp(mlir::Operation* op) {
  if (!llvm::isa<mlir::stablehlo::ConvertOp, mlir::stablehlo::NegOp,
                 mlir::stablehlo::ExpOp, mlir::stablehlo::Expm1Op,
                 mlir::stablehlo::LogOp, mlir::stablehlo::Log1pOp,
                 mlir::stablehlo::SqrtOp, mlir::stablehlo::RsqrtOp,
                 mlir::stablehlo::TanhOp, mlir::stablehlo::LogisticOp,
                 mlir::stablehlo::SineOp, mlir::stablehlo::CosineOp,
                 mlir::stablehlo::AbsOp, mlir::stablehlo::CeilOp,
                 mlir::stablehlo::FloorOp, mlir::stablehlo::SignOp,
                 mlir::stablehlo::NotOp, mlir::chlo::ErfOp,
                 mlir::stablehlo::AddOp, mlir::stablehlo::SubtractOp,
                 mlir::stablehlo::MulOp, mlir::stablehlo::DivOp,
                 mlir::stablehlo::RemOp, mlir::stablehlo::MaxOp,
                 mlir::stablehlo::MinOp, mlir::stablehlo::PowOp,
                 mlir::stablehlo::AndOp, mlir::stablehlo::OrOp,
                 mlir::stablehlo::XorOp, mlir::stablehlo::Atan2Op,
                 mlir::stablehlo::CompareOp, mlir::stablehlo::SelectOp>(op)) {
    return false;
  }
  if (op->getNumResults() != 1) {
    return false;
  }
  auto res_ty = GetStaticRankedType(op->getResult(0));
  if (!res_ty) {
    return false;
  }
  for (mlir::Value opnd : op->getOperands()) {
    auto opnd_ty = GetStaticRankedType(opnd);
    if (!opnd_ty) {
      return false;
    }
    if (llvm::isa<mlir::stablehlo::SelectOp>(op) && opnd == op->getOperand(0) &&
        opnd_ty.getRank() == 0) {
      continue;
    }
    if (opnd_ty.getShape() != res_ty.getShape()) {
      return false;
    }
  }
  return true;
}

// Maps each sliced dimension `d` of `dst_shape` (`limits[d] - starts[d] <
// dst_shape[d]`) back to a preserved dimension of `src_shape` when `ReshapeOp`
// only inserts or removes unit dimensions around the sliced axes.
[[nodiscard]] bool MapPreservedSlicedDimsToReshapeInput(
    llvm::ArrayRef<int64_t> src_shape, llvm::ArrayRef<int64_t> dst_shape,
    llvm::ArrayRef<int64_t> dst_starts, llvm::ArrayRef<int64_t> dst_limits,
    llvm::SmallVectorImpl<int64_t>& src_starts,
    llvm::SmallVectorImpl<int64_t>& src_limits) {
  const int64_t src_rank = static_cast<int64_t>(src_shape.size());
  const int64_t dst_rank = static_cast<int64_t>(dst_shape.size());
  src_starts.assign(src_rank, 0);
  src_limits.assign(src_shape.begin(), src_shape.end());

  llvm::DenseSet<int64_t> used_src_axes;
  for (int64_t d = 0; d < dst_rank; ++d) {
    if (dst_starts[d] == 0 && dst_limits[d] == dst_shape[d]) {
      continue;
    }
    int64_t dst_prefix = 1;
    for (int64_t j = 0; j < d; ++j) {
      dst_prefix *= dst_shape[j];
    }
    int64_t dst_suffix = 1;
    for (int64_t j = d + 1; j < dst_rank; ++j) {
      dst_suffix *= dst_shape[j];
    }

    int64_t matched_src = -1;
    int64_t src_prefix = 1;
    for (int64_t i = 0; i < src_rank; ++i) {
      if (src_prefix == dst_prefix && src_shape[i] == dst_shape[d] &&
          !used_src_axes.contains(i)) {
        int64_t src_suffix = 1;
        for (int64_t k = i + 1; k < src_rank; ++k) {
          src_suffix *= src_shape[k];
        }
        if (src_suffix == dst_suffix) {
          matched_src = i;
          break;
        }
      }
      src_prefix *= src_shape[i];
    }
    if (matched_src < 0) {
      return false;
    }
    used_src_axes.insert(matched_src);
    src_starts[matched_src] = dst_starts[d];
    src_limits[matched_src] = dst_limits[d];
  }
  return true;
}

// Hoists `slice_op = SliceOp(prod)` one step upstream across `prod` when `prod`
// has a single use (`slice_op`) and all resulting sliced tensors have `> 1`
// elements.
bool TryHoistSingleUseSliceStep(mlir::stablehlo::SliceOp slice_op) {
  if (!IsUnitStrideSlice(slice_op)) {
    return false;
  }
  mlir::Value src = slice_op.getOperand();
  if (!src.hasOneUse()) {
    return false;
  }
  mlir::Operation* prod = src.getDefiningOp();
  if (!prod || prod->getBlock() != slice_op->getBlock()) {
    return false;
  }

  auto in_ty = GetStaticRankedType(src);
  auto out_ty = GetStaticRankedType(slice_op.getResult());
  if (!in_ty || !out_ty || out_ty.getNumElements() <= 1) {
    return false;
  }
  llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
  llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();
  const int64_t rank = in_ty.getRank();
  mlir::OpBuilder builder(slice_op);
  mlir::Location loc = slice_op.getLoc();

  // 1. Same-shape pointwise elementwise operations.
  if (IsSameShapePointwiseOp(prod)) {
    mlir::IRMapping mapping;
    for (mlir::Value opnd : prod->getOperands()) {
      if (mapping.contains(opnd)) {
        continue;
      }
      auto opnd_ty = GetStaticRankedType(opnd);
      if (opnd_ty.getRank() == 0) {
        mapping.map(opnd, opnd);
      } else {
        mapping.map(opnd,
                    GetOrCreateUnitSlice(builder, loc, opnd, starts, limits));
      }
    }
    mlir::Operation* cloned = builder.clone(*prod, mapping);
    cloned->getResult(0).setType(out_ty);
    slice_op.replaceAllUsesWith(cloned->getResult(0));
    slice_op.erase();
    prod->erase();
    return true;
  }

  // 2. BroadcastInDimOp: hoist when the sliced operand has > 1 elements (or is
  // a rank-0 scalar); if the sliced operand would have 1 element, shrink any
  // sliced broadcast-expanded dimensions on `BroadcastInDimOp` without slicing
  // the operand down to 1 element.
  if (auto bcast = llvm::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(prod)) {
    mlir::Value b_in = bcast.getOperand();
    auto b_in_ty = GetStaticRankedType(b_in);
    if (!b_in_ty) {
      return false;
    }
    if (b_in_ty.getRank() == 0) {
      auto new_bcast = mlir::stablehlo::BroadcastInDimOp::create(
          builder, loc, out_ty, b_in, bcast.getBroadcastDimensionsAttr());
      slice_op.replaceAllUsesWith(new_bcast.getResult());
      slice_op.erase();
      bcast.erase();
      return true;
    }

    const int64_t b_rank = b_in_ty.getRank();
    llvm::ArrayRef<int64_t> b_dims = bcast.getBroadcastDimensions();
    llvm::SmallVector<int64_t, 6> b_starts(b_rank, 0);
    llvm::SmallVector<int64_t, 6> b_limits(b_in_ty.getShape());
    llvm::DenseSet<int64_t> mapped_out_dims;
    for (int64_t k = 0; k < b_rank; ++k) {
      const int64_t out_d = b_dims[k];
      if (b_in_ty.getDimSize(k) > 1) {
        b_starts[k] = starts[out_d];
        b_limits[k] = limits[out_d];
        mapped_out_dims.insert(out_d);
      }
    }

    if (ComputeSliceNumElements(b_starts, b_limits) > 1) {
      mlir::Value b_in_sl =
          GetOrCreateUnitSlice(builder, loc, b_in, b_starts, b_limits);
      auto new_bcast = mlir::stablehlo::BroadcastInDimOp::create(
          builder, loc, out_ty, b_in_sl, bcast.getBroadcastDimensionsAttr());
      slice_op.replaceAllUsesWith(new_bcast.getResult());
      slice_op.erase();
      bcast.erase();
      return true;
    }

    // If slicing `b_in` would reduce it to 1 element, shrink any sliced
    // broadcast-expanded axes (`d not in mapped_out_dims`) on `bcast` itself.
    bool has_expanded_slice = false;
    llvm::SmallVector<int64_t, 6> mid_shape(in_ty.getShape().begin(),
                                            in_ty.getShape().end());
    llvm::SmallVector<int64_t, 6> rem_starts(starts.begin(), starts.end());
    llvm::SmallVector<int64_t, 6> rem_limits(limits.begin(), limits.end());
    for (int64_t d = 0; d < rank; ++d) {
      if (!mapped_out_dims.contains(d) &&
          (limits[d] - starts[d]) < in_ty.getDimSize(d)) {
        mid_shape[d] = limits[d] - starts[d];
        rem_starts[d] = 0;
        rem_limits[d] = mid_shape[d];
        has_expanded_slice = true;
      }
    }
    if (!has_expanded_slice) {
      return false;
    }
    auto mid_ty =
        mlir::RankedTensorType::get(mid_shape, in_ty.getElementType());
    auto mid_bcast = mlir::stablehlo::BroadcastInDimOp::create(
        builder, loc, mid_ty, b_in, bcast.getBroadcastDimensionsAttr());
    mlir::Value final_sl = GetOrCreateUnitSlice(
        builder, loc, mid_bcast.getResult(), rem_starts, rem_limits);
    slice_op.replaceAllUsesWith(final_sl);
    slice_op.erase();
    bcast.erase();
    return true;
  }

  // 3. ReverseOp.
  if (auto rev = llvm::dyn_cast<mlir::stablehlo::ReverseOp>(prod)) {
    llvm::DenseSet<int64_t> rev_dims(rev.getDimensions().begin(),
                                     rev.getDimensions().end());
    llvm::SmallVector<int64_t, 6> r_starts(rank, 0);
    llvm::SmallVector<int64_t, 6> r_limits(rank, 0);
    for (int64_t d = 0; d < rank; ++d) {
      if (rev_dims.contains(d)) {
        r_starts[d] = in_ty.getDimSize(d) - limits[d];
        r_limits[d] = in_ty.getDimSize(d) - starts[d];
      } else {
        r_starts[d] = starts[d];
        r_limits[d] = limits[d];
      }
    }
    mlir::Value in_sl = GetOrCreateUnitSlice(builder, loc, rev.getOperand(),
                                             r_starts, r_limits);
    auto new_rev = mlir::stablehlo::ReverseOp::create(
        builder, loc, out_ty, in_sl, rev.getDimensionsAttr());
    slice_op.replaceAllUsesWith(new_rev.getResult());
    slice_op.erase();
    rev.erase();
    return true;
  }

  // 4. TransposeOp.
  if (auto tr = llvm::dyn_cast<mlir::stablehlo::TransposeOp>(prod)) {
    llvm::ArrayRef<int64_t> perm = tr.getPermutation();
    llvm::SmallVector<int64_t, 6> t_starts(rank, 0);
    llvm::SmallVector<int64_t, 6> t_limits(rank, 0);
    for (int64_t d = 0; d < rank; ++d) {
      t_starts[perm[d]] = starts[d];
      t_limits[perm[d]] = limits[d];
    }
    mlir::Value in_sl =
        GetOrCreateUnitSlice(builder, loc, tr.getOperand(), t_starts, t_limits);
    auto new_tr = mlir::stablehlo::TransposeOp::create(
        builder, loc, out_ty, in_sl, tr.getPermutationAttr());
    slice_op.replaceAllUsesWith(new_tr.getResult());
    slice_op.erase();
    tr.erase();
    return true;
  }

  // 5. ConcatenateOp (when slicing axes orthogonal to concat_dim, or falling
  // within a single operand along concat_dim).
  if (auto concat = llvm::dyn_cast<mlir::stablehlo::ConcatenateOp>(prod)) {
    const int64_t c_dim = static_cast<int64_t>(concat.getDimension());
    if (starts[c_dim] == 0 && limits[c_dim] == in_ty.getDimSize(c_dim)) {
      llvm::SmallVector<mlir::Value, 4> new_inputs;
      for (mlir::Value c_in : concat.getInputs()) {
        auto c_in_ty = GetStaticRankedType(c_in);
        if (!c_in_ty) {
          return false;
        }
        llvm::SmallVector<int64_t, 6> c_starts(starts);
        llvm::SmallVector<int64_t, 6> c_limits(limits);
        c_starts[c_dim] = 0;
        c_limits[c_dim] = c_in_ty.getDimSize(c_dim);
        if (ComputeSliceNumElements(c_starts, c_limits) <= 1) {
          return false;
        }
        new_inputs.push_back(
            GetOrCreateUnitSlice(builder, loc, c_in, c_starts, c_limits));
      }
      auto new_concat = mlir::stablehlo::ConcatenateOp::create(
          builder, loc, out_ty, new_inputs, concat.getDimension());
      slice_op.replaceAllUsesWith(new_concat.getResult());
      slice_op.erase();
      concat.erase();
      return true;
    }

    int64_t offset = 0;
    for (mlir::Value c_in : concat.getInputs()) {
      auto c_in_ty = GetStaticRankedType(c_in);
      if (!c_in_ty) {
        return false;
      }
      const int64_t ext = c_in_ty.getDimSize(c_dim);
      if (starts[c_dim] >= offset && limits[c_dim] <= offset + ext) {
        llvm::SmallVector<int64_t, 6> c_starts(starts);
        llvm::SmallVector<int64_t, 6> c_limits(limits);
        c_starts[c_dim] = starts[c_dim] - offset;
        c_limits[c_dim] = limits[c_dim] - offset;
        mlir::Value repl =
            GetOrCreateUnitSlice(builder, loc, c_in, c_starts, c_limits);
        slice_op.replaceAllUsesWith(repl);
        slice_op.erase();
        concat.erase();
        return true;
      }
      offset += ext;
    }
    return false;
  }

  // 6. ReshapeOp (preserving all sliced axes across unit-dimension insertions
  // or removals).
  if (auto resh = llvm::dyn_cast<mlir::stablehlo::ReshapeOp>(prod)) {
    auto r_in_ty = GetStaticRankedType(resh.getOperand());
    if (!r_in_ty) {
      return false;
    }
    llvm::SmallVector<int64_t, 6> r_starts;
    llvm::SmallVector<int64_t, 6> r_limits;
    if (!MapPreservedSlicedDimsToReshapeInput(r_in_ty.getShape(),
                                              in_ty.getShape(), starts, limits,
                                              r_starts, r_limits)) {
      return false;
    }
    if (ComputeSliceNumElements(r_starts, r_limits) <= 1) {
      return false;
    }
    mlir::Value r_in_sl = GetOrCreateUnitSlice(builder, loc, resh.getOperand(),
                                               r_starts, r_limits);
    auto new_resh =
        mlir::stablehlo::ReshapeOp::create(builder, loc, out_ty, r_in_sl);
    slice_op.replaceAllUsesWith(new_resh.getResult());
    slice_op.erase();
    resh.erase();
    return true;
  }

  // 7. DotGeneralOp (slicing non-contracting batch and/or LHS/RHS free dims).
  if (auto dot_gen = llvm::dyn_cast<mlir::stablehlo::DotGeneralOp>(prod)) {
    mlir::Value lhs = dot_gen.getLhs();
    mlir::Value rhs = dot_gen.getRhs();
    auto lhs_ty = GetStaticRankedType(lhs);
    auto rhs_ty = GetStaticRankedType(rhs);
    if (!lhs_ty || !rhs_ty) {
      return false;
    }

    auto dnums = dot_gen.getDotDimensionNumbers();
    llvm::ArrayRef<int64_t> lhs_batch = dnums.getLhsBatchingDimensions();
    llvm::ArrayRef<int64_t> rhs_batch = dnums.getRhsBatchingDimensions();
    llvm::ArrayRef<int64_t> lhs_contract = dnums.getLhsContractingDimensions();
    llvm::ArrayRef<int64_t> rhs_contract = dnums.getRhsContractingDimensions();

    llvm::SmallVector<int64_t, 4> lhs_free;
    for (int64_t i = 0; i < lhs_ty.getRank(); ++i) {
      if (!llvm::is_contained(lhs_batch, i) &&
          !llvm::is_contained(lhs_contract, i)) {
        lhs_free.push_back(i);
      }
    }
    llvm::SmallVector<int64_t, 4> rhs_free;
    for (int64_t i = 0; i < rhs_ty.getRank(); ++i) {
      if (!llvm::is_contained(rhs_batch, i) &&
          !llvm::is_contained(rhs_contract, i)) {
        rhs_free.push_back(i);
      }
    }

    const int64_t n_batch = static_cast<int64_t>(lhs_batch.size());
    const int64_t n_lhs_free = static_cast<int64_t>(lhs_free.size());
    const int64_t n_rhs_free = static_cast<int64_t>(rhs_free.size());
    if (n_batch + n_lhs_free + n_rhs_free != rank) {
      return false;
    }

    llvm::SmallVector<int64_t, 6> lhs_starts(lhs_ty.getRank(), 0);
    llvm::SmallVector<int64_t, 6> lhs_limits(lhs_ty.getShape());
    llvm::SmallVector<int64_t, 6> rhs_starts(rhs_ty.getRank(), 0);
    llvm::SmallVector<int64_t, 6> rhs_limits(rhs_ty.getShape());

    for (int64_t b = 0; b < n_batch; ++b) {
      lhs_starts[lhs_batch[b]] = starts[b];
      lhs_limits[lhs_batch[b]] = limits[b];
      rhs_starts[rhs_batch[b]] = starts[b];
      rhs_limits[rhs_batch[b]] = limits[b];
    }
    for (int64_t lf = 0; lf < n_lhs_free; ++lf) {
      const int64_t out_d = n_batch + lf;
      lhs_starts[lhs_free[lf]] = starts[out_d];
      lhs_limits[lhs_free[lf]] = limits[out_d];
    }
    for (int64_t rf = 0; rf < n_rhs_free; ++rf) {
      const int64_t out_d = n_batch + n_lhs_free + rf;
      rhs_starts[rhs_free[rf]] = starts[out_d];
      rhs_limits[rhs_free[rf]] = limits[out_d];
    }

    if (ComputeSliceNumElements(lhs_starts, lhs_limits) <= 1 ||
        ComputeSliceNumElements(rhs_starts, rhs_limits) <= 1) {
      return false;
    }

    mlir::Value lhs_sl =
        GetOrCreateUnitSlice(builder, loc, lhs, lhs_starts, lhs_limits);
    mlir::Value rhs_sl =
        GetOrCreateUnitSlice(builder, loc, rhs, rhs_starts, rhs_limits);
    auto new_dot = mlir::stablehlo::DotGeneralOp::create(
        builder, loc, out_ty, lhs_sl, rhs_sl, dnums,
        dot_gen.getPrecisionConfigAttr(), dot_gen.getAlgorithmAttr());
    slice_op.replaceAllUsesWith(new_dot.getResult());
    slice_op.erase();
    dot_gen.erase();
    return true;
  }

  // 8. 2D DotOp.
  if (auto dot_op = llvm::dyn_cast<mlir::stablehlo::DotOp>(prod)) {
    mlir::Value lhs = dot_op.getLhs();
    mlir::Value rhs = dot_op.getRhs();
    auto lhs_ty = GetStaticRankedType(lhs);
    auto rhs_ty = GetStaticRankedType(rhs);
    // GENERIC: semantic rank-2 matrix-matrix DotOp check
    if (!lhs_ty || !rhs_ty || lhs_ty.getRank() != 2 || rhs_ty.getRank() != 2 ||
        rank != 2) {
      return false;
    }
    llvm::SmallVector<int64_t, 2> lhs_starts = {starts[0], 0};
    llvm::SmallVector<int64_t, 2> lhs_limits = {limits[0],
                                                lhs_ty.getDimSize(1)};
    llvm::SmallVector<int64_t, 2> rhs_starts = {0, starts[1]};
    llvm::SmallVector<int64_t, 2> rhs_limits = {rhs_ty.getDimSize(0),
                                                limits[1]};
    if (ComputeSliceNumElements(lhs_starts, lhs_limits) <= 1 ||
        ComputeSliceNumElements(rhs_starts, rhs_limits) <= 1) {
      return false;
    }
    mlir::Value lhs_sl =
        GetOrCreateUnitSlice(builder, loc, lhs, lhs_starts, lhs_limits);
    mlir::Value rhs_sl =
        GetOrCreateUnitSlice(builder, loc, rhs, rhs_starts, rhs_limits);
    auto new_dot = mlir::stablehlo::DotOp::create(
        builder, loc, out_ty, lhs_sl, rhs_sl, dot_op.getPrecisionConfigAttr());
    slice_op.replaceAllUsesWith(new_dot.getResult());
    slice_op.erase();
    dot_op.erase();
    return true;
  }

  return false;
}

// Hoists single-use unit-stride slices upstream within func.
bool HoistSingleUseSlicesInFunc(mlir::func::FuncOp func) {
  bool changed = false;
  for (mlir::Block& block : func.getBody()) {
    llvm::SmallVector<mlir::stablehlo::SliceOp, 32> slices;
    for (mlir::Operation& op : llvm::reverse(block)) {
      if (auto slice_op = llvm::dyn_cast<mlir::stablehlo::SliceOp>(&op)) {
        slices.push_back(slice_op);
      }
    }
    for (mlir::stablehlo::SliceOp slice_op : slices) {
      if (slice_op && slice_op->getBlock() != nullptr) {
        if (TryHoistSingleUseSliceStep(slice_op)) {
          changed = true;
        }
      }
    }
  }
  return changed;
}

// Pass that factors shared multi-slice bounding boxes, pushes block-aligned
// unit-stride slices across cross-axis split/merge ReshapeOps, and hoists
// single-window slices upstream across elementwise, broadcast, reverse,
// transpose, concatenate, reshape, and non-contracting DotGeneral/Dot cones.
class FactorCrossAxisReshapeSlicesAndHoistWindowConesPass
    : public mlir::PassWrapper<
          FactorCrossAxisReshapeSlicesAndHoistWindowConesPass,
          mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      FactorCrossAxisReshapeSlicesAndHoistWindowConesPass)

  llvm::StringRef getArgument() const final {
    return "factor-cross-axis-reshape-slices-and-hoist-window-cones";
  }

  llvm::StringRef getDescription() const final {
    return "Factors multi-slice bounding boxes, pushes block-aligned slices "
           "through cross-axis reshapes, and hoists window cones.";
  }

  void runOnOperation() final {
    mlir::func::FuncOp func = getOperation();
    constexpr int kMaxIterations = 32;
    for (int iter = 0; iter < kMaxIterations; ++iter) {
      bool progress = false;
      progress |= SimplifyUnitSlicesAndDeadOps(func);
      progress |= FactorMultiSliceSharedIntervals(func);
      progress |= FactorCrossAxisReshapeSlices(func);
      progress |= HoistSingleUseSlicesInFunc(func);
      if (!progress) {
        break;
      }
    }
    SimplifyUnitSlicesAndDeadOps(func);
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createFactorCrossAxisReshapeSlicesAndHoistWindowConesPass() {
  return std::make_unique<
      FactorCrossAxisReshapeSlicesAndHoistWindowConesPass>();
}

}  // namespace torch_tpu
