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

#include "csrc/mlir/passes/canonicalize_slices/canonicalize_slices.h"  // NOLINT

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/absl_log.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // NOLINT
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"  // NOLINT
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"  // NOLINT
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

// Returns the static ranked tensor type of val, or nullptr if dynamic/invalid.
mlir::RankedTensorType GetStaticRankedType(mlir::Value val) {
  if (!val) {
    return nullptr;
  }
  auto ty = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
  if (!ty || !ty.hasStaticShape()) {
    return nullptr;
  }
  return ty;
}

// Returns true if all strides of the slice operation are 1.
[[nodiscard]] bool IsUnitStrideSlice(mlir::stablehlo::SliceOp slice_op) {
  return llvm::all_of(slice_op.getStrides(), [](int64_t s) { return s == 1; });
}

// Returns the scalar integer constant value of val if available.
std::optional<int64_t> GetConstantScalarInt(mlir::Value val) {
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return std::nullopt;
  }
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!dense || !dense.isSplat() ||
      !mlir::isa<mlir::IntegerType>(dense.getElementType())) {
    return std::nullopt;
  }
  auto int_type = mlir::cast<mlir::IntegerType>(dense.getElementType());
  llvm::APInt splat_val = dense.getSplatValue<llvm::APInt>();
  if (int_type.isUnsignedInteger()) {
    const uint64_t uval = splat_val.getZExtValue();
    if (uval > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return std::numeric_limits<int64_t>::max();
    }
    return static_cast<int64_t>(uval);
  }
  return splat_val.getSExtValue();
}

// Computes the total element count of a slice given its start and limit
// indices.
int64_t ComputeSliceNumElements(llvm::ArrayRef<int64_t> starts,
                                llvm::ArrayRef<int64_t> limits) {
  int64_t n = 1;
  for (std::size_t i = 0; i < starts.size(); ++i) {
    const int64_t len = limits[i] - starts[i];
    if (len <= 0) {
      return 0;
    }
    n *= len;
  }
  return n;
}

// Creates a unit-stride slice operation or returns src directly if full-extent.
mlir::Value GetOrCreateUnitSlice(mlir::OpBuilder& builder, mlir::Location loc,
                                 mlir::Value src,
                                 llvm::ArrayRef<int64_t> starts,
                                 llvm::ArrayRef<int64_t> limits,
                                 bool fold_parent_slice = true) {
  auto src_ty = GetStaticRankedType(src);
  if (!src_ty) {
    return nullptr;
  }
  const int64_t rank = src_ty.getRank();
  bool is_full = true;
  for (int64_t d = 0; d < rank; ++d) {
    if (starts[d] != 0 || limits[d] != src_ty.getDimSize(d)) {
      is_full = false;
      break;
    }
  }
  if (is_full) {
    return src;
  }

  // Compose nested unit-stride slices: slice(slice(x, p_starts), starts)
  // folds directly to slice(x, p_starts + starts).
  if (fold_parent_slice) {
    if (auto parent_slice = src.getDefiningOp<mlir::stablehlo::SliceOp>()) {
      if (IsUnitStrideSlice(parent_slice) &&
          GetStaticRankedType(parent_slice.getOperand()) != nullptr) {
        llvm::ArrayRef<int64_t> p_starts = parent_slice.getStartIndices();
        llvm::SmallVector<int64_t, 6> comp_starts(rank);
        llvm::SmallVector<int64_t, 6> comp_limits(rank);
        for (int64_t d = 0; d < rank; ++d) {
          comp_starts[d] = p_starts[d] + starts[d];
          comp_limits[d] = p_starts[d] + limits[d];
        }
        return GetOrCreateUnitSlice(builder, loc, parent_slice.getOperand(),
                                    comp_starts, comp_limits,
                                    fold_parent_slice);
      }
    }
  }

  llvm::SmallVector<int64_t, 6> strides(rank, 1);
  llvm::SmallVector<int64_t, 6> slice_shape(rank, 0);
  for (int64_t d = 0; d < rank; ++d) {
    slice_shape[d] = limits[d] - starts[d];
  }
  auto slice_ty =
      mlir::RankedTensorType::get(slice_shape, src_ty.getElementType());
  auto slice_op = mlir::stablehlo::SliceOp::create(
      builder, loc, slice_ty, src, builder.getDenseI64ArrayAttr(starts),
      builder.getDenseI64ArrayAttr(limits),
      builder.getDenseI64ArrayAttr(strides));
  return slice_op.getResult();
}

// =============================================================================
// Mechanism: Fold SliceOp of DynamicUpdateSliceOp
// =============================================================================

// Extracts constant clamped start and end bounds for DynamicUpdateSliceOp.
bool ExtractConstantDusBounds(mlir::stablehlo::DynamicUpdateSliceOp dus_op,
                              llvm::SmallVectorImpl<int64_t>& starts,
                              llvm::SmallVectorImpl<int64_t>& ends) {
  auto base_type =
      mlir::dyn_cast<mlir::RankedTensorType>(dus_op.getOperand().getType());
  auto update_type =
      mlir::dyn_cast<mlir::RankedTensorType>(dus_op.getUpdate().getType());
  if (!base_type || !update_type || !base_type.hasStaticShape() ||
      !update_type.hasStaticShape()) {
    return false;
  }
  const int64_t rank = base_type.getRank();
  auto start_indices = dus_op.getStartIndices();
  if (static_cast<int64_t>(start_indices.size()) != rank) {
    return false;
  }
  starts.resize(rank);
  ends.resize(rank);
  for (int64_t d = 0; d < rank; ++d) {
    auto s = GetConstantScalarInt(start_indices[d]);
    if (!s.has_value()) {
      return false;
    }
    const int64_t base_dim = base_type.getDimSize(d);
    const int64_t upd_dim = update_type.getDimSize(d);
    const int64_t clamped_s =
        std::clamp<int64_t>(*s, 0, std::max<int64_t>(0, base_dim - upd_dim));
    starts[d] = clamped_s;
    ends[d] = clamped_s + upd_dim;
  }
  return true;
}

// Returns true if slice bounds are fully contained within update bounds,
// and sets exact_match to true if they match identically with unit strides.
bool IsSliceContainedInDus(llvm::ArrayRef<int64_t> s_starts,
                           llvm::ArrayRef<int64_t> s_limits,
                           llvm::ArrayRef<int64_t> s_strides,
                           llvm::ArrayRef<int64_t> u_starts,
                           llvm::ArrayRef<int64_t> u_ends, bool& exact_match) {
  const int64_t rank = s_starts.size();
  exact_match = true;
  for (int64_t d = 0; d < rank; ++d) {
    if (s_starts[d] < u_starts[d] || s_limits[d] > u_ends[d]) {
      return false;
    }
    if (s_starts[d] != u_starts[d] || s_limits[d] != u_ends[d] ||
        s_strides[d] != 1) {
      exact_match = false;
    }
  }
  return true;
}

// Attempts to fold a slice of a DynamicUpdateSliceOp into a slice of update.
bool TryFoldSliceOfDynamicUpdateSlice(mlir::stablehlo::SliceOp slice_op) {
  auto res_type = mlir::dyn_cast<mlir::RankedTensorType>(slice_op.getType());
  if (!res_type || !res_type.hasStaticShape()) {
    return false;
  }
  llvm::ArrayRef<int64_t> s_starts = slice_op.getStartIndices();
  llvm::ArrayRef<int64_t> s_limits = slice_op.getLimitIndices();
  llvm::ArrayRef<int64_t> s_strides = slice_op.getStrides();
  const int64_t rank = s_starts.size();

  auto dus = slice_op.getOperand()
                 .getDefiningOp<mlir::stablehlo::DynamicUpdateSliceOp>();
  if (!dus) {
    return false;
  }

  llvm::SmallVector<int64_t, 6> u_starts;
  llvm::SmallVector<int64_t, 6> u_ends;
  if (!ExtractConstantDusBounds(dus, u_starts, u_ends)) {
    return false;
  }

  bool exact_match = false;
  if (!IsSliceContainedInDus(s_starts, s_limits, s_strides, u_starts, u_ends,
                             exact_match)) {
    return false;
  }

  if (exact_match && dus.getUpdate().getType() == slice_op.getType()) {
    slice_op.replaceAllUsesWith(dus.getUpdate());
    slice_op.erase();
    if (dus->use_empty()) {
      dus.erase();
    }
    return true;
  }

  llvm::SmallVector<int64_t, 6> new_starts(rank);
  llvm::SmallVector<int64_t, 6> new_limits(rank);
  for (int64_t d = 0; d < rank; ++d) {
    new_starts[d] = s_starts[d] - u_starts[d];
    new_limits[d] = s_limits[d] - u_starts[d];
  }
  mlir::OpBuilder builder(slice_op);
  auto new_slice = mlir::stablehlo::SliceOp::create(
      builder, slice_op.getLoc(), slice_op.getType(), dus.getUpdate(),
      builder.getDenseI64ArrayAttr(new_starts),
      builder.getDenseI64ArrayAttr(new_limits), slice_op.getStridesAttr());
  slice_op.replaceAllUsesWith(new_slice.getResult());
  slice_op.erase();
  if (dus->use_empty()) {
    dus.erase();
  }
  return true;
}

// =============================================================================
// Mechanism: Canonicalize Slice Boundaries (Merge contiguous concat slices)
// =============================================================================

// Returns true if slice sj is contiguous to s0 along dimension cdim.
bool CanMergeConsecutiveConcatSlices(mlir::stablehlo::SliceOp s0,
                                     mlir::stablehlo::SliceOp sj, uint64_t cdim,
                                     int64_t expected_start, int64_t rank) {
  if (!sj || !IsUnitStrideSlice(sj) || sj.getOperand() != s0.getOperand()) {
    return false;
  }
  llvm::ArrayRef<int64_t> s0_starts = s0.getStartIndices();
  llvm::ArrayRef<int64_t> s0_limits = s0.getLimitIndices();
  llvm::ArrayRef<int64_t> sj_starts = sj.getStartIndices();
  llvm::ArrayRef<int64_t> sj_limits = sj.getLimitIndices();
  if (static_cast<int64_t>(sj_starts.size()) != rank ||
      sj_starts[cdim] != expected_start) {
    return false;
  }
  for (int64_t d = 0; d < rank; ++d) {
    if (d != static_cast<int64_t>(cdim) &&
        (sj_starts[d] != s0_starts[d] || sj_limits[d] != s0_limits[d])) {
      return false;
    }
  }
  return true;
}

// Merges contiguous slice inputs in a concatenate operation.
bool TryMergeContiguousConcatInputs(mlir::stablehlo::ConcatenateOp concat) {
  const uint64_t cdim = concat.getDimension();
  llvm::SmallVector<mlir::Value> new_inputs(concat.getInputs().begin(),
                                            concat.getInputs().end());
  bool concat_changed = false;
  std::size_t idx = 0;
  while (idx + 1 < new_inputs.size()) {
    auto s0 = new_inputs[idx].getDefiningOp<mlir::stablehlo::SliceOp>();
    if (!s0 || !IsUnitStrideSlice(s0)) {
      ++idx;
      continue;
    }
    auto s0_ty = GetStaticRankedType(s0.getResult());
    auto s0_opnd_ty = GetStaticRankedType(s0.getOperand());
    if (!s0_ty || !s0_opnd_ty ||
        static_cast<int64_t>(cdim) >= s0_ty.getRank()) {
      ++idx;
      continue;
    }
    const int64_t rank = s0_ty.getRank();
    int64_t cur_end = s0.getLimitIndices()[cdim];
    std::size_t end_idx = idx + 1;
    while (end_idx < new_inputs.size()) {
      auto sj = new_inputs[end_idx].getDefiningOp<mlir::stablehlo::SliceOp>();
      if (!CanMergeConsecutiveConcatSlices(s0, sj, cdim, cur_end, rank)) {
        break;
      }
      cur_end = sj.getLimitIndices()[cdim];
      ++end_idx;
    }

    if (end_idx > idx + 1) {
      llvm::SmallVector<int64_t, 6> merged_starts(s0.getStartIndices().begin(),
                                                  s0.getStartIndices().end());
      llvm::SmallVector<int64_t, 6> merged_limits(s0.getLimitIndices().begin(),
                                                  s0.getLimitIndices().end());
      merged_limits[cdim] = cur_end;

      mlir::OpBuilder builder(concat);
      mlir::Value merged = GetOrCreateUnitSlice(
          builder, s0.getLoc(), s0.getOperand(), merged_starts, merged_limits);
      new_inputs.erase(new_inputs.begin() + idx, new_inputs.begin() + end_idx);
      new_inputs.insert(new_inputs.begin() + idx, merged);
      concat_changed = true;
    }
    ++idx;
  }

  if (!concat_changed) {
    return false;
  }

  if (new_inputs.size() == 1) {
    concat.replaceAllUsesWith(new_inputs[0]);
    concat.erase();
  } else {
    mlir::OpBuilder builder(concat);
    auto new_concat = mlir::stablehlo::ConcatenateOp::create(
        builder, concat.getLoc(), concat.getType(), new_inputs, cdim);
    new_concat->setDiscardableAttrs(concat->getDiscardableAttrDictionary());
    concat.replaceAllUsesWith(new_concat.getResult());
    concat.erase();
  }
  return true;
}

// Folds full-extent or rank-0 unit-stride slices directly to their input.
bool TryFoldIdentitySlice(mlir::stablehlo::SliceOp slice_op) {
  if (!IsUnitStrideSlice(slice_op)) {
    return false;
  }
  mlir::Value src = slice_op.getOperand();
  auto src_ty = GetStaticRankedType(src);
  auto res_ty = GetStaticRankedType(slice_op.getResult());
  if (!src_ty || !res_ty || src_ty.getRank() != res_ty.getRank()) {
    return false;
  }
  const int64_t rank = src_ty.getRank();
  llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
  llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();
  for (int64_t d = 0; d < rank; ++d) {
    if (starts[d] != 0 || limits[d] != src_ty.getDimSize(d)) {
      return false;
    }
  }
  slice_op.replaceAllUsesWith(src);
  slice_op.erase();
  return true;
}

// Canonicalizes concatenate and identity slice boundaries.
bool CanonicalizeSliceBoundaries(mlir::func::FuncOp func) {
  bool changed = false;
  for (mlir::Block& block : func.getBody()) {
    for (mlir::Operation& op : llvm::make_early_inc_range(block)) {
      if (auto slice = llvm::dyn_cast<mlir::stablehlo::SliceOp>(&op)) {
        changed |= TryFoldIdentitySlice(slice);
      } else if (auto concat =
                     llvm::dyn_cast<mlir::stablehlo::ConcatenateOp>(&op)) {
        changed |= TryMergeContiguousConcatInputs(concat);
      }
    }
  }
  return changed;
}

// =============================================================================
// Mechanism: Factor Multi-Slice Shared Intervals
// =============================================================================

// Finds shared sub-slice interval across a collection of slices.
bool FindSharedSliceIntervals(llvm::ArrayRef<mlir::stablehlo::SliceOp> slices,
                              mlir::RankedTensorType val_ty,
                              llvm::SmallVectorImpl<int64_t>& shared_starts,
                              llvm::SmallVectorImpl<int64_t>& shared_limits) {
  const int64_t rank = val_ty.getRank();
  shared_starts.assign(rank, 0);
  shared_limits.assign(val_ty.getShape().begin(), val_ty.getShape().end());
  bool has_shared_subslice = false;

  mlir::stablehlo::SliceOp s0 = slices[0];
  llvm::ArrayRef<int64_t> s0_starts = s0.getStartIndices();
  llvm::ArrayRef<int64_t> s0_limits = s0.getLimitIndices();
  for (int64_t d = 0; d < rank; ++d) {
    const int64_t st = s0_starts[d];
    const int64_t lim = s0_limits[d];
    if (lim - st >= val_ty.getDimSize(d)) {
      continue;
    }
    bool same_across_all = true;
    for (std::size_t k = 1; k < slices.size(); ++k) {
      mlir::stablehlo::SliceOp sk = slices[k];
      if (sk.getStartIndices()[d] != st || sk.getLimitIndices()[d] != lim) {
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
  return has_shared_subslice &&
         ComputeSliceNumElements(shared_starts, shared_limits) > 0;
}

// Factors shared sub-slice intervals out of multiple SliceOp consumers on the
// same tensor so redundant intermediate tensor slices are computed once.
bool TryFactorMultiSlicesOnValue(
    mlir::Value val, mlir::Block& block,
    llvm::SmallDenseSet<mlir::Operation*, 32>& erased_ops) {
  auto val_ty = GetStaticRankedType(val);
  if (!val_ty) {
    return false;
  }

  llvm::SmallVector<mlir::stablehlo::SliceOp, 8> slices;
  for (mlir::Operation* user : val.getUsers()) {
    auto s = llvm::dyn_cast<mlir::stablehlo::SliceOp>(user);
    if (!s || s->getBlock() != &block || !IsUnitStrideSlice(s)) {
      return false;
    }
    slices.push_back(s);
  }
  if (slices.size() < 2) {
    return false;
  }

  llvm::SmallVector<int64_t, 6> shared_starts;
  llvm::SmallVector<int64_t, 6> shared_limits;
  if (!FindSharedSliceIntervals(slices, val_ty, shared_starts, shared_limits)) {
    return false;
  }

  mlir::OpBuilder builder(&block, block.begin());
  if (auto def_op = val.getDefiningOp()) {
    builder.setInsertionPointAfter(def_op);
  } else {
    builder.setInsertionPointToStart(&block);
  }
  mlir::Value shared_slice_val = GetOrCreateUnitSlice(
      builder, slices[0].getLoc(), val, shared_starts, shared_limits);
  const int64_t rank = val_ty.getRank();
  for (mlir::stablehlo::SliceOp s : slices) {
    llvm::SmallVector<int64_t, 6> rel_starts(rank);
    llvm::SmallVector<int64_t, 6> rel_limits(rank);
    for (int64_t d = 0; d < rank; ++d) {
      rel_starts[d] = s.getStartIndices()[d] - shared_starts[d];
      rel_limits[d] = s.getLimitIndices()[d] - shared_starts[d];
    }
    mlir::OpBuilder s_builder(s);
    mlir::Value repl = GetOrCreateUnitSlice(
        s_builder, s.getLoc(), shared_slice_val, rel_starts, rel_limits,
        /*fold_parent_slice=*/false);
    s.replaceAllUsesWith(repl);
    erased_ops.insert(s.getOperation());
    s.erase();
  }
  return true;
}

// Factors multi-slice shared intervals within a single block.
bool FactorMultiSliceSharedIntervalsInBlock(mlir::Block& block) {
  bool changed = false;
  llvm::SmallDenseSet<mlir::Operation*, 32> erased_ops;
  for (mlir::BlockArgument arg : block.getArguments()) {
    if (TryFactorMultiSlicesOnValue(arg, block, erased_ops)) {
      changed = true;
    }
  }
  llvm::SmallVector<mlir::Operation*, 32> ops;
  for (mlir::Operation& op : block) {
    if (!llvm::isa<mlir::stablehlo::SliceOp>(&op)) {
      ops.push_back(&op);
    }
  }
  for (mlir::Operation* op_ptr : ops) {
    if (erased_ops.contains(op_ptr)) {
      continue;
    }
    for (mlir::Value res : op_ptr->getResults()) {
      if (TryFactorMultiSlicesOnValue(res, block, erased_ops)) {
        changed = true;
      }
    }
  }
  return changed;
}

// Factors multi-slice shared intervals across all blocks in a function.
bool FactorMultiSliceSharedIntervals(mlir::func::FuncOp func) {
  bool changed = false;
  for (mlir::Block& block : func.getBody()) {
    changed |= FactorMultiSliceSharedIntervalsInBlock(block);
  }
  return changed;
}

// =============================================================================
// Mechanism: Push Slice Across Cross-Axis Reshape
// =============================================================================

// Returns true if intermediate dimensions between a_outer and a_inner are 1.
bool CheckUnitDimsBetween(llvm::ArrayRef<int64_t> a_shape, int64_t a_outer,
                          int64_t a_inner) {
  for (int64_t mid = a_outer + 1; mid < a_inner; ++mid) {
    if (a_shape[mid] != 1) {
      return false;
    }
  }
  return true;
}

// Matches Case A: [lo, hi) falls inside a single outer chunk m0 = lo / K.
bool TryMatchCrossAxisCaseA(llvm::ArrayRef<int64_t> a_shape, int64_t a_outer,
                            int64_t a_inner, int64_t m_ext, int64_t k_ext,
                            int64_t block_stride, int64_t lo, int64_t hi,
                            llvm::SmallVectorImpl<int64_t>& x_starts,
                            llvm::SmallVectorImpl<int64_t>& x_limits) {
  if (lo / k_ext != (hi - 1) / k_ext) {
    return false;
  }
  const int64_t m0 = lo / k_ext;
  const int64_t k_lo = lo % k_ext;
  const int64_t k_hi = (hi - 1) % k_ext + 1;
  if (m_ext == 1 && (k_hi - k_lo) >= k_ext) {
    return false;
  }
  if (a_outer != a_inner) {
    x_starts[a_outer] = m0;
    x_limits[a_outer] = m0 + 1;
  }
  x_starts[a_inner] = k_lo * block_stride;
  x_limits[a_inner] = k_hi * block_stride;
  return true;
}

// Matches Case B: [lo, hi) is aligned to multiples of K across distinct axes.
bool TryMatchCrossAxisCaseB(int64_t a_outer, int64_t a_inner, int64_t k_ext,
                            int64_t lo, int64_t hi,
                            llvm::SmallVectorImpl<int64_t>& x_starts,
                            llvm::SmallVectorImpl<int64_t>& x_limits) {
  if (lo % k_ext != 0 || hi % k_ext != 0 || a_outer == a_inner) {
    return false;
  }
  x_starts[a_outer] = lo / k_ext;
  x_limits[a_outer] = hi / k_ext;
  return true;
}

// Matches slices of reshaped tensors that split/merge dimensions so the slice
// can be hoisted across the reshape boundary into the unreshaped tensor.
bool MatchCrossAxisReshapeSlice(mlir::RankedTensorType x_ty,
                                mlir::RankedTensorType r_ty, int64_t b_idx,
                                int64_t lo, int64_t hi,
                                llvm::SmallVectorImpl<int64_t>& x_starts,
                                llvm::SmallVectorImpl<int64_t>& x_limits) {
  llvm::ArrayRef<int64_t> a_shape = x_ty.getShape();
  llvm::ArrayRef<int64_t> b_shape = r_ty.getShape();
  const int64_t a_rank = x_ty.getRank();
  const int64_t b_rank = r_ty.getRank();
  const int64_t b_len = b_shape[b_idx];

  x_starts.assign(a_rank, 0);
  x_limits.assign(a_shape.begin(), a_shape.end());

  int64_t out_prefix = 1;
  for (int64_t j = 0; j < b_idx; ++j) {
    out_prefix *= b_shape[j];
  }
  int64_t out_suffix = 1;
  for (int64_t j = b_idx + 1; j < b_rank; ++j) {
    out_suffix *= b_shape[j];
  }

  for (int64_t a_inner = 0; a_inner < a_rank; ++a_inner) {
    int64_t in_suffix = 1;
    for (int64_t i = a_inner + 1; i < a_rank; ++i) {
      in_suffix *= a_shape[i];
    }
    if (out_suffix < in_suffix || out_suffix % in_suffix != 0) {
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

    for (int64_t a_outer = 0; a_outer <= a_inner; ++a_outer) {
      if (a_outer == a_inner) {
        if (m_ext != 1) {
          continue;
        }
      } else {
        if (a_shape[a_outer] != m_ext ||
            !CheckUnitDimsBetween(a_shape, a_outer, a_inner)) {
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

      if (m_ext == 1 && block_stride == 1) {
        continue;
      }
      // Require k_ext > 1 when a_outer != a_inner to avoid intercepting
      // preserved axes that are handled by Step 6 HoistSlicesInFunc.
      if (a_outer != a_inner && k_ext <= 1) {
        continue;
      }

      llvm::SmallVector<int64_t, 6> cand_starts(x_starts.begin(),
                                                x_starts.end());
      llvm::SmallVector<int64_t, 6> cand_limits(x_limits.begin(),
                                                x_limits.end());
      if (TryMatchCrossAxisCaseA(a_shape, a_outer, a_inner, m_ext, k_ext,
                                 block_stride, lo, hi, cand_starts,
                                 cand_limits) ||
          TryMatchCrossAxisCaseB(a_outer, a_inner, k_ext, lo, hi, cand_starts,
                                 cand_limits)) {
        x_starts = std::move(cand_starts);
        x_limits = std::move(cand_limits);
        return true;
      }
    }
  }
  return false;
}

// Finds the first sliced dimension in slice_op that matches cross-axis reshape.
bool FindCrossAxisSliceCandidate(mlir::RankedTensorType x_ty,
                                 mlir::RankedTensorType r_ty,
                                 llvm::ArrayRef<int64_t> b_shape,
                                 llvm::ArrayRef<int64_t> starts,
                                 llvm::ArrayRef<int64_t> limits, int64_t& b_idx,
                                 llvm::SmallVectorImpl<int64_t>& x_starts,
                                 llvm::SmallVectorImpl<int64_t>& x_limits) {
  const int64_t q_rank = r_ty.getRank();
  for (int64_t j = 0; j < q_rank; ++j) {
    if (limits[j] - starts[j] < b_shape[j]) {
      const int64_t lo = starts[j];
      const int64_t hi = limits[j];
      if (MatchCrossAxisReshapeSlice(x_ty, r_ty, j, lo, hi, x_starts,
                                     x_limits)) {
        b_idx = j;
        return true;
      }
    }
  }
  return false;
}

// Attempts to factor a slice across a reshape operation.
bool TryFactorCrossAxisSlice(mlir::stablehlo::SliceOp slice_op,
                             mlir::Block& block) {
  if (!slice_op || slice_op->getBlock() != &block ||
      !IsUnitStrideSlice(slice_op)) {
    return false;
  }
  auto reshape_op =
      slice_op.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!reshape_op) {
    return false;
  }

  for (mlir::Operation* u : reshape_op->getUsers()) {
    auto us = llvm::dyn_cast<mlir::stablehlo::SliceOp>(u);
    if (!us || !IsUnitStrideSlice(us)) {
      return false;
    }
  }

  mlir::Value x_val = reshape_op.getOperand();
  auto x_ty = GetStaticRankedType(x_val);
  auto r_ty = GetStaticRankedType(reshape_op.getResult());
  auto s_ty = GetStaticRankedType(slice_op.getResult());
  if (!x_ty || !r_ty || !s_ty || s_ty.getNumElements() <= 0) {
    return false;
  }

  llvm::ArrayRef<int64_t> b_shape = r_ty.getShape();
  llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
  llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();

  int64_t b_idx = -1;
  llvm::SmallVector<int64_t, 6> x_starts;
  llvm::SmallVector<int64_t, 6> x_limits;
  if (!FindCrossAxisSliceCandidate(x_ty, r_ty, b_shape, starts, limits, b_idx,
                                   x_starts, x_limits)) {
    return false;
  }

  const int64_t lo = starts[b_idx];
  const int64_t hi = limits[b_idx];
  mlir::OpBuilder builder(slice_op);
  mlir::Value x_sl = GetOrCreateUnitSlice(builder, slice_op.getLoc(), x_val,
                                          x_starts, x_limits);
  llvm::SmallVector<int64_t, 6> r_sl_shape(b_shape.begin(), b_shape.end());
  r_sl_shape[b_idx] = hi - lo;
  auto r_sl_ty = mlir::RankedTensorType::get(r_sl_shape, r_ty.getElementType());
  mlir::Value r_sl = mlir::stablehlo::ReshapeOp::create(
      builder, slice_op.getLoc(), r_sl_ty, x_sl);
  r_sl.getDefiningOp()->setDiscardableAttrs(
      reshape_op->getDiscardableAttrDictionary());

  llvm::SmallVector<int64_t, 6> rem_starts(starts.begin(), starts.end());
  llvm::SmallVector<int64_t, 6> rem_limits(limits.begin(), limits.end());
  rem_starts[b_idx] = 0;
  rem_limits[b_idx] = hi - lo;
  mlir::Value final_val = GetOrCreateUnitSlice(builder, slice_op.getLoc(), r_sl,
                                               rem_starts, rem_limits);

  slice_op.replaceAllUsesWith(final_val);
  slice_op.erase();
  if (reshape_op->use_empty()) {
    reshape_op.erase();
  }
  return true;
}

// Pushes slices across cross-axis reshape operations in a function.
bool FactorCrossAxisReshapeSlices(mlir::func::FuncOp func) {
  bool changed = false;
  for (mlir::Block& block : func.getBody()) {
    llvm::SmallVector<mlir::stablehlo::SliceOp, 16> candidate_slices;
    for (mlir::Operation& op : block) {
      if (auto s = llvm::dyn_cast<mlir::stablehlo::SliceOp>(&op)) {
        if (s.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
          candidate_slices.push_back(s);
        }
      }
    }
    for (mlir::stablehlo::SliceOp s : candidate_slices) {
      if (s && s->getBlock()) {
        changed |= TryFactorCrossAxisSlice(s, block);
      }
    }
  }
  return changed;
}

// =============================================================================
// Mechanism: Hoist Slices Across Producers
// =============================================================================

// Returns true if op is a supported elementwise operation for slice hoisting.
bool IsSupportedElementwiseOp(mlir::Operation* op) {
  if (llvm::isa<mlir::stablehlo::AbsOp, mlir::stablehlo::AddOp,
                mlir::stablehlo::AndOp, mlir::stablehlo::Atan2Op,
                mlir::stablehlo::CbrtOp, mlir::stablehlo::CeilOp,
                mlir::stablehlo::ClampOp, mlir::stablehlo::ClzOp,
                mlir::stablehlo::CompareOp, mlir::stablehlo::ConvertOp,
                mlir::stablehlo::CosineOp, mlir::stablehlo::DivOp,
                mlir::stablehlo::ExpOp, mlir::stablehlo::Expm1Op,
                mlir::stablehlo::FloorOp, mlir::stablehlo::IsFiniteOp,
                mlir::stablehlo::LogOp, mlir::stablehlo::Log1pOp,
                mlir::stablehlo::LogisticOp, mlir::stablehlo::MaxOp,
                mlir::stablehlo::MinOp, mlir::stablehlo::MulOp,
                mlir::stablehlo::NegOp, mlir::stablehlo::NotOp,
                mlir::stablehlo::OrOp, mlir::stablehlo::PopulationCountOp,
                mlir::stablehlo::PowOp, mlir::stablehlo::RemOp,
                mlir::stablehlo::RoundOp, mlir::stablehlo::RoundNearestEvenOp,
                mlir::stablehlo::RsqrtOp, mlir::stablehlo::SelectOp,
                mlir::stablehlo::ShiftLeftOp,
                mlir::stablehlo::ShiftRightArithmeticOp,
                mlir::stablehlo::ShiftRightLogicalOp, mlir::stablehlo::SignOp,
                mlir::stablehlo::SineOp, mlir::stablehlo::SqrtOp,
                mlir::stablehlo::SubtractOp, mlir::stablehlo::TanhOp,
                mlir::stablehlo::XorOp, mlir::chlo::BroadcastAddOp,
                mlir::chlo::BroadcastSubOp, mlir::chlo::BroadcastMulOp,
                mlir::chlo::BroadcastDivOp, mlir::chlo::BroadcastMaxOp,
                mlir::chlo::BroadcastMinOp, mlir::chlo::BroadcastCompareOp>(
          op)) {
    return true;
  }
  return false;
}

// Finds matching source axis for an unreshaped preserved dimension by matching
// prefix and suffix element counts, identifying axes that can be sliced.
int64_t FindMatchingSrcAxisForPreservedDim(
    int64_t d, llvm::ArrayRef<int64_t> src_shape,
    llvm::ArrayRef<int64_t> dst_shape,
    const llvm::DenseSet<int64_t>& used_src_axes) {
  const int64_t target_dim = dst_shape[d];
  int64_t target_prefix = 1;
  for (int64_t i = 0; i < d; ++i) {
    target_prefix *= dst_shape[i];
  }
  int64_t target_suffix = 1;
  for (int64_t i = d + 1; i < static_cast<int64_t>(dst_shape.size()); ++i) {
    target_suffix *= dst_shape[i];
  }

  for (int64_t s = 0; s < static_cast<int64_t>(src_shape.size()); ++s) {
    if (used_src_axes.contains(s) || src_shape[s] != target_dim) {
      continue;
    }
    int64_t s_prefix = 1;
    for (int64_t i = 0; i < s; ++i) {
      s_prefix *= src_shape[i];
    }
    int64_t s_suffix = 1;
    for (int64_t i = s + 1; i < static_cast<int64_t>(src_shape.size()); ++i) {
      s_suffix *= src_shape[i];
    }
    if (s_prefix == target_prefix && s_suffix == target_suffix) {
      return s;
    }
  }
  return -1;
}

// Maps preserved sliced dimensions through a reshape into input slice bounds.
bool MapPreservedSlicedDimsToReshapeInput(
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
    const int64_t matched_src = FindMatchingSrcAxisForPreservedDim(
        d, src_shape, dst_shape, used_src_axes);
    if (matched_src < 0) {
      return false;
    }
    used_src_axes.insert(matched_src);
    src_starts[matched_src] = dst_starts[d];
    src_limits[matched_src] = dst_limits[d];
  }
  return true;
}

// Replaces all user slices with new_val and erases producer prod.
void ReplaceUserSlicesAndEraseProducer(
    const llvm::SmallVector<mlir::stablehlo::SliceOp, 4>& user_slices,
    mlir::Value new_val, mlir::Operation* prod) {
  for (mlir::stablehlo::SliceOp u_slice : user_slices) {
    u_slice.replaceAllUsesWith(new_val);
    u_slice.erase();
  }
  prod->erase();
}

// Returns true if op is a CHLO broadcast operation.
bool IsChloBroadcastOp(mlir::Operation* op) {
  return llvm::isa<mlir::chlo::BroadcastAddOp, mlir::chlo::BroadcastSubOp,
                   mlir::chlo::BroadcastMulOp, mlir::chlo::BroadcastDivOp,
                   mlir::chlo::BroadcastMaxOp, mlir::chlo::BroadcastMinOp,
                   mlir::chlo::BroadcastCompareOp>(op);
}

// Hoists slice across an elementwise operation.
bool TryHoistAcrossElementwise(
    mlir::Operation* prod, mlir::OpBuilder& builder, mlir::Location loc,
    mlir::RankedTensorType sliced_res_ty,
    const llvm::SmallVector<mlir::stablehlo::SliceOp, 4>& user_slices,
    llvm::ArrayRef<int64_t> starts, llvm::ArrayRef<int64_t> limits) {
  if (!IsSupportedElementwiseOp(prod)) {
    return false;
  }
  auto out_ty = GetStaticRankedType(prod->getResult(0));
  if (!out_ty) {
    return false;
  }
  const int64_t out_rank = out_ty.getRank();
  llvm::ArrayRef<int64_t> out_shape = out_ty.getShape();

  struct OperandSliceInfo {
    bool is_scalar = false;
    llvm::SmallVector<int64_t, 6> opnd_starts;
    llvm::SmallVector<int64_t, 6> opnd_limits;
  };
  llvm::SmallVector<OperandSliceInfo, 4> opnd_infos;
  opnd_infos.reserve(prod->getNumOperands());

  const bool is_bcast_op = IsChloBroadcastOp(prod);

  for (mlir::Value opnd : prod->getOperands()) {
    auto opnd_ty = GetStaticRankedType(opnd);
    if (!opnd_ty) {
      return false;
    }
    const int64_t opnd_rank = opnd_ty.getRank();
    if (opnd_rank == 0) {
      opnd_infos.push_back({/*is_scalar=*/true, {}, {}});
      continue;
    }
    if (opnd_ty.getShape() == out_shape) {
      llvm::SmallVector<int64_t, 6> op_starts(starts.begin(), starts.end());
      llvm::SmallVector<int64_t, 6> op_limits(limits.begin(), limits.end());
      opnd_infos.push_back(
          {/*is_scalar=*/false, std::move(op_starts), std::move(op_limits)});
      continue;
    }
    if (!is_bcast_op || opnd_rank > out_rank) {
      return false;
    }

    llvm::SmallVector<int64_t, 6> mapped_dims;
    if (auto bcast_dims_attr = prod->getAttrOfType<mlir::DenseI64ArrayAttr>(
            "broadcast_dimensions")) {
      llvm::ArrayRef<int64_t> bcast_dims = bcast_dims_attr.asArrayRef();
      if (static_cast<int64_t>(bcast_dims.size()) != opnd_rank) {
        return false;
      }
      mapped_dims.assign(bcast_dims.begin(), bcast_dims.end());
    } else {
      for (int64_t j = 0; j < opnd_rank; ++j) {
        mapped_dims.push_back((out_rank - opnd_rank) + j);
      }
    }

    llvm::SmallVector<int64_t, 6> op_starts;
    llvm::SmallVector<int64_t, 6> op_limits;
    op_starts.reserve(opnd_rank);
    op_limits.reserve(opnd_rank);
    for (int64_t j = 0; j < opnd_rank; ++j) {
      const int64_t out_d = mapped_dims[j];
      if (out_d < 0 || out_d >= out_rank) {
        return false;
      }
      const int64_t opnd_dim = opnd_ty.getDimSize(j);
      const int64_t out_dim = out_shape[out_d];
      if (opnd_dim == 1) {
        op_starts.push_back(0);
        op_limits.push_back(limits[out_d] == starts[out_d] ? 0 : 1);
      } else if (opnd_dim == out_dim) {
        if (starts[out_d] < 0 || limits[out_d] > out_dim ||
            starts[out_d] > limits[out_d]) {
          return false;
        }
        op_starts.push_back(starts[out_d]);
        op_limits.push_back(limits[out_d]);
      } else {
        return false;
      }
    }
    opnd_infos.push_back(
        {/*is_scalar=*/false, std::move(op_starts), std::move(op_limits)});
  }

  mlir::IRMapping mapping;
  for (std::size_t i = 0; i < prod->getNumOperands(); ++i) {
    mlir::Value opnd = prod->getOperand(i);
    if (mapping.contains(opnd)) {
      continue;
    }
    const auto& info = opnd_infos[i];
    if (info.is_scalar) {
      mapping.map(opnd, opnd);
    } else {
      mlir::Value sliced_opnd = GetOrCreateUnitSlice(
          builder, loc, opnd, info.opnd_starts, info.opnd_limits);
      if (!sliced_opnd) {
        return false;
      }
      mapping.map(opnd, sliced_opnd);
    }
  }
  mlir::Operation* cloned = builder.clone(*prod, mapping);
  cloned->getResult(0).setType(sliced_res_ty);
  ReplaceUserSlicesAndEraseProducer(user_slices, cloned->getResult(0), prod);
  return true;
}

// Hoists slice across BroadcastInDimOp.
bool TryHoistAcrossBroadcastInDim(
    mlir::Operation* prod, mlir::OpBuilder& builder, mlir::Location loc,
    mlir::RankedTensorType sliced_res_ty,
    const llvm::SmallVector<mlir::stablehlo::SliceOp, 4>& user_slices,
    llvm::ArrayRef<int64_t> starts, llvm::ArrayRef<int64_t> limits) {
  auto bcast = llvm::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(prod);
  if (!bcast) {
    return false;
  }
  mlir::Value b_in = bcast.getOperand();
  auto b_in_ty = GetStaticRankedType(b_in);
  if (!b_in_ty) {
    return false;
  }
  if (b_in_ty.getRank() == 0) {
    auto new_bcast = mlir::stablehlo::BroadcastInDimOp::create(
        builder, loc, sliced_res_ty, b_in, bcast.getBroadcastDimensionsAttr());
    new_bcast->setDiscardableAttrs(bcast->getDiscardableAttrDictionary());
    ReplaceUserSlicesAndEraseProducer(user_slices, new_bcast.getResult(),
                                      bcast);
    return true;
  }

  const int64_t b_rank = b_in_ty.getRank();
  llvm::ArrayRef<int64_t> b_dims = bcast.getBroadcastDimensions();
  llvm::SmallVector<int64_t, 6> b_starts(b_rank, 0);
  llvm::SmallVector<int64_t, 6> b_limits(b_in_ty.getShape().begin(),
                                         b_in_ty.getShape().end());
  for (int64_t k = 0; k < b_rank; ++k) {
    const int64_t out_d = b_dims[k];
    if (b_in_ty.getDimSize(k) > 1) {
      b_starts[k] = starts[out_d];
      b_limits[k] = limits[out_d];
    }
  }
  mlir::Value b_in_sl =
      GetOrCreateUnitSlice(builder, loc, b_in, b_starts, b_limits);
  auto new_bcast = mlir::stablehlo::BroadcastInDimOp::create(
      builder, loc, sliced_res_ty, b_in_sl, bcast.getBroadcastDimensionsAttr());
  new_bcast->setDiscardableAttrs(bcast->getDiscardableAttrDictionary());
  ReplaceUserSlicesAndEraseProducer(user_slices, new_bcast.getResult(), bcast);
  return true;
}

// Hoists slice across ReverseOp.
bool TryHoistAcrossReverse(
    mlir::Operation* prod, mlir::OpBuilder& builder, mlir::Location loc,
    mlir::RankedTensorType sliced_res_ty,
    const llvm::SmallVector<mlir::stablehlo::SliceOp, 4>& user_slices,
    llvm::ArrayRef<int64_t> starts, llvm::ArrayRef<int64_t> limits,
    int64_t rank, mlir::RankedTensorType out_ty) {
  auto rev = llvm::dyn_cast<mlir::stablehlo::ReverseOp>(prod);
  if (!rev) {
    return false;
  }
  llvm::DenseSet<int64_t> rev_dims;
  for (int64_t d : rev.getDimensions()) {
    rev_dims.insert(d);
  }
  llvm::SmallVector<int64_t, 6> r_starts(starts.begin(), starts.end());
  llvm::SmallVector<int64_t, 6> r_limits(limits.begin(), limits.end());
  for (int64_t d = 0; d < rank; ++d) {
    if (rev_dims.contains(d)) {
      const int64_t dim_len = out_ty.getDimSize(d);
      r_starts[d] = dim_len - limits[d];
      r_limits[d] = dim_len - starts[d];
    }
  }
  mlir::Value in_sl =
      GetOrCreateUnitSlice(builder, loc, rev.getOperand(), r_starts, r_limits);
  auto new_rev = mlir::stablehlo::ReverseOp::create(
      builder, loc, sliced_res_ty, in_sl, rev.getDimensionsAttr());
  new_rev->setDiscardableAttrs(rev->getDiscardableAttrDictionary());
  ReplaceUserSlicesAndEraseProducer(user_slices, new_rev.getResult(), rev);
  return true;
}

// Hoists slice across TransposeOp.
bool TryHoistAcrossTranspose(
    mlir::Operation* prod, mlir::OpBuilder& builder, mlir::Location loc,
    mlir::RankedTensorType sliced_res_ty,
    const llvm::SmallVector<mlir::stablehlo::SliceOp, 4>& user_slices,
    llvm::ArrayRef<int64_t> starts, llvm::ArrayRef<int64_t> limits,
    int64_t rank) {
  auto tr = llvm::dyn_cast<mlir::stablehlo::TransposeOp>(prod);
  if (!tr) {
    return false;
  }
  llvm::ArrayRef<int64_t> perm = tr.getPermutation();
  llvm::SmallVector<int64_t, 6> in_starts(rank, 0);
  llvm::SmallVector<int64_t, 6> in_limits(rank, 0);
  for (int64_t i = 0; i < rank; ++i) {
    in_starts[perm[i]] = starts[i];
    in_limits[perm[i]] = limits[i];
  }
  mlir::Value in_sl =
      GetOrCreateUnitSlice(builder, loc, tr.getOperand(), in_starts, in_limits);
  auto new_tr = mlir::stablehlo::TransposeOp::create(
      builder, loc, sliced_res_ty, in_sl, tr.getPermutationAttr());
  new_tr->setDiscardableAttrs(tr->getDiscardableAttrDictionary());
  ReplaceUserSlicesAndEraseProducer(user_slices, new_tr.getResult(), tr);
  return true;
}

// Hoists slice across ConcatenateOp.
bool TryHoistAcrossConcatenate(
    mlir::Operation* prod, mlir::OpBuilder& builder, mlir::Location loc,
    mlir::RankedTensorType sliced_res_ty,
    const llvm::SmallVector<mlir::stablehlo::SliceOp, 4>& user_slices,
    llvm::ArrayRef<int64_t> starts, llvm::ArrayRef<int64_t> limits,
    int64_t rank) {
  auto concat = llvm::dyn_cast<mlir::stablehlo::ConcatenateOp>(prod);
  if (!concat) {
    return false;
  }
  const int64_t cdim = concat.getDimension();
  int64_t offset = 0;
  llvm::SmallVector<mlir::Value, 4> new_concat_inputs;
  for (mlir::Value in_val : concat.getInputs()) {
    auto in_ty = GetStaticRankedType(in_val);
    if (!in_ty) {
      return false;
    }
    const int64_t extent = in_ty.getDimSize(cdim);
    const int64_t in_start = offset;
    const int64_t in_limit = offset + extent;
    offset += extent;

    const int64_t inter_start = std::max(starts[cdim], in_start);
    const int64_t inter_limit = std::min(limits[cdim], in_limit);
    if (inter_start >= inter_limit) {
      continue;
    }
    llvm::SmallVector<int64_t, 6> sub_starts(starts.begin(), starts.end());
    llvm::SmallVector<int64_t, 6> sub_limits(limits.begin(), limits.end());
    sub_starts[cdim] = inter_start - in_start;
    sub_limits[cdim] = inter_limit - in_start;
    mlir::Value sub_sl =
        GetOrCreateUnitSlice(builder, loc, in_val, sub_starts, sub_limits);
    new_concat_inputs.push_back(sub_sl);
  }
  if (new_concat_inputs.empty()) {
    return false;
  }

  mlir::Value final_res = nullptr;
  if (new_concat_inputs.size() == 1) {
    final_res = new_concat_inputs[0];
  } else {
    auto new_concat = mlir::stablehlo::ConcatenateOp::create(
        builder, loc, sliced_res_ty, new_concat_inputs, cdim);
    new_concat->setDiscardableAttrs(concat->getDiscardableAttrDictionary());
    final_res = new_concat.getResult();
  }
  ReplaceUserSlicesAndEraseProducer(user_slices, final_res, concat);
  return true;
}

// Hoists slice across ReshapeOp (preserved axes with unit dims).
bool TryHoistAcrossReshape(
    mlir::Operation* prod, mlir::OpBuilder& builder, mlir::Location loc,
    mlir::RankedTensorType sliced_res_ty,
    const llvm::SmallVector<mlir::stablehlo::SliceOp, 4>& user_slices,
    llvm::ArrayRef<int64_t> starts, llvm::ArrayRef<int64_t> limits,
    mlir::RankedTensorType out_ty) {
  auto resh = llvm::dyn_cast<mlir::stablehlo::ReshapeOp>(prod);
  if (!resh) {
    return false;
  }
  auto r_in_ty = GetStaticRankedType(resh.getOperand());
  if (!r_in_ty) {
    return false;
  }
  llvm::SmallVector<int64_t, 6> r_starts;
  llvm::SmallVector<int64_t, 6> r_limits;
  if (!MapPreservedSlicedDimsToReshapeInput(r_in_ty.getShape(),
                                            out_ty.getShape(), starts, limits,
                                            r_starts, r_limits)) {
    return false;
  }
  mlir::Value r_in_sl =
      GetOrCreateUnitSlice(builder, loc, resh.getOperand(), r_starts, r_limits);
  auto new_resh =
      mlir::stablehlo::ReshapeOp::create(builder, loc, sliced_res_ty, r_in_sl);
  new_resh->setDiscardableAttrs(resh->getDiscardableAttrDictionary());
  ReplaceUserSlicesAndEraseProducer(user_slices, new_resh.getResult(), resh);
  return true;
}

// Hoists slice across DotGeneralOp non-contracting and batching dimensions.
bool TryHoistAcrossDotGeneral(
    mlir::Operation* prod, mlir::OpBuilder& builder, mlir::Location loc,
    mlir::RankedTensorType sliced_res_ty,
    const llvm::SmallVector<mlir::stablehlo::SliceOp, 4>& user_slices,
    llvm::ArrayRef<int64_t> starts, llvm::ArrayRef<int64_t> limits,
    int64_t rank) {
  auto dot_gen = llvm::dyn_cast<mlir::stablehlo::DotGeneralOp>(prod);
  if (!dot_gen) {
    return false;
  }
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
  llvm::SmallVector<int64_t, 6> lhs_limits(lhs_ty.getShape().begin(),
                                           lhs_ty.getShape().end());
  llvm::SmallVector<int64_t, 6> rhs_starts(rhs_ty.getRank(), 0);
  llvm::SmallVector<int64_t, 6> rhs_limits(rhs_ty.getShape().begin(),
                                           rhs_ty.getShape().end());

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

  mlir::Value lhs_sl =
      GetOrCreateUnitSlice(builder, loc, lhs, lhs_starts, lhs_limits);
  mlir::Value rhs_sl =
      GetOrCreateUnitSlice(builder, loc, rhs, rhs_starts, rhs_limits);
  auto new_dot = mlir::stablehlo::DotGeneralOp::create(
      builder, loc, sliced_res_ty, lhs_sl, rhs_sl,
      dot_gen.getDotDimensionNumbers(), dot_gen.getPrecisionConfigAttr(),
      dot_gen.getAlgorithmAttr());
  new_dot->setDiscardableAttrs(dot_gen->getDiscardableAttrDictionary());
  ReplaceUserSlicesAndEraseProducer(user_slices, new_dot.getResult(), dot_gen);
  return true;
}

// Dispatches hoisting of user_slices across producer operation prod.
bool TryHoistSliceAcrossProducer(
    mlir::Operation* prod,
    const llvm::SmallVector<mlir::stablehlo::SliceOp, 4>& user_slices,
    llvm::ArrayRef<int64_t> starts, llvm::ArrayRef<int64_t> limits) {
  if (prod->getNumResults() != 1) {
    return false;
  }
  mlir::Value res = prod->getResult(0);
  auto out_ty = GetStaticRankedType(res);
  if (!out_ty) {
    return false;
  }
  const int64_t rank = out_ty.getRank();
  mlir::OpBuilder builder(prod);
  mlir::Location loc = prod->getLoc();

  llvm::SmallVector<int64_t, 6> sliced_shape(rank, 0);
  for (int64_t d = 0; d < rank; ++d) {
    sliced_shape[d] = limits[d] - starts[d];
  }
  auto sliced_res_ty =
      mlir::RankedTensorType::get(sliced_shape, out_ty.getElementType());

  if (TryHoistAcrossElementwise(prod, builder, loc, sliced_res_ty, user_slices,
                                starts, limits)) {
    return true;
  }
  if (TryHoistAcrossBroadcastInDim(prod, builder, loc, sliced_res_ty,
                                   user_slices, starts, limits)) {
    return true;
  }
  if (TryHoistAcrossReverse(prod, builder, loc, sliced_res_ty, user_slices,
                            starts, limits, rank, out_ty)) {
    return true;
  }
  if (TryHoistAcrossTranspose(prod, builder, loc, sliced_res_ty, user_slices,
                              starts, limits, rank)) {
    return true;
  }
  if (TryHoistAcrossConcatenate(prod, builder, loc, sliced_res_ty, user_slices,
                                starts, limits, rank)) {
    return true;
  }
  if (TryHoistAcrossReshape(prod, builder, loc, sliced_res_ty, user_slices,
                            starts, limits, out_ty)) {
    return true;
  }
  if (TryHoistAcrossDotGeneral(prod, builder, loc, sliced_res_ty, user_slices,
                               starts, limits, rank)) {
    return true;
  }
  return false;
}

// Collects user slices of res if all users are identical unit-stride slices.
bool CollectIdenticalUserSlices(
    mlir::Value res, mlir::Block* block, int64_t rank,
    llvm::SmallVectorImpl<mlir::stablehlo::SliceOp>& user_slices,
    llvm::SmallVectorImpl<int64_t>& common_starts,
    llvm::SmallVectorImpl<int64_t>& common_limits) {
  for (mlir::Operation* user : res.getUsers()) {
    auto sl = llvm::dyn_cast<mlir::stablehlo::SliceOp>(user);
    if (!sl || sl->getBlock() != block || !IsUnitStrideSlice(sl)) {
      return false;
    }
    llvm::ArrayRef<int64_t> s_starts = sl.getStartIndices();
    llvm::ArrayRef<int64_t> s_limits = sl.getLimitIndices();
    if (static_cast<int64_t>(s_starts.size()) != rank) {
      return false;
    }
    if (user_slices.empty()) {
      common_starts.assign(s_starts.begin(), s_starts.end());
      common_limits.assign(s_limits.begin(), s_limits.end());
    } else if (common_starts != s_starts || common_limits != s_limits) {
      return false;
    }
    user_slices.push_back(sl);
  }
  return !user_slices.empty();
}

// Hoists unit-stride slices across supported producer operations in func.
bool HoistSlicesInFunc(mlir::func::FuncOp func) {
  bool changed = false;
  for (mlir::Block& block : func.getBody()) {
    for (mlir::Operation& op :
         llvm::make_early_inc_range(llvm::reverse(block))) {
      if (op.getNumResults() != 1 || op.use_empty() ||
          llvm::isa<mlir::stablehlo::SliceOp, mlir::stablehlo::ConstantOp>(
              &op)) {
        continue;
      }

      mlir::Value res = op.getResult(0);
      auto res_ty = GetStaticRankedType(res);
      if (!res_ty) {
        continue;
      }
      const int64_t rank = res_ty.getRank();

      llvm::SmallVector<mlir::stablehlo::SliceOp, 4> user_slices;
      llvm::SmallVector<int64_t, 6> common_starts;
      llvm::SmallVector<int64_t, 6> common_limits;
      if (!CollectIdenticalUserSlices(res, &block, rank, user_slices,
                                      common_starts, common_limits)) {
        continue;
      }

      if (TryHoistSliceAcrossProducer(&op, user_slices, common_starts,
                                      common_limits)) {
        changed = true;
      }
    }
  }
  return changed;
}

// =============================================================================
// Mechanism: Fold Slice of Pad
// =============================================================================

// Folds slice(pad(x)) into slice(x) with shifted bounds when the slice window
// falls strictly inside x's unpadded region across all dimensions and no
// interior padding is present. Valid for any pad value and any unpadded tensor.
bool TryFoldSliceOfPad(mlir::stablehlo::SliceOp slice_op) {
  if (!IsUnitStrideSlice(slice_op)) {
    return false;
  }
  auto pad_op = slice_op.getOperand().getDefiningOp<mlir::stablehlo::PadOp>();
  if (!pad_op) {
    return false;
  }
  mlir::Value x_val = pad_op.getOperand();
  auto x_ty = GetStaticRankedType(x_val);
  auto pad_ty = GetStaticRankedType(pad_op.getResult());
  if (!x_ty || !pad_ty) {
    return false;
  }
  const int64_t rank = x_ty.getRank();
  llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
  llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();
  llvm::ArrayRef<int64_t> edge_low = pad_op.getEdgePaddingLow();
  llvm::ArrayRef<int64_t> interior = pad_op.getInteriorPadding();

  llvm::SmallVector<int64_t, 6> new_starts(rank, 0);
  llvm::SmallVector<int64_t, 6> new_limits(rank, 0);

  for (int64_t d = 0; d < rank; ++d) {
    if (interior[d] != 0) {
      return false;
    }
    const int64_t low = edge_low[d];
    const int64_t x_dim = x_ty.getDimSize(d);
    if (starts[d] < low || limits[d] > low + x_dim) {
      return false;
    }
    new_starts[d] = starts[d] - low;
    new_limits[d] = limits[d] - low;
  }

  mlir::OpBuilder builder(slice_op);
  mlir::Value folded = GetOrCreateUnitSlice(builder, slice_op.getLoc(), x_val,
                                            new_starts, new_limits);
  slice_op.replaceAllUsesWith(folded);
  slice_op.erase();
  if (pad_op->use_empty()) {
    pad_op.erase();
  }
  return true;
}

// Folds slice of pad operations across all blocks in func.
bool FoldSliceOfPad(mlir::func::FuncOp func) {
  bool changed = false;
  for (mlir::Block& block : func.getBody()) {
    llvm::SmallVector<mlir::stablehlo::SliceOp, 16> slices;
    for (mlir::Operation& op : block) {
      if (auto slice = llvm::dyn_cast<mlir::stablehlo::SliceOp>(&op)) {
        slices.push_back(slice);
      }
    }
    for (mlir::stablehlo::SliceOp slice : slices) {
      if (TryFoldSliceOfPad(slice)) {
        changed = true;
      }
    }
  }
  return changed;
}

// Erases trivially dead side-effect-free operations in reverse order.
void EraseDeadOps(mlir::func::FuncOp func) {
  bool erased = true;
  while (erased) {
    erased = false;
    for (mlir::Block& block : func.getBody()) {
      for (mlir::Operation& op :
           llvm::make_early_inc_range(llvm::reverse(block))) {
        if (mlir::isOpTriviallyDead(&op)) {
          op.erase();
          erased = true;
        }
      }
    }
  }
}

class CanonicalizeSlicesPass
    : public mlir::PassWrapper<CanonicalizeSlicesPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeSlicesPass)

  llvm::StringRef getArgument() const final {
    return "torch-tpu-canonicalize-slices";
  }

  llvm::StringRef getDescription() const final {
    return "Canonicalizes, factors, hoists, and propagates StableHLO slices.";
  }

  void runOnOperation() final {
    mlir::func::FuncOp func = getOperation();
    // Cap fixed-point passes to 8 iterations to bound compiler execution time
    // on deeply nested slice-producer chains while allowing typical network
    // patterns (e.g. multi-head slicing through reshape/transpose/concat/pad)
    // to fully converge.
    constexpr int kMaxIterations = 8;
    int iter = 0;
    for (; iter < kMaxIterations; ++iter) {
      bool progress = false;

      // Step 1: Fold SliceOp of DynamicUpdateSliceOp.
      llvm::SmallVector<mlir::stablehlo::SliceOp, 16> slices;
      func.walk(
          [&slices](mlir::stablehlo::SliceOp op) { slices.push_back(op); });
      bool step1_progress = false;
      for (mlir::stablehlo::SliceOp op : slices) {
        if (op && op->getBlock()) {
          step1_progress |= TryFoldSliceOfDynamicUpdateSlice(op);
        }
      }
      if (step1_progress) {
        EraseDeadOps(func);
        progress = true;
      }

      // Step 2: Canonicalize slice boundaries and concatenate operations.
      if (CanonicalizeSliceBoundaries(func)) {
        EraseDeadOps(func);
        progress = true;
      }

      // Step 3: Fold slice of pad operations.
      if (FoldSliceOfPad(func)) {
        EraseDeadOps(func);
        progress = true;
      }

      // Step 4: Factor multi-slice shared intervals.
      if (FactorMultiSliceSharedIntervals(func)) {
        EraseDeadOps(func);
        progress = true;
      }

      // Step 5: Push slice across cross-axis reshape.
      if (FactorCrossAxisReshapeSlices(func)) {
        EraseDeadOps(func);
        progress = true;
      }

      // Step 6: Hoist slice across elementwise/broadcast/reverse/transpose/etc.
      if (HoistSlicesInFunc(func)) {
        EraseDeadOps(func);
        progress = true;
      }

      if (!progress) {
        break;
      }
    }
    if (iter == kMaxIterations) {
      ABSL_VLOG(1) << "CanonicalizeSlicesPass reached maximum iteration cap ("
                   << kMaxIterations << ") on function @"
                   << func.getName().str();
    }
    EraseDeadOps(func);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createCanonicalizeSlicesPass() {
  return std::make_unique<CanonicalizeSlicesPass>();
}

}  // namespace torch_tpu
