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

#include "csrc/ops/matmul/matmul.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/reductions/sum.h"
#include "csrc/ops/unary.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"

namespace torch_tpu {
namespace {

namespace stablehlo = mlir::stablehlo;

// Returns `[0, 1, ..., count - 1]`, used to construct contiguous batch or
// contracting dimension index lists for `stablehlo.dot_general`.
Dimensions MakeIotaDims(const int64_t count) {
  Dimensions dims(count);
  absl::c_iota(dims, 0);
  return dims;
}

// Emits `stablehlo.reshape` only when `op`'s static shape differs from
// `target_shape`, avoiding redundant identity reshapes in the MLIR graph.
mlir::MlirOp ReshapeIfChanged(mlir::MlirOp op,
                              absl::Span<const int64_t> target_shape) {
  const llvm::ArrayRef<int64_t> target_ref = AsArrayRef(target_shape);
  if (GetTensorTypeOrDie(op).getShape() == target_ref) {
    return op;
  }
  return stablehlo::Reshape(op, target_ref);
}

// Emits `stablehlo.transpose` only when `perm` is not already the identity
// permutation `[0, 1, ..., rank - 1]`.
mlir::MlirOp TransposeIfNeeded(mlir::MlirOp op,
                               absl::Span<const int64_t> perm) {
  if (absl::c_is_sorted(perm)) {
    return op;
  }
  return stablehlo::Transpose(op, AsArrayRef(perm));
}

// Returns the contracting dimension index (`K`) of `lhs`:
//   - 1D `[K]`: axis `0`
//   - ND `[..., M, K]` (`kNone`): axis `lhs_rank - 1`
//   - ND `[..., K, M]` (`kTransposeLastTwo`): axis `lhs_rank - 2`
int64_t GetLhsContractingDim(const int64_t lhs_rank,
                             const MatrixTranspose transpose_lhs) {
  if (lhs_rank == 1) {
    return 0;
  }
  return transpose_lhs == MatrixTranspose::kTransposeLastTwo ? lhs_rank - 2
                                                             : lhs_rank - 1;
}

// Returns the free row dimension index (`M`) of `lhs` (requires `lhs_rank >=
// 2`):
//   - ND `[..., M, K]` (`kNone`): axis `lhs_rank - 2`
//   - ND `[..., K, M]` (`kTransposeLastTwo`): axis `lhs_rank - 1`
int64_t GetLhsFreeMatrixDim(const int64_t lhs_rank,
                            const MatrixTranspose transpose_lhs) {
  return transpose_lhs == MatrixTranspose::kTransposeLastTwo ? lhs_rank - 1
                                                             : lhs_rank - 2;
}

// Returns the contracting dimension index (`K`) of `rhs`:
//   - 1D `[K]`: axis `0`
//   - ND `[..., K, N]` (`kNone`): axis `rhs_rank - 2`
//   - ND `[..., N, K]` (`kTransposeLastTwo`): axis `rhs_rank - 1`
int64_t GetRhsContractingDim(const int64_t rhs_rank,
                             const MatrixTranspose transpose_rhs) {
  if (rhs_rank == 1) {
    return 0;
  }
  return transpose_rhs == MatrixTranspose::kTransposeLastTwo ? rhs_rank - 1
                                                             : rhs_rank - 2;
}

// Returns the free column dimension index (`N`) of `rhs` (requires `rhs_rank
// >= 2`):
//   - ND `[..., K, N]` (`kNone`): axis `rhs_rank - 1`
//   - ND `[..., N, K]` (`kTransposeLastTwo`): axis `rhs_rank - 2`
int64_t GetRhsFreeMatrixDim(const int64_t rhs_rank,
                            const MatrixTranspose transpose_rhs) {
  return transpose_rhs == MatrixTranspose::kTransposeLastTwo ? rhs_rank - 2
                                                             : rhs_rank - 1;
}

// Applies optional pre-contraction transformations to a matmul operand:
//   1. Physical complex conjugation (`BuildConjPhysicalShlo`), which negates
//      the imaginary component of complex tensors and is a no-op for real
//      tensors.
//   2. Optional computation dtype conversion (`stablehlo.convert`) for
//      sub-64-bit integer/boolean operands (e.g., promoting `i8`, `u8`, `i16`,
//      `i32` to `f32` when requested via `target_element_type`).
//      - Floating-point and complex operands retain their input element type so
//        `target_element_type` is passed as `preferred_element_type` on
//        `stablehlo.dot_general` (enabling hardware mixed-precision
//        accumulation without pre-casting).
//      - 64-bit integer (`i64`) operands are skipped here because
//        `BuildTypedDotGeneral` emulates `i64` contractions via `f64` with
//        `Precision::HIGHEST` before converting to `target_element_type`.
absl::StatusOr<mlir::MlirOp> PrepareMatmulOperand(
    mlir::MlirOp op, const ConjugateMode conjugate,
    const std::optional<mlir::ElementType> target_element_type) {
  if (conjugate == ConjugateMode::kConjugate) {
    TT_ASSIGN_OR_RETURN(op, BuildConjPhysicalShlo(op));
  }
  const mlir::RankedTensorType tensor_type = GetTensorTypeOrDie(op);
  const mlir::Type elem_type = tensor_type.getElementType();
  if (target_element_type.has_value() && !IsFloatType(tensor_type) &&
      !IsComplexType(tensor_type) && !elem_type.isInteger(64)) {
    op = stablehlo::ConvertElementType(op, *target_element_type);
  }
  return op;
}

// Bounds the dimensions of the static `input` that have a `boundOp` in `dims`.
mlir::MlirOp ApplyDimensionBounds(mlir::MlirOp input,
                                  const stablehlo::Dimensions& dims) {
  for (int64_t dim = 0; dim < static_cast<int64_t>(dims.size()); ++dim) {
    if (!dims[dim].boundOp.has_value()) continue;
    mlir::MlirOp bound_op(input.getBuilder(), *dims[dim].boundOp);
    mlir::MlirOp size =
        stablehlo::GetDimensionSize(bound_op, dims[dim].boundOpDim);
    input = stablehlo::SetDimensionSize(input, size, dim);
  }
  return input;
}

// Resizes the upper bound of dimension `dim` on `op` down to `target_bound`
// (when `dims[dim].size > target_bound`) and sets its dynamic runtime size to
// `new_runtime_size`, preserving dynamic bounds on all other axes.
mlir::MlirOp ResizeBoundedDimension(mlir::MlirOp op, const int64_t dim,
                                    const int64_t target_bound,
                                    mlir::MlirOp new_runtime_size) {
  stablehlo::Dimensions dims = GetDimensions(op);
  if (dims[dim].size != target_bound) {
    const mlir::Type i32_type = op.getBuilder().getOpBuilder().getI32Type();
    mlir::MlirOp static_op = op;
    for (int64_t k = 0; k < static_cast<int64_t>(dims.size()); ++k) {
      if (!dims[k].boundOp.has_value()) continue;
      mlir::MlirOp bound =
          MakeScalarConstant(op.getBuilder(), dims[k].size, i32_type);
      static_op = stablehlo::SetDimensionSize(static_op, bound, k);
    }
    const Dimensions starts(dims.size(), 0);
    Dimensions limits = GetStaticShape(dims);
    limits[dim] = target_bound;
    const Dimensions strides(dims.size(), 1);
    mlir::MlirOp sliced = stablehlo::Slice(static_op, starts, limits, strides);
    dims[dim].boundOp = std::nullopt;
    sliced = ApplyDimensionBounds(sliced, dims);
    return stablehlo::SetDimensionSize(sliced, new_runtime_size, dim);
  }
  return stablehlo::SetDimensionSize(op, new_runtime_size, dim);
}

// Computes the runtime `i32` dimension size op for a reconciled bounded-dynamic
// dimension of upper bound `target_bound`, combining both dynamic sources when
// both `a` and `b` carry dynamic bounds and clamping to `target_bound` when a
// single dynamic source's upper bound exceeds `target_bound`.
mlir::MlirOp GetReconciledRuntimeSize(mlir::MlirBuilder& builder,
                                      const stablehlo::DimensionInfo& a,
                                      const stablehlo::DimensionInfo& b,
                                      const int64_t target_bound) {
  if (a.boundOp.has_value() && b.boundOp.has_value()) {
    mlir::MlirOp a_bound(builder, *a.boundOp);
    mlir::MlirOp b_bound(builder, *b.boundOp);
    mlir::MlirOp a_size = stablehlo::GetDimensionSize(a_bound, a.boundOpDim);
    mlir::MlirOp b_size = stablehlo::GetDimensionSize(b_bound, b.boundOpDim);
    return stablehlo::Min(a_size, b_size);
  }
  const stablehlo::DimensionInfo& dyn_dim = a.boundOp.has_value() ? a : b;
  mlir::MlirOp dyn_bound(builder, *dyn_dim.boundOp);
  mlir::MlirOp runtime_size =
      stablehlo::GetDimensionSize(dyn_bound, dyn_dim.boundOpDim);
  if (dyn_dim.size > target_bound) {
    mlir::MlirOp max_bound = MakeScalarConstant(
        builder, target_bound, builder.getOpBuilder().getI32Type());
    return stablehlo::Min(runtime_size, max_bound);
  }
  return runtime_size;
}

// Validates that all indices in `batch_dims` and `contract_dims` are within
// `[0, rank)` and disjoint.
absl::Status ValidateOperandDotDims(
    const int64_t rank, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> contract_dims,
    const absl::string_view operand_name) {  // ABSL_STRING_VIEW_OK
  llvm::SmallVector<bool, 8> seen(rank, false);
  for (const int64_t d : batch_dims) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by matmul_test.
        d >= 0 && d < rank, error::kInvalidArgument)
        << operand_name << " batch dimension " << d
        << " out of bounds for rank " << rank;
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by matmul_test.
        !seen[d], error::kInvalidArgument)
        << "duplicate " << operand_name << " batch dimension " << d;
    seen[d] = true;
  }
  for (const int64_t d : contract_dims) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by matmul_test.
        d >= 0 && d < rank, error::kInvalidArgument)
        << operand_name << " contracting dimension " << d
        << " out of bounds for rank " << rank;
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by matmul_test.
        !seen[d], error::kInvalidArgument)
        << operand_name << " dimension " << d
        << " cannot be both batch/contracting or duplicated";
    seen[d] = true;
  }
  return absl::OkStatus();
}

// Validates and reconciles paired dimension indices (`l_indices` on `lhs` vs
// `r_indices` on `rhs`) for either batch or contracting axes.
absl::Status ReconcilePairedDotDimensions(
    mlir::MlirOp& lhs, mlir::MlirOp& rhs, const stablehlo::Dimensions& lhs_dims,
    const stablehlo::Dimensions& rhs_dims, absl::Span<const int64_t> l_indices,
    absl::Span<const int64_t> r_indices,
    const absl::string_view kind) {  // ABSL_STRING_VIEW_OK
  for (size_t i = 0; i < l_indices.size(); ++i) {
    const int64_t d_l = l_indices[i];
    const int64_t d_r = r_indices[i];
    const stablehlo::DimensionInfo& l_dim = lhs_dims[d_l];
    const stablehlo::DimensionInfo& r_dim = rhs_dims[d_r];
    if (!l_dim.boundOp.has_value() && !r_dim.boundOp.has_value()) {
      TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by matmul_test.
          l_dim.size == r_dim.size, error::kInvalidArgument)
          << kind << " dimension sizes do not match at pair " << i << ": "
          << l_dim.size << " vs " << r_dim.size;
      continue;
    }
    if (!l_dim.boundOp.has_value()) {
      TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by matmul_test.
          l_dim.size <= r_dim.size, error::kInvalidArgument)
          << kind << " static dimension size " << l_dim.size
          << " exceeds bounded dynamic upper bound " << r_dim.size
          << " at pair " << i;
    } else if (!r_dim.boundOp.has_value()) {
      TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by matmul_test.
          r_dim.size <= l_dim.size, error::kInvalidArgument)
          << kind << " static dimension size " << r_dim.size
          << " exceeds bounded dynamic upper bound " << l_dim.size
          << " at pair " << i;
    } else if (l_dim.size == r_dim.size) {
      continue;
    }
    const int64_t target_bound = std::min(l_dim.size, r_dim.size);
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by matmul_test.
        target_bound > 0, error::kInvalidArgument)
        << "cannot reconcile bounded dynamic " << kind
        << " dimension with zero bound: " << l_dim.size << " vs " << r_dim.size;
    mlir::MlirOp runtime_size =
        GetReconciledRuntimeSize(lhs.getBuilder(), l_dim, r_dim, target_bound);
    lhs = ResizeBoundedDimension(lhs, d_l, target_bound, runtime_size);
    rhs = ResizeBoundedDimension(rhs, d_r, target_bound, runtime_size);
  }
  return absl::OkStatus();
}

// Validates paired batch and contracting dimensions of `lhs` and `rhs` and
// reconciles any paired axis where one or both operands have bounded dynamic
// dimensions with mismatched upper bounds or mixed static/bounded metadata.
absl::Status ValidateAndReconcileDotDimensions(
    mlir::MlirOp& lhs, mlir::MlirOp& rhs,
    absl::Span<const int64_t> lhs_batching,
    absl::Span<const int64_t> rhs_batching,
    absl::Span<const int64_t> lhs_contracting,
    absl::Span<const int64_t> rhs_contracting) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by matmul_test.
      lhs_batching.size() == rhs_batching.size(), error::kInvalidArgument)
      << "lhs and rhs batching dimension counts must match: "
      << lhs_batching.size() << " vs " << rhs_batching.size();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by matmul_test.
      lhs_contracting.size() == rhs_contracting.size(), error::kInvalidArgument)
      << "lhs and rhs contracting dimension counts must match: "
      << lhs_contracting.size() << " vs " << rhs_contracting.size();

  const stablehlo::Dimensions lhs_dims = GetDimensions(lhs);
  const stablehlo::Dimensions rhs_dims = GetDimensions(rhs);
  TT_RETURN_IF_ERROR(ValidateOperandDotDims(
      lhs_dims.size(), lhs_batching, lhs_contracting, /*operand_name=*/"lhs"));
  TT_RETURN_IF_ERROR(ValidateOperandDotDims(
      rhs_dims.size(), rhs_batching, rhs_contracting, /*operand_name=*/"rhs"));
  TT_RETURN_IF_ERROR(ReconcilePairedDotDimensions(
      lhs, rhs, lhs_dims, rhs_dims, lhs_batching, rhs_batching, "batch"));
  return ReconcilePairedDotDimensions(lhs, rhs, lhs_dims, rhs_dims,
                                      lhs_contracting, rhs_contracting,
                                      "contracting");
}

// Validates and reconciles two non-unit (`size > 1`) aligned batch dimensions
// for NumPy batch broadcasting.
absl::StatusOr<stablehlo::DimensionInfo> ReconcileNonUnitBroadcastBatchDim(
    const stablehlo::DimensionInfo& l_dim,
    const stablehlo::DimensionInfo& r_dim) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by matmul_test.
      (l_dim.boundOp.has_value() || l_dim.size <= r_dim.size) &&
          (r_dim.boundOp.has_value() || r_dim.size <= l_dim.size),
      error::kInvalidArgument)
      << "batch dimensions are not broadcastable: " << l_dim.size << " vs "
      << r_dim.size;
  const int64_t target_bound = std::min(l_dim.size, r_dim.size);
  const bool prefer_lhs =
      l_dim.boundOp.has_value() &&
      (!r_dim.boundOp.has_value() || l_dim.size <= r_dim.size);
  const stablehlo::DimensionInfo& chosen = prefer_lhs ? l_dim : r_dim;
  return stablehlo::DimensionInfo{target_bound, chosen.boundOp,
                                  chosen.boundOpDim};
}

// Computes the broadcasted `DimensionInfo` at aligned batch axis `i` (out of
// `batch_rank = max(lhs_batch_rank, rhs_batch_rank)`), right-aligning the
// leading batch dimensions `[0 .. rank - 3]` of `lhs` and `rhs` per NumPy
// broadcasting rules and reconciling bounded dynamic dimension metadata.
absl::StatusOr<stablehlo::DimensionInfo> ComputeSingleBroadcastBatchDim(
    const int64_t i, const int64_t batch_rank,
    const stablehlo::Dimensions& lhs_dims,
    const stablehlo::Dimensions& rhs_dims) {
  const int64_t lhs_batch_rank =
      std::max<int64_t>(0, static_cast<int64_t>(lhs_dims.size()) - 2);
  const int64_t rhs_batch_rank =
      std::max<int64_t>(0, static_cast<int64_t>(rhs_dims.size()) - 2);
  const int64_t lhs_idx = i - (batch_rank - lhs_batch_rank);
  const int64_t rhs_idx = i - (batch_rank - rhs_batch_rank);
  if (lhs_idx < 0) {
    return rhs_dims[rhs_idx];
  }
  if (rhs_idx < 0) {
    return lhs_dims[lhs_idx];
  }
  const stablehlo::DimensionInfo& l_dim = lhs_dims[lhs_idx];
  const stablehlo::DimensionInfo& r_dim = rhs_dims[rhs_idx];
  if (l_dim.size == 1 && r_dim.size == 1) {
    return !l_dim.boundOp.has_value() ? l_dim : r_dim;
  }
  if (l_dim.size == 1) {
    return r_dim;
  }
  if (r_dim.size == 1) {
    return l_dim;
  }
  return ReconcileNonUnitBroadcastBatchDim(l_dim, r_dim);
}

// Computes the common broadcasted batch prefix `[B_0, ..., B_{batch_rank - 1}]`
// for the leading batch dimensions of `lhs` and `rhs`.
absl::StatusOr<stablehlo::Dimensions> ComputeBroadcastBatchDims(
    const stablehlo::Dimensions& lhs_dims,
    const stablehlo::Dimensions& rhs_dims) {
  const int64_t lhs_batch_rank =
      std::max<int64_t>(0, static_cast<int64_t>(lhs_dims.size()) - 2);
  const int64_t rhs_batch_rank =
      std::max<int64_t>(0, static_cast<int64_t>(rhs_dims.size()) - 2);
  const int64_t batch_rank = std::max(lhs_batch_rank, rhs_batch_rank);
  stablehlo::Dimensions batch_dims;
  batch_dims.reserve(batch_rank);
  for (int64_t i = 0; i < batch_rank; ++i) {
    TT_ASSIGN_OR_RETURN(
        const stablehlo::DimensionInfo dim,
        ComputeSingleBroadcastBatchDim(i, batch_rank, lhs_dims, rhs_dims));
    batch_dims.push_back(dim);
  }
  return batch_dims;
}

// Reconciles a single leading batch axis `axis` of `matrix` against the
// broadcasted `target_dim` before `BroadcastIfNeeded`:
//   - If `cur_dim` is bounded dynamic `?<=1` and `target_dim` is static `1` or
//     `> 1`, converts `cur_dim` to static `1` so `BroadcastIfNeeded` can
//     expand it cleanly.
//   - If `cur_dim` is non-unit (`size > 1`), preserves `cur_dim` in
//     `target_dim` so `BuildTypedDotGeneral` reconciles paired batch axes.
void ReconcileMatrixBatchAxis(mlir::MlirOp& matrix,
                              stablehlo::Dimensions& matrix_dims,
                              const int64_t axis,
                              stablehlo::DimensionInfo& target_dim) {
  const stablehlo::DimensionInfo& cur_dim = matrix_dims[axis];
  if (cur_dim.size > 1) {
    target_dim = cur_dim;
    return;
  }
  if (!cur_dim.boundOp.has_value()) {
    return;
  }
  if (target_dim.size > 1 || !target_dim.boundOp.has_value()) {
    mlir::MlirOp one =
        MakeScalarConstant(matrix.getBuilder(), 1,
                           matrix.getBuilder().getOpBuilder().getI32Type());
    matrix = stablehlo::SetDimensionSize(matrix, one, axis);
    matrix_dims = GetDimensions(matrix);
    return;
  }
  target_dim = cur_dim;
}

// Broadcasts the leading batch dimensions of a rank >= 2 `matrix` from
// `[orig_batch..., R, C]` to `[batch_dims..., R, C]`, reconciling any batch
// dimension whose upper bound or bounded-dynamic status differs from
// `batch_dims` before calling `BroadcastIfNeeded`.
absl::StatusOr<mlir::MlirOp> BroadcastMatrixBatchDims(
    mlir::MlirOp matrix, const stablehlo::Dimensions& batch_dims) {
  stablehlo::Dimensions matrix_dims = GetDimensions(matrix);
  const int64_t rank = matrix_dims.size();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by matmul_test.
      rank >= 2, error::kInvalidArgument)
      << "expected matrix rank >= 2, got " << rank;
  const int64_t matrix_batch_rank = rank - 2;
  const int64_t rank_offset =
      static_cast<int64_t>(batch_dims.size()) - matrix_batch_rank;
  stablehlo::Dimensions target_dims = batch_dims;

  for (int64_t i = 0; i < matrix_batch_rank; ++i) {
    ReconcileMatrixBatchAxis(matrix, matrix_dims, i,
                             target_dims[rank_offset + i]);
  }

  target_dims.push_back(matrix_dims[rank - 2]);
  target_dims.push_back(matrix_dims[rank - 1]);
  return BroadcastIfNeeded(matrix, target_dims);
}

// Classification of each aligned forward batch axis `(l_size, r_size)` for
// zero-broadcast static batched matmul:
//   - `kTrivial` (`l_size == 1 && r_size == 1`): Dropped before `dot_general`
//     and restored by the final output reshape.
//   - `kShared` (`l_size == r_size > 1`): Present in both operands; passed as a
//     true `dot_general` batch dimension (`lhs_batch_dims`, `rhs_batch_dims`).
//   - `kLhsOnly` (`l_size > 1 && r_size == 1`): Broadcasted across `rhs`; kept
//     as a free (non-batch, non-contracting) dimension on `lhs` so `rhs` is not
//     broadcast-materialized.
//   - `kRhsOnly` (`l_size == 1 && r_size > 1`): Broadcasted across `lhs`; kept
//     as a free (non-batch, non-contracting) dimension on `rhs` so `lhs` is not
//     broadcast-materialized.
enum class ForwardBatchAxisKind {
  kTrivial,
  kShared,
  kLhsOnly,
  kRhsOnly,
};

struct CoalescedForwardAxis {
  ForwardBatchAxisKind kind;
  int64_t size;
};

absl::StatusOr<ForwardBatchAxisKind> ClassifyForwardBatchAxis(
    const int64_t l_size, const int64_t r_size) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by matmul_test.
      l_size == r_size || l_size == 1 || r_size == 1, error::kInvalidArgument)
      << "batch dimensions are not broadcastable: " << l_size << " vs "
      << r_size;
  if (l_size == 1 && r_size == 1) {
    return ForwardBatchAxisKind::kTrivial;
  }
  if (l_size == r_size) {
    return ForwardBatchAxisKind::kShared;
  }
  return l_size > 1 ? ForwardBatchAxisKind::kLhsOnly
                    : ForwardBatchAxisKind::kRhsOnly;
}

// Right-aligns the batch dimensions of `lhs_shape` and `rhs_shape`, drops
// `kTrivial` (`1 vs 1`) axes, and coalesces adjacent batch axes of the same
// `ForwardBatchAxisKind` into single flattened axes to minimize tensor rank
// during `dot_general`.
absl::StatusOr<llvm::SmallVector<CoalescedForwardAxis>>
CoalesceForwardBatchAxes(llvm::ArrayRef<int64_t> lhs_shape,
                         llvm::ArrayRef<int64_t> rhs_shape) {
  const int64_t lhs_batch_rank =
      std::max<int64_t>(0, static_cast<int64_t>(lhs_shape.size()) - 2);
  const int64_t rhs_batch_rank =
      std::max<int64_t>(0, static_cast<int64_t>(rhs_shape.size()) - 2);
  const int64_t batch_rank = std::max(lhs_batch_rank, rhs_batch_rank);
  llvm::SmallVector<CoalescedForwardAxis> axes;
  axes.reserve(batch_rank);
  for (int64_t i = 0; i < batch_rank; ++i) {
    const int64_t lhs_idx = i - (batch_rank - lhs_batch_rank);
    const int64_t rhs_idx = i - (batch_rank - rhs_batch_rank);
    const int64_t l_size = lhs_idx < 0 ? 1 : lhs_shape[lhs_idx];
    const int64_t r_size = rhs_idx < 0 ? 1 : rhs_shape[rhs_idx];
    TT_ASSIGN_OR_RETURN(const ForwardBatchAxisKind kind,
                        ClassifyForwardBatchAxis(l_size, r_size));
    const int64_t bcast_size = std::max(l_size, r_size);
    if (kind == ForwardBatchAxisKind::kTrivial) {
      continue;
    }
    if (!axes.empty() && axes.back().kind == kind) {
      TT_ASSIGN_OR_RETURN(axes.back().size,
                          SafeMultiply(axes.back().size, bcast_size));
    } else {
      axes.push_back({kind, bcast_size});
    }
  }
  return axes;
}

// Holds the reshaped operand shapes and `dot_general` batch dimension lists
// constructed from `CoalescedForwardAxis`.
struct PreparedForwardBatchShapes {
  Dimensions lhs_target_shape;
  Dimensions rhs_target_shape;
  Dimensions lhs_batch_dims;
  Dimensions rhs_batch_dims;
  int64_t num_lhs_only = 0;
  int64_t num_rhs_only = 0;
};

PreparedForwardBatchShapes PrepareForwardBatchShapes(
    absl::Span<const CoalescedForwardAxis> axes,
    llvm::ArrayRef<int64_t> lhs_shape, llvm::ArrayRef<int64_t> rhs_shape) {
  PreparedForwardBatchShapes shapes;
  for (const CoalescedForwardAxis& axis : axes) {
    if (axis.kind == ForwardBatchAxisKind::kShared) {
      shapes.lhs_batch_dims.push_back(shapes.lhs_target_shape.size());
      shapes.lhs_target_shape.push_back(axis.size);
      shapes.rhs_batch_dims.push_back(shapes.rhs_target_shape.size());
      shapes.rhs_target_shape.push_back(axis.size);
    } else if (axis.kind == ForwardBatchAxisKind::kLhsOnly) {
      shapes.lhs_target_shape.push_back(axis.size);
      ++shapes.num_lhs_only;
    } else {
      shapes.rhs_target_shape.push_back(axis.size);
      ++shapes.num_rhs_only;
    }
  }
  const int64_t lhs_rank = lhs_shape.size();
  const int64_t rhs_rank = rhs_shape.size();
  shapes.lhs_target_shape.push_back(lhs_shape[lhs_rank - 2]);
  shapes.lhs_target_shape.push_back(lhs_shape[lhs_rank - 1]);
  shapes.rhs_target_shape.push_back(rhs_shape[rhs_rank - 2]);
  shapes.rhs_target_shape.push_back(rhs_shape[rhs_rank - 1]);
  return shapes;
}

// Computes the output `stablehlo::Dimensions` (including any bounded dynamic
// dimension metadata) of `matmul(lhs, rhs)` across all rank combinations.
absl::StatusOr<stablehlo::Dimensions> ComputeMatmulOutputDimensions(
    const stablehlo::Dimensions& lhs_dims,
    const stablehlo::Dimensions& rhs_dims, const MatmulShloOptions& options) {
  const int64_t lhs_rank = lhs_dims.size();
  const int64_t rhs_rank = rhs_dims.size();
  TT_ASSIGN_OR_RETURN(stablehlo::Dimensions output_dims,
                      ComputeBroadcastBatchDims(lhs_dims, rhs_dims));
  if (lhs_rank >= 2) {
    output_dims.push_back(
        lhs_dims[GetLhsFreeMatrixDim(lhs_rank, options.transpose_lhs)]);
  }
  if (rhs_rank >= 2) {
    output_dims.push_back(
        rhs_dims[GetRhsFreeMatrixDim(rhs_rank, options.transpose_rhs)]);
  }
  return output_dims;
}

// Resolves `preferred_element_type` for complex inputs: if `input_elem_type` is
// complex and `preferred_element_type` was given as a real float (`F32` or
// `F64`), maps it to the corresponding complex element type (`COMPLEXF32` or
// `COMPLEXF64`) so `stablehlo.dot_general` receives a valid complex result
// type.
std::optional<mlir::ElementType> ResolvePreferredElementType(
    const mlir::Type input_elem_type,
    const std::optional<mlir::ElementType> preferred_element_type) {
  if (!preferred_element_type.has_value() ||
      !mlir::isa<mlir::ComplexType>(input_elem_type)) {
    return preferred_element_type;
  }
  if (*preferred_element_type == mlir::ElementType::F32) {
    return mlir::ElementType::COMPLEXF32;
  }
  if (*preferred_element_type == mlir::ElementType::F64) {
    return mlir::ElementType::COMPLEXF64;
  }
  return preferred_element_type;
}

// Handles matmul when `lhs` or `rhs` contains a zero-sized dimension (including
// when other dimensions are bounded dynamic):
//   - If the output shape itself has a zero dimension (e.g., `M == 0`, `N ==
//     0`, or a batch dimension is `0`), returns an empty tensor of that shape.
//   - If the output shape has non-zero numel because only the contracting
//     dimension `K == 0` (e.g., `[M, 0] x [0, N] -> [M, N]`), the sum over an
//     empty contracting axis is the additive identity `0`, so a zero-filled
//     constant tensor is emitted.
//   - Preserves any bounded dynamic dimensions in `output_dims` via
//     `ApplyDimensionBounds`.
absl::StatusOr<mlir::MlirOp> EmitZeroElementMatmul(
    mlir::MlirOp lhs, const stablehlo::Dimensions& lhs_dims,
    const stablehlo::Dimensions& rhs_dims, const MatmulShloOptions& options) {
  TT_ASSIGN_OR_RETURN(
      const stablehlo::Dimensions output_dims,
      ComputeMatmulOutputDimensions(lhs_dims, rhs_dims, options));
  const Dimensions output_shape = GetStaticShape(output_dims);
  const std::optional<mlir::ElementType> resolved_target_type =
      ResolvePreferredElementType(GetElementTypeOrSelf(lhs),
                                  options.target_element_type);
  const mlir::Type element_type =
      resolved_target_type.has_value()
          ? mlir::getElementType(lhs.getContext(), *resolved_target_type)
          : GetElementTypeOrSelf(lhs);
  const llvm::ArrayRef<int64_t> out_shape_ref =
      AsArrayRef(absl::MakeConstSpan(output_shape));
  mlir::MlirOp result;
  if (llvm::is_contained(output_shape, 0)) {
    TT_ASSIGN_OR_RETURN(
        result,
        MakeZeroSizedTensor(lhs.getBuilder(), element_type, out_shape_ref));
  } else {
    result = MakeConstant(lhs.getBuilder(), 0, element_type, out_shape_ref);
  }
  return ApplyDimensionBounds(result, output_dims);
}

// Returns true if `element_type` is `complex<f64>`.
bool IsComplexF64(const mlir::Type element_type) {
  if (const auto complex_type =
          mlir::dyn_cast<mlir::ComplexType>(element_type)) {
    return complex_type.getElementType().isF64();
  }
  return false;
}

// Returns true when an `ND x 1D` or `ND x 2D` matmul (`lhs_rank >= 3`,
// `rhs_rank <= 2`) has static shapes, `LeadingDimFolding::kAuto`, and
// non-transposed `lhs` and `rhs`, allowing all leading batch and row dimensions
// `[..., M]` of `lhs` to be collapsed into a single 2D matrix
// `[prod(..., M), K]`.
//
// `transpose_rhs == kTransposeLastTwo` (`[B, S, K] @ [N, K]^T`) is
// intentionally excluded so XLA's `TpuConvCanonicalizer` does not misclassify
// `Reshape(3D->2D) + Dot(lhs_contract=1, rhs_contract=1) + Reshape(2D->3D)` as
// a 3D windowed convolution (`0bf_oi0->0bf`).
bool CanFoldLhsTo2D(const int64_t lhs_rank, const int64_t rhs_rank,
                    const bool use_broadcast_batched_path,
                    const MatmulShloOptions& options) {
  return !use_broadcast_batched_path &&
         options.leading_dim_folding == LeadingDimFolding::kAuto &&
         lhs_rank >= 3 && rhs_rank <= 2 &&
         options.transpose_lhs == MatrixTranspose::kNone &&
         options.transpose_rhs == MatrixTranspose::kNone;
}

// Executes `ND x 1D` (`[..., M, K] x [K] -> [..., M]`) or `ND x 2D`
// (`[..., M, K] x [K, N] -> [..., M, N]`) as a single unbatched 2D GEMM by
// reshaping `lhs` to `[prod(..., M), K]`, contracting with `rhs`, and reshaping
// the result back to `[..., M]` or `[..., M, N]`.
absl::StatusOr<mlir::MlirOp> EmitFoldedLhsMatmul(
    mlir::MlirOp lhs, mlir::MlirOp rhs, const MatmulShloOptions& options,
    const MatmulPrecisionPair precision) {
  const llvm::ArrayRef<int64_t> lhs_shape = GetTensorTypeOrDie(lhs).getShape();
  const llvm::ArrayRef<int64_t> rhs_shape = GetTensorTypeOrDie(rhs).getShape();
  const int64_t rhs_rank = rhs_shape.size();
  TT_ASSIGN_OR_RETURN(const int64_t leading_rows,
                      NumElements(absl::MakeConstSpan(lhs_shape.drop_back(1))));
  const int64_t k_dim = lhs_shape.back();
  mlir::MlirOp lhs_2d = ReshapeIfChanged(lhs, Dimensions{leading_rows, k_dim});
  const int64_t rhs_contract_dim =
      GetRhsContractingDim(rhs_rank, options.transpose_rhs);
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp dot_2d,
      BuildTypedDotGeneral(lhs_2d, rhs, /*lhs_batching=*/{},
                           /*rhs_batching=*/{},
                           /*lhs_contracting=*/{1},
                           /*rhs_contracting=*/{rhs_contract_dim}, precision,
                           options.target_element_type));
  Dimensions output_shape(lhs_shape.begin(), lhs_shape.end() - 1);
  if (rhs_rank == 2) {
    output_shape.push_back(rhs_shape.back());
  }
  return ReshapeIfChanged(dot_2d, output_shape);
}

// Computes the transpose permutation needed to restore the canonical batch axis
// order after `EmitStaticBatchedMatmul`'s `dot_general`.
//
// `stablehlo.dot_general` orders its output dimensions as:
//   `[shared_batch (0 .. num_shared - 1),`
//   ` lhs_free (kLhsOnly..., M),`
//   ` rhs_free (kRhsOnly..., N)]`
// This function builds the permutation that re-interleaves `kShared`,
// `kLhsOnly`, and `kRhsOnly` axes in their original coalesced batch order,
// followed by `[M, N]`.
Dimensions BuildForwardOutputPermutation(
    absl::Span<const CoalescedForwardAxis> axes, const int64_t num_shared,
    const int64_t num_lhs_only, const int64_t num_rhs_only) {
  Dimensions perm;
  perm.reserve(axes.size() + 2);
  int64_t shared_idx = 0;
  int64_t lhs_only_idx = num_shared;
  const int64_t m_idx = num_shared + num_lhs_only;
  int64_t rhs_only_idx = m_idx + 1;
  const int64_t n_idx = rhs_only_idx + num_rhs_only;
  for (const CoalescedForwardAxis& axis : axes) {
    if (axis.kind == ForwardBatchAxisKind::kShared) {
      perm.push_back(shared_idx++);
    } else if (axis.kind == ForwardBatchAxisKind::kLhsOnly) {
      perm.push_back(lhs_only_idx++);
    } else {
      perm.push_back(rhs_only_idx++);
    }
  }
  perm.push_back(m_idx);
  perm.push_back(n_idx);
  return perm;
}

// Lowers static `ND x ND` batched matmul (`[..., M, K] x [..., K, N] -> [...,
// M, N]`) without materializing `stablehlo.broadcast_in_dim` copies:
//   - Fast path: When `lhs` and `rhs` have identical leading batch dimensions
//     (e.g., `3D x 3D` `bmm`, including `B == 1`), emits a single
//     `dot_general` directly with zero reshapes.
//   - General broadcast path: Coalesces aligned batch axes into `kShared`,
//     `kLhsOnly`, and `kRhsOnly` groups (dropping `kTrivial` size-1 axes),
//     reshapes `lhs` and `rhs`, passes only `kShared` axes as `dot_general`
//     batch dimensions while leaving `kLhsOnly` and `kRhsOnly` axes as free
//     dimensions, and permutes/reshapes the output.
absl::StatusOr<mlir::MlirOp> EmitStaticBatchedMatmul(
    mlir::MlirOp lhs, mlir::MlirOp rhs, const MatmulShloOptions& options,
    const MatmulPrecisionPair precision) {
  const llvm::ArrayRef<int64_t> lhs_shape = GetTensorTypeOrDie(lhs).getShape();
  const llvm::ArrayRef<int64_t> rhs_shape = GetTensorTypeOrDie(rhs).getShape();
  const int64_t lhs_rank = lhs_shape.size();
  const int64_t rhs_rank = rhs_shape.size();
  if (lhs_rank == rhs_rank &&
      lhs_shape.drop_back(2) == rhs_shape.drop_back(2)) {
    const Dimensions batch_dims = MakeIotaDims(lhs_rank - 2);
    const int64_t lhs_contract =
        GetLhsContractingDim(lhs_rank, options.transpose_lhs);
    const int64_t rhs_contract =
        GetRhsContractingDim(rhs_rank, options.transpose_rhs);
    return BuildTypedDotGeneral(lhs, rhs, batch_dims, batch_dims,
                                {lhs_contract}, {rhs_contract}, precision,
                                options.target_element_type);
  }

  TT_ASSIGN_OR_RETURN(const llvm::SmallVector<CoalescedForwardAxis> axes,
                      CoalesceForwardBatchAxes(lhs_shape, rhs_shape));
  TT_ASSIGN_OR_RETURN(
      Dimensions output_shape,
      InferSize(lhs_shape.drop_back(2), rhs_shape.drop_back(2)));
  output_shape.push_back(
      lhs_shape[GetLhsFreeMatrixDim(lhs_rank, options.transpose_lhs)]);
  output_shape.push_back(
      rhs_shape[GetRhsFreeMatrixDim(rhs_rank, options.transpose_rhs)]);

  const PreparedForwardBatchShapes shapes =
      PrepareForwardBatchShapes(axes, lhs_shape, rhs_shape);
  lhs = ReshapeIfChanged(lhs, shapes.lhs_target_shape);
  rhs = ReshapeIfChanged(rhs, shapes.rhs_target_shape);
  const int64_t lhs_contract = GetLhsContractingDim(
      shapes.lhs_target_shape.size(), options.transpose_lhs);
  const int64_t rhs_contract = GetRhsContractingDim(
      shapes.rhs_target_shape.size(), options.transpose_rhs);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp result,
                      BuildTypedDotGeneral(
                          lhs, rhs, shapes.lhs_batch_dims,
                          shapes.rhs_batch_dims, {lhs_contract}, {rhs_contract},
                          precision, options.target_element_type));
  const Dimensions perm =
      BuildForwardOutputPermutation(axes, shapes.lhs_batch_dims.size(),
                                    shapes.num_lhs_only, shapes.num_rhs_only);
  result = TransposeIfNeeded(result, perm);
  return ReshapeIfChanged(result, output_shape);
}

// Explicitly broadcast-aligns leading batch dimensions of `lhs` and `rhs` and
// emits a batched `dot_general` (used when bounded dynamic dimensions or 3D+
// `complex<f64>` operands are present).
absl::StatusOr<mlir::MlirOp> EmitBroadcastBatchedMatmul(
    mlir::MlirOp lhs, mlir::MlirOp rhs, const stablehlo::Dimensions& lhs_dims,
    const stablehlo::Dimensions& rhs_dims, const MatmulShloOptions& options,
    const MatmulPrecisionPair precision) {
  TT_ASSIGN_OR_RETURN(const stablehlo::Dimensions batch_dims,
                      ComputeBroadcastBatchDims(lhs_dims, rhs_dims));
  TT_ASSIGN_OR_RETURN(lhs, BroadcastMatrixBatchDims(lhs, batch_dims));
  TT_ASSIGN_OR_RETURN(rhs, BroadcastMatrixBatchDims(rhs, batch_dims));
  const int64_t batch_rank = batch_dims.size();
  const Dimensions batch_indices = MakeIotaDims(batch_rank);
  const int64_t bcast_lhs_contract =
      GetLhsContractingDim(batch_rank + 2, options.transpose_lhs);
  const int64_t bcast_rhs_contract =
      GetRhsContractingDim(batch_rank + 2, options.transpose_rhs);
  return BuildTypedDotGeneral(lhs, rhs, batch_indices, batch_indices,
                              {bcast_lhs_contract}, {bcast_rhs_contract},
                              precision, options.target_element_type);
}

// Classification of each aligned batch axis `(grad_size, target_size,
// operand_size)` during `matmul_backward` when computing the gradient with
// respect to `target` (`self` or `other`) using the paired forward `operand`:
//   - `kTrivial` (`grad_size == 1`): Size-1 in `grad`, `target`, and `operand`;
//     dropped before `dot_general` and restored by the final output reshape.
//   - `kShared` (`grad_size > 1 && target_size > 1 && operand_size > 1`):
//     Present in `grad`, `target`, and `operand`; passed as a true
//     `dot_general` batch dimension.
//   - `kContract` (`grad_size > 1 && target_size == 1`): `target` had a
//     missing or singleton batch dimension that was broadcasted across
//     `operand` (`operand_size == grad_size`) during the forward pass. Instead
//     of computing an unreduced batched `dot_general` and emitting a separate
//     `stablehlo.reduce(add)` pass, this batch axis is passed as an
//     **additional contracting dimension** in both `grad_contract_dims` and
//     `operand_contract_dims` so the TPU MXU accumulates the broadcast
//     sum-reduction in-place without materializing the unreduced tensor in HBM.
//   - `kGradFree` (`grad_size > 1 && target_size > 1 && operand_size == 1`):
//     `operand` was broadcasted across `target` during the forward pass; kept
//     as a free dimension on `grad` (and squeezed out of `operand`) so
//     `operand` does not need to be broadcast-materialized before
//     `dot_general`.
enum class BackwardBatchAxisKind {
  kTrivial,
  kShared,
  kContract,
  kGradFree,
};

struct CoalescedBackwardAxis {
  BackwardBatchAxisKind kind;
  int64_t size;
};

BackwardBatchAxisKind ClassifyBackwardBatchAxis(const int64_t grad_size,
                                                const int64_t target_size,
                                                const int64_t operand_size) {
  if (grad_size == 1) {
    return BackwardBatchAxisKind::kTrivial;
  }
  if (target_size == 1) {
    return BackwardBatchAxisKind::kContract;
  }
  if (operand_size == 1) {
    return BackwardBatchAxisKind::kGradFree;
  }
  return BackwardBatchAxisKind::kShared;
}

// Right-aligns the batch dimensions of `grad_shape`, `target_shape`, and
// `operand_shape`, validates broadcast compatibility, drops `kTrivial`
// (`grad_size == 1`) axes, and coalesces adjacent batch axes of the same
// `BackwardBatchAxisKind`.
absl::StatusOr<llvm::SmallVector<CoalescedBackwardAxis>>
CoalesceBackwardBatchAxes(llvm::ArrayRef<int64_t> grad_shape,
                          absl::Span<const int64_t> target_shape,
                          llvm::ArrayRef<int64_t> operand_shape) {
  const int64_t batch_rank = grad_shape.size() - 2;
  const int64_t target_batch_rank =
      std::max<int64_t>(0, static_cast<int64_t>(target_shape.size()) - 2);
  const int64_t operand_batch_rank =
      std::max<int64_t>(0, static_cast<int64_t>(operand_shape.size()) - 2);
  llvm::SmallVector<CoalescedBackwardAxis> axes;
  axes.reserve(batch_rank);
  for (int64_t i = 0; i < batch_rank; ++i) {
    const int64_t target_idx = i - (batch_rank - target_batch_rank);
    const int64_t operand_idx = i - (batch_rank - operand_batch_rank);
    const int64_t target_size = target_idx < 0 ? 1 : target_shape[target_idx];
    const int64_t operand_size =
        operand_idx < 0 ? 1 : operand_shape[operand_idx];
    const int64_t grad_size = grad_shape[i];
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by matmul_test.
        (target_size == 1 || target_size == grad_size) &&
            (operand_size == 1 || operand_size == grad_size) &&
            std::max(target_size, operand_size) == grad_size,
        error::kInvalidArgument)
        << "backward batch dimensions are not compatible at axis " << i
        << ": grad=" << grad_size << ", target=" << target_size
        << ", operand=" << operand_size;
    const BackwardBatchAxisKind kind =
        ClassifyBackwardBatchAxis(grad_size, target_size, operand_size);
    if (kind == BackwardBatchAxisKind::kTrivial) {
      continue;
    }
    if (!axes.empty() && axes.back().kind == kind) {
      TT_ASSIGN_OR_RETURN(axes.back().size,
                          SafeMultiply(axes.back().size, grad_size));
    } else {
      axes.push_back({kind, grad_size});
    }
  }
  return axes;
}

// Holds the reshaped operand shapes and `dot_general` batch/contracting
// dimension index lists constructed from `CoalescedBackwardAxis`.
struct PreparedBackwardBatchShapes {
  Dimensions grad_target_shape;
  Dimensions operand_target_shape;
  Dimensions grad_batch_dims;
  Dimensions operand_batch_dims;
  Dimensions grad_contract_dims;
  Dimensions operand_contract_dims;
  int64_t num_grad_free = 0;
};

// Populates the coalesced shapes and `dot_general` batching/contracting
// dimension lists for `grad` and `operand`:
//   - `kShared` axes are added to both `grad_batch_dims` and
//     `operand_batch_dims`.
//   - `kContract` axes (broadcasted for `target` in forward) are added to both
//     `grad_contract_dims` and `operand_contract_dims` for in-place MXU
//     reduction.
//   - `kGradFree` axes (broadcasted for `operand` in forward) are kept only on
//     `grad_target_shape` as free dimensions.
PreparedBackwardBatchShapes PrepareBackwardBatchShapes(
    absl::Span<const CoalescedBackwardAxis> axes,
    llvm::ArrayRef<int64_t> grad_shape, llvm::ArrayRef<int64_t> operand_shape) {
  PreparedBackwardBatchShapes shapes;
  shapes.grad_target_shape.reserve(axes.size() + 2);
  for (const CoalescedBackwardAxis& axis : axes) {
    if (axis.kind == BackwardBatchAxisKind::kShared) {
      shapes.grad_batch_dims.push_back(shapes.grad_target_shape.size());
      shapes.grad_target_shape.push_back(axis.size);
      shapes.operand_batch_dims.push_back(shapes.operand_target_shape.size());
      shapes.operand_target_shape.push_back(axis.size);
    } else if (axis.kind == BackwardBatchAxisKind::kContract) {
      shapes.grad_contract_dims.push_back(shapes.grad_target_shape.size());
      shapes.grad_target_shape.push_back(axis.size);
      shapes.operand_contract_dims.push_back(
          shapes.operand_target_shape.size());
      shapes.operand_target_shape.push_back(axis.size);
    } else {
      shapes.grad_target_shape.push_back(axis.size);
      ++shapes.num_grad_free;
    }
  }
  const int64_t grad_rank = grad_shape.size();
  const int64_t operand_rank = operand_shape.size();
  shapes.grad_target_shape.push_back(grad_shape[grad_rank - 2]);
  shapes.grad_target_shape.push_back(grad_shape[grad_rank - 1]);
  shapes.operand_target_shape.push_back(operand_shape[operand_rank - 2]);
  shapes.operand_target_shape.push_back(operand_shape[operand_rank - 1]);
  return shapes;
}

// Computes the transpose permutation that restores the canonical batch axis
// order `[coalesced_target_batch..., R, C]` after backward `dot_general`:
//   - When `grad_is_lhs == true` (`GradSelf`: `dot_general(grad, conj_other)`),
//     `dot_general` outputs `[kShared..., kGradFree..., M, K]`.
//   - When `grad_is_lhs == false` (`GradOther`: `dot_general(conj_self,
//     grad)`), `dot_general` outputs `[kShared..., K, kGradFree..., N]`.
Dimensions BuildBackwardOutputPermutation(
    absl::Span<const CoalescedBackwardAxis> axes, const int64_t num_shared,
    const int64_t num_grad_free, const bool grad_is_lhs) {
  Dimensions perm;
  perm.reserve(num_shared + num_grad_free + 2);
  int64_t shared_idx = 0;
  int64_t grad_free_idx = grad_is_lhs ? num_shared : num_shared + 1;
  const int64_t first_mat_idx =
      grad_is_lhs ? (num_shared + num_grad_free) : num_shared;
  const int64_t second_mat_idx = num_shared + num_grad_free + 1;
  for (const CoalescedBackwardAxis& axis : axes) {
    if (axis.kind == BackwardBatchAxisKind::kShared) {
      perm.push_back(shared_idx++);
    } else if (axis.kind == BackwardBatchAxisKind::kGradFree) {
      perm.push_back(grad_free_idx++);
    }
  }
  perm.push_back(first_mat_idx);
  perm.push_back(second_mat_idx);
  return perm;
}

// Shared helper for `EmitStaticBatchedGradSelf` (`grad_is_lhs == true`,
// `matrix_contract_offset == 1`) and `EmitStaticBatchedGradOther`
// (`grad_is_lhs == false`, `matrix_contract_offset == 2`) that coalesces batch
// axes, fuses any `target`-broadcasted batch axes (`kContract`) with the matrix
// contracting axis into a single `dot_general`, permutes free/shared batch
// axes, and reshapes to `target_static_dims`.
absl::StatusOr<mlir::MlirOp> EmitCoalescedBackwardDotGeneral(
    mlir::MlirOp grad, mlir::MlirOp conj_operand,
    absl::Span<const int64_t> target_static_dims, const bool grad_is_lhs,
    const int64_t matrix_contract_offset, const MatmulPrecisionPair precision) {
  const llvm::ArrayRef<int64_t> grad_shape =
      GetTensorTypeOrDie(grad).getShape();
  const llvm::ArrayRef<int64_t> operand_shape =
      GetTensorTypeOrDie(conj_operand).getShape();
  const llvm::ArrayRef<int64_t> target_shape_ref =
      AsArrayRef(target_static_dims);
  if (grad_shape.size() == target_shape_ref.size() &&
      grad_shape.size() == operand_shape.size() &&
      grad_shape.drop_back(2) == target_shape_ref.drop_back(2) &&
      grad_shape.drop_back(2) == operand_shape.drop_back(2)) {
    const int64_t rank = grad_shape.size();
    const Dimensions batch_dims = MakeIotaDims(rank - 2);
    const int64_t contract_dim = rank - matrix_contract_offset;
    const mlir::MlirOp lhs = grad_is_lhs ? grad : conj_operand;
    const mlir::MlirOp rhs = grad_is_lhs ? conj_operand : grad;
    return BuildTypedDotGeneral(lhs, rhs, batch_dims, batch_dims,
                                /*lhs_contracting=*/{contract_dim},
                                /*rhs_contracting=*/{contract_dim}, precision);
  }
  TT_ASSIGN_OR_RETURN(
      const llvm::SmallVector<CoalescedBackwardAxis> axes,
      CoalesceBackwardBatchAxes(grad_shape, target_static_dims, operand_shape));
  PreparedBackwardBatchShapes shapes =
      PrepareBackwardBatchShapes(axes, grad_shape, operand_shape);
  shapes.grad_contract_dims.push_back(
      static_cast<int64_t>(shapes.grad_target_shape.size()) -
      matrix_contract_offset);
  shapes.operand_contract_dims.push_back(
      static_cast<int64_t>(shapes.operand_target_shape.size()) -
      matrix_contract_offset);
  grad = ReshapeIfChanged(grad, shapes.grad_target_shape);
  conj_operand = ReshapeIfChanged(conj_operand, shapes.operand_target_shape);
  const mlir::MlirOp lhs = grad_is_lhs ? grad : conj_operand;
  const mlir::MlirOp rhs = grad_is_lhs ? conj_operand : grad;
  const Dimensions& lhs_batch =
      grad_is_lhs ? shapes.grad_batch_dims : shapes.operand_batch_dims;
  const Dimensions& rhs_batch =
      grad_is_lhs ? shapes.operand_batch_dims : shapes.grad_batch_dims;
  const Dimensions& lhs_contract =
      grad_is_lhs ? shapes.grad_contract_dims : shapes.operand_contract_dims;
  const Dimensions& rhs_contract =
      grad_is_lhs ? shapes.operand_contract_dims : shapes.grad_contract_dims;
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp result,
      BuildTypedDotGeneral(lhs, rhs, lhs_batch, rhs_batch, lhs_contract,
                           rhs_contract, precision));
  const Dimensions perm = BuildBackwardOutputPermutation(
      axes, shapes.grad_batch_dims.size(), shapes.num_grad_free, grad_is_lhs);
  result = TransposeIfNeeded(result, perm);
  return ReshapeIfChanged(result, target_static_dims);
}

// Lowers static `2D x ND` and general static `ND x ND` `grad_self`
// (`[..., M, N] x [..., K, N]^H -> [..., M, K]`).
absl::StatusOr<mlir::MlirOp> EmitStaticBatchedGradSelf(
    mlir::MlirOp grad, mlir::MlirOp conj_other,
    absl::Span<const int64_t> self_static_dims,
    const MatmulPrecisionPair precision) {
  return EmitCoalescedBackwardDotGeneral(
      grad, conj_other, self_static_dims, /*grad_is_lhs=*/true,
      /*matrix_contract_offset=*/1, precision);
}

// Lowers static `ND x ND` `grad_other` (`[..., M, K]^H x [..., M, N] -> [...,
// K, N]`).
absl::StatusOr<mlir::MlirOp> EmitStaticBatchedGradOther(
    mlir::MlirOp grad, mlir::MlirOp conj_self,
    absl::Span<const int64_t> other_static_dims,
    const MatmulPrecisionPair precision) {
  return EmitCoalescedBackwardDotGeneral(
      grad, conj_self, other_static_dims, /*grad_is_lhs=*/false,
      /*matrix_contract_offset=*/2, precision);
}

// Identifies which leading batch dimensions in `grad_dims` (`[bcast_batch...,
// R, C]`) were broadcasted relative to `target_shape` (`[target_batch..., R,
// C]`):
//   - `reduce_dims`: batch axes that are not static size-1 in `grad_dims` where
//     `target` either had a lower batch rank (`target_idx < 0`) or a singleton
//     dimension (`target_shape[target_idx] == 1`).
//   - `squeeze_dims`: leading batch axes (`target_idx < 0`) that were prepended
//     during forward broadcasting and must be dropped to restore `target`'s
//     rank.
void CollectBackwardReductionDims(const stablehlo::Dimensions& grad_dims,
                                  llvm::ArrayRef<int64_t> target_shape,
                                  Dimensions& reduce_dims,
                                  Dimensions& squeeze_dims) {
  const int64_t bcast_batch_rank = static_cast<int64_t>(grad_dims.size()) - 2;
  const int64_t target_batch_rank =
      static_cast<int64_t>(target_shape.size()) - 2;
  const int64_t rank_diff = bcast_batch_rank - target_batch_rank;
  for (int64_t i = 0; i < bcast_batch_rank; ++i) {
    const int64_t target_idx = i - rank_diff;
    const bool grad_is_static_one =
        !grad_dims[i].boundOp.has_value() && grad_dims[i].size == 1;
    if (target_idx < 0) {
      if (!grad_is_static_one) {
        reduce_dims.push_back(i);
      }
      squeeze_dims.push_back(i);
    } else if (target_shape[target_idx] == 1 && !grad_is_static_one) {
      reduce_dims.push_back(i);
    }
  }
}

// Reduces `op` across `reduce_dims` with `kKeepDims` and squeezes
// `squeeze_dims`.
absl::StatusOr<mlir::MlirOp> ReduceAndSqueezeRealBatchDims(
    mlir::MlirOp op, const Dimensions& reduce_dims,
    const Dimensions& squeeze_dims) {
  if (!reduce_dims.empty()) {
    TT_ASSIGN_OR_RETURN(
        op, BuildSumShlo(op, reduce_dims, ReductionMode::kKeepDims));
  }
  if (!squeeze_dims.empty()) {
    TT_ASSIGN_OR_RETURN(op, Squeeze(op, squeeze_dims));
  }
  return op;
}

// Sum-reduces and squeezes broadcasted batch dimensions from an unreduced
// backward `dot_general` output `grad_bcast` so its shape matches
// `target_static_dims` (used when bounded dynamic dimensions or `complex<f64>`
// prevent fusing the batch reduction into `dot_general`).
//
// For `complex<f64>` when kept size-1 dimensions are present (`reduce_dims !=
// squeeze_dims`), reduces the real and imaginary `f64` components separately
// before recombining via `stablehlo.complex`. This avoids emitting any
// `stablehlo.reshape` on `complex<f64>` tensors (which would trigger
// `TpuLayoutAssignment`'s 128-bit `X128Combine` layout `kCopy` bug) while
// supporting both static and bounded-dynamic batch dimensions in O(1) HLO ops.
absl::StatusOr<mlir::MlirOp> ReduceBroadcastedBatchDims(
    mlir::MlirOp grad_bcast, absl::Span<const int64_t> target_static_dims) {
  const mlir::RankedTensorType grad_type = GetTensorTypeOrDie(grad_bcast);
  const stablehlo::Dimensions grad_dims = GetDimensions(grad_bcast);
  const llvm::ArrayRef<int64_t> target_shape = AsArrayRef(target_static_dims);
  Dimensions reduce_dims;
  Dimensions squeeze_dims;
  CollectBackwardReductionDims(grad_dims, target_shape, reduce_dims,
                               squeeze_dims);
  if (reduce_dims.empty() && squeeze_dims.empty()) {
    return grad_bcast;
  }
  if (reduce_dims == squeeze_dims) {
    return BuildSumShlo(grad_bcast, reduce_dims, ReductionMode::kDropDims);
  }
  if (IsComplexF64(grad_type.getElementType())) {
    auto complex_op =
        grad_bcast.getValue().getDefiningOp<stablehlo::ComplexOp>();
    mlir::MlirOp real_part =
        complex_op ? mlir::MlirOp(grad_bcast.getBuilder(), complex_op.getLhs())
                   : stablehlo::Real(grad_bcast);
    mlir::MlirOp imag_part =
        complex_op ? mlir::MlirOp(grad_bcast.getBuilder(), complex_op.getRhs())
                   : stablehlo::Imag(grad_bcast);
    if (complex_op && complex_op.getResult().use_empty()) {
      complex_op.erase();
    }
    TT_ASSIGN_OR_RETURN(real_part, ReduceAndSqueezeRealBatchDims(
                                       real_part, reduce_dims, squeeze_dims));
    TT_ASSIGN_OR_RETURN(imag_part, ReduceAndSqueezeRealBatchDims(
                                       imag_part, reduce_dims, squeeze_dims));
    return stablehlo::Complex(real_part, imag_part);
  }
  return ReduceAndSqueezeRealBatchDims(grad_bcast, reduce_dims, squeeze_dims);
}

// Broadcast-aligns the leading batch prefixes of `lhs` and `rhs`, emits a
// batched `dot_general` with `matrix_contract_offset` (`1` for `GradSelf`, `0`
// for `GradOther` relative to `batch_rank`), and sum-reduces/squeezes any batch
// dimensions that were broadcasted for `target`.
absl::StatusOr<mlir::MlirOp> EmitBatchedGradWithReduction(
    mlir::MlirOp lhs, mlir::MlirOp rhs,
    absl::Span<const int64_t> target_static_dims,
    const int64_t matrix_contract_offset, const MatmulPrecisionPair precision) {
  const stablehlo::Dimensions lhs_dims = GetDimensions(lhs);
  const stablehlo::Dimensions rhs_dims = GetDimensions(rhs);
  TT_ASSIGN_OR_RETURN(const stablehlo::Dimensions batch_dims,
                      ComputeBroadcastBatchDims(lhs_dims, rhs_dims));
  TT_ASSIGN_OR_RETURN(lhs, BroadcastMatrixBatchDims(lhs, batch_dims));
  TT_ASSIGN_OR_RETURN(rhs, BroadcastMatrixBatchDims(rhs, batch_dims));
  const int64_t batch_rank = batch_dims.size();
  const int64_t matrix_contract_dim = batch_rank + matrix_contract_offset;
  const Dimensions batch_indices = MakeIotaDims(batch_rank);
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp grad_bcast,
      BuildTypedDotGeneral(lhs, rhs, batch_indices, batch_indices,
                           /*lhs_contracting=*/{matrix_contract_dim},
                           /*rhs_contracting=*/{matrix_contract_dim},
                           precision));
  return ReduceBroadcastedBatchDims(grad_bcast, target_static_dims);
}

// Decomposes a `complex<f64>` `dot_general` into four real `f64`
// `dot_general` operations:
//   `(a + bi) @ (c + di) = (a @ c - b @ d) + (a @ d + b @ c)i`
// This avoids XLA:TPU's `C128` `ComplexRewriter` + `TpuBroadcastRewriter` bugs
// on rank-1 / multi-axis contractions (`ND x 1D` and `1D x ND`).
absl::StatusOr<mlir::MlirOp> EmitComplexF64DotGeneral(
    mlir::MlirOp lhs, mlir::MlirOp rhs, absl::Span<const int64_t> lhs_batching,
    absl::Span<const int64_t> rhs_batching,
    absl::Span<const int64_t> lhs_contracting,
    absl::Span<const int64_t> rhs_contracting,
    const MatmulPrecisionPair precision,
    const std::optional<mlir::ElementType> preferred_element_type) {
  if (!IsComplexF64(GetElementTypeOrSelf(lhs))) {
    TT_ASSIGN_OR_RETURN(
        lhs, ComplexSafeConvertElementType(lhs, mlir::ElementType::COMPLEXF64));
  }
  if (!IsComplexF64(GetElementTypeOrSelf(rhs))) {
    TT_ASSIGN_OR_RETURN(
        rhs, ComplexSafeConvertElementType(rhs, mlir::ElementType::COMPLEXF64));
  }
  mlir::MlirOp lhs_real = stablehlo::Real(lhs);
  mlir::MlirOp lhs_imag = stablehlo::Imag(lhs);
  mlir::MlirOp rhs_real = stablehlo::Real(rhs);
  mlir::MlirOp rhs_imag = stablehlo::Imag(rhs);
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp ac,
      BuildTypedDotGeneral(lhs_real, rhs_real, lhs_batching, rhs_batching,
                           lhs_contracting, rhs_contracting, precision));
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp bd,
      BuildTypedDotGeneral(lhs_imag, rhs_imag, lhs_batching, rhs_batching,
                           lhs_contracting, rhs_contracting, precision));
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp ad,
      BuildTypedDotGeneral(lhs_real, rhs_imag, lhs_batching, rhs_batching,
                           lhs_contracting, rhs_contracting, precision));
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp bc,
      BuildTypedDotGeneral(lhs_imag, rhs_real, lhs_batching, rhs_batching,
                           lhs_contracting, rhs_contracting, precision));
  mlir::MlirOp real_part = stablehlo::Subtract(ac, bd);
  mlir::MlirOp imag_part = stablehlo::Add(ad, bc);
  const std::optional<mlir::ElementType> resolved_preferred =
      ResolvePreferredElementType(GetElementTypeOrSelf(lhs),
                                  preferred_element_type);
  if (resolved_preferred == mlir::ElementType::COMPLEXF32) {
    real_part =
        stablehlo::ConvertElementType(real_part, mlir::ElementType::F32);
    imag_part =
        stablehlo::ConvertElementType(imag_part, mlir::ElementType::F32);
  }
  mlir::MlirOp result = stablehlo::Complex(real_part, imag_part);
  if (resolved_preferred.has_value() &&
      *resolved_preferred != mlir::ElementType::COMPLEXF64 &&
      *resolved_preferred != mlir::ElementType::COMPLEXF32) {
    TT_ASSIGN_OR_RETURN(
        result, ComplexSafeConvertElementType(result, *resolved_preferred));
  }
  return result;
}

// Validates that backward inputs (`grad`, `self`, `other`) have valid ranks
// (`self_rank >= 1`, `other_rank >= 1`, `grad_rank` matching the forward output
// rank) and matching element types.
absl::Status ValidateBackwardRanksAndTypes(
    mlir::MlirOp grad, const int64_t self_rank, const int64_t other_rank,
    mlir::MlirOp paired_operand, std::optional<mlir::MlirOp> target_op) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by matmul_test.
      self_rank >= 1 && other_rank >= 1, error::kInvalidArgument)
      << "both arguments to matmul backward need to be at least 1D, but they "
         "are "
      << self_rank << "D and " << other_rank << "D";
  const mlir::RankedTensorType grad_type = GetTensorTypeOrDie(grad);
  const mlir::RankedTensorType operand_type =
      GetTensorTypeOrDie(paired_operand);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by matmul_test.
      grad_type.getElementType() == operand_type.getElementType() &&
          (!target_op.has_value() ||
           GetElementTypeOrSelf(*target_op) == grad_type.getElementType()),
      error::kInvalidArgument)
      << "expected grad and matmul operands to have the same element type";
  const int64_t expected_grad_rank =
      (self_rank == 1 && other_rank == 1)
          ? 0
          : ((self_rank == 1 || other_rank == 1)
                 ? std::max(self_rank, other_rank) - 1
                 : std::max(self_rank, other_rank));
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by matmul_test.
      grad_type.getRank() == expected_grad_rank, error::kInvalidArgument)
      << "expected grad rank " << expected_grad_rank << " for " << self_rank
      << "D x " << other_rank << "D matmul backward, got "
      << grad_type.getRank();
  return absl::OkStatus();
}

// Emits the gradient tensor for `target` (`self` or `other`) when any operand
// has a zero-sized dimension:
//   - If `target_static_dims` itself contains `0`, returns an empty zero-sized
//     tensor of shape `target_static_dims` (bounding any dynamic dimensions on
//     `*target_op` via `ApplyDimensionBounds`).
//   - Otherwise (`target` has non-zero numel, but the other operand or `grad`
//     was empty), returns a zero-filled tensor matching `target` (preserving
//     dynamic bounds via `MakeConstantLike(*target_op, 0)` when `target_op` is
//     provided).
absl::StatusOr<mlir::MlirOp> EmitZeroGradForTarget(
    mlir::MlirOp grad, absl::Span<const int64_t> target_static_dims,
    std::optional<mlir::MlirOp> target_op) {
  const mlir::Type element_type = GetElementTypeOrSelf(grad);
  if (llvm::is_contained(target_static_dims, 0)) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp zero_sized,
                        MakeZeroSizedTensor(grad.getBuilder(), element_type,
                                            AsArrayRef(target_static_dims)));
    if (target_op.has_value()) {
      zero_sized = ApplyDimensionBounds(zero_sized, GetDimensions(*target_op));
    }
    return zero_sized;
  }
  if (target_op.has_value()) {
    return MakeConstantLike(*target_op, 0);
  }
  return MakeConstant(grad.getBuilder(), 0, element_type,
                      AsArrayRef(target_static_dims));
}

// Lowers `grad_self` when `self` is 1D or `other` is 1D (`conj_other` is
// already physically conjugated):
//   1. `1D x 1D` (`self: [K], other: [K], grad: []`):
//      Broadcasts scalar `grad` `[] -> [K]` and multiplies element-wise with
//      `conj_other` `[K] -> [K]`.
//   2. `1D x ND` (`self: [K], other: [..., K, N], grad: [..., N]`):
//      Contracts `grad` `[..., N]` with `conj_other` `[..., K, N]` across all
//      batch dimensions `0 .. other_rank - 3` AND column dimension `N`
//      (`other_rank - 1`) in a single `dot_general`, leaving `K`
//      (`other_rank - 2`) as the sole free dimension `[K]`.
//   3. `ND x 1D` (`self: [..., M, K], other: [K], grad: [..., M]`):
//      Unsqueezes `grad` `[..., M] -> [..., M, 1]`, broadcasts with
//      `conj_other` `[K]`, and computes the outer product `[..., M, 1] * [K] ->
//      [..., M, K]`.
absl::StatusOr<mlir::MlirOp> EmitGradSelfRank1OrOtherRank1(
    mlir::MlirOp grad, absl::Span<const int64_t> self_static_dims,
    mlir::MlirOp conj_other, std::optional<mlir::MlirOp> self_op,
    const MatmulPrecisionPair precision) {
  const int64_t self_rank = self_static_dims.size();
  const int64_t other_rank = GetTensorTypeOrDie(conj_other).getRank();
  const int64_t grad_rank = GetTensorTypeOrDie(grad).getRank();
  if (self_rank == 1 && other_rank == 1) {
    mlir::MlirOp target = self_op.value_or(conj_other);
    TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_bcast,
                        BroadcastIfNeeded(grad, target));
    return stablehlo::Mul(grad_bcast, conj_other);
  }
  if (self_rank == 1) {
    const Dimensions grad_contract = GetAllDimensions(grad);
    Dimensions other_contract = MakeIotaDims(other_rank - 2);
    other_contract.push_back(other_rank - 1);
    return BuildTypedDotGeneral(grad, conj_other, /*lhs_batching=*/{},
                                /*rhs_batching=*/{}, grad_contract,
                                other_contract, precision);
  }
  TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_unsqueezed, Unsqueeze(grad, grad_rank));
  TT_ASSIGN_OR_RETURN((auto [grad_bcast, other_bcast]),
                      ApplyBroadcastIfNeeded(grad_unsqueezed, conj_other));
  return stablehlo::Mul(grad_bcast, other_bcast);
}

// Lowers `grad_other` when `self` is 1D or `other` is 1D (`conj_self` is
// already physically conjugated):
//   1. `1D x 1D` (`self: [K], other: [K], grad: []`):
//      Broadcasts scalar `grad` `[] -> [K]` and multiplies element-wise with
//      `conj_self` `[K] -> [K]`.
//   2. `1D x ND` (`self: [K], other: [..., K, N], grad: [..., N]`):
//      Unsqueezes `conj_self` `[K] -> [K, 1]` and `grad` `[..., N] -> [..., 1,
//      N]`, then broadcasts and multiplies element-wise to form the batched
//      outer product `[..., K, N]`.
//   3. `ND x 1D` (`self: [..., M, K], other: [K], grad: [..., M]`):
//      Contracts `conj_self` `[..., M, K]` with `grad` `[..., M]` over all
//      leading dimensions `[..., M]` to produce `[K]` (flattening `[..., M]` to
//      `leading_rows` first when `fold_to_2d` is true and `self_rank >= 3`).
absl::StatusOr<mlir::MlirOp> EmitGradOtherRank1OrSelfRank1(
    mlir::MlirOp grad, mlir::MlirOp conj_self,
    absl::Span<const int64_t> other_static_dims,
    std::optional<mlir::MlirOp> other_op, const bool fold_to_2d,
    const MatmulPrecisionPair precision) {
  const llvm::ArrayRef<int64_t> self_shape =
      GetTensorTypeOrDie(conj_self).getShape();
  const int64_t self_rank = self_shape.size();
  const int64_t other_rank = other_static_dims.size();
  const int64_t grad_rank = GetTensorTypeOrDie(grad).getRank();
  if (self_rank == 1 && other_rank == 1) {
    mlir::MlirOp target = other_op.value_or(conj_self);
    TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_bcast,
                        BroadcastIfNeeded(grad, target));
    return stablehlo::Mul(conj_self, grad_bcast);
  }
  if (self_rank == 1) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp self_unsqueezed, Unsqueeze(conj_self, 1));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_unsqueezed,
                        Unsqueeze(grad, grad_rank - 1));
    TT_ASSIGN_OR_RETURN(
        (auto [self_bcast, grad_bcast]),
        ApplyBroadcastIfNeeded(self_unsqueezed, grad_unsqueezed));
    return stablehlo::Mul(self_bcast, grad_bcast);
  }
  if (fold_to_2d && self_rank >= 3) {
    TT_ASSIGN_OR_RETURN(
        const int64_t leading_rows,
        NumElements(absl::MakeConstSpan(self_shape.drop_back(1))));
    mlir::MlirOp self_2d = ReshapeIfChanged(
        conj_self, Dimensions{leading_rows, self_shape.back()});
    mlir::MlirOp grad_1d = ReshapeIfChanged(grad, {leading_rows});
    return BuildTypedDotGeneral(self_2d, grad_1d, /*lhs_batching=*/{},
                                /*rhs_batching=*/{}, /*lhs_contracting=*/{0},
                                /*rhs_contracting=*/{0}, precision);
  }
  const Dimensions contract_dims = GetAllDimensions(grad);
  return BuildTypedDotGeneral(conj_self, grad, /*lhs_batching=*/{},
                              /*rhs_batching=*/{}, contract_dims, contract_dims,
                              precision);
}

// Lowers `grad_self` for `2D x 2D` (`[M, N] x [K, N]^H -> [M, K]`) and
// `ND x 2D` (`[..., M, N] x [K, N]^H -> [..., M, K]`, emitted as an unbatched
// `dot_general` with `grad`'s batch dimensions as free dimensions so XLA does
// not misclassify `Reshape + Dot(rhs_contract=1) + Reshape` as a 3D windowed
// convolution).
absl::StatusOr<mlir::MlirOp> EmitGradSelfWithRank2Other(
    mlir::MlirOp grad, mlir::MlirOp conj_other,
    const MatmulPrecisionPair precision) {
  const int64_t grad_rank = GetTensorTypeOrDie(grad).getRank();
  return BuildTypedDotGeneral(grad, conj_other, /*lhs_batching=*/{},
                              /*rhs_batching=*/{},
                              /*lhs_contracting=*/{grad_rank - 1},
                              /*rhs_contracting=*/{1}, precision);
}

// Lowers `grad_other` for `2D x 2D` (`[M, K]^H x [M, N] -> [K, N]`) and
// `ND x 2D` (`[..., M, K]^H x [..., M, N] -> [K, N]`, flattening `[..., M]` to
// `leading_rows` when `fold_to_2d` is true, or contracting all `[..., M]`
// dimensions directly in `dot_general`).
absl::StatusOr<mlir::MlirOp> EmitGradOtherWithRank2Other(
    mlir::MlirOp grad, mlir::MlirOp conj_self, const bool fold_to_2d,
    const MatmulPrecisionPair precision) {
  const llvm::ArrayRef<int64_t> self_shape =
      GetTensorTypeOrDie(conj_self).getShape();
  if (fold_to_2d && self_shape.size() >= 3) {
    const llvm::ArrayRef<int64_t> grad_shape =
        GetTensorTypeOrDie(grad).getShape();
    TT_ASSIGN_OR_RETURN(
        const int64_t leading_rows,
        NumElements(absl::MakeConstSpan(self_shape.drop_back(1))));
    mlir::MlirOp self_2d = ReshapeIfChanged(
        conj_self, Dimensions{leading_rows, self_shape.back()});
    mlir::MlirOp grad_2d =
        ReshapeIfChanged(grad, Dimensions{leading_rows, grad_shape.back()});
    return BuildTypedDotGeneral(self_2d, grad_2d, /*lhs_batching=*/{},
                                /*rhs_batching=*/{}, /*lhs_contracting=*/{0},
                                /*rhs_contracting=*/{0}, precision);
  }
  const Dimensions contract_dims =
      MakeIotaDims(static_cast<int64_t>(self_shape.size()) - 1);
  return BuildTypedDotGeneral(conj_self, grad, /*lhs_batching=*/{},
                              /*rhs_batching=*/{}, contract_dims, contract_dims,
                              precision);
}

}  // namespace

mlir::ArrayAttr MakeDotPrecisionConfig(mlir::MLIRContext& ctx,
                                       const MatmulPrecisionPair precision) {
  return stablehlo::PrecisionConfigAttr::get(&ctx,
                                             {precision.lhs, precision.rhs});
}

absl::StatusOr<mlir::MlirOp> BuildTypedDotGeneral(
    mlir::MlirOp lhs, mlir::MlirOp rhs, absl::Span<const int64_t> lhs_batching,
    absl::Span<const int64_t> rhs_batching,
    absl::Span<const int64_t> lhs_contracting,
    absl::Span<const int64_t> rhs_contracting,
    const MatmulPrecisionPair precision,
    const std::optional<mlir::ElementType> preferred_element_type) {
  TT_RETURN_IF_ERROR(ValidateAndReconcileDotDimensions(
      lhs, rhs, lhs_batching, rhs_batching, lhs_contracting, rhs_contracting));
  const mlir::RankedTensorType lhs_type = GetTensorTypeOrDie(lhs);
  const mlir::RankedTensorType rhs_type = GetTensorTypeOrDie(rhs);
  if (IsComplexF64(lhs_type.getElementType()) ||
      IsComplexF64(rhs_type.getElementType())) {
    return EmitComplexF64DotGeneral(lhs, rhs, lhs_batching, rhs_batching,
                                    lhs_contracting, rhs_contracting, precision,
                                    preferred_element_type);
  }
  const bool is_i64 = lhs_type.getElementType().isInteger(64) ||
                      rhs_type.getElementType().isInteger(64);
  if (is_i64) {
    // XLA/TPU MXU `dot_general` does not natively support 64-bit integer
    // contractions, so we emulate `i64` contractions via `f64` with
    // `Precision::HIGHEST` (full IEEE-754 53-bit mantissa accuracy instead of
    // BF16-pass truncation) and convert the result back to `i64` (or
    // `preferred_element_type` if specified).
    lhs = stablehlo::ConvertElementType(lhs, mlir::ElementType::F64);
    rhs = stablehlo::ConvertElementType(rhs, mlir::ElementType::F64);
  }
  const MatmulPrecisionPair effective_precision =
      is_i64 ? MatmulPrecisionPair{stablehlo::Precision::
                                       HIGHEST,  // EXPLICIT_PRECISION_OK=i64
                                   stablehlo::Precision::
                                       HIGHEST  // EXPLICIT_PRECISION_OK=i64
      }
             : precision;
  const int64_t out_rank = lhs_type.getRank() + rhs_type.getRank() -
                           static_cast<int64_t>(lhs_batching.size()) -
                           2 * static_cast<int64_t>(lhs_contracting.size());
  const bool squeeze_scalar_out =
      out_rank == 0 &&
      (lhs_type.getNumDynamicDims() > 0 || rhs_type.getNumDynamicDims() > 0);
  if (squeeze_scalar_out) {
    // Work around StableHLO inferDotGeneralOp bug where a 0-D output from
    // bounded dynamic inputs retains the input's non-empty bounds attribute.
    TT_ASSIGN_OR_RETURN(lhs, Unsqueeze(lhs, -1));
  }
  mlir::MLIRContext& ctx = lhs.getContext();
  const auto dot_dimension_numbers = stablehlo::DotDimensionNumbersAttr::get(
      &ctx, AsArrayRef(lhs_batching), AsArrayRef(rhs_batching),
      AsArrayRef(lhs_contracting), AsArrayRef(rhs_contracting));
  const mlir::ArrayAttr precision_config =
      MakeDotPrecisionConfig(ctx, effective_precision);
  const std::optional<mlir::ElementType> resolved_preferred =
      is_i64 ? std::nullopt
             : ResolvePreferredElementType(lhs_type.getElementType(),
                                           preferred_element_type);
  mlir::MlirOp result = stablehlo::DotGeneral(
      lhs, rhs, dot_dimension_numbers, precision_config, resolved_preferred);
  if (squeeze_scalar_out) {
    result = stablehlo::Reshape(result, {});
  }
  if (is_i64) {
    result = stablehlo::ConvertElementType(
        result, preferred_element_type.value_or(mlir::ElementType::I64));
  }
  return result;
}

absl::StatusOr<mlir::MlirOp> BuildMatmulShlo(
    mlir::MlirOp lhs, mlir::MlirOp rhs, const MatmulPrecisionPair precision,
    const MatmulShloOptions& options) {
  const mlir::RankedTensorType init_lhs_type = GetTensorTypeOrDie(lhs);
  const mlir::RankedTensorType init_rhs_type = GetTensorTypeOrDie(rhs);
  const int64_t lhs_rank = init_lhs_type.getRank();
  const int64_t rhs_rank = init_rhs_type.getRank();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by matmul_test.
      lhs_rank >= 1 && rhs_rank >= 1, error::kInvalidArgument)
      << "both arguments to matmul need to be at least 1D, but they are "
      << lhs_rank << "D and " << rhs_rank << "D";

  // Apply optional physical complex conjugation (`BuildConjPhysicalShlo`) and
  // computation dtype conversion before dispatching by rank.
  TT_ASSIGN_OR_RETURN(lhs, PrepareMatmulOperand(lhs, options.conjugate_lhs,
                                                options.target_element_type));
  TT_ASSIGN_OR_RETURN(rhs, PrepareMatmulOperand(rhs, options.conjugate_rhs,
                                                options.target_element_type));

  const mlir::RankedTensorType lhs_type = GetTensorTypeOrDie(lhs);
  const mlir::RankedTensorType rhs_type = GetTensorTypeOrDie(rhs);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by matmul_test.
      lhs_type.getElementType() == rhs_type.getElementType(),
      error::kInvalidArgument)
      << "expected both arguments to have the same element type";

  const stablehlo::Dimensions lhs_dims = GetDimensions(lhs);
  const stablehlo::Dimensions rhs_dims = GetDimensions(rhs);
  const int64_t lhs_contract =
      GetLhsContractingDim(lhs_rank, options.transpose_lhs);
  const int64_t rhs_contract =
      GetRhsContractingDim(rhs_rank, options.transpose_rhs);
  if (!lhs_dims[lhs_contract].boundOp.has_value() &&
      !rhs_dims[rhs_contract].boundOp.has_value()) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by matmul_test.
        lhs_dims[lhs_contract].size == rhs_dims[rhs_contract].size,
        error::kInvalidArgument)
        << "contracting dimensions do not match: "
        << lhs_dims[lhs_contract].size << " vs " << rhs_dims[rhs_contract].size;
  }

  const Dimensions lhs_static_shape = GetStaticShape(lhs_dims);
  const Dimensions rhs_static_shape = GetStaticShape(rhs_dims);
  if (llvm::is_contained(lhs_static_shape, 0) ||
      llvm::is_contained(rhs_static_shape, 0)) {
    return EmitZeroElementMatmul(lhs, lhs_dims, rhs_dims, options);
  }
  const bool has_dynamic_dims =
      lhs_type.getNumDynamicDims() > 0 || rhs_type.getNumDynamicDims() > 0;
  const bool use_broadcast_batched_path =
      has_dynamic_dims || IsComplexF64(lhs_type.getElementType());

  // Lower all 5 `torch.matmul` rank combinations:
  //   1. `1D x 1D` (`[K] x [K] -> []`): Inner product.
  //   2. `2D x 2D` (`[M, K] x [K, N] -> [M, N]`): Matrix-matrix product.
  //   3. `ND x 1D` (`[..., M, K] x [K] -> [..., M]`): Batched matrix-vector
  //      product (folded to 2D when `CanFoldLhsTo2D` holds, or lowered via
  //      unbatched `dot_general` with `[..., M]` as free dimensions).
  //   4. `1D x ND` (`[K] x [..., K, N] -> [..., N]`): Vector-batched-matrix
  //      product via unbatched `dot_general`.
  //   5. `ND x ND` (`[..., M, K] x [..., K, N] -> [..., M, N]`):
  //      - `ND x 2D`: Folded to 2D when `CanFoldLhsTo2D` holds, or lowered via
  //        unbatched `dot_general` (with `lhs` batch dims as free dims).
  //      - General static `ND x ND`: Lowered via `EmitStaticBatchedMatmul`.
  //      - Dynamic shapes or 3D+ `complex<f64>`: Lowered via
  //        `EmitBroadcastBatchedMatmul`.
  if (CanFoldLhsTo2D(lhs_rank, rhs_rank, use_broadcast_batched_path, options)) {
    return EmitFoldedLhsMatmul(lhs, rhs, options, precision);
  }
  if (lhs_rank == 1 || rhs_rank == 1 || rhs_rank == 2) {
    return BuildTypedDotGeneral(
        lhs, rhs, /*lhs_batching=*/{}, /*rhs_batching=*/{}, {lhs_contract},
        {rhs_contract}, precision, options.target_element_type);
  }
  if (!use_broadcast_batched_path) {
    return EmitStaticBatchedMatmul(lhs, rhs, options, precision);
  }
  return EmitBroadcastBatchedMatmul(lhs, rhs, lhs_dims, rhs_dims, options,
                                    precision);
}

absl::StatusOr<mlir::MlirOp> BuildMatmulGradSelfShlo(
    mlir::MlirOp grad, absl::Span<const int64_t> self_static_dims,
    mlir::MlirOp other, const MatmulPrecisionPair precision,
    std::optional<mlir::MlirOp> self_op) {
  const mlir::RankedTensorType other_type = GetTensorTypeOrDie(other);
  const mlir::RankedTensorType grad_type = GetTensorTypeOrDie(grad);
  const int64_t self_rank = self_static_dims.size();
  const int64_t other_rank = other_type.getRank();
  TT_RETURN_IF_ERROR(ValidateBackwardRanksAndTypes(grad, self_rank, other_rank,
                                                   other, self_op));
  const Dimensions grad_static_shape = GetStaticShape(grad);
  const Dimensions other_static_shape = GetStaticShape(other);
  if (llvm::is_contained(self_static_dims, 0) ||
      llvm::is_contained(grad_static_shape, 0) ||
      llvm::is_contained(other_static_shape, 0)) {
    return EmitZeroGradForTarget(grad, self_static_dims, self_op);
  }
  const bool has_dynamic_dims =
      grad_type.getNumDynamicDims() > 0 || other_type.getNumDynamicDims() > 0 ||
      (self_op.has_value() &&
       GetTensorTypeOrDie(*self_op).getNumDynamicDims() > 0);
  const bool use_broadcast_batched_path =
      has_dynamic_dims || IsComplexF64(grad_type.getElementType());
  // Under PyTorch's Wirtinger complex derivative convention, for C = A @ B,
  // dL/dA = dL/dC @ B^H = grad @ conj(other)^T.
  TT_ASSIGN_OR_RETURN(mlir::MlirOp conj_other, BuildConjPhysicalShlo(other));

  // Rank cases `1D x 1D`, `1D x ND`, and `ND x 1D`:
  if (self_rank == 1 || other_rank == 1) {
    return EmitGradSelfRank1OrOtherRank1(grad, self_static_dims, conj_other,
                                         self_op, precision);
  }
  // Rank cases `2D x 2D` and `ND x 2D`:
  if (other_rank == 2) {
    return EmitGradSelfWithRank2Other(grad, conj_other, precision);
  }
  // Rank cases `2D x ND` and general static `ND x ND`:
  if (!use_broadcast_batched_path) {
    return EmitStaticBatchedGradSelf(grad, conj_other, self_static_dims,
                                     precision);
  }
  return EmitBatchedGradWithReduction(grad, conj_other, self_static_dims,
                                      /*matrix_contract_offset=*/1, precision);
}

absl::StatusOr<mlir::MlirOp> BuildMatmulGradSelfShlo(
    mlir::MlirOp grad, mlir::MlirOp self, mlir::MlirOp other,
    const MatmulPrecisionPair precision) {
  const Dimensions self_static_dims = GetStaticShape(self);
  return BuildMatmulGradSelfShlo(grad, self_static_dims, other, precision,
                                 self);
}

absl::StatusOr<mlir::MlirOp> BuildMatmulGradOtherShlo(
    mlir::MlirOp grad, mlir::MlirOp self,
    absl::Span<const int64_t> other_static_dims,
    const MatmulPrecisionPair precision, std::optional<mlir::MlirOp> other_op) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  const mlir::RankedTensorType grad_type = GetTensorTypeOrDie(grad);
  const int64_t self_rank = self_type.getRank();
  const int64_t other_rank = other_static_dims.size();
  TT_RETURN_IF_ERROR(ValidateBackwardRanksAndTypes(grad, self_rank, other_rank,
                                                   self, other_op));
  const Dimensions grad_static_shape = GetStaticShape(grad);
  const Dimensions self_static_shape = GetStaticShape(self);
  if (llvm::is_contained(other_static_dims, 0) ||
      llvm::is_contained(grad_static_shape, 0) ||
      llvm::is_contained(self_static_shape, 0)) {
    return EmitZeroGradForTarget(grad, other_static_dims, other_op);
  }
  const bool has_dynamic_dims =
      grad_type.getNumDynamicDims() > 0 || self_type.getNumDynamicDims() > 0 ||
      (other_op.has_value() &&
       GetTensorTypeOrDie(*other_op).getNumDynamicDims() > 0);
  const bool use_broadcast_batched_path =
      has_dynamic_dims || IsComplexF64(grad_type.getElementType());
  // Under PyTorch's Wirtinger complex derivative convention, for C = A @ B,
  // dL/dB = A^H @ dL/dC = conj(self)^T @ grad.
  TT_ASSIGN_OR_RETURN(mlir::MlirOp conj_self, BuildConjPhysicalShlo(self));

  // Rank cases `1D x 1D`, `1D x ND`, and `ND x 1D`:
  if (self_rank == 1 || other_rank == 1) {
    return EmitGradOtherRank1OrSelfRank1(
        grad, conj_self, other_static_dims, other_op,
        /*fold_to_2d=*/!use_broadcast_batched_path, precision);
  }
  // Rank cases `2D x 2D` and `ND x 2D`:
  if (other_rank == 2) {
    return EmitGradOtherWithRank2Other(
        grad, conj_self, /*fold_to_2d=*/!use_broadcast_batched_path, precision);
  }
  // General static `ND x ND` (including `2D x ND`):
  if (!use_broadcast_batched_path) {
    return EmitStaticBatchedGradOther(grad, conj_self, other_static_dims,
                                      precision);
  }
  return EmitBatchedGradWithReduction(conj_self, grad, other_static_dims,
                                      /*matrix_contract_offset=*/0, precision);
}

absl::StatusOr<mlir::MlirOp> BuildMatmulGradOtherShlo(
    mlir::MlirOp grad, mlir::MlirOp self, mlir::MlirOp other,
    const MatmulPrecisionPair precision) {
  const Dimensions other_static_dims = GetStaticShape(other);
  return BuildMatmulGradOtherShlo(grad, self, other_static_dims, precision,
                                  other);
}

}  // namespace torch_tpu
