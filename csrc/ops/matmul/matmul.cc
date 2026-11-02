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
#include <numeric>
#include <optional>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
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
  std::iota(dims.begin(), dims.end(), 0);
  return dims;
}

// Returns the product of static dimension sizes in `dims` (1 if `dims` is
// empty), used when flattening leading batch and row dimensions into a single
// 2D matrix row axis.
int64_t ProductOfDims(llvm::ArrayRef<int64_t> dims) {
  int64_t prod = 1;
  for (const int64_t d : dims) {
    prod *= d;
  }
  return prod;
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
//      non-floating-point operands (e.g., promoting sub-32-bit integers `i8`,
//      `u8`, `i16`, `u16` to `f32` for TPU MXU execution). Floating-point
//      operands retain their input element type so `target_element_type` can be
//      passed as `preferredElementType` on `stablehlo.dot_general`.
absl::StatusOr<mlir::MlirOp> PrepareMatmulOperand(
    mlir::MlirOp op, const ConjugateMode conjugate,
    const std::optional<mlir::ElementType> target_element_type) {
  if (conjugate == ConjugateMode::kConjugate) {
    TT_ASSIGN_OR_RETURN(op, BuildConjPhysicalShlo(op));
  }
  const mlir::Type elem_type = GetTensorTypeOrDie(op).getElementType();
  if (target_element_type.has_value() &&
      GetElementTypeOrDie(op) != *target_element_type &&
      !mlir::isa<mlir::FloatType, mlir::ComplexType>(elem_type) &&
      !elem_type.isInteger(64)) {
    op = stablehlo::ConvertElementType(op, *target_element_type);
  }
  return op;
}

// Checks whether two `stablehlo::Dimensions` arrays have identical static sizes
// and bounded dynamic dimension metadata (`boundOp` and `boundOpDim`).
bool AreDimensionsEqual(const stablehlo::Dimensions& a,
                        const stablehlo::Dimensions& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].size != b[i].size || a[i].boundOp != b[i].boundOp ||
        a[i].boundOpDim != b[i].boundOpDim) {
      return false;
    }
  }
  return true;
}

// Computes the broadcasted `DimensionInfo` at aligned batch axis `i` (out of
// `batch_rank = max(lhs_batch_rank, rhs_batch_rank)`), right-aligning the
// leading batch dimensions `[0 .. rank - 3]` of `lhs` and `rhs` per NumPy
// broadcasting rules and preserving bounded dynamic dimension metadata.
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
  const auto& l_dim = lhs_dims[lhs_idx];
  const auto& r_dim = rhs_dims[rhs_idx];
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Checked before MLIR lowering.
      l_dim.size == r_dim.size || l_dim.size == 1 || r_dim.size == 1,
      error::kInvalidArgument)
      << "batch dimensions are not broadcastable: " << l_dim.size << " vs "
      << r_dim.size;
  if (l_dim.size == 1) {
    return r_dim;
  }
  if (r_dim.size == 1) {
    return l_dim;
  }
  return l_dim.boundOp.has_value() ? l_dim : r_dim;
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

// Broadcasts the leading batch dimensions of a rank >= 2 `matrix` from
// `[orig_batch..., R, C]` to `[batch_dims..., R, C]`, keeping its trailing two
// matrix dimensions `[R, C]` intact.
absl::StatusOr<mlir::MlirOp> BroadcastMatrixBatchDims(
    mlir::MlirOp matrix, const stablehlo::Dimensions& batch_dims) {
  const stablehlo::Dimensions matrix_dims = GetDimensions(matrix);
  const int64_t rank = matrix_dims.size();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal invariant (callers ensure rank
                 // >= 2).
      rank >= 2, error::kInvalidArgument)
      << "expected matrix rank >= 2, got " << rank;
  stablehlo::Dimensions target_dims = batch_dims;
  target_dims.push_back(matrix_dims[rank - 2]);
  target_dims.push_back(matrix_dims[rank - 1]);
  if (AreDimensionsEqual(matrix_dims, target_dims)) {
    return matrix;
  }
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
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Checked before MLIR lowering.
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

// Right-aligns the batch dimensions of `lhs_shape` and `rhs_shape`, appends the
// broadcasted batch sizes to `output_shape`, drops `kTrivial` (`1 vs 1`) axes,
// and coalesces adjacent batch axes of the same `ForwardBatchAxisKind` into
// single flattened axes to minimize tensor rank during `dot_general`.
absl::StatusOr<llvm::SmallVector<CoalescedForwardAxis>>
CoalesceForwardBatchAxes(llvm::ArrayRef<int64_t> lhs_shape,
                         llvm::ArrayRef<int64_t> rhs_shape,
                         Dimensions& output_shape) {
  const int64_t lhs_batch_rank =
      std::max<int64_t>(0, static_cast<int64_t>(lhs_shape.size()) - 2);
  const int64_t rhs_batch_rank =
      std::max<int64_t>(0, static_cast<int64_t>(rhs_shape.size()) - 2);
  const int64_t batch_rank = std::max(lhs_batch_rank, rhs_batch_rank);
  output_shape.reserve(batch_rank + 2);
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
    output_shape.push_back(bcast_size);
    if (kind == ForwardBatchAxisKind::kTrivial) {
      continue;
    }
    if (!axes.empty() && axes.back().kind == kind) {
      axes.back().size *= bcast_size;
    } else {
      axes.push_back({kind, bcast_size});
    }
  }
  return axes;
}

// Computes the static output shape of `matmul(lhs, rhs)` across all rank
// combinations (`[broadcast_batch..., M?, N?]`).
absl::StatusOr<Dimensions> ComputeStaticMatmulOutputShape(
    llvm::ArrayRef<int64_t> lhs_shape, llvm::ArrayRef<int64_t> rhs_shape,
    const MatmulShloOptions& options) {
  const int64_t lhs_rank = lhs_shape.size();
  const int64_t rhs_rank = rhs_shape.size();
  Dimensions output_shape;
  TT_RETURN_IF_ERROR(
      CoalesceForwardBatchAxes(lhs_shape, rhs_shape, output_shape).status());
  if (lhs_rank >= 2) {
    output_shape.push_back(
        lhs_shape[GetLhsFreeMatrixDim(lhs_rank, options.transpose_lhs)]);
  }
  if (rhs_rank >= 2) {
    output_shape.push_back(
        rhs_shape[GetRhsFreeMatrixDim(rhs_rank, options.transpose_rhs)]);
  }
  return output_shape;
}

// Handles static matmul when `lhs` or `rhs` contains a zero-sized dimension:
//   - If the output shape itself has a zero dimension (e.g., `M == 0`, `N ==
//     0`, or a batch dimension is `0`), returns an empty tensor of that shape.
//   - If the output shape has non-zero numel because only the contracting
//     dimension `K == 0` (e.g., `[M, 0] x [0, N] -> [M, N]`), the sum over an
//     empty contracting axis is the additive identity `0`, so a zero-filled
//     constant tensor is emitted.
absl::StatusOr<mlir::MlirOp> EmitZeroElementMatmul(
    mlir::MlirOp lhs, llvm::ArrayRef<int64_t> lhs_shape,
    llvm::ArrayRef<int64_t> rhs_shape, const MatmulShloOptions& options) {
  TT_ASSIGN_OR_RETURN(
      const Dimensions output_shape,
      ComputeStaticMatmulOutputShape(lhs_shape, rhs_shape, options));
  const mlir::Type element_type =
      options.target_element_type.has_value()
          ? mlir::getElementType(lhs.getContext(), *options.target_element_type)
          : GetTensorTypeOrDie(lhs).getElementType();
  const llvm::ArrayRef<int64_t> out_shape_ref =
      AsArrayRef(absl::MakeConstSpan(output_shape));
  if (llvm::is_contained(output_shape, 0)) {
    return MakeZeroSizedTensor(lhs.getBuilder(), element_type, out_shape_ref);
  }
  return MakeConstant(lhs.getBuilder(), 0, element_type, out_shape_ref);
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
// `rhs_rank <= 2`) has static shapes and non-transposed `lhs` (`[..., M, K]`),
// allowing all leading batch and row dimensions `[..., M]` of `lhs` to be
// collapsed into a single 2D matrix `[prod(..., M), K]`.
bool CanFoldLhsTo2D(const int64_t lhs_rank, const int64_t rhs_rank,
                    const bool has_dynamic_dims,
                    const MatrixTranspose transpose_lhs) {
  return !has_dynamic_dims && lhs_rank >= 3 && rhs_rank <= 2 &&
         transpose_lhs == MatrixTranspose::kNone;
}

// Executes `ND x 1D` (`[..., M, K] x [K] -> [..., M]`) or `ND x 2D`
// (`[..., M, K] x [K, N] -> [..., M, N]`) as a single unbatched 2D GEMM by
// reshaping `lhs` to `[prod(..., M), K]`, contracting with `rhs`, and reshaping
// the result back to `[..., M]` or `[..., M, N]`.
mlir::MlirOp EmitFoldedLhsMatmul(mlir::MlirOp lhs, mlir::MlirOp rhs,
                                 const MatmulShloOptions& options,
                                 const stablehlo::Precision precision) {
  const llvm::ArrayRef<int64_t> lhs_shape = GetTensorTypeOrDie(lhs).getShape();
  const llvm::ArrayRef<int64_t> rhs_shape = GetTensorTypeOrDie(rhs).getShape();
  const int64_t rhs_rank = rhs_shape.size();
  const int64_t leading_rows = ProductOfDims(lhs_shape.drop_back(1));
  const int64_t k_dim = lhs_shape.back();
  mlir::MlirOp lhs_2d = ReshapeIfChanged(lhs, Dimensions{leading_rows, k_dim});
  const int64_t rhs_contract_dim =
      GetRhsContractingDim(rhs_rank, options.transpose_rhs);
  mlir::MlirOp dot_2d = BuildTypedDotGeneral(
      lhs_2d, rhs, /*lhs_batching=*/{}, /*rhs_batching=*/{},
      /*lhs_contracting=*/{1}, /*rhs_contracting=*/{rhs_contract_dim},
      precision, options.target_element_type);
  Dimensions output_shape(lhs_shape.begin(), lhs_shape.end() - 1);
  if (rhs_rank == 2) {
    output_shape.push_back(
        rhs_shape[GetRhsFreeMatrixDim(rhs_rank, options.transpose_rhs)]);
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
  for (const auto& axis : axes) {
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
//   1. Coalesces aligned batch axes into `kShared`, `kLhsOnly`, and `kRhsOnly`
//      groups (dropping `kTrivial` size-1 axes).
//   2. Reshapes `lhs` and `rhs` to their squeezed/coalesced shapes, passing
//      only `kShared` axes as `dot_general` batch dimensions while leaving
//      `kLhsOnly` and `kRhsOnly` axes as free dimensions on `lhs` and `rhs`.
//   3. Permutes the `dot_general` result to restore the interleaved batch axis
//      order and reshapes to `output_shape`.
absl::StatusOr<mlir::MlirOp> EmitStaticBatchedMatmul(
    mlir::MlirOp lhs, mlir::MlirOp rhs, const MatmulShloOptions& options,
    const stablehlo::Precision precision) {
  const llvm::ArrayRef<int64_t> lhs_shape = GetTensorTypeOrDie(lhs).getShape();
  const llvm::ArrayRef<int64_t> rhs_shape = GetTensorTypeOrDie(rhs).getShape();
  const int64_t lhs_rank = lhs_shape.size();
  const int64_t rhs_rank = rhs_shape.size();
  Dimensions output_shape;
  TT_ASSIGN_OR_RETURN(
      const llvm::SmallVector<CoalescedForwardAxis> axes,
      CoalesceForwardBatchAxes(lhs_shape, rhs_shape, output_shape));
  output_shape.push_back(
      lhs_shape[GetLhsFreeMatrixDim(lhs_rank, options.transpose_lhs)]);
  output_shape.push_back(
      rhs_shape[GetRhsFreeMatrixDim(rhs_rank, options.transpose_rhs)]);

  Dimensions lhs_target_shape;
  Dimensions rhs_target_shape;
  Dimensions lhs_batch_dims;
  Dimensions rhs_batch_dims;
  int64_t num_lhs_only = 0;
  int64_t num_rhs_only = 0;
  for (const auto& axis : axes) {
    if (axis.kind == ForwardBatchAxisKind::kShared) {
      lhs_batch_dims.push_back(lhs_target_shape.size());
      lhs_target_shape.push_back(axis.size);
      rhs_batch_dims.push_back(rhs_target_shape.size());
      rhs_target_shape.push_back(axis.size);
    } else if (axis.kind == ForwardBatchAxisKind::kLhsOnly) {
      lhs_target_shape.push_back(axis.size);
      ++num_lhs_only;
    } else {
      rhs_target_shape.push_back(axis.size);
      ++num_rhs_only;
    }
  }
  lhs_target_shape.push_back(lhs_shape[lhs_rank - 2]);
  lhs_target_shape.push_back(lhs_shape[lhs_rank - 1]);
  rhs_target_shape.push_back(rhs_shape[rhs_rank - 2]);
  rhs_target_shape.push_back(rhs_shape[rhs_rank - 1]);

  lhs = ReshapeIfChanged(lhs, lhs_target_shape);
  rhs = ReshapeIfChanged(rhs, rhs_target_shape);
  const int64_t lhs_contract =
      GetLhsContractingDim(lhs_target_shape.size(), options.transpose_lhs);
  const int64_t rhs_contract =
      GetRhsContractingDim(rhs_target_shape.size(), options.transpose_rhs);
  mlir::MlirOp result = BuildTypedDotGeneral(
      lhs, rhs, lhs_batch_dims, rhs_batch_dims, {lhs_contract}, {rhs_contract},
      precision, options.target_element_type);
  const Dimensions perm = BuildForwardOutputPermutation(
      axes, lhs_batch_dims.size(), num_lhs_only, num_rhs_only);
  result = TransposeIfNeeded(result, perm);
  return ReshapeIfChanged(result, output_shape);
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
// `operand_shape`, drops `kTrivial` (`grad_size == 1`) axes, and coalesces
// adjacent batch axes of the same `BackwardBatchAxisKind`.
llvm::SmallVector<CoalescedBackwardAxis> CoalesceBackwardBatchAxes(
    llvm::ArrayRef<int64_t> grad_shape, absl::Span<const int64_t> target_shape,
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
    const BackwardBatchAxisKind kind =
        ClassifyBackwardBatchAxis(grad_shape[i], target_size, operand_size);
    if (kind == BackwardBatchAxisKind::kTrivial) {
      continue;
    }
    if (!axes.empty() && axes.back().kind == kind) {
      axes.back().size *= grad_shape[i];
    } else {
      axes.push_back({kind, grad_shape[i]});
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
  for (const auto& axis : axes) {
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
  for (const auto& axis : axes) {
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

// Lowers static `2D x ND` (`self: [M, K], other: [..., K, N], grad: [..., M,
// N]`, contracting all batch axes and `N` in one `dot_general`) and general
// static `ND x ND` `grad_self` (`[..., M, N] x [..., K, N]^H -> [..., M, K]`)
// by coalescing batch axes, fusing any `self`-broadcasted batch axes
// (`kContract`) with the matrix contracting axis `N` (`last_dim`) into a single
// `dot_general`, permuting free/shared batch axes, and reshaping to
// `self_static_dims`.
mlir::MlirOp EmitStaticBatchedGradSelf(
    mlir::MlirOp grad, mlir::MlirOp conj_other,
    absl::Span<const int64_t> self_static_dims,
    const stablehlo::Precision precision) {
  const llvm::ArrayRef<int64_t> grad_shape =
      GetTensorTypeOrDie(grad).getShape();
  if (self_static_dims.size() == 2) {
    const int64_t batch_rank = static_cast<int64_t>(grad_shape.size()) - 2;
    Dimensions contract_dims = MakeIotaDims(batch_rank);
    contract_dims.push_back(batch_rank + 1);
    return BuildTypedDotGeneral(grad, conj_other, /*lhs_batching=*/{},
                                /*rhs_batching=*/{}, contract_dims,
                                contract_dims, precision);
  }
  const llvm::ArrayRef<int64_t> other_shape =
      GetTensorTypeOrDie(conj_other).getShape();
  const auto axes =
      CoalesceBackwardBatchAxes(grad_shape, self_static_dims, other_shape);
  PreparedBackwardBatchShapes shapes =
      PrepareBackwardBatchShapes(axes, grad_shape, other_shape);
  shapes.grad_contract_dims.push_back(shapes.grad_target_shape.size() - 1);
  shapes.operand_contract_dims.push_back(shapes.operand_target_shape.size() -
                                         1);
  grad = ReshapeIfChanged(grad, shapes.grad_target_shape);
  conj_other = ReshapeIfChanged(conj_other, shapes.operand_target_shape);
  mlir::MlirOp result = BuildTypedDotGeneral(
      grad, conj_other, shapes.grad_batch_dims, shapes.operand_batch_dims,
      shapes.grad_contract_dims, shapes.operand_contract_dims, precision);
  const Dimensions perm = BuildBackwardOutputPermutation(
      axes, shapes.grad_batch_dims.size(), shapes.num_grad_free,
      /*grad_is_lhs=*/true);
  result = TransposeIfNeeded(result, perm);
  return ReshapeIfChanged(result, self_static_dims);
}

// Lowers static `ND x ND` `grad_other` (`[..., M, K]^H x [..., M, N] -> [...,
// K, N]`) by coalescing batch axes, fusing any `other`-broadcasted batch axes
// (`kContract`) with the matrix contracting axis `M` (`rank - 2`) into a single
// `dot_general`, permuting free/shared batch axes, and reshaping to
// `other_static_dims`.
mlir::MlirOp EmitStaticBatchedGradOther(
    mlir::MlirOp grad, mlir::MlirOp conj_self,
    absl::Span<const int64_t> other_static_dims,
    const stablehlo::Precision precision) {
  const llvm::ArrayRef<int64_t> grad_shape =
      GetTensorTypeOrDie(grad).getShape();
  const llvm::ArrayRef<int64_t> self_shape =
      GetTensorTypeOrDie(conj_self).getShape();
  const auto axes =
      CoalesceBackwardBatchAxes(grad_shape, other_static_dims, self_shape);
  PreparedBackwardBatchShapes shapes =
      PrepareBackwardBatchShapes(axes, grad_shape, self_shape);
  shapes.operand_contract_dims.push_back(shapes.operand_target_shape.size() -
                                         2);
  shapes.grad_contract_dims.push_back(shapes.grad_target_shape.size() - 2);
  conj_self = ReshapeIfChanged(conj_self, shapes.operand_target_shape);
  grad = ReshapeIfChanged(grad, shapes.grad_target_shape);
  mlir::MlirOp result = BuildTypedDotGeneral(
      conj_self, grad, shapes.operand_batch_dims, shapes.grad_batch_dims,
      shapes.operand_contract_dims, shapes.grad_contract_dims, precision);
  const Dimensions perm = BuildBackwardOutputPermutation(
      axes, shapes.grad_batch_dims.size(), shapes.num_grad_free,
      /*grad_is_lhs=*/false);
  result = TransposeIfNeeded(result, perm);
  return ReshapeIfChanged(result, other_static_dims);
}

// Identifies which leading batch dimensions in `grad_shape` (`[bcast_batch...,
// R, C]`) were broadcasted relative to `target_shape` (`[target_batch..., R,
// C]`):
//   - `reduce_dims`: batch axes with `grad_shape[i] != 1` where `target` either
//     had a lower batch rank (`target_idx < 0`) or a singleton dimension
//     (`target_shape[target_idx] == 1`).
//   - `squeeze_dims`: leading batch axes (`target_idx < 0`) that were prepended
//     during forward broadcasting and must be dropped to restore `target`'s
//     rank.
void CollectBackwardReductionDims(llvm::ArrayRef<int64_t> grad_shape,
                                  llvm::ArrayRef<int64_t> target_shape,
                                  Dimensions& reduce_dims,
                                  Dimensions& squeeze_dims) {
  const int64_t bcast_batch_rank = grad_shape.size() - 2;
  const int64_t target_batch_rank = target_shape.size() - 2;
  const int64_t rank_diff = bcast_batch_rank - target_batch_rank;
  for (int64_t i = 0; i < bcast_batch_rank; ++i) {
    const int64_t target_idx = i - rank_diff;
    if (target_idx < 0) {
      if (grad_shape[i] != 1) {
        reduce_dims.push_back(i);
      }
      squeeze_dims.push_back(i);
    } else if (target_shape[target_idx] == 1 && grad_shape[i] != 1) {
      reduce_dims.push_back(i);
    }
  }
}

// Sums along static dimension `dim` of `op` in place, producing size 1 at
// `dim` via `stablehlo.slice` + `stablehlo.add` without changing rank or
// emitting `stablehlo.reshape`. For `complex<f64>`, avoiding a post-contraction
// `kReshape` into a size-1 batch dimension prevents `TpuLayoutAssignment`'s
// `AssignReshapeLayout` from assigning a non-default minor-to-major layout on
// `X128Combine` that would require a 128-bit layout-changing `kCopy`.
mlir::MlirOp SumDimKeepSizeOneViaSlices(mlir::MlirOp op, const int64_t dim) {
  const llvm::ArrayRef<int64_t> shape = GetTensorTypeOrDie(op).getShape();
  const int64_t extent = shape[dim];
  const Dimensions strides(shape.size(), 1);
  Dimensions starts(shape.size(), 0);
  Dimensions limits(shape.begin(), shape.end());
  limits[dim] = 1;
  mlir::MlirOp acc = stablehlo::Slice(op, starts, limits, strides);
  for (int64_t i = 1; i < extent; ++i) {
    starts[dim] = i;
    limits[dim] = i + 1;
    mlir::MlirOp slice = stablehlo::Slice(op, starts, limits, strides);
    acc = stablehlo::Add(acc, slice);
  }
  return acc;
}

// Reduces broadcasted batch dimensions for static `complex<f64>` gradients
// without emitting a post-contraction `stablehlo.reshape` on kept size-1 batch
// dimensions.
absl::StatusOr<mlir::MlirOp> ReduceComplexF64BroadcastedBatchDims(
    mlir::MlirOp grad_bcast, llvm::ArrayRef<int64_t> target_shape,
    const Dimensions& squeeze_dims) {
  mlir::MlirOp result = grad_bcast;
  if (!squeeze_dims.empty()) {
    TT_ASSIGN_OR_RETURN(
        result, BuildSumShlo(result, squeeze_dims, ReductionMode::kDropDims));
  }
  const llvm::ArrayRef<int64_t> current_shape =
      GetTensorTypeOrDie(result).getShape();
  for (int64_t d = 0; d < static_cast<int64_t>(target_shape.size()) - 2; ++d) {
    if (target_shape[d] == 1 && current_shape[d] > 1) {
      result = SumDimKeepSizeOneViaSlices(result, d);
    }
  }
  return result;
}

// Sum-reduces and squeezes broadcasted batch dimensions from an unreduced
// backward `dot_general` output `grad_bcast` so its shape matches
// `target_static_dims` (used when bounded dynamic dimensions or `complex<f64>`
// prevent fusing the batch reduction into `dot_general`).
absl::StatusOr<mlir::MlirOp> ReduceBroadcastedBatchDims(
    mlir::MlirOp grad_bcast, absl::Span<const int64_t> target_static_dims) {
  const mlir::RankedTensorType grad_type = GetTensorTypeOrDie(grad_bcast);
  const llvm::ArrayRef<int64_t> grad_shape = grad_type.getShape();
  const llvm::ArrayRef<int64_t> target_shape = AsArrayRef(target_static_dims);
  if (grad_shape == target_shape) {
    return grad_bcast;
  }
  Dimensions reduce_dims;
  Dimensions squeeze_dims;
  CollectBackwardReductionDims(grad_shape, target_shape, reduce_dims,
                               squeeze_dims);
  if (reduce_dims == squeeze_dims) {
    if (reduce_dims.empty()) {
      return grad_bcast;
    }
    return BuildSumShlo(grad_bcast, reduce_dims, ReductionMode::kDropDims);
  }
  const bool can_slice_reduce_complex_f64 =
      IsComplexF64(grad_type.getElementType()) &&
      grad_type.getNumDynamicDims() == 0 &&
      llvm::all_of(squeeze_dims,
                   [grad_shape](int64_t d) { return grad_shape[d] != 1; });
  if (can_slice_reduce_complex_f64) {
    return ReduceComplexF64BroadcastedBatchDims(grad_bcast, target_shape,
                                                squeeze_dims);
  }
  mlir::MlirOp result = grad_bcast;
  if (!reduce_dims.empty()) {
    TT_ASSIGN_OR_RETURN(
        result, BuildSumShlo(result, reduce_dims, ReductionMode::kKeepDims));
  }
  if (!squeeze_dims.empty()) {
    TT_ASSIGN_OR_RETURN(result, Squeeze(result, squeeze_dims));
  }
  return result;
}

// Emits a backward batched contraction between broadcast-aligned `lhs` and
// `rhs` (both of rank `batch_rank + 2`) with all `batch_indices` as
// `dot_general` batch dimensions, followed by `ReduceBroadcastedBatchDims` to
// sum-reduce and squeeze any batch dimensions that were broadcasted for
// `target`.
absl::StatusOr<mlir::MlirOp> EmitBatchedGradWithReduction(
    mlir::MlirOp lhs, mlir::MlirOp rhs,
    absl::Span<const int64_t> target_static_dims, const int64_t batch_rank,
    const int64_t matrix_contract_dim, const stablehlo::Precision precision) {
  const Dimensions batch_indices = MakeIotaDims(batch_rank);
  mlir::MlirOp grad_bcast = BuildTypedDotGeneral(
      lhs, rhs, batch_indices, batch_indices,
      /*lhs_contracting=*/{matrix_contract_dim},
      /*rhs_contracting=*/{matrix_contract_dim}, precision);
  return ReduceBroadcastedBatchDims(grad_bcast, target_static_dims);
}

// Decomposes a `complex<f64>` `dot_general` into four real `f64`
// `dot_general` operations:
//   `(a + bi) @ (c + di) = (a @ c - b @ d) + (a @ d + b @ c)i`
// This avoids XLA:TPU's `C128` `ComplexRewriter` + `TpuBroadcastRewriter` bugs
// on rank-1 / multi-axis contractions (`ND x 1D` and `1D x ND`).
mlir::MlirOp EmitComplexF64DotGeneral(mlir::MlirOp lhs, mlir::MlirOp rhs,
                                      absl::Span<const int64_t> lhs_batching,
                                      absl::Span<const int64_t> rhs_batching,
                                      absl::Span<const int64_t> lhs_contracting,
                                      absl::Span<const int64_t> rhs_contracting,
                                      const stablehlo::Precision precision) {
  mlir::MlirOp lhs_real = stablehlo::Real(lhs);
  mlir::MlirOp lhs_imag = stablehlo::Imag(lhs);
  mlir::MlirOp rhs_real = stablehlo::Real(rhs);
  mlir::MlirOp rhs_imag = stablehlo::Imag(rhs);
  mlir::MlirOp ac =
      BuildTypedDotGeneral(lhs_real, rhs_real, lhs_batching, rhs_batching,
                           lhs_contracting, rhs_contracting, precision);
  mlir::MlirOp bd =
      BuildTypedDotGeneral(lhs_imag, rhs_imag, lhs_batching, rhs_batching,
                           lhs_contracting, rhs_contracting, precision);
  mlir::MlirOp ad =
      BuildTypedDotGeneral(lhs_real, rhs_imag, lhs_batching, rhs_batching,
                           lhs_contracting, rhs_contracting, precision);
  mlir::MlirOp bc =
      BuildTypedDotGeneral(lhs_imag, rhs_real, lhs_batching, rhs_batching,
                           lhs_contracting, rhs_contracting, precision);
  mlir::MlirOp real_part = stablehlo::Subtract(ac, bd);
  mlir::MlirOp imag_part = stablehlo::Add(ad, bc);
  return stablehlo::Complex(real_part, imag_part);
}

}  // namespace

mlir::MlirOp BuildTypedDotGeneral(
    mlir::MlirOp lhs, mlir::MlirOp rhs, absl::Span<const int64_t> lhs_batching,
    absl::Span<const int64_t> rhs_batching,
    absl::Span<const int64_t> lhs_contracting,
    absl::Span<const int64_t> rhs_contracting,
    const stablehlo::Precision precision,
    const std::optional<mlir::ElementType> preferred_element_type) {
  const mlir::RankedTensorType lhs_type = GetTensorTypeOrDie(lhs);
  const mlir::RankedTensorType rhs_type = GetTensorTypeOrDie(rhs);
  if (IsComplexF64(lhs_type.getElementType())) {
    return EmitComplexF64DotGeneral(lhs, rhs, lhs_batching, rhs_batching,
                                    lhs_contracting, rhs_contracting,
                                    precision);
  }
  const bool is_i64 = lhs_type.getElementType().isInteger(64);
  if (is_i64) {
    // XLA/TPU MXU `dot_general` does not natively support 64-bit integer
    // contractions, so we emulate `i64` contractions via `f64` and convert the
    // result back to `i64` (or `preferred_element_type` if specified).
    lhs = stablehlo::ConvertElementType(lhs, mlir::ElementType::F64);
    rhs = stablehlo::ConvertElementType(rhs, mlir::ElementType::F64);
  }
  const int64_t out_rank = lhs_type.getRank() + rhs_type.getRank() -
                           lhs_batching.size() - lhs_contracting.size() -
                           rhs_contracting.size();
  const bool squeeze_scalar_out =
      out_rank == 0 &&
      (lhs_type.getNumDynamicDims() > 0 || rhs_type.getNumDynamicDims() > 0);
  if (squeeze_scalar_out) {
    // Work around StableHLO inferDotGeneralOp bug where a 0-D output from
    // bounded dynamic inputs retains the input's non-empty bounds attribute.
    stablehlo::Dimensions output_dims = GetDimensions(lhs);
    output_dims.push_back({1});
    const mlir::Type lhs_elem_type = GetTensorTypeOrDie(lhs).getElementType();
    lhs = stablehlo::Reshape(
        stablehlo::getRankedTensorType(output_dims, lhs_elem_type), lhs);
  }
  mlir::MLIRContext& ctx = lhs.getContext();
  const auto dot_dimension_numbers = stablehlo::DotDimensionNumbersAttr::get(
      &ctx, AsArrayRef(lhs_batching), AsArrayRef(rhs_batching),
      AsArrayRef(lhs_contracting), AsArrayRef(rhs_contracting));
  // Attach the requested operand precision config (`{precision, precision}`,
  // typically obtained via `GetAndAddPrecisionTo` or an explicit precision
  // override) to `stablehlo.dot_general`.
  const mlir::ArrayAttr precision_config =
      stablehlo::PrecisionConfigAttr::get(&ctx, {precision, precision});
  mlir::MlirOp result =
      stablehlo::DotGeneral(lhs, rhs, dot_dimension_numbers, precision_config,
                            is_i64 ? std::nullopt : preferred_element_type);
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
    mlir::MlirOp lhs, mlir::MlirOp rhs, const stablehlo::Precision precision,
    const MatmulShloOptions& options) {
  const mlir::RankedTensorType init_lhs_type = GetTensorTypeOrDie(lhs);
  const mlir::RankedTensorType init_rhs_type = GetTensorTypeOrDie(rhs);
  const int64_t lhs_rank = init_lhs_type.getRank();
  const int64_t rhs_rank = init_rhs_type.getRank();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Rank constraint is checked before MLIR
                 // lowering.
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
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Type constraint is checked before MLIR
                 // lowering.
      lhs_type.getElementType() == rhs_type.getElementType(),
      error::kInvalidArgument)
      << "expected both arguments to have the same element type";
  const bool has_dynamic_dims =
      lhs_type.getNumDynamicDims() > 0 || rhs_type.getNumDynamicDims() > 0;
  if (!has_dynamic_dims && (llvm::is_contained(lhs_type.getShape(), 0) ||
                            llvm::is_contained(rhs_type.getShape(), 0))) {
    return EmitZeroElementMatmul(lhs, lhs_type.getShape(), rhs_type.getShape(),
                                 options);
  }
  const bool use_broadcast_batched_path =
      has_dynamic_dims || IsComplexF64(lhs_type.getElementType());

  // Lower all 5 `torch.matmul` rank combinations:
  //   1. `1D x 1D` (`[K] x [K] -> []`): Inner product. Handled by the
  //      unbatched `dot_general` branch below with `lhs_contract = {0}`,
  //      `rhs_contract = {0}`.
  //   2. `2D x 2D` (`[M, K] x [K, N] -> [M, N]`): Matrix-matrix product.
  //      Handled by the unbatched `dot_general` branch below.
  //   3. `ND x 1D` (`[..., M, K] x [K] -> [..., M]`): Batched matrix-vector
  //      product. When static and `transpose_lhs == kNone`, `CanFoldLhsTo2D`
  //      flattens `lhs` to `[prod(..., M), K]` for a 2D-x-1D `dot_general`;
  //      otherwise, the unbatched `dot_general` branch contracts `K` directly
  //      with `[..., M]` as `lhs` free dimensions.
  //   4. `1D x ND` (`[K] x [..., K, N] -> [..., N]`): Vector-batched-matrix
  //      product. Handled by the unbatched `dot_general` branch with
  //      `lhs_contract = {0}` and `rhs_contract = {rhs_rank - 2}` (or
  //      `rhs_rank - 1` if transposed), which treats all batch axes `...` and
  //      `N` of `rhs` as free dimensions and directly produces `[..., N]`.
  //   5. `ND x ND` (`[..., M, K] x [..., K, N] -> [..., M, N]`): Batched
  //      matrix-matrix product:
  //      - `ND x 2D`: Folded to 2D via `EmitFoldedLhsMatmul` when static and
  //        non-transposed, or lowered via unbatched `dot_general` (with `lhs`
  //        batch dims as free dims).
  //      - General static `ND x ND`: Lowered via `EmitStaticBatchedMatmul`,
  //        which coalesces shared, lhs-only, and rhs-only batch axes without
  //        materializing broadcast copies.
  //      - Dynamic shapes or 3D+ `complex<f64>` (`use_broadcast_batched_path`):
  //        Explicitly broadcast-aligns leading batch dimensions via
  //        `ComputeBroadcastBatchDims` and `BroadcastMatrixBatchDims` before
  //        emitting a batched `dot_general`.
  if (CanFoldLhsTo2D(lhs_rank, rhs_rank, use_broadcast_batched_path,
                     options.transpose_lhs)) {
    return EmitFoldedLhsMatmul(lhs, rhs, options, precision);
  }
  if (lhs_rank == 1 || rhs_rank == 1 ||
      (rhs_rank == 2 && (!has_dynamic_dims || lhs_rank == 2))) {
    const int64_t lhs_contract =
        GetLhsContractingDim(lhs_rank, options.transpose_lhs);
    const int64_t rhs_contract =
        GetRhsContractingDim(rhs_rank, options.transpose_rhs);
    return BuildTypedDotGeneral(
        lhs, rhs, /*lhs_batching=*/{}, /*rhs_batching=*/{}, {lhs_contract},
        {rhs_contract}, precision, options.target_element_type);
  }
  if (!use_broadcast_batched_path) {
    return EmitStaticBatchedMatmul(lhs, rhs, options, precision);
  }

  const stablehlo::Dimensions lhs_dims = GetDimensions(lhs);
  const stablehlo::Dimensions rhs_dims = GetDimensions(rhs);
  TT_ASSIGN_OR_RETURN(const stablehlo::Dimensions batch_dims,
                      ComputeBroadcastBatchDims(lhs_dims, rhs_dims));
  TT_ASSIGN_OR_RETURN(lhs, BroadcastMatrixBatchDims(lhs, batch_dims));
  TT_ASSIGN_OR_RETURN(rhs, BroadcastMatrixBatchDims(rhs, batch_dims));
  const int64_t batch_rank = batch_dims.size();
  const Dimensions batch_indices = MakeIotaDims(batch_rank);
  const int64_t lhs_contract =
      GetLhsContractingDim(batch_rank + 2, options.transpose_lhs);
  const int64_t rhs_contract =
      GetRhsContractingDim(batch_rank + 2, options.transpose_rhs);
  return BuildTypedDotGeneral(lhs, rhs, batch_indices, batch_indices,
                              {lhs_contract}, {rhs_contract}, precision,
                              options.target_element_type);
}

namespace {

// Emits the gradient tensor for `target` (`self` or `other`) when any operand
// has a zero-sized dimension:
//   - If `target_static_dims` itself contains `0`, returns an empty zero-sized
//     tensor of shape `target_static_dims`.
//   - Otherwise (`target` has non-zero numel, but the other operand or `grad`
//     was empty), returns a zero-filled tensor matching `target` (preserving
//     dynamic bounds via `MakeConstantLike(*target_op, 0)` when `target_op` is
//     provided).
absl::StatusOr<mlir::MlirOp> EmitZeroGradForTarget(
    mlir::MlirOp grad, absl::Span<const int64_t> target_static_dims,
    std::optional<mlir::MlirOp> target_op) {
  const mlir::Type element_type = GetTensorTypeOrDie(grad).getElementType();
  if (llvm::is_contained(target_static_dims, 0)) {
    return MakeZeroSizedTensor(grad.getBuilder(), element_type,
                               AsArrayRef(target_static_dims));
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
    const stablehlo::Precision precision) {
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
    const Dimensions grad_contract = MakeIotaDims(grad_rank);
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
    const stablehlo::Precision precision) {
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
    const int64_t leading_rows = ProductOfDims(self_shape.drop_back(1));
    mlir::MlirOp self_2d = ReshapeIfChanged(
        conj_self, Dimensions{leading_rows, self_shape.back()});
    mlir::MlirOp grad_1d = ReshapeIfChanged(grad, {leading_rows});
    return BuildTypedDotGeneral(self_2d, grad_1d, /*lhs_batching=*/{},
                                /*rhs_batching=*/{}, /*lhs_contracting=*/{0},
                                /*rhs_contracting=*/{0}, precision);
  }
  const Dimensions contract_dims = MakeIotaDims(grad_rank);
  return BuildTypedDotGeneral(conj_self, grad, /*lhs_batching=*/{},
                              /*rhs_batching=*/{}, contract_dims, contract_dims,
                              precision);
}

// Lowers `grad_self` for `2D x 2D` (`[M, N] x [K, N]^H -> [M, K]`) and
// `ND x 2D` (`[..., M, N] x [K, N]^H -> [..., M, K]`, folded to 2D
// `[prod(..., M), N] x [K, N]` when `fold_to_2d` is true, or emitted as an
// unbatched `dot_general` with `grad`'s batch dimensions as free dimensions).
mlir::MlirOp EmitGradSelfWithRank2Other(
    mlir::MlirOp grad, mlir::MlirOp conj_other,
    absl::Span<const int64_t> self_static_dims, const bool fold_to_2d,
    const stablehlo::Precision precision) {
  const llvm::ArrayRef<int64_t> grad_shape =
      GetTensorTypeOrDie(grad).getShape();
  if (fold_to_2d && self_static_dims.size() >= 3) {
    const int64_t leading_rows = ProductOfDims(grad_shape.drop_back(1));
    mlir::MlirOp grad_2d =
        ReshapeIfChanged(grad, Dimensions{leading_rows, grad_shape.back()});
    mlir::MlirOp dot_2d = BuildTypedDotGeneral(
        grad_2d, conj_other, /*lhs_batching=*/{}, /*rhs_batching=*/{},
        /*lhs_contracting=*/{1}, /*rhs_contracting=*/{1}, precision);
    return ReshapeIfChanged(dot_2d, self_static_dims);
  }
  const int64_t grad_rank = grad_shape.size();
  return BuildTypedDotGeneral(grad, conj_other, /*lhs_batching=*/{},
                              /*rhs_batching=*/{},
                              /*lhs_contracting=*/{grad_rank - 1},
                              /*rhs_contracting=*/{1}, precision);
}

// Lowers `grad_other` for `2D x 2D` (`[M, K]^H x [M, N] -> [K, N]`) and static
// `ND x 2D` (`[..., M, K]^H x [..., M, N] -> [K, N]`, flattening `[..., M]` to
// `leading_rows` when `fold_to_2d` is true, or contracting all `[..., M]`
// dimensions directly in `dot_general`).
mlir::MlirOp EmitGradOtherWithRank2Other(mlir::MlirOp grad,
                                         mlir::MlirOp conj_self,
                                         const bool fold_to_2d,
                                         const stablehlo::Precision precision) {
  const llvm::ArrayRef<int64_t> self_shape =
      GetTensorTypeOrDie(conj_self).getShape();
  if (fold_to_2d && self_shape.size() >= 3) {
    const llvm::ArrayRef<int64_t> grad_shape =
        GetTensorTypeOrDie(grad).getShape();
    const int64_t leading_rows = ProductOfDims(self_shape.drop_back(1));
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

absl::StatusOr<mlir::MlirOp> BuildMatmulGradSelfShlo(
    mlir::MlirOp grad, absl::Span<const int64_t> self_static_dims,
    mlir::MlirOp other, const stablehlo::Precision precision,
    std::optional<mlir::MlirOp> self_op) {
  const mlir::RankedTensorType other_type = GetTensorTypeOrDie(other);
  const mlir::RankedTensorType grad_type = GetTensorTypeOrDie(grad);
  const bool has_dynamic_dims =
      grad_type.getNumDynamicDims() > 0 || other_type.getNumDynamicDims() > 0 ||
      (self_op.has_value() &&
       GetTensorTypeOrDie(*self_op).getNumDynamicDims() > 0);
  if (!has_dynamic_dims && (llvm::is_contained(self_static_dims, 0) ||
                            llvm::is_contained(grad_type.getShape(), 0) ||
                            llvm::is_contained(other_type.getShape(), 0))) {
    return EmitZeroGradForTarget(grad, self_static_dims, self_op);
  }
  const int64_t self_rank = self_static_dims.size();
  const int64_t other_rank = other_type.getRank();
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
  if (other_rank == 2 && (self_rank == 2 || !has_dynamic_dims)) {
    return EmitGradSelfWithRank2Other(
        grad, conj_other, self_static_dims,
        /*fold_to_2d=*/!use_broadcast_batched_path, precision);
  }
  // Rank cases `2D x ND` and general static `ND x ND`:
  if (!use_broadcast_batched_path) {
    return EmitStaticBatchedGradSelf(grad, conj_other, self_static_dims,
                                     precision);
  }

  // Fallback for bounded dynamic shapes and 3D+ `complex<f64>`: broadcast
  // `conj_other` to `grad`'s batch prefix and emit a batched contraction with
  // broadcast-axis reduction.
  const stablehlo::Dimensions grad_dims = GetDimensions(grad);
  const stablehlo::Dimensions batch_dims(grad_dims.begin(),
                                         grad_dims.end() - 2);
  TT_ASSIGN_OR_RETURN(conj_other,
                      BroadcastMatrixBatchDims(conj_other, batch_dims));
  const int64_t batch_rank = batch_dims.size();
  return EmitBatchedGradWithReduction(grad, conj_other, self_static_dims,
                                      batch_rank, batch_rank + 1, precision);
}

absl::StatusOr<mlir::MlirOp> BuildMatmulGradSelfShlo(
    mlir::MlirOp grad, mlir::MlirOp self, mlir::MlirOp other,
    const stablehlo::Precision precision) {
  return BuildMatmulGradSelfShlo(grad, GetTensorTypeOrDie(self).getShape(),
                                 other, precision, self);
}

absl::StatusOr<mlir::MlirOp> BuildMatmulGradOtherShlo(
    mlir::MlirOp grad, mlir::MlirOp self,
    absl::Span<const int64_t> other_static_dims,
    const stablehlo::Precision precision,
    std::optional<mlir::MlirOp> other_op) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  const mlir::RankedTensorType grad_type = GetTensorTypeOrDie(grad);
  const bool has_dynamic_dims =
      grad_type.getNumDynamicDims() > 0 || self_type.getNumDynamicDims() > 0 ||
      (other_op.has_value() &&
       GetTensorTypeOrDie(*other_op).getNumDynamicDims() > 0);
  if (!has_dynamic_dims && (llvm::is_contained(other_static_dims, 0) ||
                            llvm::is_contained(grad_type.getShape(), 0) ||
                            llvm::is_contained(self_type.getShape(), 0))) {
    return EmitZeroGradForTarget(grad, other_static_dims, other_op);
  }
  const int64_t self_rank = self_type.getRank();
  const int64_t other_rank = other_static_dims.size();
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
  // Rank cases `2D x 2D` and static `ND x 2D`:
  if (other_rank == 2 && (self_rank == 2 || !has_dynamic_dims)) {
    return EmitGradOtherWithRank2Other(
        grad, conj_self, /*fold_to_2d=*/!use_broadcast_batched_path, precision);
  }
  // General static `ND x ND` (including `2D x ND`):
  if (!use_broadcast_batched_path) {
    return EmitStaticBatchedGradOther(grad, conj_self, other_static_dims,
                                      precision);
  }

  // Fallback for bounded dynamic shapes and 3D+ `complex<f64>`: broadcast
  // `conj_self` to `grad`'s batch prefix and emit a batched contraction with
  // broadcast-axis reduction.
  const stablehlo::Dimensions grad_dims = GetDimensions(grad);
  const stablehlo::Dimensions batch_dims(grad_dims.begin(),
                                         grad_dims.end() - 2);
  TT_ASSIGN_OR_RETURN(conj_self,
                      BroadcastMatrixBatchDims(conj_self, batch_dims));
  const int64_t batch_rank = batch_dims.size();
  return EmitBatchedGradWithReduction(conj_self, grad, other_static_dims,
                                      batch_rank, batch_rank, precision);
}

absl::StatusOr<mlir::MlirOp> BuildMatmulGradOtherShlo(
    mlir::MlirOp grad, mlir::MlirOp self, mlir::MlirOp other,
    const stablehlo::Precision precision) {
  return BuildMatmulGradOtherShlo(
      grad, self, GetTensorTypeOrDie(other).getShape(), precision, other);
}

}  // namespace torch_tpu
