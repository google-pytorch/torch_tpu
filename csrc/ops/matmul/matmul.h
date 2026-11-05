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

#ifndef TORCH_TPU_CSRC_OPS_MATMUL_MATMUL_H_
#define TORCH_TPU_CSRC_OPS_MATMUL_MATMUL_H_

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

// Controls whether the last two dimensions of a rank >= 2 matrix operand are
// logically transposed before contracting:
//   - `kNone`: For `lhs` of shape `[..., M, K]`, contracts along the last
//     dimension `K` (`rank - 1`). For `rhs` of shape `[..., K, N]`, contracts
//     along the second-to-last dimension `K` (`rank - 2`).
//   - `kTransposeLastTwo`: Swaps the contracting and free matrix dimensions of
//     the operand inside `DotDimensionNumbersAttr` without emitting a separate
//     `stablehlo.transpose` op. For `lhs` of shape `[..., K, M]`, contracts
//     along `rank - 2`; for `rhs` of shape `[..., N, K]`, contracts along
//     `rank - 1`. Ignored for 1D operands.
enum class MatrixTranspose {
  kNone,
  kTransposeLastTwo,
};

// Controls whether physical complex conjugation (`BuildConjPhysicalShlo`) is
// applied to an operand before contracting:
//   - `kNone`: Leave the operand unchanged.
//   - `kConjugate`: Negate the imaginary component of complex operands before
//     contracting (a no-op on real/integer operands). Used by Hermitian inner
//     products (`aten::vdot`, `linalg.vecdot`) and complex backward passes.
enum class ConjugateMode {
  kNone,
  kConjugate,
};

// Controls whether `BuildMatmulShlo` flattens `lhs` `[..., M, K]` to 2D
// `[prod(..., M), K]` when `rhs` is 1D or 2D:
//   - `kAuto`: Flatten static non-transposed `lhs` when `rhs` is 1D or
//     non-transposed 2D.
//   - `kNever`: Preserve `lhs` rank as free dimensions in `dot_general` (used
//     by `aten::linear` so XLA sees the un-reshaped 3D+ activation tensor).
enum class LeadingDimFolding {
  kAuto,
  kNever,
};

// Holds the per-operand precision configuration `(lhs, rhs)` for
// `stablehlo.dot_general`. Constructible from a single
// `mlir::stablehlo::Precision` (which sets both `lhs` and `rhs` to the same
// precision) or from `{lhs_precision, rhs_precision}` for asymmetric operand
// precision (such as `scaled_mm` FP4/FP8 mixed-dtype contractions).
struct MatmulPrecisionPair {
  mlir::stablehlo::Precision lhs =
      mlir::stablehlo::Precision::DEFAULT;  // EXPLICIT_PRECISION_OK=default
  mlir::stablehlo::Precision rhs =
      mlir::stablehlo::Precision::DEFAULT;  // EXPLICIT_PRECISION_OK=default

  constexpr MatmulPrecisionPair() = default;
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr MatmulPrecisionPair(mlir::stablehlo::Precision precision)
      : lhs(precision), rhs(precision) {}
  constexpr MatmulPrecisionPair(mlir::stablehlo::Precision lhs_precision,
                                mlir::stablehlo::Precision rhs_precision)
      : lhs(lhs_precision), rhs(rhs_precision) {}
};

// Creates a 2-element `PrecisionConfigAttr` (`{precision.lhs, precision.rhs}`)
// for `stablehlo.dot_general`.
mlir::ArrayAttr MakeDotPrecisionConfig(mlir::MLIRContext& ctx,
                                       MatmulPrecisionPair precision);

// Configuration options for `BuildMatmulShlo`.
//
// Allows callers such as `aten::linear` (`x @ w^T` via `transpose_rhs =
// kTransposeLastTwo, leading_dim_folding = kNever`), `aten::vdot` (`conj(x) .
// y` via `conjugate_lhs = kConjugate`), and `aten::addmm` / `aten::bmm` /
// `aten::addmv` (`out_dtype` or integer-to-float compute promotion via
// `target_element_type`) to express fused contractions through the single
// shared `BuildMatmulShlo` builder.
struct MatmulShloOptions {
  MatrixTranspose transpose_lhs = MatrixTranspose::kNone;
  MatrixTranspose transpose_rhs = MatrixTranspose::kNone;
  ConjugateMode conjugate_lhs = ConjugateMode::kNone;
  ConjugateMode conjugate_rhs = ConjugateMode::kNone;
  LeadingDimFolding leading_dim_folding = LeadingDimFolding::kAuto;
  std::optional<mlir::ElementType> target_element_type = std::nullopt;
};

// Emits a `stablehlo.dot_general` operation over arbitrary batching and
// contracting dimension lists with unified TPU type and shape handling:
//   - Dimension validation & bounded-dynamic reconciliation: Validates batch
//     and contracting dimension lists and reconciles mismatched upper bounds or
//     mixed static/bounded-dynamic dimensions on paired axes.
//   - `complex<f64>`: Decomposes into 4 real `f64` `dot_general` calls
//     (`ac - bd`, `ad + bc`) and recombines via `stablehlo.complex` to avoid
//     XLA:TPU `C128` `ComplexRewriter` / `TpuBroadcastRewriter` failures.
//   - `i64`: Converts both operands to `f64`, runs `dot_general` with
//     `Precision::HIGHEST`, and converts the result back to `i64` (or
//     `preferred_element_type` if specified), since XLA/TPU does not natively
//     support `i64` `dot_general`.
//   - Bounded-dynamic 0D scalar output (`out_rank == 0`): Unsqueezes `lhs` with
//     a trailing size-1 dimension before `dot_general` and reshapes the 1D
//     result back to `{}` to work around StableHLO `inferDotGeneralOp`
//     retaining non-empty bounds on rank-0 results.
//   - Attaches `PrecisionConfigAttr{precision.lhs, precision.rhs}` and
//     optional `preferred_element_type`.
absl::StatusOr<mlir::MlirOp> BuildTypedDotGeneral(
    mlir::MlirOp lhs, mlir::MlirOp rhs, absl::Span<const int64_t> lhs_batching,
    absl::Span<const int64_t> rhs_batching,
    absl::Span<const int64_t> lhs_contracting,
    absl::Span<const int64_t> rhs_contracting, MatmulPrecisionPair precision,
    std::optional<mlir::ElementType> preferred_element_type = std::nullopt);

// Builds the StableHLO computation for general matrix multiplication
// (`torch.matmul` and all derived contraction ops: `dot`, `vdot`, `mm`, `mv`,
// `bmm`, `addmm`, `baddbmm`, `addmv`, `linear`), supporting all 5 PyTorch rank
// combinations (`1D x 1D`, `2D x 2D`, `ND x 1D`, `1D x ND`, `ND x ND`) with
// NumPy batch broadcasting, bounded dynamic shapes, zero-sized dimensions, and
// optional operand transpose/conjugation/dtype promotion via `options`.
absl::StatusOr<mlir::MlirOp> BuildMatmulShlo(
    mlir::MlirOp lhs, mlir::MlirOp rhs, MatmulPrecisionPair precision,
    const MatmulShloOptions& options = {});

// Builds the StableHLO computation for the gradient of `matmul(self, other)`
// with respect to `self` (`dL/d(self) = grad @ conj(other)^T`), handling all
// rank combinations (`1D..ND`), complex Wirtinger conjugation, zero-sized
// tensors, and reduction/squeezing across any batch dimensions that were
// broadcasted for `self` during the forward pass.
//
// `self_static_dims` specifies the static (or upper-bound) shape of `self`.
// When `self_op` is provided, bounded dynamic dimension metadata on `self` is
// preserved when constructing zero-gradient constants or broadcasting scalar
// gradients.
absl::StatusOr<mlir::MlirOp> BuildMatmulGradSelfShlo(
    mlir::MlirOp grad, absl::Span<const int64_t> self_static_dims,
    mlir::MlirOp other, MatmulPrecisionPair precision,
    std::optional<mlir::MlirOp> self_op = std::nullopt);

// Overload of `BuildMatmulGradSelfShlo` that extracts `self_static_dims`
// (including upper bounds for bounded dynamic dimensions) from `self` and
// forwards `self` as `self_op`.
absl::StatusOr<mlir::MlirOp> BuildMatmulGradSelfShlo(
    mlir::MlirOp grad, mlir::MlirOp self, mlir::MlirOp other,
    MatmulPrecisionPair precision);

// Builds the StableHLO computation for the gradient of `matmul(self, other)`
// with respect to `other` (`dL/d(other) = conj(self)^T @ grad`), handling all
// rank combinations (`1D..ND`), complex Wirtinger conjugation, zero-sized
// tensors, and reduction/squeezing across any batch dimensions that were
// broadcasted for `other` during the forward pass.
//
// `other_static_dims` specifies the static (or upper-bound) shape of `other`.
// When `other_op` is provided, bounded dynamic dimension metadata on `other` is
// preserved when constructing zero-gradient constants or broadcasting scalar
// gradients.
absl::StatusOr<mlir::MlirOp> BuildMatmulGradOtherShlo(
    mlir::MlirOp grad, mlir::MlirOp self,
    absl::Span<const int64_t> other_static_dims, MatmulPrecisionPair precision,
    std::optional<mlir::MlirOp> other_op = std::nullopt);

// Overload of `BuildMatmulGradOtherShlo` that extracts `other_static_dims`
// (including upper bounds for bounded dynamic dimensions) from `other` and
// forwards `other` as `other_op`.
absl::StatusOr<mlir::MlirOp> BuildMatmulGradOtherShlo(
    mlir::MlirOp grad, mlir::MlirOp self, mlir::MlirOp other,
    MatmulPrecisionPair precision);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_MATMUL_MATMUL_H_
