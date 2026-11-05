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
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

// Controls whether the trailing two matrix dimensions `[..., R, C]` of an
// operand are logically transposed to `[..., C, R]` by swapping the free and
// contracting dimension indices passed to `stablehlo.dot_general` (avoiding a
// physical `stablehlo.transpose` op).
enum class MatrixTranspose {
  kNone,
  kTransposeLastTwo,
};

// Controls whether physical element-wise complex conjugation (`a + bi -> a -
// bi`) is applied to an operand prior to contraction (a no-op for real types).
enum class ConjugateMode {
  kNone,
  kConjugate,
};

// Optional pre-contraction transformations fused into `BuildMatmulShlo`.
// Callers such as `aten::linear`, `aten::bmm`, or conjugate-transpose kernels
// can specify logical matrix transposes, physical complex conjugation, and
// computation/accumulation dtype overrides (`target_element_type`):
//   - For non-floating-point operands (e.g., promoting sub-32-bit integers
//     `i8`, `u8`, `i16`, `u16` to `f32`), operands are cast via
//     `stablehlo.convert` prior to `dot_general`.
//   - For floating-point operands (e.g., `bf16`/`f16` inputs with `f32`
//     output in `bmm_dtype` / `baddbmm.dtype`), `target_element_type` is
//     forwarded as `preferredElementType` on `stablehlo.dot_general` so XLA
//     accumulates directly in the target type without upcasting inputs.
struct MatmulShloOptions {
  MatrixTranspose transpose_lhs = MatrixTranspose::kNone;
  MatrixTranspose transpose_rhs = MatrixTranspose::kNone;
  ConjugateMode conjugate_lhs = ConjugateMode::kNone;
  ConjugateMode conjugate_rhs = ConjugateMode::kNone;
  std::optional<mlir::ElementType> target_element_type = std::nullopt;
};

// Emits a `stablehlo.dot_general` operation with the given batching and
// contracting dimensions, uniform `{precision, precision}` `PrecisionConfig`
// attribute, and optional `preferred_element_type` accumulation/output type.
// Transparently handles two XLA/StableHLO constraints:
//   1. 64-bit integer emulation: XLA/TPU MXU `dot_general` does not natively
//      support `i64` contractions, so `i64` operands are cast `i64 -> f64`
//      before `dot_general` and cast back `f64 -> i64` (or
//      `preferred_element_type`) afterward.
//   2. Bounded dynamic 0D output workaround: When contracting bounded dynamic
//      1D tensors into a 0D scalar (`1D x 1D -> []`), temporarily unsqueezes
//      `lhs` before `dot_general` and reshapes the 1D result to `[]` to avoid a
//      StableHLO `inferDotGeneralOp` bug that attaches non-empty bounds to 0D
//      types.
mlir::MlirOp BuildTypedDotGeneral(
    mlir::MlirOp lhs, mlir::MlirOp rhs, absl::Span<const int64_t> lhs_batching,
    absl::Span<const int64_t> rhs_batching,
    absl::Span<const int64_t> lhs_contracting,
    absl::Span<const int64_t> rhs_contracting,
    mlir::stablehlo::Precision precision,
    std::optional<mlir::ElementType> preferred_element_type = std::nullopt);

// Builds a StableHLO representation for `aten::matmul(lhs, rhs)` across all 5
// PyTorch rank combinations with NumPy-style batch broadcasting:
//   1. `1D x 1D`: inner product `[K] x [K] -> []`
//   2. `2D x 2D`: matrix-matrix product `[M, K] x [K, N] -> [M, N]`
//   3. `ND x 1D`: batched matrix-vector product `[..., M, K] x [K] -> [..., M]`
//   4. `1D x ND`: vector-batched-matrix product `[K] x [..., K, N] -> [..., N]`
//   5. `ND x ND`: batched matrix-matrix product
//      `[..., M, K] x [..., K, N] -> [..., M, N]`
// For static shapes, broadcast-only batch axes are coalesced as free matrix
// dimensions rather than materialized via `stablehlo.broadcast_in_dim`.
absl::StatusOr<mlir::MlirOp> BuildMatmulShlo(
    mlir::MlirOp lhs, mlir::MlirOp rhs, mlir::stablehlo::Precision precision,
    const MatmulShloOptions& options = {});

// Builds a StableHLO representation for `grad_self` (`dL/d(self)`) in
// `aten::matmul_backward`, following the Wirtinger complex conjugate transpose
// convention (`grad @ conj(other)^T`) and reducing any batch dimensions that
// were broadcasted for `self` during the forward pass.
//
// When `self` has static dimensions, only `self_static_dims` is required and
// `self` does not need to be passed as an XLA parameter; when bounded dynamic
// dimensions are present, pass `self_op` to preserve dynamic dimension bounds.
absl::StatusOr<mlir::MlirOp> BuildMatmulGradSelfShlo(
    mlir::MlirOp grad, absl::Span<const int64_t> self_static_dims,
    mlir::MlirOp other, mlir::stablehlo::Precision precision,
    std::optional<mlir::MlirOp> self_op = std::nullopt);

// Convenience overload of `BuildMatmulGradSelfShlo` that extracts static shape
// and dynamic dimension bounds directly from `self`.
absl::StatusOr<mlir::MlirOp> BuildMatmulGradSelfShlo(
    mlir::MlirOp grad, mlir::MlirOp self, mlir::MlirOp other,
    mlir::stablehlo::Precision precision);

// Builds a StableHLO representation for `grad_other` (`dL/d(other)`) in
// `aten::matmul_backward`, following the Wirtinger complex conjugate transpose
// convention (`conj(self)^T @ grad`) and reducing any batch dimensions that
// were broadcasted for `other` during the forward pass.
//
// When `other` has static dimensions, only `other_static_dims` is required and
// `other` does not need to be passed as an XLA parameter; when bounded dynamic
// dimensions are present, pass `other_op` to preserve dynamic dimension bounds.
absl::StatusOr<mlir::MlirOp> BuildMatmulGradOtherShlo(
    mlir::MlirOp grad, mlir::MlirOp self,
    absl::Span<const int64_t> other_static_dims,
    mlir::stablehlo::Precision precision,
    std::optional<mlir::MlirOp> other_op = std::nullopt);

// Convenience overload of `BuildMatmulGradOtherShlo` that extracts static shape
// and dynamic dimension bounds directly from `other`.
absl::StatusOr<mlir::MlirOp> BuildMatmulGradOtherShlo(
    mlir::MlirOp grad, mlir::MlirOp self, mlir::MlirOp other,
    mlir::stablehlo::Precision precision);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_MATMUL_MATMUL_H_
