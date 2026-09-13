/*
 * Copyright 2025 Google LLC
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

#ifndef TORCH_TPU_CSRC_OPS_CONTRACTION_UTILS_H_
#define TORCH_TPU_CSRC_OPS_CONTRACTION_UTILS_H_  // NON_TT_MACRO_OK
                                                 // = Numerics
                                                 // Helper

#include "absl/status/statusor.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

// Extracts the `part_idx`-th BF16 slice (0, 1, or 2) of an F32 tensor using
// residual subtraction with NaN zeroing. Computing each slice on demand allows
// XLA to fuse the elementwise extraction directly into each DotGeneral loop
// without allocating intermediate BF16 buffers in HBM.
mlir::MlirOp GetBf16Part(mlir::MlirOp x_f32, int part_idx);

// Casts or converts `op` to `expected_output_type`. Uses `ConvertElementType`
// directly when the target type is integral (e.g. converting an F64 accumulator
// back to i64), and `CastIfNeeded` otherwise.
absl::StatusOr<mlir::MlirOp> CastOrConvertTo(
    mlir::MlirOp op, mlir::ElementType expected_output_type);

// Computes a high-accuracy matrix contraction returning the unrounded
// high-precision accumulator:
// - FP8 inputs  -> converted to BF16, single-pass DotGeneral with F32 accum
// - BF16 inputs -> single-pass DotGeneral with preferredElementType = F32
// - F16 inputs  -> 2-part (3-pass) BF16 split accumulated in F32
// - F32 inputs  -> 3-part (6-pass) BF16 split accumulated in F32
// - Complex64   -> 4x 3-part (6-pass) BF16 split accumulated in Complex<F32>
// - Int64       -> converted to F64 and accumulated in F64
absl::StatusOr<mlir::MlirOp> BuildAccurateDotGeneralAccumulator(
    mlir::MlirOp lhs, mlir::MlirOp rhs,
    mlir::stablehlo::DotDimensionNumbersAttr dot_dimension_numbers,
    mlir::stablehlo::Precision precision);

// Computes a high-accuracy matrix contraction and casts/converts the resulting
// accumulator to `out_dtype`.
absl::StatusOr<mlir::MlirOp> BuildAccurateDotGeneral(
    mlir::MlirOp lhs, mlir::MlirOp rhs,
    mlir::stablehlo::DotDimensionNumbersAttr dot_dimension_numbers,
    mlir::ElementType out_dtype, mlir::stablehlo::Precision precision);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_CONTRACTION_UTILS_H_
