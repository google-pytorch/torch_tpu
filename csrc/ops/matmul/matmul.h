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

#include "absl/status/statusor.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

// Builds a StableHLO representation for `aten::matmul(lhs, rhs)` across
// arbitrary tensor ranks (1D, 2D, 3D, 4D+) with NumPy-style batch broadcasting.
absl::StatusOr<mlir::MlirOp> BuildMatmulShlo(
    mlir::MlirOp lhs, mlir::MlirOp rhs, mlir::stablehlo::Precision precision);

// Builds a StableHLO representation for `grad_self` in `aten::matmul_backward`.
absl::StatusOr<mlir::MlirOp> BuildMatmulGradSelfShlo(
    mlir::MlirOp grad, mlir::MlirOp self, mlir::MlirOp other,
    mlir::stablehlo::Precision precision);

// Builds a StableHLO representation for `grad_other` in
// `aten::matmul_backward`.
absl::StatusOr<mlir::MlirOp> BuildMatmulGradOtherShlo(
    mlir::MlirOp grad, mlir::MlirOp self, mlir::MlirOp other,
    mlir::stablehlo::Precision precision);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_MATMUL_MATMUL_H_
