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

#ifndef TORCH_TPU_OPS_LINALG_QR_QR_LIB_H_
#define TORCH_TPU_OPS_LINALG_QR_QR_LIB_H_

#include "absl/status/statusor.h"
#include "c10/util/string_view.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

// Generates StableHLO for "geqrf" using the XLA "Qr" custom call.
//
// Inputs:
//   input: Matrix of shape (..., M, N).
//
// Returns:
//   A result containing two MlirOps:
//     - result[0]: Packed QR matrix of shape (..., M, N).
//     - result[1]: Elementary reflectors (tau) of shape (..., min(M, N)).
absl::StatusOr<MlirOpResults<2>> BuildGeqrfShlo(mlir::MlirOp input);

// Generates StableHLO for QR decomposition using the XLA
// "ProductOfElementaryHouseholderReflectors" custom call.
//
// Inputs:
//   input: Matrix of shape (..., M, N).
//   mode: "reduced", "complete", or "r".
//
// Returns:
//   A result containing two MlirOps:
//     - result[0]: Q after decomposition.
//     - result[1]: R after decomposition.
absl::StatusOr<MlirOpResults<2>> BuildQrShlo(mlir::MlirOp input,
                                             c10::string_view mode);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_LINALG_QR_QR_LIB_H_
