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

#include "torch_tpu/ops/hardtanh/hardtanh.h"

#include "absl/status/statusor.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildHardtanhShlo(mlir::MlirOp input_op,
                                               mlir::MlirOp min_val_op,
                                               mlir::MlirOp max_val_op) {
  TT_ASSIGN_OR_RETURN((auto [input_, min_val_, max_val_]),
                      ApplyBroadcastIfNeeded(input_op, min_val_op, max_val_op));
  return stablehlo::Clamp(min_val_, input_, max_val_);
}

}  // namespace torch_tpu
