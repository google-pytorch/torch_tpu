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

#ifndef TORCH_TPU_OPS_DROPOUT_DROPOUT_H_
#define TORCH_TPU_OPS_DROPOUT_DROPOUT_H_

#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

// Builds the SHLO for dropout in the train mode. When train=False, dropout is
// a no-op, so doesn't require a builder.
absl::StatusOr<MlirOpResults<3>> BuildDropoutTrainShlo(
    mlir::MlirOp rng_input_state, mlir::MlirOp input, double p);

// Builds the SHLO for dropout backward.
absl::StatusOr<MlirOpResults<1>> BuildDropoutBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp mask, double scale);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_DROPOUT_DROPOUT_H_
