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

#ifndef TORCH_TPU_CSRC_OPS_DROPOUT_DROPOUT_H_
#define TORCH_TPU_CSRC_OPS_DROPOUT_DROPOUT_H_

#include "absl/status/statusor.h"
#include "csrc/ops/op_builder_utils.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

enum class DropoutProbabilityType {
  kDropProbability,
  kKeepProbability,
};

enum class DropoutMaskType {
  kBool,
  kUInt8,
};

// Builds the SHLO for dropout in the train mode. When train=False, dropout is
// a no-op, so doesn't require a builder.
//
// Returns:
//   A pair of MlirOpResults, the first for the output tensor and the second for
//   the mask tensor.
absl::StatusOr<MlirOpResults<2>> BuildDropoutTrainShlo(
    mlir::MlirOp rng_input_state, mlir::MlirOp input, double p,
    DropoutProbabilityType prob_type = DropoutProbabilityType::kDropProbability,
    DropoutMaskType mask_type = DropoutMaskType::kBool);

// Builds the SHLO for dropout backward (`native_dropout_backward` and
// `_masked_scale`). Intentional divergences from CUDA `masked_scale_kernel`:
// non-boolean (`uint8`) masks are treated as binary (`mask != 0`) rather than
// integer multipliers, and masked-out elements select `0.0` even when `scale`
// is `inf` (e.g. `fused_dropout` backward at `p=0.0`) or `grad_output` is
// `inf`/`nan`.
absl::StatusOr<MlirOpResults<1>> BuildDropoutBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp mask, double scale);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_DROPOUT_DROPOUT_H_
