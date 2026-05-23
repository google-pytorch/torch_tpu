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

#ifndef TORCH_TPU_OPS_A_MIN_MAX_A_MIN_MAX_H_
#define TORCH_TPU_OPS_A_MIN_MAX_A_MIN_MAX_H_

#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/ops/reductions/reductions.h"

namespace torch_tpu {

enum class AMinMaxOp {
  kAmax,
  kAmin,
};

absl::StatusOr<mlir::MlirOp> BuildAMinMaxShlo(Dimensions dims,
                                              ReductionMode mode, AMinMaxOp op,
                                              mlir::MlirOp input_op);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_A_MIN_MAX_A_MIN_MAX_H_
