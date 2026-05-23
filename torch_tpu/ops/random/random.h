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

#ifndef TORCH_TPU_OPS_RANDOM_RANDOM_H_
#define TORCH_TPU_OPS_RANDOM_RANDOM_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

absl::StatusOr<mlir::MlirOp> RandomBitsToUniform(
    mlir::MlirOp random_bits, mlir::ElementType element_type);

absl::StatusOr<MlirOpResults<2>> BuildRandomShlo(mlir::MlirOp rng_input_state,
                                                 Dimensions dims,
                                                 mlir::ElementType output_dtype,
                                                 int64_t from, int64_t to);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_RANDOM_RANDOM_H_
