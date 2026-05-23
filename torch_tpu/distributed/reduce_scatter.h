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

#ifndef TORCH_TPU_DISTRIBUTED_REDUCE_SCATTER_H_
#define TORCH_TPU_DISTRIBUTED_REDUCE_SCATTER_H_

#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch_tpu/distributed/types.h"

namespace torch_tpu {

absl::StatusOr<mlir::MlirOp> BuildDistributedReduceScatterShlo(
    mlir::MlirOp input, c10d::ReduceOp::RedOpType reduce_op_type,
    const DeviceGroupList& device_groups);

}  // namespace torch_tpu

#endif  // TORCH_TPU_DISTRIBUTED_REDUCE_SCATTER_H_
