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

#ifndef TORCH_TPU_DISTRIBUTED_ALLGATHER_H_
#define TORCH_TPU_DISTRIBUTED_ALLGATHER_H_

#include "absl/status/statusor.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/distributed/types.h"

namespace torch_tpu {

enum class AllGatherOutputMode {
  kList,    // Output a list of tensors.
  kConcat,  // Output a single tensor concatenated along dim 0.
  kStack    // Output a single tensor stacked along the new dim 0.
};

absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> BuildDistributedAllGatherShlo(
    mlir::MlirOp input, const DeviceGroupList& device_groups,
    AllGatherOutputMode output_mode = AllGatherOutputMode::kList);

}  // namespace torch_tpu

#endif  // TORCH_TPU_DISTRIBUTED_ALLGATHER_H_
