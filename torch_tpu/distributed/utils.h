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

#ifndef TORCH_TPU_DISTRIBUTED_UTILS_H_
#define TORCH_TPU_DISTRIBUTED_UTILS_H_

#include "absl/status/status.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/distributed/types.h"

namespace torch_tpu {

// Given device ids organized into (sub)groups, construct a MLIR constant
// used for `replica_groups` attribute in all the StableHLO collectives.
mlir::DenseIntElementsAttr BuildReplicaGroupsAttr(
    mlir::MlirBuilder& builder, const DeviceGroupList& device_groups);

// Returns the next channel handle for a collective op emitted into the
// module currently being built, and advances the counter. Handles start at 1
// and follow emission order, which matches the deferred-graph execution order
// on every rank; giving each collective instance its own channel makes the
// cross-device rendezvous independent of the per-program instruction
// schedule (with a shared handle, devices pair collectives purely by launch
// order, and two ranks whose executables schedule the same logical
// collectives differently exchange data between the wrong ops).
int NextCollectiveChannelHandle();

// Resets the emission counter. Called at the start of building each MLIR
// module so that handle assignment restarts at 1 for every program.
void ResetCollectiveChannelCounter();

// Given a channel handle and type, construct a MLIR constant used for the
// `channel` attribute in StableHLO collectives.
//
// Channels are generally used to allow multiple concurrent communication
// operations between same device pairs (see
// https://openxla.org/stablehlo/spec#send). They are also used to control the
// interpretation of device groups (see
// https://openxla.org/stablehlo/spec#all_gather).
mlir::stablehlo::ChannelHandleAttr BuildChannelHandleAttr(
    mlir::MlirBuilder& builder, int handle, int type);

// Enter SPMD safe region by setting the top of the python thread stack
// context to true.
void EnterSpmdSafeRegion();

// Exit the SPMD safe region by popping the top of the python thread stack.
void ExitSpmdSafeRegion();

// Returns true if the current execution is within a SPMD safe region.
[[nodiscard]] bool IsSpmdSafe();

absl::Status ValidateReductionOp(c10d::ReduceOp reduce_op,
                                 c10::ScalarType scalar_type);

}  // namespace torch_tpu

#endif  // TORCH_TPU_DISTRIBUTED_UTILS_H_
