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

#ifndef TORCH_TPU_CSRC_DISTRIBUTED_UTILS_H_
#define TORCH_TPU_CSRC_DISTRIBUTED_UTILS_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/distributed/types.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

// Virtual multi-host (TORCH_TPU_VIRTUAL_MULTIHOST, pjrt_state.h WorldConfig):
// a torch world of N ranks runs on K per-host PJRT slices that cannot talk to
// each other. A process group whose members span slices is "folded": its
// collectives are compiled for the physical devices of the local slice only,
// and their outputs are padded to the logical group's shapes. Results are
// shape-correct, never numerically meaningful across slices.
struct VirtualMultiHostFold {
  // The device ids the group's collectives run on (one per physical member,
  // in process-group rank order of the first slice).
  DeviceGroup physical_device_ids;
  // logical group size / physical group size (1: nothing to fold).
  int64_t fold = 1;
};

// Folds a process group's member device ids (one per process-group rank, in
// rank order) when they repeat the same per-slice devices: `member_device_ids`
// must be `physical_device_ids` tiled `fold` times, `physical_device_ids`
// having no duplicates. Groups confined to one slice fold to themselves
// (`fold == 1`). Any other layout (interleaved slices, partial tiles, empty
// groups) is an invalid-argument error, so a misconfigured world fails loudly
// instead of compiling for the wrong devices.
absl::StatusOr<VirtualMultiHostFold> FoldVirtualMultiHostGroup(
    const DeviceGroup& member_device_ids);

// The distinct device ids of `device_ids` in order of first appearance: the
// local slice's devices behind a virtual multi-host world's device assignment
// (which lists every slice's devices once per subgroup).
DeviceGroup UniqueDeviceIds(const DeviceGroup& device_ids);

// The groups a folded list's collectives actually run on: each group's first
// `size / virtual_fold()` partitions. The identity for `virtual_fold() == 1`.
DeviceGroupList FoldReplicaGroups(const DeviceGroupList& device_groups);

// The size of a group the collectives are compiled for:
// `device_groups[0].size() / device_groups.virtual_fold()`.
int64_t PhysicalGroupSize(const DeviceGroupList& device_groups);

// Given device ids organized into (sub)groups, construct a MLIR constant
// used for `replica_groups` attribute in all the StableHLO collectives. A
// folded list (virtual multi-host) yields its physical groups.
mlir::DenseIntElementsAttr BuildReplicaGroupsAttr(
    mlir::MlirBuilder& builder, const DeviceGroupList& device_groups);

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

#endif  // TORCH_TPU_CSRC_DISTRIBUTED_UTILS_H_
