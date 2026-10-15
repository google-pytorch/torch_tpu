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

#ifndef TORCH_TPU_CSRC_EAGER_PROCESS_GROUP_PARTITIONS_H_
#define TORCH_TPU_CSRC_EAGER_PROCESS_GROUP_PARTITIONS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace torch_tpu {

// The partition of a process group: every group of the layout the group was
// created in (e.g. all tensor-parallel groups), each as the device ids of its
// members in group-rank order. Together the groups cover every device once.
using DeviceIdPartition = std::vector<std::vector<int64_t>>;  // INT_VEC_OK

// Registers `partition` as the partition of the process group whose
// collectives carry `device_assignment` (its members' device ids in group-rank
// order). The order of the groups does not matter. Registering the same
// partition again has no effect.
void RegisterProcessGroupPartition(absl::Span<const int64_t> device_assignment,
                                   DeviceIdPartition partition);

// Returns the partition registered for `device_assignment`, or std::nullopt if
// no process group registered one. Returns an error if process groups with the
// same device assignment registered different partitions.
absl::StatusOr<std::optional<DeviceIdPartition>> LookupProcessGroupPartition(
    absl::Span<const int64_t> device_assignment);

// Forgets every registered partition. For tests.
void ClearProcessGroupPartitions();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_PROCESS_GROUP_PARTITIONS_H_
