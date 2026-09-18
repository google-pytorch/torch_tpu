// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TORCH_TPU_CSRC_COMMON_COLLECTIVE_SUBGROUP_H_
#define TORCH_TPU_CSRC_COMMON_COLLECTIVE_SUBGROUP_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/client/executable_build_options.h"

namespace torch_tpu {

// Programs whose collectives all run over one proper subgroup of the slice's
// devices are compiled for that subgroup alone: only its devices launch and
// synchronize on the program, so the other devices may run unrelated programs
// concurrently. A collective in such a program is emitted with the slice-wide
// `replica_groups` (every subgroup, in logical device ids: a device's world
// rank, its index in the slice-wide device assignment); these helpers
// restrict the executable to the subgroup's devices and renumber the groups to
// the executable's own device indices.

// Returns the one device group (logical device ids) shared by every entry of
// `groups`, skipping empty entries. The result is empty when no entry names a
// group, and when two entries name different groups: such a program keeps the
// slice-wide device assignment, under which every device launches it.
std::vector<int64_t> CommonCollectiveDeviceGroup(    // INT_VEC_OK
    absl::Span<const std::vector<int64_t>> groups);  // INT_VEC_OK

// Restricts `options` to the devices of `group` (logical device ids, in group
// order). `global_device_ids[i]` is the PJRT global device id of logical
// device i. With `replicated` false the program has one replica and one
// partition per group member, matching the default multi-partition layout;
// with `replicated` true it has one replica per member and one partition
// (the layout used for SparseCore programs). Either way the flattened device
// index of member j is j, which is what the rewritten replica groups use.
absl::Status RestrictBuildOptionsToDeviceGroup(
    xla::ExecutableBuildOptions& options, absl::Span<const int64_t> group,
    absl::Span<const int64_t> global_device_ids, bool replicated);

// Renumbers every collective in `module` for an executable over `group`:
// a `replica_groups` attribute must list `group` (in order) as one of its
// rows and becomes the single row [0, 1, ..., n-1]; `source_target_pairs`
// entries inside the group are mapped to member indices and entries between
// other devices are dropped. A row that overlaps the group only partly, a
// permute pair that crosses the group boundary, or a collective that does not
// involve the group at all are errors.
absl::Status RewriteCollectivesForDeviceGroup(mlir::ModuleOp module,
                                              absl::Span<const int64_t> group);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_COLLECTIVE_SUBGROUP_H_
