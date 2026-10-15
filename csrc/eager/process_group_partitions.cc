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

#include "csrc/eager/process_group_partitions.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "csrc/common/error_utils.h"

namespace torch_tpu {
namespace {

struct Registry {
  absl::Mutex mu;
  absl::flat_hash_map<std::vector<int64_t>,  // INT_VEC_OK
                      std::vector<DeviceIdPartition>>
      by_device_assignment ABSL_GUARDED_BY(mu);
};

Registry& GetRegistry() {
  static absl::NoDestructor<Registry> registry;
  return *registry;
}

}  // namespace

void RegisterProcessGroupPartition(absl::Span<const int64_t> device_assignment,
                                   DeviceIdPartition partition) {
  // Every process then tiles collectives over the groups in the same order.
  absl::c_sort(partition);
  Registry& registry = GetRegistry();
  absl::MutexLock lock(&registry.mu);
  std::vector<DeviceIdPartition>& registered =
      registry.by_device_assignment[std::vector<int64_t>(  // INT_VEC_OK
          device_assignment.begin(), device_assignment.end())];
  if (!absl::c_linear_search(registered, partition)) {
    registered.push_back(std::move(partition));
  }
}

absl::StatusOr<std::optional<DeviceIdPartition>> LookupProcessGroupPartition(
    absl::Span<const int64_t> device_assignment) {
  const std::vector<int64_t> key(device_assignment.begin(),  // INT_VEC_OK
                                 device_assignment.end());
  Registry& registry = GetRegistry();
  absl::MutexLock lock(&registry.mu);
  auto it = registry.by_device_assignment.find(key);
  if (it == registry.by_device_assignment.end()) {
    return std::nullopt;
  }
  TT_RET_CHECK(it->second.size() == 1, error::kInvalidArgument)
      << "process groups on devices [" << absl::StrJoin(key, ", ")
      << "] were created in different layouts";
  return it->second.front();
}

void ClearProcessGroupPartitions() {
  Registry& registry = GetRegistry();
  absl::MutexLock lock(&registry.mu);
  registry.by_device_assignment.clear();
}

}  // namespace torch_tpu
