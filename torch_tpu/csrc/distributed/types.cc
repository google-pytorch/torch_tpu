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

#include "torch_tpu/csrc/distributed/types.h"

#include <algorithm>
#include <initializer_list>
#include <ostream>
#include <utility>
#include <vector>

#include "torch_tpu/csrc/common/fingerprint_utils.h"
#include "torch_tpu/csrc/common/to_string.h"

namespace torch_tpu {
namespace {

// Converts `groups` to enforce the normalized format:
// The list of device groups is sorted in lexicographical order and
// deduplicated.
std::vector<DeviceGroup> Normalize(std::vector<DeviceGroup> groups) {
  // We use std::vector instead of std::set because device group lists are
  // immutable after construction. std::vector provides contiguous memory
  // layout, better cache locality, O(1) indexing, zero node-allocation
  // overhead compared to pointer-linked tree nodes in std::set.
  std::sort(groups.begin(), groups.end());
  groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
  return groups;
}

}  // namespace

DeviceGroupList::DeviceGroupList(std::vector<DeviceGroup> groups)
    : groups_(Normalize(std::move(groups))),
      fingerprint_(Fingerprint(groups_)) {}

DeviceGroupList::DeviceGroupList(std::initializer_list<DeviceGroup> groups)
    : DeviceGroupList(std::vector<DeviceGroup>(groups)) {}

std::ostream& operator<<(std::ostream& os, const DeviceGroupList& list) {
  return os << ToString(list.groups());
}

}  // namespace torch_tpu
