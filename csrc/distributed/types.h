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

#ifndef TORCH_TPU_CSRC_DISTRIBUTED_TYPES_H_
#define TORCH_TPU_CSRC_DISTRIBUTED_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <ostream>
#include <vector>

#include "csrc/common/fingerprint_utils.h"

namespace torch_tpu {

// TODO(b/445264222) Create a more purpose-specific type for device groups
using DeviceGroup = std::vector<int64_t>;  // INT_VEC_OK

// Represents a normalized list of TPU device groups for distributed collective
// operations.
//
// Invariant Guarantees:
// The list of device groups is sorted in lexicographical order and
// deduplicated.
//
// This guarantees that any two `DeviceGroupList` instances representing
// equivalent device group configurations will compare equal and produce
// identical fingerprints.
//
// Immutability:
// Once constructed, a `DeviceGroupList` is immutable (except via assignment
// operations). All accessors return const references or const iterators to
// maintain normalized invariants.
class DeviceGroupList {
 public:
  DeviceGroupList() = default;

  explicit DeviceGroupList(std::vector<DeviceGroup> groups);

  // Constructs from initializer list of DeviceGroups.
  DeviceGroupList(std::initializer_list<DeviceGroup> groups);

  const std::vector<DeviceGroup>& groups() const { return groups_; }

  FingerprintType fingerprint() const { return fingerprint_; }

  size_t size() const { return groups_.size(); }
  bool empty() const { return groups_.empty(); }

  const DeviceGroup& operator[](size_t index) const { return groups_[index]; }
  const DeviceGroup& at(size_t index) const { return groups_.at(index); }

  auto begin() const { return groups_.begin(); }
  auto end() const { return groups_.end(); }
  auto cbegin() const { return groups_.cbegin(); }
  auto cend() const { return groups_.cend(); }

  bool operator==(const DeviceGroupList& other) const {
    return groups_ == other.groups_;
  }
  bool operator!=(const DeviceGroupList& other) const {
    return groups_ != other.groups_;
  }
  bool operator<(const DeviceGroupList& other) const {
    return groups_ < other.groups_;
  }

 private:
  // Invariant:
  // The list of device groups is sorted in lexicographical order and
  // deduplicated.
  std::vector<DeviceGroup> groups_;
  // Fingerprint of `groups_`.
  FingerprintType fingerprint_ = 0;
};

std::ostream& operator<<(std::ostream& os, const DeviceGroupList& list);

[[nodiscard]] inline FingerprintType EncodeParamCacheKey(
    const DeviceGroupList& list) {
  return Fingerprint(list);
}

namespace internal {
template <>
struct Fingerprint64Impl<DeviceGroupList, /*kIsSmallIntegral=*/false> {
  [[nodiscard]] static FingerprintType Compute(const DeviceGroupList& list) {
    return list.fingerprint();
  }
};
}  // namespace internal

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_DISTRIBUTED_TYPES_H_
