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

#ifndef TORCH_TPU_CSRC_EAGER_DEVICE_ASSIGNMENT_H_
#define TORCH_TPU_CSRC_EAGER_DEVICE_ASSIGNMENT_H_

#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

namespace torch_tpu {

// The devices an op runs on.
struct DeviceAssignment {
  DeviceAssignment() = default;
  // An op that runs only on its own devices.
  /* implicit */ DeviceAssignment(  // NOLINT(google-explicit-constructor)
      std::vector<int64_t> local)   // INT_VEC_OK
      : local(std::move(local)) {}
  /* implicit */ DeviceAssignment(  // NOLINT(google-explicit-constructor)
      std::initializer_list<int64_t> local)
      : local(local) {}
  DeviceAssignment(std::vector<int64_t> local,   // INT_VEC_OK
                   std::vector<int64_t> global,  // INT_VEC_OK
                   bool contains_pallas_kernel = false)
      : local(std::move(local)),
        global(std::move(global)),
        contains_pallas_kernel(contains_pallas_kernel) {}

  // The op's devices, by rank.
  std::vector<int64_t> local;  // INT_VEC_OK
  // For a collective of a process group: the devices of every group of the
  // layout the group was created in, group by group, each in group-rank order;
  // in a program that combines process groups, the collective is tiled over
  // them. Empty otherwise.
  std::vector<int64_t> global;  // INT_VEC_OK
  // Whether this op is (or contains, e.g. in a scan body) a Pallas kernel with
  // a fixed device assignment that cannot be overridden.
  bool contains_pallas_kernel = false;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_DEVICE_ASSIGNMENT_H_
