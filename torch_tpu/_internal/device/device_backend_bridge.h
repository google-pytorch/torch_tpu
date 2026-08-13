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

#ifndef TORCH_TPU_INTERNAL_DEVICE_DEVICE_BACKEND_BRIDGE_H_
#define TORCH_TPU_INTERNAL_DEVICE_DEVICE_BACKEND_BRIDGE_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "torch_tpu/runtime/backend.h"

namespace torch_tpu {

// Struct describing an addressable device in the active backend.
struct BackendDeviceInfo {
  int id;
  std::string kind;
  std::string description;
  std::vector<int64_t> coords;  // INT_VEC_OK
};

// Provides device discovery and topology routing through the active TTR
// backend.
class DeviceBackendBridge {
 public:
  // Returns number of local addressable devices for this host process.
  static int GetLocalDeviceCount();

  // Returns number of local addressable devices in the active backend.
  static int GetDeviceCount();

  // Returns total number of global devices in the cluster.
  static absl::StatusOr<int> GetGlobalDeviceCount();

  // Returns metadata list of all local addressable devices.
  static std::vector<BackendDeviceInfo> GetAddressableDevices();

  // Returns active backend kind string ("pjrt", "cerebro", "mock").
  static std::string GetActiveBackendKind();
};

}  // namespace torch_tpu

#endif  // THIRD_PARTY_PY_TORCH_TPU__INTERNAL_DEVICE_DEVICE_BACKEND_BRIDGE_H_
