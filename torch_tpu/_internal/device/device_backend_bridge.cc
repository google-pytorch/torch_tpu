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

#include "torch_tpu/_internal/device/device_backend_bridge.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/backend_config.h"

namespace torch_tpu {

int DeviceBackendBridge::GetDeviceCount() { return GetLocalDeviceCount(); }

int DeviceBackendBridge::GetLocalDeviceCount() {
  Backend* backend = Backend::GetInstance();
  if (backend == nullptr) {
    return 0;
  }
  // Single-host Linux process on TPU architecture can address at most 8 cores.
  constexpr int kMaxLocalDevices = 8;
  return std::min(static_cast<int>(backend->GetDevices().size()),
                  kMaxLocalDevices);
}

absl::StatusOr<int> DeviceBackendBridge::GetGlobalDeviceCount() {
  Backend* backend = Backend::GetInstance();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Backend pointer check
      backend != nullptr, error::kInternal)
      << "global backend instance is null";
  return backend->GetGlobalDeviceCount();
}

std::vector<BackendDeviceInfo> DeviceBackendBridge::GetAddressableDevices() {
  std::vector<BackendDeviceInfo> result;
  Backend* backend = Backend::GetInstance();
  if (backend == nullptr) {
    return result;
  }
  auto devices = backend->GetDevices();
  result.reserve(devices.size());
  for (Device* dev : devices) {
    if (dev == nullptr) {
      continue;
    }
    BackendDeviceInfo info;
    info.id = dev->id();
    info.kind = dev->kind();
    info.description = dev->to_string();
    if (auto coords_opt = dev->coords(); coords_opt.has_value()) {
      info.coords = *coords_opt;
    }
    result.push_back(std::move(info));
  }
  return result;
}

std::string DeviceBackendBridge::GetActiveBackendKind() {
  BackendKind kind = GetGlobalBackendConfig().default_backend;
  switch (kind) {
    case BackendKind::kPjrt:
      return "pjrt";
    case BackendKind::kCerebro:
      return "cerebro";
    case BackendKind::kMock:
      return "mock";
  }
  return "unknown";
}

}  // namespace torch_tpu
