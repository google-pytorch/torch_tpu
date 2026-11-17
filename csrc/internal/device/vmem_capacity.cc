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

#include "csrc/internal/device/vmem_capacity.h"

#include <cstdint>
#include <string>

#include "absl/status/statusor.h"
#include "csrc/pjrt/pjrt_state.h"

namespace torch_tpu {

int64_t GetDeviceVmemCapacityBytes() {
  static const int64_t capacity_bytes = []() -> int64_t {
    // Default fallback: 16 MB
    int64_t capacity = 16777216;
    if (absl::StatusOr<PjRtDeviceAttributes> attrs =
            PjrtBackend::GetInstance().GetDeviceAttributes();
        attrs.ok()) {
      const std::string& device_kind = attrs->device_kind;
      if (device_kind == "TPU v5e" || device_kind == "TPU v5 lite") {
        capacity = 134217728;
      } else if (device_kind == "TPU v6e" || device_kind == "TPU v6 lite") {
        capacity = 134152192;
      } else if (device_kind == "TPU v5" || device_kind == "TPU v5p") {
        capacity = 67043328;
      } else if (device_kind == "TPU7" || device_kind == "TPU7x" ||
                 device_kind == "TPU 7" || device_kind == "TPU 7x" ||
                 device_kind == "TPU8i" || device_kind == "TPU8t" ||
                 device_kind == "TPU 8i" || device_kind == "TPU 8t") {
        capacity = 67043328;
      }
    }
    return capacity;
  }();
  return capacity_bytes;
}

}  // namespace torch_tpu
