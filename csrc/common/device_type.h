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

#ifndef TORCH_TPU_COMMON_DEVICE_TYPE_H_
#define TORCH_TPU_COMMON_DEVICE_TYPE_H_

#include <string>

#include "torch/headeronly/core/DeviceType.h"

namespace torch_tpu {

// The actual physical device type used by PjRt.
// PyTorch always sees TorchTPU as the PrivateUse1 device, but TorchTPU can
// be used for any XLA device type with a PjRt plugin.
// However, some kernel implementations may only work on some specific physical
// XLA device types.
enum class PjRtDeviceType {
  kUnknown = 0,
  kCpu,
  kCuda,
  kTpu,
};

// Returns the device type for TPU.
[[nodiscard]] inline constexpr c10::DeviceType GetPrivateUse1DeviceType() {
  return c10::DeviceType::PrivateUse1;
}

// Returns a user-friendly name for the PrivateUse1 device. Mainly useful for
// constructing user-facing error messages. Instead of hard-coding "tpu", call
// this function to get the device type to use in error messages so that we have
// the flexibility to support other hardware types in the future if we want to.
[[nodiscard]] inline std::string GetPrivateUse1DeviceDebugName() {
  return "tpu";
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_DEVICE_TYPE_H_
