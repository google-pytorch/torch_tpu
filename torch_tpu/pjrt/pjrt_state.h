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

#ifndef TORCH_TPU_PJRT_PJRT_STATE_H_
#define TORCH_TPU_PJRT_PJRT_STATE_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ATen/core/ATen_fwd.h"
#include "c10/core/Device.h"
#include "torch_tpu/eager/device_types.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/framework/allocator.h"

namespace torch_tpu {

// Returns the PjRtClient singleton. Triggers lazy initialization if needed.
[[nodiscard]] xla::PjRtClient* GetPjRtClient();

// Returns the PjRtDevice singleton. Triggers lazy initialization if needed.
[[nodiscard]] xla::PjRtDevice* GetPjRtDevice();

// Sets the PjRt device.
void SetPjRtDevice(xla::PjRtDevice* device);

// Returns whether the PjRt runtime is initialized.
[[nodiscard]] bool IsPjRtInitialized();

// Returns the global device count. Triggers lazy initialization if needed.
absl::StatusOr<int> GetGlobalDeviceCount();

// Returns the PjRt device type.
[[nodiscard]] PjRtDeviceType GetPjRtDeviceType();

// Sets the PjRt backend name for implicit initialization.
void SetPjRtBackendName(const std::string& device_type);

// Returns the PjRt backend name.
[[nodiscard]] std::string GetPjRtBackendName();

// Internal function to set the initialized state.
void SetPjRtState(xla::PjRtClient* client, xla::PjRtDevice* device,
                  PjRtDeviceType device_type, int global_device_count);

// Returns allocator stats for the current PjRt device.
absl::StatusOr<tsl::AllocatorStats> GetAllocatorStats();

// Returns the PjRt host allocator from the PjRt client.
absl::StatusOr<xla::PjRtClient::HostAllocator*> GetHostAllocator();

absl::Status MarkStreamActive(const at::Tensor& tensor,
                              xla::Future<void> future);

// Blocks until all pending operations on the given device have completed.
void SynchronizeStream(c10::DeviceIndex device_index);

// Blocks until pending operations on the given tensor have completed.
absl::Status SynchronizeHostPinnedTensors(
    const std::vector<at::Tensor>& tensors);

}  // namespace torch_tpu

#endif  // THIRD_PARTY_PY_TORCH_TPU_PJRT_PJRT_STATE_H_
