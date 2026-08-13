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

#ifndef TORCH_TPU_RUNTIME_EAGER_RUNTIME_ADAPTER_H_
#define TORCH_TPU_RUNTIME_EAGER_RUNTIME_ADAPTER_H_

#include <cstdint>
#include <memory>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/device_buffer.h"
#include "xla/tsl/framework/allocator.h"

namespace torch_tpu {

// EagerRuntimeAdapter bridges eager PyTorch tensor execution with the active
// TorchTPU Runtime (TTR) backend singleton.
class EagerRuntimeAdapter {
 public:
  // Transfers tensor data from host memory to an on-device TTR DeviceBuffer
  // using the currently active backend instance.
  static absl::StatusOr<std::unique_ptr<DeviceBuffer>> CreateBufferFromHost(
      const void* data, at::ScalarType dtype, absl::Span<const int64_t> shape,
      Device* device = nullptr);

  // Wraps an on-device TTR DeviceBuffer inside a PyTorch at::Tensor.
  static at::Tensor CreateTensorFromBuffer(
      std::unique_ptr<DeviceBuffer> buffer);

  // Extracts the underlying TTR DeviceBuffer from a PyTorch at::Tensor.
  static DeviceBuffer* ExtractBufferFromTensor(const at::Tensor& tensor);

  // Synchronizes the specified device on the active backend.
  static absl::Status SynchronizeDevice(int device_id = 0);

  // Synchronizes all available devices on the active backend.
  static absl::Status SynchronizeAllDevices();

  // Queries memory allocation statistics from the active backend.
  static absl::StatusOr<tsl::AllocatorStats> GetAllocatorStats();

  // Clears memory allocation statistics on the active backend.
  static absl::Status ClearAllocatorStats();

  // Gets the active default device from the active backend.
  static Device* GetDefaultDevice();

  // Gets the global device ID from the active backend.
  static absl::StatusOr<int> GetGlobalDeviceId();

  // Gets the global device count from the active backend.
  static absl::StatusOr<int> GetGlobalDeviceCount();
};

}  // namespace torch_tpu

#endif  // THIRD_PARTY_PY_TORCH_TPU_RUNTIME_EAGER_RUNTIME_ADAPTER_H_
