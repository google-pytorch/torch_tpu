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

#include "torch_tpu/runtime/eager_runtime_adapter.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/device_buffer.h"
#include "torch_tpu/runtime/eager_bridge.h"
#include "xla/tsl/framework/allocator.h"

namespace torch_tpu {

absl::StatusOr<std::unique_ptr<DeviceBuffer>>
EagerRuntimeAdapter::CreateBufferFromHost(const void* data,
                                          at::ScalarType dtype,
                                          absl::Span<const int64_t> shape,
                                          Device* device) {
  Backend* backend = Backend::GetInstance();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Backend singleton check
      backend != nullptr, error::kInternal)
      << "active backend instance is null";
  return backend->MakeBufferFromHost(data, dtype, shape, device);
}

at::Tensor EagerRuntimeAdapter::CreateTensorFromBuffer(
    std::unique_ptr<DeviceBuffer> buffer) {
  return CreateAtTensorFromTtrBuffer(std::move(buffer));
}

DeviceBuffer* EagerRuntimeAdapter::ExtractBufferFromTensor(
    const at::Tensor& tensor) {
  return GetTtrBufferFromAtTensor(tensor);
}

absl::Status EagerRuntimeAdapter::SynchronizeDevice(int device_id) {
  Backend* backend = Backend::GetInstance();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Backend singleton check
      backend != nullptr, error::kInternal)
      << "active backend instance is null";
  return backend->SynchronizeDevice(device_id);
}

absl::Status EagerRuntimeAdapter::SynchronizeAllDevices() {
  Backend* backend = Backend::GetInstance();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Backend singleton check
      backend != nullptr, error::kInternal)
      << "active backend instance is null";
  return backend->SynchronizeAllDevices();
}

absl::StatusOr<tsl::AllocatorStats> EagerRuntimeAdapter::GetAllocatorStats() {
  Backend* backend = Backend::GetInstance();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Backend singleton check
      backend != nullptr, error::kInternal)
      << "active backend instance is null";
  return backend->GetAllocatorStats();
}

absl::Status EagerRuntimeAdapter::ClearAllocatorStats() {
  Backend* backend = Backend::GetInstance();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Backend singleton check
      backend != nullptr, error::kInternal)
      << "active backend instance is null";
  return backend->ClearAllocatorStats();
}

Device* EagerRuntimeAdapter::GetDefaultDevice() {
  Backend* backend = Backend::GetInstance();
  return (backend != nullptr) ? backend->GetDefaultDevice() : nullptr;
}

absl::StatusOr<int> EagerRuntimeAdapter::GetGlobalDeviceId() {
  Backend* backend = Backend::GetInstance();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Backend singleton check
      backend != nullptr, error::kInternal)
      << "active backend instance is null";
  return backend->GetGlobalDeviceId();
}

absl::StatusOr<int> EagerRuntimeAdapter::GetGlobalDeviceCount() {
  Backend* backend = Backend::GetInstance();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Backend singleton check
      backend != nullptr, error::kInternal)
      << "active backend instance is null";
  return backend->GetGlobalDeviceCount();
}

}  // namespace torch_tpu
