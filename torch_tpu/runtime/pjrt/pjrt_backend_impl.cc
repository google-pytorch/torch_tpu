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

#include "torch_tpu/runtime/pjrt/pjrt_backend_impl.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/device_buffer.h"
#include "torch_tpu/runtime/executable.h"
#include "torch_tpu/runtime/pjrt/pjrt_device_buffer.h"
#include "torch_tpu/runtime/pjrt/pjrt_executable.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/framework/allocator.h"

namespace torch_tpu {

PjrtDevice::PjrtDevice(xla::PjRtDevice* pjrt_device)
    : pjrt_device_(pjrt_device) {}

int PjrtDevice::id() const {
  return pjrt_device_ != nullptr ? pjrt_device_->id() : -1;
}

std::string PjrtDevice::kind() const {
  return pjrt_device_ != nullptr ? std::string(pjrt_device_->device_kind())
                                 : "Unknown";
}

std::string PjrtDevice::to_string() const {
  return pjrt_device_ != nullptr ? std::string(pjrt_device_->DebugString())
                                 : "PjrtDevice(null)";
}

bool PjrtDevice::is_addressable() const {
  return pjrt_device_ != nullptr && pjrt_device_->IsAddressable();
}

PjrtBackendImpl::PjrtBackendImpl(const BackendOptions& options)
    : options_(options), client_(nullptr) {}

xla::PjRtClient* PjrtBackendImpl::GetPjrtClient() const {
  absl::MutexLock lock(mutex_);
  if (client_ == nullptr && PjrtBackend::GetInstance().IsInitialized()) {
    client_ = PjrtBackend::GetInstance().GetClient();
  }
  return client_;
}

void PjrtBackendImpl::EnsureDevicesInitialized() const {
  absl::MutexLock lock(mutex_);
  if (!devices_ptrs_.empty()) {
    return;
  }
  if (client_ == nullptr && PjrtBackend::GetInstance().IsInitialized()) {
    client_ = PjrtBackend::GetInstance().GetClient();
  }
  if (client_ != nullptr) {
    for (xla::PjRtDevice* dev : client_->devices()) {
      auto pjrt_dev = std::make_unique<PjrtDevice>(dev);
      devices_ptrs_.push_back(pjrt_dev.get());
      if (dev->IsAddressable()) {
        addressable_devices_ptrs_.push_back(pjrt_dev.get());
      }
      owned_devices_.push_back(std::move(pjrt_dev));
    }
  }
}

absl::Span<Device* const> PjrtBackendImpl::GetDevices() const {
  EnsureDevicesInitialized();
  absl::MutexLock lock(mutex_);
  return devices_ptrs_;
}

absl::Span<Device* const> PjrtBackendImpl::GetAddressableDevices() const {
  EnsureDevicesInitialized();
  absl::MutexLock lock(mutex_);
  return addressable_devices_ptrs_;
}

absl::StatusOr<int> PjrtBackendImpl::GetGlobalDeviceId() const {
  xla::PjRtClient* c = GetPjrtClient();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal runtime null check
      c != nullptr, error::kInternal)
      << "pjrt client is not initialized";
  return PjrtBackend::GetInstance().GetGlobalDeviceId();
}

absl::StatusOr<int> PjrtBackendImpl::GetGlobalDeviceCount() const {
  xla::PjRtClient* c = GetPjrtClient();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal runtime null check
      c != nullptr, error::kInternal)
      << "pjrt client is not initialized";
  return c->device_count();
}

Device* PjrtBackendImpl::GetDefaultDevice() const {
  EnsureDevicesInitialized();
  absl::MutexLock lock(mutex_);
  return addressable_devices_ptrs_.empty() ? nullptr
                                           : addressable_devices_ptrs_.front();
}

absl::StatusOr<tsl::AllocatorStats> PjrtBackendImpl::GetAllocatorStats() {
  xla::PjRtClient* c = GetPjrtClient();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal runtime null check
      c != nullptr, error::kInternal)
      << "pjrt client is uninitialized";
  return PjrtBackend::GetInstance().GetAllocatorStats();
}

absl::Status PjrtBackendImpl::ClearAllocatorStats() {
  xla::PjRtClient* c = GetPjrtClient();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal runtime null check
      c != nullptr, error::kInternal)
      << "pjrt client is uninitialized";
  return PjrtBackend::GetInstance().ClearAllocatorStats();
}

absl::StatusOr<xla::HostMemoryAllocator*> PjrtBackendImpl::GetHostAllocator() {
  xla::PjRtClient* c = GetPjrtClient();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal runtime null check
      c != nullptr, error::kInternal)
      << "pjrt client is uninitialized";
  return PjrtBackend::GetInstance().GetHostAllocator();
}

absl::Status PjrtBackendImpl::SynchronizeDevice(int device_id) {
  return absl::OkStatus();
}

absl::Status PjrtBackendImpl::SynchronizeAllDevices() {
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<DeviceBuffer>>
PjrtBackendImpl::MakeBufferFromHost(const void* data, at::ScalarType dtype,
                                    absl::Span<const int64_t> shape,
                                    Device* device) {
  xla::PjRtClient* c = GetPjrtClient();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal runtime null check
      c != nullptr, error::kInternal)
      << "pjrt client is uninitialized";

  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=Internal dtype conversion
      xla::PrimitiveType xla_type, ConvertTo<xla::PrimitiveType>(dtype));

  xla::PjRtDevice* target_pjrt_dev = nullptr;
  if (device != nullptr) {
    auto* pjrt_dev_wrapper = dynamic_cast<PjrtDevice*>(device);
    if (pjrt_dev_wrapper != nullptr) {
      target_pjrt_dev = pjrt_dev_wrapper->pjrt_device();
    }
  }
  EnsureDevicesInitialized();
  absl::MutexLock lock(mutex_);
  if (target_pjrt_dev == nullptr && !addressable_devices_ptrs_.empty()) {
    target_pjrt_dev =
        static_cast<PjrtDevice*>(addressable_devices_ptrs_.front())
            ->pjrt_device();
  }

  Dimensions dims(shape.begin(), shape.end());
  Strides byte_strides(dims.size());
  int64_t stride = c10::elementSize(dtype);
  for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
    byte_strides[i] = stride;
    stride *= dims[i];
  }

  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=Internal PJRT buffer creation
      auto raw_buffer,
      c->BufferFromHostBuffer(
          data, xla_type, dims, byte_strides,
          xla::PjRtClient::HostBufferSemantics::
              kImmutableUntilTransferCompletes,
          /*on_done_with_host_buffer=*/nullptr,
          /*memory_space=*/static_cast<xla::PjRtMemorySpace*>(nullptr),
          /*device_layout=*/nullptr));

  std::shared_ptr<xla::PjRtBuffer> shared_buf(std::move(raw_buffer));
  return std::make_unique<PjrtDeviceBuffer>(std::move(shared_buf), dtype,
                                            device);
}

absl::StatusOr<Compiler*> PjrtBackendImpl::GetCompiler() {
  xla::PjRtClient* c = GetPjrtClient();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal runtime null check
      c != nullptr, error::kInternal)
      << "pjrt client is uninitialized";
  absl::MutexLock lock(mutex_);
  if (compiler_ == nullptr) {
    compiler_ = std::make_unique<PjrtCompiler>(c);
  }
  return compiler_.get();
}

}  // namespace torch_tpu
