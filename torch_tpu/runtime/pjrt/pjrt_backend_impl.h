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

#ifndef TORCH_TPU_RUNTIME_PJRT_PJRT_BACKEND_IMPL_H_
#define TORCH_TPU_RUNTIME_PJRT_PJRT_BACKEND_IMPL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/device_buffer.h"
#include "torch_tpu/runtime/executable.h"
#include "torch_tpu/runtime/pjrt/pjrt_executable.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/framework/allocator.h"

namespace torch_tpu {

class PjrtDevice final : public Device {
 public:
  explicit PjrtDevice(xla::PjRtDevice* pjrt_device);
  ~PjrtDevice() override = default;

  [[nodiscard]] int id() const override;
  [[nodiscard]] std::string kind() const override;
  [[nodiscard]] std::string to_string() const override;
  [[nodiscard]] bool is_addressable() const override;

  [[nodiscard]] xla::PjRtDevice* pjrt_device() const { return pjrt_device_; }

 private:
  xla::PjRtDevice* pjrt_device_;
};

class PjrtBackendImpl final : public Backend {
 public:
  explicit PjrtBackendImpl(const BackendOptions& options);
  ~PjrtBackendImpl() override = default;

  [[nodiscard]] absl::Span<Device* const> GetDevices() const override;
  [[nodiscard]] absl::Span<Device* const> GetAddressableDevices()
      const override;
  absl::StatusOr<int> GetGlobalDeviceId() const override;
  absl::StatusOr<int> GetGlobalDeviceCount() const override;
  [[nodiscard]] Device* GetDefaultDevice() const override;

  absl::StatusOr<tsl::AllocatorStats> GetAllocatorStats() override;
  absl::Status ClearAllocatorStats() override;
  absl::StatusOr<xla::HostMemoryAllocator*> GetHostAllocator() override;

  absl::StatusOr<std::unique_ptr<DeviceBuffer>> MakeBufferFromHost(
      const void* data, at::ScalarType dtype, absl::Span<const int64_t> shape,
      Device* device) override;

  absl::StatusOr<Compiler*> GetCompiler() override;

  absl::Status SynchronizeDevice(int device_id) override;
  absl::Status SynchronizeAllDevices() override;

  [[nodiscard]] xla::PjRtClient* client() const { return GetPjrtClient(); }

 private:
  xla::PjRtClient* GetPjrtClient() const;
  void EnsureDevicesInitialized() const;

  BackendOptions options_;
  mutable xla::PjRtClient* client_ ABSL_GUARDED_BY(mutex_);
  mutable absl::Mutex mutex_;
  mutable std::vector<std::unique_ptr<PjrtDevice>> owned_devices_
      ABSL_GUARDED_BY(mutex_);
  mutable std::vector<Device*> devices_ptrs_ ABSL_GUARDED_BY(mutex_);
  mutable std::vector<Device*> addressable_devices_ptrs_
      ABSL_GUARDED_BY(mutex_);
  mutable std::unique_ptr<PjrtCompiler> compiler_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_RUNTIME_PJRT_PJRT_BACKEND_IMPL_H_
