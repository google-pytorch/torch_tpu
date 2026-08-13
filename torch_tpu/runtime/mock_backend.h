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

#ifndef TORCH_TPU_RUNTIME_MOCK_BACKEND_H_
#define TORCH_TPU_RUNTIME_MOCK_BACKEND_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/device_buffer.h"
#include "torch_tpu/runtime/executable.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/framework/allocator.h"

namespace torch_tpu {

class MockDevice final : public Device {
 public:
  explicit MockDevice(int id) : id_(id) {}
  ~MockDevice() override = default;

  [[nodiscard]] int id() const override { return id_; }
  [[nodiscard]] std::string kind() const override { return "MockTPU"; }
  [[nodiscard]] std::string to_string() const override {
    return "MockDevice(0)";
  }
  [[nodiscard]] bool is_addressable() const override { return true; }

 private:
  int id_;
};

class MockDeviceBuffer final : public DeviceBuffer {
 public:
  MockDeviceBuffer(at::ScalarType dtype, Dimensions shape, Device* device);
  ~MockDeviceBuffer() override = default;

  [[nodiscard]] at::ScalarType dtype() const override { return dtype_; }
  [[nodiscard]] absl::Span<const int64_t> shape() const override {
    return shape_;
  }
  [[nodiscard]] Device* device() const override { return device_; }
  [[nodiscard]] size_t GetOnDeviceSizeInBytes() const override {
    return storage_.size();
  }

  [[nodiscard]] tsl::Future<> GetReadyFuture() const override;
  absl::Status BlockHostUntilReady() override;
  tsl::Future<> CopyToHostAsync(void* dst, size_t size_in_bytes) override;
  absl::StatusOr<std::unique_ptr<DeviceBuffer>> CreateView(
      absl::Span<const int64_t> new_shape) override;

  [[nodiscard]] const std::vector<uint8_t>& storage() const { return storage_; }
  std::vector<uint8_t>& mutable_storage() { return storage_; }

 private:
  at::ScalarType dtype_;
  Dimensions shape_;
  Device* device_;
  std::vector<uint8_t> storage_;
};

class MockLoadedExecutable final : public LoadedExecutable {
 public:
  MockLoadedExecutable() = default;
  ~MockLoadedExecutable() override = default;

  absl::StatusOr<std::vector<std::unique_ptr<DeviceBuffer>>> Execute(
      absl::Span<DeviceBuffer* const> arguments,
      const ExecuteOptions& options) override;

  absl::StatusOr<std::string> Fingerprint() const override;
};

class MockCompiler final : public Compiler {
 public:
  MockCompiler() = default;
  ~MockCompiler() override = default;

  absl::StatusOr<std::unique_ptr<LoadedExecutable>> Compile(
      std::string_view serialized_program,
      const CompileOptions& options) override;
};

class MockBackend final : public Backend {
 public:
  MockBackend();
  ~MockBackend() override = default;

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

 private:
  MockDevice device_{0};
  std::vector<Device*> devices_ptrs_{&device_};
  MockCompiler compiler_;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_RUNTIME_MOCK_BACKEND_H_
