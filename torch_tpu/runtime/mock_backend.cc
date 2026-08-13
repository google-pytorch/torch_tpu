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

#include "torch_tpu/runtime/mock_backend.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/device_buffer.h"
#include "torch_tpu/runtime/executable.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/framework/allocator.h"

namespace torch_tpu {

MockDeviceBuffer::MockDeviceBuffer(at::ScalarType dtype, Dimensions shape,
                                   Device* device)
    : dtype_(dtype), shape_(std::move(shape)), device_(device) {
  size_t num_elements = 1;
  for (int64_t dim : shape_) {
    num_elements *= dim;
  }
  size_t byte_size = num_elements * c10::elementSize(dtype);
  storage_.resize(byte_size, 0);
}

tsl::Future<> MockDeviceBuffer::GetReadyFuture() const {
  return tsl::Future<>(absl::OkStatus());
}

absl::Status MockDeviceBuffer::BlockHostUntilReady() {
  return absl::OkStatus();
}

tsl::Future<> MockDeviceBuffer::CopyToHostAsync(void* dst,
                                                size_t size_in_bytes) {
  if (dst != nullptr && size_in_bytes <= storage_.size()) {
    std::memcpy(dst, storage_.data(), size_in_bytes);
  }
  return tsl::Future<>(absl::OkStatus());
}

absl::StatusOr<std::unique_ptr<DeviceBuffer>> MockDeviceBuffer::CreateView(
    absl::Span<const int64_t> new_shape) {
  Dimensions shape_vec(new_shape.begin(), new_shape.end());
  auto view =
      std::make_unique<MockDeviceBuffer>(dtype_, std::move(shape_vec), device_);
  view->mutable_storage() = storage_;
  return view;
}

absl::StatusOr<std::vector<std::unique_ptr<DeviceBuffer>>>
MockLoadedExecutable::Execute(absl::Span<DeviceBuffer* const> arguments,
                              const ExecuteOptions& options) {
  std::vector<std::unique_ptr<DeviceBuffer>> results;
  for (DeviceBuffer* arg : arguments) {
    if (arg != nullptr) {
      Dimensions out_shape(arg->shape().begin(), arg->shape().end());
      results.push_back(std::make_unique<MockDeviceBuffer>(
          arg->dtype(), std::move(out_shape), arg->device()));
    }
  }
  return results;
}

absl::StatusOr<std::string> MockLoadedExecutable::Fingerprint() const {
  return "mock_executable_fingerprint_12345";
}

absl::StatusOr<std::unique_ptr<LoadedExecutable>> MockCompiler::Compile(
    std::string_view serialized_program, const CompileOptions& options) {
  return std::make_unique<MockLoadedExecutable>();
}

MockBackend::MockBackend() = default;

absl::Span<Device* const> MockBackend::GetDevices() const {
  return devices_ptrs_;
}

absl::Span<Device* const> MockBackend::GetAddressableDevices() const {
  return devices_ptrs_;
}

absl::StatusOr<int> MockBackend::GetGlobalDeviceId() const { return 0; }

absl::StatusOr<int> MockBackend::GetGlobalDeviceCount() const { return 1; }

Device* MockBackend::GetDefaultDevice() const {
  return const_cast<MockDevice*>(&device_);
}

absl::StatusOr<tsl::AllocatorStats> MockBackend::GetAllocatorStats() {
  tsl::AllocatorStats stats;
  stats.num_allocs = 10;
  stats.bytes_in_use = 1024;
  stats.peak_bytes_in_use = 2048;
  return stats;
}

absl::Status MockBackend::ClearAllocatorStats() { return absl::OkStatus(); }

absl::StatusOr<xla::HostMemoryAllocator*> MockBackend::GetHostAllocator() {
  return nullptr;
}

absl::StatusOr<std::unique_ptr<DeviceBuffer>> MockBackend::MakeBufferFromHost(
    const void* data, at::ScalarType dtype, absl::Span<const int64_t> shape,
    Device* device) {
  Dimensions shape_vec(shape.begin(), shape.end());
  auto buf = std::make_unique<MockDeviceBuffer>(
      dtype, std::move(shape_vec), device != nullptr ? device : &device_);
  if (data != nullptr) {
    std::memcpy(buf->mutable_storage().data(), data,
                buf->GetOnDeviceSizeInBytes());
  }
  return buf;
}

absl::StatusOr<Compiler*> MockBackend::GetCompiler() { return &compiler_; }

}  // namespace torch_tpu
