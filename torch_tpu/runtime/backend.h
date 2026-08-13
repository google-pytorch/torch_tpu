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

#ifndef TORCH_TPU_RUNTIME_BACKEND_H_
#define TORCH_TPU_RUNTIME_BACKEND_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/runtime/device_buffer.h"
#include "torch_tpu/runtime/executable.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/tsl/framework/allocator.h"

namespace torch_tpu {

// Supported execution backend kinds.
enum class BackendKind {
  kMock,
  kPjrt,
  kCerebro,
};

// Representation of a physical or logical device on the backend.
class Device {
 public:
  virtual ~Device() = default;

  [[nodiscard]] virtual int id() const = 0;
  [[nodiscard]] virtual std::string kind() const = 0;
  [[nodiscard]] virtual std::string to_string() const = 0;
  [[nodiscard]] virtual bool is_addressable() const = 0;

  [[nodiscard]] virtual std::optional<std::vector<int64_t> /* INT_VEC_OK */>
  coords() const {
    return std::nullopt;
  }
  [[nodiscard]] virtual std::optional<int64_t> core_on_chip() const {
    return std::nullopt;
  }

 protected:
  Device() = default;
};

// Configuration options passed during backend initialization.
struct BackendOptions {
  std::string platform_name = "TPU";
  bool enable_async_execution = true;
};

// Pure abstract interface representing a TorchTPU execution backend.
class Backend {
 public:
  virtual ~Backend() = default;

  static absl::StatusOr<std::unique_ptr<Backend>> Create(
      BackendKind kind, const BackendOptions& options);

  // Creates a backend tailored for the specified device_type (e.g. "tpu",
  // "xla_cpu", "xla_cuda").
  static absl::StatusOr<std::unique_ptr<Backend>> CreateForDevice(
      const std::string& device_type, BackendKind kind,
      const BackendOptions& options);

  static Backend* GetInstance();
  static void SetInstance(std::unique_ptr<Backend> backend);
  static void ResetInstance();

  [[nodiscard]] virtual absl::Span<Device* const> GetDevices() const = 0;
  [[nodiscard]] virtual absl::Span<Device* const> GetAddressableDevices()
      const = 0;
  virtual absl::StatusOr<int> GetGlobalDeviceId() const = 0;
  virtual absl::StatusOr<int> GetGlobalDeviceCount() const = 0;
  [[nodiscard]] virtual Device* GetDefaultDevice() const = 0;

  virtual absl::StatusOr<tsl::AllocatorStats> GetAllocatorStats() = 0;
  virtual absl::Status ClearAllocatorStats() = 0;
  virtual absl::StatusOr<xla::HostMemoryAllocator*> GetHostAllocator() = 0;

  virtual absl::StatusOr<std::unique_ptr<DeviceBuffer>> MakeBufferFromHost(
      const void* data, at::ScalarType dtype, absl::Span<const int64_t> shape,
      Device* device) = 0;

  virtual absl::StatusOr<Compiler*> GetCompiler() = 0;

  virtual absl::Status SynchronizeDevice(int device_id) {
    return absl::OkStatus();
  }
  virtual absl::Status SynchronizeAllDevices() { return absl::OkStatus(); }

  [[nodiscard]] virtual bool IsSingleController() const { return false; }

 protected:
  Backend() = default;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_RUNTIME_BACKEND_H_
