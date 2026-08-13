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

#ifndef TORCH_TPU_RUNTIME_CEREBRO_CEREBRO_BACKEND_IMPL_H_
#define TORCH_TPU_RUNTIME_CEREBRO_CEREBRO_BACKEND_IMPL_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
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
#include "xla/tsl/framework/allocator.h"

namespace torch_tpu {

class CerebroDevice final : public Device {
 public:
  CerebroDevice(int id, std::string kind,
                std::vector<int64_t> coords,  // INT_VEC_OK
                int64_t core_on_chip);
  ~CerebroDevice() override = default;

  [[nodiscard]] int id() const override { return id_; }
  [[nodiscard]] std::string kind() const override { return kind_; }
  [[nodiscard]] std::string to_string() const override;
  [[nodiscard]] bool is_addressable() const override { return true; }
  [[nodiscard]] std::optional<std::vector<int64_t> /* INT_VEC_OK */> coords()
      const override {
    return coords_;
  }
  [[nodiscard]] std::optional<int64_t> core_on_chip() const override {
    return core_on_chip_;
  }

 private:
  int id_;
  std::string kind_;
  std::vector<int64_t> coords_;  // INT_VEC_OK
  int64_t core_on_chip_;
};

class CerebroLoadedExecutable final : public LoadedExecutable {
 public:
  explicit CerebroLoadedExecutable(std::string program_id);
  ~CerebroLoadedExecutable() override = default;

  absl::StatusOr<std::vector<std::unique_ptr<DeviceBuffer>>> Execute(
      absl::Span<DeviceBuffer* const> arguments,
      const ExecuteOptions& options) override;

  absl::StatusOr<std::string> Fingerprint() const override;

 private:
  std::string program_id_;
};

class CerebroCompiler final : public Compiler {
 public:
  CerebroCompiler() = default;
  ~CerebroCompiler() override = default;

  absl::StatusOr<std::unique_ptr<LoadedExecutable>> Compile(
      std::string_view serialized_program,
      const CompileOptions& options) override;
};

class CerebroBackendImpl final : public Backend {
 public:
  explicit CerebroBackendImpl(const BackendOptions& options);
  ~CerebroBackendImpl() override = default;

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
  [[nodiscard]] bool IsSingleController() const override { return true; }

 private:
  void InitializeClusterTopology(int num_nodes, int chips_per_node);

  BackendOptions options_;
  std::vector<std::unique_ptr<CerebroDevice>> owned_devices_;
  std::vector<Device*> devices_ptrs_;
  CerebroCompiler compiler_;
  uint64_t next_cluster_handle_ = 1000;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_RUNTIME_CEREBRO_CEREBRO_BACKEND_IMPL_H_
