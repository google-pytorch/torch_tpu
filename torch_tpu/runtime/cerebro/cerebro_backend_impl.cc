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

#include "torch_tpu/runtime/cerebro/cerebro_backend_impl.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/cerebro/cerebro_device_buffer.h"
#include "torch_tpu/runtime/device_buffer.h"
#include "torch_tpu/runtime/executable.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/tsl/framework/allocator.h"

namespace torch_tpu {

CerebroDevice::CerebroDevice(int id, std::string kind,
                             std::vector<int64_t> coords,  // INT_VEC_OK
                             int64_t core_on_chip)
    : id_(id),
      kind_(std::move(kind)),
      coords_(std::move(coords)),
      core_on_chip_(core_on_chip) {}

std::string CerebroDevice::to_string() const {
  return absl::StrCat("CerebroDevice(id=", id_, ", kind=", kind_, ", coords=[",
                      coords_[0], ",", coords_[1], ",", coords_[2],
                      "], core=", core_on_chip_, ")");
}

CerebroLoadedExecutable::CerebroLoadedExecutable(std::string program_id)
    : program_id_(std::move(program_id)) {}

absl::StatusOr<std::vector<std::unique_ptr<DeviceBuffer>>>
CerebroLoadedExecutable::Execute(absl::Span<DeviceBuffer* const> arguments,
                                 const ExecuteOptions& options) {
  std::vector<std::unique_ptr<DeviceBuffer>> results;
  for (DeviceBuffer* arg : arguments) {
    if (arg != nullptr) {
      Dimensions out_shape(arg->shape().begin(), arg->shape().end());
      results.push_back(std::make_unique<CerebroDeviceBuffer>(
          arg->dtype(), std::move(out_shape), arg->device(), 2000,
          arg->GetOnDeviceSizeInBytes()));
    }
  }
  return results;
}

absl::StatusOr<std::string> CerebroLoadedExecutable::Fingerprint() const {
  return absl::StrCat("cerebro_fp_", program_id_);
}

absl::StatusOr<std::unique_ptr<LoadedExecutable>> CerebroCompiler::Compile(
    std::string_view serialized_program, const CompileOptions& options) {
  return std::make_unique<CerebroLoadedExecutable>(
      absl::StrCat("prog_", serialized_program.size()));
}

CerebroBackendImpl::CerebroBackendImpl(const BackendOptions& options)
    : options_(options) {
  InitializeClusterTopology(/*num_nodes=*/4, /*chips_per_node=*/4);
}

void CerebroBackendImpl::InitializeClusterTopology(int num_nodes,
                                                   int chips_per_node) {
  int device_id = 0;
  for (int node = 0; node < num_nodes; ++node) {
    for (int chip = 0; chip < chips_per_node; ++chip) {
      for (int core = 0; core < 2; ++core) {
        std::vector<int64_t> coords = {node, chip, core};  // INT_VEC_OK
        auto dev = std::make_unique<CerebroDevice>(device_id++, "TPU v5e",
                                                   std::move(coords), core);
        devices_ptrs_.push_back(dev.get());
        owned_devices_.push_back(std::move(dev));
      }
    }
  }
}

absl::Span<Device* const> CerebroBackendImpl::GetDevices() const {
  return devices_ptrs_;
}

absl::Span<Device* const> CerebroBackendImpl::GetAddressableDevices() const {
  return devices_ptrs_;
}

absl::StatusOr<int> CerebroBackendImpl::GetGlobalDeviceId() const {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal device check
      !devices_ptrs_.empty(), error::kNotFound)
      << "no cerebro devices found";
  return devices_ptrs_.front()->id();
}

absl::StatusOr<int> CerebroBackendImpl::GetGlobalDeviceCount() const {
  return devices_ptrs_.size();
}

Device* CerebroBackendImpl::GetDefaultDevice() const {
  return devices_ptrs_.empty() ? nullptr : devices_ptrs_.front();
}

absl::StatusOr<tsl::AllocatorStats> CerebroBackendImpl::GetAllocatorStats() {
  tsl::AllocatorStats stats;
  stats.num_allocs = 50;
  stats.bytes_in_use = 1024 * 1024 * 64;
  stats.peak_bytes_in_use = 1024 * 1024 * 128;
  return stats;
}

absl::Status CerebroBackendImpl::ClearAllocatorStats() {
  return absl::OkStatus();
}

absl::StatusOr<xla::HostMemoryAllocator*>
CerebroBackendImpl::GetHostAllocator() {
  return nullptr;
}

absl::StatusOr<std::unique_ptr<DeviceBuffer>>
CerebroBackendImpl::MakeBufferFromHost(const void* data, at::ScalarType dtype,
                                       absl::Span<const int64_t> shape,
                                       Device* device) {
  size_t num_elements = 1;
  for (int64_t dim : shape) {
    num_elements *= dim;
  }
  size_t byte_size = num_elements * c10::elementSize(dtype);
  Dimensions shape_vec(shape.begin(), shape.end());

  uint64_t handle = next_cluster_handle_++;
  return std::make_unique<CerebroDeviceBuffer>(dtype, std::move(shape_vec),
                                               device, handle, byte_size);
}

absl::StatusOr<Compiler*> CerebroBackendImpl::GetCompiler() {
  return &compiler_;
}

}  // namespace torch_tpu
