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

#ifndef TORCH_TPU_RUNTIME_CEREBRO_CEREBRO_DEVICE_BUFFER_H_
#define TORCH_TPU_RUNTIME_CEREBRO_CEREBRO_DEVICE_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/device_buffer.h"
#include "xla/tsl/concurrency/future.h"

namespace torch_tpu {

// Sharded cluster device buffer representing a global tensor across a Cerebro
// mesh.
class CerebroDeviceBuffer final : public DeviceBuffer {
 public:
  CerebroDeviceBuffer(at::ScalarType dtype, Dimensions shape, Device* device,
                      uint64_t cluster_handle, size_t size_in_bytes);
  ~CerebroDeviceBuffer() override = default;

  [[nodiscard]] at::ScalarType dtype() const override { return dtype_; }
  [[nodiscard]] absl::Span<const int64_t> shape() const override {
    return shape_;
  }
  [[nodiscard]] Device* device() const override { return device_; }
  [[nodiscard]] size_t GetOnDeviceSizeInBytes() const override {
    return size_in_bytes_;
  }

  [[nodiscard]] tsl::Future<> GetReadyFuture() const override;
  absl::Status BlockHostUntilReady() override;
  tsl::Future<> CopyToHostAsync(void* dst, size_t size_in_bytes) override;
  absl::StatusOr<std::unique_ptr<DeviceBuffer>> CreateView(
      absl::Span<const int64_t> new_shape) override;

  [[nodiscard]] std::optional<std::vector<int64_t> /* INT_VEC_OK */>
  mesh_coordinates() const override {
    if (device_ != nullptr) {
      return device_->coords();
    }
    return std::nullopt;
  }

  [[nodiscard]] uint64_t cluster_handle() const { return cluster_handle_; }

 private:
  at::ScalarType dtype_;
  Dimensions shape_;
  Device* device_;
  uint64_t cluster_handle_;
  size_t size_in_bytes_;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_RUNTIME_CEREBRO_CEREBRO_DEVICE_BUFFER_H_
