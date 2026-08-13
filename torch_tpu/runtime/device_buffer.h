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

#ifndef TORCH_TPU_RUNTIME_DEVICE_BUFFER_H_
#define TORCH_TPU_RUNTIME_DEVICE_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch/headeronly/core/ScalarType.h"
#include "xla/tsl/concurrency/future.h"

namespace torch_tpu {

class Device;

// Polymorphic handle representing on-device memory across backends (PJRT /
// Cerebro).
class DeviceBuffer {
 public:
  virtual ~DeviceBuffer() = default;

  [[nodiscard]] virtual at::ScalarType dtype() const = 0;
  [[nodiscard]] virtual absl::Span<const int64_t> shape() const = 0;
  [[nodiscard]] virtual Device* device() const = 0;
  [[nodiscard]] virtual size_t GetOnDeviceSizeInBytes() const = 0;

  [[nodiscard]] virtual tsl::Future<> GetReadyFuture() const = 0;
  virtual absl::Status BlockHostUntilReady() = 0;
  virtual tsl::Future<> CopyToHostAsync(void* dst, size_t size_in_bytes) = 0;

  virtual absl::StatusOr<std::unique_ptr<DeviceBuffer>> CreateView(
      absl::Span<const int64_t> new_shape) = 0;

  // Returns multi-dimensional cluster mesh coordinates if sharded across a
  // distributed topology.
  [[nodiscard]] virtual std::optional<std::vector<int64_t> /* INT_VEC_OK */>
  mesh_coordinates() const {
    return std::nullopt;
  }

  // Tracks buffer donation status for in-place optimization and alias reuse.
  [[nodiscard]] virtual bool IsDonated() const { return is_donated_; }
  virtual void MarkDonated() { is_donated_ = true; }

 protected:
  DeviceBuffer() = default;

  bool is_donated_ = false;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_RUNTIME_DEVICE_BUFFER_H_
