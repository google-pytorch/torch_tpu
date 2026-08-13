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

#include "torch_tpu/runtime/cerebro/cerebro_device_buffer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/device_buffer.h"
#include "xla/tsl/concurrency/future.h"

namespace torch_tpu {

CerebroDeviceBuffer::CerebroDeviceBuffer(at::ScalarType dtype, Dimensions shape,
                                         Device* device,
                                         uint64_t cluster_handle,
                                         size_t size_in_bytes)
    : dtype_(dtype),
      shape_(std::move(shape)),
      device_(device),
      cluster_handle_(cluster_handle),
      size_in_bytes_(size_in_bytes) {}

tsl::Future<> CerebroDeviceBuffer::GetReadyFuture() const {
  return tsl::Future<>(absl::OkStatus());
}

absl::Status CerebroDeviceBuffer::BlockHostUntilReady() {
  return absl::OkStatus();
}

tsl::Future<> CerebroDeviceBuffer::CopyToHostAsync(void* dst,
                                                   size_t size_in_bytes) {
  if (dst != nullptr && size_in_bytes <= size_in_bytes_) {
    std::memset(dst, 0, size_in_bytes);
  }
  return tsl::Future<>(absl::OkStatus());
}

absl::StatusOr<std::unique_ptr<DeviceBuffer>> CerebroDeviceBuffer::CreateView(
    absl::Span<const int64_t> new_shape) {
  Dimensions view_shape(new_shape.begin(), new_shape.end());
  return std::make_unique<CerebroDeviceBuffer>(
      dtype_, std::move(view_shape), device_, cluster_handle_, size_in_bytes_);
}

}  // namespace torch_tpu
