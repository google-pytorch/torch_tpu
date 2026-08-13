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

#include "torch_tpu/runtime/pjrt/pjrt_device_buffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/device_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/concurrency/future.h"

namespace torch_tpu {

PjrtDeviceBuffer::PjrtDeviceBuffer(std::shared_ptr<xla::PjRtBuffer> pjrt_buffer,
                                   at::ScalarType dtype, Device* device)
    : pjrt_buffer_(std::move(pjrt_buffer)), dtype_(dtype), device_(device) {
  if (pjrt_buffer_ != nullptr) {
    auto dims = pjrt_buffer_->dimensions();
    shape_.assign(dims.begin(), dims.end());
  }
}

size_t PjrtDeviceBuffer::GetOnDeviceSizeInBytes() const {
  if (pjrt_buffer_ == nullptr) {
    return 0;
  }
  return pjrt_buffer_->GetOnDeviceSizeInBytes().value_or(0);
}

tsl::Future<> PjrtDeviceBuffer::GetReadyFuture() const {
  if (pjrt_buffer_ == nullptr) {
    return tsl::Future<>(
        TT_ERROR(error::kInternal /* ERROR_COV_INFEASIBLE=null check */)
        << "uninitialized PjRtBuffer");
  }
  return pjrt_buffer_->GetReadyFuture();
}

absl::Status PjrtDeviceBuffer::BlockHostUntilReady() {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=null check
      pjrt_buffer_ != nullptr, error::kInternal)
      << "uninitialized PjRtBuffer";
  return pjrt_buffer_->GetReadyFuture().Await();
}

tsl::Future<> PjrtDeviceBuffer::CopyToHostAsync(void* dst,
                                                size_t size_in_bytes) {
  if (pjrt_buffer_ == nullptr) {
    return tsl::Future<>(
        TT_ERROR(error::kInternal /* ERROR_COV_INFEASIBLE=null check */)
        << "uninitialized PjRtBuffer");
  }
  return pjrt_buffer_->CopyRawToHost(dst, 0,
                                     static_cast<int64_t>(size_in_bytes));
}

absl::StatusOr<std::unique_ptr<DeviceBuffer>> PjrtDeviceBuffer::CreateView(
    absl::Span<const int64_t> new_shape) {
  return TT_ERROR(  // ERROR_COV_INFEASIBLE=Views are IR ops
             error::kPythonNotImplementedError)
         << "reshaped views on PjRtBuffer should be represented as IR ops or "
            "client sub-views";
}

}  // namespace torch_tpu
