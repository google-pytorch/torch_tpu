/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "csrc/api/tensor_buffer.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/error_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/materialize.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {

class TensorBufferHandle::Impl {
 public:
  explicit Impl(DeviceBufferRef buffer_ref)
      : buffer_ref_(std::move(buffer_ref)) {}

  const DeviceBufferRef& buffer_ref() const { return buffer_ref_; }
  DeviceBufferRef& buffer_ref() { return buffer_ref_; }

 private:
  DeviceBufferRef buffer_ref_;
};

TensorBufferHandle::TensorBufferHandle(absl_nonnull std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

TensorBufferHandle::TensorBufferHandle(TensorBufferHandle&& other) noexcept =
    default;

TensorBufferHandle& TensorBufferHandle::operator=(
    TensorBufferHandle&& other) noexcept = default;

TensorBufferHandle::~TensorBufferHandle() = default;

int64_t TensorBufferHandle::size_bytes() const {
  return static_cast<int64_t>(impl_->buffer_ref().size_bytes());
}

absl::Span<const int64_t> TensorBufferHandle::dimensions() const {
  return impl_->buffer_ref().dimensions();
}

int64_t TensorBufferHandle::num_elements() const {
  return impl_->buffer_ref().num_elements();
}

DeviceBufferState TensorBufferHandle::state() const {
  if (impl_->buffer_ref().is_materialized()) {
    return DeviceBufferState::kMaterialized;
  }
  if (impl_->buffer_ref().is_executing()) {
    return DeviceBufferState::kExecuting;
  }
  if (impl_->buffer_ref().is_materializing()) {
    return DeviceBufferState::kMaterializing;
  }
  if (impl_->buffer_ref().is_deferred()) {
    return DeviceBufferState::kDeferred;
  }
  return DeviceBufferState::kPlaceholder;
}

absl::Status TensorBufferHandle::Materialize() {
  return torch_tpu::Materialize(impl_->buffer_ref(),
                                MaterializationReason::kExplicitSync);
}

absl::StatusOr<xla::PjRtBuffer* absl_nonnull>
TensorBufferHandle::AwaitBuffer() {
  return impl_->buffer_ref().AwaitBuffer();
}

absl::Status TensorBufferHandle::Synchronize() {
  return impl_->buffer_ref().Synchronize();
}

absl::StatusOr<TensorBufferHandle> GetBaseTensorBuffer(
    const at::Tensor& tensor) {
  auto buffer_ref_or = GetBaseBuffer(tensor);
  if (!buffer_ref_or.ok()) {
    return buffer_ref_or.status();
  }
  return TensorBufferHandle(
      std::make_unique<TensorBufferHandle::Impl>(std::move(*buffer_ref_or)));
}

}  // namespace torch_tpu
