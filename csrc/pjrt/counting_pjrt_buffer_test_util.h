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

#ifndef TORCH_TPU_CSRC_PJRT_COUNTING_PJRT_BUFFER_TEST_UTIL_H_
#define TORCH_TPU_CSRC_PJRT_COUNTING_PJRT_BUFFER_TEST_UTIL_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/error_utils.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/service/device_assignment.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

// Forwarding `xla::PjRtBuffer` test double that counts `on_device_shape()`
// calls while delegating lightweight metadata accessors (`dimensions()`,
// `element_type()`, etc.) directly to `wrapped_`, matching `PjRtCApiBuffer`
// where `dimensions()` does not populate the lazy `on_device_shape_` cache.
//
// Holds the counter via `std::shared_ptr` so buffers retained asynchronously by
// stream state or worker tasks never outlive a stack counter.
class CountingPjRtBuffer final : public xla::PjRtBuffer {
 public:
  CountingPjRtBuffer(
      absl_nonnull std::unique_ptr<xla::PjRtBuffer> wrapped,
      absl_nonnull std::shared_ptr<std::atomic<int>> on_device_shape_calls)
      : wrapped_(std::move(wrapped)),
        on_device_shape_calls_(std::move(on_device_shape_calls)) {}

  xla::PjRtBuffer* wrapped() const { return wrapped_.get(); }

  xla::PrimitiveType element_type() const override {
    return wrapped_->element_type();
  }
  absl::Span<const int64_t> dimensions() const override {
    return wrapped_->dimensions();
  }
  std::shared_ptr<const xla::PjRtLayout> layout() const override {
    return wrapped_->layout();
  }
  bool IsTuple() const override { return wrapped_->IsTuple(); }
  const xla::Shape& on_device_shape() const override {
    on_device_shape_calls_->fetch_add(1, std::memory_order_relaxed);
    return wrapped_->on_device_shape();
  }
  bool has_dynamic_dimensions() const override {
    return wrapped_->has_dynamic_dimensions();
  }
  absl::Span<const bool> is_dynamic_dimension() const override {
    return wrapped_->is_dynamic_dimension();
  }
  absl::StatusOr<std::vector<int64_t>>  // INT_VEC_OK
  logical_dimensions() override {
    return wrapped_->logical_dimensions();
  }
  absl::StatusOr<xla::Shape> logical_on_device_shape() override {
    return wrapped_->logical_on_device_shape();
  }
  xla::PjRtMemorySpace* memory_space() const override {
    return wrapped_->memory_space();
  }
  xla::PjRtDevice* device() const override { return wrapped_->device(); }
  xla::PjRtClient* client() const override { return wrapped_->client(); }
  absl::StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override {
    return wrapped_->AcquireExternalReference();
  }
  xla::Future<> ToLiteral(xla::MutableLiteralBase* literal) override {
    return wrapped_->ToLiteral(literal);
  }
  xla::Future<> LazyToLiteral(
      absl::AnyInvocable<xla::Future<xla::MutableLiteralBase*>() &&> generator)
      override {
    return wrapped_->LazyToLiteral(std::move(generator));
  }
  absl::StatusOr<size_t> GetOnDeviceSizeInBytes() const override {
    return wrapped_->GetOnDeviceSizeInBytes();
  }
  xla::Future<> CopyRawToHost(void* dst, int64_t offset,
                              int64_t transfer_size) override {
    return wrapped_->CopyRawToHost(dst, offset, transfer_size);
  }
  xla::Future<> CopyRawToHostFuture(xla::Future<void*> dst, int64_t offset,
                                    int64_t transfer_size) override {
    return wrapped_->CopyRawToHostFuture(std::move(dst), offset, transfer_size);
  }
  void Delete() override { wrapped_->Delete(); }
  absl::StatusOr<std::unique_ptr<ExternalReference>>
  ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) override {
    return wrapped_->ReleaseDeviceMemoryOwnership(
        wait_for_operations_to_complete);
  }
  bool IsDeleted() const override { return wrapped_->IsDeleted(); }
  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> CopyToMemorySpace(
      xla::PjRtMemorySpace* dst_memory_space) override {
    return wrapped_->CopyToMemorySpace(dst_memory_space);
  }
  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> CopyToMemorySpace(
      xla::PjRtBuffer* donated_dst) override {
    return wrapped_->CopyToMemorySpace(donated_dst);
  }
  void CopyToRemoteDevice(xla::Future<std::string> serialized_descriptor,
                          RemoteSendCallback on_done) override {
    wrapped_->CopyToRemoteDevice(std::move(serialized_descriptor),
                                 std::move(on_done));
  }
  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> Bitcast(
      xla::PrimitiveType element_type, absl::Span<const int64_t> dims,
      const xla::Layout* device_layout) override {
    return wrapped_->Bitcast(element_type, dims, device_layout);
  }
  xla::Future<> GetReadyFuture() override { return wrapped_->GetReadyFuture(); }
  bool IsOnCpu() const override { return wrapped_->IsOnCpu(); }

 private:
  absl_nonnull std::unique_ptr<xla::PjRtBuffer> wrapped_;
  absl_nonnull std::shared_ptr<std::atomic<int>> on_device_shape_calls_;
};

// Returns the underlying `PjRtBuffer` if `buf` is a `CountingPjRtBuffer`
// (allowing tests to inspect `on_device_shape()` without incrementing the
// counter), or `buf` itself otherwise.
inline xla::PjRtBuffer* absl_nullable UnwrapCountingPjRtBuffer(
    xla::PjRtBuffer* absl_nullable buf) {
  if (const auto* counting = dynamic_cast<CountingPjRtBuffer*>(buf)) {
    return counting->wrapped();
  }
  return buf;
}

// Forwarding `xla::PjRtLoadedExecutable` test double that unwraps
// `CountingPjRtBuffer` arguments before dispatching to `wrapped_` (which casts
// arguments to its backend buffer type). When `output_on_device_shape_calls` is
// non-null, wraps every returned output buffer in a `CountingPjRtBuffer`
// sharing that counter.
class CountingPjRtLoadedExecutable final : public xla::PjRtLoadedExecutable {
 public:
  explicit CountingPjRtLoadedExecutable(
      absl_nonnull std::unique_ptr<xla::PjRtLoadedExecutable> wrapped,
      absl_nullable std::shared_ptr<std::atomic<int>>
          output_on_device_shape_calls = nullptr)
      : wrapped_(std::move(wrapped)),
        output_on_device_shape_calls_(std::move(output_on_device_shape_calls)) {
  }

  xla::PjRtExecutable* GetExecutable() const override {
    return wrapped_->GetExecutable();
  }
  xla::PjRtClient* client() const override { return wrapped_->client(); }
  const xla::DeviceAssignment& device_assignment() const override {
    return wrapped_->device_assignment();
  }
  absl::Span<xla::PjRtDevice* const> addressable_devices() const override {
    return wrapped_->addressable_devices();
  }
  void Delete() override { wrapped_->Delete(); }
  bool IsDeleted() const override { return wrapped_->IsDeleted(); }

  absl::StatusOr<std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>>
  Execute(absl::Span<const std::vector<xla::PjRtBuffer*>> argument_handles,
          const xla::ExecuteOptions& options,
          std::optional<std::vector<xla::Future<>>>& returned_futures)
      const override {
    std::vector<std::vector<xla::PjRtBuffer*>> unwrapped_args;
    unwrapped_args.reserve(argument_handles.size());
    for (const auto& device_args : argument_handles) {
      unwrapped_args.push_back(UnwrapBuffers(device_args));
    }
    TT_ASSIGN_OR_RETURN(auto outputs, wrapped_->Execute(unwrapped_args, options,
                                                        returned_futures));
    for (auto& device_outputs : outputs) {
      MaybeWrapOutputs(device_outputs);
    }
    return outputs;
  }

  absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>> ExecuteSharded(
      absl::Span<xla::PjRtBuffer* const> argument_handles,
      xla::PjRtDevice* device, const xla::ExecuteOptions& options,
      std::optional<xla::Future<>>& returned_future,
      bool fill_future) const override {
    const std::vector<xla::PjRtBuffer*> unwrapped_args =
        UnwrapBuffers(argument_handles);
    TT_ASSIGN_OR_RETURN(
        auto outputs, wrapped_->ExecuteSharded(unwrapped_args, device, options,
                                               returned_future, fill_future));
    MaybeWrapOutputs(outputs);
    return outputs;
  }

  absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>> ExecutePortable(
      absl::Span<xla::PjRtBuffer* const> argument_handles,
      xla::PjRtDevice* device, const xla::ExecuteOptions& options,
      std::optional<xla::Future<>>& returned_future,
      bool fill_future) const override {
    const std::vector<xla::PjRtBuffer*> unwrapped_args =
        UnwrapBuffers(argument_handles);
    TT_ASSIGN_OR_RETURN(
        auto outputs, wrapped_->ExecutePortable(unwrapped_args, device, options,
                                                returned_future, fill_future));
    MaybeWrapOutputs(outputs);
    return outputs;
  }

 private:
  static std::vector<xla::PjRtBuffer*> UnwrapBuffers(
      absl::Span<xla::PjRtBuffer* const> buffers) {
    std::vector<xla::PjRtBuffer*> unwrapped;
    unwrapped.reserve(buffers.size());
    for (xla::PjRtBuffer* buf : buffers) {
      unwrapped.push_back(UnwrapCountingPjRtBuffer(buf));
    }
    return unwrapped;
  }

  void MaybeWrapOutputs(
      std::vector<std::unique_ptr<xla::PjRtBuffer>>& outputs) const {
    if (output_on_device_shape_calls_ == nullptr) {
      return;
    }
    for (auto& out : outputs) {
      if (out != nullptr) {
        out = std::make_unique<CountingPjRtBuffer>(
            std::move(out), output_on_device_shape_calls_);
      }
    }
  }

  absl_nonnull std::unique_ptr<xla::PjRtLoadedExecutable> wrapped_;
  absl_nullable std::shared_ptr<std::atomic<int>> output_on_device_shape_calls_;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_PJRT_COUNTING_PJRT_BUFFER_TEST_UTIL_H_
