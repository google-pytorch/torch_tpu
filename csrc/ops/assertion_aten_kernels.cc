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
#include "csrc/ops/assertion_aten_kernels.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/error_utils.h"
#include "csrc/eager/current_stream.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/graph_capture_state.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"
#include "csrc/pjrt/pjrt_transfer.h"
#include "xla/future.h"

namespace torch_tpu {
namespace {

// Validates that the assertion condition tensor is a single-element scalar.
// Matches PyTorch reference semantics where empty or multi-element tensors
// produce an ambiguous boolean value error.
void ValidateAssertionInput(const at::Tensor& self) {
  const int64_t n = self.numel();
  TT_CHECK_THROW(n != 0, error::kInvalidArgument)
      << "boolean value of Tensor with no values is ambiguous";
  TT_CHECK_THROW(n < 2, error::kInvalidArgument)
      << "boolean value of Tensor with more than one value is ambiguous";
}

// Determines whether the assertion check should be scheduled asynchronously on
// the host. Returns false during graph compilation or when dealing with
// symbolic / placeholder tensors (e.g. during FX graph tracing or
// torch.compile).
//
// Also returns false while the current thread captures a
// `torch.accelerator.Graph`: the condition is only known when the graph
// replays, and the replays of a graph do not check assertions.
bool ShouldScheduleAsyncCheck(const at::Tensor& self) {
  const EagerMode mode = GetEagerMode();
  if (mode == EagerMode::kInternalCompileFxGraph ||
      mode == EagerMode::kInternalDeferAll || IsCurrentThreadCapturingGraph()) {
    return false;
  }
  const auto buffer_ref_or = GetBuffer(self);
  if (!buffer_ref_or.ok() || buffer_ref_or->is_placeholder()) {
    return false;
  }
  return true;
}

// Schedules an asynchronous assertion check for `self` on the background thread
// pool. Materializes the tensor's TPU buffer, copies the 1-element scalar to
// CPU, and evaluates its truth value without blocking the main execution
// thread.
void ScheduleAsyncAssertionCheck(const at::Tensor& self,
                                 const std::string_view assert_msg) {
  if (!ShouldScheduleAsyncCheck(self)) {
    return;
  }

  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();

  // Start materialization of `self` on TPU, but don't wait for it to complete.
  auto materialized_buf_or =
      MaterializeAndReturn(self, MaterializationReason::kCpuTransfer);
  if (!materialized_buf_or.ok()) {
    internal::MarkStreamActive(device_index, stream_id,
                               materialized_buf_or.status());
    return;
  }

  // Timing details:
  // On CUDA, the assertion check happens on-device, during the stream of
  // execution. No such device-side mechanism exists on TPU.
  // To get as close as possible, we need to compute the boolean condition on
  // device, copy it to host, and if it's an error, inject a failure into the
  // stream to be picked up on the next assertion.
  // This is not a normal TPU-to-CPU copy because we allow it to bypass
  // cross-stream wait dependencies; this is simulating a zero-copy assertion.

  // We don't want to hold an at::Tensor on a background thread as this can
  // cause GIL deadlocks on free. Instead, we copy the raw bytes to a
  // host-allocated vector that is kept alive until we can check it.
  // Any byte value that is not entirely zero is "true" and is a pass; all zeros
  // is "false" and returns the user-specified error message.
  std::vector<uint8_t> dst_vector(self.element_size());
  DeviceToHostTransfer transfer = DeviceToHostTransfer::IntoBytes(
      std::move(*materialized_buf_or), dst_vector.data());

  // The assertion must be checked on the next stream sync.
  auto [assertion_promise, assertion_future] = xla::MakePromise<void>();
  internal::MarkStreamActive(device_index, stream_id,
                             std::move(assertion_future));

  // Start the transfer immediately but don't record it on the stream;
  // the assertion future is sufficient.
  xla::Future<void> transfer_done = transfer.Run();
  transfer_done.OnReady([promise = std::move(assertion_promise),
                         msg = (assert_msg.empty() ? "assertion failed"
                                                   : std::string(assert_msg)),
                         dst_vector = std::move(dst_vector)](
                            absl::Status status) mutable {
    if (!status.ok()) {
      promise.Set(status);
      return;
    }
    if (absl::c_any_of(dst_vector, [](uint8_t byte) { return byte != 0; })) {
      promise.Set(absl::OkStatus());
      return;
    }
    promise.Set(TT_ERROR(error::kInternal) << msg);
  });
}

}  // namespace

// aten::_assert_async(Tensor self) -> ()
void AtenAssertAsync(const at::Tensor& self) {
  AtenAssertAsyncMsg(self, "assertion failed");
}

// aten::_assert_async.msg(Tensor self, str assert_msg) -> ()
void AtenAssertAsyncMsg(const at::Tensor& self, std::string_view assert_msg) {
  TT_KERNEL(
      OpName::kAssertAsyncMsg, _,
      (self, IgnoreInCacheKey(assert_msg, "message does not affect TPU graph")),
      {
        ValidateAssertionInput(self);
        ScheduleAsyncAssertionCheck(self, assert_msg);
      });
}

}  // namespace torch_tpu
