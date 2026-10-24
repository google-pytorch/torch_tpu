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
#include <exception>
#include <string>
#include <string_view>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/base/no_destructor.h"
#include "absl/base/optimization.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/reroute_warnings.h"
#include "csrc/common/thread_pool.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/copy_from/copy_common.h"
#include "csrc/ops/copy_from/tpu_to_cpu.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"

namespace torch_tpu {
namespace {

// Returns the background thread pool for evaluating asynchronous assertion
// checks. Uses a single thread to guarantee that assertions are processed in
// strict FIFO program order, ensuring deterministic error reporting when
// multiple assertions fail in sequence.
ThreadPool& GetAssertionThreadPool() {
  static absl::NoDestructor<ThreadPool> pool("assert_async", /*num_threads=*/1);
  return *pool;
}

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
bool ShouldScheduleAsyncCheck(const at::Tensor& self) {
  const EagerMode mode = GetEagerMode();
  if (mode == EagerMode::kInternalCompileFxGraph ||
      mode == EagerMode::kInternalDeferAll) {
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

  // Materialize the condition tensor on device so its PjRtBuffer is populated
  // and ready for device-to-host transfer.
  auto materialized_buf_or =
      MaterializeAndReturn(self, MaterializationReason::kCpuTransfer);
  if (!materialized_buf_or.ok()) {
    SetStickyError(materialized_buf_or.status());
    return;
  }
  const DeviceBufferRef materialized_buf = *materialized_buf_or;

  const std::string msg =
      assert_msg.empty() ? "assertion failed" : std::string(assert_msg);

  IncrementPendingAssertionChecks();
  GetAssertionThreadPool().Schedule([materialized_buf, msg = std::move(msg)]() {
    // Let dispatch threads surface this thread's warnings to Python.
    RouteWarningsOnThisThreadToDispatchThread();
    absl::Cleanup cleanup = [] { DecrementPendingAssertionChecks(); };

    // Copy the condition scalar to CPU host memory.
    // This is a blocking transfer on a background thread, which is async
    // relative to the thread that called `ScheduleAsyncAssertionCheck`.
    // It needs to be blocking so that we can immediately call ".item()" below.
    // It also needs to be isolated so that we don't interrupt the stream
    // from a background thread.
    auto status_or_tensor = CopyDeviceBufferRefToCpuTensor(
        materialized_buf, TransferSyncMode::kIsolated);
    if (!status_or_tensor.ok()) {
      SetStickyError(status_or_tensor.status());
      return;
    }

    const at::Tensor& cpu_tensor = status_or_tensor.value();
    bool condition_val = false;
    try {
      condition_val = cpu_tensor.item().toBool();
    } catch (const std::exception& e) {
      SetStickyError(TT_ERROR(error::kInvalidArgument)
                     << "assert_async expects a boolean or scalar condition");
      return;
    }

    if (!condition_val) {
      SetStickyError(TT_ERROR(error::kInternal) << msg);
    }
  });
}

}  // namespace

// Implementation of aten::_assert_async and aten::_assert_async.msg for TPU.
//
// PyTorch's `torch._assert_async` verifies a boolean condition tensor on device
// without synchronously blocking host execution on the critical path.
//
// Why on-device assertions cannot be used for `torch._assert_async` on TPU:
// 1. Hardware halt semantics: While low-level TPU kernels (e.g. via Mosaic
//    cf.AssertOp) can trigger on-device hardware assertion instructions, doing
//    so permanently halts the execution core ("halt state"), preventing the
//    device from executing any subsequent programs without a full reset.
//    Unlike CUDA where an assertion failure only corrupts the user's host
//    driver context (cudaErrorAssert) leaving the underlying GPU silicon
//    healthy for subsequent processes, a TPU shalt_err is a bare-metal machine
//    halt on the physical core that wedges lockstep pod execution and requires
//    a hardware or partition reset. In PyTorch, assertion failures must produce
//    standard catchable `RuntimeError` exceptions while keeping the TPU device
//    context intact.
// 2. StableHLO limitations: Standard StableHLO has no dynamic runtime assert op
//    lowering for TPU—`@shape_assertion` is strictly a compile-time static
//    shape refinement pass that requires static integer operands and is erased
//    before code generation.
// 3. Void return signature: `aten::_assert_async` returns void, producing no
//    downstream consumer tensors in the computation graph.
//
// In TorchTPU, non-blocking asynchronous assertion checking is achieved via an
// asynchronous host worker:
// 1. Validating that the condition is a single-element scalar tensor on the
//    host.
// 2. Checking if any previous assertion has already failed via
//    HasStickyError().
// 3. Materializing the condition tensor's TPU buffer and scheduling an
//    asynchronous DtoH transfer and condition check onto a dedicated single-
//    threaded background worker.
// 4. If the condition evaluates to false, latching an error into the global
//    sticky error state (SetStickyError).
// 5. Subsequent op dispatches (via DispatchOp) and explicit device/stream syncs
//    (via SyncAndCheckStickyError) detect the sticky error and raise a C++
//    exception into PyTorch.
// TODO(bawilson): can we do this through events_queue to reuse existing
// infrastructure?

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
        // Fail fast if a previous assertion has already set a sticky error.
        if (ABSL_PREDICT_FALSE(HasStickyError())) {
          TT_THROW_IF_ERROR(GetStickyError());
        }
        ValidateAssertionInput(self);
        ScheduleAsyncAssertionCheck(self, assert_msg);
      });
}

}  // namespace torch_tpu
