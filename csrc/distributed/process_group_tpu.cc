/*
 * Copyright 2025 Google LLC
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

#include "csrc/distributed/process_group_tpu.h"

#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>  // NOLINT - needed for PyTorch API
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <thread>  // NOLINT - needed for P2pSendQueue worker thread
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/ivalue.h"
#include "ATen/core/ivalue_inl.h"
#include "ATen/core/jit_type.h"
#include "ATen/ops/cat.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/empty_like.h"
#include "ATen/ops/flatten.h"
#include "ATen/ops/ones.h"
#include "ATen/ops/stack.h"
#include "ATen/ops/zeros.h"
#include "absl/algorithm/container.h"
#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/Exception.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/common/macro_utils.h"
#include "csrc/eager/device_assignment.h"
#include "torch/csrc/distributed/c10d/Backend.hpp"
#include "torch/csrc/distributed/c10d/FileStore.hpp"
#include "torch/csrc/distributed/c10d/PrefixStore.hpp"
#include "torch/csrc/distributed/c10d/Store.hpp"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "xla/pjrt/pjrt_common.h"
#if TT_TORCH_VERSION_GE(2, 14)
#include "torch/csrc/distributed/c10d/Window.hpp"
#endif
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fingerprint_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/distributed/allgather.h"
#include "csrc/distributed/allreduce.h"
#include "csrc/distributed/alltoall.h"
#include "csrc/distributed/reduce_scatter.h"
#include "csrc/distributed/types.h"
#include "csrc/distributed/utils.h"
#include "csrc/distributed/window_tpu.h"
#include "csrc/eager/current_stream.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/materialize.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/pjrt/pjrt_state.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/csrc/distributed/c10d/Work.hpp"
#include "torch/headeronly/core/ScalarType.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/runtime/device_id.h"
#include "xla/shape.h"

namespace torch_tpu {

namespace {

// Waits up to `timeout` for `future` to become ready.
bool AwaitFutureWithTimeout(const xla::Future<void>& future,
                            std::chrono::milliseconds timeout) {
  if (future.IsKnownReady()) {
    return true;
  }
  auto done = std::make_shared<absl::Notification>();
  future.OnReady([done](const absl::Status&) { done->Notify(); });
  return done->WaitForNotificationWithTimeout(absl::FromChrono(timeout));
}

// Returns the base buffers of `tensors`, one per DeviceBufferList.
absl::StatusOr<std::vector<DeviceBufferRef>> GetBaseBuffers(
    absl::Span<const at::Tensor> tensors) {
  std::vector<DeviceBufferRef> buffers;
  buffers.reserve(tensors.size());
  for (const at::Tensor& tensor : tensors) {
    TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer, GetBaseBuffer(tensor));
    if (absl::c_none_of(buffers, [&buffer](const DeviceBufferRef& existing) {
          return existing.device_buffer_list() == buffer.device_buffer_list();
        })) {
      buffers.push_back(std::move(buffer));
    }
  }
  return buffers;
}

// Materializes `buffers` and returns a future that is ready when they are
// ready on the device.
absl::StatusOr<xla::Future<void>> MaterializeAndGetReadyFuture(
    absl::Span<const DeviceBufferRef> buffers) {
  TT_RETURN_IF_ERROR(
      Materialize(buffers, MaterializationReason::kDistributedOp));
  std::vector<xla::Future<void>> ready_futures;
  ready_futures.reserve(buffers.size());
  for (const DeviceBufferRef& buffer : buffers) {
    ready_futures.push_back(buffer.GetReadyFuture());
  }
  return xla::JoinFutures(ready_futures);
}

// Flushes `stream` and returns a future that is ready when the work issued on
// it so far has completed.
absl::StatusOr<xla::Future<void>> FlushAndGetCompletionFuture(
    const DeviceStreamId& stream) {
  TT_ASSIGN_OR_RETURN(const EventSnapshot snapshot,
                      MaterializeStream(stream.device_index, stream.stream_id,
                                        MaterializationReason::kDistributedOp));
  auto [promise, future] = xla::MakePromise<void>();
  // The completion future of a snapshot is only valid once it has started.
  snapshot.OkCallback().OnReady(
      [promise = std::move(promise), snapshot](const absl::Status&) mutable {
        promise.Set(snapshot.completion_future().Await());
      });
  // The snapshot took the stream's failures; the next sync must report them.
  internal::MarkStreamActive(stream.device_index, stream.stream_id, future);
  return future;
}

class TpuWork : public c10d::Work {
 public:
  // A Work completed by `future` (P2P), or at once when `future` is null.
  explicit TpuWork(std::vector<at::Tensor> result_tensors, int rank = -1,
                   c10d::OpType op_type = c10d::OpType::UNKNOWN,
                   c10::intrusive_ptr<c10::ivalue::Future> future = nullptr)
      : Work(rank, op_type), results_(std::move(result_tensors)) {
    results_future_ = c10::make_intrusive<c10::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()));
    if (future) {
      future->addCallback([results_future = results_future_,
                           results = results_](c10::ivalue::Future& future) {
        if (future.hasError()) {
          results_future->setError(std::make_exception_ptr(TtError(
              TT_ERROR(error::kInternal) << future.tryRetrieveErrorMessage(),
              TT_SOURCE_LOCATION)));
        } else {
          results_future->markCompleted(c10::IValue(results));
        }
      });
    } else {
      results_future_->markCompleted(c10::IValue(results_));
    }
  }

  // A Work that tracks the collective that writes `outputs` on `stream`. With
  // no `completion` the collective is deferred and writes `deferred_buffers`.
  TpuWork(std::vector<at::Tensor> outputs, int rank, c10d::OpType op_type,
          DeviceStreamId stream, std::optional<xla::Future<void>> completion,
          absl::Span<const DeviceBufferRef> deferred_buffers = {})
      : Work(rank, op_type),
        results_(std::move(outputs)),
        issuing_stream_(stream),
        completion_(std::move(completion)) {
    deferred_buffers_.reserve(deferred_buffers.size());
    for (const DeviceBufferRef& buffer : deferred_buffers) {
      deferred_buffers_.push_back(buffer.weak());
    }
  }

  // This class is neither copyable nor movable.
  TpuWork(const TpuWork&) = delete;
  TpuWork& operator=(const TpuWork&) = delete;
  TpuWork(TpuWork&&) = delete;
  TpuWork& operator=(TpuWork&&) = delete;

  bool isCompleted() override {
    if (!issuing_stream_.has_value()) {
      return results_future_->completed();
    }
    return StatusIfCompleted().has_value();
  }

  bool isSuccess() const override {
    if (!issuing_stream_.has_value()) {
      return results_future_->completed() && !results_future_->hasError();
    }
    const std::optional<absl::Status> status = StatusIfCompleted();
    return status.has_value() && status->ok();
  }

  std::exception_ptr exception() const override {
    if (!issuing_stream_.has_value()) {
      return results_future_->exception_ptr();
    }
    const std::optional<absl::Status> status = StatusIfCompleted();
    if (!status.has_value() || status->ok()) {
      return nullptr;
    }
    return std::make_exception_ptr(TtError(*status, TT_SOURCE_LOCATION));
  }

  void synchronize() override {
    if (issuing_stream_.has_value()) {
      const DeviceStreamId current = GetCurrentDeviceStreamId();
      if (current.device_index != issuing_stream_->device_index ||
          current.stream_id != issuing_stream_->stream_id) {
        TT_ASSIGN_OR_THROW(xla::Future<void> completion, Completion());
        EventSnapshot snapshot(issuing_stream_->device_index,
                               issuing_stream_->stream_id);
        snapshot.Start(std::move(completion));
        InsertStreamWait(current.device_index, current.stream_id, snapshot);
      }
    }
    c10d::Work::synchronize();
  }

  void blockCurrentStream() override { synchronize(); }

  bool wait(std::chrono::milliseconds timeout) override {
    if (!issuing_stream_.has_value()) {
      results_future_->wait();
      TT_CHECK_THROW(!results_future_->hasError(), error::kInternal)
          << results_future_->tryRetrieveErrorMessage();
      return true;
    }
    synchronize();
    if (timeout > kNoTimeout) {
      TT_ASSIGN_OR_THROW(const xla::Future<void> completion, Completion());
      TT_CHECK_THROW(AwaitFutureWithTimeout(completion, timeout),
                     error::kDeadlineExceeded)
          << "timed out waiting for collective after " << timeout.count()
          << " ms";
      TT_THROW_IF_ERROR(completion.Await());
      return true;
    }
    if (opType_ == c10d::OpType::BARRIER) {
      TT_ASSIGN_OR_THROW(const xla::Future<void> completion, Completion());
      TT_THROW_IF_ERROR(completion.Await());
      return true;
    }
    // Reports a known failure without blocking the host or flushing the
    // stream, so that a deferred collective can still fuse with later ops.
    const std::optional<xla::Future<void>> completion = CompletionIfStarted();
    if (completion.has_value() && completion->IsKnownReady()) {
      TT_THROW_IF_ERROR(completion->Await());
    }
    return true;
  }

  std::vector<at::Tensor> result() override { return results_; }

  void abort() override {}

  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    if (!issuing_stream_.has_value()) {
      return results_future_;
    }
    // As on a GPU stream, the future is complete once the collective is issued.
    absl::MutexLock lock(mu_);
    if (results_future_ == nullptr) {
      results_future_ = c10::make_intrusive<c10::ivalue::Future>(
          c10::ListType::create(c10::TensorType::get()));
      results_future_->markCompleted(c10::IValue(results_));
    }
    return results_future_;
  }

 private:
  // Returns the collective's completion, starting the collective if needed.
  absl::StatusOr<xla::Future<void>> Completion() const
      ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(mu_);
    if (!completion_.has_value()) {
      TT_ASSIGN_OR_RETURN(completion_, StartDeferredCollective());
      deferred_buffers_.clear();
    }
    return *completion_;
  }

  // Starts the deferred collective by flushing its stream. Returns a future
  // that is ready when the collective has completed.
  absl::StatusOr<xla::Future<void>> StartDeferredCollective() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    std::vector<DeviceBufferRef> buffers;
    buffers.reserve(deferred_buffers_.size());
    for (const DeviceBufferRef::Weak& deferred_buffer : deferred_buffers_) {
      std::optional<DeviceBufferRef> buffer = deferred_buffer.lock();
      if (!buffer.has_value() || buffer->live_data_ptrs() == 0) {
        // No tensor holds what the collective writes any more (updated in
        // place, overwritten or released), so track its stream instead.
        return FlushAndGetCompletionFuture(*issuing_stream_);
      }
      buffers.push_back(std::move(*buffer));
    }
    return MaterializeAndGetReadyFuture(buffers);
  }

  // Returns the collective's status if it has completed, starting it first if
  // needed. Never blocks or throws.
  std::optional<absl::Status> StatusIfCompleted() const
      ABSL_LOCKS_EXCLUDED(mu_) {
    const absl::StatusOr<xla::Future<void>> completion = Completion();
    if (!completion.ok() || !completion->IsKnownReady()) {
      return std::nullopt;
    }
    return completion->Await();
  }

  // Returns the collective's completion if the collective has been started.
  std::optional<xla::Future<void>> CompletionIfStarted() const
      ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(mu_);
    return completion_;
  }

  std::vector<at::Tensor> results_;
  // Null for a collective Work until getFuture() is called.
  c10::intrusive_ptr<c10::ivalue::Future> results_future_;
  // Set for a collective Work; unset for P2P and completed-at-once Works.
  std::optional<DeviceStreamId> issuing_stream_;
  mutable absl::Mutex mu_;
  // What a deferred collective writes, until the collective is started.
  mutable std::vector<DeviceBufferRef::Weak> deferred_buffers_
      ABSL_GUARDED_BY(mu_);
  // Mutable: the const queries of c10d::Work also start a deferred collective.
  mutable std::optional<xla::Future<void>> completion_ ABSL_GUARDED_BY(mu_);

  friend class ::torch_tpu::ProcessGroupTpu;  // To allow changing op type.
};

// Returns the Work of a collective that produced `result_tensors` backed by
// explicit `buffers`. During an FX trace nothing runs on the device, so the
// Work is complete at once.
c10::intrusive_ptr<c10d::Work> MakeCollectiveWorkWithBuffers(
    std::vector<at::Tensor> result_tensors,
    std::vector<DeviceBufferRef> buffers, const int rank,
    const c10d::OpType op_type) {
  const EagerMode eager_mode = GetEagerMode();
  if (eager_mode == EagerMode::kInternalCompileFxGraph ||
      result_tensors.empty()) {
    return c10::make_intrusive<TpuWork>(std::move(result_tensors), rank,
                                        op_type);
  }
  const DeviceStreamId stream = GetCurrentDeviceStreamId();
  if (!IsDeferNeverMode(eager_mode)) {
    // A deferred collective starts like any deferred op: at the next flush of
    // its stream, or when its Work needs its completion.
    return c10::make_intrusive<TpuWork>(std::move(result_tensors), rank,
                                        op_type, stream, std::nullopt,
                                        std::move(buffers));
  }
  // The collective has started. Its completion is taken now: a later in-place
  // update donates the buffers that it wrote.
  TT_ASSIGN_OR_THROW(xla::Future<void> completion,
                     MaterializeAndGetReadyFuture(buffers));
  return c10::make_intrusive<TpuWork>(std::move(result_tensors), rank, op_type,
                                      stream, std::move(completion));
}

// Returns the Work of a collective that produced `result_tensors`. During an FX
// trace nothing runs on the device, so the Work is complete at once.
c10::intrusive_ptr<c10d::Work> MakeCollectiveWork(
    std::vector<at::Tensor> result_tensors, const int rank,
    const c10d::OpType op_type) {
  const EagerMode eager_mode = GetEagerMode();
  if (eager_mode == EagerMode::kInternalCompileFxGraph ||
      result_tensors.empty()) {
    return c10::make_intrusive<TpuWork>(std::move(result_tensors), rank,
                                        op_type);
  }
  TT_ASSIGN_OR_THROW(std::vector<DeviceBufferRef> buffers,
                     GetBaseBuffers(result_tensors));
  return MakeCollectiveWorkWithBuffers(std::move(result_tensors),
                                       std::move(buffers), rank, op_type);
}

std::string DeviceBufferRefDebugString(const at::Tensor& input) {
  auto maybe_base_buffer_ref = GetBaseBuffer(input);
  if (!maybe_base_buffer_ref.ok()) {
    return std::string(maybe_base_buffer_ref.status().message());
  }
  return maybe_base_buffer_ref->DebugString();
}

absl::Status ValidateTensorsUniformShape(
    const std::vector<at::Tensor>& tensors) {
  if (tensors.empty()) {
    return absl::OkStatus();
  }

  const auto& first_tensor_size = tensors[0].sizes();
  for (int i = 1; i < tensors.size(); ++i) {
    TT_RET_CHECK(tensors[i].sizes() == first_tensor_size,
                 error::kInvalidArgument)
        << "expected all tensors to have the same shape, got "
        << first_tensor_size << " at index 0 and " << tensors[i].sizes()
        << " at index " << i;
  }

  return absl::OkStatus();
}

enum class ReduceScatterShapeMode {
  kConcat,  // Input tensor is concatenated along dimension 0
  kStack,   // Input tensor is stacked along a new dimension (at position 0)
};

// Helper to validate the reduce-scatter input tensor shape, and determine if
// input tensor is concatenated or stacked (PyTorch API allows both).
absl::StatusOr<ReduceScatterShapeMode> ValidateReduceScatterInputTensorShape(
    at::Tensor& output, at::Tensor& input, size_t process_group_size) {
  // Concat mode: same shape as output, but dim=0 is pg_size times larger:
  Dimensions concat_mode_dims;
  if (output.dim() == 0) {
    concat_mode_dims = {static_cast<int64_t>(process_group_size)};
  } else {
    concat_mode_dims = CopyIntVector(output.sizes());
    concat_mode_dims[0] *= process_group_size;
  }
  ABSL_VLOG(1) << "ValidateReduceScatterInputTensorShape: concat_mode_dims: "
               << ToString(concat_mode_dims);

  // Stack mode: prepend new dimension of size pg_size, and remaining dims same.
  Dimensions stack_mode_dims(output.dim() + 1);
  stack_mode_dims[0] = process_group_size;
  for (int i = 0; i < output.dim(); ++i) {
    stack_mode_dims[i + 1] = output.size(i);
  }
  ABSL_VLOG(1) << "ValidateReduceScatterInputTensorShape: stack_mode_dims: "
               << ToString(stack_mode_dims);

  Dimensions actual_input_dims = CopyIntVector(input.sizes());
  ABSL_VLOG(1) << "ValidateReduceScatterInputTensorShape: actual_input_dims: "
               << ToString(actual_input_dims);

  if (actual_input_dims == concat_mode_dims) {
    return ReduceScatterShapeMode::kConcat;
  } else if (actual_input_dims == stack_mode_dims) {
    return ReduceScatterShapeMode::kStack;
  } else {
    return TT_ERROR(error::kInvalidArgument)
           << "expected the input tensor to be either of shape "
           << ToString(concat_mode_dims) << " or of shape "
           << ToString(stack_mode_dims) << ", got "
           << ToString(actual_input_dims);
  }
}

absl::Status ValidateSplitSizesForAllToAllSingle(c10::IntArrayRef split_sizes,
                                                 const at::Tensor& tensor,
                                                 size_t group_size) {
  auto dim0 = tensor.sizes()[0];
  if (split_sizes.empty()) {
    TT_RET_CHECK(dim0 % group_size == 0, error::kInvalidArgument)
        << "expected tensor first dimension to be divisible by process group "
        << "size, got " << dim0 << " is not divisible by " << group_size;
  } else {
    TT_RET_CHECK(split_sizes.size() == group_size, error::kInvalidArgument)
        << "expected split sizes to have the same size as process group size, "
        << "got " << split_sizes.size() << " for split sizes and " << group_size
        << " for process group size";

    for (int64_t size : split_sizes) {
      TT_RET_CHECK(size >= 0, error::kInvalidArgument)
          << "expected split sizes to be >= 0, got " << size;
    }

    const int64_t split_sizes_sum = absl::c_accumulate(split_sizes, 0L);
    TT_RET_CHECK(split_sizes_sum == dim0, error::kInvalidArgument)
        << "expected split sizes sum to be equal to tensor first dimension, "
        << "got " << split_sizes_sum << " for split sizes sum and " << dim0
        << " for tensor shape " << tensor.sizes() << " dim 0";
  }
  return absl::OkStatus();
}

// Validates tensor list invariants for all-to-all: matching process group
// sizes, matching dtypes, matching dimension counts, and matching trailing
// feature dimensions (dim 1 onwards).
absl::Status ValidateAllToAllTensors(
    const std::vector<at::Tensor>& output_tensors,
    const std::vector<at::Tensor>& input_tensors) {
  ABSL_CHECK_EQ(output_tensors.size(), input_tensors.size());  // CRASH_OK
  ABSL_CHECK_GT(input_tensors.size(), 0);                      // CRASH_OK

  const at::ScalarType dtype = input_tensors[0].scalar_type();
  const int64_t dim = input_tensors[0].dim();
  const auto trailing_dims =
      dim > 0 ? input_tensors[0].sizes().slice(1) : c10::IntArrayRef();

  for (size_t i = 0; i < input_tensors.size(); ++i) {
    TT_RET_CHECK(input_tensors[i].scalar_type() == dtype,
                 error::kInvalidArgument)
        << "expected all input tensors to have the same dtype, got "
        << ToString(input_tensors[i].scalar_type()) << " and "
        << ToString(dtype);
    TT_RET_CHECK(output_tensors[i].scalar_type() == dtype,
                 error::kInvalidArgument)
        << "expected all output tensors to have the same dtype, got "
        << ToString(output_tensors[i].scalar_type()) << " and "
        << ToString(dtype);

    TT_RET_CHECK(input_tensors[i].dim() == dim, error::kInvalidArgument)
        << "expected all input tensors to have the same number of dimensions, "
        << "got " << input_tensors[i].dim() << " and " << dim;
    TT_RET_CHECK(output_tensors[i].dim() == dim, error::kInvalidArgument)
        << "expected all output tensors to have the same number of "
        << "dimensions, got " << output_tensors[i].dim() << " and " << dim;

    if (dim > 0) {
      TT_RET_CHECK(input_tensors[i].sizes().slice(1) == trailing_dims,
                   error::kInvalidArgument)
          << "expected trailing dimensions of input tensors to match, got "
          << input_tensors[i].sizes() << " and " << input_tensors[0].sizes();
      TT_RET_CHECK(output_tensors[i].sizes().slice(1) == trailing_dims,
                   error::kInvalidArgument)
          << "expected trailing dimensions of output tensors to match, got "
          << output_tensors[i].sizes() << " and " << input_tensors[0].sizes();
    }
  }
  return absl::OkStatus();
}

// Returns true if all input and output tensors have identical shapes, enabling
// the uniform all-to-all lowering path without ragged collective slicing.
bool IsUniformAllToAll(const std::vector<at::Tensor>& output_tensors,
                       const std::vector<at::Tensor>& input_tensors) {
  const auto first_shape = input_tensors[0].sizes();
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    if (input_tensors[i].sizes() != first_shape ||
        output_tensors[i].sizes() != first_shape) {
      return false;
    }
  }
  return true;
}

// Converts `c10::elementSize()` byte counts into bit widths for
// `Select1DRaggedAllToAllFactor`.
constexpr int kBitsPerByte = 8;

enum class RaggedBufferLayout {
  kSymmetric,
  kAsymmetric,
};

// Local offsets, slice sizes, and group-agreed attributes for uneven
// `all_to_all_single`.
struct RaggedAllToAllOffsets {
  std::vector<int32_t> input_offsets;
  std::vector<int32_t> send_sizes;
  std::vector<int32_t> output_offsets;
  std::vector<int32_t> recv_sizes;
  int64_t flat_factor = 1;
  RaggedBufferLayout buffer_layout = RaggedBufferLayout::kSymmetric;
  bool uniform_equal_splits = false;
};

// Per-rank buffer metadata exchanged on the host to validate cross-rank splits
// and agree on SPMD compilation attributes.
struct RaggedAllToAllBufferInfo {
  int32_t input_buffer_size = 0;
  int32_t output_buffer_size = 0;
  int64_t dispatch_input_size = 0;
  int64_t dispatch_output_size = 0;
  c10::ScalarType scalar_type = c10::ScalarType::Undefined;
  int element_bitwidth = 0;
  int32_t tensor_dim = 0;
  uint64_t trailing_shape_fingerprint = 0;
};

struct SynchronizedRaggedAllToAllMetadata {
  std::vector<int32_t> output_offsets;
  int64_t group_gcd = 0;
  RaggedBufferLayout buffer_layout = RaggedBufferLayout::kSymmetric;
  bool uniform_equal_splits = false;
};

struct UnevenMetadataPayloadLayout {
  size_t recv_sizes;
  size_t send_sizes;
  size_t in_cap;
  size_t out_cap;
  size_t local_gcd;
  size_t scalar_type;
  size_t tensor_dim;
  size_t trailing_shape_fp_lo;
  size_t trailing_shape_fp_hi;
  size_t total_ints;
};

absl::StatusOr<std::vector<std::vector<uint8_t>>>
ExchangeUnevenAllToAllPayloads(c10d::Store* store, int my_rank,
                               size_t group_size, uint64_t seq,
                               AllToAllShmNonceCache& shm_nonce_cache,
                               const std::vector<uint8_t>& my_bytes);

absl::StatusOr<int32_t> ValidateDimToInt32(const int64_t value,
                                           const std::string_view label) {
  // RaggedAllToAll metadata uses 32-bit offsets and counts, so reject sizes
  // that would silently wrap before narrowing.
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Allocating >INT32_MAX elements per
                 // rank exceeds test TPU HBM.
      value >= 0 && value <= std::numeric_limits<int32_t>::max(),
      error::kInvalidArgument)
      << label << " (" << value << ") exceeds maximum supported int32_t size ("
      << std::numeric_limits<int32_t>::max() << ")";
  return static_cast<int32_t>(value);
}

bool PeerSplitsEqual(const absl::Span<const int32_t> splits,
                     const int32_t expected_size) {
  for (const int32_t split_size : splits) {
    if (split_size != expected_size) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<std::vector<int32_t>> BuildAllToAllSplitSizes(
    const c10::IntArrayRef split_sizes, const int64_t total_dim0,
    const size_t group_size) {
  // PyTorch allows omitting one split vector while the other is uneven, so
  // expand empty split lists into explicit uniform counts for validation.
  if (split_sizes.empty()) {
    TT_ASSIGN_OR_RETURN(
        const int32_t chunk,
        ValidateDimToInt32(total_dim0 / static_cast<int64_t>(group_size),
                           "uniform split size"));
    return std::vector<int32_t>(group_size, chunk);
  }
  std::vector<int32_t> result(group_size, 0);
  for (size_t i = 0; i < group_size; ++i) {
    TT_ASSIGN_OR_RETURN(result[i],
                        ValidateDimToInt32(split_sizes[i], "split size"));
  }
  return result;
}

int RaggedAllToAllElementBitwidth(const c10::ScalarType scalar_type) {
  // Bool, complex, quantized, and packed FP4 tensors do not map to contiguous
  // 8-, 16-, or 32-bit `(8, 128)` TPU tiles, so return 0 to disable factoring.
  if (scalar_type == c10::kBool || c10::isComplexType(scalar_type) ||
      c10::isQIntType(scalar_type) || scalar_type == c10::kFloat4_e2m1fn_x2) {
    return 0;
  }
  return static_cast<int>(c10::elementSize(scalar_type) * kBitsPerByte);
}

UnevenMetadataPayloadLayout MakeUnevenMetadataPayloadLayout(
    const size_t group_size) {
  return UnevenMetadataPayloadLayout{
      .recv_sizes = 0,
      .send_sizes = group_size,
      .in_cap = 2 * group_size,
      .out_cap = 2 * group_size + 1,
      .local_gcd = 2 * group_size + 2,
      .scalar_type = 2 * group_size + 3,
      .tensor_dim = 2 * group_size + 4,
      .trailing_shape_fp_lo = 2 * group_size + 5,
      .trailing_shape_fp_hi = 2 * group_size + 6,
      .total_ints = 2 * group_size + 7,
  };
}

std::vector<uint8_t> SerializeUnevenMetadataPayload(
    const UnevenMetadataPayloadLayout& layout,
    const absl::Span<const int32_t> send_sizes,
    const absl::Span<const int32_t> recv_sizes,
    const RaggedAllToAllBufferInfo& buffer_info, const int64_t local_gcd) {
  std::vector<int32_t> payload(layout.total_ints);
  absl::c_copy(recv_sizes, payload.begin() + layout.recv_sizes);
  absl::c_copy(send_sizes, payload.begin() + layout.send_sizes);
  payload[layout.in_cap] = buffer_info.input_buffer_size;
  payload[layout.out_cap] = buffer_info.output_buffer_size;
  payload[layout.local_gcd] = static_cast<int32_t>(local_gcd);
  payload[layout.scalar_type] = static_cast<int32_t>(buffer_info.scalar_type);
  payload[layout.tensor_dim] = buffer_info.tensor_dim;
  payload[layout.trailing_shape_fp_lo] = static_cast<int32_t>(
      static_cast<uint32_t>(buffer_info.trailing_shape_fingerprint));
  payload[layout.trailing_shape_fp_hi] = static_cast<int32_t>(
      static_cast<uint32_t>(buffer_info.trailing_shape_fingerprint >> 32));

  const auto* payload_bytes = reinterpret_cast<const uint8_t*>(payload.data());
  return std::vector<uint8_t>(payload_bytes,
                              payload_bytes + payload.size() * sizeof(int32_t));
}

absl::Status ValidateAndReducePeerPayload(
    const size_t peer, const int my_rank, const size_t group_size,
    const int32_t uniform_chunk, const UnevenMetadataPayloadLayout& layout,
    const absl::Span<const uint8_t> peer_val,
    const absl::Span<const int32_t> send_sizes,
    const absl::Span<const int32_t> recv_sizes,
    const RaggedAllToAllBufferInfo& buffer_info,
    SynchronizedRaggedAllToAllMetadata& result) {
  const size_t expected_payload_bytes = layout.total_ints * sizeof(int32_t);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Peers exchange fixed-size payloads.
      peer_val.size() == expected_payload_bytes, error::kInternal)
      << "peer " << peer << " payload size mismatch: " << peer_val.size();
  const absl::Span<const int32_t> peer_data(
      reinterpret_cast<const int32_t*>(peer_val.data()), layout.total_ints);

  // Verify both directions of the pairwise contract so every rank fails fast
  // before dispatching mismatched RDMAs onto the TPU.
  const int32_t peer_expected_recv = peer_data[layout.recv_sizes + my_rank];
  TT_RET_CHECK(peer_expected_recv == send_sizes[peer], error::kInvalidArgument)
      << "send and recv slice size mismatch between rank " << my_rank
      << " and peer rank " << peer << ": rank " << my_rank << " sends "
      << send_sizes[peer] << " elements, but peer rank " << peer
      << " expects to receive " << peer_expected_recv;

  const int32_t peer_send_to_me = peer_data[layout.send_sizes + my_rank];
  TT_RET_CHECK(peer_send_to_me == recv_sizes[peer], error::kInvalidArgument)
      << "send and recv slice size mismatch between peer rank " << peer
      << " and rank " << my_rank << ": peer rank " << peer << " sends "
      << peer_send_to_me << " elements, but rank " << my_rank
      << " expects to receive " << recv_sizes[peer];

  // PyTorch only validates input vs. output dtype on the local rank, so check
  // cross-rank agreement here before emitting an SPMD collective.
  const auto peer_scalar_type =
      static_cast<c10::ScalarType>(peer_data[layout.scalar_type]);
  TT_RET_CHECK(peer_scalar_type == buffer_info.scalar_type,
               error::kInvalidArgument)
      << "expected matching element dtype across ranks, got "
      << ToString(buffer_info.scalar_type) << " on rank " << my_rank << " and "
      << ToString(peer_scalar_type) << " on peer rank " << peer;

  // All ranks must agree on tensor rank and trailing dimensions so every
  // replica chooses the same 1D factoring factor and SPMD HLO shape.
  const int32_t peer_tensor_dim = peer_data[layout.tensor_dim];
  const uint64_t peer_trailing_fp =
      (static_cast<uint64_t>(
           static_cast<uint32_t>(peer_data[layout.trailing_shape_fp_hi]))
       << 32) |
      static_cast<uint64_t>(
          static_cast<uint32_t>(peer_data[layout.trailing_shape_fp_lo]));
  TT_RET_CHECK(peer_tensor_dim == buffer_info.tensor_dim &&
                   peer_trailing_fp == buffer_info.trailing_shape_fingerprint,
               error::kInvalidArgument)
      << "expected matching tensor rank and trailing dimensions across ranks "
         "for all_to_all_single, got rank "
      << buffer_info.tensor_dim << " on rank " << my_rank << " and rank "
      << peer_tensor_dim << " on peer rank " << peer;

  // All ranks must agree on whether capacities are symmetric and whether every
  // peer's splits are uniform so every replica selects the same collective op.
  const int32_t peer_input_size = peer_data[layout.in_cap];
  const int32_t peer_output_size = peer_data[layout.out_cap];
  if (peer_input_size != buffer_info.input_buffer_size ||
      peer_output_size != buffer_info.output_buffer_size) {
    result.buffer_layout = RaggedBufferLayout::kAsymmetric;
    result.uniform_equal_splits = false;
  } else if (result.uniform_equal_splits &&
             (!PeerSplitsEqual(peer_data.subspan(layout.send_sizes, group_size),
                               uniform_chunk) ||
              !PeerSplitsEqual(peer_data.subspan(layout.recv_sizes, group_size),
                               uniform_chunk))) {
    result.uniform_equal_splits = false;
  }

  // `ragged_all_to_all` writes rank `my_rank`'s slice into `peer`'s output
  // buffer directly after slices received from ranks `0..my_rank - 1`.
  result.output_offsets[peer] = absl::c_accumulate(
      peer_data.subspan(layout.recv_sizes, my_rank), int32_t{0});
  // Reduce `group_gcd` across every peer so all ranks pick the exact same 1D
  // factoring factor and emit identical SPMD tensor ranks.
  result.group_gcd = std::gcd(
      result.group_gcd, static_cast<int64_t>(peer_data[layout.local_gcd]));
  return absl::OkStatus();
}

// Validates pairwise split sizes and dtypes across ranks, reduces SPMD
// compilation invariants (`group_gcd`, `uniform_equal_splits`, and
// `buffer_layout`), and derives remote `output_offsets` from peer `recv_sizes`.
absl::StatusOr<SynchronizedRaggedAllToAllMetadata>
SynchronizeRaggedAllToAllMetadata(c10d::Store* const store, const int my_rank,
                                  std::atomic<uint64_t>& seq_counter,
                                  AllToAllShmNonceCache& shm_nonce_cache,
                                  const absl::Span<const int32_t> send_sizes,
                                  const absl::Span<const int32_t> recv_sizes,
                                  const RaggedAllToAllBufferInfo& buffer_info,
                                  const int64_t local_gcd) {
  const size_t group_size = send_sizes.size();
  SynchronizedRaggedAllToAllMetadata result{
      .output_offsets = std::vector<int32_t>(group_size, 0),
      .group_gcd = local_gcd,
  };
  // A single-rank group has no peers to validate or coordinate with.
  if (group_size <= 1) {
    return result;
  }
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Store is required for multi-rank setup.
      store != nullptr, error::kInvalidArgument)
      << "c10d::Store is required for multi-rank uneven all-to-all";

  const int64_t total_send_size = absl::c_accumulate(send_sizes, int64_t{0});
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Checked in ValidateSplitSizes.
      total_send_size <= buffer_info.input_buffer_size, error::kInvalidArgument)
      << "expected sum of send_sizes to be <= input buffer size, got "
      << total_send_size << " and " << buffer_info.input_buffer_size;
  const int64_t total_recv_size = absl::c_accumulate(recv_sizes, int64_t{0});
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Checked in ValidateSplitSizes.
      total_recv_size <= buffer_info.output_buffer_size,
      error::kInvalidArgument)
      << "expected sum of recv_sizes to be <= output buffer size, got "
      << total_recv_size << " and " << buffer_info.output_buffer_size;
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Buffer sizes fit in int32_t.
      local_gcd >= 0 && local_gcd <= std::numeric_limits<int32_t>::max(),
      error::kInternal)
      << "1D all-to-all local gcd out of int32 range: " << local_gcd;

  const uint64_t seq = seq_counter.fetch_add(1, std::memory_order_relaxed);
  const UnevenMetadataPayloadLayout layout =
      MakeUnevenMetadataPayloadLayout(group_size);
  const std::vector<uint8_t> my_bytes = SerializeUnevenMetadataPayload(
      layout, send_sizes, recv_sizes, buffer_info, local_gcd);

  TT_ASSIGN_OR_RETURN(
      const std::vector<std::vector<uint8_t>> peer_values,
      ExchangeUnevenAllToAllPayloads(store, my_rank, group_size, seq,
                                     shm_nonce_cache, my_bytes));
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Store exchange returns group_size rows.
      peer_values.size() == group_size, error::kInternal)
      << "expected " << group_size << " peer values, got "
      << peer_values.size();

  // Only non-empty, evenly divisible symmetric buffers can lower to the
  // uniform `stablehlo.all_to_all` fast path.
  const bool can_be_uniform_equal =
      buffer_info.input_buffer_size > 0 &&
      buffer_info.input_buffer_size == buffer_info.output_buffer_size &&
      buffer_info.input_buffer_size % static_cast<int32_t>(group_size) == 0;
  const int32_t uniform_chunk =
      can_be_uniform_equal
          ? buffer_info.input_buffer_size / static_cast<int32_t>(group_size)
          : 0;
  result.uniform_equal_splits = can_be_uniform_equal;

  // Reduce across every peer (including `my_rank`) so all ranks compute the
  // exact same `buffer_layout`, `uniform_equal_splits`, and `group_gcd`.
  for (size_t peer = 0; peer < group_size; ++peer) {
    TT_RETURN_IF_ERROR(ValidateAndReducePeerPayload(
        peer, my_rank, group_size, uniform_chunk, layout, peer_values[peer],
        send_sizes, recv_sizes, buffer_info, result));
  }
  return result;
}

absl::StatusOr<RaggedAllToAllOffsets> ComputeRaggedAllToAllOffsets(
    c10d::Store* const store, const int rank,
    std::atomic<uint64_t>& seq_counter, AllToAllShmNonceCache& shm_nonce_cache,
    const absl::Span<const int32_t> send_sizes,
    const absl::Span<const int32_t> recv_sizes,
    const RaggedAllToAllBufferInfo& buffer_info) {
  const size_t group_size = send_sizes.size();
  RaggedAllToAllOffsets offsets{
      .input_offsets = std::vector<int32_t>(group_size, 0),
      .send_sizes = std::vector<int32_t>(send_sizes.begin(), send_sizes.end()),
      .recv_sizes = std::vector<int32_t>(recv_sizes.begin(), recv_sizes.end()),
  };
  std::vector<int32_t> local_output_offsets(group_size, 0);
  int32_t in_offset = 0;
  int32_t out_offset = 0;
  // Accumulate prefix sums so each peer slice starts immediately after prior
  // ranks. Include `local_output_offsets` in `local_gcd` so remote write
  // offsets stay aligned to 1024-element tiles when 1D factoring is active.
  for (size_t i = 0; i < group_size; ++i) {
    offsets.input_offsets[i] = in_offset;
    in_offset += send_sizes[i];
    local_output_offsets[i] = out_offset;
    out_offset += recv_sizes[i];
  }

  const int64_t local_gcd = Compute1DRaggedAllToAllLocalGcd(
      buffer_info.dispatch_input_size, buffer_info.dispatch_output_size,
      offsets.input_offsets, send_sizes, local_output_offsets, recv_sizes);

  TT_ASSIGN_OR_RETURN(SynchronizedRaggedAllToAllMetadata synchronized,
                      SynchronizeRaggedAllToAllMetadata(
                          store, rank, seq_counter, shm_nonce_cache, send_sizes,
                          recv_sizes, buffer_info, local_gcd));
  offsets.output_offsets = std::move(synchronized.output_offsets);
  offsets.buffer_layout = synchronized.buffer_layout;
  offsets.uniform_equal_splits = synchronized.uniform_equal_splits;
  offsets.flat_factor = Select1DRaggedAllToAllFactor(
      synchronized.group_gcd, buffer_info.element_bitwidth);

  // Reject unfactored asymmetric buffers because differing per-rank capacities
  // can diverge in XLA:TPU HBM layout padding, whereas 1024-element factored
  // 3D shapes share identical `(8, 128)` tile strides across all ranks.
  TT_RET_CHECK(synchronized.buffer_layout != RaggedBufferLayout::kAsymmetric ||
                   offsets.flat_factor == k1DRaggedAllToAllTileElements,
               error::kInvalidArgument)
      << "asymmetric all_to_all_single buffer sizes across ranks (rank " << rank
      << ": input " << buffer_info.input_buffer_size << ", output "
      << buffer_info.output_buffer_size
      << ") are only supported on TPU for 1D tensors whose sizes, split "
         "sizes, and offsets on every rank are multiples of "
      << k1DRaggedAllToAllTileElements << " elements";
  return offsets;
}

// Populates kernel cache parameter keys for ragged all-to-all collectives.
absl::Status SetRaggedAllToAllParamKeys(OpParamCacheKeys& param_keys,
                                        const RaggedAllToAllOffsets& offsets) {
  TT_RETURN_IF_ERROR(param_keys.SetParam("flat_factor", offsets.flat_factor));
  TT_RETURN_IF_ERROR(
      param_keys.SetParam("input_offsets", offsets.input_offsets));
  TT_RETURN_IF_ERROR(param_keys.SetParam("send_sizes", offsets.send_sizes));
  TT_RETURN_IF_ERROR(
      param_keys.SetParam("output_offsets", offsets.output_offsets));
  TT_RETURN_IF_ERROR(param_keys.SetParam("recv_sizes", offsets.recv_sizes));
  return absl::OkStatus();
}

}  // namespace

OpSplitMode GetCollectiveSplitMode() {
  if (IsSpmdSafe()) {
    return OpSplitMode::kNone;
  }

  if (GetBooleanEnvOnce<kTorchTpuInternalMaterializeCollectiveTensorsEnvVar>()
          .value_or(true)) {
    return OpSplitMode::kSplitBoth;
  }
  return OpSplitMode::kNone;
}

xla::CrossHostTransferKey GetCrossHostTransferKey(int64_t src_device_id,
                                                  int64_t dst_device_id,
                                                  int tag,
                                                  size_t tensor_index) {
  const FingerprintType fp =
      FingerprintCat(src_device_id, dst_device_id, tag, tensor_index);
  return xla::CrossHostTransferKey(static_cast<int64_t>(fp));
}

namespace {

std::string GetCrossHostTransferDescriptorStoreKey(
    xla::CrossHostTransferKey key) {
  static constexpr std::string_view kKeyPrefix =
      "cross_host_transfer_descriptor:";
  return absl::StrCat(kKeyPrefix, key.value());
}

std::string SerializeDeviceAttributes(const PjRtDeviceAttributes& attrs) {
  const absl::Span<const int64_t> coords =
      attrs.coords.has_value() ? absl::MakeConstSpan(*attrs.coords)
                               : absl::Span<const int64_t>();
  int64_t x = !coords.empty() ? coords[0] : 0;
  int64_t y = coords.size() > 1 ? coords[1] : 0;
  int64_t z = coords.size() > 2 ? coords[2] : 0;
  return absl::StrFormat("%d,%d,%d,%d,%d,%d,%d,%s", attrs.id,
                         attrs.process_index, x, y, z,
                         attrs.core_on_chip.value_or(0),
                         attrs.slice_index.value_or(-1), attrs.device_kind);
}

absl::StatusOr<PjRtDeviceAttributes> DeserializeDeviceAttributes(
    std::string_view payload) {
  std::vector<std::string_view> parts =
      absl::StrSplit(payload, absl::MaxSplits(',', 7));
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=defensive check on internally
                 // serialized store payload
      parts.size() == 8, error::kInternal)
      << "cannot parse device attributes: " << payload;
  PjRtDeviceAttributes attrs;
  int64_t x = 0, y = 0, z = 0, core = 0, slice = -1;
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=defensive check on internally
                 // serialized store payload
      absl::SimpleAtoi(parts[0], &attrs.id) &&
          absl::SimpleAtoi(parts[1], &attrs.process_index) &&
          absl::SimpleAtoi(parts[2], &x) && absl::SimpleAtoi(parts[3], &y) &&
          absl::SimpleAtoi(parts[4], &z) && absl::SimpleAtoi(parts[5], &core) &&
          absl::SimpleAtoi(parts[6], &slice),
      error::kInternal)
      << "cannot parse integer fields in device attributes: " << payload;
  attrs.coords = {x, y, z};
  attrs.core_on_chip = core;
  if (slice >= 0) {
    attrs.slice_index = slice;
  }
  attrs.device_kind = std::string(parts[7]);
  return attrs;
}

}  // namespace

P2pSendQueue::P2pSendQueue(c10::intrusive_ptr<c10d::Store> store)
    : store_(std::move(store)) {}

P2pSendQueue::~P2pSendQueue() { Shutdown(); }

absl::StatusOr<P2pSendQueue::PeerChannel*> P2pSendQueue::GetOrCreateChannel(
    int dst_rank) {
  auto it = channels_.find(dst_rank);
  if (it != channels_.end()) {
    return it->second.get();
  }
  c10::intrusive_ptr<c10d::Store> cloned_store;
  try {
    cloned_store = store_->clone();
    cloned_store->setTimeout(store_->getTimeout());
  } catch (const std::exception& e) {
    return TT_ERROR(  // ERROR_COV_INFEASIBLE=store clone failure
               error::kInternal)
           << "failed to clone store for p2p send: " << e.what();
  }
  auto channel = std::make_unique<PeerChannel>(std::move(cloned_store));
  channel->worker = std::thread(&P2pSendQueue::PeerWorkerLoop, channel.get());
  PeerChannel* ptr = channel.get();
  channels_.emplace(dst_rank, std::move(channel));
  return ptr;
}

absl::Status P2pSendQueue::Enqueue(int dst_rank,
                                   std::vector<std::string> descriptor_keys,
                                   ReadyCallback on_ready,
                                   ErrorCallback on_error) {
  absl::MutexLock lock(mu_);
  if (shutdown_) {
    return TT_ERROR(  // ERROR_COV_INFEASIBLE=shutdown race during teardown
               error::kCancelled)
           << "process group is shut down";
  }
  TT_ASSIGN_OR_RETURN(PeerChannel * channel, GetOrCreateChannel(dst_rank));
  {
    absl::MutexLock channel_lock(channel->mu);
    channel->queue.push_back(PendingSend{
        .descriptor_keys = std::move(descriptor_keys),
        .on_ready = std::move(on_ready),
        .on_error = std::move(on_error),
    });
    channel->cv.Signal();
  }
  return absl::OkStatus();
}

void P2pSendQueue::Shutdown() {
  absl::flat_hash_map<int, std::unique_ptr<PeerChannel>> channels;
  {
    absl::MutexLock lock(mu_);
    if (shutdown_) {
      return;
    }
    shutdown_ = true;
    channels = std::move(channels_);
  }

  std::vector<std::string> keys_to_wake;
  for (const auto& [rank, channel] : channels) {
    {
      absl::MutexLock channel_lock(channel->mu);
      channel->shutdown = true;
      keys_to_wake = channel->inflight_keys;
      channel->cv.SignalAll();
    }
    WakeInflightWaiters(keys_to_wake);
  }

  for (const auto& [rank, channel] : channels) {
    if (channel->worker.joinable() &&
        channel->worker.get_id() != std::this_thread::get_id()) {
      channel->worker.join();
    }
  }
}

void P2pSendQueue::CancelQueuedTasks(std::deque<PendingSend>& tasks) {
  for (auto& task : tasks) {
    std::move(task.on_error)(std::make_exception_ptr(TtError(
        TT_ERROR(  // ERROR_COV_INFEASIBLE=shutdown race with queued send
            error::kCancelled)
            << "process group is shut down",
        TT_SOURCE_LOCATION)));
  }
  tasks.clear();
}

bool P2pSendQueue::PopNextOrWait(PeerChannel* channel, PendingSend& out_task) {
  absl::MutexLock lock(channel->mu);
  while (channel->queue.empty() && !channel->shutdown) {
    channel->cv.Wait(&channel->mu);
  }
  if (channel->shutdown) {
    CancelQueuedTasks(channel->queue);
    return false;
  }
  out_task = std::move(channel->queue.front());
  channel->queue.pop_front();
  channel->inflight_keys = out_task.descriptor_keys;
  return true;
}

void P2pSendQueue::DeleteConsumedKeys(c10d::Store& store,
                                      absl::Span<const std::string> keys) {
  for (const std::string& key : keys) {
    try {
      store.deleteKey(key);
    } catch (const std::exception& e) {
      ABSL_LOG(WARNING) << "failed to delete P2P descriptor key " << key
                        << " from the store: " << e.what();
    }
  }
}

void P2pSendQueue::WakeInflightWaiters(absl::Span<const std::string> keys) {
  const std::vector<uint8_t> empty_sentinel;
  for (const std::string& key : keys) {
    try {
      store_->set(key, empty_sentinel);
    } catch (const std::exception& e) {
      ABSL_LOG(WARNING) << "failed to set wake sentinel for key " << key << ": "
                        << e.what();
    }
  }
}

void P2pSendQueue::ExecuteSendTask(PeerChannel* channel, PendingSend task) {
  std::vector<std::vector<uint8_t>> descriptors;
  std::exception_ptr fetch_error;
  std::string fetch_error_msg;
  try {
    descriptors = channel->store->multiGet(task.descriptor_keys);
  } catch (const std::exception& e) {
    fetch_error = std::current_exception();
    fetch_error_msg = e.what();
  }

  bool is_shutdown = false;
  {
    absl::MutexLock lock(channel->mu);
    channel->inflight_keys.clear();
    is_shutdown = channel->shutdown;
  }

  if (is_shutdown) {
    DeleteConsumedKeys(*channel->store, task.descriptor_keys);
    std::move(task.on_error)(std::make_exception_ptr(TtError(
        TT_ERROR(  // ERROR_COV_INFEASIBLE=shutdown race with active send
            error::kCancelled)
            << "process group is shut down",
        TT_SOURCE_LOCATION)));
    return;
  }

  if (fetch_error) {
    if (absl::StrContainsIgnoreCase(fetch_error_msg, "timeout")) {
      std::move(task.on_error)(std::make_exception_ptr(TtError(
          TT_ERROR(  // ERROR_COV_INFEASIBLE=background store wait timeout
              error::kDeadlineExceeded)
              << "p2p send timed out waiting for descriptors: "
              << fetch_error_msg,
          TT_SOURCE_LOCATION)));
    } else {
      std::move(task.on_error)(std::move(fetch_error));
    }
    return;
  }

  std::move(task.on_ready)(std::move(descriptors));
  DeleteConsumedKeys(*channel->store, task.descriptor_keys);
}

void P2pSendQueue::PeerWorkerLoop(PeerChannel* channel) {
  PendingSend task;
  while (PopNextOrWait(channel, task)) {
    ExecuteSendTask(channel, std::move(task));
  }
}

absl::StatusOr<c10::intrusive_ptr<c10::ivalue::Future>>
ProcessGroupTpu::CrossHostSendBuffers(
    int dst_rank, std::shared_ptr<std::vector<DeviceBufferRef>> held_buffers,
    std::vector<xla::PjRtBuffer*> buffers,
    absl::Span<const xla::CrossHostTransferKey> transfer_keys) {
  TT_RET_CHECK(transfer_keys.size() == buffers.size(), error::kInvalidArgument)
      << "expected transfer keys to be the same size as buffers, got "
      << transfer_keys.size() << " transfer keys and " << buffers.size()
      << " buffers";

  auto all_sent_future =
      c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());

  if (buffers.empty()) {
    all_sent_future->markCompleted(c10::IValue());
    return all_sent_future;
  }

  // Map transfer keys to their corresponding store keys to retrieve the
  // serialized descriptors generated by the remote host.
  std::vector<std::string> descriptor_keys;
  descriptor_keys.reserve(transfer_keys.size());
  for (const auto& transfer_key : transfer_keys) {
    descriptor_keys.push_back(
        GetCrossHostTransferDescriptorStoreKey(transfer_key));
  }

  auto ready_callback = [all_sent_future,
                         held_buffers = std::move(held_buffers),
                         buffers = std::move(buffers)](
                            std::vector<std::vector<uint8_t>> descriptors) {
    auto remaining_sends =
        std::make_shared<std::atomic<size_t>>(buffers.size());
    for (size_t i = 0; i < descriptors.size(); ++i) {
      const std::string descriptor(descriptors[i].begin(),
                                   descriptors[i].end());
      buffers[i]->CopyToRemoteDevice(
          std::move(descriptor),
          [all_sent_future, remaining_sends, held_buffers](
              absl::Status status, bool sends_were_enqueued) {
            if (all_sent_future->completed()) {
              return;
            }
            if (!status.ok()) {
              all_sent_future->setErrorIfNeeded(std::make_exception_ptr(
                  TtError(TT_ERROR(error::kInternal)
                              << "PjRtBuffer::CopyToRemoteDevice: " << status,
                          TT_SOURCE_LOCATION)));
            } else if (!sends_were_enqueued) {
              all_sent_future->setErrorIfNeeded(std::make_exception_ptr(
                  TtError(TT_ERROR(error::kInternal)
                              << "PjRtBuffer::CopyToRemoteDevice: "
                                 "sends were not enqueued",
                          TT_SOURCE_LOCATION)));
            } else if (remaining_sends->fetch_sub(
                           1, std::memory_order_acq_rel) == 1) {
              all_sent_future->markCompleted(c10::IValue());
            }
          });
    }
  };

  TT_RETURN_IF_ERROR(  // ERROR_COV_INFEASIBLE=shutdown race during teardown
      p2p_send_queue_.Enqueue(
          dst_rank, std::move(descriptor_keys), std::move(ready_callback),
          [all_sent_future](std::exception_ptr eptr) {
            all_sent_future->setErrorIfNeeded(std::move(eptr));
          }));

  return all_sent_future;
}

absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
ProcessGroupTpu::CrossHostReceiveBuffers(
    absl::Span<const xla::Shape> shapes,
    absl::Span<const xla::CrossHostTransferKey> transfer_keys) {
  auto setup_future =
      c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());

  auto notifier =
      [this, setup_future,
       transfer_keys = std::vector<xla::CrossHostTransferKey>(
           transfer_keys.begin(), transfer_keys.end())](
          absl::StatusOr<xla::PjRtCrossHostRecvState> recv_state) -> void {
    if (!recv_state.ok()) {
      setup_future->setError(std::make_exception_ptr(TtError(
          TT_ERROR(error::kInternal)
              << "xla::PjRtClient::MakeCrossHostReceiveBuffers: invalid "
                 "recv_state: "
              << recv_state.status(),
          TT_SOURCE_LOCATION)));
      return;
    }

    if (recv_state->descriptors.size() != transfer_keys.size()) {
      setup_future->setError(std::make_exception_ptr(TtError(
          TT_ERROR(error::kInternal)
              << "xla::PjRtClient::MakeCrossHostReceiveBuffers: expected "
                 "descriptors to be the same size as transfer keys, got "
              << recv_state->descriptors.size() << " descriptors and "
              << transfer_keys.size() << " keys",
          TT_SOURCE_LOCATION)));
      return;
    }

    // Collect the serialized descriptors and push them to the distributed
    // store. This act signals the sender that the receiver is ready.
    std::vector<std::string> descriptor_keys;
    std::vector<std::vector<uint8_t>> descriptors;
    descriptor_keys.reserve(transfer_keys.size());
    descriptors.reserve(transfer_keys.size());

    for (int i = 0; i < transfer_keys.size(); ++i) {
      descriptor_keys.push_back(
          GetCrossHostTransferDescriptorStoreKey(transfer_keys[i]));

      std::string_view descriptor =
          recv_state->descriptors[i].serialized_descriptors.front();
      descriptors.push_back(
          std::vector<uint8_t>(descriptor.begin(), descriptor.end()));
    }

    // Update the store to unblock the sender's multiGet call.
    try {
      this->store_->multiSet(descriptor_keys, descriptors);
      setup_future->markCompleted(c10::IValue());
    } catch (const std::exception& e) {
      setup_future->setError(std::current_exception());
    }
  };

  // Initiate the receive process on the PJRT client. The resulting buffers
  // will be populated once the remote sender initiates its transfer.
  TT_ASSIGN_OR_RETURN(xla::PjRtClient* const client,
                      PjrtBackend::GetInstance().GetClient());
  TT_ASSIGN_OR_RETURN(xla::PjRtDevice* const device,
                      PjrtBackend::GetInstance().GetDevice());
  TT_ASSIGN_OR_RETURN(auto buffers, client->MakeCrossHostReceiveBuffers(
                                        shapes, device, std::move(notifier)));

  // Wait for the buffer to be ready to transfer and descriptors to be
  // posted to the store. This ensures the enqueueing order of send and receive
  // can be maintained from the caller's perspective.
  setup_future->waitAndThrow();
  return buffers;
}

namespace {

struct WorldTopologyState {
  std::vector<PjRtDeviceAttributes> device_attrs;
  std::vector<int64_t> rank_to_device_id;  // INT_VEC_OK
  // rank_to_topo_order[r] is world rank r's position in the physical-topology
  // order (see `ComputeRankToTopologyOrder`).
  std::vector<int64_t> rank_to_topo_order;  // INT_VEC_OK
  int64_t world_rank = -1;
};

WorldTopologyState& WorldTopology() {
  static absl::NoDestructor<WorldTopologyState> state;
  return *state;
}

}  // namespace

const std::vector<PjRtDeviceAttributes>& GetGlobalDeviceAttributes() {
  return WorldTopology().device_attrs;
}

const std::vector<int64_t>& GetGlobalRankToDeviceId() {  // INT_VEC_OK
  return WorldTopology().rank_to_device_id;
}

absl::StatusOr<std::vector<int64_t>> GetWorldDeviceAssignment() {  // INT_VEC_OK
  TT_ASSIGN_OR_RETURN(const int num_devices,
                      PjrtBackend::GetInstance().GetGlobalDeviceCount());
  if (num_devices <= 1) {
    return std::vector<int64_t>();  // INT_VEC_OK
  }
  const std::vector<int64_t>& world_rank_to_device_id =  // INT_VEC_OK
      GetGlobalRankToDeviceId();
  if (!world_rank_to_device_id.empty()) {
    return world_rank_to_device_id;
  }
  std::vector<int64_t> device_assignment(num_devices);  // INT_VEC_OK
  std::iota(device_assignment.begin(), device_assignment.end(), 0);
  return device_assignment;
}

std::vector<int64_t> ComputeRankToTopologyOrder(  // INT_VEC_OK
    absl::Span<const PjRtDeviceAttributes> device_attrs) {
  // (slice_index, x, y, z, core_on_chip, id); defaults match
  // `SerializeDeviceAttributes`.
  using TopologyKey =
      std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
  const auto key_of = [](const PjRtDeviceAttributes& attrs) -> TopologyKey {
    const absl::Span<const int64_t> coords =
        attrs.coords.has_value() ? absl::MakeConstSpan(*attrs.coords)
                                 : absl::Span<const int64_t>();
    return {
        attrs.slice_index.value_or(-1),    !coords.empty() ? coords[0] : 0,
        coords.size() > 1 ? coords[1] : 0, coords.size() > 2 ? coords[2] : 0,
        attrs.core_on_chip.value_or(0),    attrs.id};
  };
  std::vector<TopologyKey> keys;
  keys.reserve(device_attrs.size());
  for (const PjRtDeviceAttributes& attrs : device_attrs) {
    keys.push_back(key_of(attrs));
  }
  std::vector<int64_t> ranks_in_topo_order(device_attrs.size());  // INT_VEC_OK
  std::iota(ranks_in_topo_order.begin(), ranks_in_topo_order.end(), 0);
  absl::c_stable_sort(ranks_in_topo_order, [&keys](int64_t a, int64_t b) {
    return keys[a] < keys[b];
  });
  std::vector<int64_t> rank_to_topo_order(device_attrs.size());  // INT_VEC_OK
  for (size_t pos = 0; pos < ranks_in_topo_order.size(); ++pos) {
    rank_to_topo_order[ranks_in_topo_order[pos]] = static_cast<int64_t>(pos);
  }
  return rank_to_topo_order;
}

void PermuteDeviceAssignmentForSubgroups(
    const DeviceGroupList& subgroup_member_world_ranks,
    absl::Span<const int64_t> rank_to_device_id,
    std::vector<int64_t>& device_assignment,  // INT_VEC_OK
    DeviceGroupList& replica_groups) {
  device_assignment.clear();
  device_assignment.reserve(rank_to_device_id.size());
  std::vector<DeviceGroup> iota_groups;
  iota_groups.reserve(subgroup_member_world_ranks.size());
  int64_t partition = 0;
  for (const DeviceGroup& group : subgroup_member_world_ranks) {
    DeviceGroup iota_group;
    iota_group.reserve(group.size());
    for (int64_t world_rank : group) {
      device_assignment.push_back(rank_to_device_id[world_rank]);
      iota_group.push_back(partition++);
    }
    iota_groups.push_back(std::move(iota_group));
  }
  replica_groups = DeviceGroupList(std::move(iota_groups), /*normalize=*/false);
}

namespace {

int64_t NowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

class PrefixStoreAccessor : public c10d::PrefixStore {
 public:
  static const std::string& GetPrefix(const c10d::PrefixStore* store) {
    return static_cast<const PrefixStoreAccessor*>(store)->prefix_;
  }
};

struct UnwrappedStoreChain {
  c10d::Store* root_store = nullptr;
  std::string full_prefix;
};

UnwrappedStoreChain UnwrapPrefixStoreChain(c10d::Store* store) {
  UnwrappedStoreChain result;
  c10d::Store* curr = store;
  std::vector<std::string_view> prefixes;
  while (auto* prefix = dynamic_cast<c10d::PrefixStore*>(curr)) {
    prefixes.push_back(PrefixStoreAccessor::GetPrefix(prefix));
    curr = prefix->getUnderlyingStore().get();
    if (curr == nullptr) {
      break;
    }
  }
  absl::c_reverse(prefixes);
  result.root_store = curr;
  result.full_prefix = absl::StrJoin(prefixes, "/");
  return result;
}

// Runs `f`, a call into c10d::Store, converting any exception it throws (e.g.
// c10::DistStoreError on timeout) into an error status. Returns absl::Status if
// `f` returns void, else absl::StatusOr of its result. This is the only place
// in this file that catches store exceptions.
template <typename F>
auto CallStore(std::string_view op, F&& f) {
  using R = std::invoke_result_t<F>;
  const auto to_status = [op](const std::exception& e) -> absl::Status {
    // c10::Error::what() appends a C++ backtrace, which doesn't belong in a
    // TorchTPU error message.
    const auto* c10_error = dynamic_cast<const c10::Error*>(&e);
    const std::string message = AdaptExternalErrorMessage(
        c10_error != nullptr ? c10_error->what_without_backtrace() : e.what());
    const bool timed_out = absl::StrContainsIgnoreCase(message, "timeout");
    return TT_ERROR(timed_out ? error::kDeadlineExceeded : error::kUnavailable)
           << op << ": " << message;
  };
  if constexpr (std::is_void_v<R>) {
    try {
      std::forward<F>(f)();
    } catch (const std::exception& e) {
      return to_status(e);
    }
    return absl::OkStatus();
  } else {
    try {
      return absl::StatusOr<R>(std::forward<F>(f)());
    } catch (const std::exception& e) {
      return absl::StatusOr<R>(to_status(e));
    }
  }
}

using ByteVector = std::vector<uint8_t>;

std::vector<uint8_t> PackKeyAndValue(std::string_view key,
                                     std::string_view value) {
  const std::string packed = absl::StrCat(key.size(), ":", key, value);
  return std::vector<uint8_t>(packed.begin(), packed.end());
}

struct KeyAndValue {
  std::string key;
  std::vector<uint8_t> value;
};

absl::StatusOr<KeyAndValue> UnpackKeyAndValue(
    absl::Span<const uint8_t> packed) {
  const std::string_view view(reinterpret_cast<const char*>(packed.data()),
                              packed.size());
  const size_t colon_pos = view.find(':');
  size_t key_len = 0;
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=defensive check on internally
                 // serialized store payload
      colon_pos != std::string_view::npos &&
          absl::SimpleAtoi(view.substr(0, colon_pos), &key_len) &&
          key_len <= view.size() - colon_pos - 1,
      error::kInternal)
      << "malformed node-local store key/value entry (" << packed.size()
      << " bytes)";
  const size_t key_start = colon_pos + 1;
  return KeyAndValue{
      .key = std::string(view.substr(key_start, key_len)),
      .value = std::vector<uint8_t>(packed.begin() + key_start + key_len,
                                    packed.end()),
  };
}

std::vector<uint8_t> PackValueList(absl::Span<const ByteVector> values) {
  std::string packed;
  for (const auto& val : values) {
    absl::StrAppend(&packed, val.size(), ":",
                    std::string_view(reinterpret_cast<const char*>(val.data()),
                                     val.size()));
  }
  return std::vector<uint8_t>(packed.begin(), packed.end());
}

absl::StatusOr<std::vector<ByteVector>> UnpackAllValues(
    absl::Span<const uint8_t> packed) {
  std::vector<ByteVector> values;
  std::string_view cursor(reinterpret_cast<const char*>(packed.data()),
                          packed.size());
  while (!cursor.empty()) {
    const size_t colon_pos = cursor.find(':');
    size_t val_len = 0;
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=defensive check on internally
                   // serialized store payload
        colon_pos != std::string_view::npos &&
            absl::SimpleAtoi(cursor.substr(0, colon_pos), &val_len) &&
            val_len <= cursor.size() - colon_pos - 1,
        error::kInternal)
        << "malformed node-local store value list (" << packed.size()
        << " bytes)";
    cursor.remove_prefix(colon_pos + 1);
    const auto* val_ptr = reinterpret_cast<const uint8_t*>(cursor.data());
    values.emplace_back(val_ptr, val_ptr + val_len);
    cursor.remove_prefix(val_len);
  }
  return values;
}

// Prefixes of the table published by the AllGatherViaNodeLocalStore collector.
constexpr std::string_view kNodeLocalTableOkPrefix = "OK\n";
constexpr std::string_view kNodeLocalTableErrPrefix = "ERR\n";

std::vector<uint8_t> EncodeOkTable(absl::Span<const ByteVector> values) {
  std::vector<uint8_t> bytes(kNodeLocalTableOkPrefix.begin(),
                             kNodeLocalTableOkPrefix.end());
  const std::vector<uint8_t> packed = PackValueList(values);
  bytes.insert(bytes.end(), packed.begin(), packed.end());
  return bytes;
}

std::vector<uint8_t> EncodeErrTable(std::string_view message) {
  const std::string bytes = absl::StrCat(kNodeLocalTableErrPrefix, message);
  return std::vector<uint8_t>(bytes.begin(), bytes.end());
}

// Decodes a table published by the collector. Returns an error if the
// collector reported one.
absl::StatusOr<std::vector<ByteVector>> DecodeTable(
    absl::Span<const uint8_t> bytes, size_t expected_count) {
  const std::string_view view(reinterpret_cast<const char*>(bytes.data()),
                              bytes.size());
  if (absl::StartsWith(view, kNodeLocalTableErrPrefix)) {
    return TT_ERROR(error::kAborted)
           << "failed to all-gather via the node-local store: "
           << view.substr(kNodeLocalTableErrPrefix.size());
  }
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=the collector only publishes OK or
                 // ERR tables
      absl::StartsWith(view, kNodeLocalTableOkPrefix), error::kInternal)
      << "malformed node-local store table (" << bytes.size() << " bytes)";
  TT_ASSIGN_OR_RETURN(
      std::vector<ByteVector> values,
      UnpackAllValues(bytes.subspan(kNodeLocalTableOkPrefix.size())));
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=the collector fills every peer key
                 // before publishing an OK table
      values.size() == expected_count, error::kInternal)
      << "node-local store table has " << values.size() << " entries for "
      << expected_count << " peer keys";
  return values;
}

// Returns up to `max_keys` of the peer keys that are not yet filled.
std::string DescribeMissingKeys(const std::vector<std::string>& peer_keys,
                                const std::vector<bool>& filled,
                                size_t max_keys) {
  std::vector<std::string_view> missing;
  size_t num_missing = 0;
  for (size_t i = 0; i < peer_keys.size(); ++i) {
    if (filled[i]) {
      continue;
    }
    ++num_missing;
    if (missing.size() < max_keys) {
      missing.push_back(peer_keys[i]);
    }
  }
  std::string result = absl::StrCat("[", absl::StrJoin(missing, ", "));
  if (num_missing > missing.size()) {
    absl::StrAppend(&result, ", ... (", num_missing - missing.size(), " more)");
  }
  absl::StrAppend(&result, "]");
  return result;
}

}  // namespace

// Protocol. All keys are scoped by `scope_key` = "<peer_keys[0]>:<num_peers>".
//   1. A worker (local_rank > 0) sets its packed (key, value) under
//      "local_up:<scope>:<local_rank>" in the node-local store, then waits for
//      the table under "local_down:<scope>".
//   2. A leader multiGets its workers' entries, claims a node slot with
//      world_store->add("nlag:<scope>:count", 1) and publishes all of its
//      node's entries as one blob under "nlag:<scope>:node:<slot>".
//   3. The leader whose node holds peer_keys[0] (the collector) reads node
//      blobs as their slots fill until every peer key is filled, then
//      publishes "OK\n<values>" (or "ERR\n<message>") under
//      "nlag:<scope>:table".
//   4. Every other leader gets that table, and every leader forwards it to its
//      workers under "local_down:<scope>".
// A leader that fails forwards an ERR table to its workers (and, if it is the
// collector, to the world table) so that they fail promptly instead of timing
// out.
absl::StatusOr<std::vector<std::vector<uint8_t>>> AllGatherViaNodeLocalStore(
    c10d::Store* world_store, c10d::Store* local_store, int64_t local_rank,
    int64_t local_world_size, std::string_view my_key,
    std::string_view my_value, const std::vector<std::string>& peer_keys) {
  if (peer_keys.empty()) {
    return std::vector<ByteVector>();
  }
  const int64_t start_ms = NowMs();
  const int64_t num_peers = static_cast<int64_t>(peer_keys.size());

  // Without a usable node-local store, this rank is the leader of a node that
  // contains only itself.
  const bool use_local = local_store != nullptr && local_world_size > 1;
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=GetNodeLocalStore validates LOCAL_RANK
                 // before any caller gets here
      !use_local || (local_rank >= 0 && local_rank < local_world_size),
      error::kInvalidArgument)
      << absl::StrFormat(
             "local_rank=%d is out of range [0, %d) for the node-local store "
             "(my_key=%s)",
             local_rank, local_world_size, my_key);
  if (use_local && world_store != nullptr &&
      world_store->getTimeout() > local_store->getTimeout()) {
    local_store->setTimeout(world_store->getTimeout());
  }

  const std::string scope_key = absl::StrCat(peer_keys[0], ":", num_peers);
  const std::string down_key = absl::StrCat("local_down:", scope_key);

  if (use_local && local_rank != 0) {
    const std::string up_key =
        absl::StrCat("local_up:", scope_key, ":", local_rank);
    TT_RETURN_IF_ERROR(
        CallStore(absl::StrCat("local_store set ", up_key),
                  [&] {  // REF_CAPTURE_OK=called synchronously
                    local_store->set(up_key, PackKeyAndValue(my_key, my_value));
                  }));
    TT_ASSIGN_OR_RETURN(const ByteVector table,
                        CallStore(absl::StrCat("local_store get ", down_key),
                                  [&] {  // REF_CAPTURE_OK=called synchronously
                                    return local_store->get(down_key);
                                  }));
    return DecodeTable(table, peer_keys.size());
  }

  const std::string count_key = absl::StrCat("nlag:", scope_key, ":count");
  const std::string table_key = absl::StrCat("nlag:", scope_key, ":table");
  const auto node_key = [&scope_key](int64_t slot) {
    return absl::StrCat("nlag:", scope_key, ":node:", slot);
  };
  bool is_collector = my_key == peer_keys[0];
  bool world_table_written = false;
  bool local_down_written = false;
  int64_t num_nodes = 0;

  // Reads node blobs until every peer key is filled. Collector only.
  const auto collect = [&]  // REF_CAPTURE_OK=called locally
      () -> absl::StatusOr<std::vector<ByteVector>> {
    absl::flat_hash_map<std::string_view, size_t> key_to_index;
    key_to_index.reserve(peer_keys.size());
    for (size_t i = 0; i < peer_keys.size(); ++i) {
      key_to_index.emplace(peer_keys[i], i);
    }
    std::vector<ByteVector> values(peer_keys.size());
    std::vector<bool> filled(peer_keys.size(), false);
    int64_t num_filled = 0;
    const auto run = [&]() -> absl::Status {  // REF_CAPTURE_OK=called locally
      while (num_filled < num_peers) {
        const std::string next_key = node_key(num_nodes);
        TT_RETURN_IF_ERROR(
            CallStore(absl::StrCat("world_store wait ", next_key),
                      [&] {  // REF_CAPTURE_OK=called synchronously
                        world_store->wait({next_key});
                      }));
        TT_ASSIGN_OR_RETURN(
            const int64_t avail,
            CallStore(absl::StrCat("world_store add ", count_key),
                      [&] {  // REF_CAPTURE_OK=called synchronously
                        return world_store->add(count_key, 0);
                      }));
        TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=internal invariant: slots are
                       // claimed only by this gather's peers
            avail <= num_peers, error::kFailedPrecondition)
            << avail << " nodes published for " << num_peers
            << " peer keys (scope reused, or too many nodes)";
        std::vector<std::string> batch_keys;
        batch_keys.reserve(avail - num_nodes);
        for (int64_t slot = num_nodes; slot < avail; ++slot) {
          batch_keys.push_back(node_key(slot));
        }
        TT_ASSIGN_OR_RETURN(
            const std::vector<ByteVector> blobs,
            CallStore("world_store multiGet node blobs",
                      [&] {  // REF_CAPTURE_OK=called synchronously
                        return world_store->multiGet(batch_keys);
                      }));
        for (const ByteVector& blob : blobs) {
          TT_ASSIGN_OR_RETURN(const std::vector<ByteVector> entries,
                              UnpackAllValues(blob));
          for (const ByteVector& packed_entry : entries) {
            TT_ASSIGN_OR_RETURN(KeyAndValue entry,
                                UnpackKeyAndValue(packed_entry));
            const auto it = key_to_index.find(entry.key);
            TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by
                           // process_group_tpu_test
                it != key_to_index.end(), error::kFailedPrecondition)
                << "unknown key " << entry.key;
            TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Covered by
                           // process_group_tpu_test
                !filled[it->second], error::kFailedPrecondition)
                << "duplicate key " << entry.key;
            values[it->second] = std::move(entry.value);
            filled[it->second] = true;
            ++num_filled;
          }
        }
        num_nodes = avail;
        ABSL_VLOG(1) << "AllGatherViaNodeLocalStore(" << scope_key
                     << "): collector has " << num_filled << "/" << num_peers
                     << " entries from " << num_nodes << " nodes after "
                     << NowMs() - start_ms << " ms";
      }
      return absl::OkStatus();
    };
    if (absl::Status status = run(); !status.ok()) {
      return TT_ERROR(status.code()) << absl::StrFormat(
                 "collector (node holding %s): failed after %dms with %d/%d "
                 "peer keys filled from %d arrived nodes; missing keys %s: %s",
                 peer_keys[0], NowMs() - start_ms, num_filled, num_peers,
                 num_nodes,
                 DescribeMissingKeys(peer_keys, filled,
                                     /*max_keys=*/16),
                 status.message());
    }
    return values;
  };

  // The leader's whole sequence. Any error is forwarded below.
  const auto lead = [&]  // REF_CAPTURE_OK=called locally
      () -> absl::StatusOr<std::vector<ByteVector>> {
    std::vector<ByteVector> node_entries = {PackKeyAndValue(my_key, my_value)};
    if (use_local) {
      std::vector<std::string> up_keys;
      up_keys.reserve(local_world_size - 1);
      for (int64_t lr = 1; lr < local_world_size; ++lr) {
        up_keys.push_back(absl::StrCat("local_up:", scope_key, ":", lr));
      }
      absl::StatusOr<std::vector<ByteVector>> up_entries =
          CallStore("local_store multiGet",
                    [&] {  // REF_CAPTURE_OK=called synchronously
                      return local_store->multiGet(up_keys);
                    });
      if (!up_entries.ok()) {
        return TT_ERROR(  // ERROR_COV_INFEASIBLE=needs a local rank that never
                          // joins; blocks for the 300s FileStore timeout
                   up_entries.status().code())
               << absl::StrFormat(
                      "leader failed collecting %d local ranks for scope %s "
                      "(node-local store /dev/shm/torch_tpu_pg_store_%d): %s",
                      local_world_size - 1, scope_key, getppid(),
                      up_entries.status().message());
      }
      for (ByteVector& packed_entry : *up_entries) {
        TT_ASSIGN_OR_RETURN(const KeyAndValue entry,
                            UnpackKeyAndValue(packed_entry));
        is_collector = is_collector || entry.key == peer_keys[0];
        node_entries.push_back(std::move(packed_entry));
      }
    }

    // Publish this node's entries as one blob in a slot claimed with add().
    TT_ASSIGN_OR_RETURN(const int64_t count,
                        CallStore(absl::StrCat("world_store add ", count_key),
                                  [&] {  // REF_CAPTURE_OK=called synchronously
                                    return world_store->add(count_key, 1);
                                  }));
    const int64_t slot = count - 1;
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=internal invariant: slots are
                   // claimed only by this gather's peers
        slot < num_peers, error::kFailedPrecondition)
        << "claimed node slot " << slot << " but there are only " << num_peers
        << " peer keys (scope " << scope_key << " reused, or too many nodes)";
    const std::string blob_key = node_key(slot);
    TT_RETURN_IF_ERROR(CallStore(absl::StrCat("world_store set ", blob_key),
                                 [&] {  // REF_CAPTURE_OK=called synchronously
                                   world_store->set(
                                       blob_key, PackValueList(node_entries));
                                 }));
    ABSL_VLOG(1) << "AllGatherViaNodeLocalStore(" << scope_key
                 << "): leader published " << node_entries.size()
                 << " entries in node slot " << slot;

    ByteVector table;
    if (is_collector) {
      TT_ASSIGN_OR_RETURN(const std::vector<ByteVector> values, collect());
      table = EncodeOkTable(values);
      TT_RETURN_IF_ERROR(CallStore(absl::StrCat("world_store set ", table_key),
                                   [&] {  // REF_CAPTURE_OK=called synchronously
                                     world_store->set(table_key, table);
                                   }));
      world_table_written = true;
    } else {
      TT_ASSIGN_OR_RETURN(
          table, CallStore(absl::StrCat("world_store get ", table_key),
                           [&] {  // REF_CAPTURE_OK=called synchronously
                             return world_store->get(table_key);
                           }));
    }
    // Forward the table before decoding it so workers also see errors.
    if (use_local) {
      TT_RETURN_IF_ERROR(CallStore(absl::StrCat("local_store set ", down_key),
                                   [&] {  // REF_CAPTURE_OK=called synchronously
                                     local_store->set(down_key, table);
                                   }));
      local_down_written = true;
    }
    return DecodeTable(table, peer_keys.size());
  };

  absl::StatusOr<std::vector<ByteVector>> result = lead();
  if (!result.ok()) {
    // Best effort: failures to forward the error are ignored.
    const ByteVector err_table = EncodeErrTable(absl::StrCat(
        "node leader (my_key=", my_key, "): ", result.status().message()));
    if (is_collector && !world_table_written) {
      CallStore("world_store set error table",
                [&] {  // REF_CAPTURE_OK=called synchronously
                  world_store->set(table_key, err_table);
                })
          .IgnoreError();  // IGNORE_ERROR_OK=best effort; original error
                           // returned
    }
    if (use_local && !local_down_written) {
      CallStore("local_store set error table",
                [&] {  // REF_CAPTURE_OK=called synchronously
                  local_store->set(down_key, err_table);
                })
          .IgnoreError();  // IGNORE_ERROR_OK=best effort; original error
                           // returned
    }
    return result.status();
  }
  if (is_collector) {
    ABSL_LOG(INFO) << "AllGatherViaNodeLocalStore(" << scope_key
                   << "): gathered " << num_peers << " entries from "
                   << num_nodes << " nodes in " << NowMs() - start_ms << " ms";
  }
  return result;
}

namespace {

struct NodeLocalStoreConfig {
  c10d::Store* store = nullptr;
  int64_t local_rank = 0;
  int64_t local_world_size = 1;
};

// Returns the node-local store used by AllGatherViaNodeLocalStore. Returns a
// default config (no local store, so every rank publishes on its own) when the
// node layout (LOCAL_RANK and TORCH_TPU_DEVICES_PER_HOST) is unknown or the
// store is unavailable.
absl::StatusOr<NodeLocalStoreConfig> GetNodeLocalStore(
    const int64_t world_size,
    const std::optional<std::chrono::milliseconds> min_timeout = std::nullopt) {
  const std::optional<int64_t> local_rank =
      GetIntegerEnvOnce<kLocalRankEnvVar>();
  const std::optional<int64_t> local_world_size =
      GetIntegerEnvOnce<kTorchTpuDevicesPerHostEnvVar>();
  const std::string env_summary = absl::StrFormat(
      "local_rank=%s devices_per_host=%s world_size=%d",
      local_rank.has_value() ? absl::StrCat(*local_rank) : "unset",
      local_world_size.has_value() ? absl::StrCat(*local_world_size) : "unset",
      world_size);
  if (!local_rank.has_value() || !local_world_size.has_value()) {
    ABSL_LOG_FIRST_N(WARNING, 1)
        << "Node-local store disabled: LOCAL_RANK or "
           "TORCH_TPU_DEVICES_PER_HOST is unset ("
        << env_summary
        << "). Every rank will publish to the world store individually, "
           "which is slower at scale.";
    return NodeLocalStoreConfig{};
  }
  if (*local_world_size <= 1) {
    return NodeLocalStoreConfig{};
  }
  TT_RET_CHECK(*local_rank >= 0 && *local_rank < *local_world_size,
               error::kInvalidArgument)
      << absl::StrFormat(
             "expected environment variable LOCAL_RANK to be in [0, %d) "
             "(TORCH_TPU_DEVICES_PER_HOST), got %d (%s)",
             *local_world_size, *local_rank, env_summary);
  if (::access("/dev/shm", W_OK) != 0) {
    ABSL_LOG_FIRST_N(WARNING, 1)
        << "Node-local store disabled: /dev/shm is not writable ("
        << env_summary
        << "). Every rank will publish to the world store individually, "
           "which is slower at scale.";
    return NodeLocalStoreConfig{};
  }
  const std::string shm_path = absl::StrFormat("/dev/shm/torch_tpu_pg_store_%d",
                                               static_cast<int>(getppid()));
  static absl::NoDestructor<c10::intrusive_ptr<c10d::FileStore>> shm_store(
      c10::make_intrusive<c10d::FileStore>(
          shm_path, static_cast<int>(*local_world_size)));
  // Raise the cached FileStore timeout whenever a caller passes a larger
  // timeout so later PGs or subgroups do not fail early on `/dev/shm` waits.
  if (min_timeout.has_value()) {
    static absl::NoDestructor<absl::Mutex> shm_timeout_mu;
    absl::MutexLock lock(*shm_timeout_mu);
    if (*min_timeout > (*shm_store)->getTimeout()) {
      (*shm_store)->setTimeout(*min_timeout);
    }
  }
  return NodeLocalStoreConfig{
      .store = shm_store->get(),
      .local_rank = *local_rank,
      .local_world_size = *local_world_size,
  };
}

std::atomic<uint64_t> next_alltoall_shm_nonce{0};

// Coordinates a per-PG nonce once over `store` so `/dev/shm` metadata keys
// never collide across `destroy_process_group()`/`init_process_group()` cycles
// or repeated launches sharing the same parent PID.
absl::StatusOr<std::string> GetOrInitAllToAllShmNonce(
    c10d::Store* const store, const int my_rank,
    AllToAllShmNonceCache& shm_nonce_cache) {
  {
    absl::MutexLock lock(shm_nonce_cache.mu);
    if (!shm_nonce_cache.value.empty()) {
      return shm_nonce_cache.value;
    }
  }
  constexpr std::string_view kNonceKey = "a2a_meta_shm_nonce";
  std::string nonce;
  if (my_rank == 0) {
    nonce = absl::StrCat(
        static_cast<int>(getpid()), "_", absl::GetCurrentTimeNanos(), "_",
        next_alltoall_shm_nonce.fetch_add(1, std::memory_order_relaxed));
    const std::vector<uint8_t> nonce_bytes(nonce.begin(), nonce.end());
    TT_RETURN_IF_ERROR(CallStore("group_store set a2a_meta_shm_nonce",
                                 [&] {  // REF_CAPTURE_OK=called synchronously
                                   store->set(std::string(kNonceKey),
                                              nonce_bytes);
                                 }));
  } else {
    TT_ASSIGN_OR_RETURN(const std::vector<uint8_t> nonce_bytes,
                        CallStore("group_store get a2a_meta_shm_nonce",
                                  [&] {  // REF_CAPTURE_OK=called synchronously
                                    return store->get(std::string(kNonceKey));
                                  }));
    nonce.assign(nonce_bytes.begin(), nonce_bytes.end());
  }
  absl::MutexLock lock(shm_nonce_cache.mu);
  if (shm_nonce_cache.value.empty()) {
    shm_nonce_cache.value = std::move(nonce);
  }
  return shm_nonce_cache.value;
}

std::vector<std::string> BuildPeerKeys(const std::string_view key_prefix,
                                       const size_t group_size) {
  std::vector<std::string> peer_keys;
  peer_keys.reserve(group_size);
  for (size_t peer = 0; peer < group_size; ++peer) {
    peer_keys.push_back(absl::StrCat(key_prefix, peer));
  }
  return peer_keys;
}

absl::StatusOr<std::vector<std::vector<uint8_t>>> ExchangeViaStore(
    c10d::Store* const store, const std::string_view store_label,
    const std::string_view key_prefix, const int my_rank,
    const size_t group_size, const std::vector<uint8_t>& my_bytes) {
  const std::string my_key = absl::StrCat(key_prefix, my_rank);
  const std::vector<std::string> peer_keys =
      BuildPeerKeys(key_prefix, group_size);
  // Publish this rank's payload before `multiGet` so peer fetches cannot
  // deadlock waiting on an unwritten key.
  TT_RETURN_IF_ERROR(CallStore(absl::StrCat(store_label, " set ", my_key),
                               [&] {  // REF_CAPTURE_OK=called synchronously
                                 store->set(my_key, my_bytes);
                               }));
  return CallStore(absl::StrCat(store_label, " multiGet a2a_meta"),
                   [&] {  // REF_CAPTURE_OK=called synchronously
                     return store->multiGet(peer_keys);
                   });
}

// Selects the lowest-latency safe store path for uneven all-to-all host
// validation: direct `/dev/shm` `FileStore` on single-host topologies,
// `AllGatherViaNodeLocalStore` on multi-host world-sized groups, and the group
// store otherwise.
absl::StatusOr<std::vector<std::vector<uint8_t>>>
ExchangeUnevenAllToAllPayloads(c10d::Store* const store, const int my_rank,
                               const size_t group_size, const uint64_t seq,
                               AllToAllShmNonceCache& shm_nonce_cache,
                               const std::vector<uint8_t>& my_bytes) {
  const int64_t global_world_size =
      static_cast<int64_t>(GetGlobalRankToDeviceId().size());
  if (global_world_size > 0) {
    TT_ASSIGN_OR_RETURN(
        const NodeLocalStoreConfig node_local,
        GetNodeLocalStore(global_world_size,
                          store != nullptr ? std::optional(store->getTimeout())
                                           : std::nullopt));
    if (node_local.store != nullptr && node_local.local_world_size > 1) {
      const UnwrappedStoreChain unwrapped = UnwrapPrefixStoreChain(store);
      TT_ASSIGN_OR_RETURN(
          const std::string shm_nonce,
          GetOrInitAllToAllShmNonce(store, my_rank, shm_nonce_cache));
      const std::string key_prefix = absl::StrCat(
          unwrapped.full_prefix, "/a2a_meta:", shm_nonce, ":", seq, ":");
      // All world ranks share one host, so both world-sized groups and
      // subgroups can exchange directly through `/dev/shm` without TCP traffic.
      if (global_world_size == node_local.local_world_size) {
        return ExchangeViaStore(node_local.store, "local_store", key_prefix,
                                my_rank, group_size, my_bytes);
      }
      // On multi-host world-sized groups, aggregate per-host payloads through
      // one node leader so cross-host store traffic scales as O(nodes).
      if (unwrapped.root_store != nullptr &&
          static_cast<int64_t>(group_size) == global_world_size) {
        const std::string my_key = absl::StrCat(key_prefix, my_rank);
        const std::vector<std::string> peer_keys =
            BuildPeerKeys(key_prefix, group_size);
        const std::string_view my_val(
            reinterpret_cast<const char*>(my_bytes.data()), my_bytes.size());
        return AllGatherViaNodeLocalStore(
            unwrapped.root_store, node_local.store, node_local.local_rank,
            node_local.local_world_size, my_key, my_val, peer_keys);
      }
    }
  }
  // Multi-host subgroups and setups without `/dev/shm` cannot assume peers
  // share a host or span every node, so fall back to the group store.
  const std::string key_prefix = absl::StrCat("a2a_meta:", seq, ":");
  return ExchangeViaStore(store, "group_store", key_prefix, my_rank, group_size,
                          my_bytes);
}

std::atomic<int64_t> next_subgroup_gather_id{0};

}  // namespace

absl::Status GatherSubgroupsViaNodeLocalStore(
    c10d::Store* store, c10d::Store* local_store, int64_t local_rank,
    int64_t local_world_size, int64_t pg_rank, int64_t pg_size,
    int64_t world_rank, absl::Span<const int64_t> global_rank_to_device_id,
    absl::Span<const int64_t> global_rank_to_topo_order, int64_t gather_id,
    DeviceGroup& out_member_world_ranks,
    DeviceGroupList& out_subgroup_member_world_ranks) {
  const UnwrappedStoreChain unwrapped = UnwrapPrefixStoreChain(store);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=internal store chain invariant
      unwrapped.root_store != nullptr, error::kInternal)
      << "root store is null when unwrapping the store chain";
  const int64_t world_size =
      static_cast<int64_t>(global_rank_to_device_id.size());
  const std::string my_key =
      absl::StrFormat("pg_sub:%d:%d", gather_id, world_rank);
  const std::string my_val =
      absl::StrFormat("%d,%d:%s", pg_rank, world_rank, unwrapped.full_prefix);
  std::vector<std::string> peer_keys(world_size);
  for (int64_t r = 0; r < world_size; ++r) {
    peer_keys[r] = absl::StrFormat("pg_sub:%d:%d", gather_id, r);
  }

  TT_ASSIGN_OR_RETURN(
      const std::vector<std::vector<uint8_t>> peer_values,
      AllGatherViaNodeLocalStore(unwrapped.root_store, local_store, local_rank,
                                 local_world_size, my_key, my_val, peer_keys));

  absl::flat_hash_map<std::string, DeviceGroup> groups_by_prefix;
  absl::flat_hash_set<int64_t> seen_world_ranks;
  for (const auto& bytes : peer_values) {
    const std::string_view payload(reinterpret_cast<const char*>(bytes.data()),
                                   bytes.size());
    const size_t comma_pos = payload.find(',');
    const size_t colon_pos = (comma_pos == std::string_view::npos)
                                 ? std::string_view::npos
                                 : payload.find(':', comma_pos + 1);
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=defensive check on internally
                   // serialized store payload
        comma_pos != std::string_view::npos &&
            colon_pos != std::string_view::npos,
        error::kInternal)
        << "malformed subgroup gather payload '" << payload << "'";
    int64_t entry_pg_rank = -1;
    int64_t entry_world_rank = -1;
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=defensive check on internally
                   // serialized store payload
        absl::SimpleAtoi(payload.substr(0, comma_pos), &entry_pg_rank) &&
            absl::SimpleAtoi(
                payload.substr(comma_pos + 1, colon_pos - comma_pos - 1),
                &entry_world_rank),
        error::kInternal)
        << "unparsable ranks in subgroup gather payload '" << payload << "'";
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=internal rank range invariant
        entry_pg_rank >= 0 && entry_pg_rank < pg_size &&
            entry_world_rank >= 0 && entry_world_rank < world_size,
        error::kInternal)
        << "rank out of range in subgroup gather payload '" << payload << "'";
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=internal partition invariant
        seen_world_ranks.insert(entry_world_rank).second,
        error::kInvalidArgument)
        << "duplicate world rank " << entry_world_rank << " in subgroup gather";

    const std::string_view entry_prefix = payload.substr(colon_pos + 1);
    DeviceGroup& group = groups_by_prefix[entry_prefix];
    if (group.empty()) {
      group.assign(pg_size, -1);
    }
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=internal partition invariant
        group[entry_pg_rank] == -1, error::kInvalidArgument)
        << "duplicate pg_rank " << entry_pg_rank << " in subgroup '"
        << entry_prefix << "'";
    group[entry_pg_rank] = entry_world_rank;
  }

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=internal partition invariant
      static_cast<int64_t>(groups_by_prefix.size()) * pg_size == world_size &&
          groups_by_prefix.contains(unwrapped.full_prefix),
      error::kInvalidArgument)
      << "incomplete or mismatched subgroups in SPMD gather";

  std::vector<DeviceGroup> unique_subgroups;
  unique_subgroups.reserve(groups_by_prefix.size());
  for (auto& [prefix, group] : groups_by_prefix) {
    if (prefix == unwrapped.full_prefix) {
      out_member_world_ranks = group;
    }
    unique_subgroups.push_back(std::move(group));
  }

  // Order subgroups by their leader's physical topology position only. Never
  // order by world rank: world ranks come from launcher order and would make
  // the per-slice device_assignment order (and thus Megascale's per-slice
  // collective pattern) depend on pod launch order. Members within a subgroup
  // keep PG-rank order.
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=world topology initialized before PGs
      static_cast<int64_t>(global_rank_to_topo_order.size()) == world_size,
      error::kInternal)
      << "world topology order not initialized";
  absl::c_sort(unique_subgroups, [global_rank_to_topo_order](
                                     const DeviceGroup& a,
                                     const DeviceGroup& b) {
    return global_rank_to_topo_order[a[0]] < global_rank_to_topo_order[b[0]];
  });
  out_subgroup_member_world_ranks =
      DeviceGroupList(std::move(unique_subgroups), /*normalize=*/false);
  return absl::OkStatus();
}

absl::Status ProcessGroupTpu::InitializeRankToDeviceMappings(
    int64_t world_size) {
  TT_RET_CHECK(size_ == world_size, error::kInvalidArgument) << absl::StrFormat(
      "expected the first initialized TPU process group to have size %d "
      "(world size), got %d",
      world_size, size_);

  TT_ASSIGN_OR_RETURN(PjRtDeviceAttributes local_attrs,
                      PjrtBackend::GetInstance().GetDeviceAttributes());

  const std::string my_key = absl::StrFormat("world_topo:%d", rank_);
  std::vector<std::string> peer_keys(world_size);
  for (int64_t r = 0; r < world_size; ++r) {
    peer_keys[r] = absl::StrFormat("world_topo:%d", r);
  }
  TT_ASSIGN_OR_RETURN(
      const NodeLocalStoreConfig local,
      GetNodeLocalStore(world_size, store_ != nullptr
                                        ? std::optional(store_->getTimeout())
                                        : std::nullopt));
  TT_ASSIGN_OR_RETURN(
      const std::vector<std::vector<uint8_t>> peer_values,
      AllGatherViaNodeLocalStore(
          store_.get(), local.store, local.local_rank, local.local_world_size,
          my_key, SerializeDeviceAttributes(local_attrs), peer_keys));

  std::vector<PjRtDeviceAttributes> device_attrs(world_size);
  std::vector<int64_t> rank_to_device_id(world_size);  // INT_VEC_OK
  for (int64_t r = 0; r < world_size; ++r) {
    const std::string_view payload(
        reinterpret_cast<const char*>(peer_values[r].data()),
        peer_values[r].size());
    TT_ASSIGN_OR_RETURN(device_attrs[r], DeserializeDeviceAttributes(payload));
    rank_to_device_id[r] = device_attrs[r].id;
  }
  std::vector<int64_t> rank_to_topo_order =  // INT_VEC_OK
      ComputeRankToTopologyOrder(device_attrs);

  // World ranks are defined by the rank order of the first world-sized
  // ProcessGroupTpu (the default WORLD group in practice). The global state is
  // only updated once all peers' attributes have been parsed successfully.
  auto& topo = WorldTopology();
  topo.world_rank = rank_;
  topo.device_attrs = std::move(device_attrs);
  topo.rank_to_device_id = std::move(rank_to_device_id);
  topo.rank_to_topo_order = std::move(rank_to_topo_order);
  return absl::OkStatus();
}

absl::StatusOr<DeviceGroup> ProcessGroupTpu::GatherMemberWorldRanks() {
  const auto& topo = WorldTopology();
  const int64_t world_size = topo.rank_to_device_id.size();
  if (size_ == 1) {
    return DeviceGroup{topo.world_rank};
  }

  DeviceGroup member_world_ranks;
  if (rank_ == 0) {
    member_world_ranks.resize(size_);
    member_world_ranks[0] = topo.world_rank;
    std::vector<std::string> keys(size_ - 1);
    for (int64_t i = 1; i < size_; ++i) {
      keys[i - 1] = absl::StrFormat("local_rank:%d", i);
    }
    TT_ASSIGN_OR_RETURN(const std::vector<std::vector<uint8_t>> values,
                        CallStore("store multiGet local_rank:*",
                                  [&] {  // REF_CAPTURE_OK=called synchronously
                                    return store_->multiGet(keys);
                                  }));
    for (int64_t i = 1; i < size_; ++i) {
      const std::string_view value(
          reinterpret_cast<const char*>(values[i - 1].data()),
          values[i - 1].size());
      TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=defensive check on internally
                     // serialized store payload
          absl::SimpleAtoi(value, &member_world_ranks[i]), error::kInternal)
          << "unparsable world rank '" << value << "' in store";
      TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=internal rank range invariant
          member_world_ranks[i] >= 0 && member_world_ranks[i] < world_size,
          error::kInternal)
          << "world rank " << member_world_ranks[i]
          << " out of range for world size " << world_size;
    }
    TT_RETURN_IF_ERROR(CallStore("store set local_subgroup_ranks",
                                 [&] {  // REF_CAPTURE_OK=called synchronously
                                   store_->set(
                                       "local_subgroup_ranks",
                                       absl::StrJoin(member_world_ranks, ","));
                                 }));
    return member_world_ranks;
  }

  const std::string my_key = absl::StrFormat("local_rank:%d", rank_);
  TT_RETURN_IF_ERROR(CallStore(absl::StrCat("store set ", my_key),
                               [&] {  // REF_CAPTURE_OK=called synchronously
                                 store_->set(my_key,
                                             std::to_string(topo.world_rank));
                               }));
  TT_ASSIGN_OR_RETURN(const std::vector<uint8_t> bytes,
                      CallStore("store get local_subgroup_ranks",
                                [&] {  // REF_CAPTURE_OK=called synchronously
                                  return store_->get("local_subgroup_ranks");
                                }));
  const std::string_view csv(reinterpret_cast<const char*>(bytes.data()),
                             bytes.size());
  member_world_ranks.reserve(size_);
  for (std::string_view part : absl::StrSplit(csv, ',')) {
    int64_t r = 0;
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=defensive check on internally
                   // serialized store payload
        absl::SimpleAtoi(part, &r), error::kInternal)
        << "unparsable world rank '" << part << "' in store";
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=internal rank range invariant
        r >= 0 && r < world_size, error::kInternal)
        << "world rank " << r << " out of range for world size " << world_size;
    member_world_ranks.push_back(r);
  }
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=internal subgroup size invariant
      static_cast<int64_t>(member_world_ranks.size()) == size_,
      error::kInternal)
      << "expected " << size_ << " subgroup ranks, got "
      << member_world_ranks.size();
  return member_world_ranks;
}

absl::StatusOr<DeviceGroupList> ProcessGroupTpu::GatherAllSubgroups(
    const DeviceGroup& member_world_ranks, int64_t world_size) {
  c10::intrusive_ptr<c10d::Store> root_store = store_;
  while (auto* prefix_store =
             dynamic_cast<c10d::PrefixStore*>(root_store.get())) {
    root_store = prefix_store->getUnderlyingNonPrefixStore();
  }

  const auto& topo = WorldTopology();
  const int64_t num_subgroups = world_size / size_;
  const int64_t gather_id = next_subgroup_gather_id.fetch_add(1);
  const std::string subgroup_store_key =
      absl::StrFormat("all_subgroups:%d", gather_id);

  std::vector<std::string> subgroup_csvs;
  subgroup_csvs.reserve(num_subgroups);
  if (rank_ == 0) {
    const std::string count_key =
        absl::StrFormat("subgroup_count:%d", gather_id);
    TT_ASSIGN_OR_RETURN(const int64_t count,
                        CallStore(absl::StrCat("root_store add ", count_key),
                                  [&] {  // REF_CAPTURE_OK=called synchronously
                                    return root_store->add(count_key, 1);
                                  }));
    const std::string my_subgroup_key =
        absl::StrFormat("subgroup:%d:%d", gather_id, count - 1);
    TT_RETURN_IF_ERROR(
        CallStore(absl::StrCat("root_store set ", my_subgroup_key),
                  [&] {  // REF_CAPTURE_OK=called synchronously
                    root_store->set(my_subgroup_key,
                                    absl::StrJoin(member_world_ranks, ","));
                  }));
    std::vector<std::string> peer_keys(num_subgroups);
    for (int64_t s = 0; s < num_subgroups; ++s) {
      peer_keys[s] = absl::StrFormat("subgroup:%d:%d", gather_id, s);
    }
    TT_ASSIGN_OR_RETURN(const std::vector<std::vector<uint8_t>> peer_values,
                        CallStore("root_store multiGet subgroup:*",
                                  [&] {  // REF_CAPTURE_OK=called synchronously
                                    return root_store->multiGet(peer_keys);
                                  }));
    for (const auto& bytes : peer_values) {
      subgroup_csvs.emplace_back(reinterpret_cast<const char*>(bytes.data()),
                                 bytes.size());
    }
    if (size_ > 1) {
      TT_RETURN_IF_ERROR(CallStore(
          absl::StrCat("store set ", subgroup_store_key),
          [&] {  // REF_CAPTURE_OK=called synchronously
            store_->set(subgroup_store_key, absl::StrJoin(subgroup_csvs, ";"));
          }));
    }
  } else {
    TT_ASSIGN_OR_RETURN(
        const std::vector<uint8_t> raw_bytes,
        CallStore(absl::StrCat("store get ", subgroup_store_key),
                  [&] {  // REF_CAPTURE_OK=called synchronously
                    return store_->get(subgroup_store_key);
                  }));
    const std::string_view all_payload(
        reinterpret_cast<const char*>(raw_bytes.data()), raw_bytes.size());
    for (std::string_view csv :
         absl::StrSplit(all_payload, ';', absl::SkipEmpty())) {
      subgroup_csvs.emplace_back(csv);
    }
  }

  std::vector<DeviceGroup> unique_subgroups;
  absl::flat_hash_set<int64_t> seen_ranks;
  for (std::string_view payload : subgroup_csvs) {
    DeviceGroup group;
    for (std::string_view p : absl::StrSplit(payload, ',')) {
      int64_t r = 0;
      TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=defensive check on internally
                     // serialized store payload
          absl::SimpleAtoi(p, &r), error::kInternal)
          << "Failed to parse subgroup world rank from store: " << p;
      TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=internal rank range invariant
          r >= 0 && r < world_size, error::kInternal)
          << "subgroup world rank " << r << " out of range for world size "
          << world_size;
      group.push_back(r);
    }
    if (!group.empty() && !seen_ranks.contains(group[0])) {
      for (int64_t r : group) {
        TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=internal partition invariant
            seen_ranks.insert(r).second, error::kInvalidArgument)
            << "Overlapping subgroups are not supported in SPMD.";
      }
      unique_subgroups.push_back(std::move(group));
    }
  }

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=internal partition invariant
      static_cast<int64_t>(unique_subgroups.size()) * size_ == world_size,
      error::kInvalidArgument)
      << "Incomplete or overlapping subgroups are not supported in SPMD.";
  // Order subgroups by their leader's physical topology position only (see
  // `ComputeRankToTopologyOrder`), matching Python `topology_aware_mesh`. Never
  // order by world rank: world ranks come from launcher order and would make
  // the per-slice device_assignment order (and thus Megascale's per-slice
  // collective pattern) depend on pod launch order. Members within a subgroup
  // keep PG-rank order.
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=world topology initialized before PGs
      static_cast<int64_t>(topo.rank_to_topo_order.size()) == world_size,
      error::kInternal)
      << "world topology order not initialized";
  absl::c_sort(
      unique_subgroups, [&topo](const DeviceGroup& a, const DeviceGroup& b) {
        return topo.rank_to_topo_order[a[0]] < topo.rank_to_topo_order[b[0]];
      });
  return DeviceGroupList(std::move(unique_subgroups), /*normalize=*/false);
}

ProcessGroupTpu::ProcessGroupTpu(c10::intrusive_ptr<c10d::Store> store,
                                 int rank, int group_size)
    : c10d::Backend(rank, group_size),
      store_(std::move(store)),
      p2p_send_queue_(store_) {
  static std::atomic<int64_t> next_pg_id{0};
  const int64_t pg_id = next_pg_id.fetch_add(1);
  const int64_t ctor_t0 = NowMs();
  ABSL_VLOG(1) << "[ProcessGroupTpu] ctor: "
               << "rank: " << rank << ", group size: " << group_size;

  TT_ASSIGN_OR_THROW(const xla::PjRtClient* const pjrt_client,
                     PjrtBackend::GetInstance().GetClient());

  const int64_t world_size = pjrt_client->devices().size();

  // Addressable device "assigned" to this process/rank:
  TT_CHECK_THROW(pjrt_client->addressable_devices().size() == 1,
                 error::kInternal)
      << "expected exactly one TPU device per PyTorch process";
  addressable_device_id_ =
      pjrt_client->addressable_devices()[0]->global_device_id().value();

  // We support creating "flat" process groups, as well as "2D+ mesh" process
  // (sub)groups. The (sub)group size must evenly divide the world size, i.e.
  // the total number of devices.
  TT_CHECK_THROW(world_size % group_size == 0, error::kInternal)
      << "TPU slice size (" << world_size << ") must be divisible by "
      << "process group size (" << group_size << ")";

  const bool is_initial_world_pg = WorldTopology().rank_to_device_id.empty();
  if (is_initial_world_pg) {
    TT_THROW_IF_ERROR(InitializeRankToDeviceMappings(world_size));
  }
  const auto& topo = WorldTopology();
  DeviceGroupList subgroup_member_world_ranks;
  if (is_initial_world_pg) {
    member_world_ranks_.resize(size_);
    std::iota(member_world_ranks_.begin(), member_world_ranks_.end(), 0);
    subgroup_member_world_ranks = DeviceGroupList({member_world_ranks_});
  } else if (size_ == 1) {
    member_world_ranks_ = DeviceGroup{topo.world_rank};
    std::vector<DeviceGroup> unique_subgroups(world_size);
    for (int64_t r = 0; r < world_size; ++r) {
      unique_subgroups[r] = DeviceGroup{r};
    }
    // Order subgroups by their leader's physical topology position only. Never
    // order by world rank: world ranks come from launcher order and would make
    // the per-slice device_assignment order (and thus Megascale's per-slice
    // collective pattern) depend on pod launch order. Members within a
    // subgroup keep PG-rank order.
    TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=world topology initialized before
                     // PGs
        static_cast<int64_t>(topo.rank_to_topo_order.size()) == world_size,
        error::kInternal)
        << "world topology order not initialized";
    absl::c_sort(
        unique_subgroups, [&topo](const DeviceGroup& a, const DeviceGroup& b) {
          return topo.rank_to_topo_order[a[0]] < topo.rank_to_topo_order[b[0]];
        });
    subgroup_member_world_ranks =
        DeviceGroupList(std::move(unique_subgroups), /*normalize=*/false);
  } else if (size_ == world_size ||
             dynamic_cast<c10d::PrefixStore*>(store_.get()) != nullptr) {
    TT_ASSIGN_OR_THROW(
        const NodeLocalStoreConfig local,
        GetNodeLocalStore(world_size, store_ != nullptr
                                          ? std::optional(store_->getTimeout())
                                          : std::nullopt));
    const int64_t gather_id = next_subgroup_gather_id.fetch_add(1);
    TT_THROW_IF_ERROR(GatherSubgroupsViaNodeLocalStore(
        store_.get(), local.store, local.local_rank, local.local_world_size,
        rank_, size_, topo.world_rank, topo.rank_to_device_id,
        topo.rank_to_topo_order, gather_id, member_world_ranks_,
        subgroup_member_world_ranks));
  } else {
    TT_ASSIGN_OR_THROW(member_world_ranks_, GatherMemberWorldRanks());
    TT_ASSIGN_OR_THROW(subgroup_member_world_ranks,
                       GatherAllSubgroups(member_world_ranks_, world_size));
  }
  member_device_ids_.resize(size_);
  for (int64_t i = 0; i < size_; ++i) {
    member_device_ids_[i] = topo.rank_to_device_id[member_world_ranks_[i]];
  }
  // A program that combines this group's collectives with other groups' runs
  // on every subgroup's devices, with the collectives tiled over the
  // subgroups.
  DeviceGroupList tiled_replica_groups;
  PermuteDeviceAssignmentForSubgroups(
      subgroup_member_world_ranks, topo.rank_to_device_id,
      device_assignment_.global, tiled_replica_groups);
  // Compile this group's collectives for its own devices only: partition p is
  // the device of PG rank p. Each process group thus runs independent
  // executables, and XLA's start-of-program barrier spans only these devices.
  device_assignment_.local.assign(member_device_ids_.begin(),
                                  member_device_ids_.end());
  DeviceGroup local_partitions(size_);
  std::iota(local_partitions.begin(), local_partitions.end(), 0);
  replica_groups_ = DeviceGroupList({std::move(local_partitions)});
  ABSL_VLOG(1) << "[ProcessGroupTpu] ctor done: pg_id=" << pg_id
               << ", elapsed=" << NowMs() - ctor_t0
               << " ms, num_subgroups=" << subgroup_member_world_ranks.size();

  ABSL_VLOG(1) << "[ProcessGroupTpu] Rank-to-device-id mapping for "
               << "rank=" << rank << ", group_size=" << group_size << ": "
               << ToString(member_device_ids_);

  ABSL_VLOG(1) << "[ProcessGroupTpu] rank: " << getRank()
               << ", group size: " << getSize()
               << ", device id: " << addressable_device_id_
               << ", group device ids: " << ToString(member_device_ids_)
               << ", replica groups: " << ToString(replica_groups_);
}

void ProcessGroupTpu::shutdown() { p2p_send_queue_.Shutdown(); }

ProcessGroupTpu::~ProcessGroupTpu() { shutdown(); }

// Sets the subgroups fingerprint in the given param cache keys.
#define TT_SET_SUBGROUPS_FINGERPRINT(param_keys) \
  TT_THROW_IF_ERROR(                             \
      param_keys.SetParam("subgroups_fingerprint", replica_groups_))

const DeviceGroupList& ProcessGroupTpu::GetReplicaGroups() const {
  return replica_groups_;
}

absl::Status ProcessGroupTpu::DispatchAllReduceInplace(
    at::Tensor& tensor, c10d::ReduceOp reduce_op, OpParamCacheKeys param_keys,
    bool fence_operand) {
  auto op_builder = [subgroups = replica_groups_,
                     reduce_op = std::move(reduce_op),
                     fence_operand](mlir::MlirOp input) {
    if (fence_operand) {
      input =
          mlir::stablehlo::OptimizationBarrier(input.getBuilder(), {input})[0];
    }
    return BuildDistributedAllReduceShlo(input, reduce_op, subgroups);
  };
  TT_ASSIGN_OR_RETURN(const mlir::ElementType output_dtype,
                      ConvertTo<mlir::ElementType>(tensor.scalar_type()));
  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef result_buffer,
      DispatchOp<1>(std::move(op_builder), tensor,
                    {.out_dtype = output_dtype,
                     .out_dims = tensor.sizes(),
                     .op_param_cache_keys = std::move(param_keys),
                     .split_mode = GetCollectiveSplitMode(),
                     .device_assignment = device_assignment()}));
  return AssignBufferToAtTensor(std::move(result_buffer), tensor);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::allreduce(
    std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts) {
  TT_KERNEL(OpName::kDistributedAllReduce, param_keys, (tensors, opts), {
    TT_SET_SUBGROUPS_FINGERPRINT(param_keys);
    // TODO(vladbelous): Implement support for multiple input/output
    // tensors:
    TT_CHECK_THROW(tensors.size() == 1, error::kPythonNotImplementedError)
        << "does not yet support multiple tensors";

    at::Tensor& tensor = tensors[0];

    TT_THROW_IF_ERROR(ValidateReductionOp(opts.reduceOp, tensor.scalar_type()));

    ABSL_VLOG(1) << "[ProcessGroupTpu::allreduce] Rank: " << getRank()
                 << ", DeviceBufferRef: " << DeviceBufferRefDebugString(tensor);

    TT_THROW_IF_ERROR(DispatchAllReduceInplace(
        tensor, opts.reduceOp, std::move(param_keys), /*fence_operand=*/false));

    // NOTE: from distributed/c10d/ProcessGroupNCCL.cpp;l=3604 in
    // pytorch/pytorch if not async should return nullptr.
    if (opts.asyncOp) {
      return MakeCollectiveWork(tensors, getRank(), c10d::OpType::ALLREDUCE);
    }
    return nullptr;
  });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::broadcast(
    std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts) {
  TT_KERNEL(OpName::kDistributedBroadcast, _,
            (tensors, IgnoreInCacheKey(opts, "Doesn't affect SHLO")), {
              auto src_rank = opts.rootRank;
              auto src_dev_id = member_device_ids_[src_rank];
              auto cur_dev_id = addressable_device_id_;

              // NOTE: NCCL implementation does the same check. Gloo
              // implementation does something different. In either case, the
              // python-side of the API doesn't actually directly expose
              // multi-tensor variant. Revisit this later.
              TT_CHECK_THROW(tensors.size() == 1, error::kInvalidArgument)
                  << "expected a single tensor, got " << tensors.size();
              auto& tensor = tensors[0];

              if (src_dev_id != cur_dev_id) {
                // NOTE: At the pybind layer (distributed/c10d/init.cpp), torch
                // effectively holds on to the TensorImpl of the original
                // input/output. See
                // https://github.com/pytorch/pytorch/issues/159686 for details.
                // This is why we need to copy_ here, and cannot simply
                // re-assign.
                tensor.copy_(at::zeros(tensor.sizes(), tensor.options()));
              } else {
                // Rank 0 (root rank): We must push the exact same sequence of
                // operations to the dispatcher to keep the thread-local
                // `detect_repeated_ops` heuristic synchronized across all
                // ranks. We execute the same ops on a discarded dummy tensor so
                // we don't corrupt the actual broadcast data.
                auto dummy = at::empty(tensor.sizes(), tensor.options());
                dummy.copy_(at::zeros(tensor.sizes(), tensor.options()));
              }
              ABSL_VLOG(1) << OpDebugString("broadcast") << "DeviceBufferRef: "
                           << DeviceBufferRefDebugString(tensor);

              // In the short-term, we implement broadcast in terms of
              // all-reduce, which is suboptimal. This is because
              // stablehlo.collective_broadcast is not implemented for TPUs. We
              // can re-visit this later and implement a more efficient version.
              c10d::AllreduceOptions allreduce_opts{c10d::ReduceOp::SUM,
                                                    opts.timeout, opts.asyncOp};
              auto work_ptr = allreduce(tensors, allreduce_opts);

              if (work_ptr != nullptr) {
                dynamic_cast<TpuWork* absl_nonnull>(work_ptr.get())->opType_ =
                    c10d::OpType::BROADCAST;
              }
              return work_ptr;
            });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::experimental_send(
    std::vector<at::Tensor>& tensors, int dst_rank, int tag) {
  auto held_buffers = std::make_shared<std::vector<DeviceBufferRef>>();
  held_buffers->reserve(tensors.size());
  std::vector<xla::PjRtBuffer*> pjrt_buffers;
  pjrt_buffers.reserve(tensors.size());
  std::vector<xla::CrossHostTransferKey> transfer_keys;
  transfer_keys.reserve(tensors.size());

  int64_t src_device_id = addressable_device_id_;
  int64_t dst_device_id = member_device_ids_[dst_rank];
  for (const auto& tensor : tensors) {
    // Extract the underlying hardware buffer from the tensor.
    TT_ASSIGN_OR_THROW(
        DeviceBufferRef device_buffer,
        MaterializeAndReturn(tensor, MaterializationReason::kDistributedOp));
    TT_ASSIGN_OR_THROW(xla::PjRtBuffer * pjrt_buffer,
                       device_buffer.AwaitBuffer());

    held_buffers->push_back(std::move(device_buffer));
    pjrt_buffers.push_back(pjrt_buffer);
    int64_t seq;
    {
      absl::MutexLock lock(p2p_seq_mu_);
      seq = send_seq_[{dst_rank, tag}]++;
    }
    transfer_keys.push_back(
        GetCrossHostTransferKey(src_device_id, dst_device_id, tag, seq));
  }

  // Initiate the cross-host send logic asynchronously on a background worker
  // thread without blocking on store_.
  TT_ASSIGN_OR_THROW(
      auto send_future,
      CrossHostSendBuffers(dst_rank, std::move(held_buffers),
                           std::move(pjrt_buffers), transfer_keys));

  return c10::make_intrusive<TpuWork>(tensors, getRank(), c10d::OpType::SEND,
                                      std::move(send_future));
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::experimental_recv(
    std::vector<at::Tensor>& tensors, int src_rank, int tag) {
  if (tensors.empty()) {
    return c10::make_intrusive<TpuWork>(tensors, getRank(), c10d::OpType::RECV);
  }

  std::vector<xla::Shape> recv_shapes;
  std::vector<xla::CrossHostTransferKey> transfer_keys;
  transfer_keys.reserve(tensors.size());
  recv_shapes.reserve(tensors.size());

  int64_t src_device_id = member_device_ids_[src_rank];
  int64_t dst_device_id = addressable_device_id_;
  for (const auto& tensor : tensors) {
    TT_ASSIGN_OR_THROW(auto xla_dtype,
                       ConvertTo<xla::PrimitiveType>(tensor.scalar_type()));
    xla::Shape xla_shape = xla::ShapeUtil::MakeShape(xla_dtype, tensor.sizes());
    recv_shapes.push_back(std::move(xla_shape));
    int64_t seq;
    {
      absl::MutexLock lock(p2p_seq_mu_);
      seq = recv_seq_[{src_rank, tag}]++;
    }
    transfer_keys.push_back(
        GetCrossHostTransferKey(src_device_id, dst_device_id, tag, seq));
  }

  TT_ASSIGN_OR_THROW(auto recv_buffers,
                     CrossHostReceiveBuffers(recv_shapes, transfer_keys));

  auto recv_ready_future =
      c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());
  auto remaining_recvs = std::make_shared<std::atomic<size_t>>(tensors.size());

  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    std::unique_ptr<xla::PjRtBuffer> buffer = std::move(recv_buffers[i]);
    auto ready_future = buffer->GetReadyFuture();
    TT_ASSIGN_OR_THROW(DeviceBufferRef device_buffer,
                       DeviceBufferList::CreateMaterialized(std::move(buffer)));
    DeviceBufferRef held_buffer = device_buffer;

    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(device_buffer), tensor));

    ready_future.OnReady(
        [recv_ready_future, remaining_recvs,
         held_buffer = std::move(held_buffer)](const absl::Status& status) {
          if (!status.ok()) {
            recv_ready_future->setErrorIfNeeded(std::make_exception_ptr(
                TtError(TT_ERROR(  // ERROR_COV_INFEASIBLE=internal PJRT receive
                                   // completion failure
                            error::kInternal)
                            << "PjRtBuffer::GetReadyFuture failed: " << status,
                        TT_SOURCE_LOCATION)));
          } else if (remaining_recvs->fetch_sub(1, std::memory_order_acq_rel) ==
                     1) {
            recv_ready_future->markCompleted(c10::IValue());
          }
        });
  }

  return c10::make_intrusive<TpuWork>(tensors, getRank(), c10d::OpType::RECV,
                                      std::move(recv_ready_future));
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::allgather(
    std::vector<std::vector<at::Tensor>>& output_tensors,
    std::vector<at::Tensor>& input_tensors,
    const c10d::AllgatherOptions& opts) {
  TT_KERNEL(
      OpName::kDistributedAllGather, param_keys,
      (output_tensors, input_tensors, opts), {
        TT_SET_SUBGROUPS_FINGERPRINT(param_keys);
        const int64_t group_size = getSize();
        TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=PyTorch only ever passes
                         // single-element input lists.
            input_tensors.size() == 1 && output_tensors.size() == 1,
            error::kPythonNotImplementedError)
            << "multiple input tensors not supported";

        auto& input_tensor = input_tensors[0];
        std::vector<at::Tensor>& output_tensor_list = output_tensors[0];
        TT_CHECK_THROW(output_tensor_list.size() == group_size,
                       error::kInvalidArgument)
            << "expected output tensor list to have one tensor per process, "
            << "got "
            << FormatCount(output_tensor_list.size(), "tensor", "tensors")
            << " and " << FormatCount(group_size, "process", "processes");

        // NOTE: PyTorch NCCL backend does support non-uniform shapes
        // (implemented as a sequence of broadcasts, one per tensor). We decided
        // not to support this yet. Revisit later if needed.
        TT_THROW_IF_ERROR(ValidateTensorsUniformShape(output_tensor_list))
                .SetPrepend()
            << "failed to validate output tensors: ";

        TT_CHECK_THROW(input_tensor.sizes() == output_tensor_list[0].sizes(),
                       error::kInvalidArgument)
            << "input tensor shape " << input_tensor.sizes()
            << " must match output tensor shape "
            << output_tensor_list[0].sizes();

        ABSL_VLOG(1) << OpDebugString("allgather") << "DeviceBufferRef: "
                     << DeviceBufferRefDebugString(input_tensor);

        auto& maybe_materialized_input_tensor = input_tensor;

        auto op_builder =
            [subgroups = replica_groups_](
                mlir::MlirOp input) -> absl::StatusOr<DynamicMlirOpResults> {
          return BuildDistributedAllGatherShlo(input, subgroups,
                                               AllGatherOutputMode::kList);
        };

        // The PyTorch layer ensures that all input and output tensors
        // have the same dtype.
        TT_ASSIGN_OR_THROW(auto output_dtype, ConvertTo<mlir::ElementType>(
                                                  input_tensor.scalar_type()));
        const std::vector<mlir::ElementType> output_dtypes(group_size,
                                                           output_dtype);
        // When all-gathering scalars, StableHLO will return a list of shape [1]
        // MlirOps/PjRtBuffers.
        int64_t scalar_size[1] = {1};
        std::vector<absl::Span<const int64_t>> output_dims;
        output_dims.reserve(group_size);
        for (const auto& output_tensor : output_tensor_list) {
          if (output_tensor.dim() == 0) {
            output_dims.push_back(scalar_size);
          } else {
            output_dims.push_back(output_tensor.sizes());
          }
        }

        TT_ASSIGN_OR_THROW(
            auto result_buffers,
            (DispatchOp<1, kDynamicSize>(
                std::move(op_builder), maybe_materialized_input_tensor,
                {.out_dtypes = output_dtypes,
                 .out_dims_list = output_dims,
                 .op_param_cache_keys = std::move(param_keys),
                 .split_mode = GetCollectiveSplitMode(),
                 .device_assignment = device_assignment()})));

        // TODO: respect async.

        ABSL_CHECK_EQ(result_buffers.size(), group_size);  // CRASH_OK
        for (int64_t i = 0; i < group_size; ++i) {
          TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buffers[i]),
                                                   output_tensor_list[i]));
        }

        if (opts.asyncOp) {
          return MakeCollectiveWork(output_tensor_list, getRank(),
                                    c10d::OpType::ALLGATHER);
        }
        return nullptr;
      });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::_allgather_base(
    at::Tensor& output_tensor, at::Tensor& input_tensor,
    const c10d::AllgatherOptions& opts) {
  TT_KERNEL(
      OpName::kDistributedAllGatherIntoTensor, param_keys,
      (output_tensor, input_tensor, opts), {
        TT_SET_SUBGROUPS_FINGERPRINT(param_keys);
        const int64_t world_size = getSize();
        auto input_sizes = input_tensor.sizes();
        auto output_sizes = output_tensor.sizes();
        AllGatherOutputMode output_mode;

        if (input_tensor.dim() == 0) {
          // Stacks scalars into a 1-dimensional tensor.
          // The MLIR Op builder wraps each scalar as a 1-dimensional tensor.
          // Concatenating these tensors achieves the desired output shape.
          output_mode = AllGatherOutputMode::kConcat;
          TT_CHECK_THROW(
              output_tensor.dim() == 1 && output_sizes[0] == world_size,
              error::kInvalidArgument)
              << "expected output for scalar inputs to be a 1D tensor with "
                 "size equal to the world size, got output shape "
              << output_sizes << " and world size " << world_size;
        } else if (output_tensor.dim() == input_tensor.dim() + 1) {
          // Stacks tensors.
          output_mode = AllGatherOutputMode::kStack;
          TT_CHECK_THROW(output_sizes[0] == world_size, error::kInvalidArgument)
              << "expected output for stacking to be a tensor with size equal "
                 "to the world size at dimension 0, got output shape "
              << output_sizes << ", input shape " << input_sizes
              << ", and world size " << world_size;
          for (int i = 0; i < input_sizes.size(); ++i) {
            TT_CHECK_THROW(output_sizes[i + 1] == input_sizes[i],
                           error::kInvalidArgument)
                << "expected output for stacking to match the input tensor "
                   "shape along all other dimensions, got output shape "
                << output_sizes << " and input shape " << input_sizes;
          }
        } else if (output_tensor.dim() == input_tensor.dim()) {
          // Concatenates tensors.
          output_mode = AllGatherOutputMode::kConcat;
          TT_CHECK_THROW(output_sizes[0] == input_sizes[0] * world_size,
                         error::kInvalidArgument)
              << "expected output for concatenation to be a tensor with size "
                 "equal to world size * input size at dimension 0, got output "
                 "shape "
              << output_sizes << ", input shape " << input_sizes
              << ", and world size " << world_size;
          for (int i = 1; i < input_sizes.size(); ++i) {
            TT_CHECK_THROW(output_sizes[i] == input_sizes[i],
                           error::kInvalidArgument)
                << "expected output for concatenation to match the input "
                   "tensor shape along all other dimensions, got output shape "
                << output_sizes << " and input shape " << input_sizes;
          }
        } else {
          // Arbitrary value to make compiler happy.
          output_mode = AllGatherOutputMode::kConcat;
          TT_CHECK_THROW(false, error::kInvalidArgument)
              << "expected number of output dimensions to be equal the number "
                 "of input dimensions (concatenation) or input dimensions + 1 "
                 "(stacking), got output shape "
              << output_sizes << " and input shape " << input_sizes;
        }

        ABSL_VLOG(1) << OpDebugString("_allgather_base") << "DeviceBufferRef: "
                     << DeviceBufferRefDebugString(input_tensor);

        auto& maybe_materialized_input_tensor = input_tensor;

        auto op_builder =
            [subgroups = replica_groups_,
             output_mode](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          TT_ASSIGN_OR_RETURN(auto results, BuildDistributedAllGatherShlo(
                                                input, subgroups, output_mode));
          // `results` must have a single element because this kernel only
          // invokes BuildDistributedAllGatherShlo with kConcat or kStack.
          ABSL_CHECK_EQ(results.size(), 1);  // CRASH_OK
          return results[0];
        };

        TT_ASSIGN_OR_THROW(
            const auto output_dtype,
            ConvertTo<mlir::ElementType>(input_tensor.scalar_type()));
        TT_ASSIGN_OR_THROW(
            auto result_buffer,
            DispatchOp<1>(std::move(op_builder),
                          maybe_materialized_input_tensor,
                          {.out_dtype = output_dtype,
                           .out_dims = output_tensor.sizes(),
                           .op_param_cache_keys = std::move(param_keys),
                           .split_mode = GetCollectiveSplitMode(),
                           .device_assignment = device_assignment()}));

        // TODO: b/443120101 - defer if opts.asyncOp is true.

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result_buffer), output_tensor));

        if (opts.asyncOp) {
          return MakeCollectiveWork(std::vector<at::Tensor>{output_tensor},
                                    getRank(), c10d::OpType::_ALLGATHER_BASE);
        }
        return nullptr;
      });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs,
    const c10d::AllgatherOptions& opts) {
  TT_KERNEL(
      OpName::kDistributedAllGatherIntoTensor, _, (outputs, inputs, opts), {
        // NOTE: there's no such op in `torch.distributed` module, so it
        // is called indirectly.
        TT_CHECK_THROW(inputs.size() == outputs.size(), error::kInvalidArgument)
            << "inputs and outputs must have the same size.";

        // Calls allgather once for each input. This is an acceptable
        // simplification for now as we are primarily focused on supporting
        // higher level distributed APIs such as DTensor, which only ever call
        // this method with a single input.
        c10d::AllgatherOptions inner_opts = opts;
        inner_opts.asyncOp = false;
        for (int i = 0; i < inputs.size(); ++i) {
          _allgather_base(outputs[i], inputs[i], inner_opts);
        }

        if (opts.asyncOp) {
          return MakeCollectiveWork(outputs, getRank(),
                                    c10d::OpType::ALLGATHER_COALESCED);
        }
        return nullptr;
      });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::gather(
    std::vector<std::vector<at::Tensor>>& output_tensors,
    std::vector<at::Tensor>& input_tensors, const c10d::GatherOptions& opts) {
  TT_KERNEL(
      OpName::kDistributedGather, param_keys,
      (output_tensors, input_tensors, opts), {
        TT_SET_SUBGROUPS_FINGERPRINT(param_keys);
        const int64_t rank = getRank();
        const int64_t root_rank = opts.rootRank;
        TT_CHECK_THROW(input_tensors.size() == 1, error::kInvalidArgument)
            << "expected a single input tensor, got " << input_tensors.size();
        auto& input = input_tensors[0];

        if (rank == root_rank) {
          TT_CHECK_THROW(output_tensors.size() == 1, error::kInvalidArgument)
              << "expected a single list of output tensors on the root rank, "
                 "got "
              << output_tensors.size();
          const auto& output_tensor_list = output_tensors[0];
          TT_CHECK_THROW(output_tensor_list.size() == getSize(),
                         error::kInvalidArgument)
              << "expected the number of output tensors on the root rank to be "
                 "equal to the group size, got "
              << output_tensor_list.size() << " tensors and " << getSize()
              << " processes";
          TT_THROW_IF_ERROR(ValidateTensorsUniformShape(output_tensor_list))
                  .SetPrepend()
              << "failed to validate output tensors on root rank: ";
          TT_CHECK_THROW(input.sizes() == output_tensor_list[0].sizes(),
                         error::kInvalidArgument)
              << "input tensor shape must match output tensor shape";
        } else {
          TT_CHECK_THROW(output_tensors.empty(), error::kInvalidArgument)
              << "on non-root rank " << rank
              << " the list of output tensors must be empty";
        }

        std::vector<at::Tensor> allgather_outputs;
        if (rank == root_rank) {
          allgather_outputs = output_tensors[0];
        } else {
          allgather_outputs.reserve(getSize());
          std::vector<DeviceBufferRef> empty_buffers;
          empty_buffers.reserve(getSize());
          for (int i = 0; i < getSize(); ++i) {
            auto tensor = at::empty_like(input);
            // Materialize the outputs before the collective to ensure symmetry.
            // This is needed because on the root rank, output_tensors[0] might
            // have deferred ops. On non-root ranks, the temporary tensors are
            // fresh, which can result in an asymmetry.
            TT_ASSIGN_OR_THROW(auto empty_buffer, GetBaseBuffer(tensor));
            empty_buffers.push_back(std::move(empty_buffer));
            allgather_outputs.push_back(std::move(tensor));
          }
          TT_THROW_IF_ERROR(Materialize(empty_buffers,
                                        MaterializationReason::kDistributedOp));
        }

        c10d::AllgatherOptions allgather_opts;
        allgather_opts.timeout = opts.timeout;
        allgather_opts.asyncOp = opts.asyncOp;

        std::vector<std::vector<at::Tensor>> allgather_output_tensors = {
            std::move(allgather_outputs)};
        auto work_ptr =
            allgather(allgather_output_tensors, input_tensors, allgather_opts);

        if (work_ptr != nullptr) {
          auto* tpu_work = dynamic_cast<TpuWork* absl_nonnull>(work_ptr.get());
          tpu_work->opType_ = c10d::OpType::GATHER;
          if (rank != root_rank) {
            tpu_work->results_.clear();
          }
        }

        return work_ptr;
      });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const c10d::ScatterOptions& opts) {
  TT_KERNEL(
      OpName::kDistributedScatter, _,
      (outputs, inputs, IgnoreInCacheKey(opts, "Doesn't affect SHLO")), {
        const int64_t rank = getRank();
        const int64_t root_rank = opts.rootRank;
        TT_CHECK_THROW(outputs.size() == 1, error::kInvalidArgument)
            << "expected a single output tensor, got " << outputs.size();
        auto& output = outputs[0];
        bool is_scalar = output.dim() == 0;

        at::Tensor scatter_input;  // UNINITIALIZED_TENSOR_OK
        if (rank == root_rank) {
          TT_CHECK_THROW(inputs.size() == 1, error::kInvalidArgument)
              << "expected a single list of input tensors on the root rank, "
                 "got "
              << inputs.size();
          const auto& input_tensors = inputs[0];
          TT_CHECK_THROW(input_tensors.size() == getSize(),
                         error::kInvalidArgument)
              << "expected the number of input tensors on the root rank to be "
                 "equal to the group size, got "
              << input_tensors.size() << " tensors and " << getSize()
              << " processes";

          TT_THROW_IF_ERROR(ValidateTensorsUniformShape(input_tensors))
                  .SetPrepend()
              << "failed to validate input tensors on root rank: ";
          TT_CHECK_THROW(output.sizes() == input_tensors[0].sizes(),
                         error::kInvalidArgument)
              << "expected output tensor shape to match input tensor shape, "
                 "got "
              << output.sizes() << " and " << input_tensors[0].sizes();
          if (is_scalar) {
            scatter_input = at::stack(input_tensors);
          } else {
            scatter_input = at::cat(input_tensors, 0);
          }
        } else {
          TT_CHECK_THROW(inputs.empty(), error::kInvalidArgument)
              << "on non-root rank " << rank
              << " the list of input tensors must be empty";

          if (is_scalar) {
            scatter_input = at::zeros({getSize()}, output.options());
          } else {
            auto scatter_input_dims = CopyIntVector(output.sizes());
            scatter_input_dims[0] *= getSize();
            scatter_input = at::zeros(scatter_input_dims, output.options());
          }
        }

        c10d::ReduceScatterOptions reduce_scatter_opts;
        reduce_scatter_opts.reduceOp = c10d::ReduceOp::SUM;
        reduce_scatter_opts.timeout = opts.timeout;
        reduce_scatter_opts.asyncOp = opts.asyncOp;

        c10::intrusive_ptr<c10d::Work> work_ptr =
            _reduce_scatter_base(output, scatter_input, reduce_scatter_opts);
        if (work_ptr != nullptr) {
          // If async updates the Work object to have the correct op type.
          dynamic_cast<TpuWork* absl_nonnull>(work_ptr.get())->opType_ =
              c10d::OpType::SCATTER;
        }
        return work_ptr;
      });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::reduce_scatter(
    std::vector<at::Tensor>& output_tensors,
    std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10d::ReduceScatterOptions& opts) {
  TT_KERNEL(
      OpName::kDistributedReduceScatter, _,
      (output_tensors, input_tensors,
       IgnoreInCacheKey(opts, "Delegates to _reduce_scatter_base")),
      {
        // NOTE: Python side API only exposes single-element reduce_scatter op.
        // Same validation is done in NCCL backend.
        TT_CHECK_THROW(input_tensors.size() == 1,
                       error::kPythonNotImplementedError)
            << "multiple input tensor lists not supported";
        TT_CHECK_THROW(output_tensors.size() == 1,
                       error::kPythonNotImplementedError)
            << "multiple output tensor lists not supported";

        std::vector<at::Tensor>& inputs = input_tensors[0];
        at::Tensor& output = output_tensors[0];

        TT_CHECK_THROW(inputs.size() == getSize(), error::kInvalidArgument)
            << "expected 1 input tensor for each process, got " << inputs.size()
            << " input tensors and " << getSize() << " processes";

        // NOTE: NCCL does support this case, but is doing that by running a
        // sequence of world_size separate reduce calls (coalesced). Revisit
        // later if we want to support this case too. It would be very
        // problematic for error handling.
        TT_THROW_IF_ERROR(ValidateTensorsUniformShape(inputs)).SetPrepend()
            << "failed to validate input tensors: ";

        TT_CHECK_THROW(output.sizes() == inputs[getRank()].sizes(),
                       error::kInvalidArgument)
            << "output tensor shape " << output.sizes()
            << " must match input tensor shape " << inputs[getRank()].sizes();

        // We have `_reduce_scatter_base`, which aligns much better with
        // what StableHLO expects for reduce-scatter, so we delegate to that.
        at::Tensor input = at::cat(inputs);
        auto work_ptr = _reduce_scatter_base(output, input, opts);
        if (work_ptr != nullptr) {
          dynamic_cast<TpuWork* absl_nonnull>(work_ptr.get())->opType_ =
              c10d::OpType::REDUCE_SCATTER;
        }
        return work_ptr;
      });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::_reduce_scatter_base(
    at::Tensor& output, at::Tensor& input,
    const c10d::ReduceScatterOptions& opts) {
  TT_KERNEL(
      OpName::kDistributedReduceScatterTensor, param_keys,
      (output, input, opts), {
        TT_SET_SUBGROUPS_FINGERPRINT(param_keys);
        TT_THROW_IF_ERROR(
            ValidateReductionOp(opts.reduceOp, input.scalar_type()));

        TT_ASSIGN_OR_THROW(
            auto input_mode,
            ValidateReduceScatterInputTensorShape(output, input, getSize()));

        // Reduces to the concat case as that's what StableHLO expects.
        at::Tensor input_concat;  // UNINITIALIZED_TENSOR_OK
        if (input_mode == ReduceScatterShapeMode::kConcat) {
          input_concat = input;
        } else {
          input_concat = at::flatten(input, /*start_dim=*/0, /*end_dim=*/1);
        }

        ABSL_VLOG(1) << OpDebugString("_reduce_scatter_base")
                     << "DeviceBufferRef: "
                     << DeviceBufferRefDebugString(input_concat);

        auto& maybe_materialized_input_concat_tensor = input_concat;

        auto op_builder = [reduce_op = opts.reduceOp,
                           subgroups = replica_groups_](mlir::MlirOp input) {
          return BuildDistributedReduceScatterShlo(input, reduce_op, subgroups);
        };

        TT_ASSIGN_OR_THROW(auto out_dtype,
                           ConvertTo<mlir::ElementType>(output.scalar_type()));
        // Calculate the output shape by splitting on dim=0 (our scatter
        // dimension).
        Dimensions out_dims =
            CopyIntVector(maybe_materialized_input_concat_tensor.sizes());
        size_t group_size = replica_groups_[0].size();
        ABSL_CHECK_EQ(out_dims[0] % group_size, 0);  // CRASH_OK
        out_dims[0] /= group_size;

        TT_ASSIGN_OR_THROW(
            auto result_buffer,
            DispatchOp<1>(std::move(op_builder),
                          maybe_materialized_input_concat_tensor,
                          {.out_dtype = out_dtype,
                           .out_dims = std::move(out_dims),
                           .op_param_cache_keys = std::move(param_keys),
                           .split_mode = GetCollectiveSplitMode(),
                           .device_assignment = device_assignment()}));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result_buffer), output));

        if (opts.asyncOp) {
          return MakeCollectiveWork(std::vector<at::Tensor>{output}, getRank(),
                                    c10d::OpType::_REDUCE_SCATTER_BASE);
        }
        return nullptr;
      });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs,
    const c10d::ReduceScatterOptions& opts) {
  TT_KERNEL(
      OpName::kDistributedReduceScatterTensorCoalesced, _,
      (outputs, inputs,
       IgnoreInCacheKey(opts, "Delegates to _reduce_scatter_base")),
      {
        TT_CHECK_THROW(inputs.size() == outputs.size(), error::kInvalidArgument)
            << "inputs and outputs must have the same size, got "
            << inputs.size() << " inputs and " << outputs.size() << " outputs";

        // Calls reduce_scatter once for each input. This is an acceptable
        // simplification for now as we are primarily focused on supporting
        // higher level distributed APIs such as DTensor,
        // which only ever call this method with a single input.
        c10d::ReduceScatterOptions inner_opts = opts;
        inner_opts.asyncOp = false;
        for (int i = 0; i < inputs.size(); ++i) {
          _reduce_scatter_base(outputs[i], inputs[i], inner_opts);
        }

        if (opts.asyncOp) {
          return MakeCollectiveWork(outputs, getRank(),
                                    c10d::OpType::_REDUCE_SCATTER_BASE);
        }
        return nullptr;
      });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::alltoall_base(
    at::Tensor& output, at::Tensor& input,
    std::vector<int64_t>& output_split_sizes,  // INT_VEC_OK
    std::vector<int64_t>& input_split_sizes,   // INT_VEC_OK
    const c10d::AllToAllOptions& opts) {
  TT_KERNEL(OpName::kDistributedAllToAllSingle, _,
            (output, input,
             IgnoreInCacheKey(output_split_sizes, "Doesn't affect SHLO"),
             IgnoreInCacheKey(input_split_sizes, "Doesn't affect SHLO"), opts),
            {
              const int64_t rank = getRank();
              const bool async = opts.asyncOp;
              const int64_t group_size = replica_groups_[0].size();

              // Check on input and output dtypes already done in the
              // PyTorch layer. Hence, we do not need to check for dtypes
              // here.

              TT_THROW_IF_ERROR(ValidateSplitSizesForAllToAllSingle(
                  input_split_sizes, input, group_size));
              TT_THROW_IF_ERROR(ValidateSplitSizesForAllToAllSingle(
                  output_split_sizes, output, group_size));

              // Take the equal-splits fast path only when both split lists are
              // empty and shapes match. Explicit split lists or mismatched
              // local shapes must validate pairwise peer contracts first.
              const bool use_equal_splits_fast_path =
                  output.sizes() == input.sizes() &&
                  output_split_sizes.empty() && input_split_sizes.empty();

              std::optional<DeviceBufferRef> zero_elem_buf;
              if (use_equal_splits_fast_path) {
                TT_ASSIGN_OR_THROW(DeviceBufferRef result_buf,
                                   AllToAllBaseEqualSplits(output, input));
                TT_THROW_IF_ERROR(
                    AssignBufferToAtTensor(std::move(result_buf), output));
              } else {
                TT_ASSIGN_OR_THROW(
                    DeviceBufferRef result_buf,
                    AllToAllBaseUnevenSplits(output, input, output_split_sizes,
                                             input_split_sizes));
                // A 0-element output was dispatched with a 1-tile dummy buffer
                // whose shape differs from `output`. Keep that dummy buffer
                // directly so async work tracking or sync materialization
                // still executes the collective on this rank.
                if (output.numel() > 0) {
                  TT_THROW_IF_ERROR(
                      AssignBufferToAtTensor(std::move(result_buf), output));
                } else {
                  zero_elem_buf = std::move(result_buf);
                }
              }

              if (async) {
                if (zero_elem_buf.has_value()) {
                  return MakeCollectiveWorkWithBuffers(
                      std::vector<at::Tensor>{output},
                      std::vector<DeviceBufferRef>{*std::move(zero_elem_buf)},
                      rank, c10d::OpType::ALLTOALL_BASE);
                }
                return MakeCollectiveWork(std::vector<at::Tensor>{output}, rank,
                                          c10d::OpType::ALLTOALL_BASE);
              }
              if (zero_elem_buf.has_value()) {
                TT_THROW_IF_ERROR(Materialize(
                    *zero_elem_buf, MaterializationReason::kDistributedOp));
              }
              return nullptr;
            });
}

namespace {

// Materializes `tensor` into an allocated device buffer when its buffer or
// underlying storage base is still an unmaterialized constant or empty op.
absl::StatusOr<at::Tensor> MaterializeIfConstant(const at::Tensor& tensor) {
  TT_ASSIGN_OR_RETURN(DeviceBufferRef buf, GetBuffer(tensor));
  TT_ASSIGN_OR_RETURN(DeviceBufferRef base_buf, GetBaseBuffer(tensor));
  // Skip materializing non-constant tensors so deferred ops on the stream do
  // not flush unnecessarily.
  if (!buf.is_constant() && !base_buf.is_constant()) {
    return tensor;
  }
  TT_RETURN_IF_ERROR(Materialize(buf, MaterializationReason::kDistributedOp));
  // Return `tensor` directly when `buf` is already its storage base so we do
  // not wrap the base buffer in a second `c10::DataPtr`. For a view of a
  // constant base, return `MakeTensor(buf)` so `DispatchOp` consumes the
  // materialized slice instead of re-slicing the still-constant base.
  if (buf.device_buffer_list() == base_buf.device_buffer_list() &&
      buf.index() == base_buf.index()) {
    return tensor;
  }
  return MakeTensor(std::move(buf));
}

// Substitutes a 1-tile dummy buffer when `tensor` has 0 rows along dim 0 so
// XLA does not fold 0-element reshapes or reject output buffer aliasing in the
// XLA:TPU backend.
absl::StatusOr<at::Tensor> SubstituteZeroRowDispatchTensor(
    const at::Tensor& tensor) {
  if (tensor.size(0) != 0) {
    return tensor;
  }
  Dimensions dummy_sizes = CopyIntVector(tensor.sizes());
  // Use 1024 elements for 1D tensors so this rank keeps tile alignment with
  // non-empty 1D peers during `(8, 128)` factoring.
  dummy_sizes[0] = (tensor.dim() == 1) ? k1DRaggedAllToAllTileElements : 1;
  return MakeEmptyTensor(dummy_sizes, tensor.scalar_type(), tensor.device());
}

absl::StatusOr<DeviceBufferRef> DispatchUnevenAllToAll(
    const at::Tensor& materialized_input, const at::Tensor& materialized_output,
    RaggedAllToAllOffsets offsets, const DeviceGroupList& replica_groups,
    const DeviceAssignment& dev_assignment) {
  TT_ASSIGN_OR_RETURN(auto param_keys,
                      TT_MAKE_OP_PARAM_CACHE_KEYS(replica_groups));
  TT_RETURN_IF_ERROR(SetRaggedAllToAllParamKeys(param_keys, offsets));

  auto op_builder =
      [device_groups = replica_groups,
       offsets = std::move(offsets)](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
        auto& [in_op, out_op] = inputs;
        return BuildDistributedAllToAllBaseUnevenSplitsShlo(
            in_op, out_op, offsets.input_offsets, offsets.send_sizes,
            offsets.output_offsets, offsets.recv_sizes, offsets.flat_factor,
            device_groups);
      };

  TT_ASSIGN_OR_RETURN(
      auto output_dtype,
      ConvertTo<mlir::ElementType>(materialized_output.scalar_type()));
  return DispatchOp<2>(std::move(op_builder),
                       {materialized_input, materialized_output},
                       {.out_dtype = output_dtype,
                        .out_dims = materialized_output.sizes(),
                        .op_param_cache_keys = std::move(param_keys),
                        .split_mode = GetCollectiveSplitMode(),
                        .device_assignment = dev_assignment});
}

}  // namespace

absl::StatusOr<DeviceBufferRef> ProcessGroupTpu::AllToAllBaseEqualSplits(
    at::Tensor& output, at::Tensor& input) {
  auto& maybe_materialized_input_tensor = input;
  TT_RET_CHECK(output.sizes() == input.sizes(), error::kInvalidArgument)
      << "expected shape of input and output to be the same for equal splits, "
      << "got " << input.sizes() << " and " << output.sizes();

  TT_ASSIGN_OR_RETURN(auto param_keys,
                      TT_MAKE_OP_PARAM_CACHE_KEYS(replica_groups_));

  auto op_builder = [device_groups = replica_groups_](mlir::MlirOp input) {
    return BuildDistributedAllToAllBaseShlo(input, device_groups);
  };

  TT_ASSIGN_OR_RETURN(auto output_dtype,
                      ConvertTo<mlir::ElementType>(output.scalar_type()));

  // Give the all_to_all split is on input tensor dimension 0,
  // the output tensor shape will be the same as the input tensor shape.
  return DispatchOp<1>(std::move(op_builder), maybe_materialized_input_tensor,
                       {.out_dtype = output_dtype,
                        .out_dims = output.sizes(),
                        .op_param_cache_keys = std::move(param_keys),
                        .split_mode = GetCollectiveSplitMode(),
                        .device_assignment = device_assignment()});
}

absl::StatusOr<DeviceBufferRef> ProcessGroupTpu::AllToAllBaseUnevenSplits(
    at::Tensor& output, at::Tensor& input, c10::IntArrayRef output_split_sizes,
    c10::IntArrayRef input_split_sizes) {
  auto& maybe_materialized_input_tensor = input;
  TT_RET_CHECK(output.dim() == input.dim(), error::kInvalidArgument)
      << "expected input and output tensors to have the same number of "
      << "dimensions, got " << input.dim() << " and " << output.dim();
  TT_RET_CHECK(output.sizes().slice(1) == input.sizes().slice(1),
               error::kInvalidArgument)
      << "expected trailing dimensions of input and output to match, got "
      << input.sizes() << " and " << output.sizes();

  const size_t group_size = replica_groups_[0].size();
  TT_ASSIGN_OR_RETURN(
      const std::vector<int32_t> send_sizes,
      BuildAllToAllSplitSizes(input_split_sizes, input.size(0), group_size));
  TT_ASSIGN_OR_RETURN(
      const std::vector<int32_t> recv_sizes,
      BuildAllToAllSplitSizes(output_split_sizes, output.size(0), group_size));
  TT_ASSIGN_OR_RETURN(const int32_t input_buffer_size,
                      ValidateDimToInt32(input.size(0), "input dim 0"));
  TT_ASSIGN_OR_RETURN(const int32_t output_buffer_size,
                      ValidateDimToInt32(output.size(0), "output dim 0"));

  TT_ASSIGN_OR_RETURN(
      at::Tensor dispatch_input,
      SubstituteZeroRowDispatchTensor(maybe_materialized_input_tensor));
  TT_ASSIGN_OR_RETURN(at::Tensor dispatch_output,
                      SubstituteZeroRowDispatchTensor(output));

  // Only 1D tensors suffer degenerate rank-1 lane padding and qualify for
  // `(8, 128)` tile factoring. Setting `element_bitwidth` to 0 for ND tensors
  // keeps them unfactored.
  const int element_bitwidth =
      (dispatch_input.dim() == 1 && dispatch_output.dim() == 1)
          ? RaggedAllToAllElementBitwidth(dispatch_input.scalar_type())
          : 0;
  const c10::IntArrayRef trailing_dims = input.sizes().slice(1);
  const uint64_t trailing_shape_fp = Fingerprint(
      absl::Span<const int64_t>(trailing_dims.data(), trailing_dims.size()));
  const RaggedAllToAllBufferInfo buffer_info{
      .input_buffer_size = input_buffer_size,
      .output_buffer_size = output_buffer_size,
      .dispatch_input_size = dispatch_input.size(0),
      .dispatch_output_size = dispatch_output.size(0),
      .scalar_type = dispatch_input.scalar_type(),
      .element_bitwidth = element_bitwidth,
      .tensor_dim = static_cast<int32_t>(input.dim()),
      .trailing_shape_fingerprint = trailing_shape_fp,
  };
  TT_ASSIGN_OR_RETURN(
      RaggedAllToAllOffsets offsets,
      ComputeRaggedAllToAllOffsets(store_.get(), getRank(), alltoall_seq_,
                                   alltoall_shm_nonce_, send_sizes, recv_sizes,
                                   buffer_info));
  // Delegate to `stablehlo.all_to_all` when explicit split lists turn out to
  // be uniform across every rank, avoiding `ragged_all_to_all` overhead.
  if (offsets.uniform_equal_splits && output.sizes() == input.sizes()) {
    return AllToAllBaseEqualSplits(output, input);
  }

  // Materialize constant or empty operands before dispatching the collective so
  // `ProcessDeferredOpEvent` does not inline a TensorCore constant fill into
  // `output` that races with incoming SparseCore RDMAs.
  TT_ASSIGN_OR_RETURN(at::Tensor materialized_input,
                      MaterializeIfConstant(dispatch_input));
  TT_ASSIGN_OR_RETURN(at::Tensor materialized_output,
                      MaterializeIfConstant(dispatch_output));

  return DispatchUnevenAllToAll(materialized_input, materialized_output,
                                std::move(offsets), replica_groups_,
                                device_assignment());
}

// Dispatches the torch.distributed.all_to_all collective, supporting both
// uniform tensor shapes and non-uniform tensor shapes along dimension 0 for
// dynamic MoE token routing.
c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::alltoall(
    std::vector<at::Tensor>& output_tensors,
    std::vector<at::Tensor>& input_tensors, const c10d::AllToAllOptions& opts) {
  TT_KERNEL(
      OpName::kDistributedAllToAll, param_keys,
      (output_tensors, input_tensors, opts), {
        const int64_t rank = getRank();
        const bool async = opts.asyncOp;
        TT_THROW_IF_ERROR(param_keys.SetParam("rank", rank));
        TT_THROW_IF_ERROR(param_keys.SetParam("async", async));
        TT_SET_SUBGROUPS_FINGERPRINT(param_keys);
        TT_CHECK_THROW(output_tensors.size() == input_tensors.size(),
                       error::kInvalidArgument)
            << "expected output and input tensors to have the same number of "
               "tensors, got "
            << output_tensors.size() << " for output and "
            << input_tensors.size() << " for input";

        const int64_t group_size = getSize();

        TT_CHECK_THROW(input_tensors.size() == group_size,
                       error::kInvalidArgument)
            << "expected input tensors to have the same number of tensors as "
               "the process group size, got "
            << input_tensors.size() << " for input and " << group_size
            << " for process group size";

        TT_THROW_IF_ERROR(
            ValidateAllToAllTensors(output_tensors, input_tensors));

        TT_ASSIGN_OR_THROW(
            auto output_dtype,
            ConvertTo<mlir::ElementType>(output_tensors[0].scalar_type()));

        std::vector<mlir::ElementType> result_dtypes(output_tensors.size(),
                                                     output_dtype);

        std::vector<absl::Span<const int64_t>> result_dims_list;
        result_dims_list.reserve(output_tensors.size());
        for (const auto& output_tensor : output_tensors) {
          result_dims_list.push_back(output_tensor.sizes());
        }

        std::vector<DeviceBufferRef> result_buffers;

        if (IsUniformAllToAll(output_tensors, input_tensors)) {
          NAryMlirOpBuilder<kDynamicSize, kDynamicSize> op_builder =
              [device_groups = replica_groups_](absl::Span<mlir::MlirOp> inputs,
                                                mlir::MlirBuilder& builder) {
                return BuildDistributedAllToAllShlo(inputs, device_groups);
              };

          TT_ASSIGN_OR_THROW(result_buffers,
                             (DispatchOp<kDynamicSize, kDynamicSize>(
                                 std::move(op_builder), input_tensors,
                                 {.out_dtypes = result_dtypes,
                                  .out_dims_list = result_dims_list,
                                  .op_param_cache_keys = std::move(param_keys),
                                  .split_mode = GetCollectiveSplitMode(),
                                  .device_assignment = device_assignment()})));
        } else {
          // Non-uniform / uneven tensor list shapes across ranks (e.g. for MoE
          // dynamic token routing).
          //
          // We decompose uneven `all_to_all` into three isolated stages:
          //   1. Pre-collective dimension-0 concatenation
          //   (`concatenated_input`)
          //      and uninitialized destination buffer allocation
          //      (`output_placeholder`).
          //   2. Isolated `AllToAllBaseUnevenSplits` collective execution on
          //      the pre-materialized, symmetrically sized buffers.
          //   3. Post-collective dimension-0 slicing into individual
          //      destination tensors (`SliceTensorsAlongDim0`).
          std::vector<int64_t> input_split_sizes(group_size);   // INT_VEC_OK
          std::vector<int64_t> output_split_sizes(group_size);  // INT_VEC_OK
          std::vector<int32_t> slice_sizes(group_size);
          int64_t total_input_size = 0;
          int64_t total_output_size = 0;
          for (size_t i = 0; i < group_size; ++i) {
            input_split_sizes[i] = input_tensors[i].size(0);
            output_split_sizes[i] = output_tensors[i].size(0);
            slice_sizes[i] = static_cast<int32_t>(output_split_sizes[i]);
            total_input_size += input_split_sizes[i];
            total_output_size += output_split_sizes[i];
          }

          Dimensions total_input_shape =
              CopyIntVector(input_tensors[0].sizes());
          total_input_shape[0] = total_input_size;

          std::vector<at::Tensor> non_empty_inputs;
          non_empty_inputs.reserve(input_tensors.size());
          for (const auto& tensor : input_tensors) {
            if (tensor.size(0) == 0) continue;
            non_empty_inputs.push_back(tensor);
          }
          auto concat_builder = [output_dtype, total_input_shape](
                                    absl::Span<mlir::MlirOp> inputs,
                                    mlir::MlirBuilder& builder) {
            return ConcatenateTensorsAlongDim0(inputs, output_dtype,
                                               total_input_shape, builder);
          };
          TT_ASSIGN_OR_THROW(
              DeviceBufferRef concatenated_input_buf,
              DispatchOp<kDynamicSize>(
                  std::move(concat_builder), non_empty_inputs,
                  {.out_dtype = output_dtype,
                   .out_dims = total_input_shape,
                   .op_param_cache_keys = std::move(param_keys)}));

          Dimensions total_output_shape =
              CopyIntVector(output_tensors[0].sizes());
          total_output_shape[0] = total_output_size;

          auto empty_builder =
              [output_dtype, total_output_shape](
                  mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
            return BuildFillUninitialized(builder, output_dtype,
                                          total_output_shape);
          };
          TT_ASSIGN_OR_THROW(DeviceBufferRef output_placeholder_buf,
                             DispatchOp<0>(std::move(empty_builder), {},
                                           {.out_dtype = output_dtype,
                                            .out_dims = total_output_shape,
                                            .op_param_cache_keys =
                                                OpParamCacheKeys::Empty()}));

          at::Tensor concatenated_input =
              MakeTensor(std::move(concatenated_input_buf));
          at::Tensor output_placeholder =
              MakeTensor(std::move(output_placeholder_buf));

          TT_ASSIGN_OR_THROW(
              DeviceBufferRef ragged_output_buf,
              AllToAllBaseUnevenSplits(output_placeholder, concatenated_input,
                                       output_split_sizes, input_split_sizes));

          at::Tensor ragged_output = MakeTensor(std::move(ragged_output_buf));
          TT_ASSIGN_OR_THROW(auto slice_param_keys,
                             TT_MAKE_OP_PARAM_CACHE_KEYS(slice_sizes));
          auto slice_builder =
              [slice_sizes = std::move(slice_sizes)](
                  mlir::MlirOp input) -> absl::StatusOr<DynamicMlirOpResults> {
            return SliceTensorsAlongDim0(input, slice_sizes);
          };

          TT_ASSIGN_OR_THROW(
              result_buffers,
              (DispatchOp<1, kDynamicSize>(
                  std::move(slice_builder), ragged_output,
                  {.out_dtypes = result_dtypes,
                   .out_dims_list = result_dims_list,
                   .op_param_cache_keys = std::move(slice_param_keys)})));
        }

        for (int64_t i = 0; i < result_buffers.size(); ++i) {
          TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buffers[i]),
                                                   output_tensors[i]));
        }

        if (async) {
          return MakeCollectiveWork(output_tensors, rank,
                                    c10d::OpType::ALLTOALL);
        }
        return nullptr;
      });
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTpu::barrier(
    const c10d::BarrierOptions& opts) {
  TT_KERNEL(OpName::kDistributedBarrier, param_keys, (opts), {
    TT_SET_SUBGROUPS_FINGERPRINT(param_keys);
    // Check for unsupported options.
    //
    // A process addresses exactly one TPU, so the only device id it can name
    // is 0 (client->addressable_devices()). Accept that id as a no-op --
    // the barrier below always runs on this process's own device -- so that
    // device-agnostic callers such as
    // torch.distributed.barrier(device_ids=[torch.accelerator.current_device_index()])
    // work unchanged. Any other id names a device this process does not own.
    TT_CHECK_THROW(opts.device_ids.empty() ||
                       (opts.device_ids.size() == 1 && opts.device_ids[0] == 0),
                   error::kPythonNotImplementedError)
        << "device_ids in barrier options is not supported, except "
           "device_ids=[0] which names this process's own device";
    TT_CHECK_THROW(opts.timeout == c10d::kUnsetTimeout,
                   error::kPythonNotImplementedError)
        << "timeout in barrier options is not supported";
    TT_CHECK_THROW(!opts.device.has_value(), error::kPythonNotImplementedError)
        << "device in barrier options is not supported";

    at::Tensor dummy_tensor =
        at::ones({1}, at::device(GetPrivateUse1DeviceType()).dtype(at::kLong));
    // Fence the constant operand so XLA does not fold the barrier all-reduce.
    TT_THROW_IF_ERROR(DispatchAllReduceInplace(
        dummy_tensor, c10d::ReduceOp::SUM, std::move(param_keys),
        /*fence_operand=*/true));

    if (!opts.asyncOp) {
      dummy_tensor.item();
      return nullptr;
    }
    return MakeCollectiveWork(std::vector<at::Tensor>{dummy_tensor}, getRank(),
                              c10d::OpType::BARRIER);
  });
}

#if TT_TORCH_VERSION_GE(2, 14)
c10::intrusive_ptr<c10d::Window> ProcessGroupTpu::new_window(
    const std::optional<at::Tensor>& tensor) {
  auto win = c10::make_intrusive<WindowTpu>(
      c10::intrusive_ptr<ProcessGroupTpu>::unsafe_reclaim_from_nonowning(this),
      rank_);
  if (tensor.has_value()) {
    win->tensor_register(*tensor, /*owning=*/true);
  }
  return win;
}
#endif

}  // namespace torch_tpu
