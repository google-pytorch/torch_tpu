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

#include "csrc/eager/events_queue.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <memory>
#include <optional>
#include <ratio>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "c10/core/Stream.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/reroute_warnings.h"
#include "csrc/common/shape.h"
#include "csrc/eager/current_stream.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/execution_task.h"
#include "csrc/eager/split_traversal.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "csrc/pjrt/pjrt_transfer.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "xla/future.h"

namespace torch_tpu {

namespace {

constexpr int kMinRepeatedSubsequenceLength = 2;
constexpr int kMaxRepeatedSubsequenceLength = 128;
constexpr int kMinRepetitionCount = 2;
constexpr int kMaxTorchDevices = 8;

// Creates a new deferred DeviceBufferList which represents a data dependency
// but no actual computation.
//
// This is used to enforce device-side execution timing across event snapshots.
//
// The created op can take any number of inputs (which may be zero), ensuring
// that any executable containing them will not begin executing until the prior
// execution has completed. The returned value will be zero-sized; awaiting its
// materialization will enforce timing without actual execution or memory use.
//
// Note however that any input will still be kept alive by this op, which may
// delay memory freeing if the awaited input could otherwise have been freed.
// NOLINTNEXTLINE:add usage in future CL
SharedDeviceBufferList CreateNoOpDependency(
    absl::Span<const DeviceBufferRef> wait_for = {}) {
  std::vector<Shape> output_shapes = {Shape({0}, mlir::ElementType::UI8)};

  auto op_name = OpName::kTorchTpuInternalDataDependency;
  ScopedPythonContextCapturer capturer(op_name);
  auto op_builder = [](mlir::MlirBuilder& builder,
                       absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    // Intentionally do not check the number of inputs and discard them.
    auto ranked_tensor_type =
        mlir::makeTensorType(builder.getContext(), {0}, mlir::ElementType::UI8);
    auto dense_elements_attr =
        mlir::DenseElementsAttr::getFromRawBuffer(ranked_tensor_type, {});
    return DynamicMlirOpResults{
        mlir::stablehlo::Constant(builder, dense_elements_attr)};
  };
  auto refs_or = DeviceBufferList::CreateDeferred(
      op_name, std::move(op_builder), /*inputs=*/{}, OpParamCacheKeys::Empty(),
      std::move(output_shapes));
  ABSL_CHECK_OK(refs_or);  // CRASH_OK
  return refs_or->at(0).device_buffer_list();
}

void PruneCompletedFutures(std::vector<xla::Future<void>>& futures) {
  futures.erase(
      std::remove_if(futures.begin(), futures.end(),
                     [](const xla::Future<void>& future) {
                       return !future.IsValid() ||
                              (future.IsReady() && future.Await().ok());
                     }),
      futures.end());
}

// Sets the promises that release the next event snapshot of a stream. Setting
// one completes the next snapshot, whose callback releases the snapshot after
// it, so setting each promise directly nests one call per snapshot, and a long
// chain overflows the stack of the thread that completed the first snapshot.
// Instead, a promise released while a thread is already setting promises is
// queued, and that thread sets it after the current one returns, so a chain of
// any length completes in a loop.
class SnapshotChainReleaser {
 public:
  void Release(xla::Promise<void> promise) {
    {
      absl::MutexLock lock(mu_);
      if (releasing_) {
        pending_.push_back(std::move(promise));
        return;
      }
      releasing_ = true;
    }
    promise.Set(absl::OkStatus());
    std::vector<xla::Promise<void>> promises;
    while (true) {
      {
        absl::MutexLock lock(mu_);
        if (pending_.empty()) {
          releasing_ = false;
          return;
        }
        promises.swap(pending_);
      }
      for (xla::Promise<void>& next : promises) {
        next.Set(absl::OkStatus());
      }
      promises.clear();
    }
  }

 private:
  absl::Mutex mu_;
  // Promises released while a thread was setting promises.
  std::vector<xla::Promise<void>> pending_ ABSL_GUARDED_BY(mu_);
  // Whether a thread is setting promises.
  bool releasing_ ABSL_GUARDED_BY(mu_) = false;
};

// Releases the next event snapshot of a stream.
void ReleaseNextSnapshot(xla::Promise<void> promise) {
  static absl::NoDestructor<SnapshotChainReleaser> releaser;
  releaser->Release(std::move(promise));
}

class StreamState {
 public:
  // Clears all tracked DeviceBufferLists from the events queue, and resets the
  // repeated ops heuristic state.
  void Clear() {
    absl::MutexLock lock(mu_);
    repetition_hashes_.clear();
    repetition_i_ = 0;
    repeated_ops_graph_breaks_suspended_ = false;
    repeated_ops_graph_break_requested_ = false;
    // Drop all unresolved SnapshotEvents and fail their promises.
    absl::Status cancel_status = TT_ERROR(error::kCancelled)
                                 << "stream state was cleared";
    for (auto& ready_work_item : ready_work_) {
      if (auto* snapshot_request =
              std::get_if<SnapshotRequest>(&ready_work_item)) {
        snapshot_request->snapshot_promise.Set(cancel_status);
        snapshot_request->ok_callback_promise.Set(absl::OkStatus());
      }
    }
    ready_work_.clear();

    // At this point, there are no "live nodes" holding onto deferred ops, so
    // all ops that have no side effects (directly or indirectly) should be
    // expired.
    std::erase_if(deferred_ops_, [](const DeferredOpEvent& deferred_op_event) {
      return deferred_op_event.side_effects == nullptr &&
             deferred_op_event.no_side_effects.expired();
    });

    // Reset the deferred op hash for all remaining deferred ops.
    for (auto& deferred_op_event : deferred_ops_) {
      deferred_op_event.deferred_op_hash = 0;
    }
    futures_.clear();
    last_device_output_.reset();
    transfer_blocks_.clear();
  }

  // If a deferred op has no side effects, then as soon as all of its DataPtrs
  // go out of scope, it is dead code and can be removed. But if it does have
  // side effects, it must be kept alive until it is executed, even if it is
  // dead code.
  //
  // When using the repeated ops heuristic, we also need to hash the deferred op
  // to efficiently detect repetitions.
  struct DeferredOpEvent {
    static std::optional<DeferredOpEvent> FromDeferredOp(
        const SharedDeviceBufferList& device_buffer_list,
        bool compute_hash = false) {
      if (device_buffer_list->is_constant()) {
        // Compiled mode constants and empty tensors are handled specially;
        // normally, memoizing a computation by materializing it is used to
        // prevent later re-execution.
        // However, materializing a constant prevents later constant folding;
        // re-execution preserves their constantness in later compilations.
        return std::nullopt;
      }

      const auto deferred_op = device_buffer_list->deferred_op();
      if (!deferred_op) {
        return std::nullopt;
      }
      DeferredOpEvent event;
      if (IsSideEffectingOp(deferred_op->op_name())) {
        ABSL_VLOG(3) << "[DeferredOpEvent] Created side-effecting event for "
                     << device_buffer_list.get() << "("
                     << ToString(deferred_op->op_name()) << ")";
        event.side_effects = device_buffer_list;
      } else {
        ABSL_VLOG(3)
            << "[DeferredOpEvent] Created non-side-effecting event for "
            << device_buffer_list.get() << "("
            << ToString(deferred_op->op_name()) << ")";
        event.no_side_effects = device_buffer_list;
      }
      if (compute_hash) {
        event.deferred_op_hash = deferred_op->Hash();
      }
      return event;
    }

    absl_nullable std::shared_ptr<DeviceBufferList> lock() const {
      return side_effects ? side_effects : no_side_effects.lock();
    }

    // At most one of these will be a valid shared pointer.
    // If neither is valid, that indicates the DeviceBufferList has been freed.
    std::weak_ptr<DeviceBufferList> no_side_effects;
    absl_nullable std::shared_ptr<DeviceBufferList> side_effects;

    // If compute_hash is true, this will contain a minimal hash of the
    // deferred op. This is used for heuristically detecting repeated ops, and
    // is not sufficient to serve as a cache key.
    size_t deferred_op_hash = 0;
  };

  void RecordDeferredOpCreated(
      const SharedDeviceBufferList& device_buffer_list) {
    if (auto event = DeferredOpEvent::FromDeferredOp(device_buffer_list);
        event.has_value()) {
      absl::MutexLock lock(mu_);
      deferred_ops_.push_back(std::move(*event));
    }
  }

  bool RecordDeferredOpAndDetectRepetition(
      const SharedDeviceBufferList& device_buffer_list) {
    auto maybe_event = DeferredOpEvent::FromDeferredOp(device_buffer_list,
                                                       /*compute_hash=*/true);
    if (!maybe_event.has_value()) {
      return false;
    }
    auto& event = *maybe_event;
    const size_t new_hash = event.deferred_op_hash;

    absl::MutexLock lock(mu_);
    deferred_ops_.push_back(std::move(event));

    // Fast path: assume that the last sequence is still repeating.
    // For example, if the last six ops were [A, B, C, A, B, C], check if we're
    // part of another [A, B, C] loop.
    switch (CheckLastRepeatedSequence(new_hash)) {
      case StreamState::CheckLastRepeatedSequenceResult::kContinueWaiting:
        // We've seen [A, B] and are expecting [A, B, C]; don't materialize yet.
        return false;
      case StreamState::CheckLastRepeatedSequenceResult::kRepetitionComplete:
        // We expected to see [A, B, C] and just completed [A, B, C];
        // materialize C to take advantage of the already compiled loop region.
        return RequestRepeatedOpsGraphBreakLocked();
      case StreamState::CheckLastRepeatedSequenceResult::kPatternBreak:
        // We expected to see [A, B, C] but just saw op D.
        // Reset and scan for a new repeated pattern.
        break;
    }

    // Scan for a new repeated sequence.
    const int64_t repetition_length = FindRepeatedSequence(new_hash);
    if (repetition_length == 0) {
      // The deferred op queue does not end with a repeated sequence within
      // the min/max sequence length range. No early materializations here.
      return {};
    }

    // The last `repetition_length` ops have been repeated at least twice.
    // Record this to avoid repeated scans.
    RecordRepeatedHashPattern(repetition_length);

    // Get the last node of each loop of the repeated sequence, and the last
    // node in the non-looping prefix (if any).
    return RequestRepeatedOpsGraphBreakLocked();
  }

  void SuspendRepeatedOpsGraphBreaks() {
    absl::MutexLock lock(mu_);
    repeated_ops_graph_breaks_suspended_ = true;
  }

  bool ResumeRepeatedOpsGraphBreaks() {
    absl::MutexLock lock(mu_);
    repeated_ops_graph_breaks_suspended_ = false;
    return std::exchange(repeated_ops_graph_break_requested_, false);
  }

  void InsertGraphBreak(std::vector<SharedDeviceBufferList> must_materialize,
                        MaterializationReason reason, GraphSplitMode mode,
                        EagerMode eager_mode) {
    absl::MutexLock lock(mu_);
    InsertGraphBreakLocked(std::move(must_materialize), reason, mode,
                           eager_mode,
                           /*execution_task=*/std::nullopt);
  }

  xla::Future<std::shared_ptr<EventSnapshot>> RequestEventSnapshot(
      std::vector<SharedDeviceBufferList> must_materialize,
      MaterializationReason reason, GraphSplitMode mode, EagerMode eager_mode) {
    absl::MutexLock lock(mu_);
    return RequestEventSnapshotLocked(std::move(must_materialize), reason, mode,
                                      eager_mode);
  }

  std::vector<ReadyWork> GetReadyWork() {
    absl::MutexLock lock(mu_);
    return std::move(ready_work_);
  }

  // Returns the core pinning mode of this events queue.
  CorePinningMode core_pinning_mode() const { return core_pinning_mode_; }
  // Sets the device pinning mode of this events queue.
  void SetCorePinningMode(CorePinningMode core_pinning_mode) {
    core_pinning_mode_ = core_pinning_mode;
  }

  void ResetRepeatedOpsHeuristicState() {
    absl::MutexLock lock(mu_);
    repetition_hashes_.clear();
    repetition_i_ = 0;
    repeated_ops_graph_break_requested_ = false;
    for (auto& deferred_op_event : deferred_ops_) {
      deferred_op_event.deferred_op_hash = 0;
    }
  }

  absl::StatusOr<DeviceBufferRef> ScheduleHostToDeviceTransfer(
      HostToDeviceTransfer transfer) {
    // Copy the transfer-blocking futures so that the transfer start can start
    // without holding the mutex.
    auto [error_map_promise, error_map_future] = xla::MakePromise<void>();
    std::vector<xla::Future<void>> transfer_blocks;
    {
      absl::MutexLock lock(mu_);
      PruneCompletedFutures(transfer_blocks_);
      transfer_blocks = transfer_blocks_;
      futures_.push_back(std::move(error_map_future));
    }

    absl::StatusOr<DeviceBufferRef> device_buffer_ref =
        transfer.RunAfter(transfer_blocks);
    if (!device_buffer_ref.ok()) {
      error_map_promise.Set(
          AdaptXlaError(device_buffer_ref.status(),
                        /*context=*/"failed to copy buffer from host to "
                                    "device"));
    } else {
      // Satisfy TorchTPU error formatting requirements.
      device_buffer_ref->GetReadyFuture().OnReady(
          [error_map_promise = std::move(error_map_promise)](
              const absl::Status& status) mutable {
            error_map_promise.Set(
                AdaptXlaError(status,
                              /*context=*/"failed to copy buffer from host to "
                                          "device"));
          });
    }
    return device_buffer_ref;
  }

  void ScheduleDeviceToHostTransfer(DeviceToHostTransfer transfer) {
    // Copy the transfer-blocking futures so that the transfer start can start
    // without holding the mutex.
    std::vector<xla::Future<void>> transfer_blocks;
    {
      absl::MutexLock lock(mu_);
      PruneCompletedFutures(transfer_blocks_);
      transfer_blocks = transfer_blocks_;
    }
    xla::Future<void> transfer_future = transfer.RunAfter(transfer_blocks);
    {
      absl::MutexLock lock(mu_);
      futures_.push_back(std::move(transfer_future));
    }
  }

  void InsertStreamWait(EventSnapshot& snapshot) {
    absl::MutexLock lock(mu_);
    xla::Future<void> snapshot_ok_callback = snapshot.OkCallback();
    // The next event will not be ready until the awaited snapshot is ready.
    futures_.push_back(snapshot_ok_callback);  // intentional copy

    // Any transfers on this stream must delay until the awaited snapshot is
    // ready.
    if (transfer_blocks_.size() + 1 > transfer_blocks_.capacity()) {
      PruneCompletedFutures(transfer_blocks_);
    }
    transfer_blocks_.push_back(std::move(snapshot_ok_callback));
  }

  void MarkActive(std::vector<xla::Future<void>>&& new_futures) {
    if (new_futures.empty()) {
      return;
    }
    absl::MutexLock lock(mu_);
    if (futures_.size() + new_futures.size() > futures_.capacity()) {
      PruneCompletedFutures(futures_);
    }
    futures_.insert(futures_.end(),
                    std::make_move_iterator(new_futures.begin()),
                    std::make_move_iterator(new_futures.end()));
  }

  void RecordCompiledExecutionTask(ExecutionTask&& execution_task,
                                   EagerMode eager_mode) {
    absl::MutexLock lock(mu_);
    (void)InsertGraphBreakLocked(
        /*must_materialize=*/{}, MaterializationReason::kCompileModeExecution,
        GraphSplitMode::kSplitGraph, eager_mode, std::move(execution_task));
  }

  void WriteBackMaterializationFutures(
      std::vector<xla::Future<void>>&& futures,
      std::optional<DeviceBufferRef> last_device_output) {
    // Destroy any replaced DeviceBufferRef outside `mu_` to avoid running
    // `~PjRtCApiBuffer()` while holding `StreamState::mu_` on the hot path.
    std::optional<DeviceBufferRef> old_device_output;
    absl::MutexLock lock(mu_);
    for (auto& ready_work_item : ready_work_) {
      if (auto* snapshot_request =
              std::get_if<SnapshotRequest>(&ready_work_item)) {
        auto& existing_futures = snapshot_request->futures;
        existing_futures.insert(existing_futures.end(),
                                std::make_move_iterator(futures.begin()),
                                std::make_move_iterator(futures.end()));
        if (last_device_output.has_value()) {
          old_device_output =
              std::exchange(snapshot_request->last_device_output,
                            std::move(last_device_output));
        }
        return;
      }
    }
    futures_.insert(futures_.end(), std::make_move_iterator(futures.begin()),
                    std::make_move_iterator(futures.end()));
    if (last_device_output.has_value()) {
      old_device_output =
          std::exchange(last_device_output_, std::move(last_device_output));
    }
  }

 private:
  // Result of CheckLastRepeatedSequence, representing the relationship of the
  // newest op to the prior loop sequence.
  enum class CheckLastRepeatedSequenceResult {
    // We've seen a partial prefix of the last repeated sequence, but not the
    // full sequence. Collect more ops to see if the sequence completes.
    kContinueWaiting,
    // We've seen another full loop of the last repeated sequence. Materialize
    // the single loop we just completed and reset to check the next repetition.
    kRepetitionComplete,
    // We definitely are not following the last repeated sequence. Reset and
    // scan for a new pattern.
    kPatternBreak,
  };

  // Helper function for RecordDeferredOpAndDetectRepetition.
  // Checks if the new hash is consistent with the last repeated sequence.
  [[nodiscard]] CheckLastRepeatedSequenceResult CheckLastRepeatedSequence(
      size_t new_hash) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Helper function for RecordDeferredOpAndDetectRepetition.
  // Performs a full scan to try to find a repeated sequence.
  // Returns the length of the sequence if found, or 0 if no sequence is found
  // within the min/max sequence length range.
  [[nodiscard]] int64_t FindRepeatedSequence(size_t new_hash)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Helper function for RecordDeferredOpAndDetectRepetition.
  // Records the repeated hash pattern to repetition_hashes_.
  void RecordRepeatedHashPattern(int64_t repetition_length)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Helper function for RecordDeferredOpAndDetectRepetition, called when the
  // newest op completes a repeated sequence.
  // Returns true if the caller should insert a graph break now. If graph breaks
  // are suspended, returns false and records the request for
  // ResumeRepeatedOpsGraphBreaks() instead.
  [[nodiscard]] bool RequestRepeatedOpsGraphBreakLocked()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (!repeated_ops_graph_breaks_suspended_) {
      return true;
    }
    repeated_ops_graph_break_requested_ = true;
    return false;
  }

  void InsertGraphBreakLocked(
      std::vector<SharedDeviceBufferList> must_materialize,
      MaterializationReason reason, GraphSplitMode mode, EagerMode eager_mode,
      std::optional<ExecutionTask> execution_task)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    // Freeze the graph state at the time of the snapshot request.
    // All (non-constant) deferred ops created since the last snapshot must be
    // executed; any which have a current live Tensor reference must be
    // materialized.
    std::vector<SharedDeviceBufferList> must_execute;
    for (const auto& deferred_op_event : deferred_ops_) {
      auto maybe_node = deferred_op_event.lock();
      if (!maybe_node) continue;
      if (maybe_node->live_data_ptrs() > 0) {
        must_materialize.push_back(maybe_node);
      }
      must_execute.push_back(std::move(maybe_node));
    }
    deferred_ops_.clear();
    // The queue is now empty; reset the repetition index to start scanning for
    // an existing or new pattern.
    repetition_i_ = 0;
    // This graph break satisfies any graph break that the repeated ops
    // heuristic requested while suspended, and ends the suspension, so that a
    // suspension that is never resumed (e.g. because the dispatch that started
    // it failed) cannot outlive the next graph break.
    repeated_ops_graph_breaks_suspended_ = false;
    repeated_ops_graph_break_requested_ = false;
    if (!must_execute.empty() || !must_materialize.empty()) {
      ready_work_.push_back(
          EagerExecution{std::move(must_execute), std::move(must_materialize),
                         core_pinning_mode_,
                         MaterializationSettings(reason, mode, eager_mode)});
    }
    if (execution_task.has_value()) {
      ready_work_.push_back(std::move(execution_task.value()));
    }
  }

  xla::Future<std::shared_ptr<EventSnapshot>> RequestEventSnapshotLocked(
      std::vector<SharedDeviceBufferList> must_materialize,
      MaterializationReason reason, GraphSplitMode mode, EagerMode eager_mode)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    InsertGraphBreakLocked(std::move(must_materialize), reason, mode,
                           eager_mode, std::nullopt);
    PruneCompletedFutures(futures_);

    // Explanation: synchronizing the last EventSnapshot should also await all
    // prior EventSnapshots. We achieve this by chaining a promise/future;
    // each event snapshot will have a future that will be set by the prior
    // event snapshot.
    auto [snapshot_promise, snapshot_future] =
        xla::MakePromise<std::shared_ptr<EventSnapshot>>();
    auto [chaining_promise, chaining_future] = xla::MakePromise<void>();
    SnapshotRequest new_snapshot_request{
        .futures = std::move(futures_),
        .last_device_output = std::exchange(last_device_output_, std::nullopt),
        .snapshot_promise = std::move(snapshot_promise),
        .ok_callback_promise = std::move(chaining_promise)};
    ready_work_.push_back(std::move(new_snapshot_request));
    futures_.push_back(std::move(chaining_future));
    return snapshot_future;
  }

  absl::Mutex mu_;
  std::vector<xla::Future<void>> futures_ ABSL_GUARDED_BY(mu_);
  std::optional<DeviceBufferRef> last_device_output_ ABSL_GUARDED_BY(mu_);
  std::vector<xla::Future<void>> transfer_blocks_ ABSL_GUARDED_BY(mu_);

  std::deque<DeferredOpEvent> deferred_ops_ ABSL_GUARDED_BY(mu_);
  std::vector<ReadyWork> ready_work_ ABSL_GUARDED_BY(mu_);

  // If a sequence repeats twice, we anticipate that it may repeat three or
  // more times.
  // We store the hashes of the repeated segment to efficiently identify these
  // 3rd and future repetitions without repeatedly rescanning.
  std::vector<size_t> repetition_hashes_ ABSL_GUARDED_BY(mu_);
  // The index of the next hash in repetition_hashes_ to check.
  int64_t repetition_i_ = 0;

  // Whether graph breaks requested by the repeated ops heuristic are suspended
  // (see SuspendRepeatedOpsGraphBreaks).
  bool repeated_ops_graph_breaks_suspended_ ABSL_GUARDED_BY(mu_) = false;
  // Whether the repeated ops heuristic requested a graph break while graph
  // breaks were suspended.
  bool repeated_ops_graph_break_requested_ ABSL_GUARDED_BY(mu_) = false;

  std::atomic<CorePinningMode> core_pinning_mode_ = CorePinningMode::kUnpinned;
};
StreamState::CheckLastRepeatedSequenceResult
StreamState::CheckLastRepeatedSequence(size_t new_hash)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  if (repetition_i_ >= repetition_hashes_.size()) {
    // This should only happen when repetition_hashes_ is empty and
    // repetition_i_ is 0, meaning that there is no known repeated sequence to
    // check.
    return StreamState::CheckLastRepeatedSequenceResult::kPatternBreak;
  }

  // There was a previously-identified repetition, check if we're consistent
  // with another loop of the same sequence.
  const auto expected_hash = repetition_hashes_[repetition_i_];
  if (new_hash != expected_hash) {
    // Pattern break; we were looking for a sequence like [A, B, C] but just
    // saw a pattern like [A, D], meaning that we need to recheck for a new
    // repeated pattern.
    repetition_i_ = 0;
    repetition_hashes_.clear();
    return StreamState::CheckLastRepeatedSequenceResult::kPatternBreak;
  }

  // Consistent with the repetition so far, see if we've completed it.
  repetition_i_++;
  if (repetition_i_ == repetition_hashes_.size()) {
    // We've just fully completed another repetition. For example we just
    // pushed hash [C] when the last seen repetition was [A, B, C].
    // Reset the loop counter, we only need to materialize the list
    // we just pushed.
    repetition_i_ = 0;
    return StreamState::CheckLastRepeatedSequenceResult::kRepetitionComplete;
  }
  // Keep waiting for more ops to see if we're still repeating the last-seen
  // loop.
  return StreamState::CheckLastRepeatedSequenceResult::kContinueWaiting;
}

int64_t StreamState::FindRepeatedSequence(size_t new_hash)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  auto last_deferred_op_it = deferred_ops_.rbegin();
  if (last_deferred_op_it == deferred_ops_.rend()) {
    // The queue is empty, nothing to detect.
    return 0;
  }
  // Initialize our iterator to be kMinRepeatedSubsequenceLength ops from the
  // end of the queue.
  auto prior_repetition_end_it = deferred_ops_.rbegin();
  int64_t repetition_length = 0;
  while (prior_repetition_end_it != deferred_ops_.rend() &&
         repetition_length < kMinRepeatedSubsequenceLength) {
    ++repetition_length;
    ++prior_repetition_end_it;
  }

  // Incrementally increase the sequence length until we find 2 loops at the
  // end of the deferred ops queue, or we exceed the maximum sequence length.
  while (prior_repetition_end_it != deferred_ops_.rend() &&
         repetition_length <= kMaxRepeatedSubsequenceLength) {
    // Look for points where we pushed the same hash value as new_hash.
    if (prior_repetition_end_it->deferred_op_hash != new_hash) {
      ++repetition_length;
      ++prior_repetition_end_it;
      continue;
    }
    // Initialize curr_it at the end of the queue, and prev_it
    // `repetition_length` from the end.
    auto prev_it = prior_repetition_end_it;  // intentional copy
    auto curr_it = last_deferred_op_it;      // intentional copy

    const int64_t required_matches =
        (kMinRepetitionCount - 1) * repetition_length;
    int64_t matches = 0;
    while (matches < required_matches && prev_it != deferred_ops_.rend()) {
      if (prev_it->deferred_op_hash != curr_it->deferred_op_hash) {
        // Sequence mismatch, the current `repetition_length` is not valid.
        break;
      }
      if (++matches == required_matches) {
        // Repetition found, stop looking.
        break;
      }
      ++prev_it;
      ++curr_it;
    }
    if (matches == required_matches) {
      return repetition_length;
    }
    ++repetition_length;
    ++prior_repetition_end_it;
  }
  return 0;
}

void StreamState::RecordRepeatedHashPattern(int64_t repetition_length)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  repetition_hashes_.resize(repetition_length);
  auto reverse_it = deferred_ops_.rbegin();
  for (int i = repetition_length - 1; i >= 0; --i) {
    repetition_hashes_[i] = reverse_it->deferred_op_hash;
    ++reverse_it;
  }
}

struct DeviceState {
  StreamState* absl_nonnull GetOrCreateStreamState(int64_t stream_id);

  std::vector<StreamState* absl_nonnull> GetAllStreamStates() {
    absl::MutexLock lock(mutex);
    std::vector<StreamState* absl_nonnull> including_default;
    including_default.reserve(stream_states.size() + 1);
    including_default.push_back(&default_stream_state);
    for (const auto& stream_state : stream_states) {
      including_default.push_back(stream_state.get());
    }
    return including_default;
  }

  void Clear() {
    // Keep the default stream but clear it so it is empty.
    default_stream_state.Clear();

    // Drop all other stream states.
    absl::MutexLock lock(mutex);
    for (auto& stream_state : stream_states) {
      stream_state->Clear();
    }
    stream_states.clear();
  }

  void ResetRepeatedOpsHeuristicState() {
    default_stream_state.ResetRepeatedOpsHeuristicState();
    absl::MutexLock lock(mutex);
    for (auto& stream_state : stream_states) {
      stream_state->ResetRepeatedOpsHeuristicState();
    }
  }

  StreamState default_stream_state;
  absl::Mutex mutex;
  // TODO: currently this vector only grows and never deallocates a StreamState
  // once created. This could cause host-side OOMs if the user creates new
  // streams in a loop.
  // Possible alternatives:
  //   - Allocate a fixed number of stream states and reuse them; this is the
  //     approach used by XPU. This may have unexpected behavior for users with
  //     many streams, as multiple Python streams will actually be the same C++
  //     stream.
  //   - Use a destructor on the TorchTPU specific torch.tpu.Stream object with
  //     a free-list to reuse streams. This will *not* work for the generic
  //     c10::Stream object as this has no destructor we can hook into, so
  //     torch.Stream(device="tpu") objects cannot be reused.
  std::vector<std::unique_ptr<StreamState>> stream_states
      ABSL_GUARDED_BY(mutex);
};

StreamState* absl_nonnull DeviceState::GetOrCreateStreamState(
    int64_t stream_id) {
  ABSL_CHECK(stream_id >= 0);  // CRASH_OK
  if (stream_id == 0) {
    return &default_stream_state;
  }
  auto stream_index = stream_id - 1;
  absl::MutexLock lock(mutex);
  while (stream_states.size() <= stream_index) {
    stream_states.push_back(std::make_unique<StreamState>());
  }
  return stream_states[stream_index].get();
}

struct StreamStates {
  DeviceState& GetDeviceState(c10::DeviceIndex device_index) {
    ABSL_CHECK(device_index >= 0 &&  // CRASH_OK
               device_index < kMaxTorchDevices);
    return device_states[device_index];
  }

  void Clear() {
    for (auto& device_state : device_states) {
      device_state.Clear();
    }
  }

  std::array<DeviceState, kMaxTorchDevices> device_states;
};

StreamStates& GetStreamStates() {
  static absl::NoDestructor<StreamStates> states;
  return *states;
}

StreamState* absl_nonnull GetOrCreateStreamState(c10::DeviceIndex device_index,
                                                 int64_t stream_id) {
  return GetStreamStates()
      .GetDeviceState(device_index)
      .GetOrCreateStreamState(stream_id);
}

std::vector<StreamState* absl_nonnull> GetDeviceStreamStates(
    c10::DeviceIndex device_index) {
  return GetStreamStates().GetDeviceState(device_index).GetAllStreamStates();
}

StreamState* absl_nonnull GetStreamFor(
    const SharedDeviceBufferList& device_buffer_list) {
  return GetOrCreateStreamState(device_buffer_list->device_index(),
                                device_buffer_list->stream_id());
}
}  // namespace

namespace internal {
namespace {
void MarkStreamActive(c10::DeviceIndex device_index, int64_t stream_id,
                      std::vector<xla::Future<void>>&& new_futures) {
  GetOrCreateStreamState(device_index, stream_id)
      ->MarkActive(std::move(new_futures));
}
}  // namespace

void MarkStreamActive(c10::DeviceIndex device_index, int64_t stream_id,
                      xla::Future<void> future) {
  std::vector<xla::Future<void>> futures;
  futures.reserve(1);
  futures.push_back(std::move(future));
  MarkStreamActive(device_index, stream_id, std::move(futures));
}

}  // namespace internal

void RecordDeferredOpCreated(const SharedDeviceBufferList& device_buffer_list) {
  if (GetEagerMode() == EagerMode::kInternalCompileFxGraph) {
    // In FX trace mode, nothing happens on device, so we don't add anything to
    // any stream.
    return;
  }
  GetStreamFor(device_buffer_list)->RecordDeferredOpCreated(device_buffer_list);
}

void RecordCompiledExecutionTask(ExecutionTask&& execution_task,
                                 c10::DeviceIndex device_index,
                                 c10::StreamId stream_id,
                                 EagerMode eager_mode) {
  ABSL_CHECK_NE(  // CRASH_OK=Must exit trace before executing
      eager_mode, EagerMode::kInternalCompileFxGraph);
  GetOrCreateStreamState(device_index, stream_id)
      ->RecordCompiledExecutionTask(std::move(execution_task), eager_mode);
}

std::vector<ReadyWork> GetReadyWork(c10::DeviceIndex device_index,
                                    c10::StreamId stream_id) {
  StreamState* stream_state = GetOrCreateStreamState(device_index, stream_id);
  // Take all the deferred ops and live nodes from the stream state.
  // This is the only part than needs to hold the lock.
  std::vector<ReadyWork> ready_work = stream_state->GetReadyWork();
  for (auto& ready_work_block : ready_work) {
    if (auto* snapshot_request =
            std::get_if<SnapshotRequest>(&ready_work_block)) {
      PruneCompletedFutures(snapshot_request->futures);
    }
  }
  return ready_work;
}

absl::StatusOr<DeviceBufferRef> ScheduleHostToDeviceTransfer(
    HostToDeviceTransfer host_to_device_transfer) {
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  return GetOrCreateStreamState(device_index, stream_id)
      ->ScheduleHostToDeviceTransfer(std::move(host_to_device_transfer));
}

void ScheduleDeviceToHostTransfer(
    DeviceToHostTransfer device_to_host_transfer) {
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  GetOrCreateStreamState(device_index, stream_id)
      ->ScheduleDeviceToHostTransfer(std::move(device_to_host_transfer));
}

void InsertStreamWait(c10::DeviceIndex device_index, c10::StreamId stream_id,
                      EventSnapshot& snapshot) {
  GetOrCreateStreamState(device_index, stream_id)->InsertStreamWait(snapshot);
}

std::vector<PendingEventSnapshot> RecordDeviceSnapshots(
    c10::DeviceIndex device_index,
    std::vector<SharedDeviceBufferList> must_materialize,
    MaterializationReason reason, GraphSplitMode mode) {
  ABSL_VLOG(1) << "RecordDeviceSnapshots: device="
               << static_cast<int>(device_index);
  auto eager_mode = GetEagerMode();
  if (eager_mode == EagerMode::kInternalCompileFxGraph) {
    return {};
  }
  auto device_streams = GetDeviceStreamStates(device_index);
  std::vector<PendingEventSnapshot> snapshots;
  snapshots.reserve(device_streams.size());
  for (c10::StreamId stream_id = 0; stream_id < device_streams.size();
       ++stream_id) {
    auto* stream_state = device_streams[stream_id];
    std::vector<SharedDeviceBufferList> stream_must_materialize;
    for (auto& node : must_materialize) {
      if (node != nullptr && node->device_index() == device_index &&
          node->stream_id() == stream_id) {
        stream_must_materialize.push_back(std::move(node));
      }
    }
    snapshots.push_back({
        .device_index = device_index,
        .stream_id = stream_id,
        .snapshot_future = stream_state->RequestEventSnapshot(
            std::move(stream_must_materialize), reason, mode, eager_mode),
    });
  }
  return snapshots;
}

xla::Future<std::shared_ptr<EventSnapshot>> EventSnapshot::Record(
    c10::DeviceIndex device_index, c10::StreamId stream_id,
    std::vector<SharedDeviceBufferList> must_materialize,
    MaterializationReason reason, GraphSplitMode mode) {
  auto eager_mode = GetEagerMode();
  if (eager_mode == EagerMode::kInternalCompileFxGraph) {
    return xla::Future<std::shared_ptr<EventSnapshot>>(
        TT_ERROR(error::kInvalidArgument)
        << "cannot record EventSnapshot during an FX trace");
  }
  return GetOrCreateStreamState(device_index, stream_id)
      ->RequestEventSnapshot(std::move(must_materialize), reason, mode,
                             eager_mode);
}

EventSnapshot::EventSnapshot(xla::Future<void> future,
                             xla::Promise<void> ok_callback_promise)
    : future_(std::move(future)), completion_(std::make_shared<Completion>()) {
  completion_->chain_promise = std::move(ok_callback_promise);
  auto callback = [completion = completion_](absl::Status status) {
    std::optional<xla::Promise<void>> chain_promise;
    {
      absl::MutexLock lock(completion->mu);
      completion->time = std::chrono::steady_clock::now();
      completion->completed = true;
      chain_promise = std::exchange(completion->chain_promise, std::nullopt);
    }
    if (chain_promise.has_value()) {
      ReleaseNextSnapshot(*std::move(chain_promise));
    }
  };
  if (future_.IsReady()) {
    // All prior work is already complete, so this new event is immediately
    // complete.
    callback(future_.Await());
  } else {
    // The completion time is recorded before the next snapshot is released, so
    // the next snapshot cannot complete before this one has a time.
    future_.OnReady(std::move(callback));
  }
}

absl::Status EventSnapshot::Synchronize() const {
  const absl::Status status = future_.Await();
  {
    // The callback on `future_` records the completion right after it.
    absl::MutexLock lock(completion_->mu,
                         absl::Condition(&completion_->completed));
  }
  // Surface the warnings rerouted from background threads.
  SurfaceReroutedWarnings();
  return status;
}

absl::StatusOr<bool> EventSnapshot::Query() const {
  return CompletionTime().has_value();
}

std::optional<std::chrono::steady_clock::time_point>
EventSnapshot::CompletionTime() const {
  absl::MutexLock lock(completion_->mu);
  if (!completion_->completed) {
    return std::nullopt;
  }
  return completion_->time;
}

absl::StatusOr<double> EventSnapshot::ElapsedTime(
    const EventSnapshot& end) const {
  const std::optional<std::chrono::steady_clock::time_point> start_time =
      CompletionTime();
  const std::optional<std::chrono::steady_clock::time_point> end_time =
      end.CompletionTime();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=c10::Event::elapsedTime() raises its
                 // own error unless both events have completed.
      start_time.has_value() && end_time.has_value(),
      error::kFailedPrecondition)
      << "both events must be completed before calculating elapsed time";
  TT_RETURN_IF_ERROR(future_.Await());
  TT_RETURN_IF_ERROR(end.future_.Await());
  return std::chrono::duration<double, std::milli>(*end_time - *start_time)
      .count();
}

xla::Future<void> EventSnapshot::OkCallback() {
  absl::MutexLock lock(completion_->mu);
  if (completion_->completed || !completion_->chain_promise.has_value()) {
    return absl::OkStatus();
  }
  return completion_->chain_promise->future();
}

void ClearAllStreams() {
  GetStreamStates().Clear();
  ResetStreamIdCounters();
}

void SetCorePinningMode(c10::DeviceIndex device_index, c10::StreamId stream_id,
                        CorePinningMode pinned) {
  GetOrCreateStreamState(device_index, stream_id)->SetCorePinningMode(pinned);
}

bool RecordDeferredOpAndDetectRepetition(
    const SharedDeviceBufferList& device_buffer_list) {
  if (GetEagerMode() == EagerMode::kInternalCompileFxGraph) {
    // In FX trace mode, nothing happens on device, so we don't add anything to
    // any stream and never materialize.
    return false;
  }
  return GetStreamFor(device_buffer_list)
      ->RecordDeferredOpAndDetectRepetition(device_buffer_list);
}

void SuspendRepeatedOpsGraphBreaks(
    const SharedDeviceBufferList& device_buffer_list) {
  GetStreamFor(device_buffer_list)->SuspendRepeatedOpsGraphBreaks();
}

bool ResumeRepeatedOpsGraphBreaks(c10::DeviceIndex device_index,
                                  c10::StreamId stream_id) {
  return GetOrCreateStreamState(device_index, stream_id)
      ->ResumeRepeatedOpsGraphBreaks();
}

void ResetRepeatedOpsHeuristicState() {
  for (auto& device_state : GetStreamStates().device_states) {
    device_state.ResetRepeatedOpsHeuristicState();
  }
}

CompilationMode GetCompilationMode(EagerMode eager_mode) {
  switch (eager_mode) {
    case EagerMode::kInternalDeferAll:
    case EagerMode::kDeferAndFuse:
    case EagerMode::kInternalCompileFxGraph:
      return CompilationMode::kFastRuntime;
    case EagerMode::kDeferNever:
    case EagerMode::kDeferNeverAndLaunchBlocking:
      return CompilationMode::kFastCompile;
  };
}

void InsertGraphBreak(c10::DeviceIndex device_index, c10::StreamId stream_id,
                      std::vector<SharedDeviceBufferList> must_materialize,
                      MaterializationReason reason, GraphSplitMode mode) {
  auto eager_mode = GetEagerMode();
  if (eager_mode == EagerMode::kInternalCompileFxGraph) {
    return;
  }
  GetOrCreateStreamState(device_index, stream_id)
      ->InsertGraphBreak(std::move(must_materialize), reason, mode, eager_mode);
}

void WriteBackMaterializationFutures(
    c10::DeviceIndex device_index, c10::StreamId stream_id,
    std::vector<xla::Future<void>>&& futures,
    std::optional<DeviceBufferRef> last_device_output) {
  GetOrCreateStreamState(device_index, stream_id)
      ->WriteBackMaterializationFutures(std::move(futures),
                                        std::move(last_device_output));
}

}  // namespace torch_tpu
