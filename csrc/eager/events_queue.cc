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
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/absl_vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "c10/core/Stream.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/context_states.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/eager/current_stream.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/materialize_common.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/traversal.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
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

class StreamState {
 public:
  // Records on the events queue that a new c10::DataPtr referencing the given
  // DeviceBufferRef has been created.
  void RecordNewDataPtrCreated(const DeviceBufferRef& device_buffer_ref) {
    // Placeholders and constant tensors are never inserted into the map.
    if (device_buffer_ref.is_placeholder() || device_buffer_ref.is_constant()) {
      return;
    }
    absl::MutexLock lock(mu_);
    if (!device_buffer_ref.is_materialized()) {
      // Insert or increment the count for the DeviceBufferList.
      live_nodes_[device_buffer_ref.device_buffer_list()]++;
    } else {
      // Once the ref has a ready PjRtBuffer, we can stop tracking it.
      live_nodes_.erase(device_buffer_ref.device_buffer_list());
    }
  }

  // Records on the events queue that a c10::DataPtr referencing the given
  // DeviceBufferRef has been destroyed.
  void RecordDataPtrDestroyed(const DeviceBufferRef& device_buffer_ref) {
    // Placeholders and constant tensors are never inserted into the map.
    if (device_buffer_ref.is_placeholder() || device_buffer_ref.is_constant()) {
      return;
    }
    absl::MutexLock lock(mu_);
    // If the ref is already removed from the map, we do nothing. This can
    // happen if the ref became ready after insertion, or if the queue was
    // cleared.
    auto it = live_nodes_.find(device_buffer_ref.device_buffer_list());
    if (it == live_nodes_.end()) {
      return;
    }
    // Once the ref is materialized, or the count drops to zero, we can remove
    // it from the map.
    if (device_buffer_ref.is_materialized() || --it->second <= 0) {
      live_nodes_.erase(it);
    }
  }

  // Clears all tracked DeviceBufferLists from the events queue, and resets the
  // repeated ops heuristic state.
  void Clear() {
    absl::MutexLock lock(mu_);
    live_nodes_.clear();
    repetition_hashes_.clear();
    repetition_i_ = 0;
    for (auto& event : events_queue_) {
      if (auto* deferred_op_event = std::get_if<DeferredOpEvent>(&event)) {
        deferred_op_event->deferred_op_hash = 0;
      }
    }
    futures_.clear();

    // Drop all unresolved SnapshotEvents and fail their promises.
    absl::Status cancel_status = TT_ERROR(error::kCancelled)
                                 << "stream state was cleared";
    size_t n = events_queue_.size();
    for (size_t i = 0; i < n; ++i) {
      auto event = std::move(events_queue_.front());
      events_queue_.pop_front();
      if (auto* snapshot_event =
              std::get_if<std::unique_ptr<EventSnapshotRequest>>(&event)) {
        (*snapshot_event)->snapshot_promise.Set(cancel_status);
      } else {
        events_queue_.push_back(std::move(event));
      }
    }

    // At this point, there are no "live nodes" holding onto deferred ops, so
    // all ops that have no side effects (directly or indirectly) should be
    // expired.
    std::erase_if(events_queue_, [](const StreamEvent& event) {
      const auto* deferred_op_event = std::get_if<DeferredOpEvent>(&event);
      return deferred_op_event == nullptr ||
             (deferred_op_event->side_effects == nullptr &&
              deferred_op_event->no_side_effects.expired());
    });
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
        // prevent later reexecution.
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

  // A request to create an EventSnapshot at this point in the stream.
  struct EventSnapshotRequest {
    // Futures that must complete before the snapshot is reached.
    std::vector<xla::Future<void>> futures;
    // Nodes that explicitly must be materialized by the snapshot.
    std::vector<SharedDeviceBufferList> must_materialize;
    // Settings for the materialization. These are frozen when the request is
    // made, so that context states are only read from the dispatch thread.
    MaterializationSettings materialization_settings;
    // Live nodes that must be materialized by the snapshot if deferred.
    absl::flat_hash_map<SharedDeviceBufferList, int64_t> live_nodes;
    // If the snapshot includes a compiled mode execution task, it will be
    // stored here. This is always the last task in the snapshot.
    std::optional<ExecutionTask> execution_task;
    // The promise which can be used to provide the snapshot once is it
    // defined.
    xla::Promise<std::shared_ptr<EventSnapshot>> snapshot_promise;
  };

  // The stream collects a series of DeferredOpEvents in order, with
  // requests for device snapshots inserted to create graph breaks and
  // wait points.
  using StreamEvent =
      std::variant<DeferredOpEvent,
                   absl_nonnull std::unique_ptr<EventSnapshotRequest>>;

  void RecordDeferredOpCreated(
      const SharedDeviceBufferList& device_buffer_list) {
    if (auto event = DeferredOpEvent::FromDeferredOp(device_buffer_list);
        event.has_value()) {
      absl::MutexLock lock(mu_);
      events_queue_.push_back(std::move(*event));
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
    events_queue_.push_back(std::move(event));

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
        return true;
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
    return true;
  }

  xla::Future<std::shared_ptr<EventSnapshot>> RequestEventSnapshot(
      std::vector<SharedDeviceBufferList> must_materialize,
      MaterializationReason reason, MaterializationMode mode,
      EagerMode eager_mode) {
    absl::MutexLock lock(mu_);
    return RequestEventSnapshotLocked(std::move(must_materialize), reason, mode,
                                      eager_mode);
  }

  std::deque<StreamEvent> GetSnapshotEvents() {
    absl::MutexLock lock(mu_);
    std::deque<StreamEvent> temp;
    while (!events_queue_.empty() &&
           !std::holds_alternative<std::unique_ptr<EventSnapshotRequest>>(
               events_queue_.back())) {
      temp.push_front(std::move(events_queue_.back()));
      events_queue_.pop_back();
    }

    std::swap(temp, events_queue_);
    if (ABSL_VLOG_IS_ON(1)) {
      if (events_queue_.empty()) {
        ABSL_VLOG(1) << "[StreamState::GetSnapshotEvents] Returning all "
                     << temp.size() << " StreamEvents.";
      } else if (temp.empty()) {
        ABSL_VLOG(1) << "[StreamState::GetSnapshotEvents] No SnapshotEvents in "
                        "a queue of "
                     << events_queue_.size() << " StreamEvents.";
      } else {
        ABSL_VLOG(1) << "[StreamState::GetSnapshotEvents] Returning "
                     << temp.size() << " StreamEvents and retaining "
                     << events_queue_.size() << " DeferredOpEvents.";
      }
    }
    return temp;
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
    for (auto& event : events_queue_) {
      if (auto* deferred_op_event = std::get_if<DeferredOpEvent>(&event)) {
        deferred_op_event->deferred_op_hash = 0;
      }
    }
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
    RequestEventSnapshotLocked(
        /*must_materialize=*/{}, MaterializationReason::kCompileModeExecution,
        MaterializationMode::kSplitGraph, eager_mode);
    EventSnapshotRequest& request =
        *std::get<std::unique_ptr<EventSnapshotRequest>>(events_queue_.back());
    request.execution_task = std::move(execution_task);
  }

  xla::Future<void> JoinFutures() {
    absl::MutexLock lock(mu_);
    if (!futures_.empty()) {
      PruneCompletedFutures(futures_);
    }
    if (futures_.empty()) {
      return xla::Future<void>(absl::OkStatus());
    }
    auto join_future = xla::JoinFutures(futures_);
    futures_.clear();
    futures_.push_back(join_future);  // intentional copy
    return join_future;
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

  // Common helper for RequestEventSnapshot and RecordCompiledExecutionTask.
  xla::Future<std::shared_ptr<EventSnapshot>> RequestEventSnapshotLocked(
      std::vector<SharedDeviceBufferList> must_materialize,
      MaterializationReason reason, MaterializationMode mode,
      EagerMode eager_mode) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto [snapshot_promise, snapshot_future] =
        xla::MakePromise<std::shared_ptr<EventSnapshot>>();
    // Explanation: synchronizing the last EventSnapshot should also await all
    // prior EventSnapshots. We achieve this by chaining a promise/future;
    // each event snapshot will have a future that will be set by the prior
    // event snapshot.
    auto [chaining_promise, chaining_future] = xla::MakePromise<void>();
    snapshot_future.OnReady([chaining_promise = std::move(chaining_promise)](
                                absl::StatusOr<std::shared_ptr<EventSnapshot>>
                                    snapshot_or) mutable {
      if (!snapshot_or.ok()) {
        chaining_promise.Set(absl::OkStatus());
      } else {
        (*snapshot_or)->OkCallback(std::move(chaining_promise));
      }
    });
    events_queue_.push_back(std::make_unique<EventSnapshotRequest>(
        std::move(futures_), std::move(must_materialize),
        MaterializationSettings(reason, mode, eager_mode),
        std::move(live_nodes_), std::nullopt, std::move(snapshot_promise)));
    futures_.push_back(std::move(chaining_future));
    return snapshot_future;
  }

  absl::Mutex mu_;
  std::vector<xla::Future<void>> futures_ ABSL_GUARDED_BY(mu_);

  // Hold a strong pointer to the DeviceBufferList as the key; as long as there
  // is a live DataPtr, the DeviceBufferList can't be dropped.
  absl::flat_hash_map<SharedDeviceBufferList, int64_t> live_nodes_
      ABSL_GUARDED_BY(mu_);

  std::deque<StreamEvent> events_queue_ ABSL_GUARDED_BY(mu_);

  // If a sequence repeats twice, we anticipate that it may repeat three or
  // more times.
  // We store the hashes of the repeated segment to efficiently identify these
  // 3rd and future repetitions without repeatedly rescanning.
  std::vector<size_t> repetition_hashes_ ABSL_GUARDED_BY(mu_);
  // The index of the next hash in repetition_hashes_ to check.
  int64_t repetition_i_ = 0;

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
  // Initialize our iterator to be kMinRepeatedSubsequenceLength ops from the
  // end of the queue.
  auto prior_repetition_end_it = events_queue_.rbegin();
  while (prior_repetition_end_it != events_queue_.rend() &&
         !std::holds_alternative<DeferredOpEvent>(*prior_repetition_end_it)) {
    ++prior_repetition_end_it;
  }
  auto last_deferred_op_it = prior_repetition_end_it;  // intentional copy

  int64_t repetition_length = 0;
  while (prior_repetition_end_it != events_queue_.rend() &&
         repetition_length < kMinRepeatedSubsequenceLength) {
    if (std::holds_alternative<DeferredOpEvent>(*prior_repetition_end_it)) {
      ++repetition_length;
    }
    ++prior_repetition_end_it;
  }
  if (prior_repetition_end_it == events_queue_.rend()) {
    // There are less than kMinRepetitionCount * kMinRepeatedSubsequenceLength
    // deferred ops in the queue.
    return 0;
  }

  // Incrementally increase the sequence length until we find 2 loops at the
  // end of the deferred ops queue, or we exceed the maximum sequence length.
  while (prior_repetition_end_it != events_queue_.rend() &&
         repetition_length <= kMaxRepeatedSubsequenceLength) {
    const auto* deferred_op_event =
        std::get_if<DeferredOpEvent>(&*prior_repetition_end_it);
    if (deferred_op_event == nullptr) {
      ++prior_repetition_end_it;
      continue;
    }
    // Look for points where we pushed the same hash value as new_hash.
    if (deferred_op_event->deferred_op_hash != new_hash) {
      ++repetition_length;
      ++prior_repetition_end_it;
      continue;
    }
    // Initialize curr_it at the end of the queue, and prev_it
    // `repetition_length` from the end.
    auto prev_it = prior_repetition_end_it;  // intentional copy
    auto curr_it = last_deferred_op_it;      // intentional copy

    const int64_t required_matches = kMinRepetitionCount * repetition_length;
    int64_t matches = 0;
    while (matches < required_matches && prev_it != events_queue_.rend()) {
      const auto* prev_deferred_op_event =
          std::get_if<DeferredOpEvent>(&*prev_it);
      if (prev_deferred_op_event == nullptr) {
        ++prev_it;
        continue;
      }
      const auto* curr_deferred_op_event =
          std::get_if<DeferredOpEvent>(&*curr_it);
      if (curr_deferred_op_event == nullptr) {
        ++curr_it;
        continue;
      }
      if (prev_deferred_op_event->deferred_op_hash !=
          curr_deferred_op_event->deferred_op_hash) {
        break;
      }
      if (++matches == required_matches) {
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
  auto reverse_it = events_queue_.rbegin();
  for (int i = repetition_length - 1; i >= 0; --i) {
    const auto* deferred_op_event = std::get_if<DeferredOpEvent>(&*reverse_it);
    while (deferred_op_event == nullptr) {
      ++reverse_it;
      deferred_op_event = std::get_if<DeferredOpEvent>(&*reverse_it);
    }
    repetition_hashes_[i] = deferred_op_event->deferred_op_hash;
    ++reverse_it;
  }
}

// The usage of a node within an execution region.
enum class OpUsage {
  // This node is an output of the materialization.
  kOutput,
  // This node is an input to an output.
  kUsed,
  // This node is unused, and may need to be added as an output to ensure
  // it is executed.
  kUnused,
};

using DefinedNodeMap = absl::flat_hash_map<const DeviceBufferList*, OpUsage>;

// Returns the OpUsage of the given device buffer list, switching between
// kOutput (if it's a required output) or kUnused (if it's not).
OpUsage GetOpUsage(
    const DeferredOp& deferred_op,
    const SharedDeviceBufferList& device_buffer_list,
    const absl::flat_hash_map<SharedDeviceBufferList, int64_t>& live_nodes,
    const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        nodes_to_materialize_set) {
  // Explicitly materialized nodes are always outputs, even if they normally
  // wouldn't be (e.g. constants)
  if (nodes_to_materialize_set.contains(device_buffer_list.get())) {
    ABSL_VLOG(3)
        << "[ProcessDeferredOpEvent] Adding explicitly materialized buffer "
        << device_buffer_list.get() << " as output";
    return OpUsage::kOutput;
  }

  if (live_nodes.contains(device_buffer_list.get())) {
    ABSL_VLOG(3) << "[ProcessDeferredOpEvent] Adding live buffer "
                 << device_buffer_list.get() << " as output";
    return OpUsage::kOutput;
  }

  bool has_dynamic_dimensions = false;
  for (int i = 0; i < device_buffer_list->size(); ++i) {
    if (!device_buffer_list->dynamic_dimensions(i).empty()) {
      has_dynamic_dimensions = true;
      break;
    }
  }
  if (has_dynamic_dimensions) {
    // Nodes with dynamic dimensions must be materialized to resolve them
    // to static shapes.
    ABSL_VLOG(3) << "[ProcessDeferredOpEvent] Adding dynamic-dimension buffer "
                 << device_buffer_list.get() << " as output";
    return OpUsage::kOutput;
  }

  // Non-output nodes need to be used by at least one output to ensure they get
  // executed.
  ABSL_VLOG(3) << "[ProcessDeferredOpEvent] Adding non-output buffer "
               << device_buffer_list.get() << " to execution order";
  return OpUsage::kUnused;
}

// Helper function for PrepareMaterializationTraversals.
// Pushes the device buffer list to the execution order (unless it is empty),
// and updates the defined node map.
void ProcessDeferredOpEvent(
    SharedDeviceBufferList&& device_buffer_list, const DeferredOp& deferred_op,
    const absl::flat_hash_map<SharedDeviceBufferList, int64_t>& live_nodes,
    const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        nodes_to_materialize_set,
    std::vector<SharedDeviceBufferList>& execution_order,
    DefinedNodeMap& defined_node_map) {
  // Mark all inputs as used, and insert any newly-used empty tensors.
  for (const auto& input : deferred_op.inputs()) {
    if (!input.is_deferred()) continue;

    const DeviceBufferList* input_device_buffer_list =
        input.device_buffer_list().get();

    if (auto input_def_it = defined_node_map.find(input_device_buffer_list);
        input_def_it != defined_node_map.end()) {
      // Update the usage of the input tensor from unused -> used, but leave
      // outputs as outputs.
      switch (input_def_it->second) {
        case OpUsage::kUnused:
          input_def_it->second = OpUsage::kUsed;
          break;
        case OpUsage::kUsed:
        case OpUsage::kOutput:
          break;
      }
    } else if (input.is_constant() &&
               defined_node_map
                   .try_emplace(input_device_buffer_list, OpUsage::kUsed)
                   .second) {
      // The first time a constant tensor is read by a later op, we insert
      // it into the execution order, but not as an output as we don't want to
      // materialize it unless the user explicitly asks for it.
      ABSL_VLOG(3) << "[ProcessDeferredOpEvent] Constant buffer "
                   << input_device_buffer_list << " is read by "
                   << device_buffer_list.get() << "("
                   << ToString(deferred_op.op_name())
                   << ").\nInserting into execution order.";
      execution_order.push_back(input.device_buffer_list());
    }
  }

  defined_node_map[device_buffer_list.get()] = GetOpUsage(
      deferred_op, device_buffer_list, live_nodes, nodes_to_materialize_set);

  execution_order.push_back(std::move(device_buffer_list));
}

// Helper function for PrepareMaterializationTraversals.
// Takes an execution_order, a map defining the usage of each node, and returns
// a traversal (unless the execution order and output nodes are empty).
absl::StatusOr<absl_nullable std::unique_ptr<Traversal>> FinishTraversal(
    std::vector<SharedDeviceBufferList>& execution_order,
    DefinedNodeMap& defined_node_map,
    std::vector<SharedDeviceBufferList>& output_nodes,
    CorePinningMode core_pinning_mode) {
  if (execution_order.empty()) {
    ABSL_CHECK(defined_node_map.empty())  // CRASH_OK
        << "defined node map is not empty when execution order is empty. These "
           "should always have the same size";
    return nullptr;
  }

  // Build the output nodes vector, maintaining the execution order.
  output_nodes.clear();
  for (const auto& execution_node : execution_order) {
    auto it = defined_node_map.find(execution_node.get());
    ABSL_CHECK(it != defined_node_map.end())  // CRASH_OK
        << "node in execution order was not found in defined node map";
    switch (it->second) {
      case OpUsage::kOutput:
        output_nodes.push_back(execution_node);
        break;
      case OpUsage::kUnused:
        // Unused nodes are added as outputs to ensure they get executed.
        ABSL_VLOG(3) << "[FinishTraversal] Promoting unused buffer "
                     << execution_node.get() << " as output";
        output_nodes.push_back(execution_node);
        break;
      case OpUsage::kUsed:
        break;
    }
  }
  absl_nullable std::unique_ptr<Traversal> traversal = nullptr;
  if (!output_nodes.empty()) {
    TT_ASSIGN_OR_RETURN(traversal, Traversal::CreateFromExecutionOrder(
                                       execution_order, output_nodes));
  }
  execution_order.clear();
  defined_node_map.clear();
  output_nodes.clear();
  if (core_pinning_mode != CorePinningMode::kUnpinned) {
    traversal->SetCorePinningMode(core_pinning_mode);
  }

  return traversal;
}

absl::StatusOr<std::vector<absl_nonnull std::unique_ptr<Traversal>>>
PrepareTraversals(
    absl::Span<const StreamState::DeferredOpEvent> deferred_op_events,
    const absl::flat_hash_map<SharedDeviceBufferList, int64_t>& live_nodes,
    absl::Span<const SharedDeviceBufferList> nodes_to_materialize,
    const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        nodes_to_materialize_set,
    const CorePinningMode core_pinning_mode) {
  std::vector<absl_nonnull std::unique_ptr<Traversal>> traversals;

  // Partition the deferred ops queue into separate traversals with these rules:
  //   - If an op is SplitBefore, it must be first in its execution order.
  //   - If an op is SplitAfter, it must be last in its execution order, and
  //     must be an output.
  //   - If a node has live c10::DataPtrs, it must be an output.
  //   - If a node is in nodes_to_materialize, it must be an output.
  //   - Every non-constant op in execution_order must be executed by at least
  //     one output.
  //   - Constant and empty ops are added immediately before the first read, or
  //     at the end if they are explicitly materialized but not read.
  // Reuse working memory for efficiency.
  std::vector<SharedDeviceBufferList> execution_order;
  std::vector<SharedDeviceBufferList> output_nodes;
  DefinedNodeMap defined_node_map;

  for (const auto& event : deferred_op_events) {
    // Filter to only non-expired, deferred nodes.
    auto device_buffer_list = event.lock();
    if (!device_buffer_list) continue;
    const auto deferred_op = device_buffer_list->deferred_op();
    if (!deferred_op) continue;

    const auto split_mode = deferred_op->split_mode();

    if (IsSplitBefore(split_mode)) {
      TT_ASSIGN_OR_RETURN(auto maybe_traversal,
                          FinishTraversal(execution_order, defined_node_map,
                                          output_nodes, core_pinning_mode));
      if (maybe_traversal != nullptr) {
        ABSL_VLOG(2) << "[PrepareTraversals] Split out "
                     << maybe_traversal->execution_order().size()
                     << " nodes before SplitBefore buffer "
                     << device_buffer_list.get() << "("
                     << ToString(deferred_op->op_name()) << ").";
        traversals.push_back(std::move(maybe_traversal));
      }
    }

    ProcessDeferredOpEvent(std::move(device_buffer_list), *deferred_op,
                           live_nodes, nodes_to_materialize_set,
                           execution_order, defined_node_map);

    if (IsSplitAfter(split_mode)) {
      TT_ASSIGN_OR_RETURN(auto maybe_traversal,
                          FinishTraversal(execution_order, defined_node_map,
                                          output_nodes, core_pinning_mode));
      if (maybe_traversal != nullptr) {
        ABSL_VLOG(2) << "[PrepareTraversals] Split out "
                     << maybe_traversal->execution_order().size()
                     << " nodes after SplitAfter buffer "
                     << device_buffer_list.get() << "("
                     << ToString(deferred_op->op_name()) << ").";
        traversals.push_back(std::move(maybe_traversal));
      }
    }
  }

  // We never materialize a constant (or empty) tensor unless it is explicitly
  // requested by the user. If that does happen, then we append these constant
  // tensors to the last Traversal to make sure they have defined buffers.
  // This is necessary for downstream uses that require physical data buffers,
  // such as torch.compile invocations.
  for (const auto& node : nodes_to_materialize) {
    if (node->is_constant() && node->is_deferred()) {
      bool inserted =
          defined_node_map.insert_or_assign(node.get(), OpUsage::kOutput)
              .second;
      if (inserted) {
        ABSL_VLOG(3) << "[PrepareTraversals] constant buffer " << node.get()
                     << " is an explicit output. Appending to final traversal.";
        execution_order.push_back(node);
      }
    }
  }

  TT_ASSIGN_OR_RETURN(auto maybe_traversal,
                      FinishTraversal(execution_order, defined_node_map,
                                      output_nodes, core_pinning_mode));
  if (maybe_traversal != nullptr) {
    ABSL_VLOG(2) << "[PrepareTraversals] Final traversal has size "
                 << maybe_traversal->execution_order().size()
                 << ", ending with node "
                 << maybe_traversal->execution_order().back().get();
    traversals.push_back(std::move(maybe_traversal));
  }
  ABSL_VLOG(1) << "[PrepareTraversals] Created " << traversals.size()
               << " traversals to materialize " << nodes_to_materialize.size()
               << " nodes.";
  return traversals;
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

void MarkStreamActive(c10::DeviceIndex device_index, int64_t stream_id,
                      std::vector<xla::Future<void>>&& new_futures) {
  GetOrCreateStreamState(device_index, stream_id)
      ->MarkActive(std::move(new_futures));
}

void MarkStreamActive(c10::DeviceIndex device_index, int64_t stream_id,
                      xla::Future<void> future) {
  std::vector<xla::Future<void>> futures;
  futures.reserve(1);
  futures.push_back(std::move(future));
  MarkStreamActive(device_index, stream_id, std::move(futures));
}

StreamState* absl_nonnull GetStreamFor(
    const SharedDeviceBufferList& device_buffer_list) {
  return GetOrCreateStreamState(device_buffer_list->device_index(),
                                device_buffer_list->stream_id());
}

StreamState* absl_nonnull GetStreamFor(
    const DeviceBufferRef& device_buffer_ref) {
  return GetOrCreateStreamState(device_buffer_ref.device_index(),
                                device_buffer_ref.stream_id());
}

}  // namespace

void RecordNewDataPtrCreated(const DeviceBufferRef& device_buffer_ref) {
  if (GetEagerMode() == EagerMode::kInternalCompileFxGraph) {
    // In FX trace mode, nothing happens on device, so we don't add anything to
    // any stream.
    return;
  }
  GetStreamFor(device_buffer_ref)->RecordNewDataPtrCreated(device_buffer_ref);
}

void RecordDataPtrDestroyed(const DeviceBufferRef& device_buffer_ref) {
  if (GetEagerMode() == EagerMode::kInternalCompileFxGraph) {
    // In FX trace mode, nothing happens on device, so we don't add anything to
    // any stream.
    return;
  }
  GetStreamFor(device_buffer_ref)->RecordDataPtrDestroyed(device_buffer_ref);
}

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
                                 const EagerMode eager_mode) {
  ABSL_CHECK_NE(  // CRASH_OK=Must exit trace before executing
      eager_mode, EagerMode::kInternalCompileFxGraph);
  GetOrCreateStreamState(device_index, stream_id)
      ->RecordCompiledExecutionTask(std::move(execution_task), eager_mode);
}

std::vector<PendingSnapshotMaterialization> PrepareSnapshotMaterializations(
    c10::DeviceIndex device_index, c10::StreamId stream_id) {
  std::vector<PendingSnapshotMaterialization> snapshots_to_materialize;
  StreamState* stream_state = GetOrCreateStreamState(device_index, stream_id);
  // Take all the deferred ops and live nodes from the stream state.
  // This is the only part than needs to hold the lock.
  std::deque<StreamState::StreamEvent> stream_events =
      stream_state->GetSnapshotEvents();
  std::vector<StreamState::DeferredOpEvent> deferred_ops;
  absl::flat_hash_set<const DeviceBufferList*> must_materialize_node_set;

  while (!stream_events.empty()) {
    auto& event = stream_events.front();
    if (auto* deferred_op = std::get_if<StreamState::DeferredOpEvent>(&event)) {
      deferred_ops.push_back(std::move(*deferred_op));
    } else if (auto* snapshot_event_ptr = std::get_if<
                   std::unique_ptr<StreamState::EventSnapshotRequest>>(
                   &event)) {
      auto& snapshot_event = **snapshot_event_ptr;
      must_materialize_node_set.clear();
      for (const auto& node : snapshot_event.must_materialize) {
        must_materialize_node_set.insert(node.get());
      }

      auto snapshot_traversals_or = PrepareTraversals(
          deferred_ops, snapshot_event.live_nodes,
          snapshot_event.must_materialize, must_materialize_node_set,
          stream_state->core_pinning_mode());

      if (snapshot_traversals_or.ok()) {
        PruneCompletedFutures(snapshot_event.futures);
        snapshots_to_materialize.push_back(
            {.traversals = std::move(*snapshot_traversals_or),
             .execution_task = std::move(snapshot_event.execution_task),
             .must_materialize = std::move(snapshot_event.must_materialize),
             .settings = std::move(snapshot_event.materialization_settings),
             .futures = std::move(snapshot_event.futures),
             .snapshot_promise = std::move(snapshot_event.snapshot_promise)});
      } else {
        // If one snapshot fails, later snapshots may succeed only if there are
        // no dependencies on the failed DeferredOpEvents.
        // Mark all the failed ops as failed so that errors propagate and
        // unblock any threads waiting on the failed ops.
        for (const auto& deferred_op_event : deferred_ops) {
          if (auto node = deferred_op_event.lock();
              node != nullptr && !node->is_constant()) {
            node->SetAsError(snapshot_traversals_or.status());
          }
        }
        for (const auto& node : snapshot_event.must_materialize) {
          if (!node->is_materialized()) {
            node->SetAsError(snapshot_traversals_or.status());
          }
        }
        snapshot_event.snapshot_promise.Set(snapshot_traversals_or.status());
      }
      deferred_ops.clear();
    }
    stream_events.pop_front();
  }
  return snapshots_to_materialize;
}

void RecordAsyncHostToDevice(const DeviceBufferRef& device_buffer_ref) {
  ABSL_CHECK_NE(  // CRASH_OK=no device transfers during FX trace
      GetEagerMode(), EagerMode::kInternalCompileFxGraph);
  MarkStreamActive(device_buffer_ref.device_index(),
                   device_buffer_ref.stream_id(),
                   device_buffer_ref.GetReadyFuture());
}

void RecordAsyncDeviceToHost(xla::Future<void> to_literal_future) {
  ABSL_CHECK_NE(  // CRASH_OK=no device transfers during FX trace
      GetEagerMode(), EagerMode::kInternalCompileFxGraph);
  // TODO: should this be moved into the function signature?
  // Is there ever a case where we would want to record an async d2h on a
  // different stream than the current stream?
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  MarkStreamActive(device_index, stream_id, std::move(to_literal_future));
}

std::vector<PendingEventSnapshot> RecordDeviceSnapshots(
    c10::DeviceIndex device_index,
    std::vector<SharedDeviceBufferList> must_materialize,
    MaterializationReason reason, MaterializationMode mode) {
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
    MaterializationReason reason, MaterializationMode mode) {
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

absl::Status EventSnapshot::Synchronize() const { return future_.Await(); }

absl::StatusOr<bool> EventSnapshot::Query() const { return future_.IsReady(); }

void EventSnapshot::OkCallback(xla::Promise<void>&& chain_promise) {
  future_.OnReady(
      [chain_promise = std::move(chain_promise)](absl::Status status) mutable {
        chain_promise.Set(absl::OkStatus());
      });
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

}  // namespace torch_tpu
