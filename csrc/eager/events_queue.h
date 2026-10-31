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

#ifndef TORCH_TPU_CSRC_EAGER_EVENTS_QUEUE_H_
#define TORCH_TPU_CSRC_EAGER_EVENTS_QUEUE_H_

#include <chrono>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "c10/core/Device.h"
#include "c10/core/Stream.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/execution_task.h"
#include "csrc/eager/split_traversal.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/pjrt/pjrt_transfer.h"
#include "xla/future.h"

namespace torch_tpu {

// Records on the events queue that a new deferred op has been created.
void RecordDeferredOpCreated(const SharedDeviceBufferList& device_buffer_list);

// Records on the events queue that a new deferred op has been created, and
// applies the repeated ops heuristic.
// Returns true if the stream should be materialized. While the heuristic's
// graph breaks are suspended on the stream (see SuspendRepeatedOpsGraphBreaks),
// returns false instead, and the stream remembers that a graph break was
// requested.
bool RecordDeferredOpAndDetectRepetition(
    const SharedDeviceBufferList& device_buffer_list);

// Suspends the graph breaks requested by the repeated ops heuristic on the
// stream of `device_buffer_list`, until ResumeRepeatedOpsGraphBreaks() is
// called for that stream or a graph break is inserted on it for any reason.
//
// This keeps the heuristic from splitting a sequence of deferred ops that must
// be compiled into the same graph, such as the ops of an in-place update that
// donates the destination tensor's buffer before the tensor's c10::DataPtr is
// updated to the result.
void SuspendRepeatedOpsGraphBreaks(
    const SharedDeviceBufferList& device_buffer_list);

// Resumes the graph breaks requested by the repeated ops heuristic on the given
// stream. Returns true if the heuristic requested a graph break while they were
// suspended, in which case the caller should insert one.
[[nodiscard]] bool ResumeRepeatedOpsGraphBreaks(c10::DeviceIndex device_index,
                                                c10::StreamId stream_id);

// Resets the state of the repeated ops heuristic for all streams, including
// any graph break it requested while suspended, but leaves all events in the
// queue.
void ResetRepeatedOpsHeuristicState();

// Clears all references to DeviceBufferLists from the events queue, forgets all
// prior stream futures, and resets to only a single default stream per device.
void ClearAllStreams();

// Tracks that a compiled mode execution task has been created.
void RecordCompiledExecutionTask(ExecutionTask&& execution_task,
                                 c10::DeviceIndex device_index,
                                 c10::StreamId stream_id, EagerMode eager_mode);

// Schedules a host-to-device transfer to be executed on the current stream as
// soon as it is valid to do so. This will wait for any prior blocking waits
// (from torch.Event.wait or torch.Stream.wait_event) and then start.
// If there are no prior blocking waits, this will start immediately.
// Returns the DeviceBufferRef that will hold the transferred data. The future
// for the transfer is recorded on the stream; the next-created Event on the
// current stream will wait for the transfer to complete and report any error.
absl::StatusOr<DeviceBufferRef> ScheduleHostToDeviceTransfer(
    HostToDeviceTransfer host_to_device_transfer);

// Schedules a device-to-host transfer to be executed on the current stream as
// soon as it is valid to do so. This will begin as soon as the device buffer
// has a ready PjRtBuffer and there are no prior blocking waits on the stream.
// If already materialized and there are no prior blocking waits, this will
// start immediately.
// The future for the transfer is recorded on the stream; the next-created Event
// on the current stream will wait for the transfer to complete and report any
// error.
void ScheduleDeviceToHostTransfer(DeviceToHostTransfer device_to_host_transfer);

// An EventSnapshot is a joined XLA future that represents the state of
// a stream at a particular point in time.
class EventSnapshot {
 public:
  explicit EventSnapshot(xla::Future<void> future,
                         xla::Promise<void> ok_callback_promise);

  static xla::Future<std::shared_ptr<EventSnapshot>> Record(
      c10::DeviceIndex device_index, c10::StreamId stream_id,
      std::vector<SharedDeviceBufferList> must_materialize = {},
      MaterializationReason reason = MaterializationReason::kUnknown,
      GraphSplitMode mode = GraphSplitMode::kSplitGraph);

  // Blocks the calling thread until the event snapshot is complete, then
  // surfaces the warnings rerouted from background threads. Must be called on a
  // dispatch thread.
  absl::Status Synchronize() const;

  // Query whether the event snapshot has completed.
  absl::StatusOr<bool> Query() const;

  // Returns the milliseconds from the completion of this event snapshot to the
  // completion of `end`, as observed by the host. Both snapshots must have
  // completed.
  absl::StatusOr<double> ElapsedTime(const EventSnapshot& end) const;

  // A future that is set to OkStatus when the event snapshot completes or
  // fails.
  xla::Future<void> OkCallback();

  // Directly returns the wrapped future.
  xla::Future<void> completion_future() const { return future_; }

 private:
  // The completion of an event snapshot, shared with the callback on its future
  // that records it.
  struct Completion {
    absl::Mutex mu;
    // Whether the snapshot has completed and `time` is set.
    bool completed ABSL_GUARDED_BY(mu) = false;
    // The host time at which the snapshot completed.
    std::chrono::steady_clock::time_point time ABSL_GUARDED_BY(mu);
    // The promise passed to OkCallback before the snapshot completed.
    std::optional<xla::Promise<void>> chain_promise ABSL_GUARDED_BY(mu);
  };

  // Returns the host time at which the snapshot completed, if it has.
  std::optional<std::chrono::steady_clock::time_point> CompletionTime() const;

  // The future backing the event snapshot.
  xla::Future<void> future_;
  absl_nonnull std::shared_ptr<Completion> completion_;
};

// Instructs the stream specified by device_index and stream_id to wait for the
// given EventSnapshot to complete before starting any subsequent work.
// TODO(bawilson): this currently only blocks transfers, not op executions;
// we handle op execution waits in the case of a single Python thread with the
// block in TpuDeviceGuardImpl::exchangeStream but this is not sound under
// multithreading.
void InsertStreamWait(c10::DeviceIndex device_index, c10::StreamId stream_id,
                      EventSnapshot& snapshot);

// A future that will resolve to an EventSnapshot during the materialization
// process (after PrepareSnapshotMaterializations has been called), matched with
// the device and stream that the snapshot is for.
struct PendingEventSnapshot {
  // The device and stream that the snapshot is for.
  c10::DeviceIndex device_index;
  c10::StreamId stream_id;
  // The future that will resolve to the EventSnapshot.
  xla::Future<std::shared_ptr<EventSnapshot>> snapshot_future;
};

// Records one PendingEventSnapshot for each stream on the given device.
std::vector<PendingEventSnapshot> RecordDeviceSnapshots(
    c10::DeviceIndex device_index,
    std::vector<SharedDeviceBufferList> must_materialize = {},
    MaterializationReason reason = MaterializationReason::kUnknown,
    GraphSplitMode mode = GraphSplitMode::kSplitGraph);

// Returns the CompilationMode for the given EagerMode.
CompilationMode GetCompilationMode(EagerMode eager_mode);

// Settings for how to compile a set of traversals to reach a snapshot.
// These are frozen at the time of the snapshot request, and used to dictate
// later behavior.
struct MaterializationSettings {
  MaterializationSettings(MaterializationReason reason,
                          GraphSplitMode graph_split_mode, EagerMode eager_mode)
      : graph_split_mode(graph_split_mode),
        reason(reason),
        compilation_mode(GetCompilationMode(eager_mode)),
        compilation_context(GetCompilationContext()) {}
  GraphSplitMode graph_split_mode;
  MaterializationReason reason;
  CompilationMode compilation_mode;
  // Immutable, shared set of pre-resolved CompilationSpecs for every
  // CompilationMode. Held by shared_ptr so that freezing the settings does not
  // deep-copy xla::CompileOptions, and so that callers can borrow a spec by
  // reference for as long as these settings are alive.
  absl_nonnull std::shared_ptr<const CompilationContext> compilation_context;
};

// A block of DeferredOp nodes that should be compiled and executed together.
struct EagerExecution {
  // The nodes that must be executed by the block.
  std::vector<SharedDeviceBufferList> must_execute;
  // The nodes that must be materialized by the block.
  std::vector<SharedDeviceBufferList> must_materialize;
  // The core that the must_execute nodes should be executed on.
  CorePinningMode core_pinning_mode = CorePinningMode::kUnpinned;
  // Settings for the materialization.
  MaterializationSettings settings;
};

using CompiledExecution = ExecutionTask;

// A request for an EventSnapshot. This should capture the futures for all
// prior operations on the stream.
struct SnapshotRequest {
  // Futures that must be completed before the snapshot is reached.
  std::vector<xla::Future<void>> futures;
  // The latest device-resident TensorCore eager output buffer on the stream
  // prior to this snapshot request, if any.
  std::optional<DeviceBufferRef> last_device_output;
  // When the execution for the traversals has started, an EventSnapshot will be
  // created and provided through this promise.
  xla::Promise<std::shared_ptr<EventSnapshot>> snapshot_promise;
  // When the snapshot is completed, this promise will be set to OkStatus
  // (even if the snapshot failed).
  xla::Promise<void> ok_callback_promise;
};

using ReadyWork =
    std::variant<EagerExecution, CompiledExecution, SnapshotRequest>;

// Returns a list of work items that are ready to be processed for the given
// device and stream.
std::vector<ReadyWork> GetReadyWork(c10::DeviceIndex device_index,
                                    c10::StreamId stream_id);

// Sets the core pinning mode for the given stream.
void SetCorePinningMode(c10::DeviceIndex device_index, c10::StreamId stream_id,
                        CorePinningMode pinned);

void InsertGraphBreak(c10::DeviceIndex device_index, c10::StreamId stream_id,
                      std::vector<SharedDeviceBufferList> must_materialize,
                      MaterializationReason reason, GraphSplitMode mode);

void WriteBackMaterializationFutures(
    c10::DeviceIndex device_index, c10::StreamId stream_id,
    std::vector<xla::Future<void>>&& futures,
    std::optional<DeviceBufferRef> last_device_output = std::nullopt);

namespace internal {

// Injects a future representing some conceptual "work" onto the stream.
// This is only used for testing and debugging purposes; actual work should be
// recorded using ScheduleHostToDeviceTransfer, RecordDeferredOpCreated, or
// ScheduleDeviceToHostTransfer as appropriate.
void MarkStreamActive(c10::DeviceIndex device_index, c10::StreamId stream_id,
                      xla::Future<void> future);

}  // namespace internal

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_EVENTS_QUEUE_H_
