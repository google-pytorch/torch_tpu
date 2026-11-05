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

#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
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
#include "csrc/eager/traversal.h"
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
  // EventSnapshots must be constructed with a device and stream index.
  EventSnapshot() = delete;

  // Constructs an EventSnapshot for the given device and stream.
  // The snapshot is not yet started.
  explicit EventSnapshot(c10::DeviceIndex device_index, c10::StreamId stream_id)
      : device_index_(device_index),
        stream_id_(stream_id),
        state_(std::make_shared<State>()) {}

  // EventSnapshots are cheaply copyable and movable.
  EventSnapshot(const EventSnapshot& other) = default;
  EventSnapshot& operator=(const EventSnapshot& other) = default;
  EventSnapshot(EventSnapshot&& other) = default;
  EventSnapshot& operator=(EventSnapshot&& other) = default;

  // Records an EventSnapshot for the given device and stream.
  // The snapshot is not yet started.
  static absl::StatusOr<EventSnapshot> Record(
      c10::DeviceIndex device_index, c10::StreamId stream_id,
      std::vector<SharedDeviceBufferList> must_materialize = {},
      MaterializationReason reason = MaterializationReason::kUnknown,
      GraphSplitMode mode = GraphSplitMode::kSplitGraph);

  // Starts the event snapshot, providing the future that will be used to
  // signal completion. This must only be called once; usually by the
  // materialization worker thread (unless the stream is cleared).
  void Start(xla::Future<void> future) { state_->Start(std::move(future)); }

  // Blocks the calling thread until the event snapshot has been started by
  // the materialization worker thread.
  absl::Status BlockUntilStarted() const {
    return state_->started_future().Await();
  }

  // Blocks the calling thread until the event snapshot is complete, then
  // surfaces the warnings rerouted from background threads. Must be called on a
  // dispatch thread.
  absl::Status Synchronize() const;

  // Query whether the event snapshot has completed.
  absl::StatusOr<bool> Query() const;

  // Returns the time at which this event snapshot completed, or the error if
  // it failed.
  absl::StatusOr<std::chrono::steady_clock::time_point> CompletionTime() const {
    return state_->completion_time_future().Await();
  }

  // A future that is set to OkStatus when the event snapshot completes or
  // fails.
  xla::Future<void> OkCallback() const { return state_->ok_callback_future(); }

  // Directly returns the wrapped future.
  xla::Future<void> completion_future() const {
    return state_->completion_future();
  }

  c10::DeviceIndex device_index() const { return device_index_; }
  c10::StreamId stream_id() const { return stream_id_; }

  bool operator==(const EventSnapshot& other) const;

 private:
  // EventSnapshot::State is a lock-free state management class that tracks the
  // state of an EventSnapshot.
  class State {
   public:
    // Default constructor; initializes as not-yet-started.
    State();

    // Starts the event snapshot, providing the future that will be used to
    // signal completion. This must only be called once; usually by the
    // materialization worker thread (unless the stream is cleared first).
    void Start(xla::Future<void> future);

    // A future that will be completed when the event snapshot has started.
    // This is valid to access before Start() is called.
    xla::Future<void> started_future() const { return started_future_; }

    // A future that will be completed with the time at which the event
    // snapshot completed. This is valid to access before Start() is called;
    // it will always be completed with either the completion time or an error.
    xla::Future<std::chrono::steady_clock::time_point> completion_time_future()
        const {
      return completion_time_future_;
    }

    // A future that will be completed with OkStatus when the event snapshot
    // completes or fails. This is used to chain event snapshots to ensure later
    // snapshots do not finish before earlier ones.
    // This is valid to access before Start() is called.
    xla::Future<void> ok_callback_future() const { return ok_callback_future_; }

    // A future that will be completed when the event snapshot has completed.
    // This is **NOT VALID** to access before Start() is called; synchronization
    // and querying should first check started_future() before accessing this.
    xla::Future<void> completion_future() const { return completion_future_; }

   private:
    // Promises that must be set during the lifecycle of the event snapshot.
    struct Promises {
      // The promise backing BlockUntilStarted().
      xla::Promise<void> start;
      // The promise to supply a completion time through.
      xla::Promise<std::chrono::steady_clock::time_point> completion_time;
      // The promise backing OkCallback().
      xla::Promise<void> ok_callback;
    };

    // Before Start() is called, has_promises_ is true; the first caller to
    // Start() sets it to false and extract the promises. Second or later
    // callers to Start() will see false and crash.
    std::atomic_bool has_promises_ = false;
    std::optional<Promises> promises_;

    // Always-valid futures that are initialized in the constructor.
    xla::Future<void> started_future_;
    xla::Future<std::chrono::steady_clock::time_point> completion_time_future_;
    xla::Future<void> ok_callback_future_;

    // This is initialized with only an error message; it will be set to an
    // actual future by Start().
    xla::Future<void> completion_future_;
  };

  // The device and stream that this event snapshot is associated with.
  c10::DeviceIndex device_index_;
  c10::StreamId stream_id_;

  // An EventSnapshot is cheaply-copyable; all copies share the same underlying
  // state.
  absl_nonnull std::shared_ptr<State> state_;
};

// Instructs the stream specified by device_index and stream_id to wait for the
// given EventSnapshot to complete before starting any subsequent work.
// TODO(bawilson): this currently only blocks transfers, not op executions;
// we handle op execution waits in the case of a single Python thread with the
// block in TpuDeviceGuardImpl::exchangeStream but this is not sound under
// multithreading.
void InsertStreamWait(c10::DeviceIndex device_index, c10::StreamId stream_id,
                      EventSnapshot& snapshot);

// Records one EventSnapshot for each stream on the given device.
std::vector<EventSnapshot> RecordDeviceSnapshots(
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
  // When the execution for the traversals has started, the materialization
  // worker should call Start() on this snapshot.
  EventSnapshot snapshot_to_start;
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

// Indicates that the current stream of unexecuted DeferredOps should be split
// here and queued for async eager background materialization.
// Returns an error if the current execution mode is kInternalCompileFxGraph.
absl::Status InsertGraphBreak(
    c10::DeviceIndex device_index, c10::StreamId stream_id,
    std::vector<SharedDeviceBufferList> must_materialize,
    MaterializationReason reason, GraphSplitMode mode);

// Records that the current thread reserves the given stream to capture a graph
// on it, and owns the capture: other threads cannot use the stream (see
// IsStreamCapturedByAnotherThread()) until EndStreamGraphCapture(). The stream
// reports that it is capturing (see IsStreamCapturing()) once the capture
// begins (see ActivateStreamGraphCapture()). Returns false, and records
// nothing, if a graph capture is already in progress on the stream, if the
// device index is invalid, or if `stream_id` is not the ID of a non-default
// stream: like on CUDA, the default stream cannot be captured.
[[nodiscard]] bool TryReserveStreamGraphCapture(c10::DeviceIndex device_index,
                                                c10::StreamId stream_id);

// Records that the graph capture that the current thread reserved the given
// stream for begins: the capture records the operations of the current thread
// on the stream until EndStreamGraphCapture().
void ActivateStreamGraphCapture(c10::DeviceIndex device_index,
                                c10::StreamId stream_id);

// Records that the graph capture in progress on the given stream, if any, has
// ended.
void EndStreamGraphCapture(c10::DeviceIndex device_index,
                           c10::StreamId stream_id);

// Returns whether a graph capture is in progress on the given stream, as
// `c10::Stream::is_capturing()` reports, including a capture that an error
// invalidated (see InvalidateStreamGraphCapture()). Returns false for an
// invalid device index or stream ID. Unlike the functions that record work on a
// stream, this never creates the state of the stream.
[[nodiscard]] bool IsStreamCapturing(c10::DeviceIndex device_index,
                                     c10::StreamId stream_id);

// Returns whether the current thread owns the graph capture in progress on the
// given stream. Never creates the state of the stream.
[[nodiscard]] bool IsStreamCapturedByCurrentThread(
    c10::DeviceIndex device_index, c10::StreamId stream_id);

// Returns whether a thread other than the current one reserved the given stream
// for a graph capture, or is capturing a graph on it. Graph capture records the
// operations of the capturing thread only, so the operations of other threads
// on that stream would neither run nor be captured. Never creates the state of
// the stream.
[[nodiscard]] bool IsStreamCapturedByAnotherThread(
    c10::DeviceIndex device_index, c10::StreamId stream_id);

// Returns the error for running an operation on a stream that another thread
// captures a graph on.
absl::Status StreamCapturedByAnotherThreadError();

// Records that the thread that owns the graph capture in progress on the given
// stream exits, if it is the current thread. Never creates the state of the
// stream.
void MarkStreamGraphCaptureOwnerExited(c10::DeviceIndex device_index,
                                       c10::StreamId stream_id);

// Returns whether the thread that owns the graph capture in progress on the
// given stream exited. Never creates the state of the stream.
[[nodiscard]] bool HasStreamGraphCaptureOwnerExited(
    c10::DeviceIndex device_index, c10::StreamId stream_id);

// Invalidates the graph capture in progress on the given stream, if any:
// records `error` as the reason why the capture failed, unless an earlier error
// did. Like a CUDA graph capture after an operation that is not permitted
// during capture, the capture continues, and the stream reports that it is
// capturing, until EndStreamGraphCapture(); ending the capture then fails.
void InvalidateStreamGraphCapture(c10::DeviceIndex device_index,
                                  c10::StreamId stream_id, absl::Status error);

// Returns the error that invalidated the graph capture in progress on the given
// stream, or OK if no error invalidated it or no capture is in progress.
absl::Status GetStreamGraphCaptureError(c10::DeviceIndex device_index,
                                        c10::StreamId stream_id);

// Extracts the deferred ops recorded on the given stream since the last
// synchronization, and builds the Traversals that produce `must_materialize`
// and the side-effecting ops, with `GraphSplitMode::kFullGraph`, without
// enqueueing them for materialization. Used by `torch.accelerator.Graph`.
std::vector<absl_nonnull std::unique_ptr<Traversal>> ExtractCapturedTraversals(
    c10::DeviceIndex device_index, c10::StreamId stream_id,
    absl::Span<const SharedDeviceBufferList> must_materialize);

// Returns the error for using the contents of a tensor that a graph capture in
// progress read or wrote: until the capture ends, they are only known to the
// captured graph.
absl::Status TensorAccessedByGraphCaptureError();

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
