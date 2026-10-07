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
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/Device.h"
#include "c10/core/Stream.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/context_states.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/execution_task.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/traversal.h"
#include "xla/future.h"

namespace torch_tpu {

// Records on the events queue that a new c10::DataPtr referencing the given
// DeviceBufferRef has been created.
// As long as at least one DataPtr exists, the DeviceBufferList cannot be freed,
// and if deferred, must be materialized on any global sync
// (torch.accelerator.synchronize()).
void RecordNewDataPtrCreated(const DeviceBufferRef& device_buffer_ref);

// Records on the events queue that a c10::DataPtr referencing the given
// DeviceBufferRef has been destroyed.
// Once all references to a DeviceBufferList have been destroyed, the
// DeviceBufferList can be freed, and does not need to be materialized.
void RecordDataPtrDestroyed(const DeviceBufferRef& device_buffer_ref);

// Records on the events queue that a new deferred op has been created.
void RecordDeferredOpCreated(const SharedDeviceBufferList& device_buffer_list);

// Records on the events queue that a new deferred op has been created, and
// applies the repeated ops heuristic.
// Returns true if the stream should be materialized.
bool RecordDeferredOpAndDetectRepetition(
    const SharedDeviceBufferList& device_buffer_list);

// Resets the state of the repeated ops heuristic for all streams, but leaves
// all events in the queue.
void ResetRepeatedOpsHeuristicState();

// Clears all references to DeviceBufferLists from the events queue, forgets all
// prior stream futures, and resets to only a single default stream per device.
void ClearAllStreams();

// Tracks that a compiled mode execution task has been created.
void RecordCompiledExecutionTask(ExecutionTask&& execution_task,
                                 c10::DeviceIndex device_index,
                                 c10::StreamId stream_id, EagerMode eager_mode);

// Tracks that an async host-to-device copy has been started that will populate
// the given DeviceBufferRef.
void RecordAsyncHostToDevice(const DeviceBufferRef& device_buffer_ref);

// Tracks that an async device-to-host copy has been started.
// The future should be the return value from a call to PjRtBuffer::ToLiteral.
void RecordAsyncDeviceToHost(xla::Future<void> to_literal_future);

enum class MaterializationMode { kFullGraph, kSplitGraph };

// An EventSnapshot is a joined XLA future that represents the state of
// a stream at a particular point in time.
class EventSnapshot {
 public:
  explicit EventSnapshot(xla::Future<void> future)
      : future_(future.Map([] { return std::chrono::steady_clock::now(); })) {}

  static xla::Future<std::shared_ptr<EventSnapshot>> Record(
      c10::DeviceIndex device_index, c10::StreamId stream_id,
      std::vector<SharedDeviceBufferList> must_materialize = {},
      MaterializationReason reason = MaterializationReason::kUnknown,
      MaterializationMode mode = MaterializationMode::kSplitGraph);

  // Blocks the calling thread until the event snapshot is complete.
  absl::Status Synchronize() const;

  // Query whether the event snapshot has completed.
  absl::StatusOr<bool> Query() const;

  // Returns the milliseconds from the completion of this event snapshot to the
  // completion of `end`, as observed by the host. Both snapshots must have
  // completed.
  absl::StatusOr<double> ElapsedTime(const EventSnapshot& end) const;

  // Sets the promise to OkStatus as soon as the event snapshot is complete
  // or fails. The error is not propagated to the promise.
  void OkCallback(xla::Promise<void>&& chain_promise);

 private:
  // The future backing the event snapshot. It resolves to the host time at
  // which the snapshot completed, so anything waiting on it, such as the
  // callback releasing the next snapshot, runs after the stamp exists.
  xla::Future<std::chrono::steady_clock::time_point> future_;
};

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
    MaterializationMode mode = MaterializationMode::kSplitGraph);

// Returns the CompilationMode for the given EagerMode.
CompilationMode GetCompilationMode(EagerMode eager_mode);

// Settings for how to compile a set of traversals to reach a snapshot.
// These are frozen at the time of the snapshot request, and used to dictate
// later behavior.
struct MaterializationSettings {
  MaterializationSettings(MaterializationReason reason,
                          MaterializationMode materialization_mode,
                          EagerMode eager_mode)
      : materialization_mode(materialization_mode),
        reason(reason),
        compilation_mode(GetCompilationMode(eager_mode)),
        compilation_context(GetCompilationContext()) {}
  MaterializationMode materialization_mode;
  MaterializationReason reason;
  CompilationMode compilation_mode;
  // Immutable, shared set of pre-resolved CompilationSpecs for every
  // CompilationMode. Held by shared_ptr so that freezing the settings does not
  // deep-copy xla::CompileOptions, and so that callers can borrow a spec by
  // reference for as long as these settings are alive.
  absl_nonnull std::shared_ptr<const CompilationContext> compilation_context;
};

// A snapshot to be materialized.
struct PendingSnapshotMaterialization {
  // The traversals that must be executed to materialize the snapshot.
  // This may be empty if there has been no eager mode work since the last
  // snapshot.
  std::vector<absl_nonnull std::unique_ptr<Traversal>> traversals;
  // If the snapshot includes a compiled mode execution task, it will be stored
  // here.
  std::optional<ExecutionTask> execution_task;
  // The nodes that were expected to be materialized by the traversals.
  std::vector<SharedDeviceBufferList> must_materialize;
  // Settings for the materialization.
  MaterializationSettings settings;
  // Futures that must be completed before the snapshot is reached.
  std::vector<xla::Future<void>> futures;
  // When the execution for the traversals has started, an EventSnapshot will be
  // created and provided through this promise.
  xla::Promise<std::shared_ptr<EventSnapshot>> snapshot_promise;
};

// Returns a sequence of Traversals that, if compiled and executed, would
// materialize all live tensors created on the given device and stream.
// These are grouped by the snapshot they are satisfying.
//
// This clears these nodes from the events queue.
std::vector<PendingSnapshotMaterialization> PrepareSnapshotMaterializations(
    c10::DeviceIndex device_index, c10::StreamId stream_id);

// Sets the core pinning mode for the given stream.
void SetCorePinningMode(c10::DeviceIndex device_index, c10::StreamId stream_id,
                        CorePinningMode pinned);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_EVENTS_QUEUE_H_
