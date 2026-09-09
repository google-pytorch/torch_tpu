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

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "c10/core/Stream.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/traversal.h"
#include "xla/future.h"

namespace torch_tpu {

// Records on the events queue that a new c10::DataPtr referencing the given
// DeviceBufferRef has been created.
// As long as at least one DataPtr exists, the DeviceBufferList cannot be freed,
// and if deferred, must be materialized on any global sync
// (torch.tpu.synchronize()).
void RecordNewDataPtrCreated(const DeviceBufferRef& device_buffer_ref);

// Records on the events queue that a c10::DataPtr referencing the given
// DeviceBufferRef has been destroyed.
// Once all references to a DeviceBufferList have been destroyed, the
// DeviceBufferList can be freed, and does not need to be materialized.
void RecordDataPtrDestroyed(const DeviceBufferRef& device_buffer_ref);

// Records on the events queue that a new deferred op has been created.
void RecordDeferredOpCreated(const SharedDeviceBufferList& device_buffer_list);

// Clears all references to DeviceBufferLists from the events queue, forgets all
// prior stream futures, and resets to only a single default stream per device.
void ClearAllStreams();

// Returns a sequence of Traversals that, if compiled and executed, would
// materialize all live tensors up to and including all nodes_to_materialize
// (even if any node to materialize is not live).
//
// This clears these nodes from the events queue.
absl::StatusOr<std::vector<absl_nonnull std::unique_ptr<Traversal>>>
PrepareMaterializationTraversals(
    absl::Span<const SharedDeviceBufferList> nodes_to_materialize);

// Returns a sequence of Traversals that, if compiled and executed, would
// materialize all live tensors created on the given device and stream.
//
// This clears these nodes from the events queue.
absl::StatusOr<std::vector<absl_nonnull std::unique_ptr<Traversal>>>
PrepareStreamTraversals(c10::DeviceIndex device_index, c10::StreamId stream_id);

// Returns a sequence of Traversals that, if compiled and executed, would
// materialize all live tensors created on the given device.
//
// This clears these nodes from the events queue.
absl::StatusOr<std::vector<absl_nonnull std::unique_ptr<Traversal>>>
PrepareDeviceTraversals(c10::DeviceIndex device_index);

// Tracks that a background materialization has been started for the given
// outputs on the current stream.
void RecordBackgroundMaterialization(absl::Span<const DeviceBufferRef> outputs);

// Tracks that an async host-to-device copy has been started that will populate
// the given DeviceBufferRef.
void RecordAsyncHostToDevice(const DeviceBufferRef& device_buffer_ref);

// Tracks that an async device-to-host copy has been started.
// The future should be the return value from a call to PjRtBuffer::ToLiteral.
void RecordAsyncDeviceToHost(xla::Future<void> to_literal_future);

// An EventSnapshot is a joined XLA future that represents the state of
// a stream at a particular point in time.
class EventSnapshot {
 public:
  // Records a fence over the async d2h copies already enqueued on the device
  // and stream.
  static std::shared_ptr<EventSnapshot> Record(c10::DeviceIndex device_index,
                                               c10::StreamId stream_id);

  // Wait for the event snapshot to complete.
  absl::Status Wait() const;

  // Query whether the event snapshot has completed.
  absl::StatusOr<bool> Query() const;

 private:
  // The future backing the event snapshot.
  // Private; must use Record() so that the snapshot is tracked on the stream.
  explicit EventSnapshot(xla::Future<void> future)
      : future_(std::move(future)) {}

  friend std::vector<std::shared_ptr<EventSnapshot>> RecordDeviceSnapshots(
      c10::DeviceIndex device_index);

  xla::Future<void> future_;
};

// Records one EventSnapshot for each stream on the given device.
std::vector<std::shared_ptr<EventSnapshot>> RecordDeviceSnapshots(
    c10::DeviceIndex device_index);

// Sets the core pinning mode for the given stream.
void SetCorePinningMode(c10::DeviceIndex device_index, c10::StreamId stream_id,
                        CorePinningMode pinned);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_EVENTS_QUEUE_H_
