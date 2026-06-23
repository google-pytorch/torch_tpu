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

#ifndef TORCH_TPU_EAGER_EVENTS_QUEUE_H_
#define TORCH_TPU_EAGER_EVENTS_QUEUE_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "c10/core/Stream.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/traversal.h"
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

// Clears all references to DeviceBufferLists from all streams.
void ClearAllStreams();

// Returns a sequence of Traversals that, if compiled and executed, would
// materialize all live tensors up to and including all nodes_to_materialize
// (even if any node to materialize is not live).
//
// This clears these nodes from the events queue.
absl::StatusOr<std::vector<absl_nonnull std::unique_ptr<Traversal>>>
PrepareMaterializationTraversals(
    absl::Span<const SharedDeviceBufferList> nodes_to_materialize);

// Tracks that an async host-to-device copy has been started that will populate
// the given DeviceBufferRef.
void RecordAsyncHostToDevice(const DeviceBufferRef& device_buffer_ref);

// Tracks that an async device-to-host copy has been started.
// The future should be the return value from a call to PjRtBuffer::ToLiteral.
void RecordAsyncDeviceToHost(xla::Future<void> to_literal_future);

// An EventSnapshot is a future that represents that the device catching up to a
// specific point in a stream of work as defined in the user's program.
// When a snapshot is ready (Query() == true), this means that:
//   - All prior snapshots on the stream are also ready
//   - All async transfers (both device-to-host and host-to-device) that started
//     before the snapshot have completed
//   - All tensors on the stream are fully materialized
class EventSnapshot {
 public:
  explicit EventSnapshot(
      c10::DeviceIndex device_index, int64_t stream_id,
      xla::Future<void> async_device_to_host_future,
      absl_nullable std::shared_ptr<DeviceBufferList> dummy_list = nullptr)
      : device_index_(device_index),
        stream_id_(stream_id),
        async_device_to_host_future_(std::move(async_device_to_host_future)),
        dummy_list_(std::move(dummy_list)) {}

  // A snapshot that is ready to be executed.
  struct RecordedEventSnapshot {
    // The subgraphs that should be executed to reach this snapshot, in order.
    // This may be empty if there have been no deferred operations on the stream
    // since the last snapshot.
    std::vector<absl_nonnull std::unique_ptr<Traversal>> traversals;
    // The snapshot itself.
    absl_nonnull std::shared_ptr<EventSnapshot> event_snapshot;
  };

  // Creates a snapshot barrier on the specified stream.
  static absl::StatusOr<RecordedEventSnapshot> Record(
      c10::DeviceIndex device_index, int64_t stream_id);

  // Wait for the event snapshot to complete.
  absl::Status Wait() const;

  // Query whether the event snapshot has completed.
  absl::StatusOr<bool> Query() const;

  // A future representing the status of all async device-to-host transfers
  // that started before this snapshot.
  // Async host-to-device transfers are instead tracked through the inputs to
  // the dummy_list_ to avoid an unnecessary host-side wait.
  xla::Future<void> async_device_to_host_future() const {
    return async_device_to_host_future_;
  }

  // A dummy, zero-size DeviceBufferList that must reach the Materialized state
  // before this event is considered ready.
  // This encodes the device-side timing dependency on async device-to-host
  // transfers as well as prior executions (both prior event snapshots, and
  // background materializations started after the last snapshot but before this
  // snapshot was recorded).
  absl_nullable std::shared_ptr<DeviceBufferList> dummy_list() const {
    return dummy_list_;
  }

 private:
  const c10::DeviceIndex device_index_;
  const c10::StreamId stream_id_;

  // A future representing the status of all async device-to-host transfers
  // that started before this snapshot.
  // Because completing this event has no device-side effects, we cannot await
  // it through only device-side operations; we need to insert a host-side
  // wait through an xla::Future.
  // Async host-to-device transfers do have a visible device-side effect, so
  // they are tracked through the dummy_list_ to minimize device-host
  // synchronization.
  const xla::Future<void> async_device_to_host_future_;

  // A dummy, zero-size DeviceBufferList that must reach the Materialized state
  // before this event is considered ready.
  // This encodes the device-side timing dependency on async device-to-host
  // transfers as well as prior executions (both prior event snapshots, and
  // background materializations started after the last snapshot but before this
  // snapshot was recorded).
  absl_nullable std::shared_ptr<DeviceBufferList> dummy_list_;
};

// Records one event snapshot for each active stream on the device.
absl::StatusOr<std::vector<EventSnapshot::RecordedEventSnapshot>>
RecordDeviceSnapshot(c10::DeviceIndex device_index);

// Records on event snapshot for each active stream on all accessible devices.
absl::StatusOr<std::vector<EventSnapshot::RecordedEventSnapshot>>
RecordGlobalSnapshot();

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_EVENTS_QUEUE_H_
