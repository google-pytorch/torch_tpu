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

#ifndef TORCH_TPU_CSRC_EAGER_MATERIALIZE_H_
#define TORCH_TPU_CSRC_EAGER_MATERIALIZE_H_

#include <memory>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "c10/core/Stream.h"
#include "csrc/common/compilation.h"
#include "csrc/common/shape.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/structured_log_buffer.h"
#include "xla/future.h"

// When an aten op is dispatched, we always create a DeviceBufferList to contain
// the result. This DeviceBufferList will contain a DeferredOp, which describes
// the MLIR necessary to compute the return value of the aten function.
//
// At some point in the future, we may need the actual data for one or more
// tensors/buffers in the DeviceBufferList; this process is called
// materialization, and the behavior depends on the previous state.
//   - If the DeviceBufferList was alread fully materialized, no-op.
//   - If the DeviceBufferList has a DeferredOp, then that DeferredOp will be
//     compiled (or retrieved from the cache) and executed, and the
//     DeviceBufferList will be updated to hold the results.
//   - Compiled mode placeholder buffers cannot be materialized, as they
//     represent data that does not exist during compilation. It is an error to
//     call Materialize during the compilation phase of torch.compile.

namespace torch_tpu {

// Controls how long a materialization operation will block the calling thread.
enum class MaterializationBlockMode {
  // The calling thread will block until the materialization graph has been
  // created and enqueued, but not until it is completed.
  // This means that the calling thread and the background thread will not run
  // concurrently, which can be used to avoid some race conditions, or improve
  // error reporting.
  kBlockUntilStarted,
  // The calling thread will return immediately after scheduling the
  // materialization task. This means that materialization targets will not
  // immediately be set to the materializing state, and may resolve to either
  // a materialized result or an error at a later time.
  kNonBlocking,
};

// Executes all deferred nodes in the list, and updates them in place.  No-op
// for nodes that are already materialized.
// Optional parameter mode can be set to false to avoid splitting the op graph
// traversal and, hence, compile it as a whole. Returns an error if any of the
// nodes are placeholder buffers, or if any leaf input in the traversed graph is
// a placeholder.
// Optional parameter block_mode controls if the materialization will block the
// calling thread, and if so, for how long.
absl::Status Materialize(
    absl::Span<const SharedDeviceBufferList> nodes,
    MaterializationReason reason,
    MaterializationMode mode = MaterializationMode::kSplitGraph,
    MaterializationBlockMode block_mode =
        MaterializationBlockMode::kBlockUntilStarted);

// Materializes all of the buffers and connected leaf nodes in-place.
// This materializes all DeviceBufferLists referenced which are not already
// materialized; no-op for all materialized DeviceBufferLists.
// Errors if any of the buffers are placeholders.
absl::Status Materialize(
    absl::Span<const DeviceBufferRef> buffer_refs, MaterializationReason reason,
    MaterializationMode mode = MaterializationMode::kSplitGraph,
    MaterializationBlockMode block_mode =
        MaterializationBlockMode::kBlockUntilStarted);

// Materializes the given node in-place. Delegates to the span overload.
inline absl::Status Materialize(
    const SharedDeviceBufferList& node, MaterializationReason reason,
    MaterializationMode mode = MaterializationMode::kSplitGraph,
    MaterializationBlockMode block_mode =
        MaterializationBlockMode::kBlockUntilStarted) {
  return Materialize(absl::Span<const SharedDeviceBufferList>(&node, 1), reason,
                     mode, block_mode);
}

// Materializes the given buffer in-place. Delegates to the span overload.
inline absl::Status Materialize(
    const DeviceBufferRef& buffer_ref, MaterializationReason reason,
    MaterializationMode mode = MaterializationMode::kSplitGraph,
    MaterializationBlockMode block_mode =
        MaterializationBlockMode::kBlockUntilStarted) {
  return Materialize(absl::Span<const DeviceBufferRef>(&buffer_ref, 1), reason,
                     mode, block_mode);
}

// Enqueues an executable for materialization.
// This will create one new DeviceBufferList for each output tensor, with the
// same shapes and dtypes as the output_shapes argument.
// These outputs will be materialized, i.e. will hold PjRtBuffers, but these
// PjRtBuffers will not be ready until the executable reaches the front of the
// materialization queue, is Executed, and the execution completes.
absl::StatusOr<std::vector<DeviceBufferRef>> EnqueueExecutable(
    SharedLoadedExecutableWithMetadata executable,
    std::vector<DeviceBufferRef> arguments,
    absl::Span<const Shape> output_shapes, std::string_view task_name = "");

// Shuts down the materialization worker and joins its threads.
void ShutDownMaterializationState();

// Materializes all live tensors on the given stream.
// This is an async operation; after the live tensor state has been evaluated,
// and all work on the stream has been enqueued for materialization, the
// snapshot is returned and can be queried or awaited to determine when the
// materialization is complete.
xla::Future<std::shared_ptr<EventSnapshot>> MaterializeStream(
    c10::DeviceIndex device_index, c10::StreamId stream_id,
    MaterializationReason reason,
    MaterializationMode mode = MaterializationMode::kSplitGraph,
    absl::Span<const SharedDeviceBufferList> must_materialize = {});

// Materializes all live tensors on the current stream.
// This is an async operation; after the live tensor state has been evaluated,
// and all work on the stream has been enqueued for materialization, the
// snapshot is returned and can be queried or awaited to determine when the
// materialization is complete.
xla::Future<std::shared_ptr<EventSnapshot>> MaterializeCurrentStream(
    MaterializationReason reason,
    MaterializationMode mode = MaterializationMode::kSplitGraph,
    absl::Span<const SharedDeviceBufferList> must_materialize = {});

// Materializes all live tensors on the given device.
// This is an async operation; after the live tensor state has been evaluated,
// and all work on the device has been enqueued for materialization, one
// snapshot is returned for each stream on the device, and each stream can be
// queried or awaited to determine when the materialization is complete.
std::vector<xla::Future<std::shared_ptr<EventSnapshot>>> MaterializeDevice(
    c10::DeviceIndex device_index, MaterializationReason reason,
    MaterializationMode mode = MaterializationMode::kSplitGraph,
    absl::Span<const SharedDeviceBufferList> must_materialize = {});

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_MATERIALIZE_H_
