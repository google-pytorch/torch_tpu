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
#ifndef TORCH_TPU__INTERNAL_SYNC_SYNC_H_
#define TORCH_TPU__INTERNAL_SYNC_SYNC_H_

#include <string>

#include "ATen/core/TensorBody.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch_tpu/eager/device_buffer.h"

namespace torch_tpu {

// Materializes the given tensors, and then waits for their PjRtBuffers to be
// ready.
//
// This is a middle ground between Materialize() and CopyTpuToCpu():
//   Materialize(): forces a graph break and execution, but doesn't wait for the
//     PjRtBuffers to be ready.
//   SynchronizeTensors(): forces a graph break and waits for the PjRtBuffers
//     to be ready, but doesn't copy the data to the CPU.
//   CopyTpuToCpu(): forces a graph break, waits for the PjRtBuffers to be
//     ready, and then copies their data to the CPU.
//
// This is primarily useful for profiling; forcing a synchronization barrier
// allows for measuring the compile and execution time, without also including
// the data transfer time.
//
// If any of the tensors are views, then this function will materialize the
// base buffer, and then materialize and wait for the view buffer.
// However, because views are ephemeral, future calls to .cpu() on the tensors
// will re-execute the view logic (using the cached compilation).
absl::Status SynchronizeTensors(absl::Span<const at::Tensor> tensors);

// Whether to wait for execution to complete.
enum class WaitOnExecution {
  // Only ensures that all deferred operations have been scheduled for
  // execution on the device.
  kNo,
  // Blocks the calling thread until all deferred operations have completed
  // execution on the device.
  kYes,
};

// Materializes all deferred operations across all subgraphs and optionally
// waits for their execution to complete on the device.
//
// This function is thread-safe and can be called from multiple threads
// concurrently. It snapshots the current state of all subgraphs at the time of
// the call; any operations dispatched to the execution engine after this
// function begins may not be included in the synchronization barrier.
absl::Status SynchronizeAll(WaitOnExecution wait = WaitOnExecution::kNo);

// Checks if the tensor is materialized; that is, if it has a PjRtBuffer
// associated with it.
//
// Note that the PjRtBuffer is itself an asynchronous
// future, so just because a tensor is materialized doesn't mean it's ready
// for a host-to-device transfer.
//
// Additionally, if the tensor is a view, then this check only reflects the
// materialization status of the base buffer. The view will always be
// re-materialized for a device-to-host transfer.
absl::StatusOr<bool> IsMaterialized(const at::Tensor& tensor);

// Checks the tensor is both materialized and ready, meaning that it both has
// a PjRtBuffer and that PjRtBuffer has completed execution.
//
// If the tensor is a view, then this check only reflects the materialization
// status of the base buffer. The view will always be re-materialized for a
// device-to-host transfer.
absl::StatusOr<bool> IsReady(const at::Tensor& tensor);

// Returns a graphviz compatible representation of the computation graph of the
// given buffer refs. This traverses backwards from the given buffers through
// the graph of Aten operations that produced them.
//
// `buffer_refs` contains the buffers that are sinks of the computation graph.
// `buffer_ref_to_vars` can optionally contain a map from buffer refs to
// variable names. If any buffer in `buffer_ref_to_vars` is encountered during
// traversal, it will be labeled with the corresponding variable name in the
// graphviz output.
//
// This function will return an error if:
//  - `buffer_refs` is empty.
absl::StatusOr<std::string> GetComputationGraphviz(
    absl::Span<const DeviceBufferRef> buffer_refs,
    const absl::flat_hash_map<DeviceBufferRef, std::string>&
        buffer_ref_to_vars);

// Returns the MLIR representation of the computation graph of the given buffer
// refs. This traverses backwards from the given buffers through the graph of
// Aten operations that produced them.
//
// `buffer_refs` contains the buffers that are sinks of the computation graph.
//
// This function will return an error if:
//  - `buffer_refs` is empty.
absl::StatusOr<std::string> GetComputationMlir(
    absl::Span<const DeviceBufferRef> buffer_refs);

}  // namespace torch_tpu

#endif  // TORCH_TPU__INTERNAL_SYNC_SYNC_H_
