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

#ifndef TORCH_TPU_EAGER_MATERIALIZE_H_
#define TORCH_TPU_EAGER_MATERIALIZE_H_

#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ATen/core/TensorBody.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/split_traversal.h"

// When an aten op is dispatched, we always create a DeviceBufferList to contain
// the result. Typically, this will contain a DeferredOp, which describes the
// MLIR necessary to execute the op at some point in the future. Less commonly,
// the DeviceBufferList may instead contain zero-sized buffers,
// or empty/meta buffers (such as by torch.empty()).
//
// At some point in the future, we may need the actual data for one or more
// tensors/buffers in the DeviceBufferList; this process is called
// materialization, and the behavior depends on the previous state.
//   - If the DeviceBufferList was alread fully materialized, no-op.
//   - If the DeviceBufferList has a DeferredOp, then that DeferredOp will be
//     compiled (or retrieved from the cache) and executed, and the
//     DeviceBufferList will be updated to hold the results.
//   - Zero-sized buffers in the DeviceBufferList do not need any data, so
//     materialization is typically a no-op. However, if a DeferredOp produces
//     a mix of zero-sized and non-zero-sized buffers, then the entire op must
//     be executed, which will produce some redundant zero-sized PjRtBuffers.
//   - Empty/meta buffers cannot be materialized, as they are purely placeholder
//     buffers for compile-only warm-up runs, or out-tensor assignments to be
//     overwritten.

namespace torch_tpu {

// Executes all deferred nodes in the list, and updates them in place.  No-op
// for nodes that are already materialized, or are entirely zero-sized
// buffers. Optional parameter split_traversal can be set to false to avoid
// splitting the op graph traversal and, hence, compile it as a whole. Returns
// an error if any of the nodes are empty, or if any leaf input in the traversed
// graph is empty/meta.
absl::Status Materialize(
    absl::Span<const SharedDeviceBufferList> nodes,
    MaterializationMode mode = MaterializationMode::kSplitSafe);

// Materializes all of the buffers and connected leaf nodes in-place.
// This materializes all DeviceBufferLists referenced which are not already
// materialized; no-op for all materialized DeviceBufferLists.
// Errors if any of the buffers are empty.
absl::Status Materialize(
    absl::Span<const DeviceBufferRef> buffer_refs,
    MaterializationMode mode = MaterializationMode::kSplitSafe);

// Materializes the given node in-place. Delegates to the span overload.
inline absl::Status Materialize(
    const SharedDeviceBufferList& node,
    MaterializationMode mode = MaterializationMode::kSplitSafe) {
  return Materialize(absl::Span<const SharedDeviceBufferList>(&node, 1), mode);
}

// Materializes the given buffer in-place. Delegates to the span overload.
inline absl::Status Materialize(
    const DeviceBufferRef& buffer_ref,
    MaterializationMode mode = MaterializationMode::kSplitSafe) {
  return Materialize(absl::Span<const DeviceBufferRef>(&buffer_ref, 1), mode);
}

// Returns a materialized DeviceBufferRef for the logical data in the tensor.
// This will materialize the base tensor in-place (see Materialize()) if it is
// deferred.
// If the tensor is a contiguous base tensor, then it will return this tensor.
// If it is a view, then an ephemeral DeviceBufferList will be created and
// materialized.
// Errors if the tensor's base DeviceBufferRef is empty.
absl::StatusOr<DeviceBufferRef> GetMaterialized(const at::Tensor& tensor);

// Returns a materialized DeviceBufferRefs each input tensor.
// This will materialize each base tensor in-place (see Materialize()) if it is
// deferred.
// All contiguous base tensors will be returned as-is.
// All views will be materialized into ephemeral DeviceBufferLists.
// Errors if any of the tensors' base DeviceBufferRefs are empty.
absl::StatusOr<std::vector<DeviceBufferRef>> GetMaterialized(
    absl::Span<const at::Tensor> tensors);

// Enqueues an executable for materialization.
// This will create one new DeviceBufferList for each output tensor, with the
// same shapes and dtypes as the output_shapes argument.
// These outputs will be materialized, i.e. will hold PjRtBuffers, but these
// PjRtBuffers will not be ready until the executable reaches the front of the
// materialization queue, is Executed, and the execution completes.
absl::StatusOr<std::vector<DeviceBufferRef>> EnqueueExecutable(
    SharedLoadedExecutable executable, std::vector<DeviceBufferRef> inputs,
    absl::Span<const Shape> output_shapes, std::string_view task_name = "");

// Sets the output nodes as error.
void SetOutputNodesAsError(absl::Span<const DeviceBufferRef> outputs,
                           absl::Status status);

void SetOutputNodesAsError(absl::Span<const SharedDeviceBufferList> outputs,
                           absl::Status status);

// Given a list of target DeviceBufferLists, adds all leaf nodes of their
// subgraphs to the list.
void AddLeafNodes(std::vector<SharedDeviceBufferList>& nodes);

absl::Status ExecuteMaterializationJob(
    absl::Span<const DeviceBufferRef> inputs,
    absl::Span<const DeviceBufferRef> outputs,
    std::vector<SharedLoadedExecutable> executables,
    std::string_view task_name = "anonymous");

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_MATERIALIZE_H_
