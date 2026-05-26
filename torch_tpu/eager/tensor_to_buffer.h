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

#ifndef TORCH_TPU_EAGER_TENSOR_TO_BUFFER_H_
#define TORCH_TPU_EAGER_TENSOR_TO_BUFFER_H_

#include <vector>

#include "ATen/core/CachingHostAllocator.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/Allocator.h"
#include "c10/core/Storage.h"
#include "c10/core/TensorImpl.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/structured_log_buffer.h"

// This library connects the DeviceBufferRef/DeviceBufferList/DeferredOp graph
// structure with aten, offering bidirectional conversion between the two.
//
// The aten/c10 pointer hierarchy is:
//   at::Tensor
//   -> c10::TensorImpl { .storage = c10::Storage }
//   -> c10::StorageImpl {
//     .data_ptr = c10::DataPtr, .allocator = c10::Allocator }
//   -> c10::UniqueVoidPtr { .data_ = void*, .ctx_ = void* }
//
// In order to register a custom backend, we need to supply two things:
//  1. A c10::Allocator subclass that is able to create c10::DataPtrs of a
//     specified size_t bytes (but no shape or type information)
//  2. A mechanism to cast the void* pointers within a c10::DataPtr to concrete
//     types as required to resolve registered aten kernels.
//
// For torch_tpu, we satisfy these requirements by:
//  -  Defining a global allocator (accessed by calling GetTpuAllocator()),
//     which will create one-dimensionsal u8 tensors when requested
//  -  Defining a DeviceBufferList class, which describes the shapes, types, and
//     current evaluation state (materialized, deferred, or nonexistent) of the
//     data on the XLA device.
//  -  Defining a class for the referent of the c10::DataPtr, the
//     DeviceBufferRef. See `device_buffer.h` for more details.
//  -  Providing getter functions to resolve the DeviceBufferRef from a given
//     c10::Storage or at::Tensor. This includes the logic necessary to convert
//     from a contiguous base DeviceBufferRef to a strided view; this logic
//     is defined in view_decomposition.h and applied in GetBuffer.
//
// These functions primarily operate at the boundaries of any call from aten;
// a function call provides at::Tensors, which we extract DeviceBufferRefs from,
// perform an operation, and then either return a new tensor (using MakeTensor)
// or update an existing tensor (using AssignBufferToAtTensor).
namespace torch_tpu {

// Returns the device buffer for the given tensor's base tensor.
//
// Every tensor has a base: for a view tensor, the base is the actual tensor
// that backs up the view; for an independent tensor, the base is itself.
//
// The tensor must be allocated by the TpuAllocator
// and have a valid DeviceBufferRef via the data_ptr context.
//
// The returned DeviceBufferRef is always the contiguous buffer that was used
// to create the c10::StorageImpl backing the tensor; if tensor is a view, then
// the returned DeviceBufferRef may have different dimensions than the tensor.
absl::StatusOr<DeviceBufferRef> GetBaseBuffer(const c10::TensorImpl& tensor);
absl::StatusOr<DeviceBufferRef> GetBaseBuffer(const at::Tensor& tensor);
absl::StatusOr<DeviceBufferRef> GetBaseBuffer(const c10::Storage& storage);

// Extracts the DeviceBufferRef from the given ATen tensor.
//
// If the tensor is a view, the buffer will be a view of the base buffer.
// Otherwise, the buffer will be the base buffer itself.
//
// The tensor must be allocated by the TpuAllocator
// and have a valid DeviceBufferRef via the data_ptr context.
//
// If the tensor is a view, then the returned DeviceBufferRef will always be
// in the kDeferred state, with a DeferredOp that will convert the inner
// contiguous base buffer into the view's layout.
//
// The returned value may be in any DeviceBufferRefState, including
// kMaterialized, kDeferred, and kPlaceholder. Callers are
// responsible for handling any unexpected states; for example, erroring on a
// kPlaceholder state before calling a PjRtLoadedExecutable.
absl::StatusOr<DeviceBufferRef> GetBuffer(const at::Tensor& tensor);
absl::StatusOr<DeviceBufferRef> GetBuffer(const c10::TensorImpl& tensor);

// Extracts all DeviceBufferRefs from the list of AtenTensors.
// This is just GetBuffer in a loop, exiting on the first error.
absl::StatusOr<std::vector<DeviceBufferRef>> GetBuffers(
    absl::Span<const at::Tensor> tensors);

// Creates a c10::Storage pointer to a new c10::StorageImpl, which holds a
// c10::DataPtr to the given DeviceBufferRef.
// The nbytes of the c10::StorageImpl will be set to the size
// of the DeviceBufferRef.
// The DeviceBufferRef must be in a valid state.
c10::Storage MakeStorage(DeviceBufferRef buffer_ref);

// Returns an ATen tensor with the given DeviceBufferRef.
//
// This creates a full pointer chain of a new c10::DataPtr,
// c10::StorageImpl/c10::Storage, c10::TensorImpl, and at::Tensor.
//
// The resulting tensor will have the same sizes and dtype as the
// DeviceBufferRef (converting from mlir::ElementType to ScalarType) and will
// be contiguous.
at::Tensor MakeTensor(DeviceBufferRef device_buffer_ref);

// Assigns the given DeviceBufferRef to the given ATen tensor.
//
// This is typically used for inplace operations that overwrite a "self" tensor,
// and "out"-tensor operations that use a provided destination tensor.
//
// Args:
//    result_buf: the result buffer.
//    tensor: reference to the tensor into which to copy result_buf.
// Requires:
//    result_buf and tensor must have the same shape and dtype.
absl::Status AssignBufferToAtTensor(DeviceBufferRef result_buf,
                                    const at::Tensor& tensor);

// Returns the C10 allocator singleton for TPU.
c10::Allocator* GetTpuAllocator();

// Returns the Tpu pinned memory allocator.
at::HostAllocator* GetTpuPinnedAllocator();

// Returns true if the given pointer was allocated by the Tpu pinned memory
// allocator.
bool IsTpuPinnedPtr(const void* ptr);

// Registers the TpuAllocator as the allocator for the PrivateUse1 device.
// This must be called before using the allocator for any device operations.
void RegisterTpuAllocator();

// Returns a materialized DeviceBufferRef for the logical data in the tensor.
// This will materialize the base tensor in-place (see Materialize()) if it is
// deferred.
// If the tensor is a contiguous base tensor, then it will return this tensor.
// If it is a view, then an ephemeral DeviceBufferList will be created and
// materialized.
// Errors if the tensor's base DeviceBufferRef is a placeholder.
absl::StatusOr<DeviceBufferRef> GetMaterialized(const at::Tensor& tensor,
                                                MaterializationReason reason);

// Returns a materialized DeviceBufferRefs each input tensor.
// This will materialize each base tensor in-place (see Materialize()) if it is
// deferred.
// All contiguous base tensors will be returned as-is.
// All views will be materialized into ephemeral DeviceBufferLists.
// Errors if any of the tensors' base DeviceBufferRefs are placeholders or
// depend on placeholders.
absl::StatusOr<std::vector<DeviceBufferRef>> GetMaterialized(
    absl::Span<const at::Tensor> tensors, MaterializationReason reason);

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_TENSOR_TO_BUFFER_H_
