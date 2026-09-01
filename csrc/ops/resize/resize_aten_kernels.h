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

#ifndef TORCH_TPU_OPS_RESIZE_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_RESIZE_ATEN_KERNELS_H_

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/Optional.h"
#include "torch/headeronly/core/MemoryFormat.h"

namespace torch_tpu {

// at::resize_
const at::Tensor& AtenResize_(
    const at::Tensor& self_const, c10::IntArrayRef size,
    c10::optional<at::MemoryFormat> memory_format_opt);

// Resizes the target tensor to the given size.
//
// After resizing, the tensor's metadata is updated to be contiguous.
//
// If the requested capacity (implied by the new size and dtype) is less than or
// equal to the current storage capacity, the layout is updated on the existing
// tensor without reallocating storage.
//
// If the requested capacity exceeds the current capacity, a new larger storage
// buffer is allocated. The existing data is copied to the new buffer, and the
// tensor's storage pointer is updated. Since the storage is shared, this will
// also update the storage pointer for all other views of this tensor (resizing
// their shared storage in-place, although their shape/stride metadata remain
// unchanged).
absl::Status ResizeTensor(const at::Tensor& self, c10::IntArrayRef size);

// Resizes the target tensor to the given shape if the shape differs.
//
// This should be used in place of at::native::resize_output in TPU kernels to
// avoid dispatch overheads. Returns absl::OkStatus() if no resize was needed or
// if the resize succeeded.
inline absl::Status ResizeTensorIfShapeDiffers(const at::Tensor& tensor,
                                               c10::IntArrayRef shape) {
  if (tensor.sizes() == shape) {
    return absl::OkStatus();
  }
  return ResizeTensor(tensor, shape);
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_RESIZE_ATEN_KERNELS_H_
