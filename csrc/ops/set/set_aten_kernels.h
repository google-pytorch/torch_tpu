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

#ifndef TORCH_TPU_OPS_SET_SET_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_SET_SET_ATEN_KERNELS_H_

#include "ATen/core/TensorBody.h"
#include "c10/core/Storage.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"

namespace torch_tpu {

// Sets tensor `self` to have a zero-element storage.
at::Tensor& AtenSet_(at::Tensor& self);

// Sets tensor `self` to have storage `src`. The size/strides/offset
// are set so that `self` is a flat tensor.
at::Tensor& AtenSet_SourceStorage(at::Tensor& self, c10::Storage src);

// Sets tensor `self` to have the provided storage, offset, sizes, and strides.
at::Tensor& AtenSet_SourceStorageOffset(at::Tensor& self, c10::Storage src,
                                        c10::SymInt storage_offset,
                                        c10::SymIntArrayRef size,
                                        c10::SymIntArrayRef stride);

// Sets tensor `self` to have the storage, offset, sizes, and strides
// of tensor `src`.
at::Tensor& AtenSet_SourceTensor(at::Tensor& self, const at::Tensor& src);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_SET_SET_ATEN_KERNELS_H_
