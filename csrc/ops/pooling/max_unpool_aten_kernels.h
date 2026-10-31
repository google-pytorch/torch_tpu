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

#ifndef TORCH_TPU_CSRC_OPS_POOLING_MAX_UNPOOL_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_POOLING_MAX_UNPOOL_ATEN_KERNELS_H_

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "c10/core/SymIntArrayRef.h"

namespace torch_tpu {

// Dispatched from ATen PrivateUse1 for aten::max_unpool2d.
// Scatters `self` into a zero tensor of spatial size `output_size` at the
// flattened spatial positions in `indices` (the inverse of
// max_pool2d_with_indices).
at::Tensor AtenMaxUnpool2d(const at::Tensor& self, const at::Tensor& indices,
                           c10::SymIntArrayRef output_size);

// Dispatched from ATen PrivateUse1 for aten::max_unpool2d.out.
at::Tensor& AtenMaxUnpool2dOut(const at::Tensor& self,
                               const at::Tensor& indices,
                               c10::SymIntArrayRef output_size,
                               at::Tensor& out);

// Dispatched from ATen PrivateUse1 for aten::max_unpool3d.
// `stride` and `padding` only take part in validation, as in ATen.
at::Tensor AtenMaxUnpool3d(const at::Tensor& self, const at::Tensor& indices,
                           c10::SymIntArrayRef output_size,
                           at::IntArrayRef stride, at::IntArrayRef padding);

// Dispatched from ATen PrivateUse1 for aten::max_unpool3d.out.
at::Tensor& AtenMaxUnpool3dOut(const at::Tensor& self,
                               const at::Tensor& indices,
                               c10::SymIntArrayRef output_size,
                               at::IntArrayRef stride, at::IntArrayRef padding,
                               at::Tensor& out);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_POOLING_MAX_UNPOOL_ATEN_KERNELS_H_
