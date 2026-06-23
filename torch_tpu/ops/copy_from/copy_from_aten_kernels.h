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

#ifndef TORCH_TPU_OPS_COPY_FROM_COPY_FROM_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_COPY_FROM_COPY_FROM_ATEN_KERNELS_H_

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

// Updates the c10::Storage of self_dest to point to a new c10::StorageImpl
// that contains a copy of the data of src.
// src and self_dest must have the same dtype, but may be on different
// devices.
// Note: even though self_dest is given as `const at::Tensor&`, it is expected
// that the c10::TensorImpl referenced by self_dest will be modified inplace.
// The const-ness means that self_dest must not be redirected to a *different*
// c10::TensorImpl.
// https://github.com/pytorch/pytorch/blob/v2.8.0-rc8/aten/src/ATen/native/Copy.cpp#L225
at::Tensor AtenCopyFrom(const at::Tensor& src, const at::Tensor& self_dest,
                        bool non_blocking);

// By going through its usage in ATen/native/CPUFallback.cpp. This function is
// basically just _copy_from, either h2d or d2h
// Per https://github.com/pytorch/xla/issues/2881 it is introduced in 2021 to
// fix an issue with aten:cpufallback for pytorch_xla.
// TODO: follow up with pytorch, try to replace _copy_from_and_resize with
// _copy_from
at::Tensor AtenCopyFromAndResize(const at::Tensor& self, const at::Tensor& dst);

// Copies a 1-element tensor from TPU to CPU and converts it to a scalar.
at::Scalar AtenLocalScalarDense(const at::Tensor& self);

// Identical to _copy_from, only registered to avoid a dispatch error in torch.
//
// When the input to copy_ is a complex number with the conjugate bit set,
// copy_()' default implementation will call MathOpFallback::fallback_impl; this
// will call clone(), which decomposes into empty_strided() and copy_() again.
// This creates a stack overflow and crashes with a segmentation fault.
//
// We can break this loop by registering at least one of these ops; copy_ is the
// easiest one to break the loop as the signature is almost identical to
// _copy_from.
at::Tensor& AtenCopy_(at::Tensor& self, const at::Tensor& src,
                      bool non_blocking);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_COPY_FROM_COPY_FROM_ATEN_KERNELS_H_
