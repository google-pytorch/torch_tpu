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

#ifndef TORCH_TPU_OPS_REFLECTION_PAD_REFLECTION_PAD_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_REFLECTION_PAD_REFLECTION_PAD_ATEN_KERNELS_H_

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

at::Tensor& AtenReflectionPad1dOut(const at::Tensor& self,
                                   at::IntArrayRef padding, at::Tensor& out);
at::Tensor AtenReflectionPad2d(const at::Tensor& self, at::IntArrayRef padding);
at::Tensor& AtenReflectionPad2dOut(const at::Tensor& self,
                                   at::IntArrayRef padding, at::Tensor& out);
at::Tensor& AtenReflectionPad3dOut(const at::Tensor& self,
                                   at::IntArrayRef padding, at::Tensor& out);
at::Tensor& AtenReflectionPad1dBackwardGradInput(const at::Tensor& grad_output,
                                                 const at::Tensor& self,
                                                 at::IntArrayRef padding,
                                                 at::Tensor& grad_input);
at::Tensor& AtenReflectionPad2dBackwardGradInput(const at::Tensor& grad_output,
                                                 const at::Tensor& self,
                                                 at::IntArrayRef padding,
                                                 at::Tensor& grad_input);
at::Tensor& AtenReflectionPad3dBackwardGradInput(const at::Tensor& grad_output,
                                                 const at::Tensor& self,
                                                 at::IntArrayRef padding,
                                                 at::Tensor& grad_input);
at::Tensor AtenReflectionPad2dBackward(const at::Tensor& grad_output,
                                       const at::Tensor& self,
                                       at::IntArrayRef padding);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_REFLECTION_PAD_REFLECTION_PAD_ATEN_KERNELS_H_
