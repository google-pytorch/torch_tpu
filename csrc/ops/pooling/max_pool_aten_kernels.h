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

#ifndef TORCH_TPU_OPS_POOLING_MAX_POOL_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_POOLING_MAX_POOL_ATEN_KERNELS_H_

#include <tuple>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "torch/csrc/autograd/custom_function.h"

namespace torch_tpu {

at::Tensor AtenMaxPool2d(const at::Tensor& self, at::IntArrayRef kernel_size,
                         at::IntArrayRef stride, at::IntArrayRef padding,
                         at::IntArrayRef dilation, bool ceil_mode);

std::tuple<at::Tensor&, at::Tensor&> AtenMaxPool2dWithIndicesOut(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    at::Tensor& out, at::Tensor& indices);

at::Tensor TpuMaxPool2d(const at::Tensor& self, at::IntArrayRef kernel_size,
                        at::IntArrayRef stride, at::IntArrayRef padding,
                        at::IntArrayRef dilation, bool ceil_mode);

at::Tensor TpuMaxPool2dBackward(const at::Tensor& grad_output,
                                const at::Tensor& self,
                                at::IntArrayRef kernel_size,
                                at::IntArrayRef stride, at::IntArrayRef padding,
                                at::IntArrayRef dilation, bool ceil_mode);

std::tuple<at::Tensor, at::Tensor> AtenMaxPool3dWithIndices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode);

std::tuple<at::Tensor&, at::Tensor&> AtenMaxPool3dWithIndicesOut(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    at::Tensor& out, at::Tensor& indices);

at::Tensor& AtenMaxPool2dWithIndicesBackwardGradInput(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices, at::Tensor& grad_input);

at::Tensor AtenMaxPool3dWithIndicesBackward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices);

at::Tensor& AtenMaxPool3dWithIndicesBackwardGradInput(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices, at::Tensor& grad_input);

struct TpuMaxPool2dAutograd
    : public torch::autograd::Function<TpuMaxPool2dAutograd> {
  static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                            const at::Tensor& self, at::IntArrayRef kernel_size,
                            at::IntArrayRef stride, at::IntArrayRef padding,
                            at::IntArrayRef dilation, bool ceil_mode);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs);
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_POOLING_MAX_POOL_ATEN_KERNELS_H_
