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

#ifndef TORCH_TPU_CSRC_OPS_POOLING_MAX_POOL_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_POOLING_MAX_POOL_ATEN_KERNELS_H_

#include <tuple>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "torch/csrc/autograd/custom_function.h"

namespace torch_tpu {

// ============================================================================
// MaxPool1D Kernels
// ============================================================================

// Dispatched from ATen AutogradPrivateUse1 for aten::max_pool1d.
// Routes to TpuMaxPool1dAutograd for automatic differentiation when dilation is
// trivial, avoiding index tensor generation and variadic reduction.
at::Tensor AtenMaxPool1d(const at::Tensor& self, at::IntArrayRef kernel_size,
                         at::IntArrayRef stride, at::IntArrayRef padding,
                         at::IntArrayRef dilation, bool ceil_mode);

// Dispatched from PrivateUse1 for tpu::_max_pool1d.
// Directly lowers to StableHLO ReduceWindowOp without returning or computing
// indices.
at::Tensor TpuMaxPool1d(const at::Tensor& self, at::IntArrayRef kernel_size,
                        at::IntArrayRef stride, at::IntArrayRef padding,
                        at::IntArrayRef dilation, bool ceil_mode);

// Dispatched from PrivateUse1 for tpu::_max_pool1d_backward.
// Directly lowers to StableHLO SelectAndScatterOp.
at::Tensor TpuMaxPool1dBackward(const at::Tensor& grad_output,
                                const at::Tensor& self,
                                at::IntArrayRef kernel_size,
                                at::IntArrayRef stride, at::IntArrayRef padding,
                                at::IntArrayRef dilation, bool ceil_mode);

// Autograd function for MaxPool1D that routes to tpu::_max_pool1d during
// forward and tpu::_max_pool1d_backward during backward.
struct TpuMaxPool1dAutograd
    : public torch::autograd::Function<TpuMaxPool1dAutograd> {
  static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                            const at::Tensor& self, at::IntArrayRef kernel_size,
                            at::IntArrayRef stride, at::IntArrayRef padding,
                            at::IntArrayRef dilation, bool ceil_mode);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs);
};

// ============================================================================
// MaxPool2D Kernels
// ============================================================================

// Dispatched from ATen AutogradPrivateUse1 for aten::max_pool2d.
// Routes to TpuMaxPool2dAutograd for automatic differentiation when dilation is
// trivial, avoiding index tensor generation and variadic reduction.
at::Tensor AtenMaxPool2d(const at::Tensor& self, at::IntArrayRef kernel_size,
                         at::IntArrayRef stride, at::IntArrayRef padding,
                         at::IntArrayRef dilation, bool ceil_mode);

// Dispatched from ATen PrivateUse1 for aten::max_pool2d_with_indices.out.
// Uses StableHLO ReduceWindowOp with iota indices for backwards compatibility
// when indices are explicitly requested.
std::tuple<at::Tensor&, at::Tensor&> AtenMaxPool2dWithIndicesOut(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    at::Tensor& out, at::Tensor& indices);

// Dispatched from PrivateUse1 for tpu::_max_pool2d.
// Directly lowers to StableHLO ReduceWindowOp without returning or computing
// indices.
at::Tensor TpuMaxPool2d(const at::Tensor& self, at::IntArrayRef kernel_size,
                        at::IntArrayRef stride, at::IntArrayRef padding,
                        at::IntArrayRef dilation, bool ceil_mode);

// Dispatched from PrivateUse1 for tpu::_max_pool2d_backward.
// Directly lowers to StableHLO SelectAndScatterOp.
at::Tensor TpuMaxPool2dBackward(const at::Tensor& grad_output,
                                const at::Tensor& self,
                                at::IntArrayRef kernel_size,
                                at::IntArrayRef stride, at::IntArrayRef padding,
                                at::IntArrayRef dilation, bool ceil_mode);

// Autograd function for MaxPool2D that routes to tpu::_max_pool2d during
// forward and tpu::_max_pool2d_backward during backward.
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

// Dispatched from ATen PrivateUse1 for
// aten::max_pool2d_with_indices_backward.grad_input.
at::Tensor& AtenMaxPool2dWithIndicesBackwardGradInput(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices, at::Tensor& grad_input);

// ============================================================================
// MaxPool3D Kernels
// ============================================================================

// Dispatched from ATen AutogradPrivateUse1 for aten::max_pool3d.
// Routes to TpuMaxPool3dAutograd for automatic differentiation when dilation is
// trivial, avoiding index tensor generation and variadic reduction.
at::Tensor AtenMaxPool3d(const at::Tensor& self, at::IntArrayRef kernel_size,
                         at::IntArrayRef stride, at::IntArrayRef padding,
                         at::IntArrayRef dilation, bool ceil_mode);

// Dispatched from PrivateUse1 for tpu::_max_pool3d.
// Directly lowers to StableHLO ReduceWindowOp without returning or computing
// indices.
at::Tensor TpuMaxPool3d(const at::Tensor& self, at::IntArrayRef kernel_size,
                        at::IntArrayRef stride, at::IntArrayRef padding,
                        at::IntArrayRef dilation, bool ceil_mode);

// Dispatched from PrivateUse1 for tpu::_max_pool3d_backward.
// Directly lowers to StableHLO SelectAndScatterOp.
at::Tensor TpuMaxPool3dBackward(const at::Tensor& grad_output,
                                const at::Tensor& self,
                                at::IntArrayRef kernel_size,
                                at::IntArrayRef stride, at::IntArrayRef padding,
                                at::IntArrayRef dilation, bool ceil_mode);

// Autograd function for MaxPool3D that routes to tpu::_max_pool3d during
// forward and tpu::_max_pool3d_backward during backward.
struct TpuMaxPool3dAutograd
    : public torch::autograd::Function<TpuMaxPool3dAutograd> {
  static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                            const at::Tensor& self, at::IntArrayRef kernel_size,
                            at::IntArrayRef stride, at::IntArrayRef padding,
                            at::IntArrayRef dilation, bool ceil_mode);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs);
};

// Dispatched from ATen PrivateUse1 for aten::max_pool3d_with_indices.
std::tuple<at::Tensor, at::Tensor> AtenMaxPool3dWithIndices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode);

// Dispatched from ATen PrivateUse1 for aten::max_pool3d_with_indices.out.
std::tuple<at::Tensor&, at::Tensor&> AtenMaxPool3dWithIndicesOut(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    at::Tensor& out, at::Tensor& indices);

// Dispatched from ATen PrivateUse1 for aten::max_pool3d_with_indices_backward.
at::Tensor AtenMaxPool3dWithIndicesBackward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices);

// Dispatched from ATen PrivateUse1 for
// aten::max_pool3d_with_indices_backward.grad_input.
at::Tensor& AtenMaxPool3dWithIndicesBackwardGradInput(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices, at::Tensor& grad_input);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_POOLING_MAX_POOL_ATEN_KERNELS_H_
