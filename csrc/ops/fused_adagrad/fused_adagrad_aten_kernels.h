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

#ifndef TORCH_TPU_CSRC_OPS_FUSED_ADAGRAD_FUSED_ADAGRAD_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_FUSED_ADAGRAD_FUSED_ADAGRAD_ATEN_KERNELS_H_

#include <optional>
#include <tuple>
#include <vector>

#include "ATen/core/ATen_fwd.h"

namespace torch_tpu {

// Executes multi-tensor Adagrad optimizer step with a scalar learning rate.
void AtenFusedAdagrad(at::TensorList self, at::TensorList grads,
                      at::TensorList state_sums, at::TensorList state_steps,
                      double lr, double lr_decay, double weight_decay,
                      double eps, bool maximize,
                      const std::optional<at::Tensor>& grad_scale,
                      const std::optional<at::Tensor>& found_inf);

// Executes multi-tensor Adagrad optimizer step with a tensor learning rate.
void AtenFusedAdagradTensorLr(at::TensorList self, at::TensorList grads,
                              at::TensorList state_sums,
                              at::TensorList state_steps, const at::Tensor& lr,
                              double lr_decay, double weight_decay, double eps,
                              bool maximize,
                              const std::optional<at::Tensor>& grad_scale,
                              const std::optional<at::Tensor>& found_inf);

// Meta implementations for upstream ATen fused Adagrad ops
// (`TORCH_LIBRARY_IMPL(aten, Meta, m)`):
// - Why `TensorLr` suffix exists: ATen defines two overloads for each fused
//   optimizer—the default overload taking scalar `double lr` (`_fused_adagrad_`
//   / `_fused_adagrad`) and the `.tensor_lr` overload taking `const at::Tensor&
//   lr` (`_fused_adagrad_.tensor_lr` / `_fused_adagrad.tensor_lr`, used when
//   `lr` is passed as a Tensor to avoid `torch.compile` recompilations on LR
//   schedule updates).
// - Why in-place `MetaFusedAdagrad` and `MetaFusedAdagradTensorLr` are no-ops:
//   Meta kernels only propagate tensor metadata (shapes, strides, dtypes) for
//   `FakeTensor` / `torch.compile` without running numerical math. Because the
//   in-place ops (`-> ()`) return `void` and mutate existing tensors in place
//   without altering their shapes, strides, or dtypes, their Meta kernels are
//   no-ops (`{}`). By contrast, the `Functional` variants (`_fused_adagrad` /
//   `_fused_adagrad.tensor_lr`) are out-of-place ops used by AOTAutograd
//   functionalization and must return newly allocated `at::empty_like` Meta
//   tensors matching the mutated inputs.
void MetaFusedAdagrad(at::TensorList self, at::TensorList grads,
                      at::TensorList state_sums, at::TensorList state_steps,
                      double lr, double lr_decay, double weight_decay,
                      double eps, bool maximize,
                      const std::optional<at::Tensor>& grad_scale,
                      const std::optional<at::Tensor>& found_inf);

void MetaFusedAdagradTensorLr(at::TensorList self, at::TensorList grads,
                              at::TensorList state_sums,
                              at::TensorList state_steps, const at::Tensor& lr,
                              double lr_decay, double weight_decay, double eps,
                              bool maximize,
                              const std::optional<at::Tensor>& grad_scale,
                              const std::optional<at::Tensor>& found_inf);

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>,
           std::vector<at::Tensor>, std::vector<at::Tensor>>
MetaFusedAdagradFunctional(at::TensorList self, at::TensorList grads,
                           at::TensorList state_sums,
                           at::TensorList state_steps, double lr,
                           double lr_decay, double weight_decay, double eps,
                           bool maximize,
                           const std::optional<at::Tensor>& grad_scale,
                           const std::optional<at::Tensor>& found_inf);

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>,
           std::vector<at::Tensor>>
MetaFusedAdagradFunctionalTensorLr(at::TensorList self, at::TensorList grads,
                                   at::TensorList state_sums,
                                   at::TensorList state_steps,
                                   const at::Tensor& lr, double lr_decay,
                                   double weight_decay, double eps,
                                   bool maximize,
                                   const std::optional<at::Tensor>& grad_scale,
                                   const std::optional<at::Tensor>& found_inf);
}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_FUSED_ADAGRAD_FUSED_ADAGRAD_ATEN_KERNELS_H_
