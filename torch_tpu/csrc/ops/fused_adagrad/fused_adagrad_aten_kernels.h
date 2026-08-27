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

#ifndef TORCH_TPU_OPS_FUSED_ADAGRAD_FUSED_ADAGRAD_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_FUSED_ADAGRAD_FUSED_ADAGRAD_ATEN_KERNELS_H_

#include <optional>

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
}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_FUSED_ADAGRAD_FUSED_ADAGRAD_ATEN_KERNELS_H_
