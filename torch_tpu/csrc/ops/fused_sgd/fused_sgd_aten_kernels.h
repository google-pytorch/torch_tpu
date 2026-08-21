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

#ifndef TORCH_TPU_OPS_FUSED_SGD_FUSED_SGD_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_FUSED_SGD_FUSED_SGD_ATEN_KERNELS_H_

#include <optional>

#include "ATen/core/ATen_fwd.h"

namespace torch_tpu {

// Executes multi-tensor SGD optimizer step with a scalar learning rate.
void AtenFusedSgd(at::TensorList self, at::TensorList grads,
                  at::TensorList momentum_buffer_list, double weight_decay,
                  double momentum, double lr, double dampening, bool nesterov,
                  bool maximize, bool is_first_step,
                  const std::optional<at::Tensor>& grad_scale,
                  const std::optional<at::Tensor>& found_inf);

// Executes multi-tensor SGD optimizer step with a tensor learning rate.
void AtenFusedSgdTensorLr(at::TensorList self, at::TensorList grads,
                          at::TensorList momentum_buffer_list,
                          double weight_decay, double momentum,
                          const at::Tensor& lr, double dampening, bool nesterov,
                          bool maximize, bool is_first_step,
                          const std::optional<at::Tensor>& grad_scale,
                          const std::optional<at::Tensor>& found_inf);
}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_FUSED_SGD_FUSED_SGD_ATEN_KERNELS_H_
