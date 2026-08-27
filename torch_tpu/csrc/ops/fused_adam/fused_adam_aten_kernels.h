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

#ifndef TORCH_TPU_OPS_FUSED_ADAM_FUSED_ADAM_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_FUSED_ADAM_FUSED_ADAM_ATEN_KERNELS_H_

#include <optional>

#include "ATen/core/ATen_fwd.h"

namespace torch_tpu {
enum class AdamAmsgradMode : bool { kDisabled = false, kEnabled = true };
enum class AdamObjectiveMode : bool { kMinimize = false, kMaximize = true };

// Configuration options for an Adam optimizer step, bundling enum modes and
// optional tensor existence flags.
struct AdamStepConfig {
  AdamAmsgradMode amsgrad_mode;
  AdamObjectiveMode maximize_mode;
  bool has_grad_scale;
  bool has_found_inf;

  bool amsgrad() const { return amsgrad_mode == AdamAmsgradMode::kEnabled; }
  bool maximize() const {
    return maximize_mode == AdamObjectiveMode::kMaximize;
  }
};

// Executes multi-tensor Adam optimizer step with a scalar learning rate.
void AtenFusedAdam(at::TensorList self, at::TensorList grads,
                   at::TensorList exp_avgs, at::TensorList exp_avg_sqs,
                   at::TensorList max_exp_avg_sqs, at::TensorList state_steps,
                   double lr, double beta1, double beta2, double weight_decay,
                   double eps, bool amsgrad, bool maximize,
                   const std::optional<at::Tensor>& grad_scale,
                   const std::optional<at::Tensor>& found_inf);

// Executes multi-tensor Adam optimizer step with a tensor learning rate.
void AtenFusedAdamTensorLr(at::TensorList self, at::TensorList grads,
                           at::TensorList exp_avgs, at::TensorList exp_avg_sqs,
                           at::TensorList max_exp_avg_sqs,
                           at::TensorList state_steps, const at::Tensor& lr,
                           double beta1, double beta2, double weight_decay,
                           double eps, bool amsgrad, bool maximize,
                           const std::optional<at::Tensor>& grad_scale,
                           const std::optional<at::Tensor>& found_inf);
}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_FUSED_ADAM_FUSED_ADAM_ATEN_KERNELS_H_
