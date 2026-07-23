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

#ifndef TORCH_TPU_OPS_HUBER_LOSS_HUBER_LOSS_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_HUBER_LOSS_HUBER_LOSS_ATEN_KERNELS_H_

#include <cstdint>

#include "ATen/core/TensorBody.h"

namespace torch_tpu {

// Computes the Huber loss between `self` and `target`.
at::Tensor AtenHuberLoss(const at::Tensor& self, const at::Tensor& target,
                         int64_t reduction, double delta);

// Out-variant of AtenHuberLoss, storing result in `out`.
at::Tensor& AtenHuberLossOut(const at::Tensor& self, const at::Tensor& target,
                             int64_t reduction, double delta, at::Tensor& out);

// Out-variant of AtenHuberLossBackward, storing gradient in `grad_input`.
at::Tensor& AtenHuberLossBackwardOut(const at::Tensor& grad_output,
                                     const at::Tensor& self,
                                     const at::Tensor& target,
                                     int64_t reduction, double delta,
                                     at::Tensor& grad_input);

// Computes gradient of Huber loss with respect to `self`.
at::Tensor AtenHuberLossBackward(const at::Tensor& grad_output,
                                 const at::Tensor& self,
                                 const at::Tensor& target, int64_t reduction,
                                 double delta);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_HUBER_LOSS_HUBER_LOSS_ATEN_KERNELS_H_
