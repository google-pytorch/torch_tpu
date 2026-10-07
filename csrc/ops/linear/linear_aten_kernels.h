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

#ifndef TORCH_TPU_CSRC_OPS_LINEAR_LINEAR_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_LINEAR_LINEAR_ATEN_KERNELS_H_

#include <array>
#include <optional>
#include <tuple>

#include "ATen/core/ATen_fwd.h"

namespace torch_tpu {

// Implements `aten::linear` (`linear.default`) on `AutogradPrivateUse1`.
// Decomposes Python wrapper tensor subclasses (such as `DTensor`) via
// `at::native::linear` (matching CPU/CUDA behavior) while dispatching standard
// TPU tensors and `FakeTensor`s (`torch.compile` / `torch.export`) through
// `LinearBackward0` and `AtenLinear`.
at::Tensor AtenLinearAutograd(const at::Tensor& input, const at::Tensor& weight,
                              const std::optional<at::Tensor>& bias);

// Implements `aten::linear` (`linear.default`): y = x @ A^T + b.
at::Tensor AtenLinear(const at::Tensor& input, const at::Tensor& weight,
                      const std::optional<at::Tensor>& bias);

// Implements `aten::linear.out`: writes y = x @ A^T + b into `out`.
at::Tensor& AtenLinearOut(const at::Tensor& input, const at::Tensor& weight,
                          const std::optional<at::Tensor>& bias,
                          at::Tensor& out);

// Implements `aten::linear_backward`, returning `(grad_input, grad_weight,
// grad_bias)` according to `output_mask`.
std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenLinearBackward(
    const at::Tensor& self, const at::Tensor& grad_output,
    const at::Tensor& weight, std::array<bool, 3> output_mask);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_LINEAR_LINEAR_ATEN_KERNELS_H_
