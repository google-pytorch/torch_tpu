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

#ifndef TORCH_TPU_CSRC_OPS_DROPOUT_DROPOUT_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_DROPOUT_DROPOUT_ATEN_KERNELS_H_

#include <tuple>

#include "ATen/core/TensorBody.h"
#include "c10/util/Optional.h"

namespace torch_tpu {

// Implements `aten::dropout` (`dropout.default`) on `PrivateUse1`.
// Short-circuits no-op calls (`p <= 0.0`, `!train`, or `numel == 0`) by
// returning `input` directly without launching a device kernel, and dispatches
// a 1-output StableHLO dropout kernel when `train` is true and `p > 0.0`.
at::Tensor AtenDropout(const at::Tensor& input, double p, bool train);

// Implements `aten::dropout` (`dropout.default`) on `AutogradPrivateUse1`.
// Short-circuits no-op calls and non-differentiable training calls to
// `AtenDropout`, while delegating active autograd training calls to
// `at::native_dropout` (or a `NativeDropoutBackward0` node for complex inputs).
at::Tensor AtenDropoutAutograd(const at::Tensor& input, double p, bool train);

// Implements in-place `aten::dropout_` (`dropout_.default`) on `PrivateUse1`.
// Short-circuits no-op calls by returning `self` directly and dispatches an
// in-place 1-output StableHLO dropout kernel with buffer donation when active.
at::Tensor& AtenDropout_(at::Tensor& self, double p, bool train);

// Implements in-place `aten::dropout_` (`dropout_.default`) on
// `AutogradPrivateUse1`.
at::Tensor& AtenDropout_Autograd(at::Tensor& self, double p, bool train);

// Implements `aten::native_dropout` on `PrivateUse1`, returning `(output,
// mask)`.
std::tuple<at::Tensor, at::Tensor> AtenNativeDropout(const at::Tensor& input,
                                                     double p,
                                                     c10::optional<bool> train);

// Implements `aten::_fused_dropout` on `PrivateUse1`, returning `(output,
// mask)`.
std::tuple<at::Tensor, at::Tensor> AtenFusedDropout(
    const at::Tensor& self, double p, c10::optional<at::Generator> generator);

// Implements `aten::native_dropout_backward` on `PrivateUse1`.
at::Tensor AtenNativeDropoutBackward(const at::Tensor& grad_output,
                                     const at::Tensor& mask, double scale);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_DROPOUT_DROPOUT_ATEN_KERNELS_H_
