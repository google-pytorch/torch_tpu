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

#ifndef TORCH_TPU_OPS_BINARY_CROSS_ENTROPY_BINARY_CROSS_ENTROPY_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_BINARY_CROSS_ENTROPY_BINARY_CROSS_ENTROPY_ATEN_KERNELS_H_

#include <cstdint>
#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

at::Tensor AtenBinaryCrossEntropy(const at::Tensor& self,
                                  const at::Tensor& target,
                                  const std::optional<at::Tensor>& weight,
                                  int64_t reduction);

at::Tensor& AtenBinaryCrossEntropyOut(const at::Tensor& self,
                                      const at::Tensor& target,
                                      const std::optional<at::Tensor>& weight,
                                      int64_t reduction, at::Tensor& out);

at::Tensor AtenBinaryCrossEntropyBackward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const std::optional<at::Tensor>& weight,
    int64_t reduction);

at::Tensor& AtenBinaryCrossEntropyBackwardGradInput(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const std::optional<at::Tensor>& weight,
    int64_t reduction, at::Tensor& grad_input);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_BINARY_CROSS_ENTROPY_BINARY_CROSS_ENTROPY_ATEN_KERNELS_H_
