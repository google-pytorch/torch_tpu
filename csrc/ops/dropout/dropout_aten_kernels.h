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

#ifndef TORCH_TPU_OPS_DROPOUT_DROPOUT_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_DROPOUT_DROPOUT_ATEN_KERNELS_H_

#include <tuple>

#include "ATen/core/TensorBody.h"
#include "c10/util/Optional.h"

namespace torch_tpu {

std::tuple<at::Tensor, at::Tensor> AtenDropout(const at::Tensor& input,
                                               double p,
                                               c10::optional<bool> train);

std::tuple<at::Tensor, at::Tensor> AtenFusedDropout(
    const at::Tensor& self, double p, c10::optional<at::Generator> generator);

at::Tensor AtenNativeDropoutBackward(const at::Tensor& grad_output,
                                     const at::Tensor& mask, double scale);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_DROPOUT_DROPOUT_ATEN_KERNELS_H_
