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

#ifndef TORCH_TPU_OPS_RMS_NORM_RMS_NORM_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_RMS_NORM_RMS_NORM_ATEN_KERNELS_H_

#include <array>
#include <optional>
#include <tuple>

#include "ATen/core/ATen_fwd.h"

namespace torch_tpu {

std::tuple<at::Tensor, at::Tensor> AtenFusedRmsNorm(
    const at::Tensor& input, const at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight, const std::optional<double> eps);

std::tuple<at::Tensor, at::Tensor> AtenFusedRmsNormBackward(
    const at::Tensor& grad_out, const at::Tensor& input,
    at::IntArrayRef normalized_shape, const at::Tensor& rstd,
    const std::optional<at::Tensor>& weight, std::array<bool, 2> output_mask);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_RMS_NORM_RMS_NORM_ATEN_KERNELS_H_
