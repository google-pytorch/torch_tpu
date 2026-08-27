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

#ifndef TORCH_TPU_OPS_CONVOLUTION_CONVOLUTION_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_CONVOLUTION_CONVOLUTION_ATEN_KERNELS_H_

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>

#include "ATen/core/ATen_fwd.h"

namespace torch_tpu {

// at::convolution
at::Tensor AtenConvolution(const at::Tensor& input, const at::Tensor& weight,
                           const std::optional<at::Tensor>& bias_opt,
                           at::IntArrayRef stride, at::IntArrayRef padding,
                           at::IntArrayRef dilation, bool transposed,
                           at::IntArrayRef output_padding, int64_t groups);

// at::convolution.out
at::Tensor& AtenConvolutionOut(const at::Tensor& input,
                               const at::Tensor& weight,
                               const std::optional<at::Tensor>& bias_opt,
                               at::IntArrayRef stride, at::IntArrayRef padding,
                               at::IntArrayRef dilation, bool transposed,
                               at::IntArrayRef output_padding, int64_t groups,
                               at::Tensor& out);

// at::convolution_backward
std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenConvolutionBackward(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& weight, at::OptionalIntArrayRef bias_sizes,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups,
    std::array<bool, 3> output_mask);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_CONVOLUTION_CONVOLUTION_ATEN_KERNELS_H_
