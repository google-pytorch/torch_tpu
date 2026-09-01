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

#ifndef TORCH_TPU_OPS_GRID_SAMPLER_GRID_SAMPLER_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_GRID_SAMPLER_GRID_SAMPLER_ATEN_KERNELS_H_

#include <array>
#include <cstdint>
#include <tuple>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

at::Tensor AtenGridSampler2d(const at::Tensor& input, const at::Tensor& grid,
                             int64_t interpolation_mode, int64_t padding_mode,
                             bool align_corners);

at::Tensor AtenGridSampler3d(const at::Tensor& input, const at::Tensor& grid,
                             int64_t interpolation_mode, int64_t padding_mode,
                             bool align_corners);

std::tuple<at::Tensor, at::Tensor> AtenGridSampler2dBackward(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& grid, int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, std::array<bool, 2> output_mask);

std::tuple<at::Tensor, at::Tensor> AtenGridSampler3dBackward(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& grid, int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, std::array<bool, 2> output_mask);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_GRID_SAMPLER_GRID_SAMPLER_ATEN_KERNELS_H_
