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

#ifndef TORCH_TPU_OPS_GROUP_NORM_GROUP_NORM_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_GROUP_NORM_GROUP_NORM_ATEN_KERNELS_H_

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>

#include "ATen/core/TensorBody.h"

namespace torch_tpu {

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenNativeGroupNormBackward(
    const at::Tensor& grad_out, const at::Tensor& input, const at::Tensor& mean,
    const at::Tensor& rstd, const std::optional<at::Tensor>& weight, int64_t n,
    int64_t c, int64_t h_w, int64_t group, std::array<bool, 3> output_mask);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_GROUP_NORM_GROUP_NORM_ATEN_KERNELS_H_
