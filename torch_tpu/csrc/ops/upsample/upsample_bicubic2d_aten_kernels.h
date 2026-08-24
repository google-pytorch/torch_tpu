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

#ifndef TORCH_TPU_OPS_UPSAMPLE_UPSAMPLE_BICUBIC2D_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_UPSAMPLE_UPSAMPLE_BICUBIC2D_ATEN_KERNELS_H_

#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

at::Tensor& AtenUpsampleBicubic2dOut(const at::Tensor& self,
                                     at::IntArrayRef output_size,
                                     bool align_corners,
                                     std::optional<double> scale_h,
                                     std::optional<double> scale_w,
                                     at::Tensor& out);

at::Tensor& AtenUpsampleBicubic2dBackwardGradInput(
    const at::Tensor& grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size, bool align_corners,
    std::optional<double> scales_h, std::optional<double> scales_w,
    at::Tensor& grad_input);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_UPSAMPLE_UPSAMPLE_BICUBIC2D_ATEN_KERNELS_H_
