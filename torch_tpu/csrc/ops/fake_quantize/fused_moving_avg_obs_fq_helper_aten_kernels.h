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

#ifndef TORCH_TPU_OPS_FAKE_QUANTIZE_FUSED_MOVING_AVG_OBS_FQ_HELPER_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_FAKE_QUANTIZE_FUSED_MOVING_AVG_OBS_FQ_HELPER_ATEN_KERNELS_H_

#include <cstdint>
#include <tuple>

#include "ATen/core/ATen_fwd.h"

namespace torch_tpu {

std::tuple<at::Tensor, at::Tensor> FusedMovingAvgObsFqHelper(
    const at::Tensor& self, const at::Tensor& observer_on,
    const at::Tensor& fake_quant_on, at::Tensor& running_min,
    at::Tensor& running_max, at::Tensor& scale, at::Tensor& zero_point,
    double averaging_const, int64_t quant_min, int64_t quant_max,
    int64_t ch_axis, bool per_row_fake_quant, bool symmetric_quant);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_FAKE_QUANTIZE_FUSED_MOVING_AVG_OBS_FQ_HELPER_ATEN_KERNELS_H_
