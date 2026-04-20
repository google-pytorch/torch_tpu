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

#ifndef TORCH_TPU_OPS_CTC_LOSS_CTC_LOSS_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_CTC_LOSS_CTC_LOSS_ATEN_KERNELS_H_

#include <cstdint>
#include <tuple>

#include "ATen/core/ATen_fwd.h"

namespace torch_tpu {

std::tuple<at::Tensor, at::Tensor> AtenCtcLoss(const at::Tensor& log_probs,
                                               const at::Tensor& targets,
                                               at::IntArrayRef input_lengths,
                                               at::IntArrayRef target_lengths,
                                               int64_t blank,
                                               bool zero_infinity);

std::tuple<at::Tensor, at::Tensor> AtenCtcLossTensor(
    const at::Tensor& log_probs, const at::Tensor& targets,
    const at::Tensor& input_lengths, const at::Tensor& target_lengths,
    int64_t blank, bool zero_infinity);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_CTC_LOSS_CTC_LOSS_ATEN_KERNELS_H_
