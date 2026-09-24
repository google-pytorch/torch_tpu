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

#ifndef TORCH_TPU_CSRC_OPS_EXPERIMENTAL_DIMENSIONAL_COLLECTIVES_DIMENSIONAL_COLLECTIVES_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_EXPERIMENTAL_DIMENSIONAL_COLLECTIVES_DIMENSIONAL_COLLECTIVES_ATEN_KERNELS_H_

#include <cstdint>
#include <string_view>

#include "ATen/core/TensorBody.h"

namespace torch_tpu {

at::Tensor ExperimentalAllGatherDim(const at::Tensor& input, int64_t dim,
                                    int64_t group_size,
                                    std::string_view group_name);

at::Tensor ExperimentalReduceScatterDim(const at::Tensor& input, int64_t dim,
                                        int64_t group_size,
                                        std::string_view group_name);

at::Tensor MetaExperimentalAllGatherDim(const at::Tensor& input, int64_t dim,
                                        int64_t group_size,
                                        std::string_view group_name);

at::Tensor MetaExperimentalReduceScatterDim(const at::Tensor& input,
                                            int64_t dim, int64_t group_size,
                                            std::string_view group_name);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_EXPERIMENTAL_DIMENSIONAL_COLLECTIVES_DIMENSIONAL_COLLECTIVES_ATEN_KERNELS_H_
