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

#ifndef TORCH_TPU_OPS_UNFOLD_UNFOLD_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_UNFOLD_UNFOLD_ATEN_KERNELS_H_

#include <cstdint>

#include "ATen/core/ATen_fwd.h"

namespace torch_tpu {

at::Tensor AtenUnfold(const at::Tensor& self, int64_t dimension, int64_t size,
                      int64_t step);

at::Tensor AtenUnfoldBackward(const at::Tensor& grad_in,
                              at::IntArrayRef input_sizes, int64_t dimension,
                              int64_t size, int64_t step);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_UNFOLD_UNFOLD_ATEN_KERNELS_H_
