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
#ifndef TORCH_TPU_OPS_DISTANCE_DIST_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_DISTANCE_DIST_ATEN_KERNELS_H_

#include <cstdint>
#include <optional>

#include "ATen/core/TensorBody.h"

namespace torch_tpu {

at::Tensor AtenCdistForward(const at::Tensor& x1, const at::Tensor& x2,
                            double p, std::optional<int64_t> compute_mode);
at::Tensor AtenCdistBackward(const at::Tensor& grad, const at::Tensor& x1,
                             const at::Tensor& x2, double p,
                             const at::Tensor& cdist);
at::Tensor AtenPdistForward(const at::Tensor& self, double p);
at::Tensor AtenPdistBackward(const at::Tensor& grad, const at::Tensor& self,
                             double p, const at::Tensor& pdist);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_DISTANCE_DIST_ATEN_KERNELS_H_
