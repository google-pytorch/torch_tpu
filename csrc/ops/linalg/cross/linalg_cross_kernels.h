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

#ifndef TORCH_TPU_CSRC_OPS_LINALG_CROSS_LINALG_CROSS_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_LINALG_CROSS_LINALG_CROSS_KERNELS_H_

#include <cstdint>

#include "ATen/core/ATen_fwd.h"

namespace torch_tpu {

// Computes the cross product of 3D vectors `self` and `other` along `dim`
// and writes the result into `out`.
at::Tensor& AtenLinalgCrossOut(const at::Tensor& self, const at::Tensor& other,
                               int64_t dim, at::Tensor& out);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_LINALG_CROSS_LINALG_CROSS_KERNELS_H_
