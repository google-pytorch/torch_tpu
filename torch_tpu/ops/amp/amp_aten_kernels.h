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

#ifndef TORCH_TPU_OPS_AMP_AMP_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_AMP_AMP_ATEN_KERNELS_H_

#include <cstdint>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

void AtenAmpForeachNonFiniteCheckAndUnscale_(at::TensorList self,
                                             at::Tensor& found_inf,
                                             const at::Tensor& inv_scale);

at::Tensor& AtenAmpUpdateScale_(at::Tensor& self, at::Tensor& growth_tracker,
                                const at::Tensor& found_inf,
                                double scale_growth_factor,
                                double scale_backoff_factor,
                                int64_t growth_interval);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_AMP_AMP_ATEN_KERNELS_H_
