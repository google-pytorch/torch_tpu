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

#ifndef TORCH_TPU_OPS_MASKED_FILL_MASKED_FILL_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_MASKED_FILL_MASKED_FILL_ATEN_KERNELS_H_
#include "ATen/core/ATen_fwd.h"

namespace torch_tpu {

// masked_fill_.Scalar
at::Tensor& AtenMaskedFill_Scalar(at::Tensor& self, const at::Tensor& mask,
                                  const at::Scalar& value);

// masked_fill_.Tensor
at::Tensor& AtenMaskedFill_Tensor(at::Tensor& self, const at::Tensor& mask,
                                  const at::Tensor& value);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_MASKED_FILL_MASKED_FILL_ATEN_KERNELS_H_
