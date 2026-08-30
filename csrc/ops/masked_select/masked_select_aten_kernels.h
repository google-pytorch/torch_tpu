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

#ifndef TORCH_TPU_OPS_MASKED_SELECT_MASKED_SELECT_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_MASKED_SELECT_MASKED_SELECT_ATEN_KERNELS_H_

#include "ATen/core/TensorBody.h"

namespace torch_tpu {

// This op is implemented by first calculating and materializing the output
// shape then restarting tracing with a known data-dependent output shape
//
// This method is equivalent to:
//    torch.masked_select(x, mask, output_size=mask.sum().cpu())
at::Tensor AtenMaskedSelect(const at::Tensor& self, const at::Tensor& mask);

// This op is implemented by first calculating and materializing the output
// shape then restarting tracing with a known data-dependent output shape
//
// This method is equivalent to:
//    torch.masked_select(x, mask, output_size=mask.sum().cpu(), out=out)
at::Tensor& AtenMaskedSelectOut(const at::Tensor& self, const at::Tensor& mask,
                                at::Tensor& out);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_MASKED_SELECT_MASKED_SELECT_ATEN_KERNELS_H_
