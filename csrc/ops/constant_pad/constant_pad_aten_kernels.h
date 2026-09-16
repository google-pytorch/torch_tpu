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

#ifndef TORCH_TPU_CSRC_OPS_CONSTANT_PAD_CONSTANT_PAD_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_CONSTANT_PAD_CONSTANT_PAD_ATEN_KERNELS_H_

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "c10/core/Scalar.h"

namespace torch_tpu {

// Implements aten::constant_pad_nd (torch.nn.functional.pad with
// mode='constant') on TPU by lowering directly to StableHLO slice and
// concatenate operations instead of falling back to PyTorch's default composite
// decomposition (allocating a full-sized output tensor filled with `value` and
// mutating its interior via DynamicUpdateSlice).
at::Tensor AtenConstantPadNd(const at::Tensor& self, at::IntArrayRef pad,
                             const at::Scalar& value);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_CONSTANT_PAD_CONSTANT_PAD_ATEN_KERNELS_H_
