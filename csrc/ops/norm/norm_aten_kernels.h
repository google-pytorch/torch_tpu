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

#ifndef TORCH_TPU_CSRC_OPS_NORM_NORM_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_NORM_NORM_ATEN_KERNELS_H_

#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Scalar.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

// aten::norm.out
at::Tensor& AtenNormOut(const at::Tensor& self,
                        const std::optional<at::Scalar>& p, at::IntArrayRef dim,
                        bool keepdim, at::Tensor& out);

// aten::norm.dtype_out
at::Tensor& AtenNormDtypeOut(const at::Tensor& self,
                             const std::optional<at::Scalar>& p,
                             at::IntArrayRef dim, bool keepdim,
                             at::ScalarType dtype, at::Tensor& out);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_NORM_NORM_ATEN_KERNELS_H_
