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

#ifndef TORCH_TPU_OPS_REDUCTIONS_VAR_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_REDUCTIONS_VAR_ATEN_KERNELS_H_

#include <cstdint>
#include <optional>
#include <tuple>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "c10/util/OptionalArrayRef.h"

namespace torch_tpu {

// Implements aten::var.correction.
at::Tensor AtenVar(const at::Tensor& self, c10::OptionalArrayRef<int64_t> dim,
                   const std::optional<at::Scalar>& correction, bool keep_dim);

// Implements aten::var.correction_out.
at::Tensor& AtenVarOut(const at::Tensor& self,
                       c10::OptionalArrayRef<int64_t> dim,
                       const std::optional<at::Scalar>& correction,
                       bool keep_dim, at::Tensor& out);

// Implements aten::var_mean.correction.
std::tuple<at::Tensor, at::Tensor> AtenVarMeanCorrection(
    const at::Tensor& self, c10::OptionalArrayRef<int64_t> dim,
    const std::optional<at::Scalar>& correction, bool keep_dim);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_REDUCTIONS_VAR_ATEN_KERNELS_H_
