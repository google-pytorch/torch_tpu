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

#ifndef TORCH_TPU_OPS_NATIVE_NORM_NATIVE_NORM_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_NATIVE_NORM_NATIVE_NORM_ATEN_KERNELS_H_

#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Scalar.h"
#include "ATen/core/TensorBody.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

// Computes the vector p-norm of all elements in the input tensor `self`.
at::Tensor AtenNativeNorm(const at::Tensor& self, const at::Scalar& p);

// Computes the vector p-norm of `self` over the specified dimensions `dim`,
// retaining reduced dimensions if `keepdim` is true, with an optional
// output dtype conversion.
at::Tensor AtenNativeNormScalarOptDimDtype(const at::Tensor& self,
                                           const std::optional<at::Scalar>& p,
                                           at::IntArrayRef dim, bool keepdim,
                                           std::optional<at::ScalarType> dtype);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_NATIVE_NORM_NATIVE_NORM_ATEN_KERNELS_H_
