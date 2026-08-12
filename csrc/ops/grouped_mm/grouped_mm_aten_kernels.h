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

#ifndef TORCH_TPU_CSRC_OPS_GROUPED_MM_GROUPED_MM_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_GROUPED_MM_GROUPED_MM_ATEN_KERNELS_H_

#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "csrc/common/macro_utils.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

at::Tensor AtenGroupedMm(const at::Tensor& self, const at::Tensor& mat2,
                         const std::optional<at::Tensor>& offs,
                         const std::optional<at::Tensor>& bias,
                         std::optional<at::ScalarType> out_dtype);

at::Tensor AtenScaledGroupedMm(const at::Tensor& self, const at::Tensor& mat2,
                               const at::Tensor& scale_a,
                               const at::Tensor& scale_b,
                               const std::optional<at::Tensor>& offs,
                               const std::optional<at::Tensor>& bias,
                               const std::optional<at::Tensor>& scale_result,
                               std::optional<at::ScalarType> out_dtype,
                               bool use_fast_accum);

// PyTorch 2.14 changed the C++ binding of `Tensor[]` arguments from
// at::TensorList to at::ITensorListRef; the registered kernel's signature
// must match the torch it runs under.
#if TT_TORCH_VERSION_GE(2, 14)
at::Tensor AtenScaledGroupedMmV2(
    const at::Tensor& self, const at::Tensor& mat2,
    const at::ITensorListRef& scale_a, at::IntArrayRef recipe_a,
    at::IntArrayRef swizzle_a, const at::ITensorListRef& scale_b,
    at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    std::optional<at::ScalarType> out_dtype, at::IntArrayRef contraction_dim,
    bool use_fast_accum);

at::Tensor& AtenScaledGroupedMmV2Out(
    const at::Tensor& self, const at::Tensor& mat2,
    const at::ITensorListRef& scale_a, at::IntArrayRef recipe_a,
    at::IntArrayRef swizzle_a, const at::ITensorListRef& scale_b,
    at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    std::optional<at::ScalarType> out_dtype, at::IntArrayRef contraction_dim,
    bool use_fast_accum, at::Tensor& out);
#else
at::Tensor AtenScaledGroupedMmV2(
    const at::Tensor& self, const at::Tensor& mat2, at::TensorList scale_a,
    at::IntArrayRef recipe_a, at::IntArrayRef swizzle_a, at::TensorList scale_b,
    at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    std::optional<at::ScalarType> out_dtype, at::IntArrayRef contraction_dim,
    bool use_fast_accum);

at::Tensor& AtenScaledGroupedMmV2Out(
    const at::Tensor& self, const at::Tensor& mat2, at::TensorList scale_a,
    at::IntArrayRef recipe_a, at::IntArrayRef swizzle_a, at::TensorList scale_b,
    at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    std::optional<at::ScalarType> out_dtype, at::IntArrayRef contraction_dim,
    bool use_fast_accum, at::Tensor& out);
#endif

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_GROUPED_MM_GROUPED_MM_ATEN_KERNELS_H_
