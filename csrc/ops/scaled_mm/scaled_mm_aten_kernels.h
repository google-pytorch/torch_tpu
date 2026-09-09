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

#ifndef TORCH_TPU_CSRC_OPS_SCALED_MM_SCALED_MM_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_SCALED_MM_SCALED_MM_ATEN_KERNELS_H_

#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "csrc/common/macro_utils.h"
#include "csrc/common/utils.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

at::Tensor AtenScaledMm(const at::Tensor& self, const at::Tensor& mat2,
                        const at::Tensor& scale_a, const at::Tensor& scale_b,
                        const std::optional<at::Tensor>& bias,
                        const std::optional<at::Tensor>& scale_result,
                        std::optional<at::ScalarType> out_dtype,
                        bool use_fast_accum);

at::Tensor& AtenScaledMmOut(const at::Tensor& self, const at::Tensor& mat2,
                            const at::Tensor& scale_a,
                            const at::Tensor& scale_b,
                            const std::optional<at::Tensor>& bias,
                            const std::optional<at::Tensor>& scale_result,
                            std::optional<at::ScalarType> out_dtype,
                            bool use_fast_accum, at::Tensor& out);

// PyTorch 2.14 changed the C++ binding of `Tensor[]` arguments in
// aten::_scaled_mm_v2 from at::TensorList to at::ITensorListRef; the
// registered kernel's signature must match the torch it runs under, or
// registration aborts at library load. Internal builds always track
// torch head, so they take the >= 2.14 branch too.
#if TT_TORCH_VERSION_GE(2, 14)
at::Tensor AtenScaledMmV2(const at::Tensor& self, const at::Tensor& mat2,
                          const at::ITensorListRef& scale_a,
                          at::IntArrayRef recipe_a, at::IntArrayRef swizzle_a,
                          const at::ITensorListRef& scale_b,
                          at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b,
                          const std::optional<at::Tensor>& bias,
                          std::optional<at::ScalarType> out_dtype,
                          at::IntArrayRef contraction_dim, bool use_fast_accum);

at::Tensor& AtenScaledMmV2Out(
    const at::Tensor& self, const at::Tensor& mat2,
    const at::ITensorListRef& scale_a, at::IntArrayRef recipe_a,
    at::IntArrayRef swizzle_a, const at::ITensorListRef& scale_b,
    at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b,
    const std::optional<at::Tensor>& bias,
    std::optional<at::ScalarType> out_dtype, at::IntArrayRef contraction_dim,
    bool use_fast_accum, at::Tensor& out);
#else
at::Tensor AtenScaledMmV2(const at::Tensor& self, const at::Tensor& mat2,
                          at::TensorList scale_a, at::IntArrayRef recipe_a,
                          at::IntArrayRef swizzle_a, at::TensorList scale_b,
                          at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b,
                          const std::optional<at::Tensor>& bias,
                          std::optional<at::ScalarType> out_dtype,
                          at::IntArrayRef contraction_dim, bool use_fast_accum);

at::Tensor& AtenScaledMmV2Out(const at::Tensor& self, const at::Tensor& mat2,
                              at::TensorList scale_a, at::IntArrayRef recipe_a,
                              at::IntArrayRef swizzle_a, at::TensorList scale_b,
                              at::IntArrayRef recipe_b,
                              at::IntArrayRef swizzle_b,
                              const std::optional<at::Tensor>& bias,
                              std::optional<at::ScalarType> out_dtype,
                              at::IntArrayRef contraction_dim,
                              bool use_fast_accum, at::Tensor& out);
#endif

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_SCALED_MM_SCALED_MM_ATEN_KERNELS_H_
