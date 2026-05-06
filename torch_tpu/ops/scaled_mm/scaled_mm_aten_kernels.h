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

#ifndef TORCH_TPU_OPS_SCALED_MM_SCALED_MM_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_SCALED_MM_SCALED_MM_ATEN_KERNELS_H_

#include <optional>

#include "ATen/core/TensorBody.h"
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

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_SCALED_MM_SCALED_MM_ATEN_KERNELS_H_
