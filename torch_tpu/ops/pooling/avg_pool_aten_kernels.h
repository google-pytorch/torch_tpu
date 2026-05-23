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

#ifndef TORCH_TPU_OPS_POOLING_AVG_POOL_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_POOLING_AVG_POOL_ATEN_KERNELS_H_

#include <cstdint>
#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

// Helper function to build and dispatch N-dimensional average pooling.
// It is exposed for use in adaptive average pooling, which supports a different
// set of dtypes than standard average pooling and thus cannot always fall back
// to AtenAvgPoolNd.
absl::StatusOr<at::Tensor> BuildAvgPoolOutNd(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override, at::Tensor& out,
    int64_t spatial_dim_count, OpParamCacheKeys param_keys);

at::Tensor& AtenAvgPool2dOut(const at::Tensor& self,
                             at::IntArrayRef kernel_size,
                             at::IntArrayRef stride, at::IntArrayRef padding,
                             bool ceil_mode, bool count_include_pad,
                             std::optional<int64_t> divisor_override,
                             at::Tensor& out);

at::Tensor& AtenAvgPool3dOut(const at::Tensor& self,
                             at::IntArrayRef kernel_size,
                             at::IntArrayRef stride, at::IntArrayRef padding,
                             bool ceil_mode, bool count_include_pad,
                             std::optional<int64_t> divisor_override,
                             at::Tensor& out);

at::Tensor& AtenAvgPool2dBackwardGradInput(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override, at::Tensor& grad_input);

at::Tensor& AtenAvgPool3dBackwardGradInput(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override, at::Tensor& grad_input);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_POOLING_AVG_POOL_ATEN_KERNELS_H_
