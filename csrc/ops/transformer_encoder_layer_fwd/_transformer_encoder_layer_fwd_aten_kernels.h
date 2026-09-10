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

#ifndef TORCH_TPU_CSRC_OPS_TRANSFORMER_ENCODER_LAYER_FWD__TRANSFORMER_ENCODER_LAYER_FWD_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_TRANSFORMER_ENCODER_LAYER_FWD__TRANSFORMER_ENCODER_LAYER_FWD_ATEN_KERNELS_H_

#include <cstdint>
#include <optional>

#include "ATen/core/TensorBody.h"

namespace torch_tpu {

at::Tensor AtenTransformerEncoderLayerFwd(
    const at::Tensor& src, int64_t embed_dim, int64_t num_heads,
    const at::Tensor& qkv_weight, const at::Tensor& qkv_bias,
    const at::Tensor& proj_weight, const at::Tensor& proj_bias, bool use_gelu,
    bool norm_first, double eps, const at::Tensor& norm_weight_1,
    const at::Tensor& norm_bias_1, const at::Tensor& norm_weight_2,
    const at::Tensor& norm_bias_2, const at::Tensor& ffn_weight_1,
    const at::Tensor& ffn_bias_1, const at::Tensor& ffn_weight_2,
    const at::Tensor& ffn_bias_2, const std::optional<at::Tensor>& mask,
    std::optional<int64_t> mask_type);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_TRANSFORMER_ENCODER_LAYER_FWD__TRANSFORMER_ENCODER_LAYER_FWD_ATEN_KERNELS_H_
