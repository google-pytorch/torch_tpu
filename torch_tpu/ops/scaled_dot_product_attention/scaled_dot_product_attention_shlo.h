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

#ifndef TORCH_TPU_OPS_SCALED_DOT_PRODUCT_ATTENTION_KERNELS_SCALED_DOT_PRODUCT_ATTENTION_SHLO_H_
#define TORCH_TPU_OPS_SCALED_DOT_PRODUCT_ATTENTION_KERNELS_SCALED_DOT_PRODUCT_ATTENTION_SHLO_H_

#include <optional>
#include <tuple>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"

namespace torch_tpu {

absl::StatusOr<std::pair<at::Tensor, at::Tensor>>
ScaledDotProductFusedAttentionShlo(const at::Tensor& query,
                                   const at::Tensor& key,
                                   const at::Tensor& value,
                                   const std::optional<at::Tensor>& attn_bias,
                                   bool is_causal, std::optional<double> scale);

absl::StatusOr<std::tuple<at::Tensor, at::Tensor, at::Tensor>>
ScaledDotProductFusedAttentionShloBackward(
    const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& attn_bias,
    const at::Tensor& sum_exp, std::optional<double> scale, bool is_causal);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_SCALED_DOT_PRODUCT_ATTENTION_KERNELS_SCALED_DOT_PRODUCT_ATTENTION_SHLO_H_
