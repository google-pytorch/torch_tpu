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

#ifndef TORCH_TPU_OPS_SDPA_SDPA_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_SDPA_SDPA_ATEN_KERNELS_H_

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

// EXPLANATION:
//
// aten::scaled_dot_product_attention is an unusual kernel, in that it supports
// multiple "backends" (implementations), even on the same device type.
//
// For example, on CUDA there is both "_scaled_dot_product_flash_attention_cuda"
// and "_scaled_dot_product_cudnn_attention". The user can select an SDPA
// implementation by setting a global "SDPBackend" enum using a Python context
// manager. There is also a universal "math" backend that decomposes to an
// unfused implementation.
//
// These different "SDPBackend"s have different forward and backward kernels,
// and are not interchangeable. Therefore, there is no single "backwards" pass
// for aten::scaled_dot_product_attention_backward.
//
// The intended extension point for adding new SDPA backends is via the
// "aten::_scaled_dot_product_fused_attention_overrideable" and
// "aten::_scaled_dot_product_fused_attention_overrideable_backward"
// functions, which are used by "SDPBackend::OVERRIDEABLE". We implement these
// methods here to reuse the existing Autograd codepaths. These functions are
// extremely general; we ignore many arguments that are unused.
//
// However, for TPU, we only support two backends: "math" (logical,
// non-optimized) and our implementation using Pallas. To prevent users from
// needing to manually set the backend to "SDPBackend::OVERRIDEABLE", we also
// override the aten::_fused_sdp_choice function to force a call
// to our custom kernel on any non-"math" backend.

int64_t AtenFusedSdpChoice(const at::Tensor& query, const at::Tensor& key,
                           const at::Tensor& value,
                           const std::optional<at::Tensor>& attn_mask,
                           double dropout_p, bool is_causal,
                           std::optional<double> scale, bool enable_gqa);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt,
           c10::SymInt, at::Tensor, at::Tensor, at::Tensor>
AtenScaledDotProductFusedAttentionOverrideable(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias, double dropout_p,
    bool is_causal, bool return_debug_mask, std::optional<double> scale);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
AtenScaledDotProductFusedAttentionOverrideableBackward(
    const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& attn_bias,
    std::array<bool, 4> grad_input_mask, const at::Tensor& out,
    const at::Tensor& logsumexp, const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k, at::SymInt max_q, at::SymInt max_k,
    double dropout_p, bool is_causal, const at::Tensor& philox_seed,
    const at::Tensor& philox_offset, std::optional<double> scale);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
AtenScaledDotProductEfficientAttention(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias, bool compute_log_sumexp,
    double dropout_p, bool is_causal, std::optional<double> scale);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
AtenScaledDotProductEfficientAttentionBackward(
    const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& attn_bias, const at::Tensor& out,
    const at::Tensor& logsumexp, const at::Tensor& philox_seed,
    const at::Tensor& philox_offset, double dropout_p,
    std::array<bool, 4> grad_input_mask, bool is_causal,
    std::optional<double> scale);

std::tuple<at::Tensor, at::Tensor> AtenScaledDotProductFlashAttention(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    double dropout_p, bool is_causal,
    const std::optional<at::Tensor>& attn_mask, std::optional<double> scale);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
AtenScaledDotProductFlashAttentionBackward(
    const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& out, const at::Tensor& logsumexp,
    double dropout_p, bool is_causal,
    const std::optional<at::Tensor>& attn_mask, std::optional<double> scale);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_SDPA_SDPA_ATEN_KERNELS_H_
