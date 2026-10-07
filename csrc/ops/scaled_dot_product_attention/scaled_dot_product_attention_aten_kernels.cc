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

#include "csrc/ops/scaled_dot_product_attention/scaled_dot_product_attention_aten_kernels.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>

#include "ATen/Context.h"
#include "ATen/SDPBackend.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBase.h"
#include "ATen/ops/zeros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "c10/core/GradMode.h"
#include "c10/core/ScalarType.h"
#include "c10/core/SymInt.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/warnings.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/scaled_dot_product_attention/flash_attention_kernel_builder.h"
#include "csrc/ops/scaled_dot_product_attention/helpers.h"
#include "csrc/ops/scaled_dot_product_attention/scaled_dot_product_attention_shlo.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

struct SdpaKernelKey {
  at::ScalarType dtype = at::ScalarType::Undefined;
  bool is_causal = false;

  template <typename H>
  friend H AbslHashValue(H h, const SdpaKernelKey& k) {
    return H::combine(std::move(h), k.dtype, k.is_causal);
  }

  bool operator==(const SdpaKernelKey& other) const = default;
};

// Returns true if `tensor` has symbolic sizes (torch.compile dynamic shapes) or
// bounded dynamic dimensions on its device buffer (eager `mark_dynamic`).
bool HasDynamicShape(const at::Tensor& tensor) {
  if (!tensor.defined()) {
    return false;
  }
  if (tensor.unsafeGetTensorImpl()->has_symbolic_sizes_strides()) {
    return true;
  }
  // Wrapper tensors, such as the functorch wrappers that `torch.func`
  // transforms pass to this kernel, have no storage and throw on storage
  // access, so they have no device buffer to inspect.
  if (!tensor.has_storage()) {
    return false;
  }
  const absl::StatusOr<DeviceBufferRef> buf = GetBuffer(tensor);
  return buf.ok() && (!buf->dynamic_dimensions().empty() ||
                      buf->on_device_shape_is_dynamic());
}

bool IsSupportedFlashAttentionShape(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask = std::nullopt) {
  // Flash attention (Mosaic) does not support dynamic shapes / bounded
  // dynamism.
  if (HasDynamicShape(query) || HasDynamicShape(key) ||
      HasDynamicShape(value) ||
      (attn_mask.has_value() && HasDynamicShape(*attn_mask))) {
    return false;
  }

  // TODO(elliotenglish): Add support for attn_mask and attributes.
  constexpr int min_block_size = 128;
  constexpr int min_batch_size = 1;

  bool has_batch = query.ndimension() >= 4 &&
                   get_batch_size(query.sizes()) >= min_batch_size;
  bool valid_head_dim =
      (query.size(query.ndimension() - 1) < min_block_size ||
       query.size(query.ndimension() - 1) % min_block_size == 0);

  return has_batch && valid_head_dim;
}

}  // namespace

// Torch's extensibility is pretty gross here due to weird SDPA
// implementation.
// *  at::native::scaled_dot_product_attention calls _fused_sdp_choice_stub,
//    which then dispatches to either _fused_sdp_choice_cpp (for CPU) or
//    _fused_sdp_choice_cuda. This can be overridden by
//    REGISTER_PRIVATEUSE1_DISPATCH, which we do in tpu_aten_kernels.cc.
// *  However, these then use either sdp::select_sdp_backend_cpp (CPU) or
//    sdp::select_sdp_backend (CUDA). select_sdp_backend is *not* TORCH_API
//    overrideable, and these hardcode which backends are supported on
//    CPU and CUDA.
//
// So we have to inline the logic from sdp::select_sdp_backend* that accesses
// the at::globalContext() flags, and return either overrideable or math,
// depending on whether the arguments are supported by our current
// implementation.
//
// Also, to support easy migration for users, we only warn on an unsupported
// backend instead of erroring (we only error if *all* backends are disabled).
int64_t AtenFusedSdpChoice(const at::Tensor& query, const at::Tensor& key,
                           const at::Tensor& value,
                           const std::optional<at::Tensor>& attn_mask,
                           double dropout_p, bool is_causal,
                           std::optional<double> scale, bool enable_gqa) {
  TT_KERNEL(
      OpName::kFusedSdpChoice, _,
      (query, key, value, IgnoreInCacheKey(attn_mask, "Doesn't affect SHLO"),
       IgnoreInCacheKey(dropout_p, "Doesn't affect SHLO"),
       IgnoreInCacheKey(is_causal, "Unused"), IgnoreInCacheKey(scale, "Unused"),
       IgnoreInCacheKey(enable_gqa, "Unused")),
      {
        const auto& ctx = at::globalContext();
        bool flash_enabled = ctx.userEnabledFlashSDP();
        bool math_enabled = ctx.userEnabledMathSDP();
        bool overrideable_enabled = ctx.userEnabledOverrideableSDP();

        bool consistent_ranks = query.ndimension() == key.ndimension() &&
                                query.ndimension() == value.ndimension();
        bool has_batch = query.ndimension() >= 4;
        bool has_dropout = dropout_p != 0.0;
        bool valid_flash_dropout = dropout_p >= 0.0 && dropout_p < 1.0;
        bool mask_requires_grad =
            c10::GradMode::is_enabled() && attn_mask.has_value() &&
            attn_mask->defined() && attn_mask->requires_grad();

        if (flash_enabled) {
          bool supported_flash_dtype =
              (query.scalar_type() == at::ScalarType::Float ||
               query.scalar_type() == at::ScalarType::BFloat16);
          if (supported_flash_dtype &&
              IsSupportedFlashAttentionShape(query, key, value, attn_mask) &&
              consistent_ranks && valid_flash_dropout && !mask_requires_grad) {
            return static_cast<int64_t>(at::SDPBackend::cudnn_attention);
          } else {
            // Use c10::str rather than absl::StrCat as it will convert enums
            // into their string representation.
            TT_WARN_ONCE(
                WarningCategory::kPerformance,
                "TorchTPU only supports FLASH_ATTENTION SDPBackend for "
                "scaled_dot_product_attention when these conditions are met:\n"
                "- inputs have static shapes\n"
                "- dropout_p is in [0.0, 1.0) (current: ",
                dropout_p,
                ")\n"
                "- inputs are float32 or bfloat16 (current: ",
                query.scalar_type(),
                ")\n"
                "- inputs have the same rank (query: ",
                query.ndimension(), ", key: ", key.ndimension(),
                ", value: ", value.ndimension(),
                ")\n"
                "- query rank is at least 4 (current: ",
                query.ndimension(),
                ")\n"
                "- batch size is at least 1\n"
                "- Head dimension (dim - 1) is less than 128 or divisible by "
                "128\n"
                // `sym_sizes()` so that symbolic sizes under torch.compile are
                // printed without being specialized.
                "(query shape: ",
                query.sym_sizes(),
                ")\n"
                "- attn_mask does not require gradient (current: ",
                mask_requires_grad, ")");
          }
        }

        // We treat the SHLO implementation as an optimized version of MATH so
        // also try it when it is enabled.
        if (math_enabled || overrideable_enabled) {
          bool is_floating_type = c10::isFloatingType(query.scalar_type());
          if (consistent_ranks && has_batch && !has_dropout &&
              is_floating_type && !mask_requires_grad) {
            return static_cast<int64_t>(at::SDPBackend::overrideable);
          } else {
            TT_WARN_ONCE(
                WarningCategory::kPerformance,
                "TorchTPU only supports SHLO optimized MATH SDPBackend when "
                "these conditions are met:\n"
                "- dropout_p is 0.0 (current: ",
                dropout_p,
                ")\n"
                "- inputs are not complex (current: ",
                query.scalar_type(),
                ")\n"
                "- inputs have the same rank (query: ",
                query.ndimension(), ", key: ", key.ndimension(),
                ", value: ", value.ndimension(),
                ")\n"
                "- attn_mask does not require gradient (current: ",
                mask_requires_grad, ")");
          }
        }

        TT_CHECK_THROW(math_enabled, error::kFailedPrecondition)
            << "no viable SDPBackend found: all supported backends are "
               "disabled, including the fallback MATH backend; enable at "
               "least one of FLASH, OVERRIDEABLE, or MATH for "
               "TorchTPU";
        return static_cast<int64_t>(at::SDPBackend::math);
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt,
           c10::SymInt, at::Tensor, at::Tensor, at::Tensor>
GenerateResults(at::Tensor out, at::Tensor logsumexp,
                at::Tensor philox_seed = at::Tensor(),
                at::Tensor philox_offset = at::Tensor()) {
  int64_t batch = get_batch_size(out.sizes());
  at::Tensor cum_seq_q = at::zeros({batch + 1}, out.options().dtype(at::kInt));
  at::Tensor cum_seq_k = at::zeros({batch + 1}, out.options().dtype(at::kInt));
  if (!philox_seed.defined()) {
    philox_seed = at::zeros({1}, out.options().dtype(at::kLong));
    philox_offset = at::zeros({1}, out.options().dtype(at::kLong));
  }
  at::Tensor debug_mask = at::zeros({1}, out.options().dtype(at::kBool));

  return std::make_tuple(std::move(out), std::move(logsumexp),
                         std::move(cum_seq_q), std::move(cum_seq_k),
                         c10::SymInt(0), c10::SymInt(0), std::move(philox_seed),
                         std::move(philox_offset), std::move(debug_mask));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt,
           c10::SymInt, at::Tensor, at::Tensor, at::Tensor>
AtenScaledDotProductFusedAttentionOverrideable(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias, double dropout_p,
    bool is_causal, bool return_debug_mask, std::optional<double> scale) {
  TT_KERNEL(
      OpName::kScaledDotProductFusedAttentionOverrideable, param_keys,
      (query, key, value,
       IgnoreInCacheKey(attn_bias, "Tensors are implicitly fingerprinted."),
       IgnoreInCacheKey(dropout_p, "Unused"), is_causal,
       IgnoreInCacheKey(return_debug_mask, "Unused"), scale),
      {
        const bool allow_half_precision_reduction_math =
            at::globalContext().allowFP16BF16ReductionMathSDP();
        TT_THROW_IF_ERROR(param_keys.SetParam(
            "allow_fp16_bf16_reduction", allow_half_precision_reduction_math));

        TT_ASSIGN_OR_THROW(
            auto results,
            ScaledDotProductFusedAttentionShlo(
                query, key, value, attn_bias, is_causal, scale,
                allow_half_precision_reduction_math, std::move(param_keys)));
        auto [out, logsumexp] = results;
        return GenerateResults(out, logsumexp);
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
AtenScaledDotProductFusedAttentionOverrideableBackward(
    const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& attn_bias,
    std::array<bool, 4> grad_input_mask, const at::Tensor& out,
    const at::Tensor& logsumexp, const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k, at::SymInt max_q, at::SymInt max_k,
    double dropout_p, bool is_causal, const at::Tensor& philox_seed,
    const at::Tensor& philox_offset, std::optional<double> scale) {
  TT_KERNEL(
      OpName::kScaledDotProductFusedAttentionOverrideableBackward, param_keys,
      (grad_out, query, key, value, attn_bias,
       IgnoreInCacheKey(grad_input_mask, "Doesn't affect SHLO"), out, logsumexp,
       cum_seq_q, cum_seq_k, IgnoreInCacheKey(max_q, "Unused"),
       IgnoreInCacheKey(max_k, "Unused"), IgnoreInCacheKey(dropout_p, "Unused"),
       is_causal, philox_seed, philox_offset, scale),
      {
        const bool allow_half_precision_reduction_math =
            at::globalContext().allowFP16BF16ReductionMathSDP();
        TT_THROW_IF_ERROR(param_keys.SetParam(
            "allow_fp16_bf16_reduction", allow_half_precision_reduction_math));

        TT_ASSIGN_OR_THROW(
            auto out, ScaledDotProductFusedAttentionShloBackward(
                          grad_out, query, key, value, attn_bias, logsumexp,
                          scale, is_causal, allow_half_precision_reduction_math,
                          std::move(param_keys)));

        return std::make_tuple(
            grad_input_mask[0] ? std::get<0>(out)
                               : at::zeros({0}, grad_out.options()),
            grad_input_mask[1] ? std::get<1>(out)
                               : at::zeros({0}, grad_out.options()),
            grad_input_mask[2] ? std::get<2>(out)
                               : at::zeros({0}, grad_out.options()),
            at::zeros({0}, grad_out.options()));
      });
}

// This function implements the interface of scaled_dot_product_cudnn_attention
// See torch/aten/src/ATen/native/native_functions.yaml for arguments.
// See torch/_meta_registrations.py for layout.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt,
           c10::SymInt, at::Tensor, at::Tensor, at::Tensor>
AtenScaledDotProductFlashAttention(const at::Tensor& query,
                                   const at::Tensor& key,
                                   const at::Tensor& value,
                                   const std::optional<at::Tensor>& attn_bias,
                                   bool compute_log_sumexp, double dropout_p,
                                   bool is_causal, bool return_debug_mask,
                                   std::optional<double> scale) {
  TT_KERNEL(
      OpName::kScaledDotProductFlashAttention, _,
      (query, key, value,
       IgnoreInCacheKey(attn_bias, "Delegates to implementation"),
       IgnoreInCacheKey(compute_log_sumexp,
                        "Delegates to CreateFlashAttentionKernel"),
       IgnoreInCacheKey(dropout_p, "Delegates to implementation"),
       IgnoreInCacheKey(is_causal, "Delegates to implementation"),
       IgnoreInCacheKey(return_debug_mask, "Unused"),
       IgnoreInCacheKey(scale, "Delegates to implementation")),
      {
        TT_ASSIGN_OR_THROW(
            auto results,
            CreateFlashAttentionKernel(query, key, value, attn_bias, is_causal,
                                       scale, dropout_p, compute_log_sumexp));
        auto [out, logsumexp, philox_seed, philox_offset] = std::move(results);
        return GenerateResults(std::move(out), std::move(logsumexp),
                               std::move(philox_seed),
                               std::move(philox_offset));
      });
}

// This function implements the interface of
// _scaled_dot_product_cudnn_attention_backward
// See torch/aten/src/ATen/native/native_functions.yaml for arguments.
// See torch/_meta_registrations.py for layout.
std::tuple<at::Tensor, at::Tensor, at::Tensor>
AtenScaledDotProductFlashAttentionBackward(
    const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& out, const at::Tensor& logsumexp,
    const at::Tensor& philox_seed, const at::Tensor& philox_offset,
    const at::Tensor& attn_bias, const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k, at::SymInt max_q, at::SymInt max_k,
    double dropout_p, bool is_causal, std::optional<double> scale) {
  TT_KERNEL(
      OpName::kScaledDotProductFlashAttentionBackward, _,
      (grad_out, query, key, value, out, logsumexp, philox_seed, philox_offset,
       attn_bias, cum_seq_q, cum_seq_k, IgnoreInCacheKey(max_q, "Unused"),
       IgnoreInCacheKey(max_k, "Unused"),
       IgnoreInCacheKey(dropout_p, "Delegates to implementation"),
       IgnoreInCacheKey(is_causal, "Delegates to implementation"),
       IgnoreInCacheKey(scale, "Delegates to implementation")),
      {
        TT_ASSIGN_OR_THROW(
            auto out,
            CreateFlashAttentionBackwardKernel(
                grad_out, query, key, value, out, logsumexp, attn_bias, scale,
                is_causal, dropout_p, philox_seed, philox_offset));

        return out;
      });
}

}  // namespace torch_tpu
