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

#ifndef TORCH_TPU_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_FLASH_ATTENTION_KERNEL_BUILDER_H_
#define TORCH_TPU_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_FLASH_ATTENTION_KERNEL_BUILDER_H_

#include <cstdint>
#include <optional>
#include <tuple>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBase.h"
#include "absl/status/statusor.h"
// IWYU pragma: begin_keep
#include "ATen/core/Tensor.h"
// IWYU pragma: end_keep

namespace torch_tpu {

// TPU MXU/VPU hardware lane width; the FlashAttention kernel pads any shorter
// sequence dimension up to 128.
inline constexpr int64_t kFlashMinHardwareTileSize = 128;

// Minimum block size produced by CreateTiling's splitting loop; sequences at
// or below 256 always execute as a single 1x1 block.
inline constexpr int64_t kFlashMinSplitTileSize = 256;

// Returns 50% of on-chip VMEM capacity in bytes for the active TPU generation
// (or 50% of the conservative 16 MB fallback when TPU device attributes are
// unavailable).
[[nodiscard]] int64_t GetFlashDeviceVmemLimitBytes();

absl::StatusOr<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>>
CreateFlashAttentionKernel(const at::Tensor& query, const at::Tensor& key,
                           const at::Tensor& value,
                           const std::optional<at::Tensor>& attn_bias,
                           bool is_causal, std::optional<double> scale,
                           double dropout_p = 0.0,
                           std::optional<bool> return_lse = std::nullopt);

absl::StatusOr<std::tuple<at::Tensor, at::Tensor, at::Tensor>>
CreateFlashAttentionBackwardKernel(
    const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& out, const at::Tensor& logsumexp,
    const std::optional<at::Tensor>& attn_bias, std::optional<double> scale,
    bool is_causal, double dropout_p = 0.0,
    const std::optional<at::Tensor>& philox_seed = std::nullopt,
    const std::optional<at::Tensor>& philox_offset = std::nullopt);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_FLASH_ATTENTION_KERNEL_BUILDER_H_
