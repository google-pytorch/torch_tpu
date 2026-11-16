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

#ifndef TORCH_TPU_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_FLASH_ATTENTION_CONFIG_H_
#define TORCH_TPU_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_FLASH_ATTENTION_CONFIG_H_

#include <cstdint>
#include <numbers>
#include <optional>

#include "csrc/internal/mosaic/kiln/kiln.h"
#include "llvm/ADT/SmallVector.h"

namespace torch_tpu {

// Configuration for the flash attention kernel.
struct FlashAttnConfig {
  kiln::DType element_type = kiln::bf16;
  int64_t batch_size;
  int64_t num_heads;
  int64_t kv_num_heads;
  int64_t qk_head_dim;
  int64_t vo_head_dim;
  int64_t q_sequence_length;
  int64_t kv_sequence_length;
  int64_t padded_q_sequence_length;
  int64_t padded_kv_sequence_length;
  bool is_causal;
  float scale;
  // Broadcast dimensions of the mask.
  llvm::SmallVector<int64_t, 4> mask_broadcast_dims;
  // Whether to return LSE (m and l) tensors.
  bool return_lse;
  bool has_attn_bias;
  float dropout_rate = 0.0f;
  std::optional<kiln::ContractPrecision> precision = std::nullopt;
};

struct Tiling {
  // Q / KV block sizes that are DMA'd per grid step.
  int64_t qt;
  int64_t kt;
  // KV sub-block size the forward kernel computes on within a grid step. This
  // decouples the (large) DMA block, which amortizes the per grid step
  // overhead, from the (smaller) compute block, which bounds the size of the
  // [qt, kc] softmax intermediates. Must divide kt.
  int64_t kc;
};

constexpr int64_t kDefaultQTileSize = 512;
constexpr int64_t kDefaultKTileSize = 512;

// The forward and backward kernels evaluate the softmax in the log2 domain:
// exp(x) == exp2(x * log2(e)). Folding log2(e) into the logit scale lets us
// use the cheaper exp2 / log2 EUP ops. As a consequence the LSE returned by the
// forward kernel (and consumed by the backward kernel) is log2-sum-exp2, i.e.
// the natural LSE multiplied by log2(e).
constexpr float kLog2e = std::numbers::log2e_v<float>;

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_FLASH_ATTENTION_CONFIG_H_
