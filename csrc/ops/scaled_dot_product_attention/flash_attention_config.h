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

#include "llvm/ADT/SmallVector.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace mlir::torch_tpu {

// Configuration for the flash attention kernel.
struct FlashAttnConfig {
  ElementType element_type;
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
};

struct Tiling {
  int64_t qt;
  int64_t kt;
};

constexpr int64_t kDefaultQTileSize = 512;
constexpr int64_t kDefaultKTileSize = 512;

}  // namespace mlir::torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_FLASH_ATTENTION_CONFIG_H_
