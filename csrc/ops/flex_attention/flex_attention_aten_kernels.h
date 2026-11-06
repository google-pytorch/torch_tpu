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

#ifndef TORCH_TPU_CSRC_OPS_FLEX_ATTENTION_FLEX_ATTENTION_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_FLEX_ATTENTION_FLEX_ATTENTION_ATEN_KERNELS_H_

#include <cstdint>
#include <string_view>
#include <tuple>

#include "ATen/core/TensorBody.h"

namespace torch_tpu {

// torch.ops.tpu._flex_attention_fwd. `kv_num_blocks` / `kv_indices` are the
// BlockMask tables flattened to int32 1-D. `score_mod_mlir` / `mask_mod_mlir`
// are the mods as StableHLO text (see flex_attention.mod_to_mlir), empty
// for none. Returns (out in q's dtype, lse f32 [B, Hq, 1, T] in log2 units).
std::tuple<at::Tensor, at::Tensor> FlexAttentionFwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& kv_num_blocks, const at::Tensor& kv_indices,
    int64_t mask_batch, int64_t mask_heads, int64_t q_block, int64_t kv_block,
    double scale, std::string_view score_mod_mlir,
    std::string_view mask_mod_mlir);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_FLEX_ATTENTION_FLEX_ATTENTION_ATEN_KERNELS_H_
