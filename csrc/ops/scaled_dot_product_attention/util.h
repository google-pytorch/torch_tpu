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

#ifndef TORCH_TPU_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_UTIL_H_
#define TORCH_TPU_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_UTIL_H_

// Kiln building blocks shared by the flash attention forward and backward
// kernels. Everything here must be called inside a `kiln::CreateKernel` body.

#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "csrc/ops/scaled_dot_product_attention/flash_attention_config.h"

namespace torch_tpu {

// Validates the preconditions shared by all flash attention kernels.
absl::Status ValidateFlashAttnConfig(const FlashAttnConfig& config,
                                     const Tiling& tiling);

// -----------------------------------------------------------------------------
// Block specs
// -----------------------------------------------------------------------------

// Dropout needs a single `scalar_prefetch` operand: a 4-word SMEM buffer
// holding the Philox {seed_lo, seed_hi, offset_lo, offset_hi}. Returns
// `std::nullopt` when dropout is disabled.
std::optional<kiln::ScalarPrefetchSpec> DropoutSeedSpec(
    const FlashAttnConfig& config);

// Index map for the K / V blocks on a (batch, q_head, q_block, kv_block) grid.
// Handles GQA / MQA (kv_head = q_head / head_group, which needs a floordiv and
// therefore a function map) and, for causal attention, points skipped blocks
// above the diagonal at block 0 so the pipeline prefetches what the next q row
// will need.
kiln::IndexMap KvBlockIndexMap(const FlashAttnConfig& config,
                               const Tiling& tiling);

// Block shape of the bias: (1, 1, qt, kt), with 1 instead of qt / kt along a
// broadcast query / key dim (e.g. a [B, 1, 1, S] key padding mask), which
// `LoadBias` then broadcasts in VMEM.
kiln::Shape BiasBlockShape(const FlashAttnConfig& config, const Tiling& tiling,
                           kiln::DType dtype);

// Input spec for the user attention bias on a (batch, q_head, q_block,
// kv_block) grid, honouring `config.mask_broadcast_dims`. Returns
// `std::nullopt` when there is no bias.
std::optional<kiln::InSpec> AttnBiasSpec(const FlashAttnConfig& config,
                                         const Tiling& tiling,
                                         kiln::DType dtype);

// -----------------------------------------------------------------------------
// Kernel body helpers
// -----------------------------------------------------------------------------

// Loads the f32 (rows, cols) bias sub-block at (row_offset, col_offset),
// broadcasting a size-1 query / key dim (see `BiasBlockShape`).
kiln::Tile LoadBias(kiln::Ref bias_ref, const FlashAttnConfig& config,
                    int64_t row_offset, int64_t rows, int64_t col_offset,
                    int64_t cols);

// Whether any sequence boundary masking is required for the given
// configuration.
bool NeedsPadMask(const FlashAttnConfig& config, const Tiling& tiling);

// True when the (row_block, col_block) tile pair touches or lies below the
// diagonal, i.e. the tile contributes to a causal softmax.
kiln::Scalar BelowOrOnDiagonal(kiln::Scalar row_block, kiln::Scalar col_block,
                               const Tiling& tiling);

// True when the (row_block, col_block) tile contains at least one element
// strictly above the diagonal, i.e. whether a causal mask needs to be applied
// to it. Tiles strictly below the diagonal need no masking.
kiln::Scalar IntersectsDiagonal(kiln::Scalar row_block, kiln::Scalar col_block,
                                const Tiling& tiling);

// Masks out logits above the diagonal (causal) and in the padded tail of
// either sequence for the sub-block starting at (row_offset, col_offset)
// within the (row_block, col_block) grid tile. Returns `s` untouched when no
// masking is required.
kiln::Tile ApplyStructuredMask(kiln::Tile s, kiln::Scalar row_block,
                               kiln::Scalar col_block,
                               const FlashAttnConfig& config,
                               const Tiling& tiling, int64_t row_offset = 0,
                               int64_t col_offset = 0);

// log2-scaled logits for one (q, k) tile pair:
//   s = (q @ k^T) * scale * log2(e) [+ bias * log2(e)]
// with causal / sequence-padding masking applied. Returns an f32 (qt, kt) tile.
kiln::Tile ComputeLogits(kiln::Tile q, kiln::Tile k,
                         std::optional<kiln::Ref> bias_ref,
                         kiln::Scalar row_block, kiln::Scalar col_block,
                         const FlashAttnConfig& config, const Tiling& tiling);

// Builds a stateless, tile-size-invariant dropout keep mask: an i1 (rows, cols)
// tile (true = keep, false = drop) for the sub-block starting at
// (row_offset, col_offset) within the [qt, kt] tile. When `rows` or `cols` are
// <= 0 they default to `qt` and `kt` respectively.
//
// The mask is a pure function of (batch, q_head, absolute q, absolute kv),
// `config.kv_sequence_length`, `config.dropout_rate` and the Philox
// {seed, offset} in `seed_ref` (memref<4xi32, smem>), so the backward kernel
// regenerates it rather than storing it. It does not depend on the tiling: the
// forward and backward kernels can pick different tile sizes, and so pad the
// sequences to different lengths, but they share the unpadded lengths and drop
// the same positions.
//
// `q_head` must be the *query* head index.
kiln::Tile DropoutKeepMask(const FlashAttnConfig& config, const Tiling& tiling,
                           kiln::Scalar batch, kiln::Scalar q_head,
                           kiln::Scalar row_block, kiln::Scalar col_block,
                           kiln::Ref seed_ref, int64_t row_offset = 0,
                           int64_t col_offset = 0, int64_t rows = 0,
                           int64_t cols = 0);

// Inverted dropout: kept entries of `t` are rescaled by 1 / (1 - p), dropped
// entries are zeroed.
kiln::Tile ApplyDropout(kiln::Tile t, kiln::Tile keep,
                        const FlashAttnConfig& config);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_UTIL_H_
