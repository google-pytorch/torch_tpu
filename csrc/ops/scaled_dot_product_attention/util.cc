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

#include "csrc/ops/scaled_dot_product_attention/util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include "csrc/common/error_utils.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "csrc/ops/scaled_dot_product_attention/flash_attention_config.h"
#include "llvm/ADT/STLExtras.h"

namespace torch_tpu {

namespace {

// MurmurHash3's fmix32 avalanche; see https://en.wikipedia.org/wiki/MurmurHash.
// It is a bijection on uint32, so the packed dropout key cannot collide. The
// Kiln operators are shape-polymorphic, so this runs unchanged on the scalar
// (stream) and tile (per-element) paths.
template <typename T>
T Fmix32(T h) {
  h = h.AsType(kiln::u32);
  h ^= h >> 16;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13;
  h *= 0xC2B2AE35u;
  h ^= h >> 16;
  return h.AsType(kiln::i32);
}

}  // namespace

absl::Status ValidateFlashAttnConfig(const FlashAttnConfig& config,
                                     const Tiling& tiling) {
  if (config.padded_q_sequence_length % tiling.qt != 0 ||
      config.padded_kv_sequence_length % tiling.kt != 0) {
    return TT_ERROR(  // ERROR_COV_INFEASIBLE=Checked by caller.
               error::kInvalidArgument)
           << "Padded sequence lengths must be divisible by tile sizes.";
  }
  if (config.num_heads % config.kv_num_heads != 0) {
    return TT_ERROR(  // ERROR_COV_INFEASIBLE=Checked by caller.
               error::kInvalidArgument)
           << "num_heads must be divisible by kv_num_heads.";
  }
  if (config.dropout_rate < 0.0f || config.dropout_rate >= 1.0f) {
    return TT_ERROR(  // ERROR_COV_INFEASIBLE=Checked by caller.
               error::kInvalidArgument)
           << "dropout_rate must be in [0.0, 1.0), got " << config.dropout_rate;
  }
  return absl::OkStatus();
}

std::optional<kiln::ScalarPrefetchSpec> DropoutSeedSpec(
    const FlashAttnConfig& config) {
  if (config.dropout_rate <= 0.0f) {
    return std::nullopt;
  }
  return kiln::ScalarPrefetchSpec{kiln::i32(4)};
}

kiln::IndexMap KvBlockIndexMap(const FlashAttnConfig& config,
                               const Tiling& tiling) {
  const int64_t head_group = config.num_heads / config.kv_num_heads;
  return kiln::IndexMap([head_group, is_causal = config.is_causal,
                         tiling]() -> kiln::IndexIndices {
    kiln::Scalar kv_head = kiln::ProgramId(1) / head_group;
    kiln::Scalar kv_block = kiln::ProgramId(3);
    if (is_causal) {
      kv_block = Select(BelowOrOnDiagonal(kiln::ProgramId(2), kv_block, tiling),
                        kv_block, 0);
    }
    return {kiln::ProgramId(0), kv_head, kv_block, 0};
  });
}

kiln::Shape BiasBlockShape(const FlashAttnConfig& config, const Tiling& tiling,
                           kiln::DType dtype) {
  // Pinning the block index to 0 (see AttnBiasSpec) only broadcasts a dim
  // whose block is 1 wide.
  const bool q_bcast = llvm::is_contained(config.mask_broadcast_dims, 2);
  const bool kv_bcast = llvm::is_contained(config.mask_broadcast_dims, 3);
  return dtype(1, 1, q_bcast ? 1 : tiling.qt, kv_bcast ? 1 : tiling.kt);
}

std::optional<kiln::InSpec> AttnBiasSpec(const FlashAttnConfig& config,
                                         const Tiling& tiling,
                                         kiln::DType dtype) {
  if (!config.has_attn_bias) {
    return std::nullopt;
  }
  kiln::IndexMap bias_map = kiln::IndexMap(
      [dims = config.mask_broadcast_dims]() -> kiln::IndexIndices {
        kiln::IndexIndices idx{kiln::ProgramId(0), kiln::ProgramId(1),
                               kiln::ProgramId(2), kiln::ProgramId(3)};
        for (int64_t d : dims) idx[d] = 0;
        return idx;
      });
  return kiln::InSpec{BiasBlockShape(config, tiling, dtype), bias_map};
}

kiln::Tile LoadBias(kiln::VmemRef bias_ref, const FlashAttnConfig& config,
                    int64_t row_offset, int64_t rows, int64_t col_offset,
                    int64_t cols) {
  const bool q_bcast = llvm::is_contained(config.mask_broadcast_dims, 2);
  const bool kv_bcast = llvm::is_contained(config.mask_broadcast_dims, 3);
  const kiln::SliceDim row_slice =
      q_bcast ? kiln::Slice(0, 1) : kiln::Slice(row_offset, rows);
  const kiln::SliceDim col_slice =
      kv_bcast ? kiln::Slice(0, 1) : kiln::Slice(col_offset, cols);
  kiln::Tile bias = kiln::Load(bias_ref(0, 0, row_slice, col_slice));
  return bias.AsType(kiln::f32).BroadcastTo({rows, cols});
}

bool NeedsPadMask(const FlashAttnConfig& config, const Tiling& tiling) {
  return (config.q_sequence_length % tiling.qt) != 0 ||
         (config.kv_sequence_length % tiling.kt) != 0;
}

kiln::Scalar BelowOrOnDiagonal(kiln::Scalar row_block, kiln::Scalar col_block,
                               const Tiling& t) {
  return (row_block + 1) * t.qt - 1 >= col_block * t.kt;
}

kiln::Scalar IntersectsDiagonal(kiln::Scalar row_block, kiln::Scalar col_block,
                                const Tiling& t) {
  return (col_block + 1) * t.kt - 1 > row_block * t.qt;
}

kiln::Tile ApplyStructuredMask(kiln::Tile s, kiln::Scalar row_block,
                               kiln::Scalar col_block,
                               const FlashAttnConfig& config, const Tiling& t,
                               int64_t row_offset, int64_t col_offset) {
  const bool ragged_q = config.q_sequence_length % t.qt != 0;
  const bool ragged_kv = config.kv_sequence_length % t.kt != 0;
  if (!config.is_causal && !ragged_q && !ragged_kv) {
    return s;
  }

  const kiln::Shape i32_shape(s.dims(), kiln::i32);
  kiln::Scalar row_start = row_block * t.qt;
  if (row_offset != 0) {
    row_start += row_offset;
  }
  kiln::Scalar col_start = col_block * t.kt;
  if (col_offset != 0) {
    col_start += col_offset;
  }

  kiln::Tile row = kiln::Iota(i32_shape, 0) + row_start;
  kiln::Tile col = kiln::Iota(i32_shape, 1) + col_start;

  std::optional<kiln::Tile> valid;
  auto also = [&](kiln::Tile m) {  // REF_CAPTURE_OK=called locally
    valid = valid ? (*valid & m) : m;
  };
  if (config.is_causal) {
    also(row >= col);
  }
  if (ragged_q) {
    also(row < config.q_sequence_length);
  }
  if (ragged_kv) {
    also(col < config.kv_sequence_length);
  }
  return Select(*valid, s, kiln::NInf(s.shape()));
}

kiln::Tile ComputeLogits(kiln::Tile q, kiln::Tile k,
                         std::optional<kiln::VmemRef> bias_ref,
                         kiln::Scalar row_block, kiln::Scalar col_block,
                         const FlashAttnConfig& config, const Tiling& tiling) {
  kiln::Tile s =
      kiln::Dot(q, k, {.transpose_rhs = true, .precision = config.precision}) *
      (config.scale * kLog2e);
  if (bias_ref) {
    kiln::Tile bias = LoadBias(*bias_ref, config, /*row_offset=*/0, tiling.qt,
                               /*col_offset=*/0, tiling.kt);
    s += bias * kLog2e;
  }
  return ApplyStructuredMask(s, row_block, col_block, config, tiling);
}

kiln::Tile DropoutKeepMask(const FlashAttnConfig& config, const Tiling& tiling,
                           kiln::Scalar batch, kiln::Scalar q_head,
                           kiln::Scalar row_block, kiln::Scalar col_block,
                           kiln::SmemRef seed_ref, int64_t row_offset,
                           int64_t col_offset, int64_t rows, int64_t cols) {
  const int64_t mask_rows = rows > 0 ? rows : tiling.qt;
  const int64_t mask_cols = cols > 0 ? cols : tiling.kt;
  const kiln::Shape i32_shape = kiln::i32(mask_rows, mask_cols);

  // 1. The Philox {seed_lo, seed_hi, offset_lo, offset_hi} words.
  kiln::Scalar seed_lo = kiln::Load(seed_ref(0));
  kiln::Scalar seed_hi = kiln::Load(seed_ref(1));
  kiln::Scalar offset_lo = kiln::Load(seed_ref(2));
  kiln::Scalar offset_hi = kiln::Load(seed_ref(3));

  // 2. Fold the stream identity into one word. This is scalar work, so it costs
  // once per tile rather than once per element, and we can afford to avalanche
  // after every term.
  kiln::Scalar stream = seed_lo;
  for (kiln::Scalar part : {seed_hi, offset_lo, offset_hi, batch, q_head}) {
    stream = Fmix32(stream ^ part);
  }

  // 3. Absolute (q, kv) coordinates: block index * tile size + static
  // sub-block offset, plus the within-tile iota.
  kiln::Scalar q_start = row_block * tiling.qt;
  if (row_offset != 0) {
    q_start += row_offset;
  }
  kiln::Scalar kv_start = col_block * tiling.kt;
  if (col_offset != 0) {
    kv_start += col_offset;
  }
  kiln::Tile q_abs = kiln::Iota(i32_shape, 0) + q_start;
  kiln::Tile kv_abs = kiln::Iota(i32_shape, 1) + kv_start;

  // 4. Pack the coordinates into a single word, then avalanche. The pack is
  // exactly injective while padded_q * padded_kv <= 2^32, so within a stream
  // no two positions can share a hash.
  kiln::Tile key = q_abs * config.padded_kv_sequence_length + kv_abs;
  kiln::Tile hash = Fmix32(key ^ stream);

  // 5. Keep iff uint32(hash) >= round(dropout_rate * 2^32). Only a signed
  // compare is available, so flip the sign bit on both sides.
  const uint32_t threshold = static_cast<uint32_t>(std::min<double>(
      std::floor(static_cast<double>(config.dropout_rate) * 4294967296.0),
      4294967295.0));
  return (hash ^ 0x80000000u) >= (threshold ^ 0x80000000u);
}

kiln::Tile ApplyDropout(kiln::Tile t, kiln::Tile keep,
                        const FlashAttnConfig& config) {
  return Select(keep, t * (1.0f / (1.0f - config.dropout_rate)), 0.0);
}

}  // namespace torch_tpu
