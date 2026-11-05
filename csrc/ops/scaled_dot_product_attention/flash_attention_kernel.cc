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

#include "csrc/ops/scaled_dot_product_attention/flash_attention_kernel.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "csrc/ops/scaled_dot_product_attention/flash_attention_config.h"
#include "csrc/ops/scaled_dot_product_attention/util.h"
#include "llvm/ADT/STLExtras.h"

namespace torch_tpu {

namespace {

// Row statistics (m, l) are kept lane-replicated in (qt, 128) scratch.
constexpr int64_t kLanes = 128;

// The running online-softmax state for a band of query rows.
struct SoftmaxState {
  kiln::Tile m;    // [rows, 128] f32 running max (log2 domain).
  kiln::Tile l;    // [rows, 128] f32 running sum.
  kiln::Tile acc;  // [rows, d_vo] f32 unnormalized output.
};

// Kernel arguments that are shared by all sub-blocks of a grid step.
struct KernelArgs {
  std::optional<kiln::SmemRef> seed_ref;
  kiln::VmemRef q_ref;
  kiln::VmemRef k_ref;
  kiln::VmemRef v_ref;
  std::optional<kiln::VmemRef> bias_ref;
  kiln::VmemRef o_ref;
  std::optional<kiln::VmemRef> lse_ref;
  kiln::VmemRef acc_ref;
  kiln::VmemRef m_ref;
  kiln::VmemRef l_ref;
  kiln::Scalar qi;
  kiln::Scalar ki;
};

kiln::IndexIndices QBlockIndexMap() {
  return {kiln::ProgramId(0), kiln::ProgramId(1), kiln::ProgramId(2), 0};
}

kiln::IndexIndices LseBlockIndexMap() {
  return {kiln::ProgramId(0), kiln::ProgramId(1), 0, kiln::ProgramId(2)};
}

void InitSoftmaxScratch(const FlashAttnConfig& config, const Tiling& tiling,
                        const KernelArgs& args) {
  kiln::Store(args.acc_ref,
              kiln::Zeros(kiln::f32(tiling.qt, config.vo_head_dim)));
  kiln::Store(args.l_ref, kiln::Zeros(kiln::f32(tiling.qt, kLanes)));
  kiln::Store(args.m_ref, kiln::NInf(kiln::f32(tiling.qt, kLanes)));
}

kiln::Tile ApplyDropoutToProbabilities(const FlashAttnConfig& config,
                                       const Tiling& tiling,
                                       const KernelArgs& args,
                                       int64_t row_offset, int64_t col_offset,
                                       kiln::Tile p) {
  if (!args.seed_ref) {
    return p;
  }
  kiln::Tile keep = DropoutKeepMask(
      config, tiling, kiln::ProgramId(0), kiln::ProgramId(1), args.qi, args.ki,
      *args.seed_ref, row_offset, col_offset, p.dims()[0], p.dims()[1]);
  return ApplyDropout(p, keep, config);
}

// out = acc / l ; lse = m + Log2(l) (log2-scaled, only used by backward).
void StoreNormalizedOutput(const FlashAttnConfig& config,
                           const KernelArgs& args, const SoftmaxState& state,
                           int64_t row_offset) {
  const int64_t rows = state.acc.dims()[0];
  kiln::Tile is_zero_l = state.l == 0.0f;
  kiln::Tile safe_l = Select(is_zero_l, 1.0f, state.l);
  kiln::SliceDim row_slice = kiln::Slice(row_offset, rows);
  kiln::Tile out =
      state.acc * kiln::Reciprocal(safe_l).Lanes(config.vo_head_dim);
  kiln::Store(args.o_ref(0, 0, row_slice, kiln::all), out);
  if (args.lse_ref) {
    kiln::Tile lse =
        Select(is_zero_l, kiln::NInf(state.l.shape()), state.m + Log2(safe_l));
    kiln::Tile lse_row = kiln::Transpose(lse.Lanes(1));
    kiln::Store((*args.lse_ref)(0, 0, kiln::all, row_slice), lse_row);
  }
}

// Loads a [rows, cols] sub-block starting at `row_offset` from `ref`
// ([1, 1, block_rows, cols]). Because unaligned sequence inputs are not padded
// in VMEM, the last grid block (`block_idx == num_blocks - 1`) only has
// `seq_len % block_rows` valid rows; clamping the `vector.load` to the valid
// tail and zero-padding in VREGs avoids reading past the end of VMEM (and
// zeroes out-of-bounds V rows so 0 * NaN cannot poison the accumulator).
kiln::Tile LoadClampedRows(kiln::VmemRef ref, kiln::Scalar block_idx,
                           const int64_t num_blocks, const int64_t seq_len,
                           const int64_t row_offset, const int64_t rows) {
  const int64_t block_rows = ref.dims()[2];
  const int64_t cols = ref.dims()[3];
  const int64_t tail = seq_len % block_rows;
  auto load_full = [&] {  // REF_CAPTURE_OK=called synchronously
    return kiln::Load(ref(0, 0, kiln::Slice(row_offset, rows), kiln::all));
  };
  if (tail == 0 || row_offset + rows <= tail) {
    return load_full();
  }
  const int64_t valid_rows = std::max<int64_t>(0, tail - row_offset);
  auto load_tail = [&] {  // REF_CAPTURE_OK=called synchronously
    if (valid_rows == 0) {
      return kiln::Zeros(ref.dtype()(rows, cols));
    }
    kiln::Tile part =
        kiln::Load(ref(0, 0), {row_offset, 0}, {valid_rows, cols});
    return kiln::Concat(
        {part, kiln::Zeros(ref.dtype()(rows - valid_rows, cols))},
        /*axis=*/0);
  };
  if (num_blocks == 1) {
    return load_tail();
  }
  return kiln::If(block_idx == num_blocks - 1, load_tail, load_full);
}

enum class TailKind {
  kFull,
  kRagged,
};

// Loads the f32 [rows, cols] bias sub-block at (row_offset, col_offset),
// clamping ragged non-broadcast Q/KV dimensions on the final block so that
// unpadded VMEM bias buffers are never read out of bounds.
kiln::Tile LoadClampedBias(kiln::VmemRef bias_ref,
                           const FlashAttnConfig& config, const Tiling& tiling,
                           const KernelArgs& args, const int64_t row_offset,
                           const int64_t rows, const int64_t col_offset,
                           const int64_t cols) {
  const bool q_bcast = llvm::is_contained(config.mask_broadcast_dims, 2);
  const bool kv_bcast = llvm::is_contained(config.mask_broadcast_dims, 3);
  const int64_t num_q_blocks = config.padded_q_sequence_length / tiling.qt;
  const int64_t num_kv_blocks = config.padded_kv_sequence_length / tiling.kt;
  const int64_t q_tail = q_bcast ? 0 : (config.q_sequence_length % tiling.qt);
  const int64_t kv_tail =
      kv_bcast ? 0 : (config.kv_sequence_length % tiling.kt);
  const bool clamp_q = q_tail != 0 && row_offset + rows > q_tail;
  const bool clamp_kv = kv_tail != 0 && col_offset + cols > kv_tail;
  if (!clamp_q && !clamp_kv) {
    return LoadBias(bias_ref, config, row_offset, rows, col_offset, cols);
  }

  auto emit_tile = [&]  // REF_CAPTURE_OK=called inline
      (const TailKind q_kind, const TailKind kv_kind) -> kiln::Tile {
    if (q_kind == TailKind::kFull && kv_kind == TailKind::kFull) {
      return LoadBias(bias_ref, config, row_offset, rows, col_offset, cols);
    }
    const int64_t load_rows =
        q_bcast ? 1
                : (q_kind == TailKind::kRagged
                       ? std::max<int64_t>(0, q_tail - row_offset)
                       : rows);
    const int64_t load_cols =
        kv_bcast ? 1
                 : (kv_kind == TailKind::kRagged
                        ? std::max<int64_t>(0, kv_tail - col_offset)
                        : cols);
    if (load_rows == 0 || load_cols == 0) {
      return kiln::Zeros(kiln::f32(rows, cols));
    }
    const int64_t b_row_off = q_bcast ? 0 : row_offset;
    const int64_t b_col_off = kv_bcast ? 0 : col_offset;
    const int64_t valid_q = q_bcast ? rows : load_rows;
    const int64_t valid_kv = kv_bcast ? cols : load_cols;
    kiln::Tile bias = kiln::Load(bias_ref(0, 0), {b_row_off, b_col_off},
                                 {load_rows, load_cols})
                          .AsType(kiln::f32)
                          .BroadcastTo({valid_q, valid_kv});
    if (valid_q < rows) {
      bias =
          kiln::Concat({bias, kiln::Zeros(kiln::f32(rows - valid_q, valid_kv))},
                       /*axis=*/0);
    }
    if (valid_kv < cols) {
      bias = kiln::Concat({bias, kiln::Zeros(kiln::f32(rows, cols - valid_kv))},
                          /*axis=*/1);
    }
    return bias;
  };

  auto emit_for_q = [&]  // REF_CAPTURE_OK=called inline
      (const TailKind q_kind) -> kiln::Tile {
    if (!clamp_kv) {
      return emit_tile(q_kind, TailKind::kFull);
    }
    if (num_kv_blocks == 1) {
      return emit_tile(q_kind, TailKind::kRagged);
    }
    return kiln::If(
        args.ki == num_kv_blocks - 1,
        [&] {  // REF_CAPTURE_OK=called synchronously
          return emit_tile(q_kind, TailKind::kRagged);
        },
        [&] {  // REF_CAPTURE_OK=called synchronously
          return emit_tile(q_kind, TailKind::kFull);
        });
  };

  if (!clamp_q) {
    return emit_for_q(TailKind::kFull);
  }
  if (num_q_blocks == 1) {
    return emit_for_q(TailKind::kRagged);
  }
  return kiln::If(
      args.qi == num_q_blocks - 1,
      [&] {  // REF_CAPTURE_OK=called synchronously
        return emit_for_q(TailKind::kRagged);
      },
      [&] {  // REF_CAPTURE_OK=called synchronously
        return emit_for_q(TailKind::kFull);
      });
}

// Folds the [rows, kc] logits sub-block at (row_offset, col_offset) of the
// current grid tile into the online softmax state `prev`.
SoftmaxState UpdateSoftmaxState(const FlashAttnConfig& config,
                                const Tiling& tiling, const KernelArgs& args,
                                kiln::Tile q, int64_t row_offset, int64_t rows,
                                int64_t col_offset, bool apply_masking,
                                const std::optional<SoftmaxState>& prev) {
  const int64_t kc = tiling.kc;
  const int64_t d_vo = config.vo_head_dim;
  const int64_t num_kv_blocks = config.padded_kv_sequence_length / tiling.kt;
  kiln::DType dt = config.element_type;

  kiln::Tile k = LoadClampedRows(args.k_ref, args.ki, num_kv_blocks,
                                 config.kv_sequence_length, col_offset, kc);
  kiln::Tile s =
      kiln::Dot(q, k, {.transpose_rhs = true, .precision = config.precision}) *
      (config.scale * kLog2e);
  if (args.bias_ref) {
    kiln::Tile bias = LoadClampedBias(*args.bias_ref, config, tiling, args,
                                      row_offset, rows, col_offset, kc);
    s += bias * kLog2e;
  }
  if (apply_masking) {
    s = ApplyStructuredMask(s, args.qi, args.ki, config, tiling, row_offset,
                            col_offset, QueryTailMasking::kSkip);
  }

  kiln::Tile mij = kiln::MaxReduce(s, 1).Lanes(kLanes);
  if (args.bias_ref) {
    // Clamping mij instead of sij ensures that when a row is entirely masked
    // out with -inf, sij - mij evaluates to (-inf) - (<large negative finite>)
    // = -inf, so exp2(-inf) = 0.0 naturally without any vector compare or
    // select on p in the inner loop.
    mij = Max(mij, std::numeric_limits<float>::lowest());
  }
  kiln::Tile v = LoadClampedRows(args.v_ref, args.ki, num_kv_blocks,
                                 config.kv_sequence_length, col_offset, kc);

  if (!prev.has_value()) {
    kiln::Tile p = Exp2(s - mij.Lanes(kc));
    kiln::Tile l = kiln::SumReduce(p, 1).Lanes(kLanes);
    kiln::Tile p_pv = ApplyDropoutToProbabilities(config, tiling, args,
                                                  row_offset, col_offset, p);
    kiln::Tile acc =
        kiln::Dot(p_pv.AsType(dt), v, {.precision = config.precision});
    return SoftmaxState{.m = mij, .l = l, .acc = acc};
  }

  kiln::Tile m_new = Max(prev->m, mij);
  kiln::Tile alpha = Exp2(prev->m - m_new);
  kiln::Tile p = Exp2(s - m_new.Lanes(kc));
  kiln::Tile l_new = alpha * prev->l + kiln::SumReduce(p, 1).Lanes(kLanes);
  kiln::Tile p_pv = ApplyDropoutToProbabilities(config, tiling, args,
                                                row_offset, col_offset, p);
  kiln::Tile pv =
      kiln::Dot(p_pv.AsType(dt), v, {.precision = config.precision});
  kiln::Tile acc_new = alpha.Lanes(d_vo) * prev->acc + pv;
  return SoftmaxState{.m = m_new, .l = l_new, .acc = acc_new};
}

SoftmaxState LoadSoftmaxState(const KernelArgs& args, int64_t row_offset,
                              int64_t num_rows) {
  return SoftmaxState{
      .m = kiln::Load(args.m_ref(kiln::Slice(row_offset, num_rows), kiln::all)),
      .l = kiln::Load(args.l_ref(kiln::Slice(row_offset, num_rows), kiln::all)),
      .acc = kiln::Load(
          args.acc_ref(kiln::Slice(row_offset, num_rows), kiln::all)),
  };
}

void StoreSoftmaxState(const KernelArgs& args, const SoftmaxState& state,
                       int64_t row_offset) {
  const int64_t num_rows = state.acc.dims()[0];
  kiln::SliceDim row_slice = kiln::Slice(row_offset, num_rows);
  kiln::Store(args.m_ref(row_slice, kiln::all), state.m);
  kiln::Store(args.l_ref(row_slice, kiln::all), state.l);
  kiln::Store(args.acc_ref(row_slice, kiln::all), state.acc);
}

// Processes a single [band_rows, kt] query band across its active KV
// sub-blocks.
void EmitQueryBand(const FlashAttnConfig& config, const Tiling& tiling,
                   const KernelArgs& args, int64_t row_offset,
                   int64_t band_rows, int64_t num_kv_subblocks,
                   bool single_kv_step, bool ragged_kv,
                   bool square_causal_diag_block, bool intersects_causal_diag) {
  const int64_t num_q_blocks = config.padded_q_sequence_length / tiling.qt;
  kiln::Tile q =
      LoadClampedRows(args.q_ref, args.qi, num_q_blocks,
                      config.q_sequence_length, row_offset, band_rows);

  std::optional<SoftmaxState> state =
      single_kv_step
          ? std::nullopt
          : std::make_optional(LoadSoftmaxState(args, row_offset, band_rows));

  for (int64_t kv_sub = 0; kv_sub < num_kv_subblocks; ++kv_sub) {
    const bool apply_causal_mask = square_causal_diag_block
                                       ? (kv_sub == num_kv_subblocks - 1)
                                       : intersects_causal_diag;
    const bool apply_masking = ragged_kv || apply_causal_mask;
    state = UpdateSoftmaxState(config, tiling, args, q, row_offset, band_rows,
                               kv_sub * tiling.kc, apply_masking, state);
  }

  if (single_kv_step) {
    StoreNormalizedOutput(config, args, *state, row_offset);
  } else {
    StoreSoftmaxState(args, *state, row_offset);
  }
}

// Processes the [qt, kt] grid tile as a static grid of [band_rows, kc]
// sub-blocks.
void EmitGridStep(const FlashAttnConfig& config, const Tiling& tiling,
                  const KernelArgs& args, bool single_kv_step, bool ragged_kv,
                  bool intersects_causal_diag) {
  const int64_t qt = tiling.qt;
  const int64_t kt = tiling.kt;
  const int64_t kc = tiling.kc;
  const bool square_causal_diag_block = intersects_causal_diag && qt == kt;
  const int64_t band_rows = square_causal_diag_block ? kc : qt;
  const int64_t num_bands = qt / band_rows;

  for (int64_t band = 0; band < num_bands; ++band) {
    const int64_t num_kv_subblocks =
        square_causal_diag_block ? (band + 1) : (kt / kc);
    EmitQueryBand(config, tiling, args, band * band_rows, band_rows,
                  num_kv_subblocks, single_kv_step, ragged_kv,
                  square_causal_diag_block, intersects_causal_diag);
  }
}

}  // namespace

absl::StatusOr<kiln::Kernel> CreateKernel(const FlashAttnConfig& config,
                                          const Tiling& tiling) {
  TT_RETURN_IF_ERROR(ValidateFlashAttnConfig(config, tiling));

  Tiling normalized_tiling = tiling;
  if (normalized_tiling.kc <= 0) {
    normalized_tiling.kc = normalized_tiling.kt;
  }
  if (normalized_tiling.kt % normalized_tiling.kc != 0 ||
      normalized_tiling.kc % kLanes != 0) {
    return TT_ERROR(error::kInvalidArgument)
           << "KV compute tile size must be a multiple of 128 that divides "
              "the KV tile size.";
  }

  const int64_t qt = normalized_tiling.qt;
  const int64_t kt = normalized_tiling.kt;
  const int64_t d_qk = config.qk_head_dim;
  const int64_t d_vo = config.vo_head_dim;
  // The grid covers the sequences rounded up to whole tiles.
  const int64_t num_q_blocks = config.padded_q_sequence_length / qt;
  const int64_t num_kv_blocks = config.padded_kv_sequence_length / kt;
  const bool single_kv_step = num_kv_blocks == 1 && !config.is_causal;
  // Out-of-bounds query rows are discarded when writing the output.
  const bool ragged_kv = config.kv_sequence_length % kt != 0;
  kiln::DType dt = config.element_type;

  // ---------------------------------------------------------------------------
  // Grid & block specs.
  // ---------------------------------------------------------------------------
  kiln::GridSpec grid{
      .name = "flash_forward",
      .grid = {config.batch_size, config.num_heads, num_q_blocks,
               num_kv_blocks},
      .dimension_semantics = {kiln::DimensionSemantics::kParallel,
                              kiln::DimensionSemantics::kParallel,
                              kiln::DimensionSemantics::kParallel,
                              kiln::DimensionSemantics::kArbitrary},
  };

  kiln::IndexMap q_map(QBlockIndexMap);
  kiln::IndexMap kv_map = KvBlockIndexMap(config, normalized_tiling);

  std::optional<kiln::OutSpec> lse_spec;
  if (config.return_lse) {
    lse_spec =
        kiln::OutSpec{kiln::f32(1, 1, 1, qt), kiln::IndexMap(LseBlockIndexMap)};
  }

  // ---------------------------------------------------------------------------
  // Kernel body.
  // ---------------------------------------------------------------------------
  TT_ASSIGN_OR_RETURN(
      kiln::Kernel kernel,
      kiln::CreateKernel(
          grid, DropoutSeedSpec(config),
          kiln::InSpec{dt(1, 1, qt, d_qk), q_map},
          kiln::InSpec{dt(1, 1, kt, d_qk), kv_map},
          kiln::InSpec{dt(1, 1, kt, d_vo), kv_map},
          AttnBiasSpec(config, normalized_tiling, dt),
          kiln::OutSpec{dt(1, 1, qt, d_vo), q_map}, lse_spec,
          kiln::ScratchSpec{kiln::f32(qt, d_vo)},
          kiln::ScratchSpec{kiln::f32(qt, kLanes)},
          kiln::ScratchSpec{kiln::f32(qt, kLanes)},
          [&]  // REF_CAPTURE_OK=called synchronously
          (std::optional<kiln::SmemRef> seed_ref, kiln::VmemRef q_ref,
           kiln::VmemRef k_ref, kiln::VmemRef v_ref,
           std::optional<kiln::VmemRef> bias_ref, kiln::VmemRef o_ref,
           std::optional<kiln::VmemRef> lse_ref, kiln::VmemRef acc_ref,
           kiln::VmemRef m_ref, kiln::VmemRef l_ref) {
            KernelArgs args{
                .seed_ref = seed_ref,
                .q_ref = q_ref,
                .k_ref = k_ref,
                .v_ref = v_ref,
                .bias_ref = bias_ref,
                .o_ref = o_ref,
                .lse_ref = lse_ref,
                .acc_ref = acc_ref,
                .m_ref = m_ref,
                .l_ref = l_ref,
                .qi = kiln::ProgramId(2),
                .ki = kiln::ProgramId(3),
            };

            if (config.is_causal) {
              kiln::If(
                  BelowOrOnDiagonal(args.qi, args.ki, normalized_tiling),
                  [&] {  // REF_CAPTURE_OK=called synchronously
                    if (!single_kv_step) {
                      kiln::If(args.ki == 0,
                               [&] {  // REF_CAPTURE_OK=called synchronously
                                 InitSoftmaxScratch(config, normalized_tiling,
                                                    args);
                               });
                    }
                    kiln::If(
                        IntersectsDiagonal(args.qi, args.ki, normalized_tiling),
                        [&] {  // REF_CAPTURE_OK=called synchronously
                          EmitGridStep(config, normalized_tiling, args,
                                       single_kv_step, ragged_kv,
                                       /*intersects_causal_diag=*/true);
                        },
                        [&] {  // REF_CAPTURE_OK=called synchronously
                          EmitGridStep(config, normalized_tiling, args,
                                       single_kv_step, ragged_kv,
                                       /*intersects_causal_diag=*/false);
                        });
                  });
            } else {
              if (!single_kv_step) {
                kiln::If(args.ki == 0,
                         [&] {  // REF_CAPTURE_OK=called synchronously
                           InitSoftmaxScratch(config, normalized_tiling, args);
                         });
              }
              EmitGridStep(config, normalized_tiling, args, single_kv_step,
                           ragged_kv, /*intersects_causal_diag=*/false);
            }

            if (!single_kv_step) {
              kiln::If(args.ki == num_kv_blocks - 1,
                       [&] {  // REF_CAPTURE_OK=called synchronously
                         StoreNormalizedOutput(
                             config, args,
                             SoftmaxState{
                                 .m = kiln::Load(args.m_ref),
                                 .l = kiln::Load(args.l_ref),
                                 .acc = kiln::Load(args.acc_ref),
                             },
                             /*row_offset=*/0);
                       });
            }
          }));

  ABSL_VLOG(1) << "Forward kernel:\n" << kiln::ToString(kernel);
  return kernel;
}

}  // namespace torch_tpu
