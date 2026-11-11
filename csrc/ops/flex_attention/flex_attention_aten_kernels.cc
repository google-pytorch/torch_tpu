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

#include "csrc/ops/flex_attention/flex_attention_aten_kernels.h"

#include <cstdint>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/util/ArrayRef.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/internal/device/vmem_capacity.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "csrc/internal/mosaic/kiln/mlir/stablehlo_splice.h"
#include "csrc/internal/mosaic/kiln_custom_call.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

struct FlexAttnConfig {
  kiln::DType dtype;  // Element type of q, k, v and out (bf16 or f32).
  int64_t batch;
  int64_t q_heads;
  int64_t kv_heads;  // Divides `q_heads`.
  int64_t q_len;
  int64_t kv_len;
  int64_t qk_dim;
  int64_t v_dim;
  int64_t q_block;   // BlockMask Q_BLOCK_SIZE; must divide `q_len`.
  int64_t kv_block;  // BlockMask KV_BLOCK_SIZE; must divide `kv_len`.
  // Leading dims of the BlockMask tables: 1 or equal to `batch` / `q_heads`.
  int64_t mask_batch;
  int64_t mask_heads;
  float scale;
  // StableHLO modules over one [q_block, kv_block] tile (empty: no mod).
  std::string score_mod_mlir;  // @main(score f32, b, h, q_idx, kv_idx) -> f32
  std::string mask_mod_mlir;   // @main(b, h, q_idx, kv_idx) -> i1

  int64_t num_q_blocks() const { return q_len / q_block; }
  int64_t num_kv_blocks() const { return kv_len / kv_block; }
  int64_t gqa_groups() const { return q_heads / kv_heads; }
};

constexpr float kLog2e = std::numbers::log2e_v<float>;
constexpr int64_t kLanes = 128;

// Scoped VMEM limit, as in flash attention: half the capacity, to avoid
// spilling. flex_attention.py sizes the tile against the same limit.
int64_t VmemLimitBytes() { return GetDeviceVmemCapacityBytes() / 2; }

constexpr int kBatchAxis = 0;
constexpr int kHeadAxis = 1;
constexpr int kQBlockAxis = 2;
constexpr int kKvStepAxis = 3;

// Row of the flattened tables for the current (batch, head, q_block); a
// mask broadcast over batch / heads has a single row set at b = 0 / h = 0.
kiln::Scalar TableRow(const FlexAttnConfig& c) {
  kiln::Scalar b =
      c.mask_batch == 1 ? kiln::I32(0) : kiln::ProgramId(kBatchAxis);
  kiln::Scalar h =
      c.mask_heads == 1 ? kiln::I32(0) : kiln::ProgramId(kHeadAxis);
  return (b * c.mask_heads + h) * c.num_q_blocks() +
         kiln::ProgramId(kQBlockAxis);
}

// SMEM read: how many KV blocks this row attends to.
kiln::Scalar NumListedBlocks(kiln::SmemRef kv_num_blocks,
                             const FlexAttnConfig& c) {
  return kiln::Load(kv_num_blocks(TableRow(c)));
}

// SMEM read: the KV block to visit at grid step j, kv_indices[row, j]. Steps
// past the end of the list (j >= n) repeat the last block so the pipeline
// issues no new DMA; the body skips them.
kiln::Scalar CurrentKvBlock(kiln::SmemRef kv_num_blocks,
                            kiln::SmemRef kv_indices, const FlexAttnConfig& c) {
  kiln::Scalar n = NumListedBlocks(kv_num_blocks, c);
  kiln::Scalar j = Min(kiln::ProgramId(kKvStepAxis), Max(n - 1, 0));
  return kiln::Load(kv_indices(TableRow(c) * c.num_kv_blocks() + j));
}

kiln::Scalar KvHead(const FlexAttnConfig& c) {
  kiln::Scalar h = kiln::ProgramId(kHeadAxis);
  return c.gqa_groups() == 1 ? h : h / c.gqa_groups();
}

// Applies the score / mask mods to the scores `s` (natural units) of the
// tile of query head `head`, query block `q_blk` and key block `kv_blk`.
absl::StatusOr<kiln::Tile> ApplyMods(const FlexAttnConfig& c, kiln::Tile s,
                                     kiln::Scalar head, kiln::Scalar q_blk,
                                     kiln::Scalar kv_blk) {
  if (c.score_mod_mlir.empty() && c.mask_mod_mlir.empty()) return s;
  // The mods were traced at the full tile shape (see mod_to_mlir).
  const llvm::ArrayRef<int64_t> shape = s.dims();
  const kiln::Tile b = kiln::Splat(kiln::ProgramId(kBatchAxis), shape);
  const kiln::Tile h = kiln::Splat(head, shape);
  const kiln::Tile q_idx =
      (kiln::Iota(kiln::i32(c.q_block, 1), 0) + q_blk * c.q_block)
          .BroadcastTo(shape);
  const kiln::Tile kv_idx =
      (kiln::Iota(kiln::i32(1, c.kv_block), 1) + kv_blk * c.kv_block)
          .BroadcastTo(shape);
  // s = score_mod(s, b, h, q_idx, kv_idx)
  if (!c.score_mod_mlir.empty()) {
    TT_ASSIGN_OR_RETURN(
        s, kiln::SpliceStablehlo(c.score_mod_mlir, {s, b, h, q_idx, kv_idx}));
  }
  // s = mask_mod(b, h, q_idx, kv_idx) ? s : -inf
  if (!c.mask_mod_mlir.empty()) {
    TT_ASSIGN_OR_RETURN(
        const kiln::Tile keep,
        kiln::SpliceStablehlo(c.mask_mod_mlir, {b, h, q_idx, kv_idx}));
    s = Select(keep, s, kiln::NInf(s.shape()));
  }
  return s;
}

// Flex attention forward: flash attention that visits only the KV blocks
// listed in the BlockMask tables.
//
// Operands: scalar prefetch `kv_num_blocks` i32 [mask_batch * mask_heads * NQ]
// and `kv_indices` i32 [mask_batch * mask_heads * NQ * NKV] (the BlockMask
// tables flattened), then q [B, Hq, T, D], k [B, Hkv, S, D], v [B, Hkv, S, Dv].
// Results: out [B, Hq, T, Dv] and lse f32 [B, Hq, 1, T] in log2 units.
//
// Grid (B, Hq, NQ, NKV): the last axis walks `kv_indices[row, j]` for
// `j < kv_num_blocks[row]`, keeping the online-softmax state in scratch. Steps
// past the end of the list repeat the last block (no new DMA) and skip compute.
absl::StatusOr<kiln::Kernel> BuildFlexAttentionKernel(const FlexAttnConfig& c) {
  const kiln::DType f32 = kiln::f32;
  const kiln::DType i32 = kiln::i32;
  const int64_t nq = c.num_q_blocks();
  const int64_t nkv = c.num_kv_blocks();
  const int64_t table_rows = c.mask_batch * c.mask_heads * nq;

  // Which HBM tile sits in each window at grid step (b, h, i, j). q / out /
  // lse ignore j, so one query block is reused across its whole KV list and
  // written back once; k / v follow the block table.
  kiln::IndexMap q_map([] {
    return kiln::IndexIndices{kiln::ProgramId(kBatchAxis),
                              kiln::ProgramId(kHeadAxis),
                              kiln::ProgramId(kQBlockAxis), 0};
  });
  kiln::IndexMap kv_map([c] {
    return kiln::IndexIndices{
        kiln::ProgramId(kBatchAxis), KvHead(c),
        CurrentKvBlock(kiln::ScalarPrefetch(0), kiln::ScalarPrefetch(1), c), 0};
  });
  kiln::IndexMap lse_map([] {
    return kiln::IndexIndices{kiln::ProgramId(kBatchAxis),
                              kiln::ProgramId(kHeadAxis), 0,
                              kiln::ProgramId(kQBlockAxis)};
  });

  kiln::DotOptions qk_options{.transpose_rhs = true};
  kiln::DotOptions pv_options{};
  if (c.dtype == f32) {
    qk_options.precision = kiln::ContractPrecision::kFP32;
    pv_options.precision = kiln::ContractPrecision::kFP32;
  }

  // The body callback cannot return errors, so mod splice errors land here.
  absl::Status mod_status;
  absl::StatusOr<kiln::Kernel> kernel = kiln::CreateKernel(
      kiln::GridSpec{
          .name = "flex_attention_fwd",
          .grid = {c.batch, c.q_heads, nq, nkv},
          .dimension_semantics = {kiln::DimensionSemantics::kParallel,
                                  kiln::DimensionSemantics::kParallel,
                                  kiln::DimensionSemantics::kParallel,
                                  kiln::DimensionSemantics::kArbitrary}},
      kiln::ScalarPrefetchSpec{i32(table_rows)},
      kiln::ScalarPrefetchSpec{i32(table_rows * nkv)},
      kiln::InSpec{c.dtype(1, 1, c.q_block, c.qk_dim), q_map},
      kiln::InSpec{c.dtype(1, 1, c.kv_block, c.qk_dim), kv_map},
      kiln::InSpec{c.dtype(1, 1, c.kv_block, c.v_dim), kv_map},
      kiln::OutSpec{c.dtype(1, 1, c.q_block, c.v_dim), q_map},
      kiln::OutSpec{f32(1, 1, 1, c.q_block), lse_map},
      kiln::ScratchSpec::Vmem(f32(c.q_block, kLanes)),  // m: running row max
      kiln::ScratchSpec::Vmem(f32(c.q_block, kLanes)),  // l: running row sum
      kiln::ScratchSpec::Vmem(
          f32(c.q_block, c.v_dim)),  // acc: unnormalized out
      [&]                            // REF_CAPTURE_OK=called synchronously
      (kiln::SmemRef kv_num_blocks, kiln::SmemRef kv_indices,
       kiln::VmemRef q_ref, kiln::VmemRef k_ref, kiln::VmemRef v_ref,
       kiln::VmemRef out_ref, kiln::VmemRef lse_ref, kiln::VmemRef m_ref,
       kiln::VmemRef l_ref, kiln::VmemRef acc_ref) {
        const kiln::Scalar j = kiln::ProgramId(kKvStepAxis);
        const kiln::Scalar n = NumListedBlocks(kv_num_blocks, c);

        // Start of a row: reset the online-softmax state.
        kiln::If(j == 0, [&] {  // REF_CAPTURE_OK=called synchronously
          kiln::Store(m_ref, kiln::NInf(f32(c.q_block, kLanes)));
          kiln::Store(l_ref, kiln::Zeros(f32(c.q_block, kLanes)));
          kiln::Store(acc_ref, kiln::Zeros(f32(c.q_block, c.v_dim)));
        });

        // One listed KV block: fold its scores into (m, l, acc).
        kiln::If(j < n, [&] {  // REF_CAPTURE_OK=called synchronously
          const kiln::Tile q = kiln::Load(q_ref(0, 0));
          const kiln::Tile k = kiln::Load(k_ref(0, 0));
          const kiln::Tile v = kiln::Load(v_ref(0, 0));
          // Mods see the scores in natural units; the softmax runs in the
          // log2 domain.
          const kiln::Tile scores = kiln::Dot(q, k, qk_options) * c.scale;
          absl::StatusOr<kiln::Tile> modded =
              ApplyMods(c, scores, kiln::ProgramId(kHeadAxis),
                        kiln::ProgramId(kQBlockAxis),
                        CurrentKvBlock(kv_num_blocks, kv_indices, c));
          if (!modded.ok()) {
            mod_status = modded.status();
            return;  // The kernel is discarded below.
          }
          const kiln::Tile s = *modded * kLog2e;

          // m and l stay lane-replicated [q_block, 128]: broadcasting a
          // [q_block, 1] column across the kv_block lanes every step is
          // costlier than repeating whole vregs.
          const kiln::Tile m_prev = kiln::Load(m_ref);
          const kiln::Tile l_prev = kiln::Load(l_ref);
          const kiln::Tile m_new =
              Max(m_prev, kiln::MaxReduce(s, 1).Lanes(kLanes));
          // Rows with no visible key yet have m == -inf; use 0 in the
          // exponents so they give exp2(-inf) = 0 rather than NaN.
          const kiln::Tile m_safe =
              Select(m_new == kiln::NInf(m_new.shape()), 0.0, m_new);
          const kiln::Tile alpha = Exp2(m_prev - m_safe);
          const kiln::Tile p = Exp2(s - m_safe.Lanes(c.kv_block));
          kiln::Store(acc_ref, kiln::Load(acc_ref) * alpha.Lanes(c.v_dim) +
                                   kiln::Dot(p.AsType(c.dtype), v, pv_options));
          kiln::Store(m_ref, m_new);
          kiln::Store(l_ref,
                      l_prev * alpha + kiln::SumReduce(p, 1).Lanes(kLanes));
        });

        // End of the row: normalize and emit out / lse.
        kiln::If(j == nkv - 1, [&] {  // REF_CAPTURE_OK=called synchronously
          const kiln::Tile l = kiln::Load(l_ref).Lanes(1);
          const kiln::Tile m = kiln::Load(m_ref).Lanes(1);
          // Fully masked rows: out = 0 and lse = -inf, never NaN.
          const kiln::Tile is_empty = l == kiln::Zeros(l.shape());
          const kiln::Tile safe_l = Select(is_empty, 1.0, l);
          kiln::Store(
              out_ref(0, 0),
              (kiln::Load(acc_ref) * kiln::Reciprocal(safe_l)).AsType(c.dtype));
          kiln::Store(lse_ref(0, 0),
                      kiln::Transpose(Select(is_empty, kiln::NInf(m.shape()),
                                             m + Log2(safe_l))));
        });
      });
  TT_RETURN_IF_ERROR(mod_status);
  return kernel;
}

// Grid axes of the dK / dV kernel.
constexpr int kKvHeadAxis = 1;
constexpr int kKvBlockAxis = 2;
constexpr int kGroupAxis = 3;  // Query head within the KV head's group.
constexpr int kQStepAxis = 4;

// Query head of the current dK / dV grid step.
kiln::Scalar DkvQHead(const FlexAttnConfig& c) {
  return kiln::ProgramId(kKvHeadAxis) * c.gqa_groups() +
         kiln::ProgramId(kGroupAxis);
}

// Row of the flattened query-side tables for the current (batch, query head,
// KV block).
kiln::Scalar DkvTableRow(const FlexAttnConfig& c) {
  kiln::Scalar b =
      c.mask_batch == 1 ? kiln::I32(0) : kiln::ProgramId(kBatchAxis);
  kiln::Scalar h = c.mask_heads == 1 ? kiln::I32(0) : DkvQHead(c);
  return (b * c.mask_heads + h) * c.num_kv_blocks() +
         kiln::ProgramId(kKvBlockAxis);
}

// SMEM read: how many query blocks see this KV block.
kiln::Scalar NumListedQBlocks(kiln::SmemRef q_num_blocks,
                              const FlexAttnConfig& c) {
  return kiln::Load(q_num_blocks(DkvTableRow(c)));
}

// SMEM read: the query block to visit at step j, q_indices[row, j]. Steps past
// the end of the list repeat the last block (no new DMA); the body skips them.
kiln::Scalar CurrentQBlock(kiln::SmemRef q_num_blocks, kiln::SmemRef q_indices,
                           const FlexAttnConfig& c) {
  kiln::Scalar n = NumListedQBlocks(q_num_blocks, c);
  kiln::Scalar j = Min(kiln::ProgramId(kQStepAxis), Max(n - 1, 0));
  return kiln::Load(q_indices(DkvTableRow(c) * c.num_q_blocks() + j));
}

// Flex attention backward for dK and dV.
//
// Operands: scalar prefetch `q_num_blocks` i32 [mask_batch * mask_heads * NKV]
// and `q_indices` i32 [mask_batch * mask_heads * NKV * NQ] (the query blocks
// that see each KV block), then q [B, Hq, T, D], k [B, Hkv, S, D],
// v [B, Hkv, S, Dv], grad_out [B, Hq, T, Dv], and f32 lse (log2 units) and
// di = rowsum(grad_out * out), both [B, Hq, 1, T].
// Results: dk [B, Hkv, S, D] and dv [B, Hkv, S, Dv].
//
// Grid (B, Hkv, NKV, G, NQ): for each KV block, the last two axes walk the
// query blocks of every query head in the KV head's group, accumulating
// dK += dS^T Q * scale and dV += P^T dO in scratch.
absl::StatusOr<kiln::Kernel> BuildFlexAttentionDkvKernel(
    const FlexAttnConfig& c) {
  const kiln::DType f32 = kiln::f32;
  const kiln::DType i32 = kiln::i32;
  const int64_t nq = c.num_q_blocks();
  const int64_t nkv = c.num_kv_blocks();
  const int64_t groups = c.gqa_groups();
  const int64_t table_rows = c.mask_batch * c.mask_heads * nkv;

  kiln::IndexMap kv_map([] {
    return kiln::IndexIndices{kiln::ProgramId(kBatchAxis),
                              kiln::ProgramId(kKvHeadAxis),
                              kiln::ProgramId(kKvBlockAxis), 0};
  });
  kiln::IndexMap q_map([c] {
    return kiln::IndexIndices{
        kiln::ProgramId(kBatchAxis), DkvQHead(c),
        CurrentQBlock(kiln::ScalarPrefetch(0), kiln::ScalarPrefetch(1), c), 0};
  });
  kiln::IndexMap row_map([c] {
    return kiln::IndexIndices{
        kiln::ProgramId(kBatchAxis), DkvQHead(c), 0,
        CurrentQBlock(kiln::ScalarPrefetch(0), kiln::ScalarPrefetch(1), c)};
  });

  std::optional<kiln::ContractPrecision> precision;
  if (c.dtype == f32) precision = kiln::ContractPrecision::kFP32;
  const kiln::DotOptions nt{.transpose_rhs = true, .precision = precision};
  const kiln::DotOptions tn{.transpose_lhs = true, .precision = precision};

  // The body callback cannot return errors, so mod splice errors land here.
  absl::Status mod_status;
  absl::StatusOr<kiln::Kernel> kernel = kiln::CreateKernel(
      kiln::GridSpec{
          .name = "flex_attention_bwd_dkv",
          .grid = {c.batch, c.kv_heads, nkv, groups, nq},
          .dimension_semantics = {kiln::DimensionSemantics::kParallel,
                                  kiln::DimensionSemantics::kParallel,
                                  kiln::DimensionSemantics::kParallel,
                                  kiln::DimensionSemantics::kArbitrary,
                                  kiln::DimensionSemantics::kArbitrary}},
      kiln::ScalarPrefetchSpec{i32(table_rows)},
      kiln::ScalarPrefetchSpec{i32(table_rows * nq)},
      kiln::InSpec{c.dtype(1, 1, c.q_block, c.qk_dim), q_map},     // q
      kiln::InSpec{c.dtype(1, 1, c.kv_block, c.qk_dim), kv_map},   // k
      kiln::InSpec{c.dtype(1, 1, c.kv_block, c.v_dim), kv_map},    // v
      kiln::InSpec{c.dtype(1, 1, c.q_block, c.v_dim), q_map},      // grad_out
      kiln::InSpec{f32(1, 1, 1, c.q_block), row_map},              // lse
      kiln::InSpec{f32(1, 1, 1, c.q_block), row_map},              // di
      kiln::OutSpec{c.dtype(1, 1, c.kv_block, c.qk_dim), kv_map},  // dk
      kiln::OutSpec{c.dtype(1, 1, c.kv_block, c.v_dim), kv_map},   // dv
      kiln::ScratchSpec::Vmem(f32(c.kv_block, c.qk_dim)),  // dk accumulator
      kiln::ScratchSpec::Vmem(f32(c.kv_block, c.v_dim)),   // dv accumulator
      [&]  // REF_CAPTURE_OK=called synchronously
      (kiln::SmemRef q_num_blocks, kiln::SmemRef q_indices, kiln::VmemRef q_ref,
       kiln::VmemRef k_ref, kiln::VmemRef v_ref, kiln::VmemRef do_ref,
       kiln::VmemRef lse_ref, kiln::VmemRef di_ref, kiln::VmemRef dk_ref,
       kiln::VmemRef dv_ref, kiln::VmemRef dk_acc, kiln::VmemRef dv_acc) {
        const kiln::Scalar g = kiln::ProgramId(kGroupAxis);
        const kiln::Scalar j = kiln::ProgramId(kQStepAxis);
        const kiln::Scalar n = NumListedQBlocks(q_num_blocks, c);

        kiln::If((g == 0) & (j == 0),
                 [&] {  // REF_CAPTURE_OK=called synchronously
                   kiln::Store(dk_acc, kiln::Zeros(f32(c.kv_block, c.qk_dim)));
                   kiln::Store(dv_acc, kiln::Zeros(f32(c.kv_block, c.v_dim)));
                 });

        // One listed query block: recompute P and fold its gradients in.
        kiln::If(j < n, [&] {  // REF_CAPTURE_OK=called synchronously
          const kiln::Tile q = kiln::Load(q_ref(0, 0));
          const kiln::Tile k = kiln::Load(k_ref(0, 0));
          const kiln::Tile v = kiln::Load(v_ref(0, 0));
          const kiln::Tile d_out = kiln::Load(do_ref(0, 0));
          const kiln::Tile scores = kiln::Dot(q, k, nt) * c.scale;
          absl::StatusOr<kiln::Tile> modded = ApplyMods(
              c, scores, DkvQHead(c), CurrentQBlock(q_num_blocks, q_indices, c),
              kiln::ProgramId(kKvBlockAxis));
          if (!modded.ok()) {
            mod_status = modded.status();
            return;  // The kernel is discarded below.
          }
          const kiln::Tile s = *modded * kLog2e;

          const llvm::SmallVector<int64_t, 2> tile = {c.q_block, c.kv_block};
          const kiln::Tile lse = kiln::Load(lse_ref(0, 0))
                                     .Reshape({c.q_block, 1})
                                     .BroadcastTo(tile);
          const kiln::Tile di = kiln::Load(di_ref(0, 0))
                                    .Reshape({c.q_block, 1})
                                    .BroadcastTo(tile);
          // Fully masked rows have lse == -inf; their P is 0, never NaN.
          const kiln::Tile is_empty = lse == kiln::NInf(lse.shape());
          const kiln::Tile p =
              Select(is_empty, 0.0, Exp2(s - Select(is_empty, 0.0, lse)));
          const kiln::Tile dp = kiln::Dot(d_out, v, nt);
          const kiln::Tile ds = p * (dp - di);
          kiln::Store(dv_acc, kiln::Load(dv_acc) +
                                  kiln::Dot(p.AsType(c.dtype), d_out, tn));
          kiln::Store(dk_acc,
                      kiln::Load(dk_acc) +
                          kiln::Dot(ds.AsType(c.dtype), q, tn) * c.scale);
        });

        kiln::If(
            (g == groups - 1) & (j == nq - 1),
            [&] {  // REF_CAPTURE_OK=called synchronously
              kiln::Store(dk_ref(0, 0), kiln::Load(dk_acc).AsType(c.dtype));
              kiln::Store(dv_ref(0, 0), kiln::Load(dv_acc).AsType(c.dtype));
            });
      });
  TT_RETURN_IF_ERROR(mod_status);
  return kernel;
}

absl::StatusOr<kiln::DType> ToKilnDType(at::ScalarType dtype) {
  switch (dtype) {
    case at::kBFloat16:
      return kiln::bf16;
    case at::kFloat:
      return kiln::f32;
    default:
      return TT_ERROR(error::kPythonNotImplementedError)
             << "expected bfloat16 or float32 inputs, got " << dtype;
  }
}

absl::StatusOr<FlexAttnConfig> MakeConfig(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    int64_t mask_batch, int64_t mask_heads, int64_t q_block, int64_t kv_block,
    double scale, std::string_view score_mod_mlir,
    std::string_view mask_mod_mlir) {
  TT_RET_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
               error::kInvalidArgument)
      << "expected 4D q, k and v, got " << q.dim() << "D, " << k.dim()
      << "D and " << v.dim() << "D";
  TT_RET_CHECK(
      q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
      error::kInvalidArgument)
      << "expected q, k and v to share a dtype, got " << q.scalar_type() << ", "
      << k.scalar_type() << " and " << v.scalar_type();
  TT_RET_CHECK(k.size(0) == q.size(0) && v.size(0) == q.size(0) &&
                   v.size(1) == k.size(1) && v.size(2) == k.size(2) &&
                   k.size(3) == q.size(3),
               error::kInvalidArgument)
      << "expected consistent q, k and v shapes, got " << q.sizes() << ", "
      << k.sizes() << " and " << v.sizes();
  TT_RET_CHECK(q_block > 0 && kv_block > 0 && q.size(2) % q_block == 0 &&
                   k.size(2) % kv_block == 0,
               error::kInvalidArgument)
      << "sequence lengths must be multiples of the block sizes (got q_len="
      << q.size(2) << ", kv_len=" << k.size(2) << ", q_block=" << q_block
      << ", kv_block=" << kv_block << ")";
  TT_RET_CHECK(k.size(1) > 0 && q.size(1) % k.size(1) == 0,
               error::kInvalidArgument)
      << "q_heads must be a multiple of kv_heads";
  TT_RET_CHECK((mask_batch == 1 || mask_batch == q.size(0)) &&
                   (mask_heads == 1 || mask_heads == q.size(1)),
               error::kInvalidArgument)
      << "block mask batch/head dims must be 1 or match q";
  TT_ASSIGN_OR_RETURN(const kiln::DType dtype, ToKilnDType(q.scalar_type()));

  FlexAttnConfig c{
      .dtype = dtype,
      .batch = q.size(0),
      .q_heads = q.size(1),
      .kv_heads = k.size(1),
      .q_len = q.size(2),
      .kv_len = k.size(2),
      .qk_dim = q.size(3),
      .v_dim = v.size(3),
      .q_block = q_block,
      .kv_block = kv_block,
      .mask_batch = mask_batch,
      .mask_heads = mask_heads,
      .scale = static_cast<float>(scale),
      .score_mod_mlir = std::string(score_mod_mlir),
      .mask_mod_mlir = std::string(mask_mod_mlir),
  };
  return c;
}

// Checks the flattened block tables: `num_blocks` has `rows` entries and
// `indices` has `rows * cols`.
absl::Status ValidateTables(const at::Tensor& num_blocks,
                            const at::Tensor& indices, int64_t rows,
                            int64_t cols, std::string_view num_name,
                            std::string_view indices_name) {
  TT_RET_CHECK(
      num_blocks.scalar_type() == at::kInt && indices.scalar_type() == at::kInt,
      error::kInvalidArgument)
      << "block mask tables must be int32";
  TT_RET_CHECK(num_blocks.dim() == 1 && num_blocks.size(0) == rows,
               error::kInvalidArgument)
      << "expected " << num_name << " to be 1D with " << rows
      << " entries, got " << num_blocks.sizes();
  TT_RET_CHECK(indices.dim() == 1 && indices.size(0) == rows * cols,
               error::kInvalidArgument)
      << "expected " << indices_name << " to be 1D with " << rows * cols
      << " entries, got " << indices.sizes();
  return absl::OkStatus();
}

absl::StatusOr<std::tuple<at::Tensor, at::Tensor>> FlexAttentionFwdImpl(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& kv_num_blocks, const at::Tensor& kv_indices,
    int64_t mask_batch, int64_t mask_heads, int64_t q_block, int64_t kv_block,
    double scale, std::string_view score_mod_mlir,
    std::string_view mask_mod_mlir, OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(
      const FlexAttnConfig config,
      MakeConfig(q, k, v, mask_batch, mask_heads, q_block, kv_block, scale,
                 score_mod_mlir, mask_mod_mlir));
  TT_RETURN_IF_ERROR(
      ValidateTables(kv_num_blocks, kv_indices,
                     mask_batch * mask_heads * config.num_q_blocks(),
                     config.num_kv_blocks(), "kv_num_blocks", "kv_indices"));

  const Dimensions out_dims = {config.batch, config.q_heads, config.q_len,
                               config.v_dim};
  const Dimensions lse_dims = {config.batch, config.q_heads, 1, config.q_len};

  auto op_builder =
      [config, out_dims, lse_dims](
          absl::Span<mlir::MlirOp> inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<DynamicMlirOpResults> {
    TT_ASSIGN_OR_RETURN(kiln::Kernel kernel, BuildFlexAttentionKernel(config));
    mlir::OpBuilder& op_builder = builder.getOpBuilder();
    const mlir::Type elem_type = GetTensorTypeOrDie(inputs[2]).getElementType();
    const llvm::SmallVector<mlir::Type, 2> result_types = {
        mlir::RankedTensorType::get(out_dims, elem_type),
        mlir::RankedTensorType::get(lse_dims, op_builder.getF32Type())};
    // Kernel operand order: scalar prefetch (the two tables), then q, k, v.
    llvm::SmallVector<mlir::Value, 5> operands;
    for (mlir::MlirOp& input : inputs) {
      operands.push_back(input.getValue());
    }
    TT_ASSIGN_OR_RETURN(
        auto custom_call,
        CreateKilnCustomCallOp(
            op_builder, builder.getLoc(), std::move(kernel), operands,
            result_types, VmemLimitBytes(),
            /*needs_hlo_passes=*/!config.score_mod_mlir.empty() ||
                !config.mask_mod_mlir.empty()));
    DynamicMlirOpResults results;
    results.push_back(mlir::MlirOp(builder, custom_call.getResult(0)));
    results.push_back(mlir::MlirOp(builder, custom_call.getResult(1)));
    return results;
  };

  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype,
                      ConvertTo<mlir::ElementType>(q.scalar_type()));
  const std::vector<mlir::ElementType> out_dtypes = {out_dtype,
                                                     mlir::ElementType::F32};
  const std::vector<absl::Span<const int64_t>> out_dims_list = {out_dims,
                                                                lse_dims};
  const std::vector<at::Tensor> dispatch_inputs = {kv_num_blocks, kv_indices, q,
                                                   k, v};
  TT_ASSIGN_OR_RETURN(std::vector<DeviceBufferRef> results,
                      (DispatchOp<kDynamicSize, kDynamicSize>(
                          std::move(op_builder), dispatch_inputs,
                          {.out_dtypes = out_dtypes,
                           .out_dims_list = out_dims_list,
                           .op_param_cache_keys = std::move(param_keys)})));
  return std::make_tuple(MakeTensor(std::move(results[0])),
                         MakeTensor(std::move(results[1])));
}

absl::StatusOr<std::tuple<at::Tensor, at::Tensor>> FlexAttentionBwdDkvImpl(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& grad_out, const at::Tensor& lse, const at::Tensor& di,
    const at::Tensor& q_num_blocks, const at::Tensor& q_indices,
    int64_t mask_batch, int64_t mask_heads, int64_t q_block, int64_t kv_block,
    double scale, std::string_view mask_mod_mlir, OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const FlexAttnConfig config,
                      MakeConfig(q, k, v, mask_batch, mask_heads, q_block,
                                 kv_block, scale, "", mask_mod_mlir));
  TT_RETURN_IF_ERROR(ValidateTables(
      q_num_blocks, q_indices, mask_batch * mask_heads * config.num_kv_blocks(),
      config.num_q_blocks(), "q_num_blocks", "q_indices"));
  const Dimensions grad_out_dims = {config.batch, config.q_heads, config.q_len,
                                    config.v_dim};
  TT_RET_CHECK(grad_out.scalar_type() == q.scalar_type() &&
                   grad_out.sizes() == c10::IntArrayRef(grad_out_dims),
               error::kInvalidArgument)
      << "expected grad_out of shape " << c10::IntArrayRef(grad_out_dims)
      << " and dtype " << q.scalar_type() << ", got " << grad_out.sizes()
      << " and " << grad_out.scalar_type();

  const Dimensions dk_dims = {config.batch, config.kv_heads, config.kv_len,
                              config.qk_dim};
  const Dimensions dv_dims = {config.batch, config.kv_heads, config.kv_len,
                              config.v_dim};
  auto op_builder =
      [config, dk_dims, dv_dims](
          absl::Span<mlir::MlirOp> inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<DynamicMlirOpResults> {
    TT_ASSIGN_OR_RETURN(kiln::Kernel kernel,
                        BuildFlexAttentionDkvKernel(config));
    const mlir::Type elem_type = GetTensorTypeOrDie(inputs[2]).getElementType();
    const llvm::SmallVector<mlir::Type, 2> result_types = {
        mlir::RankedTensorType::get(dk_dims, elem_type),
        mlir::RankedTensorType::get(dv_dims, elem_type)};
    // Kernel operand order: scalar prefetch (the two tables), then q, k, v,
    // grad_out, lse, di.
    llvm::SmallVector<mlir::Value, 8> operands;
    for (mlir::MlirOp& input : inputs) {
      operands.push_back(input.getValue());
    }
    TT_ASSIGN_OR_RETURN(
        auto custom_call,
        CreateKilnCustomCallOp(
            builder.getOpBuilder(), builder.getLoc(), std::move(kernel),
            operands, result_types, VmemLimitBytes(),
            /*needs_hlo_passes=*/!config.mask_mod_mlir.empty()));
    DynamicMlirOpResults results;
    results.push_back(mlir::MlirOp(builder, custom_call.getResult(0)));
    results.push_back(mlir::MlirOp(builder, custom_call.getResult(1)));
    return results;
  };

  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype,
                      ConvertTo<mlir::ElementType>(q.scalar_type()));
  const std::vector<mlir::ElementType> out_dtypes = {out_dtype, out_dtype};
  const std::vector<absl::Span<const int64_t>> out_dims_list = {dk_dims,
                                                                dv_dims};
  const std::vector<at::Tensor> dispatch_inputs = {
      q_num_blocks, q_indices, q, k, v, grad_out, lse, di};
  TT_ASSIGN_OR_RETURN(std::vector<DeviceBufferRef> results,
                      (DispatchOp<kDynamicSize, kDynamicSize>(
                          std::move(op_builder), dispatch_inputs,
                          {.out_dtypes = out_dtypes,
                           .out_dims_list = out_dims_list,
                           .op_param_cache_keys = std::move(param_keys)})));
  return std::make_tuple(MakeTensor(std::move(results[0])),
                         MakeTensor(std::move(results[1])));
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> FlexAttentionFwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& kv_num_blocks, const at::Tensor& kv_indices,
    int64_t mask_batch, int64_t mask_heads, int64_t q_block, int64_t kv_block,
    double scale, std::string_view score_mod_mlir,
    std::string_view mask_mod_mlir) {
  TT_KERNEL(OpName::kFlexAttentionFwd, param_keys,
            (q, k, v, kv_num_blocks, kv_indices, mask_batch, mask_heads,
             q_block, kv_block, scale, score_mod_mlir, mask_mod_mlir),
            {
              TT_ASSIGN_OR_THROW(
                  auto results,
                  FlexAttentionFwdImpl(q, k, v, kv_num_blocks, kv_indices,
                                       mask_batch, mask_heads, q_block,
                                       kv_block, scale, score_mod_mlir,
                                       mask_mod_mlir, std::move(param_keys)));
              return results;
            });
}

std::tuple<at::Tensor, at::Tensor> FlexAttentionBwdDkv(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& grad_out, const at::Tensor& lse, const at::Tensor& di,
    const at::Tensor& q_num_blocks, const at::Tensor& q_indices,
    int64_t mask_batch, int64_t mask_heads, int64_t q_block, int64_t kv_block,
    double scale, std::string_view mask_mod_mlir) {
  TT_KERNEL(OpName::kFlexAttentionBwdDkv, param_keys,
            (q, k, v, grad_out, lse, di, q_num_blocks, q_indices, mask_batch,
             mask_heads, q_block, kv_block, scale, mask_mod_mlir),
            {
              TT_ASSIGN_OR_THROW(
                  auto results,
                  FlexAttentionBwdDkvImpl(
                      q, k, v, grad_out, lse, di, q_num_blocks, q_indices,
                      mask_batch, mask_heads, q_block, kv_block, scale,
                      mask_mod_mlir, std::move(param_keys)));
              return results;
            });
}

}  // namespace torch_tpu
