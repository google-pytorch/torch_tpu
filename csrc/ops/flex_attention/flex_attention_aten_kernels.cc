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
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
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
kiln::Scalar NumListedBlocks(kiln::Ref kv_num_blocks, const FlexAttnConfig& c) {
  return kiln::LoadScalar(kv_num_blocks(TableRow(c)));
}

// SMEM read: the KV block to visit at grid step j, kv_indices[row, j]. Steps
// past the end of the list (j >= n) repeat the last block so the pipeline
// issues no new DMA; the body skips them.
kiln::Scalar CurrentKvBlock(kiln::Ref kv_num_blocks, kiln::Ref kv_indices,
                            const FlexAttnConfig& c) {
  kiln::Scalar n = NumListedBlocks(kv_num_blocks, c);
  kiln::Scalar j = Min(kiln::ProgramId(kKvStepAxis), Max(n - 1, 0));
  return kiln::LoadScalar(kv_indices(TableRow(c) * c.num_kv_blocks() + j));
}

kiln::Scalar KvHead(const FlexAttnConfig& c) {
  kiln::Scalar h = kiln::ProgramId(kHeadAxis);
  return c.gqa_groups() == 1 ? h : h / c.gqa_groups();
}

// Applies the score / mask mods to the scores `s` (natural units) of the
// current tile.
absl::StatusOr<kiln::Tile> ApplyMods(const FlexAttnConfig& c,
                                     kiln::Ref kv_num_blocks,
                                     kiln::Ref kv_indices, kiln::Tile s) {
  if (c.score_mod_mlir.empty() && c.mask_mod_mlir.empty()) return s;
  // The mods were traced at the full tile shape (see mod_to_mlir).
  const llvm::ArrayRef<int64_t> shape = s.dims();
  const kiln::Tile b = kiln::Splat(kiln::ProgramId(kBatchAxis), shape);
  const kiln::Tile h = kiln::Splat(kiln::ProgramId(kHeadAxis), shape);
  const kiln::Tile q_idx = (kiln::Iota(kiln::i32(c.q_block, 1), 0) +
                            kiln::ProgramId(kQBlockAxis) * c.q_block)
                               .BroadcastTo(shape);
  const kiln::Tile kv_idx =
      (kiln::Iota(kiln::i32(1, c.kv_block), 1) +
       CurrentKvBlock(kv_num_blocks, kv_indices, c) * c.kv_block)
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
      (kiln::Ref kv_num_blocks, kiln::Ref kv_indices, kiln::Ref q_ref,
       kiln::Ref k_ref, kiln::Ref v_ref, kiln::Ref out_ref, kiln::Ref lse_ref,
       kiln::Ref m_ref, kiln::Ref l_ref, kiln::Ref acc_ref) {
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
              ApplyMods(c, kv_num_blocks, kv_indices, scores);
          if (!modded.ok()) {
            mod_status = modded.status();
            return;  // The kernel is discarded below.
          }
          const kiln::Tile s = *modded * kLog2e;

          const kiln::Tile m_prev = kiln::Load(m_ref).Lanes(1);
          const kiln::Tile l_prev = kiln::Load(l_ref).Lanes(1);
          const kiln::Tile m_new = Max(m_prev, kiln::MaxReduce(s, 1));
          // Rows with no visible key yet have m == -inf; use 0 in the
          // exponents so they give exp2(-inf) = 0 rather than NaN.
          const kiln::Tile m_safe =
              Select(m_new == kiln::NInf(m_new.shape()), 0.0, m_new);
          const kiln::Tile alpha = Exp2(m_prev - m_safe);
          const kiln::Tile p = Exp2(s - m_safe);
          kiln::Store(acc_ref, kiln::Load(acc_ref) * alpha +
                                   kiln::Dot(p.AsType(c.dtype), v, pv_options));
          kiln::Store(m_ref, m_new.Lanes(kLanes));
          kiln::Store(l_ref,
                      (l_prev * alpha + kiln::SumReduce(p, 1)).Lanes(kLanes));
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
    const at::Tensor& kv_num_blocks, const at::Tensor& kv_indices,
    int64_t mask_batch, int64_t mask_heads, int64_t q_block, int64_t kv_block,
    double scale, std::string_view score_mod_mlir,
    std::string_view mask_mod_mlir) {
  TT_RET_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
               error::kInvalidArgument)
      << "expected 4-D q, k and v, got " << q.dim() << "-D, " << k.dim()
      << "-D and " << v.dim() << "-D";
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

  const int64_t table_rows = mask_batch * mask_heads * c.num_q_blocks();
  TT_RET_CHECK(kv_num_blocks.scalar_type() == at::kInt &&
                   kv_indices.scalar_type() == at::kInt,
               error::kInvalidArgument)
      << "block mask tables must be int32";
  TT_RET_CHECK(kv_num_blocks.dim() == 1 && kv_num_blocks.size(0) == table_rows,
               error::kInvalidArgument)
      << "expected kv_num_blocks to be 1-D with " << table_rows
      << " entries, got " << kv_num_blocks.sizes();
  TT_RET_CHECK(kv_indices.dim() == 1 &&
                   kv_indices.size(0) == table_rows * c.num_kv_blocks(),
               error::kInvalidArgument)
      << "expected kv_indices to be 1-D with " << table_rows * c.num_kv_blocks()
      << " entries, got " << kv_indices.sizes();
  return c;
}

absl::StatusOr<std::tuple<at::Tensor, at::Tensor>> FlexAttentionFwdImpl(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& kv_num_blocks, const at::Tensor& kv_indices,
    int64_t mask_batch, int64_t mask_heads, int64_t q_block, int64_t kv_block,
    double scale, std::string_view score_mod_mlir,
    std::string_view mask_mod_mlir, OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(
      const FlexAttnConfig config,
      MakeConfig(q, k, v, kv_num_blocks, kv_indices, mask_batch, mask_heads,
                 q_block, kv_block, scale, score_mod_mlir, mask_mod_mlir));

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
            result_types, /*vmem_limit_bytes=*/std::nullopt,
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

}  // namespace torch_tpu
