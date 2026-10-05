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

#include <cstdint>
#include <initializer_list>
#include <optional>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "csrc/common/error_utils.h"
#include "csrc/internal/mosaic/op_builders.h"
#include "csrc/ops/scaled_dot_product_attention/flash_attention_config.h"
#include "csrc/ops/scaled_dot_product_attention/util.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

using mlir::arith::AddFOp;          // USING_DECL_OK
using mlir::arith::CmpFOp;          // USING_DECL_OK
using mlir::arith::CmpFPredicate;   // USING_DECL_OK
using mlir::arith::CmpIOp;          // USING_DECL_OK
using mlir::arith::CmpIPredicate;   // USING_DECL_OK
using mlir::arith::ConstantOp;      // USING_DECL_OK
using mlir::arith::MaximumFOp;      // USING_DECL_OK
using mlir::arith::MulFOp;          // USING_DECL_OK
using mlir::arith::SelectOp;        // USING_DECL_OK
using mlir::arith::SubFOp;          // USING_DECL_OK
using mlir::func::ReturnOp;         // USING_DECL_OK
using mlir::math::Exp2Op;           // USING_DECL_OK
using mlir::scf::IfOp;              // USING_DECL_OK
using mlir::scf::YieldOp;           // USING_DECL_OK
using mlir::vector::CombiningKind;  // USING_DECL_OK

namespace {

void SetKernelAttributes(llvm::ArrayRef<const int64_t> qk_dimensions,
                         const FlashAttnConfig& config, const Tiling& tiling,
                         mlir::func::FuncOp fn, mlir::OpBuilder& builder) {
  const int64_t scratch_operands = 3L;
  fn->setAttr("scratch_operands", builder.getI64IntegerAttr(scratch_operands));
  SetScalarPrefetchAttr(config, fn, builder);

  const int64_t num_iteration_indices = absl::c_count_if(
      fn.getArguments(),
      [](mlir::BlockArgument arg) { return arg.getType().isIntOrIndex(); });
  if (num_iteration_indices == 0) {
    // No other attributes are needed in the case there is a single iteration.
    return;
  }
  absl::InlinedVector<int64_t, 4> iteration_bounds = {qk_dimensions.begin(),
                                                      qk_dimensions.end()};
  iteration_bounds[iteration_bounds.size() - 1] =
      config.padded_kv_sequence_length / tiling.kt;
  iteration_bounds[iteration_bounds.size() - 2] =
      config.padded_q_sequence_length / tiling.qt;
  fn->setAttr("iteration_bounds",
              builder.getDenseI64ArrayAttr(
                  {iteration_bounds.begin(), iteration_bounds.end()}));
  mlir::MLIRContext* context = builder.getContext();
  absl::InlinedVector<mlir::Attribute, 4> dimension_semantics(
      num_iteration_indices,
      GetDimensionSemanticsAttr(context, DimensionSemantics::kParallel));
  // The last dimension is arbitrary, update it accordingly.
  dimension_semantics.back() =
      GetDimensionSemanticsAttr(context, DimensionSemantics::kArbitrary);
  fn->setAttr("dimension_semantics",
              builder.getArrayAttr(
                  {dimension_semantics.begin(), dimension_semantics.end()}));

  // We need transform_indices attribute for q, k, v, any auxiliary arguments,
  // and output. This boils down to all arguments except iteration indices,
  // scalar_prefetch arguments, and scratch arguments.
  // The arguments are in the following order: iteration_indices,
  // [seed_smem], q, k, v, auxiliary_parameters, output, li, mi.
  mlir::SmallVector<mlir::AffineExpr> dims(num_iteration_indices);
  mlir::bindDimsList(context, mlir::MutableArrayRef{dims});
  // Auxiliary function for creating transform_indices attribute.
  auto transform_indices = [&](mlir::ArrayRef<mlir::AffineExpr> results) {
    return builder.getDictionaryAttr({builder.getNamedAttr(
        "transform_indices", mlir::AffineMapAttr::get(mlir::AffineMap::get(
                                 /*dimCount=*/dims.size(),
                                 /*symbolCount=*/0, results, context)))});
  };

  llvm::SmallVector<mlir::AffineExpr> q_map_expr = {
      builder.getAffineDimExpr(0), builder.getAffineDimExpr(1),
      builder.getAffineDimExpr(2), builder.getAffineConstantExpr(0)};
  llvm::SmallVector<mlir::AffineExpr> aux_map_expr = {
      builder.getAffineDimExpr(0), builder.getAffineDimExpr(1),
      builder.getAffineConstantExpr(0), builder.getAffineDimExpr(2)};

  auto q_map = transform_indices(q_map_expr);
  auto aux_map = transform_indices(aux_map_expr);

  auto [k_map, v_map] = CreateKVWindowMaps(builder, fn, config, tiling);

  mlir::SmallVector<mlir::DictionaryAttr> window_attrs;
  window_attrs.push_back(q_map);  // Q
  window_attrs.push_back(k_map);  // K
  window_attrs.push_back(v_map);  // V

  if (config.has_attn_bias) {
    mlir::SmallVector<mlir::AffineExpr, 4> mask_expr{
        builder.getAffineDimExpr(0), builder.getAffineDimExpr(1),
        builder.getAffineDimExpr(2), builder.getAffineDimExpr(3)};
    for (auto dim : config.mask_broadcast_dims) {
      mask_expr[dim] = builder.getAffineConstantExpr(0);
    }

    window_attrs.push_back(transform_indices(mask_expr));
  }

  window_attrs.push_back(q_map);  // oi
  if (config.return_lse) {
    window_attrs.push_back(aux_map);  // lse
  }

  fn->setAttr("window_params",
              builder.getArrayAttr({window_attrs.begin(), window_attrs.end()}));
}

// The running online-softmax state for a band of query rows.
struct SoftmaxState {
  mlir::Value m;    // [rows, 128] f32 running max (log2 domain).
  mlir::Value l;    // [rows, 128] f32 running sum.
  mlir::Value acc;  // [rows, vo_head_dim] f32 unnormalized output.
};

// Kernel arguments that are shared by all sub-blocks of a grid step.
struct KernelArgs {
  mlir::Value seed_arg;  // Null when dropout is disabled.
  mlir::Value q_arg;
  mlir::Value k_arg;
  mlir::Value v_arg;
  mlir::Value mask_arg;  // Null when there is no user provided bias.
  mlir::Value oi_arg;
  mlir::Value lse_arg;  // Null when LSE is not returned.
  mlir::Value o_scratch_arg;
  mlir::Value mi_arg;
  mlir::Value li_arg;
  mlir::Value batch_idx;
  mlir::Value head_idx;
  mlir::Value row_block_idx;
  mlir::Value col_block_idx;
};

// Applies PyTorch's inverted dropout to the softmax probabilities `p`: kept
// entries are rescaled by 1 / (1 - dropout_rate) and dropped entries are
// zeroed. Returns `p` unchanged when dropout is disabled.
//
// Only the probabilities fed into the P @ V matmul are dropped; the softmax
// statistics (m_i, l_i and hence the LSE) must stay undropped because the
// backward pass recomputes P from the LSE.
mlir::Value ApplyDropoutToProbabilities(mlir::ImplicitLocOpBuilder& b,
                                        const FlashAttnConfig& config,
                                        const Tiling& tiling,
                                        const KernelArgs& args,
                                        int64_t row_offset, int64_t col_offset,
                                        mlir::Value p) {
  if (config.dropout_rate <= 0.0f) {
    return p;
  }
  mlir::VectorType p_type = mlir::cast<mlir::VectorType>(p.getType());
  mlir::Value keep = BuildDropoutKeepMask(
      b, config, tiling.qt, tiling.kt, args.batch_idx, args.head_idx,
      args.row_block_idx, args.col_block_idx, args.seed_arg, row_offset,
      col_offset, p_type.getDimSize(0), p_type.getDimSize(1));
  mlir::Value p_rescaled =
      ScaleValue(b, p, 1.0f / (1.0f - config.dropout_rate));
  mlir::Value zeros = ConstantOp::create(b, b.getZeroAttr(p_type));
  return mlir::arith::SelectOp::create(b, keep, p_rescaled, zeros);
}

// Folds the [rows, kc] logits sub-block at (row_offset, col_offset) of the
// current grid tile into the online softmax state `prev`. When `prev` is
// nullopt this is the first block for these rows, which avoids the rescaling
// of an (empty) previous state.
SoftmaxState UpdateSoftmaxState(mlir::ImplicitLocOpBuilder& b,
                                const FlashAttnConfig& config,
                                const Tiling& tiling, const KernelArgs& args,
                                mlir::Value q, int64_t row_offset, int64_t rows,
                                int64_t col_offset, int64_t kc,
                                bool apply_masking,
                                const std::optional<SoftmaxState>& prev) {
  mlir::Type ity = getElementType(*b.getContext(), config.element_type);
  mlir::FloatType f32 = b.getF32Type();

  mlir::Value k = LoadTileSlice(b, args.k_arg, col_offset, kc, /*col_offset=*/0,
                                config.qk_head_dim);
  mlir::Value sij = CreateMatmul(
      b, q, k,
      MatmulOptions{.transpose_rhs = true, .precision = config.precision});

  if (args.mask_arg) {
    mlir::Value user_bias =
        LoadTileSlice(b, args.mask_arg, row_offset, rows, col_offset, kc);
    user_bias = ConvertElementType(b, f32, user_bias);
    user_bias = ScaleValue(b, user_bias, kLog2e);
    sij = AddFOp::create(b, sij, user_bias);
  }

  if (apply_masking) {
    sij = ApplyStructuredMask(b, sij, args.row_block_idx, args.col_block_idx,
                              tiling.qt, tiling.kt, config.q_sequence_length,
                              config.kv_sequence_length, config.is_causal,
                              row_offset, col_offset);
  }

  mlir::Value mij = ReduceBroadcastLane(b, sij, CombiningKind::MAXIMUMF);
  if (args.mask_arg) {
    // Clamping mij instead of sij ensures that when a row is entirely masked
    // out with -inf, sij - mij evaluates to (-inf) - (<large negative finite>)
    // = -inf, so exp2(-inf) = 0.0 naturally without any vector compare or
    // select on p in the inner loop.
    mij = ClampMinFinite(b, mij);
  }
  mlir::Value v = LoadTileSlice(b, args.v_arg, col_offset, kc, /*col_offset=*/0,
                                config.vo_head_dim);

  SoftmaxState next;
  if (!prev.has_value()) {
    next.m = mij;
    mlir::Value p =
        Exp2Op::create(b, SubFOp::create(b, sij, NormalizeLaneDim(b, mij, kc)));
    next.l = ReduceBroadcastLane(b, p, CombiningKind::ADD);
    mlir::Value p_for_pv = ApplyDropoutToProbabilities(
        b, config, tiling, args, row_offset, col_offset, p);
    next.acc = CreateMatmul(b, ConvertElementType(b, ity, p_for_pv), v,
                            MatmulOptions{.precision = config.precision});
    return next;
  }

  next.m = MaximumFOp::create(b, prev->m, mij);
  mlir::Value alpha = Exp2Op::create(b, SubFOp::create(b, prev->m, next.m));
  mlir::Value p = Exp2Op::create(
      b, SubFOp::create(b, sij, NormalizeLaneDim(b, next.m, kc)));
  mlir::Value tile_sum = ReduceBroadcastLane(b, p, CombiningKind::ADD);
  next.l = AddFOp::create(b, MulFOp::create(b, alpha, prev->l), tile_sum);

  mlir::Value p_for_pv = ApplyDropoutToProbabilities(b, config, tiling, args,
                                                     row_offset, col_offset, p);
  mlir::Value current =
      CreateMatmul(b, ConvertElementType(b, ity, p_for_pv), v,
                   MatmulOptions{.precision = config.precision});
  mlir::Value adjustment = MulFOp::create(
      b, NormalizeLaneDim(b, alpha, config.vo_head_dim), prev->acc);
  next.acc = AddFOp::create(b, adjustment, current);
  return next;
}

// Normalizes the accumulated output for the query rows starting at
// `row_offset` and stores it (and optionally the LSE).
void StoreNormalizedOutput(mlir::ImplicitLocOpBuilder& b,
                           const FlashAttnConfig& config,
                           const KernelArgs& args, const SoftmaxState& state,
                           int64_t row_offset) {
  mlir::VectorType l_type = mlir::cast<mlir::VectorType>(state.l.getType());
  mlir::Value zero_l = ConstantOp::create(b, b.getZeroAttr(l_type));
  mlir::Value is_zero_l = mlir::arith::CmpFOp::create(
      b, mlir::arith::CmpFPredicate::OEQ, state.l, zero_l);
  mlir::Value one_l = SplatConstant(b, l_type, 1.0f);
  mlir::Value safe_l =
      mlir::arith::SelectOp::create(b, is_zero_l, one_l, state.l);

  mlir::Value li_recip = CreateReciprocal(b, safe_l);
  mlir::Value normalized_output = MulFOp::create(
      b, state.acc, NormalizeLaneDim(b, li_recip, config.vo_head_dim));
  StoreTileSlice(b, normalized_output, args.oi_arg, row_offset,
                 /*col_offset=*/0);
  if (config.return_lse) {
    // As LSE is *only* used for the backward pass, we can use the more
    // efficient log2 instead of log. I.e. this result is actually scaled by
    // log2(e) of the natural log version of LSE.
    mlir::Value ninf_val = SplatNegInf(b, l_type);
    mlir::Value lse = AddFOp::create(b, state.m, CreateLog2(b, safe_l));
    lse = mlir::arith::SelectOp::create(b, is_zero_l, ninf_val, lse);
    mlir::Value sliced_lse = NormalizeLaneDim(b, lse, 1);
    mlir::Value transposed_lse =
        mlir::vector::TransposeOp::create(b, sliced_lse, {1, 0});
    // The LSE tile is [1, qt], so the query rows are along the lane dimension.
    StoreTileSlice(b, transposed_lse, args.lse_arg, /*row_offset=*/0,
                   /*col_offset=*/row_offset);
  }
}

// Emits the compute for one grid step.
//
// The [qt, kt] grid tile is processed as a static grid of
// [band_rows, kc] sub-blocks: an outer loop over bands of query rows and an
// inner loop over KV sub-blocks. Keeping the compute block smaller than the DMA
// block bounds the size of the softmax intermediates while still amortizing
// the per grid step overhead over a large DMA block. The online softmax state
// stays in SSA values across the inner loop; the scratch buffers are only read
// once and written once per band.
//
// `has_padding`: whether the sequence lengths are not tile aligned, in which
//   case every sub-block is masked.
// `intersects_causal_diag`: whether this causal tile intersects the diagonal
//   and so requires causal masking. For square tiles (qt == kt) the query bands
//   are chosen to match `kc` so that sub-blocks strictly above the diagonal
//   (fully masked) are skipped entirely and only the sub-blocks on the diagonal
//   are masked. For non-square tiles every sub-block is masked.
void EmitGridStep(mlir::ImplicitLocOpBuilder& b, const FlashAttnConfig& config,
                  const Tiling& tiling, const KernelArgs& args,
                  bool is_single_kv_step, bool has_padding,
                  bool intersects_causal_diag) {
  const int64_t qt = tiling.qt;
  const int64_t kt = tiling.kt;
  const int64_t kc = tiling.kc;
  const int64_t num_kv_blocks = kt / kc;
  // On a square diagonal tile the row and column block indices are equal, so
  // with band_rows == kc, sub-block (band, kv_block) is fully masked iff
  // kv_block > band and needs masking iff kv_block == band.
  const bool square_causal_diag_block = intersects_causal_diag && qt == kt;
  const int64_t band_rows = square_causal_diag_block ? kc : qt;
  const int64_t num_bands = qt / band_rows;

  for (int64_t band = 0; band < num_bands; ++band) {
    const int64_t row_offset = band * band_rows;
    mlir::Value q = LoadTileSlice(b, args.q_arg, row_offset, band_rows,
                                  /*col_offset=*/0, config.qk_head_dim);
    q = ScaleValue(b, q, config.scale * kLog2e);

    std::optional<SoftmaxState> state;
    if (!is_single_kv_step) {
      state = SoftmaxState{
          .m = LoadTileSlice(b, args.mi_arg, row_offset, band_rows, 0, 128),
          .l = LoadTileSlice(b, args.li_arg, row_offset, band_rows, 0, 128),
          .acc = LoadTileSlice(b, args.o_scratch_arg, row_offset, band_rows, 0,
                               config.vo_head_dim),
      };
    }

    for (int64_t kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
      if (square_causal_diag_block && kv_block > band) {
        continue;
      }
      const bool apply_causal_mask =
          square_causal_diag_block ? kv_block == band : intersects_causal_diag;
      const bool apply_masking = has_padding || apply_causal_mask;
      state =
          UpdateSoftmaxState(b, config, tiling, args, q, row_offset, band_rows,
                             kv_block * kc, kc, apply_masking, state);
    }

    if (is_single_kv_step) {
      StoreNormalizedOutput(b, config, args, *state, row_offset);
    } else {
      StoreTileSlice(b, state->m, args.mi_arg, row_offset, 0);
      StoreTileSlice(b, state->l, args.li_arg, row_offset, 0);
      StoreTileSlice(b, state->acc, args.o_scratch_arg, row_offset, 0);
    }
  }
}

// q [batch_size, num_heads, q_seq_len, d_model]
// k [batch_size, num_heads, kv_seq_len, d_model]
// v [batch_size, num_heads, kv_seq_len, d_model]
mlir::func::FuncOp buildModule(mlir::ImplicitLocOpBuilder& module_builder,
                               const FlashAttnConfig& config,
                               const Tiling& tiling) {
  mlir::MLIRContext* context = module_builder.getContext();
  mlir::IntegerType i32 = module_builder.getIntegerType(32);
  // FlashAttnConfig::element_type is either F32 or BF16.
  mlir::Type ity = getElementType(*context, config.element_type);
  // The output element type is always F32.
  mlir::FloatType f32 = module_builder.getF32Type();
  const int64_t qt = tiling.qt;
  const int64_t kt = tiling.kt;
  const bool has_dropout = config.dropout_rate > 0.0f;

  mlir::SmallVector<int64_t> q_dims = {1, 1, qt, config.qk_head_dim};
  mlir::SmallVector<int64_t> k_dims = {1, 1, kt, config.qk_head_dim};
  mlir::SmallVector<int64_t> v_dims = {1, 1, kt, config.vo_head_dim};
  mlir::SmallVector<int64_t> oi_dims = {1, 1, qt, config.vo_head_dim};
  mlir::SmallVector<int64_t> lse_dims = {1, 1, 1, qt};
  mlir::SmallVector<int64_t> mi_dims = {qt, 128};
  mlir::SmallVector<int64_t> li_dims = {qt, 128};

  // We have 4 iteration indices.
  mlir::SmallVector<mlir::Type> inputs(4, i32);
  AppendScalarPrefetchTypes(context, config, inputs);
  // Add q, k, v.
  inputs.insert(inputs.end(), {GetVmemMemRefType(context, q_dims, ity),
                               GetVmemMemRefType(context, k_dims, ity),
                               GetVmemMemRefType(context, v_dims, ity)});
  bool has_mask = config.has_attn_bias;
  if (has_mask) {
    mlir::SmallVector<int64_t> mask_dims = {1, 1, qt, kt};
    inputs.push_back(GetVmemMemRefType(context, mask_dims, ity));
  }

  inputs.push_back(GetVmemMemRefType(context, oi_dims, ity));
  if (config.return_lse) {
    inputs.push_back(GetVmemMemRefType(context, lse_dims, f32));
  }
  //  Out scratch is always F32.
  inputs.insert(inputs.end(), {GetVmemMemRefType(context, oi_dims, f32),
                               GetVmemMemRefType(context, mi_dims, f32),
                               GetVmemMemRefType(context, li_dims, f32)});
  mlir::func::FuncOp fn =
      mlir::func::FuncOp::create(module_builder, "flash_forward",
                                 module_builder.getFunctionType(inputs, {}));
  SetTcCoreTypeAttr(fn);

  mlir::ImplicitLocOpBuilder fn_builder =
      mlir::ImplicitLocOpBuilder::atBlockBegin(module_builder.getLoc(),
                                               fn.addEntryBlock());

  // There are 4 induction variables.
  unsigned int next_index = 4;
  KernelArgs args;
  args.seed_arg = has_dropout ? fn.getArgument(next_index++) : nullptr;
  args.q_arg = fn.getArgument(next_index++);
  args.k_arg = fn.getArgument(next_index++);
  args.v_arg = fn.getArgument(next_index++);
  args.mask_arg = has_mask ? fn.getArgument(next_index++) : nullptr;
  args.oi_arg = fn.getArgument(next_index++);
  args.lse_arg = config.return_lse ? fn.getArgument(next_index++) : nullptr;
  args.o_scratch_arg = fn.getArgument(next_index++);
  args.mi_arg = fn.getArgument(next_index++);
  args.li_arg = fn.getArgument(next_index++);
  args.batch_idx = fn.getArgument(0);
  args.head_idx = fn.getArgument(1);
  args.row_block_idx = fn.getArgument(2);
  args.col_block_idx = fn.getArgument(3);

  const bool is_single_kv_step =
      (config.padded_kv_sequence_length == tiling.kt) && !config.is_causal;
  const bool has_padding =
      NeedsPadMask(tiling.qt, tiling.kt, config.q_sequence_length,
                   config.kv_sequence_length);

  ConstantOp zero = ConstantOp::create(fn_builder, fn_builder.getZeroAttr(i32));
  std::optional<mlir::scf::IfOp> causal_if;
  if (config.is_causal) {
    causal_if = CreateCausalIfOp(fn_builder, args.row_block_idx,
                                 args.col_block_idx, qt, kt);
    fn_builder.setInsertionPointToStart(causal_if->thenBlock());
  }

  if (!is_single_kv_step) {
    IfOp::create(
        fn_builder,
        CmpIOp::create(fn_builder, CmpIPredicate::eq, zero, args.col_block_idx),
        /*thenBuilder=*/
        [&](mlir::OpBuilder& builder, mlir::Location loc) -> void {
          mlir::ImplicitLocOpBuilder b(loc, builder);
          ZeroTile(b, args.o_scratch_arg);
          ZeroTile(b, args.li_arg);
          NInfTile(b, args.mi_arg);
          YieldOp::create(b, loc);
        });
  }

  if (config.is_causal) {
    // Only tiles that intersect the diagonal need masking; tiles strictly
    // below it are computed without any masking overhead.
    mlir::Value intersects_diag = CreateIntersectsDiagCondition(
        fn_builder, args.row_block_idx, args.col_block_idx, qt, kt);
    mlir::scf::IfOp::create(
        fn_builder, intersects_diag,
        [&](mlir::OpBuilder& then_builder, mlir::Location then_loc) {
          mlir::ImplicitLocOpBuilder then_implicit_b(then_loc, then_builder);
          EmitGridStep(then_implicit_b, config, tiling, args, is_single_kv_step,
                       has_padding,
                       /*intersects_causal_diag=*/true);
          YieldOp::create(then_implicit_b);
        },
        [&](mlir::OpBuilder& else_builder, mlir::Location else_loc) {
          mlir::ImplicitLocOpBuilder else_implicit_b(else_loc, else_builder);
          EmitGridStep(else_implicit_b, config, tiling, args, is_single_kv_step,
                       has_padding,
                       /*intersects_causal_diag=*/false);
          YieldOp::create(else_implicit_b);
        });
  } else {
    EmitGridStep(fn_builder, config, tiling, args, is_single_kv_step,
                 has_padding,
                 /*intersects_causal_diag=*/false);
  }

  if (is_single_kv_step) {
    ReturnOp::create(fn_builder);
    return fn;
  }

  if (causal_if.has_value()) {
    fn_builder.setInsertionPointAfter(*causal_if);
  }

  mlir::Value last_col_idx = ConstantOp::create(
      fn_builder, fn_builder.getI32IntegerAttr(
                      (config.padded_kv_sequence_length / tiling.kt) - 1));
  mlir::Value is_last_col = CmpIOp::create(fn_builder, CmpIPredicate::eq,
                                           args.col_block_idx, last_col_idx);
  IfOp::create(fn_builder, is_last_col,
               /*thenBuilder=*/
               [&](mlir::OpBuilder& builder, mlir::Location loc) -> void {
                 mlir::ImplicitLocOpBuilder b(loc, builder);
                 SoftmaxState state{
                     .m = LoadTile(b, args.mi_arg),
                     .l = LoadTile(b, args.li_arg),
                     .acc = LoadTile(b, args.o_scratch_arg),
                 };
                 StoreNormalizedOutput(b, config, args, state,
                                       /*row_offset=*/0);
                 YieldOp::create(b, loc);
               });

  ReturnOp::create(fn_builder);

  return fn;
}
}  // namespace

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateKernel(
    mlir::MLIRContext* context, const FlashAttnConfig& config,
    const Tiling& tiling) {
  mlir::OpBuilder builder(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder, builder.getUnknownLoc());
  mlir::ImplicitLocOpBuilder module_builder(module->getLoc(),
                                            module->getBodyRegion());
  if (config.padded_q_sequence_length % tiling.qt != 0 ||
      config.padded_kv_sequence_length % tiling.kt != 0) {
    return TT_ERROR(::torch_tpu::error::kInvalidArgument)
           << "Padded sequence lengths must be divisible by tile sizes.";
  }

  if (config.num_heads % config.kv_num_heads != 0) {
    return TT_ERROR(::torch_tpu::error::kInvalidArgument)
           << "num_heads must be divisible by kv_num_heads.";
  }

  if (config.dropout_rate < 0.0f || config.dropout_rate >= 1.0f) {
    return TT_ERROR(::torch_tpu::error::kInvalidArgument)
           << "dropout_rate must be in [0.0, 1.0), got " << config.dropout_rate;
  }

  Tiling normalized_tiling = tiling;
  if (normalized_tiling.kc <= 0) {
    normalized_tiling.kc = normalized_tiling.kt;
  }
  if (normalized_tiling.kt % normalized_tiling.kc != 0 ||
      normalized_tiling.kc % 128 != 0) {
    return TT_ERROR(::torch_tpu::error::kInvalidArgument)
           << "KV compute tile size must be a multiple of 128 that divides "
              "the KV tile size.";
  }

  auto fn = buildModule(module_builder, config, normalized_tiling);

  absl::InlinedVector<int64_t, 4> q_shape = {
      config.batch_size, config.num_heads, config.q_sequence_length,
      config.qk_head_dim};

  SetKernelAttributes(q_shape, config, normalized_tiling, fn, builder);

  ABSL_VLOG(1) << "Forward kernel:\n" << GetOpString(module.get());

  return module;
}

}  // namespace torch_tpu
