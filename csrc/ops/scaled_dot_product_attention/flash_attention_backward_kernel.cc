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

#include <cstdint>
#include <initializer_list>
#include <numbers>
#include <optional>

#include "absl/functional/any_invocable.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "csrc/common/error_utils.h"
#include "csrc/internal/mosaic/op_builders.h"
#include "csrc/ops/scaled_dot_product_attention/flash_attention_config.h"
#include "csrc/ops/scaled_dot_product_attention/flash_attention_kernel.h"
#include "csrc/ops/scaled_dot_product_attention/util.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

using mlir::arith::AddFOp;         // USING_DECL_OK
using mlir::arith::AndIOp;         // USING_DECL_OK
using mlir::arith::CmpFOp;         // USING_DECL_OK
using mlir::arith::CmpFPredicate;  // USING_DECL_OK
using mlir::arith::CmpIOp;         // USING_DECL_OK
using mlir::arith::CmpIPredicate;  // USING_DECL_OK
using mlir::arith::ConstantOp;     // USING_DECL_OK
using mlir::arith::MulFOp;         // USING_DECL_OK
using mlir::arith::SelectOp;       // USING_DECL_OK
using mlir::arith::SubFOp;         // USING_DECL_OK
using mlir::func::ReturnOp;        // USING_DECL_OK
using mlir::math::Exp2Op;          // USING_DECL_OK
using mlir::scf::IfOp;             // USING_DECL_OK
using mlir::scf::YieldOp;          // USING_DECL_OK
using mlir::vector::BroadcastOp;   // USING_DECL_OK
using mlir::vector::ShapeCastOp;   // USING_DECL_OK

struct SharedBwdResult {
  mlir::Value p;
  mlir::Value ds_trunc;
};

SharedBwdResult ComputeSharedBackwardLogic(
    mlir::ImplicitLocOpBuilder& fn_builder, const FlashAttnConfig& config,
    int64_t qt, int64_t kt, mlir::Value q_2d, mlir::Value k_2d,
    mlir::Value v_2d, mlir::Value do_2d, mlir::Value lse_tile,
    mlir::Value di_tile, mlir::Value user_bias, mlir::Value batch_idx,
    mlir::Value q_head_idx, mlir::Value row_block_idx,
    mlir::Value col_block_idx, mlir::Value seed_arg, mlir::Type oty,
    mlir::Type ity, mlir::IntegerType i32, mlir::MLIRContext* context) {
  constexpr float kLog2e = std::numbers::log2e_v<float>;
  mlir::Type vector_type_qt_kt = mlir::VectorType::get({qt, kt}, oty);
  mlir::Value unscaled_sij = CreateMatmul(
      fn_builder, q_2d, k_2d,
      MatmulOptions{.transpose_rhs = true, .precision = config.precision});

  mlir::Value sij = ScaleValue(fn_builder, unscaled_sij, config.scale * kLog2e);

  if (user_bias) {
    user_bias = ConvertElementType(fn_builder, oty, user_bias);
    user_bias = ScaleValue(fn_builder, user_bias, kLog2e);
    sij = AddFOp::create(fn_builder, sij, user_bias);
  }

  sij = ApplyStructuredMask(fn_builder, sij, row_block_idx, col_block_idx, qt,
                            kt, config.q_sequence_length,
                            config.kv_sequence_length, config.is_causal);

  mlir::Value lse_broadcast =
      BroadcastOp::create(fn_builder, vector_type_qt_kt, lse_tile);

  mlir::Value ninf_val = SplatNegInf(fn_builder, vector_type_qt_kt);
  mlir::Value is_lse_ninf = mlir::arith::CmpFOp::create(
      fn_builder, mlir::arith::CmpFPredicate::OEQ, lse_broadcast, ninf_val);
  mlir::Value zero_p =
      ConstantOp::create(fn_builder, fn_builder.getZeroAttr(vector_type_qt_kt));
  mlir::Value safe_lse = mlir::arith::SelectOp::create(fn_builder, is_lse_ninf,
                                                       zero_p, lse_broadcast);
  mlir::Value sub_op = SubFOp::create(fn_builder, sij, safe_lse);
  mlir::Value pij = Exp2Op::create(fn_builder, sub_op);
  pij = mlir::arith::SelectOp::create(fn_builder, is_lse_ninf, zero_p, pij);

  // Compute dP = dO @ V^T!
  mlir::Value dpij = CreateMatmul(
      fn_builder, do_2d, v_2d,
      MatmulOptions{.transpose_rhs = true, .precision = config.precision});

  mlir::Value p_for_dv = pij;
  if (config.dropout_rate > 0.0f) {
    mlir::Value keep =
        BuildDropoutKeepMask(fn_builder, config, qt, kt, batch_idx, q_head_idx,
                             row_block_idx, col_block_idx, seed_arg);
    float inv_keep_prob = 1.0f / (1.0f - config.dropout_rate);
    mlir::Value zeros =
        ConstantOp::create(fn_builder, fn_builder.getZeroAttr(pij.getType()));
    p_for_dv = mlir::arith::SelectOp::create(
        fn_builder, keep, ScaleValue(fn_builder, pij, inv_keep_prob), zeros);
    dpij = mlir::arith::SelectOp::create(
        fn_builder, keep, ScaleValue(fn_builder, dpij, inv_keep_prob), zeros);
  }

  // Compute ds = P * (dP - di)!
  mlir::Value di_broadcast =
      BroadcastOp::create(fn_builder, vector_type_qt_kt, di_tile);

  mlir::Value dp_sub_di = SubFOp::create(fn_builder, dpij, di_broadcast);
  mlir::Value ds = MulFOp::create(fn_builder, pij, dp_sub_di);
  ds = ConvertElementType(fn_builder, ity, ds);

  return {p_for_dv, ds};
}

struct BwdArgIndices {
  int seed = -1;
  int q;
  int k;
  int v;
  int do_arg;
  int lse;
  int di;
  int mask = -1;
  int next_idx;
};

BwdArgIndices GetBwdArgIndices(const FlashAttnConfig& config,
                               int num_iteration_dims = 4) {
  BwdArgIndices indices;
  int arg_idx = num_iteration_dims;
  if (config.dropout_rate > 0.0f) {
    indices.seed = arg_idx++;
  }
  indices.q = arg_idx++;
  indices.k = arg_idx++;
  indices.v = arg_idx++;
  indices.do_arg = arg_idx++;
  indices.lse = arg_idx++;
  indices.di = arg_idx++;
  if (config.has_attn_bias) {
    indices.mask = arg_idx++;
  }
  indices.next_idx = arg_idx;
  return indices;
}

struct LoadedTiles {
  mlir::Value lse_tile;
  mlir::Value di_tile;
  mlir::Value mask_tile;
  mlir::Value q_tile;
  mlir::Value k_tile;
  mlir::Value v_tile;
  mlir::Value do_2d;
};

LoadedTiles LoadAndCastTiles(mlir::ImplicitLocOpBuilder& fn_builder,
                             mlir::func::FuncOp fn,
                             const FlashAttnConfig& config,
                             const Tiling& tiling, mlir::Type oty,
                             const BwdArgIndices& arg_indices) {
  int64_t qt = tiling.qt;

  LoadedTiles tiles;
  tiles.q_tile = LoadTile(fn_builder, fn.getArgument(arg_indices.q));
  tiles.k_tile = LoadTile(fn_builder, fn.getArgument(arg_indices.k));
  tiles.v_tile = LoadTile(fn_builder, fn.getArgument(arg_indices.v));
  tiles.do_2d = LoadTile(fn_builder, fn.getArgument(arg_indices.do_arg));
  auto lse_loaded = LoadTile(fn_builder, fn.getArgument(arg_indices.lse));
  tiles.lse_tile = ShapeCastOp::create(
      fn_builder, mlir::VectorType::get({qt, 1}, oty), lse_loaded);

  auto di_loaded = LoadTile(fn_builder, fn.getArgument(arg_indices.di));
  tiles.di_tile = ShapeCastOp::create(
      fn_builder, mlir::VectorType::get({qt, 1}, oty), di_loaded);

  if (config.has_attn_bias) {
    tiles.mask_tile = LoadTile(fn_builder, fn.getArgument(arg_indices.mask));
  }

  return tiles;
}

mlir::func::FuncOp buildBackwardDkvModule(
    mlir::ImplicitLocOpBuilder& module_builder, const FlashAttnConfig& config,
    const Tiling& tiling) {
  mlir::MLIRContext* context = module_builder.getContext();
  mlir::Type ity = getElementType(*context, config.element_type);
  mlir::Type oty = module_builder.getF32Type();

  const int64_t qt = tiling.qt;
  const int64_t kt = tiling.kt;

  mlir::SmallVector<int64_t> q_dims = {1, 1, qt, config.qk_head_dim};
  mlir::SmallVector<int64_t> k_dims = {1, 1, kt, config.qk_head_dim};
  mlir::SmallVector<int64_t> v_dims = {1, 1, kt, config.vo_head_dim};
  mlir::SmallVector<int64_t> do_dims = {1, 1, qt, config.vo_head_dim};
  mlir::SmallVector<int64_t> lse_dims = {1, 1, 1, qt};
  mlir::SmallVector<int64_t> di_dims = {1, 1, 1, qt};

  // Inputs list!
  mlir::SmallVector<mlir::Type> inputs(5, module_builder.getI32Type());
  AppendScalarPrefetchTypes(context, config, inputs);
  inputs.push_back(GetVmemMemRefType(context, q_dims, ity));
  inputs.push_back(GetVmemMemRefType(context, k_dims, ity));
  inputs.push_back(GetVmemMemRefType(context, v_dims, ity));
  inputs.push_back(GetVmemMemRefType(context, do_dims, ity));
  inputs.push_back(GetVmemMemRefType(context, lse_dims, oty));
  inputs.push_back(GetVmemMemRefType(context, di_dims, oty));

  if (config.has_attn_bias) {
    mlir::SmallVector<int64_t> mask_dims = {1, 1, qt, kt};
    inputs.push_back(GetVmemMemRefType(context, mask_dims, ity));
  }

  // Outputs! (dk, dv)
  mlir::SmallVector<int64_t> dk_dims = {1, 1, kt, config.qk_head_dim};
  mlir::SmallVector<int64_t> dv_dims = {1, 1, kt, config.vo_head_dim};

  inputs.push_back(GetVmemMemRefType(context, dk_dims, ity));
  inputs.push_back(GetVmemMemRefType(context, dv_dims, ity));
  // Scratch outputs!
  inputs.push_back(GetVmemMemRefType(context, dk_dims, oty));
  inputs.push_back(GetVmemMemRefType(context, dv_dims, oty));

  BwdArgIndices arg_indices =
      GetBwdArgIndices(config, /*num_iteration_dims=*/5);
  int arg_idx = arg_indices.next_idx;
  int dk_bf16_idx = arg_idx++;
  int dv_bf16_idx = arg_idx++;

  int dk_f32_idx = arg_idx++;
  int dv_f32_idx = arg_idx++;

  mlir::func::FuncOp fn =
      mlir::func::FuncOp::create(module_builder, "flash_backward_dkv",
                                 module_builder.getFunctionType(inputs, {}));
  SetTcCoreTypeAttr(fn);

  mlir::ImplicitLocOpBuilder fn_builder =
      mlir::ImplicitLocOpBuilder::atBlockBegin(module_builder.getLoc(),
                                               fn.addEntryBlock());

  ConstantOp zero = ConstantOp::create(
      fn_builder, fn_builder.getZeroAttr(fn_builder.getI32Type()));

  mlir::Value is_first_q_tile =
      CmpIOp::create(fn_builder, CmpIPredicate::eq, zero, fn.getArgument(3));
  mlir::Value is_first_head_group =
      CmpIOp::create(fn_builder, CmpIPredicate::eq, zero, fn.getArgument(4));
  mlir::Value is_first_reduction_step =
      AndIOp::create(fn_builder, is_first_q_tile, is_first_head_group);

  IfOp::create(fn_builder, is_first_reduction_step,
               /*thenBuilder=*/
               [&](mlir::OpBuilder& builder, mlir::Location loc) -> void {
                 mlir::ImplicitLocOpBuilder b(loc, builder);
                 ZeroTile(b, fn.getArgument(dk_f32_idx));
                 ZeroTile(b, fn.getArgument(dv_f32_idx));
                 YieldOp::create(b, loc);
               });

  mlir::Value kv_tile_idx = fn.getArgument(2);
  mlir::Value q_tile_idx = fn.getArgument(3);

  std::optional<IfOp> causal_if;
  if (config.is_causal) {
    causal_if = CreateCausalIfOp(fn_builder, q_tile_idx, kv_tile_idx, qt, kt);
    fn_builder.setInsertionPointToStart(causal_if->thenBlock());
  }

  LoadedTiles tiles =
      LoadAndCastTiles(fn_builder, fn, config, tiling, oty, arg_indices);

  mlir::Value batch_idx = fn.getArgument(0);
  mlir::Value q_head_idx = nullptr;
  mlir::Value seed_arg = nullptr;
  if (config.dropout_rate > 0.0f) {
    mlir::Value kv_head_idx = fn.getArgument(1);
    mlir::Value head_in_group = fn.getArgument(4);
    int64_t head_group_size = config.num_heads / config.kv_num_heads;
    mlir::Value group_size_val = mlir::arith::ConstantOp::create(
        fn_builder, fn_builder.getI32IntegerAttr(head_group_size));
    q_head_idx = mlir::arith::AddIOp::create(
        fn_builder,
        mlir::arith::MulIOp::create(fn_builder, kv_head_idx, group_size_val),
        head_in_group);
    seed_arg = fn.getArgument(arg_indices.seed);
  }

  auto shared_result = ComputeSharedBackwardLogic(
      fn_builder, config, qt, kt, tiles.q_tile, tiles.k_tile, tiles.v_tile,
      tiles.do_2d, tiles.lse_tile, tiles.di_tile, tiles.mask_tile, batch_idx,
      q_head_idx, q_tile_idx, kv_tile_idx, seed_arg, oty, ity,
      module_builder.getI32Type(), context);

  mlir::Value p_contract = ConvertElementType(fn_builder, ity, shared_result.p);

  auto dv_contraction = CreateMatmul(
      fn_builder, p_contract, tiles.do_2d,
      MatmulOptions{.transpose_lhs = true, .precision = config.precision});

  auto dk_contraction = CreateMatmul(
      fn_builder, shared_result.ds_trunc, tiles.q_tile,
      MatmulOptions{.transpose_lhs = true, .precision = config.precision});

  mlir::Value scaled_dk = ScaleValue(fn_builder, dk_contraction, config.scale);

  auto prev_dk = LoadTile(fn_builder, fn.getArgument(dk_f32_idx));
  auto updated_dk = AddFOp::create(fn_builder, prev_dk, scaled_dk);
  StoreTile(fn_builder, updated_dk, fn.getArgument(dk_f32_idx));

  auto prev_dv = LoadTile(fn_builder, fn.getArgument(dv_f32_idx));
  auto updated_dv = AddFOp::create(fn_builder, prev_dv, dv_contraction);
  StoreTile(fn_builder, updated_dv, fn.getArgument(dv_f32_idx));

  StoreTile(fn_builder, updated_dk, fn.getArgument(dk_bf16_idx));
  StoreTile(fn_builder, updated_dv, fn.getArgument(dv_bf16_idx));

  if (causal_if.has_value()) {
    fn_builder.setInsertionPointAfter(*causal_if);
  }
  mlir::func::ReturnOp::create(fn_builder);
  return fn;
}
void SetBackwardDkvKernelAttributes(const FlashAttnConfig& config,
                                    const Tiling& tiling, mlir::func::FuncOp fn,
                                    mlir::OpBuilder& builder) {
  mlir::MLIRContext* context = builder.getContext();
  int64_t num_kv_tiles = config.padded_kv_sequence_length / tiling.kt;
  int64_t num_q_tiles = config.padded_q_sequence_length / tiling.qt;
  int64_t head_group_size = config.num_heads / config.kv_num_heads;

  mlir::SmallVector<int64_t> iteration_bounds = {
      config.batch_size, config.kv_num_heads, num_kv_tiles, num_q_tiles,
      head_group_size};
  fn->setAttr("iteration_bounds",
              builder.getDenseI64ArrayAttr(iteration_bounds));

  mlir::SmallVector<mlir::Attribute> dimension_semantics(
      3, GetDimensionSemanticsAttr(context, DimensionSemantics::kParallel));
  dimension_semantics.push_back(
      GetDimensionSemanticsAttr(context, DimensionSemantics::kArbitrary));
  dimension_semantics.push_back(
      GetDimensionSemanticsAttr(context, DimensionSemantics::kArbitrary));
  fn->setAttr("dimension_semantics", builder.getArrayAttr(dimension_semantics));

  auto transform_indices = [&](mlir::AffineMap map) {
    return builder.getDictionaryAttr({builder.getNamedAttr(
        "transform_indices", mlir::AffineMapAttr::get(map))});
  };

  // Iteration indices: (d0, d1, d2, d3, d4) = (batch, kv_head, kv_tile, q_tile,
  // head_group)
  mlir::AffineExpr batch_itr = builder.getAffineDimExpr(0);
  mlir::AffineExpr kv_head_itr = builder.getAffineDimExpr(1);
  mlir::AffineExpr kv_seq_itr = builder.getAffineDimExpr(2);
  mlir::AffineExpr constant_0 = builder.getAffineConstantExpr(0);

  // K, V, dK, dV always index: (batch, kv_head, kv_tile, 0)
  auto kv_map = mlir::AffineMap::get(
      5, 0, {batch_itr, kv_head_itr, kv_seq_itr, constant_0}, context);
  mlir::DictionaryAttr k_map = transform_indices(kv_map);
  mlir::DictionaryAttr v_map = k_map;
  mlir::DictionaryAttr dk_map = k_map;
  mlir::DictionaryAttr dv_map = k_map;

  // GQA / MQA: Mosaic AffineMap doesn't support Dim * Const + Dim, so use
  // transform functions.
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(fn);
  mlir::Location loc = fn.getLoc();
  mlir::Type i32 = builder.getI32Type();
  // The 5 iteration indices, followed by the scalar_prefetch operands, which
  // Mosaic requires to be present even though these functions ignore them.
  mlir::SmallVector<mlir::Type> input_types(5, i32);
  AppendScalarPrefetchTypes(context, config, input_types);
  mlir::SmallVector<mlir::Type> output_types(4, i32);
  mlir::FunctionType fn_type =
      builder.getFunctionType(input_types, output_types);

  auto create_transform_fn =
      [&](llvm::StringRef name,
          absl::AnyInvocable<mlir::SmallVector<mlir::Value, 4>(
              mlir::OpBuilder&, mlir::Value batch, mlir::Value q_head,
              mlir::Value q_tile, mlir::Value kv_tile, mlir::Value c0)>
              get_results) {
        auto transform_fn =
            mlir::func::FuncOp::create(builder, loc, name, fn_type);
        transform_fn.setPrivate();
        mlir::Block* entry = transform_fn.addEntryBlock();
        mlir::OpBuilder fn_b = mlir::OpBuilder::atBlockBegin(entry);
        mlir::Value batch = entry->getArgument(0);
        mlir::Value kv_head = entry->getArgument(1);
        mlir::Value kv_tile = entry->getArgument(2);
        mlir::Value q_tile = entry->getArgument(3);
        mlir::Value head_in_group = entry->getArgument(4);

        mlir::Value group_size_val = mlir::arith::ConstantOp::create(
            fn_b, loc, fn_b.getI32IntegerAttr(head_group_size));
        mlir::Value kv_offset =
            mlir::arith::MulIOp::create(fn_b, loc, kv_head, group_size_val);
        mlir::Value q_head =
            mlir::arith::AddIOp::create(fn_b, loc, kv_offset, head_in_group);

        mlir::Value c0 = mlir::arith::ConstantOp::create(
            fn_b, loc, fn_b.getI32IntegerAttr(0));
        mlir::SmallVector<mlir::Value, 4> results =
            get_results(fn_b, batch, q_head, q_tile, kv_tile, c0);
        mlir::func::ReturnOp::create(fn_b, loc, results);
        return transform_fn;
      };

  create_transform_fn(
      "q_transform_indices",
      [](mlir::OpBuilder&, mlir::Value batch, mlir::Value q_head,
         mlir::Value q_tile, mlir::Value kv_tile,
         mlir::Value c0) -> mlir::SmallVector<mlir::Value, 4> {
        return {batch, q_head, q_tile, c0};
      });
  mlir::DictionaryAttr q_map_attr = CreateSymbolTransformIndicesAttr(
      builder, "q_transform_indices", {1, 1, tiling.qt, config.qk_head_dim});
  mlir::DictionaryAttr do_map_attr = CreateSymbolTransformIndicesAttr(
      builder, "q_transform_indices", {1, 1, tiling.qt, config.vo_head_dim});

  create_transform_fn(
      "aux_transform_indices",
      [](mlir::OpBuilder&, mlir::Value batch, mlir::Value q_head,
         mlir::Value q_tile, mlir::Value kv_tile,
         mlir::Value c0) -> mlir::SmallVector<mlir::Value, 4> {
        return {batch, q_head, c0, q_tile};
      });
  mlir::DictionaryAttr aux_map_attr = CreateSymbolTransformIndicesAttr(
      builder, "aux_transform_indices", {1, 1, 1, tiling.qt});

  mlir::DictionaryAttr lse_map_attr = aux_map_attr;
  mlir::DictionaryAttr di_map_attr = aux_map_attr;

  mlir::DictionaryAttr mask_map_attr;
  if (config.has_attn_bias) {
    create_transform_fn(
        "mask_transform_indices",
        [&](mlir::OpBuilder&, mlir::Value batch, mlir::Value q_head,
            mlir::Value q_tile, mlir::Value kv_tile,
            mlir::Value c0) -> mlir::SmallVector<mlir::Value, 4> {
          mlir::SmallVector<mlir::Value, 4> coords = {batch, q_head, q_tile,
                                                      kv_tile};
          for (auto dim : config.mask_broadcast_dims) {
            coords[dim] = c0;
          }
          return coords;
        });
    mask_map_attr = CreateSymbolTransformIndicesAttr(
        builder, "mask_transform_indices", {1, 1, tiling.qt, tiling.kt});
  }

  mlir::SmallVector<mlir::Attribute> window_attrs;
  window_attrs.push_back(q_map_attr);    // Q
  window_attrs.push_back(k_map);         // K
  window_attrs.push_back(v_map);         // V
  window_attrs.push_back(do_map_attr);   // dO
  window_attrs.push_back(lse_map_attr);  // lse
  window_attrs.push_back(di_map_attr);   // di
  if (config.has_attn_bias) {
    window_attrs.push_back(mask_map_attr);  // mask
  }
  window_attrs.push_back(dk_map);  // dK
  window_attrs.push_back(dv_map);  // dV

  fn->setAttr("window_params", builder.getArrayAttr(window_attrs));
  fn->setAttr("scratch_operands", builder.getI64IntegerAttr(2));
  SetScalarPrefetchAttr(config, fn, builder);
}

void SetBackwardDqKernelAttributes(const FlashAttnConfig& config,
                                   const Tiling& tiling, mlir::func::FuncOp fn,
                                   mlir::OpBuilder& builder) {
  mlir::MLIRContext* context = builder.getContext();

  mlir::SmallVector<int64_t> iteration_bounds = {
      config.batch_size, config.num_heads,
      config.padded_q_sequence_length / tiling.qt,
      config.padded_kv_sequence_length / tiling.kt};

  int batch_idx = 0;
  int heads_idx = 1;
  int q_seq_idx = 2;
  int kv_seq_idx = 3;

  fn->setAttr("iteration_bounds",
              builder.getDenseI64ArrayAttr(iteration_bounds));

  mlir::SmallVector<mlir::Attribute> dimension_semantics(
      3, GetDimensionSemanticsAttr(context, DimensionSemantics::kParallel));
  dimension_semantics.push_back(
      GetDimensionSemanticsAttr(context, DimensionSemantics::kArbitrary));

  fn->setAttr("dimension_semantics", builder.getArrayAttr(dimension_semantics));

  auto transform_indices = [&](mlir::AffineMap map) {
    return builder.getDictionaryAttr({builder.getNamedAttr(
        "transform_indices", mlir::AffineMapAttr::get(map))});
  };

  mlir::AffineExpr batch_itr = builder.getAffineDimExpr(batch_idx);
  mlir::AffineExpr heads_itr = builder.getAffineDimExpr(heads_idx);
  mlir::AffineExpr q_seq_itr = builder.getAffineDimExpr(q_seq_idx);
  mlir::AffineExpr kv_seq_itr = builder.getAffineDimExpr(kv_seq_idx);
  mlir::AffineExpr constant_0 = builder.getAffineConstantExpr(0);

  llvm::SmallVector<mlir::AffineExpr> q_map_expr = {batch_itr, heads_itr,
                                                    q_seq_itr, constant_0};
  llvm::SmallVector<mlir::AffineExpr> aux_map_expr = {batch_itr, heads_itr,
                                                      constant_0, q_seq_itr};

  auto q_map = mlir::AffineMap::get(4, 0, q_map_expr, context);
  auto aux_map = mlir::AffineMap::get(4, 0, aux_map_expr, context);

  auto [k_map, v_map] = CreateKVWindowMaps(builder, fn, config, tiling);

  mlir::SmallVector<mlir::Attribute> window_attrs;
  window_attrs.push_back(transform_indices(q_map));    // Q
  window_attrs.push_back(k_map);                       // K
  window_attrs.push_back(v_map);                       // V
  window_attrs.push_back(transform_indices(q_map));    // dO
  window_attrs.push_back(transform_indices(aux_map));  // lse
  window_attrs.push_back(transform_indices(aux_map));  // di

  if (config.has_attn_bias) {
    llvm::SmallVector<mlir::AffineExpr, 4> mask_expr = {batch_itr, heads_itr,
                                                        q_seq_itr, kv_seq_itr};
    for (auto dim : config.mask_broadcast_dims) {
      mask_expr[dim] = builder.getAffineConstantExpr(0);
    }
    mlir::AffineMap mask_map = mlir::AffineMap::get(4, 0, mask_expr, context);
    window_attrs.push_back(transform_indices(mask_map));
  }

  window_attrs.push_back(transform_indices(q_map));  // Out
  window_attrs.push_back(transform_indices(q_map));  // dQ

  fn->setAttr("window_params", builder.getArrayAttr(window_attrs));
  fn->setAttr("scratch_operands", builder.getI64IntegerAttr(1));
  SetScalarPrefetchAttr(config, fn, builder);
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateBackwardDkvKernel(
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

  auto fn = buildBackwardDkvModule(module_builder, config, tiling);
  SetBackwardDkvKernelAttributes(config, tiling, fn, builder);

  ABSL_VLOG(1) << "Backward dKV kernel:\n" << GetOpString(module.get());

  return module;
}

mlir::func::FuncOp buildBackwardDqModule(
    mlir::ImplicitLocOpBuilder& module_builder, const FlashAttnConfig& config,
    const Tiling& tiling) {
  auto context = module_builder.getContext();
  mlir::Type ity = getElementType(*context, config.element_type);
  auto oty = module_builder.getF32Type();

  int64_t qt = tiling.qt;
  int64_t kt = tiling.kt;

  mlir::SmallVector<int64_t> q_dims = {1, 1, qt, config.qk_head_dim};
  mlir::SmallVector<int64_t> k_dims = {1, 1, kt, config.qk_head_dim};
  mlir::SmallVector<int64_t> v_dims = {1, 1, kt, config.vo_head_dim};
  mlir::SmallVector<int64_t> do_dims = {1, 1, qt, config.vo_head_dim};
  mlir::SmallVector<int64_t> lse_dims = {1, 1, 1, qt};
  mlir::SmallVector<int64_t> di_dims = {1, 1, 1, qt};
  mlir::SmallVector<int64_t> out_dims = {1, 1, qt, config.vo_head_dim};

  mlir::SmallVector<int64_t> dq_dims = {1, 1, qt, config.qk_head_dim};

  // Inputs list!
  mlir::SmallVector<mlir::Type> inputs(4, module_builder.getI32Type());
  AppendScalarPrefetchTypes(context, config, inputs);
  inputs.push_back(GetVmemMemRefType(context, q_dims, ity));
  inputs.push_back(GetVmemMemRefType(context, k_dims, ity));
  inputs.push_back(GetVmemMemRefType(context, v_dims, ity));
  inputs.push_back(GetVmemMemRefType(context, do_dims, ity));
  inputs.push_back(GetVmemMemRefType(context, lse_dims, oty));
  inputs.push_back(GetVmemMemRefType(context, di_dims, oty));

  if (config.has_attn_bias) {
    mlir::SmallVector<int64_t> mask_dims = {1, 1, qt, kt};
    inputs.push_back(GetVmemMemRefType(context, mask_dims, ity));
  }

  inputs.push_back(GetVmemMemRefType(context, out_dims, ity));
  inputs.push_back(GetVmemMemRefType(context, dq_dims, ity));
  // scratch dq
  inputs.push_back(GetVmemMemRefType(context, dq_dims, oty));

  BwdArgIndices arg_indices = GetBwdArgIndices(config);
  int arg_idx = arg_indices.next_idx;
  arg_idx++;  // out_idx
  int dq_bf16_idx = arg_idx++;
  int dq_f32_idx = arg_idx++;

  auto fn = mlir::func::FuncOp::create(
      module_builder, module_builder.getUnknownLoc(), "flash_backward_dq",
      module_builder.getFunctionType(inputs, {}));
  SetTcCoreTypeAttr(fn);

  auto fn_builder = mlir::ImplicitLocOpBuilder::atBlockBegin(
      module_builder.getLoc(), fn.addEntryBlock());

  IfOp::create(fn_builder,
               CmpIOp::create(
                   fn_builder, CmpIPredicate::eq,
                   ConstantOp::create(fn_builder, fn_builder.getZeroAttr(
                                                      fn_builder.getI32Type())),
                   fn.getArgument(3)),
               /*thenBuilder=*/
               [&](mlir::OpBuilder& builder, mlir::Location loc) -> void {
                 mlir::ImplicitLocOpBuilder b(loc, builder);
                 ZeroTile(b, fn.getArgument(dq_f32_idx));
                 YieldOp::create(builder, loc);
               });

  std::optional<IfOp> causal_if;
  if (config.is_causal) {
    causal_if = CreateCausalIfOp(fn_builder, fn.getArgument(2),
                                 fn.getArgument(3), qt, kt);
    fn_builder.setInsertionPointToStart(causal_if->thenBlock());
  }

  LoadedTiles tiles =
      LoadAndCastTiles(fn_builder, fn, config, tiling, oty, arg_indices);

  mlir::Value batch_idx = fn.getArgument(0);
  mlir::Value q_head_idx = fn.getArgument(1);
  mlir::Value q_tile_idx = fn.getArgument(2);
  mlir::Value kv_tile_idx = fn.getArgument(3);
  mlir::Value seed_arg =
      config.dropout_rate > 0.0f ? fn.getArgument(arg_indices.seed) : nullptr;

  auto shared_result = ComputeSharedBackwardLogic(
      fn_builder, config, qt, kt, tiles.q_tile, tiles.k_tile, tiles.v_tile,
      tiles.do_2d, tiles.lse_tile, tiles.di_tile, tiles.mask_tile, batch_idx,
      q_head_idx, q_tile_idx, kv_tile_idx, seed_arg, oty, ity,
      module_builder.getI32Type(), context);

  // Compute dQ = dS @ K!
  auto dq_contraction =
      CreateMatmul(fn_builder, shared_result.ds_trunc, tiles.k_tile,
                   MatmulOptions{.precision = config.precision});

  mlir::Value scaled_dq = ScaleValue(fn_builder, dq_contraction, config.scale);

  mlir::Value prev_dq = LoadTile(fn_builder, fn.getArgument(dq_f32_idx));
  mlir::Value updated_dq = AddFOp::create(fn_builder, prev_dq, scaled_dq);
  StoreTile(fn_builder, updated_dq, fn.getArgument(dq_f32_idx));

  StoreTile(fn_builder, updated_dq, fn.getArgument(dq_bf16_idx));

  if (causal_if.has_value()) {
    fn_builder.setInsertionPointAfter(*causal_if);
  }
  mlir::func::ReturnOp::create(fn_builder);
  return fn;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateBackwardDqKernel(
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

  auto fn = buildBackwardDqModule(module_builder, config, tiling);
  SetBackwardDqKernelAttributes(config, tiling, fn, builder);

  ABSL_VLOG(1) << "Backward dQ kernel:\n" << GetOpString(module.get());

  return module;
}

}  // namespace torch_tpu
