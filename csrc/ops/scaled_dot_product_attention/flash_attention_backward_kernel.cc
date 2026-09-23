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

namespace mlir::torch_tpu {

using mlir::arith::AddFOp;         // USING_DECL_OK
using mlir::arith::AndIOp;         // USING_DECL_OK
using mlir::arith::CmpIOp;         // USING_DECL_OK
using mlir::arith::CmpIPredicate;  // USING_DECL_OK
using mlir::arith::ConstantOp;     // USING_DECL_OK
using mlir::arith::MulFOp;         // USING_DECL_OK
using mlir::arith::SubFOp;         // USING_DECL_OK
using mlir::func::ReturnOp;        // USING_DECL_OK
using mlir::math::Exp2Op;          // USING_DECL_OK
using mlir::scf::IfOp;             // USING_DECL_OK
using mlir::scf::YieldOp;          // USING_DECL_OK
using mlir::vector::BroadcastOp;   // USING_DECL_OK
using mlir::vector::ShapeCastOp;   // USING_DECL_OK

struct SharedBwdResult {
  Value p;
  Value ds_trunc;
};

SharedBwdResult ComputeSharedBackwardLogic(
    ImplicitLocOpBuilder& fn_builder, const FlashAttnConfig& config, int64_t qt,
    int64_t kt, Value q_2d, Value k_2d, Value v_2d, Value do_2d, Value lse_tile,
    Value di_tile, Value user_bias, Value row_block_idx, Value col_block_idx,
    Type oty, Type ity, IntegerType i32, MLIRContext* context) {
  constexpr float kLog2e = std::numbers::log2e_v<float>;
  Type vector_type_qt_kt = VectorType::get({qt, kt}, oty);
  Value unscaled_sij = CreateMatmul(fn_builder, q_2d, k_2d,
                                    MatmulOptions{.transpose_rhs = true});

  Value sij = ScaleValue(fn_builder, unscaled_sij, config.scale * kLog2e);

  if (user_bias) {
    user_bias = ConvertElementType(fn_builder, oty, user_bias);
    user_bias = ScaleValue(fn_builder, user_bias, kLog2e);
    sij = AddFOp::create(fn_builder, sij, user_bias);
  }

  sij = ApplyStructuredMask(fn_builder, sij, row_block_idx, col_block_idx, qt,
                            kt, config.q_sequence_length,
                            config.kv_sequence_length, config.is_causal);

  if (user_bias) {
    // See comment in forward pass for details.
    sij = ClampLogits(fn_builder, sij);
  }

  Value lse_broadcast =
      BroadcastOp::create(fn_builder, vector_type_qt_kt, lse_tile);

  Value sub_op = SubFOp::create(fn_builder, sij, lse_broadcast);
  Value pij = Exp2Op::create(fn_builder, sub_op);

  // Compute dP = dO @ V^T!
  Value dpij = CreateMatmul(fn_builder, do_2d, v_2d,
                            MatmulOptions{.transpose_rhs = true});

  // Compute ds = P * (dP - di)!
  Value di_broadcast =
      BroadcastOp::create(fn_builder, vector_type_qt_kt, di_tile);

  Value dp_sub_di = SubFOp::create(fn_builder, dpij, di_broadcast);
  Value ds = MulFOp::create(fn_builder, pij, dp_sub_di);
  ds = ConvertElementType(fn_builder, ity, ds);

  return {pij, ds};
}

struct BwdArgIndices {
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
  Value lse_tile;
  Value di_tile;
  Value mask_tile;
  Value q_tile;
  Value k_tile;
  Value v_tile;
  Value do_2d;
};

LoadedTiles LoadAndCastTiles(ImplicitLocOpBuilder& fn_builder, func::FuncOp fn,
                             const FlashAttnConfig& config,
                             const Tiling& tiling, Type oty,
                             const BwdArgIndices& arg_indices) {
  int64_t qt = tiling.qt;

  LoadedTiles tiles;
  tiles.q_tile = LoadTile(fn_builder, fn.getArgument(arg_indices.q));
  tiles.k_tile = LoadTile(fn_builder, fn.getArgument(arg_indices.k));
  tiles.v_tile = LoadTile(fn_builder, fn.getArgument(arg_indices.v));
  tiles.do_2d = LoadTile(fn_builder, fn.getArgument(arg_indices.do_arg));
  auto lse_loaded = LoadTile(fn_builder, fn.getArgument(arg_indices.lse));
  tiles.lse_tile = ShapeCastOp::create(
      fn_builder, VectorType::get({qt, 1}, oty), lse_loaded);

  auto di_loaded = LoadTile(fn_builder, fn.getArgument(arg_indices.di));
  tiles.di_tile =
      ShapeCastOp::create(fn_builder, VectorType::get({qt, 1}, oty), di_loaded);

  if (config.has_attn_bias) {
    tiles.mask_tile = LoadTile(fn_builder, fn.getArgument(arg_indices.mask));
  }

  return tiles;
}

func::FuncOp buildBackwardDkvModule(ImplicitLocOpBuilder& module_builder,
                                    const FlashAttnConfig& config,
                                    const Tiling& tiling) {
  MLIRContext* context = module_builder.getContext();
  Type ity = getElementType(*context, config.element_type);
  Type oty = module_builder.getF32Type();

  const int64_t qt = tiling.qt;
  const int64_t kt = tiling.kt;

  SmallVector<int64_t> q_dims = {1, 1, qt, config.qk_head_dim};
  SmallVector<int64_t> k_dims = {1, 1, kt, config.qk_head_dim};
  SmallVector<int64_t> v_dims = {1, 1, kt, config.vo_head_dim};
  SmallVector<int64_t> do_dims = {1, 1, qt, config.vo_head_dim};
  SmallVector<int64_t> lse_dims = {1, 1, 1, qt};
  SmallVector<int64_t> di_dims = {1, 1, 1, qt};

  // Inputs list!
  SmallVector<Type> inputs(5, module_builder.getI32Type());
  inputs.push_back(GetVmemMemRefType(context, q_dims, ity));
  inputs.push_back(GetVmemMemRefType(context, k_dims, ity));
  inputs.push_back(GetVmemMemRefType(context, v_dims, ity));
  inputs.push_back(GetVmemMemRefType(context, do_dims, ity));
  inputs.push_back(GetVmemMemRefType(context, lse_dims, oty));
  inputs.push_back(GetVmemMemRefType(context, di_dims, oty));

  if (config.has_attn_bias) {
    SmallVector<int64_t> mask_dims = {1, 1, qt, kt};
    inputs.push_back(GetVmemMemRefType(context, mask_dims, ity));
  }

  // Outputs! (dk, dv)
  SmallVector<int64_t> dk_dims = {1, 1, kt, config.qk_head_dim};
  SmallVector<int64_t> dv_dims = {1, 1, kt, config.vo_head_dim};

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

  func::FuncOp fn =
      func::FuncOp::create(module_builder, "flash_backward_dkv",
                           module_builder.getFunctionType(inputs, {}));
  SetTcCoreTypeAttr(fn);

  ImplicitLocOpBuilder fn_builder = ImplicitLocOpBuilder::atBlockBegin(
      module_builder.getLoc(), fn.addEntryBlock());

  ConstantOp zero = ConstantOp::create(
      fn_builder, fn_builder.getZeroAttr(fn_builder.getI32Type()));

  Value is_first_q_tile =
      CmpIOp::create(fn_builder, CmpIPredicate::eq, zero, fn.getArgument(3));
  Value is_first_head_group =
      CmpIOp::create(fn_builder, CmpIPredicate::eq, zero, fn.getArgument(4));
  Value is_first_reduction_step =
      AndIOp::create(fn_builder, is_first_q_tile, is_first_head_group);

  IfOp::create(fn_builder, is_first_reduction_step,
               /*thenBuilder=*/
               [&](OpBuilder& builder, Location loc) -> void {
                 ImplicitLocOpBuilder b(loc, builder);
                 ZeroTile(b, fn.getArgument(dk_f32_idx));
                 ZeroTile(b, fn.getArgument(dv_f32_idx));
                 YieldOp::create(b, loc);
               });

  Value kv_tile_idx = fn.getArgument(2);
  Value q_tile_idx = fn.getArgument(3);

  std::optional<IfOp> causal_if;
  if (config.is_causal) {
    causal_if = CreateCausalIfOp(fn_builder, q_tile_idx, kv_tile_idx, qt, kt);
    fn_builder.setInsertionPointToStart(causal_if->thenBlock());
  }

  LoadedTiles tiles =
      LoadAndCastTiles(fn_builder, fn, config, tiling, oty, arg_indices);

  auto shared_result = ComputeSharedBackwardLogic(
      fn_builder, config, qt, kt, tiles.q_tile, tiles.k_tile, tiles.v_tile,
      tiles.do_2d, tiles.lse_tile, tiles.di_tile, tiles.mask_tile, q_tile_idx,
      kv_tile_idx, oty, ity, module_builder.getI32Type(), context);

  Value p_contract = ConvertElementType(fn_builder, ity, shared_result.p);

  auto dv_contraction = CreateMatmul(fn_builder, p_contract, tiles.do_2d,
                                     MatmulOptions{.transpose_lhs = true});

  auto dk_contraction =
      CreateMatmul(fn_builder, shared_result.ds_trunc, tiles.q_tile,
                   MatmulOptions{.transpose_lhs = true});

  Value scaled_dk = ScaleValue(fn_builder, dk_contraction, config.scale);

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
  func::ReturnOp::create(fn_builder);
  return fn;
}
void SetBackwardDkvKernelAttributes(const FlashAttnConfig& config,
                                    const Tiling& tiling, func::FuncOp fn,
                                    OpBuilder& builder) {
  MLIRContext* context = builder.getContext();
  int64_t num_kv_tiles = config.padded_kv_sequence_length / tiling.kt;
  int64_t num_q_tiles = config.padded_q_sequence_length / tiling.qt;
  int64_t head_group_size = config.num_heads / config.kv_num_heads;

  SmallVector<int64_t> iteration_bounds = {config.batch_size,
                                           config.kv_num_heads, num_kv_tiles,
                                           num_q_tiles, head_group_size};
  fn->setAttr("iteration_bounds",
              builder.getDenseI64ArrayAttr(iteration_bounds));

  SmallVector<Attribute> dimension_semantics(
      3, GetDimensionSemanticsAttr(context, DimensionSemantics::kParallel));
  dimension_semantics.push_back(
      GetDimensionSemanticsAttr(context, DimensionSemantics::kArbitrary));
  dimension_semantics.push_back(
      GetDimensionSemanticsAttr(context, DimensionSemantics::kArbitrary));
  fn->setAttr("dimension_semantics", builder.getArrayAttr(dimension_semantics));

  auto transform_indices = [&](AffineMap map) {
    return builder.getDictionaryAttr(
        {builder.getNamedAttr("transform_indices", AffineMapAttr::get(map))});
  };

  // Iteration indices: (d0, d1, d2, d3, d4) = (batch, kv_head, kv_tile, q_tile,
  // head_group)
  mlir::AffineExpr batch_itr = builder.getAffineDimExpr(0);
  mlir::AffineExpr kv_head_itr = builder.getAffineDimExpr(1);
  mlir::AffineExpr kv_seq_itr = builder.getAffineDimExpr(2);
  mlir::AffineExpr constant_0 = builder.getAffineConstantExpr(0);

  // K, V, dK, dV always index: (batch, kv_head, kv_tile, 0)
  auto kv_map = AffineMap::get(
      5, 0, {batch_itr, kv_head_itr, kv_seq_itr, constant_0}, context);
  DictionaryAttr k_map = transform_indices(kv_map);
  DictionaryAttr v_map = k_map;
  DictionaryAttr dk_map = k_map;
  DictionaryAttr dv_map = k_map;

  // GQA / MQA: Mosaic AffineMap doesn't support Dim * Const + Dim, so use
  // transform functions.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(fn);
  Location loc = fn.getLoc();
  Type i32 = builder.getI32Type();
  SmallVector<Type> input_types(5, i32);
  SmallVector<Type> output_types(4, i32);
  FunctionType fn_type = builder.getFunctionType(input_types, output_types);

  auto create_transform_fn = [&](llvm::StringRef name,
                                 absl::AnyInvocable<SmallVector<Value, 4>(
                                     OpBuilder&, Value batch, Value q_head,
                                     Value q_tile, Value kv_tile, Value c0)>
                                     get_results) {
    auto transform_fn = func::FuncOp::create(builder, loc, name, fn_type);
    transform_fn.setPrivate();
    Block* entry = transform_fn.addEntryBlock();
    OpBuilder fn_b = OpBuilder::atBlockBegin(entry);
    Value batch = entry->getArgument(0);
    Value kv_head = entry->getArgument(1);
    Value kv_tile = entry->getArgument(2);
    Value q_tile = entry->getArgument(3);
    Value head_in_group = entry->getArgument(4);

    Value group_size_val = arith::ConstantOp::create(
        fn_b, loc, fn_b.getI32IntegerAttr(head_group_size));
    Value kv_offset = arith::MulIOp::create(fn_b, loc, kv_head, group_size_val);
    Value q_head = arith::AddIOp::create(fn_b, loc, kv_offset, head_in_group);

    Value c0 = arith::ConstantOp::create(fn_b, loc, fn_b.getI32IntegerAttr(0));
    SmallVector<Value, 4> results =
        get_results(fn_b, batch, q_head, q_tile, kv_tile, c0);
    func::ReturnOp::create(fn_b, loc, results);
    return transform_fn;
  };

  create_transform_fn("q_transform_indices",
                      [](OpBuilder&, Value batch, Value q_head, Value q_tile,
                         Value kv_tile, Value c0) -> SmallVector<Value, 4> {
                        return {batch, q_head, q_tile, c0};
                      });
  DictionaryAttr q_map_attr = CreateSymbolTransformIndicesAttr(
      builder, "q_transform_indices", {1, 1, tiling.qt, config.qk_head_dim});
  DictionaryAttr do_map_attr = CreateSymbolTransformIndicesAttr(
      builder, "q_transform_indices", {1, 1, tiling.qt, config.vo_head_dim});

  create_transform_fn("aux_transform_indices",
                      [](OpBuilder&, Value batch, Value q_head, Value q_tile,
                         Value kv_tile, Value c0) -> SmallVector<Value, 4> {
                        return {batch, q_head, c0, q_tile};
                      });
  DictionaryAttr aux_map_attr = CreateSymbolTransformIndicesAttr(
      builder, "aux_transform_indices", {1, 1, 1, tiling.qt});

  DictionaryAttr lse_map_attr = aux_map_attr;
  DictionaryAttr di_map_attr = aux_map_attr;

  DictionaryAttr mask_map_attr;
  if (config.has_attn_bias) {
    create_transform_fn("mask_transform_indices",
                        [&](OpBuilder&, Value batch, Value q_head, Value q_tile,
                            Value kv_tile, Value c0) -> SmallVector<Value, 4> {
                          SmallVector<Value, 4> coords = {batch, q_head, q_tile,
                                                          kv_tile};
                          for (auto dim : config.mask_broadcast_dims) {
                            coords[dim] = c0;
                          }
                          return coords;
                        });
    mask_map_attr = CreateSymbolTransformIndicesAttr(
        builder, "mask_transform_indices", {1, 1, tiling.qt, tiling.kt});
  }

  SmallVector<Attribute> window_attrs;
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
}

void SetBackwardDqKernelAttributes(const FlashAttnConfig& config,
                                   const Tiling& tiling, func::FuncOp fn,
                                   OpBuilder& builder) {
  MLIRContext* context = builder.getContext();

  SmallVector<int64_t> iteration_bounds = {
      config.batch_size, config.num_heads,
      config.padded_q_sequence_length / tiling.qt,
      config.padded_kv_sequence_length / tiling.kt};

  int batch_idx = 0;
  int heads_idx = 1;
  int q_seq_idx = 2;
  int kv_seq_idx = 3;

  fn->setAttr("iteration_bounds",
              builder.getDenseI64ArrayAttr(iteration_bounds));

  SmallVector<Attribute> dimension_semantics(
      3, GetDimensionSemanticsAttr(context, DimensionSemantics::kParallel));
  dimension_semantics.push_back(
      GetDimensionSemanticsAttr(context, DimensionSemantics::kArbitrary));

  fn->setAttr("dimension_semantics", builder.getArrayAttr(dimension_semantics));

  auto transform_indices = [&](AffineMap map) {
    return builder.getDictionaryAttr(
        {builder.getNamedAttr("transform_indices", AffineMapAttr::get(map))});
  };

  mlir::AffineExpr batch_itr = builder.getAffineDimExpr(batch_idx);
  mlir::AffineExpr heads_itr = builder.getAffineDimExpr(heads_idx);
  mlir::AffineExpr q_seq_itr = builder.getAffineDimExpr(q_seq_idx);
  mlir::AffineExpr kv_seq_itr = builder.getAffineDimExpr(kv_seq_idx);
  mlir::AffineExpr constant_0 = builder.getAffineConstantExpr(0);

  llvm::SmallVector<AffineExpr> q_map_expr = {batch_itr, heads_itr, q_seq_itr,
                                              constant_0};
  llvm::SmallVector<AffineExpr> aux_map_expr = {batch_itr, heads_itr,
                                                constant_0, q_seq_itr};

  auto q_map = AffineMap::get(4, 0, q_map_expr, context);
  auto aux_map = AffineMap::get(4, 0, aux_map_expr, context);

  auto [k_map, v_map] = CreateKVWindowMaps(builder, fn, config, tiling);

  SmallVector<Attribute> window_attrs;
  window_attrs.push_back(transform_indices(q_map));    // Q
  window_attrs.push_back(k_map);                       // K
  window_attrs.push_back(v_map);                       // V
  window_attrs.push_back(transform_indices(q_map));    // dO
  window_attrs.push_back(transform_indices(aux_map));  // lse
  window_attrs.push_back(transform_indices(aux_map));  // di

  if (config.has_attn_bias) {
    llvm::SmallVector<AffineExpr, 4> mask_expr = {batch_itr, heads_itr,
                                                  q_seq_itr, kv_seq_itr};
    for (auto dim : config.mask_broadcast_dims) {
      mask_expr[dim] = builder.getAffineConstantExpr(0);
    }
    AffineMap mask_map = AffineMap::get(4, 0, mask_expr, context);
    window_attrs.push_back(transform_indices(mask_map));
  }

  window_attrs.push_back(transform_indices(q_map));  // Out
  window_attrs.push_back(transform_indices(q_map));  // dQ

  fn->setAttr("window_params", builder.getArrayAttr(window_attrs));
  fn->setAttr("scratch_operands", builder.getI64IntegerAttr(1));
}

absl::StatusOr<OwningOpRef<ModuleOp>> CreateBackwardDkvKernel(
    MLIRContext* context, const FlashAttnConfig& config, const Tiling& tiling) {
  OpBuilder builder(context);
  OwningOpRef<ModuleOp> module =
      ModuleOp::create(builder, builder.getUnknownLoc());
  ImplicitLocOpBuilder module_builder(module->getLoc(),
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

  auto fn = buildBackwardDkvModule(module_builder, config, tiling);
  SetBackwardDkvKernelAttributes(config, tiling, fn, builder);

  ABSL_VLOG(1) << "Backward dKV kernel:\n" << GetOpString(module.get());

  return module;
}

func::FuncOp buildBackwardDqModule(ImplicitLocOpBuilder& module_builder,
                                   const FlashAttnConfig& config,
                                   const Tiling& tiling) {
  auto context = module_builder.getContext();
  Type ity = getElementType(*context, config.element_type);
  auto oty = module_builder.getF32Type();

  int64_t qt = tiling.qt;
  int64_t kt = tiling.kt;

  SmallVector<int64_t> q_dims = {1, 1, qt, config.qk_head_dim};
  SmallVector<int64_t> k_dims = {1, 1, kt, config.qk_head_dim};
  SmallVector<int64_t> v_dims = {1, 1, kt, config.vo_head_dim};
  SmallVector<int64_t> do_dims = {1, 1, qt, config.vo_head_dim};
  SmallVector<int64_t> lse_dims = {1, 1, 1, qt};
  SmallVector<int64_t> di_dims = {1, 1, 1, qt};
  SmallVector<int64_t> out_dims = {1, 1, qt, config.vo_head_dim};

  SmallVector<int64_t> dq_dims = {1, 1, qt, config.qk_head_dim};

  // Inputs list!
  SmallVector<Type> inputs(4, module_builder.getI32Type());
  inputs.push_back(GetVmemMemRefType(context, q_dims, ity));
  inputs.push_back(GetVmemMemRefType(context, k_dims, ity));
  inputs.push_back(GetVmemMemRefType(context, v_dims, ity));
  inputs.push_back(GetVmemMemRefType(context, do_dims, ity));
  inputs.push_back(GetVmemMemRefType(context, lse_dims, oty));
  inputs.push_back(GetVmemMemRefType(context, di_dims, oty));

  if (config.has_attn_bias) {
    SmallVector<int64_t> mask_dims = {1, 1, qt, kt};
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

  auto fn = func::FuncOp::create(module_builder, module_builder.getUnknownLoc(),
                                 "flash_backward_dq",
                                 module_builder.getFunctionType(inputs, {}));
  SetTcCoreTypeAttr(fn);

  auto fn_builder = ImplicitLocOpBuilder::atBlockBegin(module_builder.getLoc(),
                                                       fn.addEntryBlock());

  IfOp::create(fn_builder,
               CmpIOp::create(
                   fn_builder, CmpIPredicate::eq,
                   ConstantOp::create(fn_builder, fn_builder.getZeroAttr(
                                                      fn_builder.getI32Type())),
                   fn.getArgument(3)),
               /*thenBuilder=*/
               [&](OpBuilder& builder, Location loc) -> void {
                 ImplicitLocOpBuilder b(loc, builder);
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

  Value q_tile_idx = fn.getArgument(2);
  Value kv_tile_idx = fn.getArgument(3);

  auto shared_result = ComputeSharedBackwardLogic(
      fn_builder, config, qt, kt, tiles.q_tile, tiles.k_tile, tiles.v_tile,
      tiles.do_2d, tiles.lse_tile, tiles.di_tile, tiles.mask_tile, q_tile_idx,
      kv_tile_idx, oty, ity, module_builder.getI32Type(), context);

  // Compute dQ = dS @ K!
  auto dq_contraction =
      CreateMatmul(fn_builder, shared_result.ds_trunc, tiles.k_tile);

  Value scaled_dq = ScaleValue(fn_builder, dq_contraction, config.scale);

  Value prev_dq = LoadTile(fn_builder, fn.getArgument(dq_f32_idx));
  Value updated_dq = AddFOp::create(fn_builder, prev_dq, scaled_dq);
  StoreTile(fn_builder, updated_dq, fn.getArgument(dq_f32_idx));

  StoreTile(fn_builder, updated_dq, fn.getArgument(dq_bf16_idx));

  if (causal_if.has_value()) {
    fn_builder.setInsertionPointAfter(*causal_if);
  }
  func::ReturnOp::create(fn_builder);
  return fn;
}

absl::StatusOr<OwningOpRef<ModuleOp>> CreateBackwardDqKernel(
    MLIRContext* context, const FlashAttnConfig& config, const Tiling& tiling) {
  OpBuilder builder(context);
  OwningOpRef<ModuleOp> module =
      ModuleOp::create(builder, builder.getUnknownLoc());
  ImplicitLocOpBuilder module_builder(module->getLoc(),
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

  auto fn = buildBackwardDqModule(module_builder, config, tiling);
  SetBackwardDqKernelAttributes(config, tiling, fn, builder);

  ABSL_VLOG(1) << "Backward dQ kernel:\n" << GetOpString(module.get());

  return module;
}

}  // namespace mlir::torch_tpu
