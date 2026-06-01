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

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
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
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/_internal/mosaic/op_builders.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/scaled_dot_product_attention/flash_attention_kernel.h"
#include "xla/xla_data.pb.h"

namespace mlir::torch_tpu {

using mlir::arith::AddFOp;         // USING_DECL_OK
using mlir::arith::AddIOp;         // USING_DECL_OK
using mlir::arith::CmpIOp;         // USING_DECL_OK
using mlir::arith::CmpIPredicate;  // USING_DECL_OK
using mlir::arith::ConstantOp;     // USING_DECL_OK
using mlir::arith::ExtFOp;         // USING_DECL_OK
using mlir::arith::MaximumFOp;     // USING_DECL_OK
using mlir::arith::MulFOp;         // USING_DECL_OK
using mlir::arith::MulIOp;         // USING_DECL_OK
using mlir::arith::SelectOp;       // USING_DECL_OK
using mlir::arith::SubFOp;         // USING_DECL_OK
using mlir::arith::SubIOp;         // USING_DECL_OK
using mlir::arith::TruncFOp;       // USING_DECL_OK
using mlir::func::ReturnOp;        // USING_DECL_OK
using mlir::math::ExpOp;           // USING_DECL_OK
using mlir::scf::IfOp;             // USING_DECL_OK
using mlir::scf::YieldOp;          // USING_DECL_OK
using mlir::vector::BroadcastOp;   // USING_DECL_OK
using mlir::vector::IteratorType;  // USING_DECL_OK
using mlir::vector::LoadOp;        // USING_DECL_OK
using mlir::vector::ShapeCastOp;   // USING_DECL_OK
using mlir::vector::StoreOp;       // USING_DECL_OK
using mlir::vector::TransposeOp;   // USING_DECL_OK

Value CreateZeroVector(ImplicitLocOpBuilder& builder, Type type,
                       ArrayRef<int64_t> shape) {
  return BroadcastOp::create(
      builder, VectorType::get(shape, type),
      ConstantOp::create(builder, builder.getZeroAttr(type)));
}

Value ApplyCausalMask2D(ImplicitLocOpBuilder& fn_builder, Value sij,
                        Value row_offset, Value col_offset, int64_t qt,
                        int64_t kt, Type oty, IntegerType i32) {
  auto row_iota = CreateIotaOp(fn_builder, VectorType::get({qt, kt}, i32), 0);
  auto col_iota = CreateIotaOp(fn_builder, VectorType::get({qt, kt}, i32), 1);

  auto row_offset_vec = BroadcastOp::create(
      fn_builder, VectorType::get({qt, kt}, i32), row_offset);
  auto col_offset_vec = BroadcastOp::create(
      fn_builder, VectorType::get({qt, kt}, i32), col_offset);

  auto row_indices = AddIOp::create(fn_builder, row_iota, row_offset_vec);
  auto col_indices = AddIOp::create(fn_builder, col_iota, col_offset_vec);

  auto causal_mask =
      CmpIOp::create(fn_builder, CmpIPredicate::sge, row_indices, col_indices);

  auto mask_val = fn_builder.getFloatAttr(oty, -10000.0f);

  return SelectOp::create(
      fn_builder, causal_mask, sij,
      ConstantOp::create(
          fn_builder,
          DenseElementsAttr::get(VectorType::get({qt, kt}, oty), mask_val)));
}

struct SharedBwdResult {
  Value p;
  Value ds_trunc;
};

SharedBwdResult ComputeSharedBackwardLogic(
    ImplicitLocOpBuilder& fn_builder, const FlashAttnConfig& config, int64_t qt,
    int64_t kt, Value q_2d, Value k_2d, Value v_2d, Value do_2d, Value lse_tile,
    Value di_tile, Value mask_2d, Value row_offset, Value col_offset, Type oty,
    Type ity, IntegerType i32, MLIRContext* context,
    bool set_qk_precision = true) {
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  AffineExpr r, c, o;
  bindDims(context, r, c, o);

  auto unscaled_sij = CreateContractionOp(
      fn_builder, q_2d, k_2d, CreateZeroVector(fn_builder, oty, {qt, kt}),
      MapList{{r, o}, {c, o}, {r, c}},
      ArrayRef<IteratorType>{IteratorType::parallel, IteratorType::parallel,
                             IteratorType::reduction},
      set_qk_precision ? std::make_optional(config.contract_precision_qk_dot)
                       : std::nullopt);

  auto scale_attr = fn_builder.getFloatAttr(oty, config.scale);
  auto scale_op = ConstantOp::create(fn_builder, scale_attr);
  auto scale_broadcast =
      BroadcastOp::create(fn_builder, VectorType::get({qt, kt}, oty), scale_op);
  Value sij = MulFOp::create(fn_builder, unscaled_sij, scale_broadcast);

  if (mask_2d) {
    Value mask_f32 = mask_2d;
    if (ity != oty) {
      mask_f32 =
          ExtFOp::create(fn_builder, VectorType::get({qt, kt}, oty), mask_2d);
    }

    auto min_finite_attr = fn_builder.getFloatAttr(oty, -10000.0f);
    auto min_finite_op = ConstantOp::create(fn_builder, min_finite_attr);
    auto min_finite_broadcast = BroadcastOp::create(
        fn_builder, VectorType::get({qt, kt}, oty), min_finite_op);

    auto clamped_mask =
        MaximumFOp::create(fn_builder, mask_f32, min_finite_broadcast);

    sij = AddFOp::create(fn_builder, sij, clamped_mask);
  }

  if (config.is_causal) {
    sij = ApplyCausalMask2D(fn_builder, sij, row_offset, col_offset, qt, kt,
                            oty, i32);
  }

  auto lse_broadcast =
      BroadcastOp::create(fn_builder, VectorType::get({qt, kt}, oty), lse_tile);

  auto sub_op = SubFOp::create(fn_builder, sij, lse_broadcast);
  auto pij = math::ExpOp::create(fn_builder, sub_op);

  // Compute dP = dO @ V^T!
  auto dpij = CreateContractionOp(
      fn_builder, do_2d, v_2d, CreateZeroVector(fn_builder, oty, {qt, kt}),
      MapList{{r, o}, {c, o}, {r, c}},
      {IteratorType::parallel, IteratorType::parallel,
       IteratorType::reduction});

  // Compute ds = P * (dP - di)!
  auto di_broadcast =
      BroadcastOp::create(fn_builder, VectorType::get({qt, kt}, oty), di_tile);

  auto dp_sub_di = SubFOp::create(fn_builder, dpij, di_broadcast);
  auto ds = MulFOp::create(fn_builder, pij, dp_sub_di);

  Value ds_trunc = ds;
  if (ity != oty) {
    ds_trunc = TruncFOp::create(fn_builder, VectorType::get({qt, kt}, ity), ds);
  }

  return {pij, ds_trunc};
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

BwdArgIndices GetBwdArgIndices(const FlashAttnConfig& config) {
  BwdArgIndices indices;
  int arg_idx = 4;
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
                             const Tiling& tiling, ValueRange indices, Type ity,
                             Type oty, const BwdArgIndices& arg_indices) {
  int64_t qt = tiling.qt;
  int64_t kt = tiling.kt;

  SmallVector<int64_t> q_dims = {qt, config.qk_head_dim};
  SmallVector<int64_t> k_dims = {kt, config.qk_head_dim};
  SmallVector<int64_t> v_dims = {kt, config.vo_head_dim};
  SmallVector<int64_t> do_dims = {qt, config.vo_head_dim};
  LoadedTiles tiles;
  tiles.q_tile = LoadOp::create(fn_builder, VectorType::get(q_dims, ity),
                                fn.getArgument(arg_indices.q), indices);
  tiles.k_tile = LoadOp::create(fn_builder, VectorType::get(k_dims, ity),
                                fn.getArgument(arg_indices.k), indices);
  tiles.v_tile = LoadOp::create(fn_builder, VectorType::get(v_dims, ity),
                                fn.getArgument(arg_indices.v), indices);
  tiles.do_2d = LoadOp::create(fn_builder, VectorType::get(do_dims, ity),
                               fn.getArgument(arg_indices.do_arg), indices);
  auto lse_loaded = LoadOp::create(fn_builder, VectorType::get({1, qt}, oty),
                                   fn.getArgument(arg_indices.lse), indices);
  tiles.lse_tile = ShapeCastOp::create(
      fn_builder, VectorType::get({qt, 1}, oty), lse_loaded);

  auto di_loaded = LoadOp::create(fn_builder, VectorType::get({1, qt}, oty),
                                  fn.getArgument(arg_indices.di), indices);
  tiles.di_tile =
      ShapeCastOp::create(fn_builder, VectorType::get({qt, 1}, oty), di_loaded);

  if (config.has_attn_bias) {
    SmallVector<int64_t> mask_dims = {qt, kt};
    tiles.mask_tile =
        LoadOp::create(fn_builder, VectorType::get(mask_dims, ity),
                       fn.getArgument(arg_indices.mask), indices);
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

  // Outputs! (dk, dv)
  SmallVector<int64_t> dk_2d_dims = {kt, config.qk_head_dim};
  SmallVector<int64_t> dk_dims = {1, 1, kt, config.qk_head_dim};

  SmallVector<int64_t> dv_2d_dims = {kt, config.vo_head_dim};
  SmallVector<int64_t> dv_dims = {1, 1, kt, config.vo_head_dim};

  inputs.push_back(GetVmemMemRefType(context, dk_dims, ity));
  inputs.push_back(GetVmemMemRefType(context, dv_dims, ity));
  // Scratch outputs!
  inputs.push_back(GetVmemMemRefType(context, dk_dims, oty));
  inputs.push_back(GetVmemMemRefType(context, dv_dims, oty));

  BwdArgIndices arg_indices = GetBwdArgIndices(config);
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

  ConstantOp i0 = ConstantOp::create(fn_builder, fn_builder.getIndexAttr(0));
  ConstantOp zero = ConstantOp::create(
      fn_builder, fn_builder.getZeroAttr(fn_builder.getI32Type()));
  using Indices = SmallVector<Value>;
  IfOp initialize_output = IfOp::create(
      fn_builder,
      CmpIOp::create(fn_builder, CmpIPredicate::eq, zero, fn.getArgument(3)),
      /*thenBuilder=*/
      [&](OpBuilder& builder, Location loc) -> void {
        ImplicitLocOpBuilder b(loc, builder);
        StoreOp::create(b,
                        ConstantOp::create(
                            b, b.getZeroAttr(VectorType::get(dk_2d_dims, oty))),
                        fn.getArgument(dk_f32_idx),
                        Indices(dk_dims.size(), i0));
        StoreOp::create(b,
                        ConstantOp::create(
                            b, b.getZeroAttr(VectorType::get(dv_2d_dims, oty))),
                        fn.getArgument(dv_f32_idx),
                        Indices(dv_dims.size(), i0));
        YieldOp::create(b, loc);
      });
  fn_builder.setInsertionPointAfter(initialize_output);

  std::optional<IfOp> causal_if;
  if (config.is_causal) {
    ConstantOp qt_const =
        ConstantOp::create(fn_builder, fn_builder.getI32IntegerAttr(qt));
    ConstantOp kt_const =
        ConstantOp::create(fn_builder, fn_builder.getI32IntegerAttr(kt));
    ConstantOp c1 =
        ConstantOp::create(fn_builder, fn_builder.getI32IntegerAttr(1));

    Value q_tile_idx = fn.getArgument(2);
    Value kv_tile_idx = fn.getArgument(3);

    Value kv_idx_plus_1 = AddIOp::create(fn_builder, kv_tile_idx, c1);
    Value lhs = MulIOp::create(fn_builder, kv_idx_plus_1, qt_const);
    Value lhs_minus_1 = SubIOp::create(fn_builder, lhs, c1);

    Value rhs = MulIOp::create(fn_builder, q_tile_idx, kt_const);

    Value cond =
        CmpIOp::create(fn_builder, CmpIPredicate::sge, lhs_minus_1, rhs);

    causal_if = IfOp::create(fn_builder, cond, /*withElse=*/false);
    fn_builder.setInsertionPointToStart(causal_if->thenBlock());
  }

  SmallVector<Value> indices(4, i0);

  LoadedTiles tiles = LoadAndCastTiles(fn_builder, fn, config, tiling, indices,
                                       ity, oty, arg_indices);

  Value q_tile_idx = fn.getArgument(2);
  Value kv_tile_idx = fn.getArgument(3);

  ConstantOp qt_const =
      ConstantOp::create(fn_builder, fn_builder.getI32IntegerAttr(qt));
  ConstantOp kt_const =
      ConstantOp::create(fn_builder, fn_builder.getI32IntegerAttr(kt));

  Value row_offset = MulIOp::create(fn_builder, kv_tile_idx, qt_const);
  Value col_offset = MulIOp::create(fn_builder, q_tile_idx, kt_const);

  auto shared_result = ComputeSharedBackwardLogic(
      fn_builder, config, qt, kt, tiles.q_tile, tiles.k_tile, tiles.v_tile,
      tiles.do_2d, tiles.lse_tile, tiles.di_tile, tiles.mask_tile, row_offset,
      col_offset, oty, ity, module_builder.getI32Type(), context);

  Value p_contract = shared_result.p;
  if (ity != oty) {
    p_contract = TruncFOp::create(fn_builder, VectorType::get({qt, kt}, ity),
                                  shared_result.p);
  }

  auto p_transposed = TransposeOp::create(fn_builder, p_contract,
                                          llvm::ArrayRef<int64_t>{1, 0});

  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  AffineExpr r, c, o;
  bindDims(context, r, c, o);

  auto dv_contraction = CreateContractionOp(
      fn_builder, p_transposed, tiles.do_2d,
      CreateZeroVector(fn_builder, oty, {kt, config.vo_head_dim}),
      MapList{{r, o}, {o, c}, {r, c}},
      ArrayRef<IteratorType>{IteratorType::parallel, IteratorType::parallel,
                             IteratorType::reduction});

  auto ds_transposed = TransposeOp::create(fn_builder, shared_result.ds_trunc,
                                           llvm::ArrayRef<int64_t>{1, 0});

  auto dk_contraction = CreateContractionOp(
      fn_builder, ds_transposed, tiles.q_tile,
      CreateZeroVector(fn_builder, oty, {kt, config.qk_head_dim}),
      MapList{{r, o}, {o, c}, {r, c}},
      ArrayRef<IteratorType>{IteratorType::parallel, IteratorType::parallel,
                             IteratorType::reduction});

  auto scale_attr = fn_builder.getFloatAttr(oty, config.scale);
  auto scale_op = ConstantOp::create(fn_builder, scale_attr);
  auto scale_broadcast_dk = BroadcastOp::create(
      fn_builder, VectorType::get({kt, config.qk_head_dim}, oty), scale_op);
  auto scaled_dk =
      MulFOp::create(fn_builder, dk_contraction, scale_broadcast_dk);

  auto prev_dk = LoadOp::create(fn_builder, VectorType::get(dk_2d_dims, oty),
                                fn.getArgument(dk_f32_idx), indices);
  auto updated_dk = AddFOp::create(fn_builder, prev_dk, scaled_dk);
  StoreOp::create(fn_builder, updated_dk, fn.getArgument(dk_f32_idx), indices);

  auto prev_dv = LoadOp::create(fn_builder, VectorType::get(dv_2d_dims, oty),
                                fn.getArgument(dv_f32_idx), indices);
  auto updated_dv = AddFOp::create(fn_builder, prev_dv, dv_contraction);
  StoreOp::create(fn_builder, updated_dv, fn.getArgument(dv_f32_idx), indices);

  Value dk_out = updated_dk;
  Value dv_out = updated_dv;
  if (ity != oty) {
    dk_out = TruncFOp::create(fn_builder, VectorType::get(dk_2d_dims, ity),
                              updated_dk);
    dv_out = TruncFOp::create(fn_builder, VectorType::get(dv_2d_dims, ity),
                              updated_dv);
  }

  // TODO: need to accumulate this for the MQA case as multiple "head"
  // iterations will contribute to the same kv grad.
  // This is probably possible by some clever indexing and ensuring the scratch
  // is zeroed on the correct index.
  StoreOp::create(fn_builder, dk_out, fn.getArgument(dk_bf16_idx), indices);
  StoreOp::create(fn_builder, dv_out, fn.getArgument(dv_bf16_idx), indices);

  if (causal_if.has_value()) {
    fn_builder.setInsertionPointAfter(*causal_if);
  }
  func::ReturnOp::create(fn_builder);
  return fn;
}
void SetBackwardKernelAttributes(const FlashAttnConfig& config,
                                 const Tiling& tiling, func::FuncOp fn,
                                 OpBuilder& builder, bool is_dq = false) {
  MLIRContext* context = builder.getContext();

  SmallVector<int64_t> iteration_bounds = {
      config.batch_size, config.num_heads, config.q_sequence_length / tiling.qt,
      config.kv_sequence_length / tiling.kt};

  int batch_idx = 0;
  int heads_idx = 1;
  int q_seq_idx = 2;
  int kv_seq_idx = 3;

  if (!is_dq) {
    std::swap(iteration_bounds[q_seq_idx], iteration_bounds[kv_seq_idx]);
    std::swap(q_seq_idx, kv_seq_idx);
  }

  bool is_mqa = config.kv_num_heads < config.num_heads;

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
  llvm::SmallVector<AffineExpr> kv_map_expr = {batch_itr, heads_itr, kv_seq_itr,
                                               constant_0};
  llvm::SmallVector<AffineExpr> aux_map_expr = {batch_itr, heads_itr,
                                                constant_0, q_seq_itr};

  if (is_mqa) {
    kv_map_expr[1] = kv_map_expr[1].floorDiv(
        builder.getAffineConstantExpr(config.num_heads / config.kv_num_heads));
  }

  auto q_map = AffineMap::get(4, 0, q_map_expr, context);
  auto kv_map = AffineMap::get(4, 0, kv_map_expr, context);
  auto aux_map = AffineMap::get(4, 0, aux_map_expr, context);

  SmallVector<Attribute> window_attrs;
  window_attrs.push_back(transform_indices(q_map));    // Q
  window_attrs.push_back(transform_indices(kv_map));   // K
  window_attrs.push_back(transform_indices(kv_map));   // V
  window_attrs.push_back(transform_indices(q_map));    // dO
  window_attrs.push_back(transform_indices(aux_map));  // lse
  window_attrs.push_back(transform_indices(aux_map));  // di

  if (config.has_attn_bias) {
    llvm::SmallVector<AffineExpr, 4> mask_expr = {batch_itr, heads_itr,
                                                  q_seq_itr, kv_seq_itr};
    for (auto dim : config.mask_broadcast_dims) {
      mask_expr[dim] = builder.getAffineDimExpr(0);
    }
    AffineMap mask_map = AffineMap::get(4, 0, mask_expr, context);
    window_attrs.push_back(transform_indices(mask_map));
  }

  if (is_dq) {
    window_attrs.push_back(transform_indices(q_map));  // Out
    window_attrs.push_back(transform_indices(q_map));  // dQ
  } else {
    window_attrs.push_back(transform_indices(kv_map));  // dK
    window_attrs.push_back(transform_indices(kv_map));  // dV
  }

  fn->setAttr("window_params", builder.getArrayAttr(window_attrs));

  if (is_dq) {
    fn->setAttr("scratch_operands", builder.getI64IntegerAttr(1));
  } else {
    fn->setAttr("scratch_operands", builder.getI64IntegerAttr(2));
  }
}

absl::StatusOr<std::string> CreateBackwardDkvKernel(
    const FlashAttnConfig& config) {
  auto context = CreateMlirContextWithDialects();
  OpBuilder builder(context.get());
  OwningOpRef<ModuleOp> module =
      ModuleOp::create(builder, builder.getUnknownLoc());
  ImplicitLocOpBuilder module_builder(module->getLoc(),
                                      module->getBodyRegion());

  Tiling tiling = {
      .qt = std::min(kDefaultQTileSize, config.q_sequence_length),
      .kt = std::min(kDefaultKTileSize, config.kv_sequence_length),
  };
  if (config.q_sequence_length % tiling.qt != 0 ||
      config.kv_sequence_length % tiling.kt != 0) {
    return TT_ERROR(::torch_tpu::error::kInvalidArgument)
           << "padding not supported";
  }

  auto fn = buildBackwardDkvModule(module_builder, config, tiling);
  SetBackwardKernelAttributes(config, tiling, fn, builder);

  if (failed(SerializeMoasicKernel(module.get()))) {
    return TT_ERROR(::torch_tpu::error::kInternal)
           << "failed to serialize mosaic kernel";
  }

  std::string output;
  llvm::raw_string_ostream os(output);
  module->print(os, OpPrintingFlags().useLocalScope());
  os.flush();
  ABSL_VLOG(1) << "Backward dKV kernel:\n" << output;
  return output;
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

  SmallVector<int64_t> dq_2d_dims = {qt, config.qk_head_dim};
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

  auto zero = fn_builder.create<ConstantOp>(
      fn_builder.getIntegerAttr(fn_builder.getIndexType(), 0));
  SmallVector<Value> indices(4, zero);

  IfOp::create(fn_builder,
               CmpIOp::create(
                   fn_builder, CmpIPredicate::eq,
                   ConstantOp::create(fn_builder, fn_builder.getZeroAttr(
                                                      fn_builder.getI32Type())),
                   fn.getArgument(3)),
               /*thenBuilder=*/
               [&](OpBuilder& builder, Location loc) -> void {
                 StoreOp::create(
                     builder, loc,
                     ConstantOp::create(
                         builder, loc,
                         builder.getZeroAttr(VectorType::get(dq_2d_dims, oty))),
                     fn.getArgument(dq_f32_idx), indices);
                 YieldOp::create(builder, loc);
               });

  std::optional<IfOp> causal_if;
  if (config.is_causal) {
    ConstantOp qt_const =
        ConstantOp::create(fn_builder, fn_builder.getI32IntegerAttr(qt));
    ConstantOp kt_const =
        ConstantOp::create(fn_builder, fn_builder.getI32IntegerAttr(kt));
    ConstantOp c1 =
        ConstantOp::create(fn_builder, fn_builder.getI32IntegerAttr(1));

    Value row_block_idx = fn.getArgument(2);
    Value col_block_idx = fn.getArgument(3);

    Value row_idx_plus_1 = AddIOp::create(fn_builder, row_block_idx, c1);
    Value lhs = MulIOp::create(fn_builder, row_idx_plus_1, qt_const);
    Value lhs_minus_1 = SubIOp::create(fn_builder, lhs, c1);

    Value rhs = MulIOp::create(fn_builder, col_block_idx, kt_const);

    Value cond =
        CmpIOp::create(fn_builder, CmpIPredicate::sge, lhs_minus_1, rhs);

    causal_if = IfOp::create(fn_builder, cond, /*withElse=*/false);
    fn_builder.setInsertionPointToStart(causal_if->thenBlock());
  }

  LoadedTiles tiles = LoadAndCastTiles(fn_builder, fn, config, tiling, indices,
                                       ity, oty, arg_indices);

  Value q_tile_idx = fn.getArgument(2);
  Value kv_tile_idx = fn.getArgument(3);

  ConstantOp qt_const =
      ConstantOp::create(fn_builder, fn_builder.getI32IntegerAttr(qt));
  ConstantOp kt_const =
      ConstantOp::create(fn_builder, fn_builder.getI32IntegerAttr(kt));

  Value row_offset = MulIOp::create(fn_builder, q_tile_idx, qt_const);
  Value col_offset = MulIOp::create(fn_builder, kv_tile_idx, kt_const);

  auto shared_result = ComputeSharedBackwardLogic(
      fn_builder, config, qt, kt, tiles.q_tile, tiles.k_tile, tiles.v_tile,
      tiles.do_2d, tiles.lse_tile, tiles.di_tile, tiles.mask_tile, row_offset,
      col_offset, oty, ity, module_builder.getI32Type(), context,
      /*set_qk_precision=*/false);

  // Compute dQ = dS @ K!
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  AffineExpr r, c, o;
  bindDims(context, r, c, o);

  auto dq_contraction = CreateContractionOp(
      fn_builder, shared_result.ds_trunc, tiles.k_tile,
      CreateZeroVector(fn_builder, oty, {qt, config.qk_head_dim}),
      MapList{{r, o}, {o, c}, {r, c}},
      ArrayRef<IteratorType>{IteratorType::parallel, IteratorType::parallel,
                             IteratorType::reduction});

  auto scale_attr = fn_builder.getFloatAttr(oty, config.scale);
  auto scale_op = ConstantOp::create(fn_builder, scale_attr);
  auto scale_broadcast_dq = BroadcastOp::create(
      fn_builder, VectorType::get({qt, config.qk_head_dim}, oty), scale_op);
  auto scaled_dq =
      MulFOp::create(fn_builder, dq_contraction, scale_broadcast_dq);

  Value prev_dq = LoadOp::create(fn_builder, VectorType::get(dq_2d_dims, oty),
                                 fn.getArgument(dq_f32_idx), indices);
  Value updated_dq = AddFOp::create(fn_builder, prev_dq, scaled_dq);
  StoreOp::create(fn_builder, updated_dq, fn.getArgument(dq_f32_idx), indices);

  Value dq_out = updated_dq;
  if (ity != oty) {
    dq_out = TruncFOp::create(fn_builder, VectorType::get(dq_2d_dims, ity),
                              updated_dq);
  }

  StoreOp::create(fn_builder, dq_out, fn.getArgument(dq_bf16_idx), indices);

  if (causal_if.has_value()) {
    fn_builder.setInsertionPointAfter(*causal_if);
  }
  func::ReturnOp::create(fn_builder);
  return fn;
}

absl::StatusOr<std::string> CreateBackwardDqKernel(
    const FlashAttnConfig& config) {
  auto context = CreateMlirContextWithDialects();
  OpBuilder builder(context.get());
  OwningOpRef<ModuleOp> module =
      ModuleOp::create(builder, builder.getUnknownLoc());
  ImplicitLocOpBuilder module_builder(module->getLoc(),
                                      module->getBodyRegion());

  Tiling tiling = {
      .qt = std::min(kDefaultQTileSize, config.q_sequence_length),
      .kt = std::min(kDefaultKTileSize, config.kv_sequence_length),
  };
  if (config.q_sequence_length % tiling.qt != 0 ||
      config.kv_sequence_length % tiling.kt != 0) {
    return TT_ERROR(::torch_tpu::error::kInvalidArgument)
           << "padding not supported";
  }

  auto fn = buildBackwardDqModule(module_builder, config, tiling);
  SetBackwardKernelAttributes(config, tiling, fn, builder, /*is_dq=*/true);

  if (failed(SerializeMoasicKernel(module.get()))) {
    return TT_ERROR(::torch_tpu::error::kInternal)
           << "failed to serialize mosaic kernel";
  }

  std::string output;
  llvm::raw_string_ostream os(output);
  module->print(os, OpPrintingFlags().useLocalScope());
  os.flush();
  ABSL_VLOG(1) << "Backward dQ kernel:\n" << output;
  return output;
}

}  // namespace mlir::torch_tpu
