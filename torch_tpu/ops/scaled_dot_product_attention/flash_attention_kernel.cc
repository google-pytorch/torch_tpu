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

#include "torch_tpu/ops/scaled_dot_product_attention/flash_attention_kernel.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
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
#include "xla/xla_data.pb.h"

namespace mlir::torch_tpu {

using mlir::arith::AddFOp;                // USING_DECL_OK
using mlir::arith::AddIOp;                // USING_DECL_OK
using mlir::arith::CmpIOp;                // USING_DECL_OK
using mlir::arith::CmpIPredicate;         // USING_DECL_OK
using mlir::arith::ConstantOp;            // USING_DECL_OK
using mlir::arith::DivFOp;                // USING_DECL_OK
using mlir::arith::ExtFOp;                // USING_DECL_OK
using mlir::arith::MaximumFOp;            // USING_DECL_OK
using mlir::arith::MulFOp;                // USING_DECL_OK
using mlir::arith::MulIOp;                // USING_DECL_OK
using mlir::arith::SelectOp;              // USING_DECL_OK
using mlir::arith::SubFOp;                // USING_DECL_OK
using mlir::arith::SubIOp;                // USING_DECL_OK
using mlir::arith::TruncFOp;              // USING_DECL_OK
using mlir::func::ReturnOp;               // USING_DECL_OK
using mlir::math::ExpOp;                  // USING_DECL_OK
using mlir::scf::IfOp;                    // USING_DECL_OK
using mlir::scf::YieldOp;                 // USING_DECL_OK
using mlir::vector::BroadcastOp;          // USING_DECL_OK
using mlir::vector::CombiningKind;        // USING_DECL_OK
using mlir::vector::IteratorType;         // USING_DECL_OK
using mlir::vector::LoadOp;               // USING_DECL_OK
using mlir::vector::MultiDimReductionOp;  // USING_DECL_OK
using mlir::vector::ShapeCastOp;          // USING_DECL_OK
using mlir::vector::StoreOp;              // USING_DECL_OK

namespace {

void SetKernelAttributes(absl::Span<const int64_t> qk_dimensions,
                         const FlashAttnConfig& config, const Tiling& tiling,
                         func::FuncOp fn, OpBuilder& builder) {
  const int64_t scratch_operands = 3L;
  fn->setAttr("scratch_operands", builder.getI64IntegerAttr(scratch_operands));

  const int64_t num_iteration_indices = absl::c_count_if(
      fn.getArguments(),
      [](BlockArgument arg) { return arg.getType().isIntOrIndex(); });
  if (num_iteration_indices == 0) {
    // No other attributes are needed in the case there is a single iteration.
    return;
  }
  auto reference_dimensions = qk_dimensions;
  absl::InlinedVector<int64_t, 4> iteration_bounds = {
      reference_dimensions.begin(), reference_dimensions.end()};
  iteration_bounds[iteration_bounds.size() - 1] =
      config.kv_sequence_length / tiling.kt;
  iteration_bounds[iteration_bounds.size() - 2] =
      config.q_sequence_length / tiling.qt;
  fn->setAttr("iteration_bounds",
              builder.getDenseI64ArrayAttr(
                  {iteration_bounds.begin(), iteration_bounds.end()}));
  MLIRContext* context = builder.getContext();
  absl::InlinedVector<Attribute, 4> dimension_semantics(
      num_iteration_indices,
      GetDimensionSemanticsAttr(context, DimensionSemantics::kParallel));
  // The last dimension is arbitrary, update it accordingly.
  dimension_semantics.back() =
      GetDimensionSemanticsAttr(context, DimensionSemantics::kArbitrary);
  fn->setAttr("dimension_semantics",
              builder.getArrayAttr(
                  {dimension_semantics.begin(), dimension_semantics.end()}));

  // We need transform_indices attribute for q, k, v, any auxiliary arguments,
  // and output. This boils down to all arguments except iteration indices and
  // scratch arguments.
  // The arguments are in the following order: iteration_indices, q, k, v,
  // auxiliary_parameters, output, li, mi. The iteration_indices and
  // auxiliary_parameters are possibly empty.
  SmallVector<AffineExpr> dims(num_iteration_indices);
  bindDimsList(context, MutableArrayRef{dims});
  // Auxiliary function for creating transform_indices attribute.
  auto transform_indices = [&](ArrayRef<AffineExpr> results) {
    return builder.getDictionaryAttr({builder.getNamedAttr(
        "transform_indices", AffineMapAttr::get(AffineMap::get(
                                 /*dimCount=*/dims.size(),
                                 /*symbolCount=*/0, results, context)))});
  };

  llvm::SmallVector<AffineExpr> inner_map_expr = {
      builder.getAffineDimExpr(0), builder.getAffineDimExpr(1),
      builder.getAffineDimExpr(3), builder.getAffineConstantExpr(0)};
  llvm::SmallVector<AffineExpr> q_map_expr = {
      builder.getAffineDimExpr(0), builder.getAffineDimExpr(1),
      builder.getAffineDimExpr(2), builder.getAffineConstantExpr(0)};
  llvm::SmallVector<AffineExpr> aux_map_expr = {
      builder.getAffineDimExpr(0), builder.getAffineDimExpr(1),
      builder.getAffineConstantExpr(0), builder.getAffineDimExpr(2)};

  auto kv_map_expr = inner_map_expr;
  if (config.kv_num_heads < config.num_heads) {
    kv_map_expr[1] = kv_map_expr[1].floorDiv(
        builder.getAffineConstantExpr(config.num_heads / config.kv_num_heads));
  }

  auto q_map = transform_indices(q_map_expr);
  auto kv_map = transform_indices(kv_map_expr);
  auto aux_map = transform_indices(aux_map_expr);

  SmallVector<DictionaryAttr> window_attrs;
  window_attrs.push_back(q_map);   // Q
  window_attrs.push_back(kv_map);  // K
  window_attrs.push_back(kv_map);  // V

  if (config.has_attn_bias) {
    SmallVector<AffineExpr, 4> mask_expr{
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

Value ApplyCausalMask(ImplicitLocOpBuilder& fn_builder, Value sij,
                      Value row_block_idx, Value col_block_idx,
                      const FlashAttnConfig& config, int64_t qt, int64_t kt,
                      FloatType f32, IntegerType i32) {
  Value is_diagonal = CmpIOp::create(fn_builder, CmpIPredicate::eq,
                                     row_block_idx, col_block_idx);

  return IfOp::create(
             fn_builder, is_diagonal,
             /*thenBuilder=*/
             [&](OpBuilder& builder, Location loc) -> void {
               ImplicitLocOpBuilder b(loc, builder);

               Value row_iota =
                   CreateIotaOp(b, VectorType::get({qt, kt}, i32), 0);
               Value col_iota =
                   CreateIotaOp(b, VectorType::get({qt, kt}, i32), 1);

               auto qt_const = ConstantOp::create(b, b.getI32IntegerAttr(qt));
               auto kt_const = ConstantOp::create(b, b.getI32IntegerAttr(kt));

               Value row_offset = MulIOp::create(b, row_block_idx, qt_const);
               Value col_offset = MulIOp::create(b, col_block_idx, kt_const);

               Value row_offset_vec = BroadcastOp::create(
                   b, VectorType::get({qt, kt}, i32), row_offset);
               Value col_offset_vec = BroadcastOp::create(
                   b, VectorType::get({qt, kt}, i32), col_offset);

               Value row_indices = AddIOp::create(b, row_iota, row_offset_vec);
               Value col_indices = AddIOp::create(b, col_iota, col_offset_vec);

               Value causal_mask = CmpIOp::create(b, CmpIPredicate::sge,
                                                  row_indices, col_indices);

               auto mask_val = b.getFloatAttr(f32, -10000.0f);

               auto masked_sij = SelectOp::create(
                   b, causal_mask, sij,
                   ConstantOp::create(
                       b, DenseElementsAttr::get(VectorType::get({qt, kt}, f32),
                                                 mask_val)));
               YieldOp::create(b, masked_sij.getResult());
             },
             /*elseBuilder=*/
             [&](OpBuilder& builder, Location loc) -> void {
               YieldOp::create(builder, loc, sij);
             })
      .getResult(0);
}

// q [batch_size, num_heads, q_seq_len, d_model]
// k [batch_size, num_heads, kv_seq_len, d_model]
// v [batch_size, num_heads, kv_seq_len, d_model]
func::FuncOp buildModule(ImplicitLocOpBuilder& module_builder,
                         const FlashAttnConfig& config, const Tiling& tiling) {
  MLIRContext* context = module_builder.getContext();
  IntegerType i32 = module_builder.getIntegerType(32);
  // FlashAttnConfig::element_type is either F32 or BF16.
  Type ity = getElementType(*context, config.element_type);
  // The output element type is always F32.
  FloatType f32 = module_builder.getF32Type();
  const int64_t qt = tiling.qt;
  const int64_t kt = tiling.kt;
  // Ranks of q, k, v are either 3 or 4.
  SmallVector<int64_t> q_2d_dims = {qt, config.qk_head_dim};
  SmallVector<int64_t> k_2d_dims = {kt, config.qk_head_dim};
  SmallVector<int64_t> v_2d_dims = {kt, config.vo_head_dim};
  SmallVector<int64_t> oi_2d_dims = {qt, config.vo_head_dim};

  SmallVector<int64_t> q_dims = {1, 1, qt, config.qk_head_dim};
  SmallVector<int64_t> k_dims = {1, 1, kt, config.qk_head_dim};
  SmallVector<int64_t> v_dims = {1, 1, kt, config.vo_head_dim};
  SmallVector<int64_t> oi_dims = {1, 1, qt, config.vo_head_dim};
  SmallVector<int64_t> lse_dims = {1, 1, 1, qt};
  SmallVector<int64_t> mi_dims = {1, qt};
  SmallVector<int64_t> li_dims = {1, qt};

  // We have 4 iteration indices.
  SmallVector<Type> inputs(4, i32);
  // Add q, k, v.
  inputs.insert(inputs.end(), {GetVmemMemRefType(context, q_dims, ity),
                               GetVmemMemRefType(context, k_dims, ity),
                               GetVmemMemRefType(context, v_dims, ity)});
  bool has_mask = config.has_attn_bias;
  if (has_mask) {
    SmallVector<int64_t> mask_dims = {1, 1, qt, kt};
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
  func::FuncOp fn =
      func::FuncOp::create(module_builder, "flash_forward",
                           module_builder.getFunctionType(inputs, {}));
  SetTcCoreTypeAttr(fn);

  ImplicitLocOpBuilder fn_builder = ImplicitLocOpBuilder::atBlockBegin(
      module_builder.getLoc(), fn.addEntryBlock());

  // Indexes of the arguments.
  ArrayRef<BlockArgument> index_args = fn.getArguments().take_front(4);

  unsigned int next_index = 4;
  unsigned int q_index = next_index++;
  unsigned int k_index = next_index++;
  unsigned int v_index = next_index++;
  unsigned int mask_index = has_mask ? next_index++ : 0;
  unsigned int oi_index = next_index++;
  unsigned int lse_index = config.return_lse ? next_index++ : 0;
  unsigned int o_scratch_index = next_index++;
  unsigned int mi_index = next_index++;
  unsigned int li_index = next_index++;

  using Indices = SmallVector<Value>;
  ConstantOp i0 = ConstantOp::create(fn_builder, fn_builder.getIndexAttr(0));
  Indices zero_indices(4, i0);
  ConstantOp zero = ConstantOp::create(fn_builder, fn_builder.getZeroAttr(i32));
  std::optional<scf::IfOp> causal_if;
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
  IfOp initialize_output = IfOp::create(
      fn_builder,
      CmpIOp::create(fn_builder, CmpIPredicate::eq, zero, fn.getArgument(3)),
      /*thenBuilder=*/
      [&](OpBuilder& builder, Location loc) -> void {
        ImplicitLocOpBuilder b(loc, builder);
        // Initialize output (oi).
        StoreOp::create(b,
                        ConstantOp::create(
                            b, b.getZeroAttr(VectorType::get(oi_2d_dims, f32))),
                        fn.getArgument(o_scratch_index), zero_indices);
        // Initialize max reduce (mi).
        auto ninf = fn_builder.getFloatAttr(
            f32, APFloat::getInf(f32.getFloatSemantics(),
                                 /*Negative=*/true));
        StoreOp::create(
            b,
            ConstantOp::create(
                b, DenseElementsAttr::get(VectorType::get({1, qt}, f32), ninf)),
            fn.getArgument(mi_index), Indices(2, i0));
        // Initialize sum reduce (li).
        StoreOp::create(
            b,
            ConstantOp::create(b, b.getZeroAttr(VectorType::get({1, qt}, f32))),
            fn.getArgument(li_index), Indices(2, i0));
        YieldOp::create(b, loc);
      });
  fn_builder.setInsertionPointAfter(initialize_output);

  auto q = LoadOp::create(fn_builder, VectorType::get(q_2d_dims, ity),
                          fn.getArgument(q_index), Indices(q_dims.size(), i0));
  auto k = LoadOp::create(fn_builder, VectorType::get(k_2d_dims, ity),
                          fn.getArgument(k_index), Indices(k_dims.size(), i0));
  AffineExpr r, c, o;
  bindDims(context, r, c, o);
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  Value sij = CreateContractionOp(
      fn_builder, q, k,
      ConstantOp::create(
          fn_builder, fn_builder.getZeroAttr(VectorType::get({qt, kt}, f32))),
      MapList{{r, o}, {c, o}, {r, c}},
      ArrayRef<IteratorType>{IteratorType::parallel, IteratorType::parallel,
                             IteratorType::reduction},
      config.contract_precision_qk_dot);

  sij = MulFOp::create(
      fn_builder, sij,
      ConstantOp::create(
          fn_builder,
          DenseElementsAttr::get(VectorType::get({qt, kt}, f32),
                                 fn_builder.getFloatAttr(f32, config.scale))));

  if (config.is_causal) {
    Value row_block_idx = fn.getArgument(2);
    Value col_block_idx = fn.getArgument(3);
    sij = ApplyCausalMask(fn_builder, sij, row_block_idx, col_block_idx, config,
                          qt, kt, f32, i32);
  }

  if (has_mask) {
    Value bias = LoadOp::create(fn_builder, VectorType::get({qt, kt}, ity),
                                fn.getArgument(mask_index), Indices(4, i0));
    if (ity != f32) {
      bias = ExtFOp::create(fn_builder, VectorType::get({qt, kt}, f32), bias);
    }
    sij = AddFOp::create(fn_builder, sij, bias);
  }

  auto ninf =
      fn_builder.getFloatAttr(f32, APFloat::getInf(f32.getFloatSemantics(),
                                                   /*Negative=*/true));
  SmallVector<int64_t> reduction_dims = {qt};
  // The last dimension is the reduction dimension.
  SmallVector<bool> reduction_mask = {false, true};

  auto mij_1d = MultiDimReductionOp::create(
      fn_builder, sij,
      ConstantOp::create(
          fn_builder,
          DenseElementsAttr::get(VectorType::get(reduction_dims, f32), ninf)),
      reduction_mask, CombiningKind::MAXIMUMF);
  auto mij =
      ShapeCastOp::create(fn_builder, VectorType::get({qt, 1}, f32), mij_1d);

  auto pij = ExpOp::create(
      fn_builder,
      SubFOp::create(fn_builder, sij,
                     BroadcastOp::create(fn_builder,
                                         VectorType::get({qt, kt}, f32), mij)));

  auto lij_1d = MultiDimReductionOp::create(
      fn_builder, pij,
      ConstantOp::create(fn_builder, fn_builder.getZeroAttr(
                                         VectorType::get(reduction_dims, f32))),
      reduction_mask, CombiningKind::ADD);
  auto lij =
      ShapeCastOp::create(fn_builder, VectorType::get({qt, 1}, f32), lij_1d);

  auto mi_loaded = LoadOp::create(fn_builder, VectorType::get({1, qt}, f32),
                                  fn.getArgument(mi_index), Indices(2, i0));
  auto mi =
      ShapeCastOp::create(fn_builder, VectorType::get({qt, 1}, f32), mi_loaded);
  auto mi_new = MaximumFOp::create(fn_builder, mi, mij);

  auto li_loaded = LoadOp::create(fn_builder, VectorType::get({1, qt}, f32),
                                  fn.getArgument(li_index), Indices(2, i0));
  auto li =
      ShapeCastOp::create(fn_builder, VectorType::get({qt, 1}, f32), li_loaded);
  auto li_new = AddFOp::create(
      fn_builder,
      MulFOp::create(
          fn_builder,
          ExpOp::create(fn_builder, SubFOp::create(fn_builder, mi, mi_new)),
          li),
      MulFOp::create(
          fn_builder,
          ExpOp::create(fn_builder, SubFOp::create(fn_builder, mij, mi_new)),
          lij));

  Value lse = nullptr;
  if (config.return_lse) {
    lse = AddFOp::create(fn_builder, mi_new,
                         math::LogOp::create(fn_builder, li_new));
  }

  auto oi = LoadOp::create(fn_builder, VectorType::get(oi_2d_dims, f32),
                           fn.getArgument(o_scratch_index),
                           Indices(oi_dims.size(), i0));
  auto adjustment = MulFOp::create(
      fn_builder,
      BroadcastOp::create(
          fn_builder, VectorType::get(oi_2d_dims, f32),
          DivFOp::create(
              fn_builder,
              MulFOp::create(
                  fn_builder,
                  ExpOp::create(fn_builder,
                                SubFOp::create(fn_builder, mi, mi_new)),
                  li),
              li_new)),
      oi);

  auto vj = LoadOp::create(fn_builder, VectorType::get(v_2d_dims, ity),
                           fn.getArgument(v_index), Indices(v_dims.size(), i0));
  Value lhs = MulFOp::create(
      fn_builder,
      BroadcastOp::create(
          fn_builder, VectorType::get({qt, kt}, f32),
          DivFOp::create(fn_builder,
                         ExpOp::create(fn_builder,
                                       SubFOp::create(fn_builder, mij, mi_new)),
                         li_new)),
      pij);
  if (ity != f32) {
    lhs = TruncFOp::create(fn_builder, VectorType::get({qt, kt}, ity), lhs);
  }
  Value current = CreateContractionOp(
      fn_builder, lhs, vj,
      ConstantOp::create(fn_builder, fn_builder.getZeroAttr(VectorType::get(
                                         {qt, config.vo_head_dim}, f32))),
      MapList{{r, o}, {o, c}, {r, c}},
      ArrayRef<IteratorType>{IteratorType::parallel, IteratorType::parallel,
                             IteratorType::reduction},
      config.contract_precision_root_dot);

  Value adjusted_output = AddFOp::create(fn_builder, adjustment, current);
  Value cast_output = adjusted_output;
  if (ity != f32) {
    cast_output = TruncFOp::create(fn_builder, VectorType::get(oi_2d_dims, ity),
                                   adjusted_output);
  }

  // Store the output, we could just do this on the last iteration.
  StoreOp::create(fn_builder, cast_output, fn.getArgument(oi_index),
                  Indices(oi_dims.size(), i0));
  StoreOp::create(fn_builder, adjusted_output, fn.getArgument(o_scratch_index),
                  Indices(oi_dims.size(), i0));
  if (config.return_lse) {
    auto lse_stored =
        ShapeCastOp::create(fn_builder, VectorType::get({1, qt}, f32), lse);
    StoreOp::create(fn_builder, lse_stored, fn.getArgument(lse_index),
                    Indices(lse_dims.size(), i0));
  }
  // Store new mi.
  auto mi_new_stored =
      ShapeCastOp::create(fn_builder, VectorType::get({1, qt}, f32), mi_new);
  StoreOp::create(fn_builder, mi_new_stored, fn.getArgument(mi_index),
                  Indices(2, i0));
  // Store new li.
  auto li_new_stored =
      ShapeCastOp::create(fn_builder, VectorType::get({1, qt}, f32), li_new);
  StoreOp::create(fn_builder, li_new_stored, fn.getArgument(li_index),
                  Indices(2, i0));

  if (causal_if.has_value()) {
    fn_builder.setInsertionPointAfter(*causal_if);
  }
  ReturnOp::create(fn_builder);

  return fn;
}
}  // namespace

absl::StatusOr<std::string> CreateKernel(const FlashAttnConfig& config) {
  auto context = CreateMlirContextWithDialects();
  OpBuilder builder(context.get());
  OwningOpRef<ModuleOp> module =
      ModuleOp::create(builder, builder.getUnknownLoc());
  ImplicitLocOpBuilder module_builder(module->getLoc(),
                                      module->getBodyRegion());
  // Use default tiling for now.
  Tiling tiling = {
      .qt = std::min(kDefaultQTileSize, config.q_sequence_length),
      .kt = std::min(kDefaultKTileSize, config.kv_sequence_length),
  };
  if (config.q_sequence_length % tiling.qt != 0 ||
      config.kv_sequence_length % tiling.kt != 0) {
    return TT_ERROR(::torch_tpu::error::kInvalidArgument)
           << "padding not supported";
  }
  auto fn = buildModule(module_builder, config, tiling);

  absl::InlinedVector<int64_t, 4> q_shape = {
      config.batch_size, config.num_heads, config.q_sequence_length,
      config.qk_head_dim};

  SetKernelAttributes(q_shape, config, tiling, fn, builder);

  if (failed(SerializeMoasicKernel(module.get()))) {
    return TT_ERROR(::torch_tpu::error::kInternal)
           << "failed to serialize mosaic kernel";
  }

  std::string output;
  llvm::raw_string_ostream os(output);
  module->print(os, OpPrintingFlags().useLocalScope());
  os.flush();
  ABSL_VLOG(1) << "Forward kernel:\n" << output;
  return output;
}

}  // namespace mlir::torch_tpu
