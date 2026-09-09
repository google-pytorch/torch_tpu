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

namespace mlir::torch_tpu {

using mlir::arith::AddFOp;          // USING_DECL_OK
using mlir::arith::CmpIOp;          // USING_DECL_OK
using mlir::arith::CmpIPredicate;   // USING_DECL_OK
using mlir::arith::ConstantOp;      // USING_DECL_OK
using mlir::arith::MaximumFOp;      // USING_DECL_OK
using mlir::arith::MulFOp;          // USING_DECL_OK
using mlir::arith::SubFOp;          // USING_DECL_OK
using mlir::func::ReturnOp;         // USING_DECL_OK
using mlir::math::ExpOp;            // USING_DECL_OK
using mlir::scf::IfOp;              // USING_DECL_OK
using mlir::scf::YieldOp;           // USING_DECL_OK
using mlir::vector::CombiningKind;  // USING_DECL_OK

namespace {

void SetKernelAttributes(llvm::ArrayRef<const int64_t> qk_dimensions,
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
  absl::InlinedVector<int64_t, 4> iteration_bounds = {qk_dimensions.begin(),
                                                      qk_dimensions.end()};
  iteration_bounds[iteration_bounds.size() - 1] =
      config.padded_kv_sequence_length / tiling.kt;
  iteration_bounds[iteration_bounds.size() - 2] =
      config.padded_q_sequence_length / tiling.qt;
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

  SmallVector<int64_t> q_dims = {1, 1, qt, config.qk_head_dim};
  SmallVector<int64_t> k_dims = {1, 1, kt, config.qk_head_dim};
  SmallVector<int64_t> v_dims = {1, 1, kt, config.vo_head_dim};
  SmallVector<int64_t> oi_dims = {1, 1, qt, config.vo_head_dim};
  SmallVector<int64_t> lse_dims = {1, 1, 1, qt};
  SmallVector<int64_t> mi_dims = {qt, 128};
  SmallVector<int64_t> li_dims = {qt, 128};

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

  // There are 4 induction variables.
  unsigned int next_index = 4;
  Value q_arg = fn.getArgument(next_index++);
  Value k_arg = fn.getArgument(next_index++);
  Value v_arg = fn.getArgument(next_index++);
  Value mask_arg = has_mask ? fn.getArgument(next_index++) : nullptr;
  Value oi_arg = fn.getArgument(next_index++);
  Value lse_arg = config.return_lse ? fn.getArgument(next_index++) : nullptr;
  Value o_scratch_arg = fn.getArgument(next_index++);
  Value mi_arg = fn.getArgument(next_index++);
  Value li_arg = fn.getArgument(next_index++);

  ConstantOp zero = ConstantOp::create(fn_builder, fn_builder.getZeroAttr(i32));
  std::optional<scf::IfOp> causal_if;
  if (config.is_causal) {
    causal_if = CreateCausalIfOp(fn_builder, fn.getArgument(2),
                                 fn.getArgument(3), qt, kt);
    fn_builder.setInsertionPointToStart(causal_if->thenBlock());
  }

  IfOp::create(
      fn_builder,
      CmpIOp::create(fn_builder, CmpIPredicate::eq, zero, fn.getArgument(3)),
      /*thenBuilder=*/
      [&](OpBuilder& builder, Location loc) -> void {
        ImplicitLocOpBuilder b(loc, builder);
        ZeroTile(b, o_scratch_arg);
        ZeroTile(b, li_arg);
        NInfTile(b, mi_arg);
        YieldOp::create(b, loc);
      });

  auto q = LoadTile(fn_builder, q_arg);
  auto k = LoadTile(fn_builder, k_arg);
  Value sij =
      CreateMatmul(fn_builder, q, k, MatmulOptions{.transpose_rhs = true});

  sij = ScaleValue(fn_builder, sij, config.scale);

  Value row_block_idx = fn.getArgument(2);
  Value col_block_idx = fn.getArgument(3);
  Value bias = GetStructuredBias(fn_builder, row_block_idx, col_block_idx, qt,
                                 kt, config.q_sequence_length,
                                 config.kv_sequence_length, config.is_causal);

  if (mask_arg) {
    Value user_bias = LoadTile(fn_builder, mask_arg);
    user_bias = ConvertElementType(fn_builder, f32, user_bias);
    if (bias) {
      bias = AddFOp::create(fn_builder, bias, user_bias);
    } else {
      bias = user_bias;
    }
  }

  if (bias) {
    bias = ConvertElementType(fn_builder, f32, bias);
    sij = AddFOp::create(fn_builder, sij, bias);
  }

  if (mask_arg) {
    // If a user provides a mask we must defend against a row being entirely
    // masked out. without this clamp we can get nan values.
    // If there is no user mask there is no possibility of this happening.
    sij = ClampLogits(fn_builder, sij);
  }

  auto mij = ReduceBroadcastLane(fn_builder, sij, CombiningKind::MAXIMUMF);

  auto mi_prev = LoadTile(fn_builder, mi_arg);
  auto mi_new = MaximumFOp::create(fn_builder, mi_prev, mij);

  auto alpha =
      ExpOp::create(fn_builder, SubFOp::create(fn_builder, mi_prev, mi_new));

  auto p_tile_scaled = ExpOp::create(
      fn_builder, SubFOp::create(fn_builder, sij,
                                 NormalizeLaneDim(fn_builder, mi_new, kt)));

  auto tile_sum =
      ReduceBroadcastLane(fn_builder, p_tile_scaled, CombiningKind::ADD);

  auto li_prev = LoadTile(fn_builder, li_arg);
  auto li_prev_scaled = MulFOp::create(fn_builder, alpha, li_prev);

  auto li_new = AddFOp::create(fn_builder, li_prev_scaled, tile_sum);

  auto oi = LoadTile(fn_builder, o_scratch_arg);
  auto adjustment = MulFOp::create(
      fn_builder, NormalizeLaneDim(fn_builder, alpha, config.vo_head_dim), oi);

  auto vj = LoadTile(fn_builder, v_arg);
  Value lhs = ConvertElementType(fn_builder, ity, p_tile_scaled);

  Value current = CreateMatmul(fn_builder, lhs, vj);

  Value unnormalized_output = AddFOp::create(fn_builder, adjustment, current);

  StoreTile(fn_builder, mi_new, mi_arg);
  StoreTile(fn_builder, li_new, li_arg);
  StoreTile(fn_builder, unnormalized_output, o_scratch_arg);

  if (causal_if.has_value()) {
    fn_builder.setInsertionPointAfter(*causal_if);
  }

  Value last_col_idx = ConstantOp::create(
      fn_builder, fn_builder.getI32IntegerAttr(
                      (config.padded_kv_sequence_length / tiling.kt) - 1));
  Value is_last_col = CmpIOp::create(fn_builder, CmpIPredicate::eq,
                                     fn.getArgument(3), last_col_idx);
  IfOp::create(fn_builder, is_last_col,
               /*thenBuilder=*/
               [&](OpBuilder& builder, Location loc) -> void {
                 ImplicitLocOpBuilder b(loc, builder);
                 Value unnormalized_output = LoadTile(b, o_scratch_arg);
                 Value li = LoadTile(b, li_arg);
                 Value li_recip = CreateReciprocal(b, li);
                 Value normalized_output = MulFOp::create(
                     b, unnormalized_output,
                     NormalizeLaneDim(b, li_recip, config.vo_head_dim));
                 StoreTile(b, normalized_output, oi_arg);
                 if (config.return_lse) {
                   Value mi = LoadTile(b, mi_arg);
                   Value lse =
                       AddFOp::create(b, mi, math::LogOp::create(b, li));
                   Value sliced_lse = NormalizeLaneDim(b, lse, 1);
                   Value transposed_lse =
                       vector::TransposeOp::create(b, sliced_lse, {1, 0});
                   StoreTile(b, transposed_lse, lse_arg);
                 }
                 YieldOp::create(b, loc);
               });

  ReturnOp::create(fn_builder);

  return fn;
}
}  // namespace

absl::StatusOr<OwningOpRef<ModuleOp>> CreateKernel(
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
  auto fn = buildModule(module_builder, config, tiling);

  absl::InlinedVector<int64_t, 4> q_shape = {
      config.batch_size, config.num_heads, config.q_sequence_length,
      config.qk_head_dim};

  SetKernelAttributes(q_shape, config, tiling, fn, builder);

  ABSL_VLOG(1) << "Forward kernel:\n" << GetOpString(module.get());

  return module;
}

}  // namespace mlir::torch_tpu
