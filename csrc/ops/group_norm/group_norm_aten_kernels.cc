/*
 * Copyright 2025 Google LLC
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

#include "csrc/ops/group_norm/group_norm_aten_kernels.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/sum.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

struct GroupNormShloResults {
  mlir::MlirOp output;
  mlir::MlirOp mean;
  mlir::MlirOp rstd;
};

struct GroupNormBackwardShloResults {
  mlir::MlirOp grad_input;
  mlir::MlirOp grad_weight;
  mlir::MlirOp grad_bias;
};

namespace {

enum class InputLayout {
  // Input shape starts with [N, C, ...], so spatial axes need not be flattened.
  kNcPrefix,
  // Input does not start with [N, C, ...] (e.g. flat 1D tensor of N*C*HxW).
  kFlat,
};

Dimensions MakeDimRange(int64_t start_dim, int64_t end_dim_exclusive) {
  Dimensions dims(end_dim_exclusive - start_dim);
  std::iota(dims.begin(), dims.end(), start_dim);
  return dims;
}

absl::StatusOr<GroupNormShloResults> BuildGroupNormShlo(
    mlir::MlirOp input_op, std::optional<mlir::MlirOp> weight_op,
    std::optional<mlir::MlirOp> bias_op, int64_t group, double eps) {
  const mlir::ElementType input_elem_type = GetElementTypeOrDie(input_op);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType comp_elem_type,
                      InferComputationDtype(input_elem_type));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp input_comp,
                      CastIfNeeded(input_op, comp_elem_type));
  const mlir::Type comp_type = GetTensorTypeOrDie(input_comp).getElementType();

  // Split channel axis [N, C, *] -> [N, G, C/G, *] without flattening spatial
  // dimensions so all dynamic batch/channel/spatial bounds are preserved.
  TT_ASSIGN_OR_RETURN(mlir::MlirOp grouped,
                      SplitDim(input_comp, /*dim=*/1, /*outer_size=*/group));
  const int64_t grouped_rank = GetTensorTypeOrDie(grouped).getRank();
  const Dimensions group_reduce_dims = MakeDimRange(2, grouped_rank);

  mlir::MlirOp group_elem_count =
      GetNumElements(grouped, comp_type, group_reduce_dims);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp sum_x,
                      BuildSumShlo(grouped, group_reduce_dims));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp count_bcast,
                      BroadcastIfNeeded(group_elem_count, sum_x));
  mlir::MlirOp mean = mlir::stablehlo::Div(sum_x, count_bcast);

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp mean_bcast,
      BroadcastIfNeeded(mean, grouped, /*broadcast_dimensions=*/{0, 1}));
  mlir::MlirOp centered = mlir::stablehlo::Subtract(grouped, mean_bcast);
  mlir::MlirOp sq_diff = mlir::stablehlo::Mul(centered, centered);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp sum_sq_diff,
                      BuildSumShlo(sq_diff, group_reduce_dims));
  mlir::MlirOp variance = mlir::stablehlo::Div(sum_sq_diff, count_bcast);

  mlir::MlirOp eps_op = MakeConstantLike(variance, eps);
  mlir::MlirOp variance_plus_eps = mlir::stablehlo::Add(variance, eps_op);
  mlir::MlirOp rstd = mlir::stablehlo::Rsqrt(variance_plus_eps);

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp rstd_bcast,
      BroadcastIfNeeded(rstd, grouped, /*broadcast_dimensions=*/{0, 1}));
  mlir::MlirOp norm_grouped = mlir::stablehlo::Mul(centered, rstd_bcast);
  mlir::MlirOp out_comp =
      MergeAdjacentDims(norm_grouped, /*start_dim=*/1, /*end_dim=*/2);

  if (weight_op.has_value()) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp weight_comp,
                        CastIfNeeded(*weight_op, comp_elem_type));
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp weight_bcast,
        BroadcastIfNeeded(weight_comp, out_comp, /*broadcast_dimensions=*/{1}));
    out_comp = mlir::stablehlo::Mul(out_comp, weight_bcast);
  }
  if (bias_op.has_value()) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp bias_comp,
                        CastIfNeeded(*bias_op, comp_elem_type));
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp bias_bcast,
        BroadcastIfNeeded(bias_comp, out_comp, /*broadcast_dimensions=*/{1}));
    out_comp = mlir::stablehlo::Add(out_comp, bias_bcast);
  }

  TT_ASSIGN_OR_RETURN(mlir::MlirOp output,
                      CastIfNeeded(out_comp, input_elem_type));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mean_out,
                      CastIfNeeded(mean, input_elem_type));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp rstd_out,
                      CastIfNeeded(rstd, input_elem_type));

  return GroupNormShloResults{
      .output = output, .mean = mean_out, .rstd = rstd_out};
}

// Math pseudo code, referencing pytorch decomposition:
// py/torch/_decomp/decompositions.py;l=1512
//
// Common parts:
//   cpg = C // group
//   ds = sum(grad_output * input, dim=spatial_dims)  -> [N, C]
//   db = sum(grad_output, dim=spatial_dims)          -> [N, C]
//
// d_input (Mask 0):
//   s = 1.0 / (HxW * cpg)
//   If weight exists:
//      ds_val = sum(ds * weight, dim=2_of_group) -> [N, G]
//      db_val = sum(db * weight, dim=2_of_group) -> [N, G]
//      c1 = rstd[..., None] * weight
//   Else:
//      ds_val = sum(ds, dim=2_of_group) -> [N, G]
//      db_val = sum(db, dim=2_of_group) -> [N, G]
//      c1 = rstd[..., None] * 1
//
//   c2 = (db_val * mean - ds_val) * rstd^3 * s
//   c3 = -c2 * mean - db_val * rstd * s
//   dX = grad * c1 + input * c2 + c3
//
// d_weight (Mask 1):
//   term = (ds - db * mean) * rstd  (all viewed as [N, G, CPG])
//   d_weight = sum(term, dim=0) -> [C]
//
// d_bias (Mask 2):
//   d_bias = sum(db, dim=0) -> [C]
absl::StatusOr<GroupNormBackwardShloResults> BuildGroupNormBackwardShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp input_op, mlir::MlirOp mean_op,
    mlir::MlirOp rstd_op, std::optional<mlir::MlirOp> weight_op, int64_t n,
    int64_t c, int64_t h_w, int64_t group, std::array<bool, 3> output_mask,
    InputLayout layout) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const Dimensions input_static_shape = GetStaticShape(input_op);
  const mlir::ElementType input_elem_type = GetElementTypeOrDie(input_op);
  const mlir::ElementType weight_elem_type =
      weight_op.has_value() ? GetElementTypeOrDie(weight_op.value())
                            : input_elem_type;

  TT_ASSIGN_OR_RETURN(const mlir::ElementType comp_elem_type,
                      InferComputationDtype(input_elem_type));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_output_comp,
                      CastIfNeeded(grad_output_op, comp_elem_type));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp input_comp,
                      CastIfNeeded(input_op, comp_elem_type));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mean_comp,
                      CastIfNeeded(mean_op, comp_elem_type));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp rstd_comp,
                      CastIfNeeded(rstd_op, comp_elem_type));
  std::optional<mlir::MlirOp> weight_comp;
  if (weight_op.has_value()) {
    TT_ASSIGN_OR_RETURN(weight_comp,
                        CastIfNeeded(weight_op.value(), comp_elem_type));
  }

  const mlir::Type comp_type = GetTensorTypeOrDie(input_comp).getElementType();

  // Keep [N, C, *] layout directly whenever input starts with [N, C] so spatial
  // dimensions are never flattened; fall back to [N, C, HxW] for static flat
  // inputs.
  if (layout == InputLayout::kFlat) {
    TT_RET_CHECK(GetTensorTypeOrDie(input_op).hasStaticShape(),
                 error::kPythonNotImplementedError)
        << "expected static input shape when input does not start with [N, C], "
           "got "
        << ToString(input_static_shape);
    const Dimensions n_c_hxw_shape = {n, c, h_w};
    TT_ASSIGN_OR_RETURN(grad_output_comp, ReshapeToStaticDimensions(
                                              grad_output_comp, n_c_hxw_shape));
    TT_ASSIGN_OR_RETURN(input_comp,
                        ReshapeToStaticDimensions(input_comp, n_c_hxw_shape));
  }

  // 1. Compute Common Terms. ds and db are [N, C].
  mlir::MlirOp mul_in_grad = mlir::stablehlo::Mul(grad_output_comp, input_comp);
  const int64_t comp_rank = GetTensorTypeOrDie(input_comp).getRank();
  mlir::MlirOp ds = mul_in_grad;
  mlir::MlirOp db = grad_output_comp;
  if (comp_rank > 2) {
    const Dimensions spatial_dims = MakeDimRange(2, comp_rank);
    TT_ASSIGN_OR_RETURN(ds, BuildSumShlo(mul_in_grad, spatial_dims));
    TT_ASSIGN_OR_RETURN(db, BuildSumShlo(grad_output_comp, spatial_dims));
  }

  // Initialize placeholder outputs for masked-out gradients.
  mlir::MlirOp d_input_op = MakeScalarConstant(builder, 0.0, input_elem_type);
  mlir::MlirOp d_weight_op = MakeScalarConstant(builder, 0.0, weight_elem_type);
  mlir::MlirOp d_bias_op = MakeScalarConstant(builder, 0.0, weight_elem_type);

  // 2. Compute d_input_op
  if (output_mask[0]) {
    mlir::MlirOp ds_val, db_val, c1;

    if (weight_comp.has_value()) {
      // weight: [C] -> Broadcast to [N, C] to multiply with ds/db
      TT_ASSIGN_OR_RETURN(mlir::MlirOp weight_broadcast_nc,
                          BroadcastIfNeeded(weight_comp.value(), ds,
                                            /*broadcast_dimensions=*/{1}));

      mlir::MlirOp ds_weighted = mlir::stablehlo::Mul(ds, weight_broadcast_nc);
      mlir::MlirOp db_weighted = mlir::stablehlo::Mul(db, weight_broadcast_nc);

      // Split weighted ds/db [N, C] -> [N, G, CPG]
      TT_ASSIGN_OR_RETURN(
          mlir::MlirOp ds_weighted_reshaped,
          SplitDim(ds_weighted, /*dim=*/1, /*outer_size=*/group));
      TT_ASSIGN_OR_RETURN(
          mlir::MlirOp db_weighted_reshaped,
          SplitDim(db_weighted, /*dim=*/1, /*outer_size=*/group));

      // Reduce on dim 2 (CPG) -> [N, G]
      TT_ASSIGN_OR_RETURN(ds_val, BuildSumShlo(ds_weighted_reshaped, {2}));
      TT_ASSIGN_OR_RETURN(db_val, BuildSumShlo(db_weighted_reshaped, {2}));

      // rstd: [N, G] -> Broadcast to [N, G, CPG] (dims 0, 1)
      TT_ASSIGN_OR_RETURN(mlir::MlirOp rstd_bcast_ngc,
                          BroadcastIfNeeded(rstd_comp, ds_weighted_reshaped,
                                            /*broadcast_dimensions=*/{0, 1}));

      // weight: [C] -> Split [G, CPG] -> Broadcast to [N, G, CPG] (dims 1, 2)
      TT_ASSIGN_OR_RETURN(
          mlir::MlirOp weight_reshaped_gc,
          SplitDim(weight_comp.value(), /*dim=*/0, /*outer_size=*/group));
      TT_ASSIGN_OR_RETURN(
          mlir::MlirOp weight_bcast_ngc,
          BroadcastIfNeeded(weight_reshaped_gc, ds_weighted_reshaped,
                            /*broadcast_dimensions=*/{1, 2}));

      // c1 = rstd.unsqueeze(-1) * weight.reshape(1, group, cpg)
      c1 = mlir::stablehlo::Mul(rstd_bcast_ngc, weight_bcast_ngc);
    } else {
      TT_ASSIGN_OR_RETURN(mlir::MlirOp ds_reshaped,
                          SplitDim(ds, /*dim=*/1, /*outer_size=*/group));
      TT_ASSIGN_OR_RETURN(mlir::MlirOp db_reshaped,
                          SplitDim(db, /*dim=*/1, /*outer_size=*/group));

      TT_ASSIGN_OR_RETURN(ds_val, BuildSumShlo(ds_reshaped, {2}));
      TT_ASSIGN_OR_RETURN(db_val, BuildSumShlo(db_reshaped, {2}));

      // c1 = rstd.unsqueeze(-1) * 1
      // Effectively just broadcasting rstd [N, G] -> [N, G, CPG]
      TT_ASSIGN_OR_RETURN(c1,
                          BroadcastIfNeeded(rstd_comp, ds_reshaped,
                                            /*broadcast_dimensions=*/{0, 1}));
    }

    // Split grad_output/input to [N, G, CPG, *]
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp grad_grouped,
        SplitDim(grad_output_comp, /*dim=*/1, /*outer_size=*/group));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp input_grouped,
                        SplitDim(input_comp, /*dim=*/1, /*outer_size=*/group));
    const int64_t grouped_rank = GetTensorTypeOrDie(grad_grouped).getRank();
    const Dimensions group_reduce_dims = MakeDimRange(2, grouped_rank);

    // Derive s = 1.0 / (HxW * CPG) from runtime dimension sizes.
    mlir::MlirOp group_elem_count =
        GetNumElements(grad_grouped, comp_type, group_reduce_dims);
    mlir::MlirOp one = MakeScalarConstant(builder, 1.0, comp_type);
    mlir::MlirOp s = mlir::stablehlo::Div(one, group_elem_count);
    TT_ASSIGN_OR_RETURN(mlir::MlirOp s_broadcast, BroadcastIfNeeded(s, ds_val));

    // All ops here are [N, G]
    // c2 = (db_val * mean - ds_val) * rstd * rstd * rstd * s
    mlir::MlirOp db_val_mean = mlir::stablehlo::Mul(db_val, mean_comp);
    mlir::MlirOp mean_sub_ds = mlir::stablehlo::Subtract(db_val_mean, ds_val);
    mlir::MlirOp rstd_pow2 = mlir::stablehlo::Mul(rstd_comp, rstd_comp);
    mlir::MlirOp rstd_pow3 = mlir::stablehlo::Mul(rstd_pow2, rstd_comp);
    mlir::MlirOp c2_term = mlir::stablehlo::Mul(mean_sub_ds, rstd_pow3);
    mlir::MlirOp c2 = mlir::stablehlo::Mul(c2_term, s_broadcast);

    // c3 = -c2 * mean - db_val * rstd * s
    mlir::MlirOp neg_c2 = mlir::stablehlo::Neg(c2);
    mlir::MlirOp neg_c2_mean = mlir::stablehlo::Mul(neg_c2, mean_comp);
    mlir::MlirOp db_val_rstd = mlir::stablehlo::Mul(db_val, rstd_comp);
    mlir::MlirOp db_val_rstd_s = mlir::stablehlo::Mul(db_val_rstd, s_broadcast);
    mlir::MlirOp c3 = mlir::stablehlo::Subtract(neg_c2_mean, db_val_rstd_s);

    // d_input = grad_output * c1 + input * c2 + c3
    // Broadcast c1, c2, c3 to [N, G, CPG, *]
    TT_ASSIGN_OR_RETURN(mlir::MlirOp c1_final,
                        BroadcastIfNeeded(c1, grad_grouped,
                                          /*broadcast_dimensions=*/{0, 1, 2}));
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp c2_final,
        BroadcastIfNeeded(c2, grad_grouped, /*broadcast_dimensions=*/{0, 1}));
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp c3_final,
        BroadcastIfNeeded(c3, grad_grouped, /*broadcast_dimensions=*/{0, 1}));

    mlir::MlirOp term1 = mlir::stablehlo::Mul(grad_grouped, c1_final);
    mlir::MlirOp term2 = mlir::stablehlo::Mul(input_grouped, c2_final);

    mlir::MlirOp dx_full = mlir::stablehlo::Add(term1, term2);
    dx_full = mlir::stablehlo::Add(dx_full, c3_final);

    // Merge [N, G, CPG, *] -> [N, C, *]
    mlir::MlirOp dx_reshaped =
        MergeAdjacentDims(dx_full, /*start_dim=*/1, /*end_dim=*/2);
    if (layout == InputLayout::kFlat) {
      TT_ASSIGN_OR_RETURN(dx_reshaped, ReshapeToStaticDimensions(
                                           dx_reshaped, input_static_shape));
    }
    TT_ASSIGN_OR_RETURN(d_input_op, CastIfNeeded(dx_reshaped, input_elem_type));
  }

  // 3. Compute d_weight_op
  if (output_mask[1]) {
    // All views are [N, G, CPG]
    // (ds.view - db.view * mean.unsqueeze) * rstd.unsqueeze
    TT_ASSIGN_OR_RETURN(mlir::MlirOp ds_view,
                        SplitDim(ds, /*dim=*/1, /*outer_size=*/group));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp db_view,
                        SplitDim(db, /*dim=*/1, /*outer_size=*/group));

    // mean/rstd [N, G] -> [N, G, CPG] (dims 0, 1)
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp mean_bcast,
        BroadcastIfNeeded(mean_comp, ds_view, /*broadcast_dimensions=*/{0, 1}));
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp rstd_bcast,
        BroadcastIfNeeded(rstd_comp, ds_view, /*broadcast_dimensions=*/{0, 1}));

    mlir::MlirOp db_mean = mlir::stablehlo::Mul(db_view, mean_bcast);
    mlir::MlirOp diff = mlir::stablehlo::Subtract(ds_view, db_mean);
    mlir::MlirOp mul_rstd = mlir::stablehlo::Mul(diff, rstd_bcast);

    // sum(dim=0) -> [G, CPG]
    TT_ASSIGN_OR_RETURN(mlir::MlirOp sum_term, BuildSumShlo(mul_rstd, {0}));

    // Merge [G, CPG] -> [C]
    mlir::MlirOp d_weight_comp =
        MergeAdjacentDims(sum_term, /*start_dim=*/0, /*end_dim=*/1);
    TT_ASSIGN_OR_RETURN(d_weight_op,
                        CastIfNeeded(d_weight_comp, weight_elem_type));
  }

  // 4. Compute d_bias_op
  if (output_mask[2]) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp d_bias_comp, BuildSumShlo(db, {0}));
    TT_ASSIGN_OR_RETURN(d_bias_op, CastIfNeeded(d_bias_comp, weight_elem_type));
  }

  return GroupNormBackwardShloResults{.grad_input = d_input_op,
                                      .grad_weight = d_weight_op,
                                      .grad_bias = d_bias_op};
}

at::Tensor MakeEmptyTensorIf(const bool enabled, at::IntArrayRef sizes,
                             const c10::ScalarType dtype,
                             const at::Tensor& device_source) {
  if (!enabled) {
    return at::Tensor();
  }
  TT_ASSIGN_OR_THROW(at::Tensor result,
                     MakeEmptyTensor(sizes, dtype, device_source.device()));
  return result;
}

absl::StatusOr<MlirOpResults<2>> BuildZeroChannelGradsShlo(
    absl::Span<const mlir::MlirOp> inputs, mlir::MlirBuilder& builder,
    const int64_t c, const mlir::ElementType weight_dtype,
    const std::array<bool, 3> output_mask) {
  mlir::MlirOp zero_scalar = MakeScalarConstant(builder, 0.0, weight_dtype);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp zero_c,
                      inputs.empty()
                          ? BroadcastIfNeeded(zero_scalar, {c})
                          : BroadcastIfNeeded(zero_scalar, inputs[0]));
  return {{output_mask[1] ? zero_c : zero_scalar,
           output_mask[2] ? zero_c : zero_scalar}};
}

// Handles empty inputs (`n == 0 || c == 0 || h_w == 0`) before DispatchOp so
// zero-sized tensors (which dynamic padding adapters omit) are not passed into
// an executable.
std::tuple<at::Tensor, at::Tensor, at::Tensor> DispatchEmptyGroupNormBackward(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const bool has_weight, const int64_t c,
    const mlir::ElementType weight_dtype, const std::array<bool, 3> output_mask,
    const Dimensions& weight_grad_dims, const Dimensions& bias_grad_dims,
    OpParamCacheKeys param_keys) {
  at::Tensor d_input = MakeEmptyTensorIf(output_mask[0], input.sizes(),
                                         input.scalar_type(), input);
  const c10::ScalarType weight_scalar_type =
      has_weight ? weight->scalar_type() : input.scalar_type();
  if (c == 0) {
    return {
        std::move(d_input),
        MakeEmptyTensorIf(output_mask[1], {0}, weight_scalar_type, input),
        MakeEmptyTensorIf(output_mask[2], {0}, weight_scalar_type, input),
    };
  }
  if (!output_mask[1] && !output_mask[2]) {
    return {std::move(d_input), at::Tensor(), at::Tensor()};
  }
  std::vector<at::Tensor> zero_inputs;
  if (has_weight) {
    zero_inputs.push_back(*weight);
  } else {
    TT_THROW_IF_ERROR(param_keys.SetParam("c", c));
  }
  auto zero_builder = [c, weight_dtype, output_mask](
                          absl::Span<const mlir::MlirOp> inputs,
                          mlir::MlirBuilder& builder) {
    return BuildZeroChannelGradsShlo(inputs, builder, c, weight_dtype,
                                     output_mask);
  };
  TT_ASSIGN_OR_THROW((auto [d_weight_buf, d_bias_buf]),
                     (DispatchOp<kDynamicSize, 2>(
                         std::move(zero_builder),
                         /*inputs=*/zero_inputs,
                         {.out_dtypes = {weight_dtype, weight_dtype},
                          .out_dims_list = {weight_grad_dims, bias_grad_dims},
                          .op_param_cache_keys = std::move(param_keys)})));
  return {
      std::move(d_input),
      output_mask[1] ? MakeTensor(std::move(d_weight_buf)) : at::Tensor(),
      output_mask[2] ? MakeTensor(std::move(d_bias_buf)) : at::Tensor(),
  };
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenNativeGroupNorm(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, int64_t n, int64_t c, int64_t h_w,
    int64_t group, double eps) {
  TT_KERNEL(
      OpName::kNativeGroupNorm, param_keys,
      (input, weight, bias,
       IgnoreInCacheKey(n, "Derived from bounded input shape"),
       IgnoreInCacheKey(c, "Derived from bounded input shape"),
       IgnoreInCacheKey(h_w, "Derived from bounded input shape"), group, eps),
      {
        const std::optional<at::Tensor> sanitized_weight =
            SanitizeOptionalTensor(weight);
        const std::optional<at::Tensor> sanitized_bias =
            SanitizeOptionalTensor(bias);
        const bool has_weight = sanitized_weight.has_value();
        const bool has_bias = sanitized_bias.has_value();

        TT_CHECK_THROW(input.dim() >= 2, error::kInvalidArgument)
            << "expected input to have >= 2 dimensions, got " << input.dim();
        TT_CHECK_THROW(c > 0, error::kInvalidArgument)
            << "expected number of channels to be > 0, got " << c;
        TT_CHECK_THROW(h_w > 0, error::kInvalidArgument)
            << "expected HxW to be > 0, got " << h_w;
        TT_CHECK_THROW(group > 0, error::kInvalidArgument)
            << "expected num_groups to be > 0, got " << group;
        TT_CHECK_THROW(c % group == 0, error::kInvalidArgument)
            << "expected number of channels to be divisible by num_groups, got "
            << c << " channels and num_groups=" << group;
        TT_ASSIGN_OR_THROW(const int64_t expected_numel,
                           NumElements({n, c, h_w}));
        TT_CHECK_THROW(input.size(0) == n && input.size(1) == c &&
                           input.numel() == expected_numel,
                       error::kInvalidArgument)
            << "expected input to have shape [" << n << ", " << c
            << ", ...] with " << expected_numel << " elements, got shape "
            << ToString(input.sizes());
        TT_CHECK_THROW(!has_weight || (sanitized_weight->dim() == 1 &&
                                       sanitized_weight->numel() == c),
                       error::kInvalidArgument)
            << "expected weight to be a 1D tensor of size " << c
            << ", got shape " << ToString(sanitized_weight->sizes());
        TT_CHECK_THROW(!has_bias || (sanitized_bias->dim() == 1 &&
                                     sanitized_bias->numel() == c),
                       error::kInvalidArgument)
            << "expected bias to be a 1D tensor of size " << c << ", got shape "
            << ToString(sanitized_bias->sizes());

        const bool is_mixed_type =
            (has_weight &&
             sanitized_weight->scalar_type() != input.scalar_type()) ||
            (has_bias && sanitized_bias->scalar_type() != input.scalar_type());
        if (is_mixed_type) {
          TT_CHECK_THROW(!has_weight || sanitized_weight->scalar_type() ==
                                            at::ScalarType::Float,
                         error::kInvalidArgument)
              << "expected weight to have dtype float32 for mixed dtype, got "
              << ToString(sanitized_weight->scalar_type());
          TT_CHECK_THROW(!has_bias || sanitized_bias->scalar_type() ==
                                          at::ScalarType::Float,
                         error::kInvalidArgument)
              << "expected bias to have dtype float32 for mixed dtype, got "
              << ToString(sanitized_bias->scalar_type());
          TT_CHECK_THROW(at::isReducedFloatingType(input.scalar_type()),
                         error::kInvalidArgument)
              << "expected input to have dtype bfloat16 or float16 for mixed "
                 "dtype, got "
              << ToString(input.scalar_type());
        }
        // Return mean and rstd in input.scalar_type() to match PyTorch's
        // Meta/FakeTensor kernel (torch/_refs/native_group_norm) and
        // math_group_norm so AOTAutograd saved tensors match at runtime.
        if (n == 0) {
          TT_ASSIGN_OR_THROW(at::Tensor out,
                             MakeEmptyTensor(input.sizes(), input.scalar_type(),
                                             input.device()));
          TT_ASSIGN_OR_THROW(
              at::Tensor mean,
              MakeEmptyTensor({0, group}, input.scalar_type(), input.device()));
          TT_ASSIGN_OR_THROW(
              at::Tensor rstd,
              MakeEmptyTensor({0, group}, input.scalar_type(), input.device()));
          return {out, mean, rstd};
        }

        TT_CHECK_THROW(IsFloatingPoint(input), error::kInvalidArgument)
            << "expected the input dtype to be floating point, got "
            << ToString(input.scalar_type());

        TT_ASSIGN_OR_THROW(const mlir::ElementType input_dtype,
                           ConvertTo<mlir::ElementType>(input.scalar_type()));

        auto op_builder = [has_weight, has_bias, group, eps](
                              absl::Span<const mlir::MlirOp> inputs,
                              mlir::MlirBuilder& builder)
            -> absl::StatusOr<MlirOpResults<3>> {
          mlir::MlirOp input_op = inputs[0];
          std::optional<mlir::MlirOp> weight_op;
          std::optional<mlir::MlirOp> bias_op;
          size_t next_idx = 1;
          if (has_weight) {
            weight_op = inputs[next_idx++];
          }
          if (has_bias) {
            bias_op = inputs[next_idx++];
          }

          TT_ASSIGN_OR_RETURN(
              GroupNormShloResults results,
              BuildGroupNormShlo(input_op, weight_op, bias_op, group, eps));
          return {{results.output, results.mean, results.rstd}};
        };

        std::vector<at::Tensor> inputs = {input};
        if (has_weight) {
          inputs.push_back(*sanitized_weight);
        }
        if (has_bias) {
          inputs.push_back(*sanitized_bias);
        }

        const Dimensions input_dims = CopyIntVector(input.sizes());
        const Dimensions stat_dims = {n, group};
        TT_ASSIGN_OR_THROW(
            (auto [output_buf, mean_buf, rstd_buf]),
            (DispatchOp<kDynamicSize, 3>(
                std::move(op_builder),
                /*inputs=*/inputs,
                {.out_dtypes = {input_dtype, input_dtype, input_dtype},
                 .out_dims_list = {input_dims, stat_dims, stat_dims},
                 .op_param_cache_keys = std::move(param_keys)})));

        return {MakeTensor(std::move(output_buf)),
                MakeTensor(std::move(mean_buf)),
                MakeTensor(std::move(rstd_buf))};
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenNativeGroupNormBackward(
    const at::Tensor& grad_out, const at::Tensor& input, const at::Tensor& mean,
    const at::Tensor& rstd, const std::optional<at::Tensor>& weight, int64_t n,
    int64_t c, int64_t h_w, int64_t group, std::array<bool, 3> output_mask) {
  TT_KERNEL(
      OpName::kNativeGroupNormBackward, param_keys,
      (grad_out, input, mean, rstd, weight,
       IgnoreInCacheKey(n, "Only affects flat-layout fallback"),
       IgnoreInCacheKey(c, "Only affects flat-layout fallback"),
       IgnoreInCacheKey(h_w, "Only affects flat-layout fallback"), group,
       output_mask),
      {
        // Input validations:
        // grad_out & input shape: (N, C, HxW)
        // mean & rstd shape: (N, group)
        // weight shape: (C) if provided else ()
        TT_ASSIGN_OR_THROW(const int64_t expected_numel,
                           NumElements({n, c, h_w}));
        TT_CHECK_THROW(grad_out.numel() == expected_numel,
                       error::kInvalidArgument)
            << "expected grad_out to have " << expected_numel << " elements"
            << ", got " << grad_out.numel();
        TT_CHECK_THROW(input.numel() == expected_numel, error::kInvalidArgument)
            << "expected input to have " << expected_numel << " elements"
            << ", got " << input.numel();
        TT_CHECK_THROW(grad_out.sizes() == input.sizes(),
                       error::kInvalidArgument)
            << "expected grad_out and input to have the same dimensions, "
               "got grad_out size "
            << grad_out.sizes() << ", input size " << input.sizes();
        TT_CHECK_THROW(mean.sizes() == at::IntArrayRef({n, group}),
                       error::kInvalidArgument)
            << "expected mean to have shape [" << n << ", " << group << "]"
            << ", got " << mean.sizes();
        TT_CHECK_THROW(mean.sizes() == rstd.sizes(), error::kInvalidArgument)
            << "expected mean and rstd to have the same shape"
            << ", got mean size " << mean.sizes() << " and rstd size "
            << rstd.sizes();
        TT_CHECK_THROW(group > 0, error::kInvalidArgument)
            << "expected num_groups to be > 0, got " << group;
        TT_CHECK_THROW(c % group == 0, error::kInvalidArgument)
            << "expected number of channels to be divisible by num_groups, got "
            << c << " channels and num_groups=" << group;

        TT_ASSIGN_OR_THROW(auto input_dtype,
                           ConvertTo<mlir::ElementType>(input.scalar_type()));
        Dimensions input_dims = CopyIntVector(input.sizes());

        // If weight exists, grad_weight is [C]
        bool has_weight = weight.has_value() && weight->defined();
        if (has_weight) {
          TT_CHECK_THROW(weight.value().numel() == c, error::kInvalidArgument)
              << "expected weight to have " << c << " elements"
              << ", got " << (weight ? weight->numel() : -1);
        }
        Dimensions d_input_dims = output_mask[0] ? input_dims : Dimensions();
        Dimensions weight_grad_dims =
            output_mask[1] ? Dimensions{c} : Dimensions();
        Dimensions bias_grad_dims =
            output_mask[2] ? Dimensions{c} : Dimensions();

        mlir::ElementType weight_dtype = input_dtype;
        if (has_weight) {
          TT_ASSIGN_OR_THROW(weight_dtype, ConvertTo<mlir::ElementType>(
                                               weight->scalar_type()));
        }

        if (n == 0 || c == 0 || h_w == 0) {
          return DispatchEmptyGroupNormBackward(
              input, weight, has_weight, c, weight_dtype, output_mask,
              weight_grad_dims, bias_grad_dims, std::move(param_keys));
        }

        const InputLayout layout =
            (input.dim() >= 2 && input.size(0) == n && input.size(1) == c)
                ? InputLayout::kNcPrefix
                : InputLayout::kFlat;
        if (layout == InputLayout::kFlat) {
          TT_THROW_IF_ERROR(param_keys.SetParam("n", n));
          TT_THROW_IF_ERROR(param_keys.SetParam("c", c));
          TT_THROW_IF_ERROR(param_keys.SetParam("h_w", h_w));
        }

        auto op_builder = [n, c, h_w, group, output_mask, layout](
                              absl::Span<const mlir::MlirOp> inputs,
                              mlir::MlirBuilder& builder)
            -> absl::StatusOr<MlirOpResults<3>> {
          auto grad_out_op = inputs[0];
          auto input_op = inputs[1];
          auto mean_op = inputs[2];
          auto rstd_op = inputs[3];

          std::optional<mlir::MlirOp> weight_op;
          if (inputs.size() == 5) {
            weight_op = inputs[4];
          }

          TT_ASSIGN_OR_RETURN(
              GroupNormBackwardShloResults results,
              BuildGroupNormBackwardShlo(grad_out_op, input_op, mean_op,
                                         rstd_op, weight_op, n, c, h_w, group,
                                         output_mask, layout));

          return {{results.grad_input, results.grad_weight, results.grad_bias}};
        };

        // Construct Inputs Vector
        std::vector<at::Tensor> inputs = {grad_out, input, mean, rstd};
        if (weight.has_value() && weight->defined()) {
          inputs.push_back(weight.value());
        }

        TT_ASSIGN_OR_THROW(
            (auto [d_input_buf, d_weight_buf, d_bias_buf]),
            (DispatchOp<kDynamicSize, 3>(
                std::move(op_builder),
                /*inputs=*/inputs,
                {.out_dtypes = {input_dtype, weight_dtype, weight_dtype},
                 .out_dims_list = {d_input_dims, weight_grad_dims,
                                   bias_grad_dims},
                 .op_param_cache_keys = std::move(param_keys)})));

        // Return Results (handling masks and optional weight)
        return {
            output_mask[0] ? MakeTensor(std::move(d_input_buf)) : at::Tensor(),
            output_mask[1] ? MakeTensor(std::move(d_weight_buf)) : at::Tensor(),
            output_mask[2] ? MakeTensor(std::move(d_bias_buf)) : at::Tensor()};
      });
}
}  // namespace torch_tpu
