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
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/device_type.h"
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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

struct GroupNormBackwardShloResults {
  mlir::MlirOp grad_input;
  mlir::MlirOp grad_weight;
  mlir::MlirOp grad_bias;
};

namespace {

struct GroupNormForwardShloResults {
  mlir::MlirOp output;
  mlir::MlirOp mean;
  mlir::MlirOp rstd;
};

// Math pseudo code, referencing pytorch decomposition:
// py/torch/_decomp/decompositions.py;l=1512
//
// Common parts:
//   cpg = C // group
//   ds = sum(grad_output * input, dim=2)  -> [N, C]
//   db = sum(grad_output, dim=2)          -> [N, C]
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
    int64_t c, int64_t h_w, int64_t group, std::array<bool, 3> output_mask) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  const mlir::ElementType input_elem_type = GetElementTypeOrDie(input_op);
  const mlir::ElementType weight_elem_type =
      weight_op.has_value() ? GetElementTypeOrDie(weight_op.value())
                            : GetElementTypeOrDie(mean_op);

  // Early return for empty channels or spatial dims to prevent division-by-zero
  // without decomposing into ATen ops (which triggers CompositeOpCheck).
  if (c == 0 || h_w == 0) {
    mlir::MlirOp zero_weight_scalar =
        MakeScalarConstant(builder, 0.0, weight_elem_type);
    mlir::MlirOp d_bias_op;
    if (c == 0) {
      TT_ASSIGN_OR_RETURN(d_bias_op,
                          MakeZeroSizedTensor(builder, weight_elem_type, {0}));
    } else {
      const mlir::RankedTensorType weight_tensor_type =
          mlir::makeTensorType(builder.getContext(), {c}, weight_elem_type);
      d_bias_op = mlir::stablehlo::BroadcastInDim(weight_tensor_type,
                                                  zero_weight_scalar,
                                                  /*broadcast_dimensions=*/{});
    }
    mlir::MlirOp d_weight_op = output_mask[1] ? d_bias_op : zero_weight_scalar;
    return GroupNormBackwardShloResults{
        .grad_input = input_op,
        .grad_weight = d_weight_op,
        .grad_bias = d_bias_op,
    };
  }

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

  const auto zeros = MakeScalarConstant(builder, 0.0, comp_type);
  const auto sum_reduce_builder = [comp_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        comp_type, rb.getRegion(), rb.getOpBuilder());
  };

  // 0. Reshape Inputs to [N, C, HxW]
  int64_t cpg = c / group;
  Dimensions n_c_hxw_shape = {n, c, h_w};
  auto grad_output_reshaped =
      mlir::stablehlo::Reshape(grad_output_comp, n_c_hxw_shape);
  auto input_reshaped = mlir::stablehlo::Reshape(input_comp, n_c_hxw_shape);

  // 1. Compute Common Terms. ds and db are [N, C]
  auto mul_in_grad = mlir::stablehlo::Mul(grad_output_reshaped, input_reshaped);
  auto ds = mlir::stablehlo::Reduce(builder, mul_in_grad, zeros,
                                    sum_reduce_builder, {2})[0];
  auto db = mlir::stablehlo::Reduce(builder, grad_output_reshaped, zeros,
                                    sum_reduce_builder, {2})[0];

  // Initialize Outputs with Default Zeros
  mlir::MlirOp d_input_op = MakeConstantLike(input_op, 0.0);
  mlir::MlirOp zero_weight_scalar =
      MakeScalarConstant(builder, 0.0, weight_elem_type);
  const mlir::RankedTensorType weight_tensor_type =
      mlir::makeTensorType(builder.getContext(), {c}, weight_elem_type);
  mlir::MlirOp d_bias_op = mlir::stablehlo::BroadcastInDim(
      weight_tensor_type, zero_weight_scalar, /*broadcast_dimensions=*/{});
  // If output_mask[1] is set, Dispatcher expects a [C] tensor.
  // Else, Dispatcher expects a Scalar, but the shape is not important.
  mlir::MlirOp d_weight_op = output_mask[1]
                                 ? mlir::stablehlo::BroadcastInDim(
                                       weight_tensor_type, zero_weight_scalar,
                                       /*broadcast_dimensions=*/{})
                                 : zero_weight_scalar;

  // 2. Compute d_input_op
  if (output_mask[0]) {
    mlir::MlirOp ds_val, db_val, c1;
    Dimensions n_g_cpg_shape = {n, group, cpg};
    auto type_ngc = mlir::RankedTensorType::get(n_g_cpg_shape, comp_type);

    if (weight_comp.has_value()) {
      // weight: [C] -> Broadcast to [N, C] to multiply with ds/db
      auto weight_broadcast_nc = mlir::stablehlo::BroadcastInDim(
          GetTensorTypeOrDie(ds), weight_comp.value(),
          /*broadcast_dimensions=*/{1});

      auto ds_weighted = mlir::stablehlo::Mul(ds, weight_broadcast_nc);
      auto db_weighted = mlir::stablehlo::Mul(db, weight_broadcast_nc);

      // Reshape weighted ds/db to [N, C] -> [N, G, CPG]
      auto ds_weighted_reshaped =
          mlir::stablehlo::Reshape(ds_weighted, n_g_cpg_shape);
      auto db_weighted_reshaped =
          mlir::stablehlo::Reshape(db_weighted, n_g_cpg_shape);

      // Reduce on dim 2 (CPG) -> [N, G]
      ds_val = mlir::stablehlo::Reduce(builder, ds_weighted_reshaped, zeros,
                                       sum_reduce_builder, {2})[0];
      db_val = mlir::stablehlo::Reduce(builder, db_weighted_reshaped, zeros,
                                       sum_reduce_builder, {2})[0];

      // rstd: [N, G] -> Broadcast to [N, G, CPG] (dims 0, 1)
      auto rstd_bcast_ngc =
          mlir::stablehlo::BroadcastInDim(type_ngc, rstd_comp,
                                          /*broadcast_dimensions=*/{0, 1});

      // weight: [C] -> Reshape [G, CPG] -> Broadcast to [N, G, CPG] (dims 1, 2)
      auto weight_reshaped_gc =
          mlir::stablehlo::Reshape(weight_comp.value(), {group, cpg});
      auto weight_bcast_ngc =
          mlir::stablehlo::BroadcastInDim(type_ngc, weight_reshaped_gc,
                                          /*broadcast_dimensions=*/{1, 2});

      // c1 = rstd.unsqueeze(-1) * weight.reshape(1, group, cpg)
      c1 = mlir::stablehlo::Mul(rstd_bcast_ngc, weight_bcast_ngc);
    } else {
      auto ds_reshaped = mlir::stablehlo::Reshape(ds, n_g_cpg_shape);
      auto db_reshaped = mlir::stablehlo::Reshape(db, n_g_cpg_shape);

      ds_val = mlir::stablehlo::Reduce(builder, ds_reshaped, zeros,
                                       sum_reduce_builder, {2})[0];
      db_val = mlir::stablehlo::Reduce(builder, db_reshaped, zeros,
                                       sum_reduce_builder, {2})[0];

      // c1 = rstd.unsqueeze(-1) * 1
      // Effectively just broadcasting rstd [N, G] -> [N, G, CPG]
      c1 = mlir::stablehlo::BroadcastInDim(type_ngc, rstd_comp,
                                           /*broadcast_dimensions=*/{0, 1});
    }

    auto s_val = 1.0 / (h_w * cpg);
    auto s = MakeScalarConstant(builder, s_val, comp_type);
    auto s_broadcast =
        mlir::stablehlo::BroadcastInDim(GetTensorTypeOrDie(ds_val), s,
                                        /*broadcast_dimensions=*/{});

    // All ops here are [N, G]
    // c2 = (db_val * mean - ds_val) * rstd * rstd * rstd * s
    auto db_val_mean = mlir::stablehlo::Mul(db_val, mean_comp);
    auto mean_sub_ds = mlir::stablehlo::Subtract(db_val_mean, ds_val);
    auto rstd_pow2 = mlir::stablehlo::Mul(rstd_comp, rstd_comp);
    auto rstd_pow3 = mlir::stablehlo::Mul(rstd_pow2, rstd_comp);
    auto c2_term = mlir::stablehlo::Mul(mean_sub_ds, rstd_pow3);
    auto c2 = mlir::stablehlo::Mul(c2_term, s_broadcast);

    // c3 = -c2 * mean - db_val * rstd * s
    auto neg_c2 = mlir::stablehlo::Neg(c2);
    auto neg_c2_mean = mlir::stablehlo::Mul(neg_c2, mean_comp);
    auto db_val_rstd = mlir::stablehlo::Mul(db_val, rstd_comp);
    auto db_val_rstd_s = mlir::stablehlo::Mul(db_val_rstd, s_broadcast);
    auto c3 = mlir::stablehlo::Subtract(neg_c2_mean, db_val_rstd_s);

    // d_input = grad_output * c1 + input * c2 + c3
    // Need to broadcast c1, c2, c3 to [N, G, CPG, HxW]
    Dimensions n_g_cpg_hxw_shape = {n, group, cpg, h_w};
    auto target_type =
        mlir::RankedTensorType::get(n_g_cpg_hxw_shape, comp_type);

    // c1 [N, G, CPG] -> [N, G, CPG, HxW] (dims 0, 1, 2)
    auto c1_final =
        mlir::stablehlo::BroadcastInDim(target_type, c1,
                                        /*broadcast_dimensions=*/{0, 1, 2});
    // c2 [N, G] -> [N, G, CPG, HxW] (dims 0, 1)
    auto c2_final =
        mlir::stablehlo::BroadcastInDim(target_type, c2,
                                        /*broadcast_dimensions=*/{0, 1});
    // c3 [N, G] -> [N, G, CPG, HxW] (dims 0, 1)
    auto c3_final =
        mlir::stablehlo::BroadcastInDim(target_type, c3,
                                        /*broadcast_dimensions=*/{0, 1});

    // Reshape grad_output/input to [N, G, CPG, HxW]
    auto grad_reshaped_full =
        mlir::stablehlo::Reshape(grad_output_comp, n_g_cpg_hxw_shape);
    auto input_reshaped_full =
        mlir::stablehlo::Reshape(input_comp, n_g_cpg_hxw_shape);

    auto term1 = mlir::stablehlo::Mul(grad_reshaped_full, c1_final);
    auto term2 = mlir::stablehlo::Mul(input_reshaped_full, c2_final);

    auto dx_full = mlir::stablehlo::Add(term1, term2);
    dx_full = mlir::stablehlo::Add(dx_full, c3_final);

    // Reshape back to [N, C, HxW]
    auto dx_reshaped = mlir::stablehlo::Reshape(dx_full, input_type.getShape());
    TT_ASSIGN_OR_RETURN(d_input_op, CastIfNeeded(dx_reshaped, input_elem_type));
  }

  // 3. Compute d_weight_op
  if (output_mask[1]) {
    // All views are [N, G, CPG]
    // (ds.view - db.view * mean.unsqueeze) * rstd.unsqueeze
    Dimensions n_g_cpg_shape = {n, group, cpg};
    auto type_ngc = mlir::RankedTensorType::get(n_g_cpg_shape, comp_type);

    auto ds_view = mlir::stablehlo::Reshape(ds, n_g_cpg_shape);
    auto db_view = mlir::stablehlo::Reshape(db, n_g_cpg_shape);

    // mean/rstd [N, G] -> [N, G, CPG] (dims 0, 1)
    auto mean_bcast =
        mlir::stablehlo::BroadcastInDim(type_ngc, mean_comp,
                                        /*broadcast_dimensions=*/{0, 1});
    auto rstd_bcast =
        mlir::stablehlo::BroadcastInDim(type_ngc, rstd_comp,
                                        /*broadcast_dimensions=*/{0, 1});

    auto db_mean = mlir::stablehlo::Mul(db_view, mean_bcast);
    auto diff = mlir::stablehlo::Subtract(ds_view, db_mean);
    auto mul_rstd = mlir::stablehlo::Mul(diff, rstd_bcast);

    // sum(dim=0) -> [G, CPG]
    const auto sum_term = mlir::stablehlo::Reduce(builder, mul_rstd, zeros,
                                                  sum_reduce_builder, {0})[0];

    // Reshape [G, CPG] -> [C]
    const auto d_weight_comp = mlir::stablehlo::Reshape(sum_term, {c});
    TT_ASSIGN_OR_RETURN(d_weight_op,
                        CastIfNeeded(d_weight_comp, weight_elem_type));
  }

  // 4. Compute d_bias_op
  if (output_mask[2]) {
    auto d_bias_comp =
        mlir::stablehlo::Reduce(builder, db, zeros, sum_reduce_builder, {0})[0];
    TT_ASSIGN_OR_RETURN(d_bias_op, CastIfNeeded(d_bias_comp, weight_elem_type));
  }

  return GroupNormBackwardShloResults{.grad_input = d_input_op,
                                      .grad_weight = d_weight_op,
                                      .grad_bias = d_bias_op};
}

absl::Status ValidateGroupNormForwardInputs(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, int64_t n, int64_t c, int64_t h_w,
    int64_t group) {
  TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(input));
  if (weight.has_value()) {
    TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(*weight));
  }
  if (bias.has_value()) {
    TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(*bias));
  }
  TT_RET_CHECK(n >= 0, error::kInvalidArgument)
      << "expected batch size to be >= 0, got " << n;
  if (n > 0) {
    TT_RET_CHECK(IsFloatingPoint(input), error::kInvalidArgument)
        << "expected input dtype to be floating point, got "
        << ToString(input.scalar_type());
  }
  TT_RET_CHECK(c > 0, error::kInvalidArgument)
      << "expected number of channels to be greater than 0, got " << c;
  TT_RET_CHECK(h_w > 0, error::kInvalidArgument)
      << "expected HxW to be greater than 0, got " << h_w;
  TT_RET_CHECK(group > 0, error::kInvalidArgument)
      << "expected num groups to be greater than 0, got " << group;
  TT_RET_CHECK(c % group == 0, error::kInvalidArgument)
      << "expected number of channels to be divisible by num_groups (" << group
      << "), got " << c;
  TT_ASSIGN_OR_RETURN(const int64_t nc, SafeMultiply(n, c));
  TT_ASSIGN_OR_RETURN(const int64_t expected_numel, SafeMultiply(nc, h_w));
  TT_RET_CHECK(input.numel() == expected_numel, error::kInvalidArgument)
      << "expected input to have " << expected_numel << " elements, got "
      << input.numel();
  if (weight.has_value()) {
    TT_RET_CHECK(weight->dim() == 1 && weight->numel() == c,
                 error::kInvalidArgument)
        << "expected weight to be a 1D tensor of size " << c << ", got "
        << ToString(weight->sizes());
  }
  if (bias.has_value()) {
    TT_RET_CHECK(bias->dim() == 1 && bias->numel() == c,
                 error::kInvalidArgument)
        << "expected bias to be a 1D tensor of size " << c << ", got "
        << ToString(bias->sizes());
  }
  const bool has_mixed =
      (weight.has_value() && weight->scalar_type() != input.scalar_type()) ||
      (bias.has_value() && bias->scalar_type() != input.scalar_type());
  if (has_mixed) {
    const bool is_half_or_bf16 =
        input.scalar_type() == at::ScalarType::Half ||
        input.scalar_type() == at::ScalarType::BFloat16;
    TT_RET_CHECK(is_half_or_bf16, error::kInvalidArgument)
        << "expected float16 or bfloat16 input for mixed-dtype group_norm, "
        << "got " << ToString(input.scalar_type());
    TT_RET_CHECK(
        !weight.has_value() || weight->scalar_type() == at::ScalarType::Float,
        error::kInvalidArgument)
        << "expected weight dtype to be float32 for mixed-dtype group_norm, "
        << "got " << ToString(weight->scalar_type());
    TT_RET_CHECK(
        !bias.has_value() || bias->scalar_type() == at::ScalarType::Float,
        error::kInvalidArgument)
        << "expected bias dtype to be float32 for mixed-dtype group_norm, "
        << "got " << ToString(bias->scalar_type());
  }
  return absl::OkStatus();
}

at::ScalarType ResolveGroupNormStatDtype(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias) {
  if (weight.has_value() && weight->scalar_type() != input.scalar_type()) {
    return weight->scalar_type();
  }
  if (bias.has_value() && bias->scalar_type() != input.scalar_type()) {
    return bias->scalar_type();
  }
  return input.scalar_type();
}

absl::StatusOr<GroupNormForwardShloResults> BuildGroupNormForwardShlo(
    mlir::MlirOp input_op, std::optional<mlir::MlirOp> weight_op,
    std::optional<mlir::MlirOp> bias_op, int64_t n, int64_t c, int64_t h_w,
    int64_t group, double eps, mlir::ElementType stat_elem_type) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  const mlir::ElementType input_elem_type = GetElementTypeOrDie(input_op);
  TT_RET_CHECK(input_type.hasStaticShape() &&
                   (!weight_op.has_value() ||
                    GetTensorTypeOrDie(*weight_op).hasStaticShape()) &&
                   (!bias_op.has_value() ||
                    GetTensorTypeOrDie(*bias_op).hasStaticShape()),
               error::kPythonNotImplementedError)
      << "bounded dynamic shapes are not yet supported";

  if (n == 0) {
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp empty_mean,
        MakeZeroSizedTensor(builder, stat_elem_type, {0, group}));
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp empty_rstd,
        MakeZeroSizedTensor(builder, stat_elem_type, {0, group}));
    return GroupNormForwardShloResults{
        .output = input_op, .mean = empty_mean, .rstd = empty_rstd};
  }

  TT_ASSIGN_OR_RETURN(const mlir::ElementType comp_elem_type,
                      InferComputationDtype(input_elem_type));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp input_comp,
                      CastIfNeeded(input_op, comp_elem_type));
  std::optional<mlir::MlirOp> weight_comp;
  if (weight_op.has_value()) {
    TT_ASSIGN_OR_RETURN(weight_comp,
                        CastIfNeeded(weight_op.value(), comp_elem_type));
  }
  std::optional<mlir::MlirOp> bias_comp;
  if (bias_op.has_value()) {
    TT_ASSIGN_OR_RETURN(bias_comp,
                        CastIfNeeded(bias_op.value(), comp_elem_type));
  }

  const mlir::Type comp_type = GetTensorTypeOrDie(input_comp).getElementType();
  const mlir::MlirOp zeros = MakeScalarConstant(builder, 0.0, comp_type);
  const auto sum_reduce_builder = [comp_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        comp_type, rb.getRegion(), rb.getOpBuilder());
  };

  TT_ASSIGN_OR_RETURN(const int64_t group_size, SafeMultiply(c / group, h_w));
  const Dimensions n_g_s_shape = {n, group, group_size};
  mlir::MlirOp input_grouped =
      mlir::stablehlo::Reshape(input_comp, n_g_s_shape);

  mlir::MlirOp sum_x = mlir::stablehlo::Reduce(builder, input_grouped, zeros,
                                               sum_reduce_builder, {2})[0];
  mlir::MlirOp inv_group_size =
      MakeConstantLike(sum_x, 1.0 / static_cast<double>(group_size));
  mlir::MlirOp mean_comp = mlir::stablehlo::Mul(sum_x, inv_group_size);

  const mlir::RankedTensorType grouped_type = GetTensorTypeOrDie(input_grouped);
  mlir::MlirOp mean_bcast = mlir::stablehlo::BroadcastInDim(
      grouped_type, mean_comp, /*broadcast_dimensions=*/{0, 1});
  mlir::MlirOp diff = mlir::stablehlo::Subtract(input_grouped, mean_bcast);
  mlir::MlirOp sq_diff = mlir::stablehlo::Mul(diff, diff);
  mlir::MlirOp sum_sq = mlir::stablehlo::Reduce(builder, sq_diff, zeros,
                                                sum_reduce_builder, {2})[0];
  mlir::MlirOp var_comp = mlir::stablehlo::Mul(sum_sq, inv_group_size);
  mlir::MlirOp eps_const = MakeConstantLike(var_comp, eps);
  mlir::MlirOp var_eps = mlir::stablehlo::Add(var_comp, eps_const);
  mlir::MlirOp rstd_comp = mlir::stablehlo::Rsqrt(var_eps);

  mlir::MlirOp rstd_bcast = mlir::stablehlo::BroadcastInDim(
      grouped_type, rstd_comp, /*broadcast_dimensions=*/{0, 1});
  mlir::MlirOp normalized = mlir::stablehlo::Mul(diff, rstd_bcast);

  const Dimensions n_c_hxw_shape = {n, c, h_w};
  mlir::MlirOp y_comp = mlir::stablehlo::Reshape(normalized, n_c_hxw_shape);
  const mlir::RankedTensorType nch_type = GetTensorTypeOrDie(y_comp);
  if (weight_comp.has_value()) {
    mlir::MlirOp weight_bcast = mlir::stablehlo::BroadcastInDim(
        nch_type, weight_comp.value(), /*broadcast_dimensions=*/{1});
    y_comp = mlir::stablehlo::Mul(y_comp, weight_bcast);
  }
  if (bias_comp.has_value()) {
    mlir::MlirOp bias_bcast = mlir::stablehlo::BroadcastInDim(
        nch_type, bias_comp.value(), /*broadcast_dimensions=*/{1});
    y_comp = mlir::stablehlo::Add(y_comp, bias_bcast);
  }

  mlir::MlirOp y_reshaped =
      mlir::stablehlo::Reshape(y_comp, input_type.getShape());
  TT_ASSIGN_OR_RETURN(mlir::MlirOp y_out,
                      CastIfNeeded(y_reshaped, input_elem_type));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mean_out,
                      CastIfNeeded(mean_comp, stat_elem_type));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp rstd_out,
                      CastIfNeeded(rstd_comp, stat_elem_type));
  return GroupNormForwardShloResults{
      .output = y_out, .mean = mean_out, .rstd = rstd_out};
}
struct GroupNormForwardDispatchParams {
  std::vector<at::Tensor> inputs;
  at::ScalarType stat_scalar_type;
  std::array<mlir::ElementType, 3> out_dtypes;
  std::array<Dimensions, 3> out_dims;
  NAryMlirOpBuilder<kDynamicSize, 3> op_builder;
};

absl::StatusOr<GroupNormForwardDispatchParams> PrepareGroupNormForwardDispatch(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, int64_t n, int64_t c, int64_t h_w,
    int64_t group, double eps) {
  const std::optional<at::Tensor> sanitized_weight =
      SanitizeOptionalTensor(weight);
  const std::optional<at::Tensor> sanitized_bias = SanitizeOptionalTensor(bias);
  TT_RETURN_IF_ERROR(ValidateGroupNormForwardInputs(
      input, sanitized_weight, sanitized_bias, n, c, h_w, group));

  const bool has_weight = sanitized_weight.has_value();
  const bool has_bias = sanitized_bias.has_value();
  const at::ScalarType stat_scalar_type =
      ResolveGroupNormStatDtype(input, sanitized_weight, sanitized_bias);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType input_dtype,
                      ConvertTo<mlir::ElementType>(input.scalar_type()));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType stat_dtype,
                      ConvertTo<mlir::ElementType>(stat_scalar_type));

  auto op_builder =
      [has_weight, has_bias, n, c, h_w, group, eps, stat_dtype](
          absl::Span<mlir::MlirOp> inputs,
          mlir::MlirBuilder& /*builder*/) -> absl::StatusOr<MlirOpResults<3>> {
    std::optional<mlir::MlirOp> weight_op;
    std::optional<mlir::MlirOp> bias_op;
    size_t idx = 1;
    if (has_weight) {
      weight_op = inputs[idx++];
    }
    if (has_bias) {
      bias_op = inputs[idx++];
    }
    TT_ASSIGN_OR_RETURN(
        GroupNormForwardShloResults results,
        BuildGroupNormForwardShlo(inputs[0], weight_op, bias_op, n, c, h_w,
                                  group, eps, stat_dtype));
    return {{results.output, results.mean, results.rstd}};
  };

  std::vector<at::Tensor> inputs = {input};
  if (has_weight) {
    inputs.push_back(*sanitized_weight);
  }
  if (has_bias) {
    inputs.push_back(*sanitized_bias);
  }

  const Dimensions stats_dims = {n, group};
  return GroupNormForwardDispatchParams{
      .inputs = std::move(inputs),
      .stat_scalar_type = stat_scalar_type,
      .out_dtypes = {input_dtype, stat_dtype, stat_dtype},
      .out_dims = {CopyIntVector(input.sizes()), stats_dims, stats_dims},
      .op_builder = std::move(op_builder),
  };
}
}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenNativeGroupNorm(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, int64_t n, int64_t c, int64_t h_w,
    int64_t group, double eps) {
  TT_KERNEL(
      OpName::kNativeGroupNorm, param_keys,
      (input, weight, bias, n, c, h_w, group, eps), {
        TT_ASSIGN_OR_THROW(GroupNormForwardDispatchParams params,
                           PrepareGroupNormForwardDispatch(
                               input, weight, bias, n, c, h_w, group, eps));
        const std::array<absl::Span<const int64_t>, 3> out_dims_list = {
            params.out_dims[0], params.out_dims[1], params.out_dims[2]};
        TT_ASSIGN_OR_THROW(
            (auto [y_buf, mean_buf, rstd_buf]),
            (DispatchOp<kDynamicSize, 3>(
                std::move(params.op_builder), params.inputs,
                {.out_dtypes = params.out_dtypes,
                 .out_dims_list = out_dims_list,
                 .op_param_cache_keys = std::move(param_keys)})));
        return {MakeTensor(std::move(y_buf)), MakeTensor(std::move(mean_buf)),
                MakeTensor(std::move(rstd_buf))};
      });
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> AtenNativeGroupNormOut(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, int64_t n, int64_t c, int64_t h_w,
    int64_t group, double eps, at::Tensor& out0, at::Tensor& out1,
    at::Tensor& out2) {
  TT_KERNEL(
      OpName::kNativeGroupNormOut, param_keys,
      (input, weight, bias, n, c, h_w, group, eps, out0, out1, out2), {
        TT_CHECK_THROW(IsPrivateUse1Device(out0) && IsPrivateUse1Device(out1) &&
                           IsPrivateUse1Device(out2),
                       error::kInvalidArgument)
            << "expected all output tensors to be on "
            << GetPrivateUse1DeviceDebugName() << ", got " << out0.device()
            << ", " << out1.device() << ", and " << out2.device();
        TT_ASSIGN_OR_THROW(GroupNormForwardDispatchParams params,
                           PrepareGroupNormForwardDispatch(
                               input, weight, bias, n, c, h_w, group, eps));
        TT_THROW_IF_ERROR(ValidateOutDtype(out0, input.scalar_type(),
                                           /*allow_out_dtype_cast=*/false));
        TT_THROW_IF_ERROR(ValidateOutDtype(out1, params.stat_scalar_type,
                                           /*allow_out_dtype_cast=*/false));
        TT_THROW_IF_ERROR(ValidateOutDtype(out2, params.stat_scalar_type,
                                           /*allow_out_dtype_cast=*/false));
        const std::array<absl::Span<const int64_t>, 3> out_dims_list = {
            params.out_dims[0], params.out_dims[1], params.out_dims[2]};
        OpOutputs<3> outputs = {out0, out1, out2};
        TT_THROW_IF_ERROR((DispatchOpOut<kDynamicSize, 3>(
            std::move(params.op_builder), params.inputs, outputs,
            {.out_dtypes = params.out_dtypes,
             .out_dims_list = out_dims_list,
             .op_param_cache_keys = std::move(param_keys)})));
        return std::tie(out0, out1, out2);
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenNativeGroupNormBackward(
    const at::Tensor& grad_out, const at::Tensor& input, const at::Tensor& mean,
    const at::Tensor& rstd, const std::optional<at::Tensor>& weight, int64_t n,
    int64_t c, int64_t h_w, int64_t group, std::array<bool, 3> output_mask) {
  TT_KERNEL(
      OpName::kNativeGroupNormBackward, param_keys,
      (grad_out, input, mean, rstd, weight, n, c, h_w, group, output_mask), {
        TT_THROW_IF_ERROR(ValidateNotLowPrecisionFloat(grad_out));
        TT_THROW_IF_ERROR(ValidateNotLowPrecisionFloat(input));
        TT_THROW_IF_ERROR(ValidateNotLowPrecisionFloat(mean));
        TT_THROW_IF_ERROR(ValidateNotLowPrecisionFloat(rstd));
        if (weight.has_value() && weight->defined()) {
          TT_THROW_IF_ERROR(ValidateNotLowPrecisionFloat(*weight));
        }

        // Input validations:
        // grad_out & input shape: (N, C, HxW)
        // mean & rstd shape: (N, group)
        // weight shape: (C) if provided else ()
        TT_CHECK_THROW(grad_out.numel() == n * c * h_w, error::kInvalidArgument)
            << "expected grad_out to have " << n * c * h_w << " elements"
            << ", got " << grad_out.numel();
        TT_CHECK_THROW(input.numel() == n * c * h_w, error::kInvalidArgument)
            << "expected input to have " << n * c * h_w << " elements"
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

        // If weight exists, grad_weight is [C]
        const bool has_weight = weight.has_value() && weight->defined();
        if (has_weight) {
          TT_CHECK_THROW(weight.value().numel() == c, error::kInvalidArgument)
              << "expected weight to have " << c << " elements"
              << ", got " << (weight ? weight->numel() : -1);
        }
        TT_CHECK_THROW(group > 0, error::kInvalidArgument)
            << "expected num groups to be greater than 0, got " << group;
        TT_CHECK_THROW(c % group == 0, error::kInvalidArgument)
            << "expected number of channels to be divisible by num_groups ("
            << group << "), got " << c;

        TT_ASSIGN_OR_THROW(auto input_dtype,
                           ConvertTo<mlir::ElementType>(input.scalar_type()));
        Dimensions input_dims = CopyIntVector(input.sizes());
        Dimensions weight_grad_dims =
            output_mask[1] ? Dimensions{c} : Dimensions();
        Dimensions bias_grad_dims = {c};

        TT_ASSIGN_OR_THROW(
            const mlir::ElementType weight_dtype,
            ConvertTo<mlir::ElementType>(has_weight ? weight->scalar_type()
                                                    : mean.scalar_type()));

        auto op_builder = [n, c, h_w, group, output_mask](
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

          TT_ASSIGN_OR_RETURN(GroupNormBackwardShloResults results,
                              BuildGroupNormBackwardShlo(
                                  grad_out_op, input_op, mean_op, rstd_op,
                                  weight_op, n, c, h_w, group, output_mask));

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
                 .out_dims_list = {input_dims, weight_grad_dims,
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
