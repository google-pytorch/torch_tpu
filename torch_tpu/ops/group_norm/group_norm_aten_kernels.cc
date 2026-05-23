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

#include "torch_tpu/ops/group_norm/group_norm_aten_kernels.h"

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/zeros.h"
#include "ATen/ops/zeros_like.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

struct GroupNormBackwardShloResults {
  mlir::MlirOp grad_input;
  mlir::MlirOp grad_weight;
  mlir::MlirOp grad_bias;
};

namespace {

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
  auto elem_type = input_type.getElementType();

  auto zeros = MakeScalarConstant(builder, 0.0, elem_type);
  auto sum_reduce_builder = [elem_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        elem_type, rb.getRegion(), rb.getOpBuilder());
  };

  // 0. Reshape Inputs to [N, C, HxW]
  int64_t cpg = c / group;
  Dimensions n_c_hxw_shape = {n, c, h_w};
  auto grad_output_reshaped =
      mlir::stablehlo::Reshape(grad_output_op, n_c_hxw_shape);
  auto input_reshaped = mlir::stablehlo::Reshape(input_op, n_c_hxw_shape);

  // 1. Compute Common Terms. ds and db are [N, C]
  auto mul_in_grad = mlir::stablehlo::Mul(grad_output_reshaped, input_reshaped);
  auto ds = mlir::stablehlo::Reduce(builder, mul_in_grad, zeros,
                                    sum_reduce_builder, {2})[0];
  auto db = mlir::stablehlo::Reduce(builder, grad_output_reshaped, zeros,
                                    sum_reduce_builder, {2})[0];

  // Initialize Outputs with Default Zeros
  mlir::MlirOp d_input_op = MakeConstantLike(input_op, 0.0);
  mlir::MlirOp d_bias_op = MakeConstant(builder, 0.0, elem_type, {c});
  // If weight exists, Dispatcher expects a [C] tensor.
  // Else, Dispatcher expects a Scalar, but the shape is not important.
  mlir::MlirOp d_weight_op;
  if (weight_op.has_value()) {
    d_weight_op = MakeConstant(builder, 0.0, elem_type, {c});
  } else {
    d_weight_op = MakeScalarConstant(builder, 0.0, elem_type);
  }

  // 2. Compute d_input_op
  if (output_mask[0]) {
    mlir::MlirOp ds_val, db_val, c1;
    Dimensions n_g_cpg_shape = {n, group, cpg};
    auto type_ngc = mlir::RankedTensorType::get(n_g_cpg_shape, elem_type);

    if (weight_op.has_value()) {
      // weight: [C] -> Broadcast to [N, C] to multiply with ds/db
      auto weight_broadcast_nc = mlir::stablehlo::BroadcastInDim(
          GetTensorTypeOrDie(ds), weight_op.value(),
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
          mlir::stablehlo::BroadcastInDim(type_ngc, rstd_op,
                                          /*broadcast_dimensions=*/{0, 1});

      // weight: [C] -> Reshape [G, CPG] -> Broadcast to [N, G, CPG] (dims 1, 2)
      auto weight_reshaped_gc =
          mlir::stablehlo::Reshape(weight_op.value(), {group, cpg});
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
      c1 = mlir::stablehlo::BroadcastInDim(type_ngc, rstd_op,
                                           /*broadcast_dimensions=*/{0, 1});
    }

    auto s_val = 1.0 / (h_w * cpg);
    auto s = MakeScalarConstant(builder, s_val, elem_type);
    auto s_broadcast =
        mlir::stablehlo::BroadcastInDim(GetTensorTypeOrDie(ds_val), s,
                                        /*broadcast_dimensions=*/{});

    // All ops here are [N, G]
    // c2 = (db_val * mean - ds_val) * rstd * rstd * rstd * s
    auto db_val_mean = mlir::stablehlo::Mul(db_val, mean_op);
    auto mean_sub_ds = mlir::stablehlo::Subtract(db_val_mean, ds_val);
    auto rstd_pow2 = mlir::stablehlo::Mul(rstd_op, rstd_op);
    auto rstd_pow3 = mlir::stablehlo::Mul(rstd_pow2, rstd_op);
    auto c2_term = mlir::stablehlo::Mul(mean_sub_ds, rstd_pow3);
    auto c2 = mlir::stablehlo::Mul(c2_term, s_broadcast);

    // c3 = -c2 * mean - db_val * rstd * s
    auto neg_c2 = mlir::stablehlo::Neg(c2);
    auto neg_c2_mean = mlir::stablehlo::Mul(neg_c2, mean_op);
    auto db_val_rstd = mlir::stablehlo::Mul(db_val, rstd_op);
    auto db_val_rstd_s = mlir::stablehlo::Mul(db_val_rstd, s_broadcast);
    auto c3 = mlir::stablehlo::Subtract(neg_c2_mean, db_val_rstd_s);

    // d_input = grad_output * c1 + input * c2 + c3
    // Need to broadcast c1, c2, c3 to [N, G, CPG, HxW]
    Dimensions n_g_cpg_hxw_shape = {n, group, cpg, h_w};
    auto target_type =
        mlir::RankedTensorType::get(n_g_cpg_hxw_shape, elem_type);

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
        mlir::stablehlo::Reshape(grad_output_op, n_g_cpg_hxw_shape);
    auto input_reshaped_full =
        mlir::stablehlo::Reshape(input_op, n_g_cpg_hxw_shape);

    auto term1 = mlir::stablehlo::Mul(grad_reshaped_full, c1_final);
    auto term2 = mlir::stablehlo::Mul(input_reshaped_full, c2_final);

    auto dx_full = mlir::stablehlo::Add(term1, term2);
    dx_full = mlir::stablehlo::Add(dx_full, c3_final);

    // Reshape back to [N, C, HxW]
    d_input_op = mlir::stablehlo::Reshape(dx_full, input_type.getShape());
  }

  // 3. Compute d_weight_op
  if (output_mask[1] && weight_op.has_value()) {
    // All views are [N, G, CPG]
    // (ds.view - db.view * mean.unsqueeze) * rstd.unsqueeze
    Dimensions n_g_cpg_shape = {n, group, cpg};
    auto type_ngc = mlir::RankedTensorType::get(n_g_cpg_shape, elem_type);

    auto ds_view = mlir::stablehlo::Reshape(ds, n_g_cpg_shape);
    auto db_view = mlir::stablehlo::Reshape(db, n_g_cpg_shape);

    // mean/rstd [N, G] -> [N, G, CPG] (dims 0, 1)
    auto mean_bcast =
        mlir::stablehlo::BroadcastInDim(type_ngc, mean_op,
                                        /*broadcast_dimensions=*/{0, 1});
    auto rstd_bcast =
        mlir::stablehlo::BroadcastInDim(type_ngc, rstd_op,
                                        /*broadcast_dimensions=*/{0, 1});

    auto db_mean = mlir::stablehlo::Mul(db_view, mean_bcast);
    auto diff = mlir::stablehlo::Subtract(ds_view, db_mean);
    auto mul_rstd = mlir::stablehlo::Mul(diff, rstd_bcast);

    // sum(dim=0) -> [G, CPG]
    auto sum_term = mlir::stablehlo::Reduce(builder, mul_rstd, zeros,
                                            sum_reduce_builder, {0})[0];

    // Reshape [G, CPG] -> [C]
    d_weight_op = mlir::stablehlo::Reshape(sum_term, {c});
  }

  // 4. Compute d_bias_op
  if (output_mask[2]) {
    d_bias_op =
        mlir::stablehlo::Reduce(builder, db, zeros, sum_reduce_builder, {0})[0];
  }

  return GroupNormBackwardShloResults{.grad_input = d_input_op,
                                      .grad_weight = d_weight_op,
                                      .grad_bias = d_bias_op};
}
}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenNativeGroupNormBackward(
    const at::Tensor& grad_out, const at::Tensor& input, const at::Tensor& mean,
    const at::Tensor& rstd, const std::optional<at::Tensor>& weight, int64_t n,
    int64_t c, int64_t h_w, int64_t group, std::array<bool, 3> output_mask) {
  TT_KERNEL(
      OpName::kNativeGroupNormBackward, param_keys,
      (grad_out, input, mean, rstd, weight, n, c, h_w, group, output_mask), {
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
            << "grad_out and input must have the same dimensions"
            << ", got grad_out size " << grad_out.sizes() << ", input size "
            << input.sizes();
        TT_CHECK_THROW(mean.sizes() == at::IntArrayRef({n, group}),
                       error::kInvalidArgument)
            << "expected mean to have shape [" << n << ", " << group << "]"
            << ", got " << mean.sizes();
        TT_CHECK_THROW(mean.sizes() == rstd.sizes(), error::kInvalidArgument)
            << "expected mean and rstd to have the same shape"
            << ", got mean size " << mean.sizes() << " and rstd size "
            << rstd.sizes();

        // Early return for empty inputs to prevent division-by-zero
        if (c == 0 || h_w == 0 || c < group) {
          at::Tensor d_input, d_weight, d_bias;
          if (output_mask[0]) {
            d_input = at::zeros_like(input, input.options());
          }
          if (output_mask[1] && weight.has_value() && weight->defined()) {
            d_weight = at::zeros_like(*weight, weight->options());
          }
          if (output_mask[2]) {
            d_bias = at::zeros({c}, input.options());
          }
          return {d_input, d_weight, d_bias};
        }

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
        Dimensions weight_grad_dims = has_weight ? Dimensions{c} : Dimensions();
        Dimensions bias_grad_dims = {c};

        mlir::ElementType weight_dtype = input_dtype;
        if (has_weight) {
          TT_ASSIGN_OR_THROW(weight_dtype, ConvertTo<mlir::ElementType>(
                                               weight->scalar_type()));
        }

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
                {.out_dtypes = {input_dtype, weight_dtype, input_dtype},
                 .out_dims_list = {input_dims, weight_grad_dims,
                                   bias_grad_dims},
                 .op_param_cache_keys = std::move(param_keys)})));

        // Return Results (handling masks and optional weight)
        return {
            output_mask[0] ? MakeTensor(std::move(d_input_buf)) : at::Tensor(),
            (output_mask[1] && weight.has_value())
                ? MakeTensor(std::move(d_weight_buf))
                : at::Tensor(),
            output_mask[2] ? MakeTensor(std::move(d_bias_buf)) : at::Tensor()};
      });
}
}  // namespace torch_tpu
