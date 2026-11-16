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

#include "csrc/ops/layer_norm/layer_norm.h"

#include <cstddef>
#include <cstdint>
#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

struct TwoMoments {
  mlir::MlirOp mean;
  mlir::MlirOp mean_squared;
};

absl::StatusOr<TwoMoments> ComputeTwoMoments(
    const mlir::RankedTensorType& input_type, mlir::MlirOp input_casted,
    mlir::MlirOp sum_x, mlir::MlirOp sum_x_squared,
    const Dimensions& reduction_axes, mlir::Type stats_type) {
  bool has_dynamic_reduction_dim = false;
  int64_t num_elements = 1;
  const auto shape = input_type.getShape();
  for (const int64_t dim_idx : reduction_axes) {
    if (input_type.isDynamicDim(dim_idx)) {
      has_dynamic_reduction_dim = true;
      break;
    }
    TT_ASSIGN_OR_RETURN(num_elements,
                        SafeMultiply(num_elements, shape[dim_idx]));
  }

  if (!has_dynamic_reduction_dim) {
    mlir::MlirOp inv_num_elements =
        MakeConstantLike(sum_x, 1.0 / static_cast<double>(num_elements));
    return TwoMoments{
        .mean = mlir::stablehlo::Mul(sum_x, inv_num_elements),
        .mean_squared = mlir::stablehlo::Mul(sum_x_squared, inv_num_elements),
    };
  }

  mlir::MlirOp count_op;
  for (size_t i = 0; i < reduction_axes.size(); ++i) {
    const int64_t dim_idx = reduction_axes[i];
    mlir::MlirOp dim_size =
        mlir::stablehlo::GetDimensionSize(input_casted, dim_idx);
    dim_size = mlir::stablehlo::ConvertElementType(dim_size, stats_type);
    if (i == 0) {
      count_op = dim_size;
    } else {
      count_op = mlir::stablehlo::Mul(count_op, dim_size);
    }
  }
  count_op =
      mlir::stablehlo::BroadcastInDim(GetTensorTypeOrDie(sum_x), count_op, {});
  return TwoMoments{
      .mean = mlir::stablehlo::Div(sum_x, count_op),
      .mean_squared = mlir::stablehlo::Div(sum_x_squared, count_op),
  };
}

}  // namespace

// This layer implements the operation as described:
//
// y = ((x - E[x]) / sqrt(Var[x] + eps))* gamma + beta
//
// The mean and standard-deviation are calculated over the last D dimensions,
// where D is the dimension of the normalized shape. For example, if input shape
// is (10, 3, 5) and the normalized shape is (3, 5) (a 2-dimensional shape), the
// mean and standard-deviation are computed over the last 2 dimensions of the
// input.
//
// gamma (weight_op) and beta (bias_op) are learnable affine transform
// parameters of shape normalized_shape.

absl::StatusOr<LayerNormShloResults> BuildLayerNormShlo(
    mlir::MlirOp input_op, std::optional<mlir::MlirOp> weight_op,
    std::optional<mlir::MlirOp> bias_op, const int normalized_num_dims,
    const double eps) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  const int input_num_dims = input_type.getShape().size();
  Dimensions reduction_axes(normalized_num_dims);  // Dims reduced over
  absl::c_iota(reduction_axes, input_num_dims - normalized_num_dims);
  Dimensions unreduced_axes;  // Dims NOT reduced over
  if (input_num_dims > normalized_num_dims) {
    unreduced_axes.resize(input_num_dims - normalized_num_dims);
    absl::c_iota(unreduced_axes, 0);
  }

  // Compute mean and variance using the two-moment formulation:
  //   E[x] = sum(x) / N
  //   Var[x] = max(0, sum(x^2) / N - (E[x])^2)
  // XLA fusions combine the two sum reductions into a single loop over input.
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::Type element_type = input_type.getElementType();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=AtenNativeLayerNorm (caller) already
                 // runs this check.
      element_type.isFloat(), error::kInvalidArgument)
      << "expected the input dtype to be floating point, got "
      << ToString(element_type);

  mlir::MlirOp input_casted = input_op;
  // For bf16 and f16, use f32 for statistics computation.
  mlir::Type stats_type = element_type;
  if (!element_type.isF32() && !element_type.isF64()) {
    stats_type = builder.getOpBuilder().getF32Type();
    input_casted = mlir::stablehlo::ConvertElementType(input_op, stats_type);
  }

  const mlir::FloatType stats_float_type =
      mlir::cast<mlir::FloatType>(stats_type);

  const mlir::MlirOp x_squared =
      mlir::stablehlo::Mul(input_casted, input_casted);
  const mlir::MlirOp zero = MakeScalarConstant(builder, 0.0, stats_float_type);
  const mlir::RankedTensorType scalar_stats_type =
      mlir::RankedTensorType::get({}, stats_float_type);
  const auto moments_reduce_builder =
      [scalar_stats_type](mlir::RegionBuilder& rb) {
        mlir::MlirOp acc_x = mlir::Argument(rb, scalar_stats_type);
        mlir::MlirOp acc_x2 = mlir::Argument(rb, scalar_stats_type);
        mlir::MlirOp val_x = mlir::Argument(rb, scalar_stats_type);
        mlir::MlirOp val_x2 = mlir::Argument(rb, scalar_stats_type);

        mlir::MlirOp new_acc_x = mlir::stablehlo::Add(acc_x, val_x);
        mlir::MlirOp new_acc_x2 = mlir::stablehlo::Add(acc_x2, val_x2);

        mlir::stablehlo::Return(rb, {new_acc_x, new_acc_x2});
      };

  const auto results =
      mlir::stablehlo::Reduce(builder, {input_casted, x_squared}, {zero, zero},
                              moments_reduce_builder, reduction_axes);
  const mlir::MlirOp sum_x = results[0];
  const mlir::MlirOp sum_x_squared = results[1];

  TT_ASSIGN_OR_RETURN(
      const auto moments,
      ComputeTwoMoments(input_type, input_casted, sum_x, sum_x_squared,
                        reduction_axes, stats_type));
  mlir::MlirOp mean = moments.mean;
  mlir::MlirOp mean_squared = moments.mean_squared;

  mlir::MlirOp mean_sq = mlir::stablehlo::Mul(mean, mean);
  mlir::MlirOp variance = mlir::stablehlo::Subtract(mean_squared, mean_sq);

  // Clamp variance to 0 to prevent negative variance due to floating-point
  // rounding.
  mlir::MlirOp zero_var = MakeConstantLike(variance, 0.0);
  variance = mlir::stablehlo::Max(variance, zero_var);

  mlir::MlirOp eps_op = MakeConstantLike(variance, eps);
  mlir::MlirOp variance_plus_eps = mlir::stablehlo::Add(variance, eps_op);
  mlir::MlirOp rstd = mlir::stablehlo::Rsqrt(variance_plus_eps);

  // Broadcast with `BroadcastIfNeeded`, which keeps bounded dynamic dimensions
  // of the input bounded.
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mean_broadcasted,
                      BroadcastIfNeeded(mean, input_casted, unreduced_axes));
  mlir::MlirOp input_minus_mean =
      mlir::stablehlo::Subtract(input_casted, mean_broadcasted);

  TT_ASSIGN_OR_RETURN(mlir::MlirOp rstd_broadcasted,
                      BroadcastIfNeeded(rstd, input_casted, unreduced_axes));

  // Compute normalized input by multiplying (input - mean) by rstd.
  mlir::MlirOp normalized_input =
      mlir::stablehlo::Mul(input_minus_mean, rstd_broadcasted);

  // Compute affine output by multiplying normalized input by gamma and adding
  // beta.
  mlir::MlirOp normalized_input_casted = normalized_input;
  if (stats_type != element_type) {
    normalized_input_casted =
        mlir::stablehlo::ConvertElementType(normalized_input, element_type);
  }

  mlir::MlirOp normalized_input_times_weight = normalized_input_casted;
  if (weight_op.has_value()) {
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp weight_broadcasted,
        BroadcastIfNeeded(*weight_op, input_op, reduction_axes));
    normalized_input_times_weight =
        mlir::stablehlo::Mul(normalized_input_casted, weight_broadcasted);
  }

  mlir::MlirOp affine_output = normalized_input_times_weight;
  if (bias_op.has_value()) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp bias_broadcasted,
                        BroadcastIfNeeded(*bias_op, input_op, reduction_axes));

    affine_output =
        mlir::stablehlo::Add(normalized_input_times_weight, bias_broadcasted);
  }

  // mean and rstd are returned without trailing size-1 dimensions.
  // Future calls to GetBuffer will use the appropriate view logic
  // to reshape them as desired for downstream use.
  return LayerNormShloResults{
      .normalized_values = affine_output, .mean = mean, .reciprocal_std = rstd};
}

absl::StatusOr<LayerNormBackwardShloResults> BuildLayerNormBackwardShlo(
    mlir::MlirOp dy, mlir::MlirOp x, mlir::MlirOp mean, mlir::MlirOp rstd,
    std::optional<mlir::MlirOp> weight, at::IntArrayRef normalized_shape,
    bool compute_dbeta) {
  mlir::MlirBuilder& builder = x.getBuilder();
  const mlir::RankedTensorType x_type = GetTensorTypeOrDie(x);
  const mlir::Type orig_elem_type = x_type.getElementType();
  bool needs_upcast = orig_elem_type.isF16() || orig_elem_type.isBF16();

  if (needs_upcast) {
    TT_ASSIGN_OR_RETURN(x, PromoteFloatDtype(x));
    TT_ASSIGN_OR_RETURN(dy, PromoteFloatDtype(dy));
    TT_ASSIGN_OR_RETURN(mean, PromoteFloatDtype(mean));
    TT_ASSIGN_OR_RETURN(rstd, PromoteFloatDtype(rstd));
  }
  mlir::Type acc_elem_type = GetTensorTypeOrDie(x).getElementType();
  auto acc_float_type = mlir::cast<mlir::FloatType>(acc_elem_type);

  mlir::MlirOp zeros = MakeScalarConstant(builder, 0.0, acc_float_type);
  auto sum_reduce_builder = [acc_float_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        acc_float_type, rb.getRegion(), rb.getOpBuilder());
  };

  // math pseudo code, referencing torch cuda kernel:
  // py/torch/aten/src/ATen/native/cuda/layer_norm_kernel.cu
  //
  // N = normalized_dim_numl
  // x_hat = (X - mean) * rstd
  // dy_gamma = dY * gamma (or dY if gamma is not present)
  //
  // dGamma = sum_batch(dY * x_hat)
  // dBeta = sum_batch(dY)
  //
  // a1 = sum_norm(dy_gamma)
  // a2 = sum_norm(dy_gamma * x_hat)
  // dX = rstd * (dy_gamma - a1 / N - x_hat * (a2 / N))

  // Shape example:
  // All code comments below will use this shape for explanation:
  // X is shape [20, 5, 10, 10]
  // dy will be shape [20, 5, 10, 10]
  // mean is shape [20, 1, 1, 1]
  // rstd is shape [20, 1, 1, 1]
  // gamma is shape [5, 10, 10]
  // bias is shape [5, 10, 10]

  int64_t x_rank = GetTensorTypeOrDie(x).getShape().size();
  int64_t norm_len = normalized_shape.size();
  int64_t batch_len = x_rank - norm_len;

  Dimensions batch_dims;
  Dimensions norm_dims;
  Dimensions all_dims;
  for (int i = 0; i < x_rank; ++i) {
    all_dims.push_back(i);
    if (i < batch_len)
      batch_dims.push_back(i);
    else
      norm_dims.push_back(i);
  }

  // Broadcast mean/rstd to x shape. As in the forward pass, `BroadcastIfNeeded`
  // keeps bounded dynamic dimensions of x bounded.
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mean_bcast,
                      BroadcastIfNeeded(mean, x, all_dims));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp rstd_bcast,
                      BroadcastIfNeeded(rstd, x, all_dims));

  // x_hat = (x - mean) * rstd
  mlir::MlirOp x_hat = mlir::stablehlo::Subtract(x, mean_bcast);
  x_hat = mlir::stablehlo::Mul(x_hat, rstd_bcast);

  // Optionally broadcast and upcast weight.
  std::optional<mlir::MlirOp> gamma_bcast;
  if (weight.has_value()) {
    mlir::MlirOp w = weight.value();
    if (needs_upcast) {
      TT_ASSIGN_OR_RETURN(w, PromoteFloatDtype(w));
    }
    TT_ASSIGN_OR_RETURN(gamma_bcast, BroadcastIfNeeded(w, x, norm_dims));
  }

  // dgamma = sum_batch(dy * x_hat)
  // Reduce over batch dimensions {0} to get [5, 10, 10]
  mlir::MlirOp dgamma;
  if (weight.has_value()) {
    mlir::MlirOp dgamma_full = mlir::stablehlo::Mul(dy, x_hat);
    dgamma = mlir::stablehlo::Reduce(builder, dgamma_full, zeros,
                                     sum_reduce_builder, batch_dims)[0];
  } else {
    dgamma = zeros;
  }

  // dbeta = sum_batch(dy)
  // Reduce over batch dimensions {0} to get [5, 10, 10]
  mlir::MlirOp dbeta;
  if (compute_dbeta) {
    dbeta = mlir::stablehlo::Reduce(builder, dy, zeros, sum_reduce_builder,
                                    batch_dims)[0];
  } else {
    dbeta = zeros;
  }

  // dy_gamma = dy * gamma (or just dy when no weight)
  mlir::MlirOp dy_gamma = dy;
  if (gamma_bcast.has_value()) {
    dy_gamma = mlir::stablehlo::Mul(dy, gamma_bcast.value());
  }

  // a1 = sum_norm(dy * gamma)
  // Reduce over norm_dims to get reduced shape [batch_dims]
  mlir::MlirOp a1 = mlir::stablehlo::Reduce(builder, dy_gamma, zeros,
                                            sum_reduce_builder, norm_dims)[0];

  // a2 = sum_norm(dy * gamma * x_hat)
  // Reduce over norm_dims to get reduced shape [batch_dims]
  mlir::MlirOp a2_full = mlir::stablehlo::Mul(dy_gamma, x_hat);
  mlir::MlirOp a2 = mlir::stablehlo::Reduce(builder, a2_full, zeros,
                                            sum_reduce_builder, norm_dims)[0];

  // dx = rstd * (dy_gamma - a1/N - x_hat * a2/N)
  // Pre-multiply (1/N) into a1 and a2 on the small reduced shape before
  // broadcasting, eliminating full-tensor n_bcast allocation, full-tensor
  // elementwise Mul(n_bcast, dy_gamma), and full-tensor inv_n multiplication.
  TT_ASSIGN_OR_RETURN(const int64_t n, NumElements(normalized_shape));
  const double inv_n_val = n > 0 ? 1.0 / static_cast<double>(n) : 0.0;
  mlir::MlirOp inv_n = MakeConstantLike(a1, inv_n_val);
  mlir::MlirOp a1_scaled = mlir::stablehlo::Mul(a1, inv_n);
  mlir::MlirOp a2_scaled = mlir::stablehlo::Mul(a2, inv_n);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp a1_bcast,
                      BroadcastIfNeeded(a1_scaled, x, batch_dims));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp a2_bcast,
                      BroadcastIfNeeded(a2_scaled, x, batch_dims));

  mlir::MlirOp dx = mlir::stablehlo::Subtract(dy_gamma, a1_bcast);
  mlir::MlirOp x_hat_a2 = mlir::stablehlo::Mul(x_hat, a2_bcast);
  dx = mlir::stablehlo::Subtract(dx, x_hat_a2);
  dx = mlir::stablehlo::Mul(dx, rstd_bcast);

  if (needs_upcast) {
    dx = mlir::stablehlo::ConvertElementType(dx, orig_elem_type);
    dgamma = mlir::stablehlo::ConvertElementType(dgamma, orig_elem_type);
    dbeta = mlir::stablehlo::ConvertElementType(dbeta, orig_elem_type);
  }

  return LayerNormBackwardShloResults{
      .grad_input = dx, .grad_weight = dgamma, .grad_bias = dbeta};
};

}  // namespace torch_tpu
