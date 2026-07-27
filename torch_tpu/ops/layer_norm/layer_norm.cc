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

#include "torch_tpu/ops/layer_norm/layer_norm.h"

#include <cstdint>
#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"

namespace torch_tpu {

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

namespace {

void BuildWelfordReductionBody(mlir::FloatType element_type,
                               mlir::RegionBuilder& builder) {
  mlir::RankedTensorType scalar_type =
      mlir::RankedTensorType::get({}, element_type);

  // 3 args for lhs (accumulators), 3 args for rhs (inputs)
  // Layout: lhs_count, lhs_mean, lhs_m2, rhs_count, rhs_mean, rhs_m2
  mlir::MlirOp n_a = mlir::Argument(builder, scalar_type);
  mlir::MlirOp mu_a = mlir::Argument(builder, scalar_type);
  mlir::MlirOp m2_a = mlir::Argument(builder, scalar_type);
  mlir::MlirOp n_b = mlir::Argument(builder, scalar_type);
  mlir::MlirOp mu_b = mlir::Argument(builder, scalar_type);
  mlir::MlirOp m2_b = mlir::Argument(builder, scalar_type);

  // n = n_a + n_b
  mlir::MlirOp n = mlir::stablehlo::Add(n_a, n_b);

  // delta = mu_b - mu_a
  mlir::MlirOp delta = mlir::stablehlo::Subtract(mu_b, mu_a);

  // delta_sq = delta * delta
  mlir::MlirOp delta_sq = mlir::stablehlo::Mul(delta, delta);

  // n_a * n_b
  mlir::MlirOp na_nb = mlir::stablehlo::Mul(n_a, n_b);

  // Guard against n=0 to avoid NaN from 0/0 division.
  // We need to create constants 0 and 1 with the correct floating-point
  // semantics.
  const llvm::fltSemantics& semantics = element_type.getFloatSemantics();
  mlir::MlirOp zero = MakeConstantLike(n, llvm::APFloat(semantics, "0"));
  mlir::MlirOp one = MakeConstantLike(n, llvm::APFloat(semantics, "1"));

  mlir::MlirOp n_is_zero = mlir::stablehlo::Compare(
      n, zero, mlir::stablehlo::ComparisonDirection::EQ);
  mlir::MlirOp safe_n = mlir::stablehlo::Select(n_is_zero, one, n);

  // term = delta^2 * (n_a * n_b) / n
  mlir::MlirOp term_num = mlir::stablehlo::Mul(delta_sq, na_nb);
  mlir::MlirOp term = mlir::stablehlo::Div(term_num, safe_n);

  // M2 = M2_a + M2_b + term
  mlir::MlirOp m2_sum = mlir::stablehlo::Add(m2_a, m2_b);
  mlir::MlirOp m2 = mlir::stablehlo::Add(m2_sum, term);

  // mu = mu_a + delta * (n_b / n)
  mlir::MlirOp nb_div_n = mlir::stablehlo::Div(n_b, safe_n);
  mlir::MlirOp delta_scaled = mlir::stablehlo::Mul(delta, nb_div_n);
  mlir::MlirOp mu = mlir::stablehlo::Add(mu_a, delta_scaled);

  // If n=0, we want to return the identity (0, 0, 0).
  // The calculations above might produce non-zero or garbage if n=0 but we
  // masked the div. However, if n=0, then n_a=0 and n_b=0. term_num = 0
  // (since na_nb=0). term = 0. nb_div_n = 0 / 1 = 0. delta_scaled = 0. m2 = 0
  // + 0 + 0 = 0. mu = 0 + 0 = 0. So the result is correctly (0, 0, 0) with
  // safe_n.
  mlir::stablehlo::Return(builder, {n, mu, m2});
}

}  // namespace

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

  // Compute mean and variance by summing over the reduction axes.
  // We compute sum(x) and sum(x^2) in a single reduction to avoid reading the
  // input twice.
  mlir::MlirBuilder& builder = input_op.getBuilder();
  mlir::Type element_type = input_type.getElementType();
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

  mlir::FloatType stats_float_type = mlir::cast<mlir::FloatType>(stats_type);
  auto stats_tensor_type =
      mlir::RankedTensorType::get(input_type.getShape(), stats_type);

  // Welford's algorithm inputs:
  // 1. Counts: All 1s (shape of input)
  // 2. Means: The input itself
  // 3. M2s: All 0s (shape of input)
  mlir::MlirOp one = MakeScalarConstant(builder, 1.0, stats_float_type);
  mlir::MlirOp zero = MakeScalarConstant(builder, 0.0, stats_float_type);

  // We prepare matching input tensors for Welford's algorithm:
  // 'counts' (all 1s) and 'm2s' (all 0s) are broadcasted to the full input
  // shape (stats_tensor_type) to correspond with 'input_casted'.
  mlir::MlirOp counts =
      mlir::stablehlo::BroadcastInDim(stats_tensor_type, one, {});
  mlir::MlirOp m2s =
      mlir::stablehlo::BroadcastInDim(stats_tensor_type, zero, {});

  // Initial values for reduction (identity):
  // Count = 0, Mean = 0, M2 = 0
  mlir::MlirOp init_count = MakeScalarConstant(builder, 0.0, stats_float_type);
  mlir::MlirOp init_mean = MakeScalarConstant(builder, 0.0, stats_float_type);
  mlir::MlirOp init_m2 = MakeScalarConstant(builder, 0.0, stats_float_type);

  auto reduce_builder = [stats_float_type](mlir::RegionBuilder& rb) {
    BuildWelfordReductionBody(stats_float_type, rb);
  };

  auto results = mlir::stablehlo::Reduce(builder, {counts, input_casted, m2s},
                                         {init_count, init_mean, init_m2},
                                         reduce_builder, reduction_axes);

  mlir::MlirOp total_count = results[0];
  mlir::MlirOp mean = results[1];
  mlir::MlirOp total_m2 = results[2];

  // Variance = M2 / n
  mlir::MlirOp variance = mlir::stablehlo::Div(total_m2, total_count);

  mlir::MlirOp mean_broadcasted =
      mlir::stablehlo::BroadcastInDim(stats_tensor_type, mean, unreduced_axes);
  mlir::MlirOp input_minus_mean =
      mlir::stablehlo::Subtract(input_casted, mean_broadcasted);

  // Compute reciprocal standard deviation by taking the reciprocal square root
  // of the variance + eps.

  mlir::MlirOp eps_op = MakeConstantLike(variance, eps);
  mlir::MlirOp variance_plus_eps = mlir::stablehlo::Add(variance, eps_op);
  mlir::MlirOp rstd = mlir::stablehlo::Rsqrt(variance_plus_eps);
  mlir::MlirOp rstd_broadcasted =
      mlir::stablehlo::BroadcastInDim(stats_tensor_type, rstd, unreduced_axes);

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
    mlir::MlirOp weight_broadcasted =
        mlir::stablehlo::BroadcastInDim(input_type, *weight_op, reduction_axes);
    normalized_input_times_weight =
        mlir::stablehlo::Mul(normalized_input_casted, weight_broadcasted);
  }

  mlir::MlirOp affine_output = normalized_input_times_weight;
  if (bias_op.has_value()) {
    mlir::MlirOp bias_broadcasted =
        mlir::stablehlo::BroadcastInDim(input_type, *bias_op, reduction_axes);

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
  // dX = (1 / N) * rstd * (N * dy_gamma - a1 - x_hat * a2)

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

  // Broadcast mean/rstd to x shape.
  mlir::MlirOp mean_bcast =
      mlir::stablehlo::BroadcastInDim(x.getType(), mean, all_dims);
  mlir::MlirOp rstd_bcast =
      mlir::stablehlo::BroadcastInDim(x.getType(), rstd, all_dims);

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
    gamma_bcast = mlir::stablehlo::BroadcastInDim(x.getType(), w, norm_dims);
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
  // Reduce over norm_dims {1, 2, 3} to get [20], then broadcast to [20, 5, 10,
  // 10]
  mlir::MlirOp a1 = mlir::stablehlo::Reduce(builder, dy_gamma, zeros,
                                            sum_reduce_builder, norm_dims)[0];
  a1 = mlir::stablehlo::BroadcastInDim(x.getType(), a1, batch_dims);

  // a2 = sum_norm(dy * gamma * x_hat)
  // Reduce over norm_dims {1, 2, 3} to get [20], then broadcast to [20, 5, 10,
  // 10]
  mlir::MlirOp a2_full = mlir::stablehlo::Mul(dy_gamma, x_hat);
  mlir::MlirOp a2 = mlir::stablehlo::Reduce(builder, a2_full, zeros,
                                            sum_reduce_builder, norm_dims)[0];
  a2 = mlir::stablehlo::BroadcastInDim(x.getType(), a2, batch_dims);

  // dx = (1/N) * rstd * (N * dy_gamma - a1 - x_hat * a2)
  TT_ASSIGN_OR_RETURN(const int64_t n, NumElements(normalized_shape));
  mlir::MlirOp n_const =
      MakeScalarConstant(builder, static_cast<double>(n), acc_float_type);
  mlir::MlirOp n_bcast =
      mlir::stablehlo::BroadcastInDim(x.getType(), n_const, {});
  mlir::MlirOp inv_n = MakeScalarConstant(builder, 1.0 / n, acc_float_type);
  mlir::MlirOp inv_n_bcast =
      mlir::stablehlo::BroadcastInDim(x.getType(), inv_n, {});

  mlir::MlirOp dx = mlir::stablehlo::Mul(n_bcast, dy_gamma);
  dx = mlir::stablehlo::Subtract(dx, a1);
  mlir::MlirOp x_hat_a2 = mlir::stablehlo::Mul(x_hat, a2);
  dx = mlir::stablehlo::Subtract(dx, x_hat_a2);
  mlir::MlirOp inv_n_rstd = mlir::stablehlo::Mul(inv_n_bcast, rstd_bcast);
  dx = mlir::stablehlo::Mul(dx, inv_n_rstd);

  if (needs_upcast) {
    dx = mlir::stablehlo::ConvertElementType(dx, orig_elem_type);
    dgamma = mlir::stablehlo::ConvertElementType(dgamma, orig_elem_type);
    dbeta = mlir::stablehlo::ConvertElementType(dbeta, orig_elem_type);
  }

  return LayerNormBackwardShloResults{
      .grad_input = dx, .grad_weight = dgamma, .grad_bias = dbeta};
};

}  // namespace torch_tpu
