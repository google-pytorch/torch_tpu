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

#include "csrc/ops/rms_norm/rms_norm.h"

#include <array>
#include <cstdint>
#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/layer_norm/layer_norm.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/reductions/reductions.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

// RMSNorm(x) = x * w / sqrt(mean(x^2) + eps)
// It is similar to LayerNorm but without mean subtraction.
absl::StatusOr<LayerNormShloResults> BuildRmsNormShlo(
    mlir::MlirOp input_op, std::optional<mlir::MlirOp> weight_op,
    const int normalized_num_dims, const double eps) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  const int input_num_dims = input_type.getShape().size();

  Dimensions reduction_axes(normalized_num_dims);  // Dims reduced over
  absl::c_iota(reduction_axes, input_num_dims - normalized_num_dims);

  Dimensions unreduced_axes;  // Dims NOT reduced over
  if (input_num_dims > normalized_num_dims) {
    unreduced_axes.resize(input_num_dims - normalized_num_dims);
    absl::c_iota(unreduced_axes, 0);
  }

  mlir::MlirBuilder& builder = input_op.getBuilder();
  mlir::Type element_type = input_type.getElementType();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Error caught by unique caller:
                 // AtenFusedRmsNorm.
      element_type.isFloat(), error::kInvalidArgument)
      << "expected the input dtype to be floating point, got "
      << ToString(element_type);

  // Perform computation in float32 to avoid overflow/underflow for f16/bf16.
  const bool need_cast = element_type.getIntOrFloatBitWidth() < 32;
  mlir::MlirOp compute_input = input_op;
  mlir::Type compute_type = element_type;
  if (need_cast) {
    compute_type = builder.getOpBuilder().getF32Type();
    compute_input = mlir::stablehlo::ConvertElementType(input_op, compute_type);
  }

  // Compute Mean(x^2)
  mlir::MlirOp x_squared = mlir::stablehlo::Mul(compute_input, compute_input);

  mlir::MlirOp zero = MakeScalarConstant(builder, 0.0, compute_type);
  auto sum_reduce_builder = [compute_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        compute_type, rb.getRegion(), rb.getOpBuilder());
  };

  // Mathematical Formulation of RMSNorm Forward Pass:
  // ---------------------------------------------------------------------------
  // For input tensor x with shape [..., N], where N is the product of reduction
  // axes:
  //   1. Sum of squares:
  //        SS = \sum_{i=1}^N x_i^2
  //   2. Mean square (variance without centering):
  //        MS = (1 / N) * SS
  //      Optimization: Computed via constant multiplication `Mul(SS, 1.0 / N)`
  //      rather than vector division `Div(SS, N)` to reduce MXU latency on TPU.
  //   3. Reciprocal root mean square:
  //        rstd = 1 / \sqrt{MS + \epsilon} = \text{rsqrt}(MS + \epsilon)
  //   4. Normalized output:
  //        y = (x * rstd) * \gamma
  // ---------------------------------------------------------------------------
  mlir::MlirOp sum_squared_elements = mlir::stablehlo::Reduce(
      builder, x_squared, zero, sum_reduce_builder, reduction_axes)[0];

  int64_t num_elements = 1;
  const auto shape = input_type.getShape();
  for (const int64_t dim_idx : reduction_axes) {
    TT_ASSIGN_OR_RETURN(num_elements,
                        SafeMultiply(num_elements, shape[dim_idx]));
  }
  mlir::MlirOp inv_num_elements = MakeConstantLike(
      sum_squared_elements, 1.0 / static_cast<double>(num_elements));
  mlir::MlirOp mean_squared_elements =
      mlir::stablehlo::Mul(sum_squared_elements, inv_num_elements);

  // rstd = 1 / sqrt(Mean(x^2) + eps)
  mlir::MlirOp epsilon_constant = MakeConstantLike(sum_squared_elements, eps);
  mlir::MlirOp variance_plus_epsilon =
      mlir::stablehlo::Add(mean_squared_elements, epsilon_constant);
  mlir::MlirOp rstd = mlir::stablehlo::Rsqrt(variance_plus_epsilon);

  const mlir::RankedTensorType compute_input_type =
      GetTensorTypeOrDie(compute_input);
  mlir::MlirOp rstd_broadcasted =
      mlir::stablehlo::BroadcastInDim(compute_input_type, rstd, unreduced_axes);

  // normalized = compute_input * rstd (in compute_type / F32)
  mlir::MlirOp normalized_input =
      mlir::stablehlo::Mul(compute_input, rstd_broadcasted);

  // output = normalized * weight
  mlir::MlirOp output = normalized_input;
  if (weight_op.has_value()) {
    mlir::MlirOp weight_compute = *weight_op;
    if (GetTensorTypeOrDie(weight_compute).getElementType() != compute_type) {
      weight_compute =
          mlir::stablehlo::ConvertElementType(weight_compute, compute_type);
    }
    mlir::MlirOp weight_broadcasted = mlir::stablehlo::BroadcastInDim(
        compute_input_type, weight_compute, reduction_axes);
    output = mlir::stablehlo::Mul(normalized_input, weight_broadcasted);
  }

  if (need_cast) {
    output = mlir::stablehlo::ConvertElementType(output, element_type);
  }

  const auto rstd_unsqueezed =
      BuildKeepDimsShlo(compute_input, rstd, reduction_axes);

  // Helper struct reuse: .mean is effectively unused/zero for RMSNorm.
  // .reciprocal_std is preserved in compute_type (F32) for backward pass
  // compatibility.
  mlir::MlirOp final_zero = MakeScalarConstant(builder, 0.0, element_type);
  return LayerNormShloResults{.normalized_values = output,
                              .mean = final_zero,
                              .reciprocal_std = rstd_unsqueezed};
}

namespace {

// Builds the StableHLO operations for the gradient with respect to weight
// (\nabla_\gamma L).
// Mathematically: \nabla_\gamma L = \sum_{batch} ( \nabla_y L \odot \hat{x} ),
// where \hat{x} = x \odot rstd.
// If affine weight is omitted, returns a tensor of zeros with shape
// normalized_shape. Otherwise, reduces across all batch dimensions in
// compute_float_type (F32) and casts to orig_weight_type if needed.
absl::StatusOr<mlir::MlirOp> BuildRmsNormDgamma(
    mlir::MlirBuilder& builder, mlir::MlirOp dy_f32, mlir::MlirOp x_hat,
    std::optional<mlir::MlirOp> weight, at::IntArrayRef normalized_shape,
    mlir::FloatType compute_float_type, mlir::Type orig_weight_type,
    const Dimensions& batch_dims, mlir::MlirOp zero) {
  if (!weight.has_value()) {
    mlir::MlirOp zero_w =
        (orig_weight_type != compute_float_type)
            ? MakeScalarConstant(builder, 0.0, orig_weight_type)
            : zero;
    return mlir::stablehlo::BroadcastInDim(
        mlir::RankedTensorType::get(normalized_shape, orig_weight_type), zero_w,
        {});
  }
  mlir::MlirOp dgamma_full = mlir::stablehlo::Mul(dy_f32, x_hat);
  if (batch_dims.empty()) {
    if (orig_weight_type != compute_float_type) {
      return mlir::stablehlo::ConvertElementType(dgamma_full, orig_weight_type);
    }
    return dgamma_full;
  }
  const auto sum_reduce_builder =
      [compute_float_type](mlir::RegionBuilder& rb) {
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
            compute_float_type, rb.getRegion(), rb.getOpBuilder());
      };
  mlir::MlirOp dgamma = mlir::stablehlo::Reduce(
      builder, dgamma_full, zero, sum_reduce_builder, batch_dims)[0];
  if (orig_weight_type != compute_float_type) {
    dgamma = mlir::stablehlo::ConvertElementType(dgamma, orig_weight_type);
  }
  return dgamma;
}

// Builds the StableHLO operations for the gradient with respect to input
// (\nabla_x L).
// Mathematically:
//   \nabla_x L = rstd \odot ( \nabla_{\hat{x}} L - \hat{x} \odot
//       ( \frac{1}{N} \sum_{norm} ( \nabla_{\hat{x}} L \odot \hat{x} ) ) )
// where \nabla_{\hat{x}} L = \nabla_y L \odot \gamma (or \nabla_y L if no
// weight).
// Multiplies by constant (1/N) rather than vector division to avoid TPU MXU
// overhead, computes entirely in compute_float_type (F32), and casts to
// orig_elem_type if needed.
absl::StatusOr<mlir::MlirOp> BuildRmsNormDx(
    mlir::MlirBuilder& builder, mlir::MlirOp dy_f32, mlir::MlirOp x_hat,
    mlir::MlirOp rstd_f32_bcast, std::optional<mlir::MlirOp> weight,
    at::IntArrayRef normalized_shape, mlir::RankedTensorType compute_x_type,
    mlir::FloatType compute_float_type, mlir::Type orig_elem_type,
    const Dimensions& norm_dims, const Dimensions& batch_dims,
    mlir::MlirOp zero) {
  mlir::MlirOp dy_gamma_f32 = dy_f32;
  if (weight.has_value()) {
    mlir::MlirOp weight_f32 =
        (GetTensorTypeOrDie(*weight).getElementType() != compute_float_type)
            ? mlir::stablehlo::ConvertElementType(*weight, compute_float_type)
            : *weight;
    mlir::MlirOp gamma_bcast =
        mlir::stablehlo::BroadcastInDim(compute_x_type, weight_f32, norm_dims);
    dy_gamma_f32 = mlir::stablehlo::Mul(dy_f32, gamma_bcast);
  }

  const auto sum_reduce_builder =
      [compute_float_type](mlir::RegionBuilder& rb) {
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
            compute_float_type, rb.getRegion(), rb.getOpBuilder());
      };
  mlir::MlirOp dy_gamma_times_x_hat = mlir::stablehlo::Mul(dy_gamma_f32, x_hat);
  mlir::MlirOp sum_dy_gamma_times_x_hat = mlir::stablehlo::Reduce(
      builder, dy_gamma_times_x_hat, zero, sum_reduce_builder, norm_dims)[0];

  TT_ASSIGN_OR_RETURN(const int64_t n, NumElements(normalized_shape));
  mlir::MlirOp inv_n =
      MakeConstantLike(sum_dy_gamma_times_x_hat, 1.0 / static_cast<double>(n));
  mlir::MlirOp reduced_grad_factor =
      mlir::stablehlo::Mul(sum_dy_gamma_times_x_hat, inv_n);
  mlir::MlirOp reduced_grad_factor_bcast = mlir::stablehlo::BroadcastInDim(
      compute_x_type, reduced_grad_factor, batch_dims);

  mlir::MlirOp x_hat_times_grad_factor =
      mlir::stablehlo::Mul(x_hat, reduced_grad_factor_bcast);
  mlir::MlirOp diff =
      mlir::stablehlo::Subtract(dy_gamma_f32, x_hat_times_grad_factor);
  mlir::MlirOp dx_f32 = mlir::stablehlo::Mul(diff, rstd_f32_bcast);

  if (orig_elem_type != compute_float_type) {
    return mlir::stablehlo::ConvertElementType(dx_f32, orig_elem_type);
  }
  return dx_f32;
}

}  // namespace

absl::StatusOr<RmsNormBackwardShloResults> BuildRmsNormBackwardShlo(
    mlir::MlirOp dy, mlir::MlirOp x, mlir::MlirOp rstd,
    std::optional<mlir::MlirOp> weight, at::IntArrayRef normalized_shape,
    std::array<bool, 2> output_mask) {
  // Mathematical Formulation of RMSNorm Backward Pass (dL/dx and dL/d\gamma):
  // ---------------------------------------------------------------------------
  // Let y = x * rstd * \gamma, where rstd = 1 / \sqrt{Mean(x^2) + \epsilon}.
  // Using the chain rule:
  //   dL/d\gamma = \sum_{batch} (dL/dy * \hat{x}), where \hat{x} = x * rstd.
  //
  //   dL/dx = (grad_x_hat - \hat{x} * reduced_grad_factor) * rstd,
  //   where:
  //     grad_x_hat = dL/dy * \gamma (or dL/dy if \gamma is omitted),
  //     reduced_grad_factor = (1 / N) * \sum_{norm} (grad_x_hat * \hat{x}).
  // ---------------------------------------------------------------------------
  const mlir::RankedTensorType x_type = GetTensorTypeOrDie(x);
  const mlir::Type orig_elem_type = x_type.getElementType();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Error caught by unique caller:
                 // AtenFusedRmsNormBackward.
      orig_elem_type.isFloat(), error::kInvalidArgument)
      << "expected the input dtype to be floating point, got "
      << ToString(orig_elem_type);

  const int64_t x_rank = x_type.getShape().size();
  const int64_t norm_len = normalized_shape.size();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Error caught by unique caller:
                 // AtenFusedRmsNormBackward.
      norm_len <= x_rank, error::kInvalidArgument)
      << "expected the normalized shape to have <= " << x_rank
      << " dimensions, got " << norm_len;

  const bool needs_upcast = orig_elem_type.isF16() || orig_elem_type.isBF16();
  mlir::Type orig_weight_type = orig_elem_type;
  if (weight.has_value()) {
    orig_weight_type = GetTensorTypeOrDie(*weight).getElementType();
  }

  mlir::MlirBuilder& builder = x.getBuilder();

  if (!output_mask[0] && !output_mask[1]) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp zero_in,
                        MakeZeroSizedTensor(builder, orig_elem_type));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp zero_w,
                        MakeZeroSizedTensor(builder, orig_weight_type));
    return RmsNormBackwardShloResults{.grad_input = zero_in,
                                      .grad_weight = zero_w};
  }

  auto compute_float_type = mlir::cast<mlir::FloatType>(orig_elem_type);
  if (needs_upcast) {
    compute_float_type = builder.getOpBuilder().getF32Type();
  }

  const int64_t batch_len = x_rank - norm_len;

  Dimensions batch_dims(batch_len);
  absl::c_iota(batch_dims, 0);

  Dimensions norm_dims(norm_len);
  absl::c_iota(norm_dims, batch_len);

  Dimensions all_dims(x_rank);
  absl::c_iota(all_dims, 0);

  mlir::MlirOp zero = MakeScalarConstant(builder, 0.0, compute_float_type);

  // Upcast inputs to compute_float_type (F32) once.
  mlir::MlirOp x_f32 =
      (orig_elem_type != compute_float_type)
          ? mlir::stablehlo::ConvertElementType(x, compute_float_type)
          : x;
  mlir::MlirOp rstd_f32 =
      (GetTensorTypeOrDie(rstd).getElementType() != compute_float_type)
          ? mlir::stablehlo::ConvertElementType(rstd, compute_float_type)
          : rstd;
  const auto compute_x_type =
      mlir::RankedTensorType::get(x_type.getShape(), compute_float_type);
  mlir::MlirOp rstd_f32_bcast =
      mlir::stablehlo::BroadcastInDim(compute_x_type, rstd_f32, all_dims);

  // x_hat = x * rstd in compute_float_type (F32).
  // Exactly matches forward pass Mul(compute_input, rstd_broadcasted) for XLA
  // CSE.
  mlir::MlirOp x_hat = mlir::stablehlo::Mul(x_f32, rstd_f32_bcast);

  mlir::MlirOp dy_f32 =
      (GetTensorTypeOrDie(dy).getElementType() != compute_float_type)
          ? mlir::stablehlo::ConvertElementType(dy, compute_float_type)
          : dy;

  // 1. Compute dgamma = sum_batch(dy * x_hat)
  mlir::MlirOp dgamma;
  if (output_mask[1]) {
    TT_ASSIGN_OR_RETURN(dgamma,
                        BuildRmsNormDgamma(builder, dy_f32, x_hat, weight,
                                           normalized_shape, compute_float_type,
                                           orig_weight_type, batch_dims, zero));
  } else {
    TT_ASSIGN_OR_RETURN(dgamma, MakeZeroSizedTensor(builder, orig_weight_type));
  }

  // 2. Compute dx = (dy_gamma - x_hat * reduced_grad_factor) * rstd
  mlir::MlirOp dx;
  if (output_mask[0]) {
    TT_ASSIGN_OR_RETURN(
        dx, BuildRmsNormDx(builder, dy_f32, x_hat, rstd_f32_bcast, weight,
                           normalized_shape, compute_x_type, compute_float_type,
                           orig_elem_type, norm_dims, batch_dims, zero));
  } else {
    TT_ASSIGN_OR_RETURN(dx, MakeZeroSizedTensor(builder, orig_elem_type));
  }

  return RmsNormBackwardShloResults{.grad_input = dx, .grad_weight = dgamma};
}

}  // namespace torch_tpu
