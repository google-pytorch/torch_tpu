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

#include <cstdint>
#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "absl/algorithm/container.h"
#include "absl/functional/any_invocable.h"
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
  bool need_cast = element_type.getIntOrFloatBitWidth() < 32;
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

  mlir::MlirOp rstd_broadcasted = mlir::stablehlo::BroadcastInDim(
      GetTensorTypeOrDie(compute_input), rstd, unreduced_axes);

  // normalized = x * rstd
  mlir::MlirOp normalized_input =
      mlir::stablehlo::Mul(compute_input, rstd_broadcasted);

  // output = normalized * weight
  mlir::MlirOp output = normalized_input;
  if (weight_op.has_value()) {
    mlir::MlirOp compute_weight = *weight_op;
    if (need_cast) {
      compute_weight =
          mlir::stablehlo::ConvertElementType(*weight_op, compute_type);
    }
    mlir::MlirOp weight_broadcasted = mlir::stablehlo::BroadcastInDim(
        GetTensorTypeOrDie(compute_input), compute_weight, reduction_axes);
    output = mlir::stablehlo::Mul(normalized_input, weight_broadcasted);
  }

  // Cast back to original type if needed
  if (need_cast) {
    output = mlir::stablehlo::ConvertElementType(output, element_type);
  }

  auto rstd_unsqueezed = BuildKeepDimsShlo(compute_input, rstd, reduction_axes);

  // Helper struct reuse: .mean is effectively unused/zero for RMSNorm.
  // .reciprocal_std is preserved for backward pass compatibility.
  mlir::MlirOp final_zero = MakeScalarConstant(builder, 0.0, element_type);
  return LayerNormShloResults{.normalized_values = output,
                              .mean = final_zero,
                              .reciprocal_std = rstd_unsqueezed};
}

namespace {

struct RmsNormBackwardInputs {
  mlir::MlirOp x;
  mlir::MlirOp dy;
  mlir::MlirOp rstd;
  std::optional<mlir::MlirOp> weight;
  mlir::Type compute_type;
  mlir::Type original_element_type;
  bool need_cast;
};

RmsNormBackwardInputs PrepareRmsNormBackwardInputs(
    mlir::MlirBuilder& builder, mlir::MlirOp dy, mlir::MlirOp x,
    mlir::MlirOp rstd, std::optional<mlir::MlirOp> weight) {
  mlir::RankedTensorType x_tensor_type = GetTensorTypeOrDie(x);
  mlir::Type x_mlir_element_type = x_tensor_type.getElementType();

  // Perform computation in float32 to avoid overflow/underflow for f16/bf16.
  bool need_cast = false;
  if (auto float_type = mlir::dyn_cast<mlir::FloatType>(x_mlir_element_type)) {
    if (float_type.getWidth() < 32) need_cast = true;
  }

  mlir::Type compute_type = x_mlir_element_type;
  if (need_cast) {
    compute_type = builder.getOpBuilder().getF32Type();
  }

  mlir::MlirOp x_comp = x;
  mlir::MlirOp dy_comp = dy;
  mlir::MlirOp rstd_comp = rstd;
  std::optional<mlir::MlirOp> weight_comp;

  if (need_cast) {
    x_comp = mlir::stablehlo::ConvertElementType(x, compute_type);
    dy_comp = mlir::stablehlo::ConvertElementType(dy, compute_type);
    rstd_comp = mlir::stablehlo::ConvertElementType(rstd, compute_type);
    if (weight.has_value()) {
      weight_comp = mlir::stablehlo::ConvertElementType(*weight, compute_type);
    }
  } else {
    weight_comp = weight;
  }

  return {x_comp,      dy_comp,      rstd_comp,
          weight_comp, compute_type, x_mlir_element_type,
          need_cast};
}

mlir::MlirOp ComputeRmsNormBackwardDGamma(
    mlir::MlirBuilder& builder, mlir::MlirOp dy, mlir::MlirOp normalized_input,
    std::optional<mlir::MlirOp> weight, at::IntArrayRef normalized_shape,
    const Dimensions& batch_dims, mlir::Type compute_type,
    absl::AnyInvocable<void(mlir::RegionBuilder&)>& sum_reduce_builder) {
  mlir::MlirOp zeros = MakeScalarConstant(builder, 0.0, compute_type);
  if (weight.has_value()) {
    mlir::MlirOp dgamma_full = mlir::stablehlo::Mul(dy, normalized_input);
    return mlir::stablehlo::Reduce(
        builder, dgamma_full, zeros,
        [&](mlir::RegionBuilder& rb) { sum_reduce_builder(rb); },
        batch_dims)[0];
  }
  return mlir::stablehlo::BroadcastInDim(
      mlir::RankedTensorType::get(normalized_shape, compute_type), zeros, {});
}

mlir::MlirOp ComputeRmsNormBackwardDX(
    mlir::MlirBuilder& builder, mlir::MlirOp dy, mlir::MlirOp normalized_input,
    mlir::MlirOp rstd_bcast, std::optional<mlir::MlirOp> gamma_bcast,
    const Dimensions& norm_dims, const Dimensions& batch_dims,
    const int64_t normalized_dim_numl, mlir::Type compute_type,
    absl::AnyInvocable<void(mlir::RegionBuilder&)>& sum_reduce_builder) {
  // Mathematical Formulation of RMSNorm Backward Pass (dL/dx):
  // ---------------------------------------------------------------------------
  // Let y = x * rstd * \gamma, where rstd = 1 / \sqrt{Mean(x^2) + \epsilon}.
  // Using the chain rule:
  //   dL/dx_i = rstd * (dL/dy_i * \gamma_i)
  //             - (x_i / N) * rstd^3 * \sum_{k=1}^N (dL/dy_k * \gamma_k * x_k)
  //
  // Let normalized_input_i = x_i * rstd.
  // Let reduced_grad_factor = (1 / N) * \sum_{k=1}^N (dL/dy_k * \gamma_k *
  // normalized_input_k). Then:
  //   dL/dx = (rstd * dL/dy * \gamma) - (normalized_input * rstd *
  //   reduced_grad_factor)
  //
  // Optimization:
  //   Compute `reduced_grad_factor * (1 / N)` on the smaller reduced tensor [B,
  //   S] *before* broadcasting to [B, S, D], avoiding an elementwise scaling
  //   loop over all B * S * D elements across every transformer layer.
  // ---------------------------------------------------------------------------
  mlir::MlirOp zeros = MakeScalarConstant(builder, 0.0, compute_type);
  mlir::MlirOp dy_times_norm_input = mlir::stablehlo::Mul(dy, normalized_input);
  mlir::MlirOp reduced_grad_factor;

  if (gamma_bcast.has_value()) {
    mlir::MlirOp temp =
        mlir::stablehlo::Mul(dy_times_norm_input, gamma_bcast.value());
    reduced_grad_factor = mlir::stablehlo::Reduce(
        builder, temp, zeros,
        [&](mlir::RegionBuilder& rb) { sum_reduce_builder(rb); }, norm_dims)[0];
  } else {
    reduced_grad_factor = mlir::stablehlo::Reduce(
        builder, dy_times_norm_input, zeros,
        [&](mlir::RegionBuilder& rb) { sum_reduce_builder(rb); }, norm_dims)[0];
  }

  mlir::MlirOp inv_normalized_dim_elements = MakeConstantLike(
      reduced_grad_factor, 1.0 / static_cast<double>(normalized_dim_numl));
  reduced_grad_factor =
      mlir::stablehlo::Mul(reduced_grad_factor, inv_normalized_dim_elements);

  mlir::MlirOp reduced_grad_factor_broadcasted =
      mlir::stablehlo::BroadcastInDim(GetTensorTypeOrDie(normalized_input),
                                      reduced_grad_factor, batch_dims);

  // direct_grad_term = rstd * dy * gamma
  mlir::MlirOp direct_grad_term = mlir::stablehlo::Mul(rstd_bcast, dy);
  if (gamma_bcast.has_value()) {
    direct_grad_term =
        mlir::stablehlo::Mul(direct_grad_term, gamma_bcast.value());
  }

  // variance_grad_term = normalized_input * (reduced_grad_factor * rstd)
  mlir::MlirOp scaled_grad_factor =
      mlir::stablehlo::Mul(reduced_grad_factor_broadcasted, rstd_bcast);
  mlir::MlirOp variance_grad_term =
      mlir::stablehlo::Mul(normalized_input, scaled_grad_factor);

  return mlir::stablehlo::Subtract(direct_grad_term, variance_grad_term);
}

}  // namespace

absl::StatusOr<RmsNormBackwardShloResults> BuildRmsNormBackwardShlo(
    mlir::MlirOp dy, mlir::MlirOp x, mlir::MlirOp rstd,
    std::optional<mlir::MlirOp> weight, at::IntArrayRef normalized_shape) {
  mlir::MlirBuilder& builder = x.getBuilder();
  RmsNormBackwardInputs inputs =
      PrepareRmsNormBackwardInputs(builder, dy, x, rstd, weight);

  absl::AnyInvocable<void(mlir::RegionBuilder&)> sum_reduce_builder =
      [compute_type = inputs.compute_type](mlir::RegionBuilder& rb) {
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
            compute_type, rb.getRegion(), rb.getOpBuilder());
      };

  // Get Dimensions
  int64_t x_rank = GetTensorTypeOrDie(inputs.x).getShape().size();
  int64_t norm_len = normalized_shape.size();
  int64_t batch_len = x_rank - norm_len;

  Dimensions batch_dims;
  Dimensions norm_dims;
  Dimensions all_dims;
  for (int i = 0; i < x_rank; ++i) {
    if (i < batch_len)
      batch_dims.push_back(i);
    else
      norm_dims.push_back(i);

    all_dims.push_back(i);
  }

  // Broadcast Rstd to X shape
  mlir::MlirOp rstd_casted = inputs.rstd;
  mlir::Type x_tensor_elem_type = GetTensorTypeOrDie(inputs.x).getElementType();
  if (GetTensorTypeOrDie(rstd_casted).getElementType() != x_tensor_elem_type) {
    rstd_casted =
        mlir::stablehlo::ConvertElementType(rstd_casted, x_tensor_elem_type);
  }

  mlir::MlirOp rstd_bcast = mlir::stablehlo::BroadcastInDim(
      GetTensorTypeOrDie(inputs.x), rstd_casted, all_dims);

  std::optional<mlir::MlirOp> gamma_bcast;
  if (inputs.weight.has_value()) {
    gamma_bcast = mlir::stablehlo::BroadcastInDim(
        GetTensorTypeOrDie(inputs.x), inputs.weight.value(), norm_dims);
  }

  mlir::MlirOp normalized_input = mlir::stablehlo::Mul(inputs.x, rstd_bcast);

  mlir::MlirOp dgamma = ComputeRmsNormBackwardDGamma(
      builder, inputs.dy, normalized_input, inputs.weight, normalized_shape,
      batch_dims, inputs.compute_type, sum_reduce_builder);

  TT_ASSIGN_OR_RETURN(const int64_t normalized_dim_numl,
                      NumElements(normalized_shape));

  mlir::MlirOp dx = ComputeRmsNormBackwardDX(
      builder, inputs.dy, normalized_input, rstd_bcast, gamma_bcast, norm_dims,
      batch_dims, normalized_dim_numl, inputs.compute_type, sum_reduce_builder);

  // Cast outputs back to original type if needed
  if (inputs.need_cast) {
    if (inputs.weight.has_value()) {
      dgamma = mlir::stablehlo::ConvertElementType(
          dgamma, inputs.original_element_type);
    }
    dx = mlir::stablehlo::ConvertElementType(dx, inputs.original_element_type);
  }

  return RmsNormBackwardShloResults{dx, dgamma};
}

}  // namespace torch_tpu
