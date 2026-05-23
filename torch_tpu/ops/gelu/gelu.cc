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

#include "torch_tpu/ops/gelu/gelu.h"

#include <cmath>
#include <numbers>
#include <string_view>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;
namespace chlo = mlir::chlo;

namespace {
// TODO:(gleasonk) Make builder API allow inline exprs
// Currently `stablehlo::Abs(stablehlo::Abs(x))` is not allowed since we use
// non-const references.
absl::StatusOr<mlir::MlirOp> BuildGeluApproximateShlo(mlir::MlirOp input_op) {
  auto k_0_5 = MakeConstantLike(input_op, 0.5);
  auto k_1_0 = MakeConstantLike(input_op, 1.0);
  auto k_sqrt_2_div_pi =
      MakeConstantLike(input_op, std::sqrt(2.0 / std::numbers::pi));
  auto k_0_044715 = MakeConstantLike(input_op, 0.044715);
  auto k_3_0 = MakeConstantLike(input_op, 3.0);

  mlir::MlirOp x_cubed = stablehlo::Pow(input_op, k_3_0);
  mlir::MlirOp x_mul = stablehlo::Mul(x_cubed, k_0_044715);
  mlir::MlirOp inner_term = stablehlo::Add(input_op, x_mul);
  mlir::MlirOp tanh_arg = stablehlo::Mul(inner_term, k_sqrt_2_div_pi);
  mlir::MlirOp tanh_out = stablehlo::Tanh(tanh_arg);
  mlir::MlirOp y_component = stablehlo::Mul(k_0_5, input_op);
  mlir::MlirOp tanh_add = stablehlo::Add(k_1_0, tanh_out);
  return stablehlo::Mul(y_component, tanh_add);
}

absl::StatusOr<mlir::MlirOp> BuildGeluNoneShlo(mlir::MlirOp input_op) {
  auto k_0_5 = MakeConstantLike(input_op, 0.5);
  auto k_1_0 = MakeConstantLike(input_op, 1.0);
  auto k_alpha = MakeConstantLike(
      input_op, 0.5 * std::numbers::sqrt2);  // 1/sqrt(2) == sqrt(2)/2

  auto self_alpha = stablehlo::Mul(input_op, k_alpha);
  auto erf_out = chlo::Erf(self_alpha);
  auto one_plus_erf = stablehlo::Add(k_1_0, erf_out);
  auto half_x = stablehlo::Mul(k_0_5, input_op);
  return stablehlo::Mul(half_x, one_plus_erf);
}

// When approximation_type == "tanh", we compute the approximate gelu formula.
// NOTE: 'output_op' is treated here as the input 'x' to the forward Gelu.
//
// Forward approximate Formula:
//   k = sqrt(2/pi) * (x + 0.044715 * x^3)
//   y = 0.5 * x * (1 + tanh(k))
//
// Backward Derivative (dy/dx):
//   Using product rule on y = u * v, where u = 0.5 * x, v = 1 + tanh(k)
//   dy/dx = u'v + uv'
//   Term 1 (u'v): 0.5 * (1 + tanh(k))
//   Term 2 (uv'): 0.5 * x * (1 - tanh^2(k)) * (dk/dx)
//   dk/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
absl::StatusOr<mlir::MlirOp> BuildGeluBackwardGradInputApproximateShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp input_op) {
  auto k_0_5 = MakeConstantLike(input_op, 0.5);
  auto k_1_0 = MakeConstantLike(input_op, 1.0);
  auto k_3_0 = MakeConstantLike(input_op, 3.0);
  // kBeta = sqrt(2/pi)
  auto kBeta = MakeConstantLike(input_op, std::sqrt(2.0 / std::numbers::pi));
  auto kKappa = MakeConstantLike(input_op, 0.044715);

  // 1. Recompute 'k' (inner) and tanh(k) used in the forward pass
  mlir::MlirOp x_sq = stablehlo::Mul(input_op, input_op);
  mlir::MlirOp x_cube = stablehlo::Mul(x_sq, input_op);
  mlir::MlirOp kKappa_x_cube = stablehlo::Mul(kKappa, x_cube);
  // inner_arg = x + kappa * x^3
  mlir::MlirOp inner_arg = stablehlo::Add(input_op, kKappa_x_cube);
  mlir::MlirOp inner = stablehlo::Mul(kBeta, inner_arg);  // k
  mlir::MlirOp tanh_inner = stablehlo::Tanh(inner);       // tanh(k)

  // 2. Calculate Term 1: 0.5 * (1 + tanh(k))
  mlir::MlirOp left = stablehlo::Mul(k_0_5, input_op);
  mlir::MlirOp right = stablehlo::Add(k_1_0, tanh_inner);
  mlir::MlirOp left_derivative = stablehlo::Mul(k_0_5, right);

  // 3. Calculate Term 2: 0.5 * x * sech^2(k) * k'
  // sech^2(k) = 1 - tanh^2(k)
  mlir::MlirOp tanh_inner_sq = stablehlo::Mul(tanh_inner, tanh_inner);
  mlir::MlirOp tanh_derivative = stablehlo::Subtract(k_1_0, tanh_inner_sq);

  // Calculate k' = sqrt(2/pi) * (1 + 3 * kappa * x^2)
  mlir::MlirOp k3Kappa = stablehlo::Mul(k_3_0, kKappa);
  mlir::MlirOp k3Kappa_x_sq = stablehlo::Mul(k3Kappa, x_sq);
  mlir::MlirOp inner_derivative_arg = stablehlo::Add(k_1_0, k3Kappa_x_sq);
  mlir::MlirOp inner_derivative = stablehlo::Mul(kBeta, inner_derivative_arg);

  mlir::MlirOp inner_tanh_derivative =
      stablehlo::Mul(tanh_derivative, inner_derivative);
  mlir::MlirOp right_derivative = stablehlo::Mul(left, inner_tanh_derivative);

  // 4. Final Gradient: grad_output * (Term 1 + Term 2)
  mlir::MlirOp left_derivative_plus_right_derivative =
      stablehlo::Add(left_derivative, right_derivative);
  return stablehlo::Mul(grad_output_op, left_derivative_plus_right_derivative);
}

// When approximation_type == "none", we compute the exact gelu formula.
// NOTE: 'output_op' is treated here as the input 'x' to the forward Gelu.
//
// Forward Exact Formula:
//   y = x * CDF(x)
//   CDF(x) = 0.5 * (1 + erf(x / sqrt(2)))
//
// Backward Derivative (dy/dx):
//   dy/dx = CDF(x) + x * PDF(x)
//   PDF(x) = (1 / sqrt(2*pi)) * exp(-x^2 / 2)
absl::StatusOr<mlir::MlirOp> BuildGeluBackwardGradInputNoneShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp input_op) {
  auto k_alpha = MakeConstantLike(
      input_op, 0.5 * std::numbers::sqrt2);  // 1/sqrt(2) == sqrt(2)/2
  // kBeta = 1 / sqrt(2 * pi)
  auto k_beta =
      MakeConstantLike(input_op, 1.0 / std::sqrt(2.0 * std::numbers::pi));
  auto k_0_5 = MakeConstantLike(input_op, 0.5);
  auto k_1_0 = MakeConstantLike(input_op, 1.0);
  auto k_neg_0_5 = MakeConstantLike(input_op, -0.5);

  // 1. Calculate CDF(x) = 0.5 * (1 + erf(x / sqrt(2)))
  mlir::MlirOp self_alpha = stablehlo::Mul(input_op, k_alpha);
  mlir::MlirOp erf_out = chlo::Erf(self_alpha);
  mlir::MlirOp one_plus_erf = stablehlo::Add(k_1_0, erf_out);
  mlir::MlirOp cdf = stablehlo::Mul(k_0_5, one_plus_erf);

  // 2. Calculate PDF(x) = (1 / sqrt(2*pi)) * exp(-x^2 / 2)
  mlir::MlirOp self_sq = stablehlo::Mul(input_op, input_op);
  mlir::MlirOp self_sq_neg_0_5 = stablehlo::Mul(self_sq, k_neg_0_5);
  mlir::MlirOp exp_out = stablehlo::Exp(self_sq_neg_0_5);
  mlir::MlirOp pdf = stablehlo::Mul(k_beta, exp_out);

  // 3. Calculate dy/dx = CDF(x) + x * PDF(x)
  mlir::MlirOp self_pdf = stablehlo::Mul(input_op, pdf);
  mlir::MlirOp self_pdf_plus_cdf = stablehlo::Add(self_pdf, cdf);

  // 4. Final Gradient: grad_output * dy/dx
  return stablehlo::Mul(grad_output_op, self_pdf_plus_cdf);
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildGeluShlo(mlir::MlirOp input_op,
                                           std::string_view approximation_type,
                                           mlir::ElementType output_dtype) {
  ABSL_VLOG(1) << "[BuildGeluShlo] input_op: "
               << mlir::debugString(input_op.getValue().getLoc());

  TT_ASSIGN_OR_RETURN(mlir::ElementType computation_dtype,
                      InferComputationDtype(output_dtype));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp computation_input_op,
                      ConvertIfInteger(input_op, computation_dtype));
  computation_input_op = mlir::stablehlo::ConvertElementType(
      computation_input_op, computation_dtype);

  absl::StatusOr<mlir::MlirOp> result_op;
  if (approximation_type == "none") {
    result_op = BuildGeluNoneShlo(computation_input_op);
  } else {  // approximation_type == "tanh"
    result_op = BuildGeluApproximateShlo(computation_input_op);
  }

  TT_ASSIGN_OR_RETURN(mlir::MlirOp result, result_op);
  return mlir::stablehlo::ConvertElementType(result, output_dtype);
}

absl::StatusOr<mlir::MlirOp> BuildGeluBackwardGradInputShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp input_op,
    std::string_view approximation_type, mlir::ElementType output_dtype) {
  TT_ASSIGN_OR_RETURN(mlir::ElementType computation_dtype,
                      InferComputationDtype(output_dtype));

  TT_ASSIGN_OR_RETURN(
      (auto [computation_grad_output_op, computation_input_op]),
      ConvertIfIntegers(grad_output_op, input_op, computation_dtype));
  computation_grad_output_op = mlir::stablehlo::ConvertElementType(
      computation_grad_output_op, computation_dtype);
  computation_input_op = mlir::stablehlo::ConvertElementType(
      computation_input_op, computation_dtype);

  mlir::MlirOp result_op_val;
  if (approximation_type == "none") {
    TT_ASSIGN_OR_RETURN(result_op_val,
                        BuildGeluBackwardGradInputNoneShlo(
                            computation_grad_output_op, computation_input_op));
  } else {  // approximation_type == "tanh"
    TT_ASSIGN_OR_RETURN(result_op_val,
                        BuildGeluBackwardGradInputApproximateShlo(
                            computation_grad_output_op, computation_input_op));
  }

  return mlir::stablehlo::ConvertElementType(result_op_val, output_dtype);
}

}  // namespace torch_tpu
