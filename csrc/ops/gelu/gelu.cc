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

#include "csrc/ops/gelu/gelu.h"

#include <cmath>
#include <numbers>
#include <string_view>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;
namespace chlo = mlir::chlo;

namespace {
// TODO:(gleasonk) Make builder API allow inline exprs
// Currently `stablehlo::Abs(stablehlo::Abs(x))` is not allowed since we use
// non-const references.
absl::StatusOr<mlir::MlirOp> BuildGeluApproximateShlo(mlir::MlirOp input_op) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::RankedTensorType type = GetTensorTypeOrDie(input_op);
  const mlir::Type element_type = type.getElementType();

  mlir::MlirOp k_0_5 = MakeScalarConstant(builder, 0.5, element_type);
  mlir::MlirOp k_1_0 = MakeScalarConstant(builder, 1.0, element_type);
  mlir::MlirOp k_sqrt_2_div_pi = MakeScalarConstant(
      builder, std::sqrt(2.0 / std::numbers::pi), element_type);
  mlir::MlirOp k_kappa = MakeScalarConstant(builder, 0.044715, element_type);

  // Algebraically, inner_term = x + kappa * x^3 = x * (1.0 + kappa * x^2).
  // Factoring avoids transcendental Pow(x, 3.0), saving an ALU operation and
  // eliminating exp(3*log(x)) evaluation on TPU.
  mlir::MlirOp x_sq = stablehlo::Mul(input_op, input_op);
  TT_ASSIGN_OR_RETURN((auto [x_sq_bcst, kappa_bcst]),
                      ApplyBroadcastIfNeeded(x_sq, k_kappa));
  mlir::MlirOp kappa_x_sq = stablehlo::Mul(x_sq_bcst, kappa_bcst);
  TT_ASSIGN_OR_RETURN((auto [k1_bcst, k_x_sq_bcst]),
                      ApplyBroadcastIfNeeded(k_1_0, kappa_x_sq));
  mlir::MlirOp poly = stablehlo::Add(k1_bcst, k_x_sq_bcst);
  mlir::MlirOp inner_term = stablehlo::Mul(input_op, poly);
  TT_ASSIGN_OR_RETURN((auto [inner_term_bcst, sqrt_2_div_pi_bcst]),
                      ApplyBroadcastIfNeeded(inner_term, k_sqrt_2_div_pi));
  mlir::MlirOp tanh_arg = stablehlo::Mul(inner_term_bcst, sqrt_2_div_pi_bcst);
  mlir::MlirOp tanh_out = stablehlo::Tanh(tanh_arg);

  TT_ASSIGN_OR_RETURN((auto [k_0_5_bcst, input_bcst]),
                      ApplyBroadcastIfNeeded(k_0_5, input_op));
  mlir::MlirOp y_component = stablehlo::Mul(k_0_5_bcst, input_bcst);
  TT_ASSIGN_OR_RETURN((auto [k_1_0_bcst_tanh, tanh_out_bcst]),
                      ApplyBroadcastIfNeeded(k_1_0, tanh_out));
  mlir::MlirOp tanh_add = stablehlo::Add(k_1_0_bcst_tanh, tanh_out_bcst);
  return stablehlo::Mul(y_component, tanh_add);
}

absl::StatusOr<mlir::MlirOp> BuildGeluNoneShlo(mlir::MlirOp input_op) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::RankedTensorType type = GetTensorTypeOrDie(input_op);
  const mlir::Type element_type = type.getElementType();

  mlir::MlirOp k_0_5 = MakeScalarConstant(builder, 0.5, element_type);
  mlir::MlirOp k_1_0 = MakeScalarConstant(builder, 1.0, element_type);
  mlir::MlirOp k_alpha =
      MakeScalarConstant(builder, 0.5 * std::numbers::sqrt2, element_type);

  TT_ASSIGN_OR_RETURN((auto [input_alpha_bcst, k_alpha_bcst]),
                      ApplyBroadcastIfNeeded(input_op, k_alpha));
  mlir::MlirOp self_alpha = stablehlo::Mul(input_alpha_bcst, k_alpha_bcst);
  mlir::MlirOp erf_out = chlo::Erf(self_alpha);
  TT_ASSIGN_OR_RETURN((auto [k_1_0_bcst, erf_out_bcst]),
                      ApplyBroadcastIfNeeded(k_1_0, erf_out));
  mlir::MlirOp one_plus_erf = stablehlo::Add(k_1_0_bcst, erf_out_bcst);
  TT_ASSIGN_OR_RETURN((auto [k_0_5_bcst, input_half_bcst]),
                      ApplyBroadcastIfNeeded(k_0_5, input_op));
  mlir::MlirOp half_x = stablehlo::Mul(k_0_5_bcst, input_half_bcst);
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
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::RankedTensorType type = GetTensorTypeOrDie(input_op);
  const mlir::Type element_type = type.getElementType();

  mlir::MlirOp k_0_5 = MakeScalarConstant(builder, 0.5, element_type);
  mlir::MlirOp k_1_0 = MakeScalarConstant(builder, 1.0, element_type);
  mlir::MlirOp k_beta = MakeScalarConstant(
      builder, std::sqrt(2.0 / std::numbers::pi), element_type);
  mlir::MlirOp k_kappa = MakeScalarConstant(builder, 0.044715, element_type);
  // Precompute 3.0 * kappa = 3.0 * 0.044715 = 0.134145 at compile time.
  mlir::MlirOp k_3_kappa = MakeScalarConstant(builder, 0.134145, element_type);

  // 1. Recompute 'k' (inner) and tanh(k) used in the forward pass.
  // Factoring: inner_arg = x + kappa * x^3 = x * (1 + kappa * x^2).
  mlir::MlirOp x_sq = stablehlo::Mul(input_op, input_op);
  TT_ASSIGN_OR_RETURN((auto [x_sq_bcst, kappa_bcst]),
                      ApplyBroadcastIfNeeded(x_sq, k_kappa));
  mlir::MlirOp kappa_x_sq = stablehlo::Mul(x_sq_bcst, kappa_bcst);
  TT_ASSIGN_OR_RETURN((auto [k1_bcst, k_x_sq_bcst]),
                      ApplyBroadcastIfNeeded(k_1_0, kappa_x_sq));
  mlir::MlirOp poly = stablehlo::Add(k1_bcst, k_x_sq_bcst);
  mlir::MlirOp inner_arg = stablehlo::Mul(input_op, poly);
  TT_ASSIGN_OR_RETURN((auto [inner_arg_bcst, beta_bcst]),
                      ApplyBroadcastIfNeeded(inner_arg, k_beta));
  mlir::MlirOp inner = stablehlo::Mul(inner_arg_bcst, beta_bcst);  // k
  mlir::MlirOp tanh_inner = stablehlo::Tanh(inner);                // tanh(k)

  // 2. Calculate Term 1: 0.5 * (1 + tanh(k))
  TT_ASSIGN_OR_RETURN((auto [k_0_5_bcst, input_bcst]),
                      ApplyBroadcastIfNeeded(k_0_5, input_op));
  mlir::MlirOp left = stablehlo::Mul(k_0_5_bcst, input_bcst);
  TT_ASSIGN_OR_RETURN((auto [k_1_0_bcst_term1, tanh_inner_bcst]),
                      ApplyBroadcastIfNeeded(k_1_0, tanh_inner));
  mlir::MlirOp right = stablehlo::Add(k_1_0_bcst_term1, tanh_inner_bcst);
  TT_ASSIGN_OR_RETURN((auto [k_0_5_bcst_term1, right_bcst]),
                      ApplyBroadcastIfNeeded(k_0_5, right));
  mlir::MlirOp left_derivative = stablehlo::Mul(k_0_5_bcst_term1, right_bcst);

  // 3. Calculate Term 2: 0.5 * x * sech^2(k) * k'
  // sech^2(k) = 1 - tanh^2(k)
  mlir::MlirOp tanh_inner_sq = stablehlo::Mul(tanh_inner, tanh_inner);
  TT_ASSIGN_OR_RETURN((auto [k_1_0_bcst_term2, tanh_inner_sq_bcst]),
                      ApplyBroadcastIfNeeded(k_1_0, tanh_inner_sq));
  mlir::MlirOp tanh_derivative =
      stablehlo::Subtract(k_1_0_bcst_term2, tanh_inner_sq_bcst);

  // Calculate k' = sqrt(2/pi) * (1 + 3 * kappa * x^2) using precomputed 3 *
  // kappa and reusing x_sq cleanly.
  TT_ASSIGN_OR_RETURN((auto [k_3_kappa_bcst, x_sq_bcst2]),
                      ApplyBroadcastIfNeeded(k_3_kappa, x_sq));
  mlir::MlirOp k3_kappa_x_sq = stablehlo::Mul(k_3_kappa_bcst, x_sq_bcst2);
  TT_ASSIGN_OR_RETURN((auto [k_1_0_bcst_kprime, k3_kappa_x_sq_bcst]),
                      ApplyBroadcastIfNeeded(k_1_0, k3_kappa_x_sq));
  mlir::MlirOp inner_derivative_arg =
      stablehlo::Add(k_1_0_bcst_kprime, k3_kappa_x_sq_bcst);
  TT_ASSIGN_OR_RETURN((auto [inner_deriv_bcst, beta_bcst2]),
                      ApplyBroadcastIfNeeded(inner_derivative_arg, k_beta));
  mlir::MlirOp inner_derivative = stablehlo::Mul(inner_deriv_bcst, beta_bcst2);

  mlir::MlirOp inner_tanh_derivative =
      stablehlo::Mul(tanh_derivative, inner_derivative);
  mlir::MlirOp right_derivative = stablehlo::Mul(left, inner_tanh_derivative);

  // 4. Final Gradient: grad_output * (Term 1 + Term 2)
  mlir::MlirOp left_derivative_plus_right_derivative =
      stablehlo::Add(left_derivative, right_derivative);
  TT_ASSIGN_OR_RETURN(
      (auto [grad_output_bcst, deriv_bcst]),
      ApplyBroadcastIfNeeded(grad_output_op,
                             left_derivative_plus_right_derivative));
  return stablehlo::Mul(grad_output_bcst, deriv_bcst);
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
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::RankedTensorType type = GetTensorTypeOrDie(input_op);
  const mlir::Type element_type = type.getElementType();

  mlir::MlirOp k_alpha =
      MakeScalarConstant(builder, 0.5 * std::numbers::sqrt2, element_type);
  mlir::MlirOp k_beta = MakeScalarConstant(
      builder, 1.0 / std::sqrt(2.0 * std::numbers::pi), element_type);
  mlir::MlirOp k_0_5 = MakeScalarConstant(builder, 0.5, element_type);
  mlir::MlirOp k_1_0 = MakeScalarConstant(builder, 1.0, element_type);
  mlir::MlirOp k_neg_0_5 = MakeScalarConstant(builder, -0.5, element_type);

  // 1. Calculate CDF(x) = 0.5 * (1 + erf(x / sqrt(2)))
  TT_ASSIGN_OR_RETURN((auto [input_alpha_bcst, k_alpha_bcst]),
                      ApplyBroadcastIfNeeded(input_op, k_alpha));
  mlir::MlirOp self_alpha = stablehlo::Mul(input_alpha_bcst, k_alpha_bcst);
  mlir::MlirOp erf_out = chlo::Erf(self_alpha);
  TT_ASSIGN_OR_RETURN((auto [k_1_0_bcst, erf_out_bcst]),
                      ApplyBroadcastIfNeeded(k_1_0, erf_out));
  mlir::MlirOp one_plus_erf = stablehlo::Add(k_1_0_bcst, erf_out_bcst);
  TT_ASSIGN_OR_RETURN((auto [k_0_5_bcst, one_plus_erf_bcst]),
                      ApplyBroadcastIfNeeded(k_0_5, one_plus_erf));
  mlir::MlirOp cdf = stablehlo::Mul(k_0_5_bcst, one_plus_erf_bcst);

  // 2. Calculate PDF(x) = (1 / sqrt(2*pi)) * exp(-x^2 / 2)
  mlir::MlirOp self_sq = stablehlo::Mul(input_op, input_op);
  TT_ASSIGN_OR_RETURN((auto [self_sq_bcst, k_neg_0_5_bcst]),
                      ApplyBroadcastIfNeeded(self_sq, k_neg_0_5));
  mlir::MlirOp self_sq_neg_0_5 = stablehlo::Mul(self_sq_bcst, k_neg_0_5_bcst);
  mlir::MlirOp exp_out = stablehlo::Exp(self_sq_neg_0_5);
  TT_ASSIGN_OR_RETURN((auto [k_beta_bcst, exp_out_bcst]),
                      ApplyBroadcastIfNeeded(k_beta, exp_out));
  mlir::MlirOp pdf = stablehlo::Mul(k_beta_bcst, exp_out_bcst);

  // 3. Calculate dy/dx = CDF(x) + x * PDF(x)
  mlir::MlirOp self_pdf = stablehlo::Mul(input_op, pdf);
  mlir::MlirOp self_pdf_plus_cdf = stablehlo::Add(self_pdf, cdf);

  // 4. Final Gradient: grad_output * dy/dx
  TT_ASSIGN_OR_RETURN(
      (auto [grad_output_bcst, deriv_bcst]),
      ApplyBroadcastIfNeeded(grad_output_op, self_pdf_plus_cdf));
  return stablehlo::Mul(grad_output_bcst, deriv_bcst);
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
