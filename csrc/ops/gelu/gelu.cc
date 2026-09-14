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
  auto k_0_5 = MakeConstantLike(input_op, 0.5);
  auto k_1_0 = MakeConstantLike(input_op, 1.0);
  auto kBeta = MakeConstantLike(input_op, std::sqrt(2.0 / std::numbers::pi));
  auto kBetaKappa =
      MakeConstantLike(input_op, 0.044715 * std::sqrt(2.0 / std::numbers::pi));

  mlir::MlirOp x_sq = stablehlo::Mul(input_op, input_op);
  mlir::MlirOp kBetaKappa_x_sq = stablehlo::Mul(kBetaKappa, x_sq);
  mlir::MlirOp inner_term = stablehlo::Add(kBeta, kBetaKappa_x_sq);
  mlir::MlirOp tanh_arg = stablehlo::Mul(input_op, inner_term);
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

  mlir::MlirOp self_alpha = stablehlo::Mul(input_op, k_alpha);
  mlir::MlirOp erf_out = chlo::Erf(self_alpha);
  mlir::MlirOp one_plus_erf = stablehlo::Add(k_1_0, erf_out);
  mlir::MlirOp half_x = stablehlo::Mul(k_0_5, input_op);
  return stablehlo::Mul(half_x, one_plus_erf);
}

// When approximation_type == "tanh", we compute the approximate gelu formula.
// NOTE: 'output_op' is treated here as the input 'x' to the forward Gelu.
//
// Forward approximate Formula (Horner form):
//   k = x * (sqrt(2/pi) + 0.044715 * sqrt(2/pi) * x^2)
//   y = 0.5 * x * (1 + tanh(k))
//
// Backward Derivative (dy/dx):
//   Using product rule on y = u * v, where u = 0.5 * x, v = 1 + tanh(k)
//   dy/dx = 0.5 * (1 + tanh(k)) + 0.5 * x * (1 - tanh^2(k)) * k'
//   Using 1 - tanh^2(k) = (1 - tanh(k)) * (1 + tanh(k)), we factor out (1 +
//   tanh(k)): dy/dx = 0.5 * (1 + tanh(k)) * [1 + x * (1 - tanh(k)) * k'] where
//   k' = sqrt(2/pi) + 3 * 0.044715 * sqrt(2/pi) * x^2
absl::StatusOr<mlir::MlirOp> BuildGeluBackwardGradInputApproximateShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp input_op) {
  auto k_0_5 = MakeConstantLike(input_op, 0.5);
  auto k_1_0 = MakeConstantLike(input_op, 1.0);
  auto kBeta = MakeConstantLike(input_op, std::sqrt(2.0 / std::numbers::pi));
  auto kBetaKappa =
      MakeConstantLike(input_op, 0.044715 * std::sqrt(2.0 / std::numbers::pi));
  auto kBeta3Kappa = MakeConstantLike(
      input_op, 3.0 * 0.044715 * std::sqrt(2.0 / std::numbers::pi));

  // 1. Recompute 'k' (inner) and tanh(k) using Horner form:
  //    k = x * (kBeta + kBetaKappa * x^2)
  mlir::MlirOp x_sq = stablehlo::Mul(input_op, input_op);
  mlir::MlirOp kBetaKappa_x_sq = stablehlo::Mul(kBetaKappa, x_sq);
  mlir::MlirOp inner_term = stablehlo::Add(kBeta, kBetaKappa_x_sq);
  mlir::MlirOp inner = stablehlo::Mul(input_op, inner_term);
  mlir::MlirOp tanh_inner = stablehlo::Tanh(inner);

  // 2. Calculate k' = kBeta + kBeta3Kappa * x^2
  mlir::MlirOp kBeta3Kappa_x_sq = stablehlo::Mul(kBeta3Kappa, x_sq);
  mlir::MlirOp inner_derivative = stablehlo::Add(kBeta, kBeta3Kappa_x_sq);

  // 3. Using algebraic identity 1 - T^2 = (1 - T)(1 + T):
  //    dy/dx = 0.5 * (1 + T) + 0.5 * x * (1 - T^2) * k'
  //          = 0.5 * (1 + T) * [ 1 + x * (1 - T) * k' ]
  mlir::MlirOp one_minus_tanh = stablehlo::Subtract(k_1_0, tanh_inner);
  mlir::MlirOp one_plus_tanh = stablehlo::Add(k_1_0, tanh_inner);
  mlir::MlirOp one_minus_tanh_x_inner_derivative =
      stablehlo::Mul(one_minus_tanh, inner_derivative);
  mlir::MlirOp bracket_term =
      stablehlo::Mul(input_op, one_minus_tanh_x_inner_derivative);
  mlir::MlirOp bracket = stablehlo::Add(k_1_0, bracket_term);
  mlir::MlirOp one_plus_tanh_x_bracket = stablehlo::Mul(one_plus_tanh, bracket);
  mlir::MlirOp scale = stablehlo::Mul(k_0_5, one_plus_tanh_x_bracket);

  // 4. Final Gradient: grad_output * scale
  return stablehlo::Mul(grad_output_op, scale);
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
//   dy/dx = 0.5 * [ (1 + erf(x / sqrt(2))) + sqrt(2/pi) * x * exp(-x^2 / 2) ]
absl::StatusOr<mlir::MlirOp> BuildGeluBackwardGradInputNoneShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp input_op) {
  auto k_alpha = MakeConstantLike(
      input_op, 0.5 * std::numbers::sqrt2);  // 1/sqrt(2) == sqrt(2)/2
  auto k_2beta = MakeConstantLike(input_op, std::sqrt(2.0 / std::numbers::pi));
  auto k_0_5 = MakeConstantLike(input_op, 0.5);
  auto k_1_0 = MakeConstantLike(input_op, 1.0);
  auto k_neg_0_5 = MakeConstantLike(input_op, -0.5);

  // 1. Calculate 1 + erf(x / sqrt(2))
  mlir::MlirOp self_alpha = stablehlo::Mul(input_op, k_alpha);
  mlir::MlirOp erf_out = chlo::Erf(self_alpha);
  mlir::MlirOp one_plus_erf = stablehlo::Add(k_1_0, erf_out);

  // 2. Calculate 2 * x * PDF(x) = sqrt(2/pi) * x * exp(-0.5 * x^2)
  mlir::MlirOp self_sq = stablehlo::Mul(input_op, input_op);
  mlir::MlirOp self_sq_neg_0_5 = stablehlo::Mul(self_sq, k_neg_0_5);
  mlir::MlirOp exp_out = stablehlo::Exp(self_sq_neg_0_5);
  mlir::MlirOp k_2beta_exp = stablehlo::Mul(k_2beta, exp_out);
  mlir::MlirOp pdf_scaled = stablehlo::Mul(input_op, k_2beta_exp);

  // 3. dy/dx * 2 = (1 + erf(x / sqrt(2))) + sqrt(2/pi) * x * exp(-0.5 * x^2)
  mlir::MlirOp dy_dx_2x = stablehlo::Add(one_plus_erf, pdf_scaled);

  // 4. Final Gradient: 0.5 * grad_output * (dy/dx * 2)
  mlir::MlirOp half_grad_output = stablehlo::Mul(k_0_5, grad_output_op);
  return stablehlo::Mul(half_grad_output, dy_dx_2x);
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
