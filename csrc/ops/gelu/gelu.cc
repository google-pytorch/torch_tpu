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
#include "csrc/ops/binary.h"
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
  auto element_type = GetElementTypeOrDie(input_op);

  auto k_0_5 = MakeScalarConstant(builder, 0.5, element_type);
  auto k_1_0 = MakeScalarConstant(builder, 1.0, element_type);
  auto k_sqrt_2_div_pi = MakeScalarConstant(
      builder, std::sqrt(2.0 / std::numbers::pi), element_type);
  auto k_0_044715 = MakeScalarConstant(builder, 0.044715, element_type);

  TT_ASSIGN_OR_RETURN(mlir::MlirOp x_sq, BuildMulShlo(input_op, input_op));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp x_cubed, BuildMulShlo(x_sq, input_op));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp x_mul, BuildMulShlo(x_cubed, k_0_044715));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp inner_term, BuildAddShlo(input_op, x_mul));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp tanh_arg,
                      BuildMulShlo(inner_term, k_sqrt_2_div_pi));
  mlir::MlirOp tanh_out = stablehlo::Tanh(tanh_arg);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp y_component, BuildMulShlo(k_0_5, input_op));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp tanh_add, BuildAddShlo(k_1_0, tanh_out));
  return BuildMulShlo(y_component, tanh_add);
}

absl::StatusOr<mlir::MlirOp> BuildGeluNoneShlo(mlir::MlirOp input_op) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  auto element_type = GetElementTypeOrDie(input_op);

  auto k_0_5 = MakeScalarConstant(builder, 0.5, element_type);
  auto k_1_0 = MakeScalarConstant(builder, 1.0, element_type);
  auto k_alpha = MakeScalarConstant(builder, 0.5 * std::numbers::sqrt2,
                                    element_type);  // 1/sqrt(2) == sqrt(2)/2

  TT_ASSIGN_OR_RETURN(mlir::MlirOp self_alpha, BuildMulShlo(input_op, k_alpha));
  mlir::MlirOp erf_out = chlo::Erf(self_alpha);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp one_plus_erf, BuildAddShlo(k_1_0, erf_out));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp half_x, BuildMulShlo(k_0_5, input_op));
  return BuildMulShlo(half_x, one_plus_erf);
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
  auto element_type = GetElementTypeOrDie(input_op);

  auto k_0_5 = MakeScalarConstant(builder, 0.5, element_type);
  auto k_1_0 = MakeScalarConstant(builder, 1.0, element_type);
  // kBeta = sqrt(2/pi)
  auto kBeta = MakeScalarConstant(builder, std::sqrt(2.0 / std::numbers::pi),
                                  element_type);
  auto kKappa = MakeScalarConstant(builder, 0.044715, element_type);
  auto k3Kappa = MakeScalarConstant(builder, 3.0 * 0.044715, element_type);

  // 1. Recompute 'k' (inner) and tanh(k) used in the forward pass
  TT_ASSIGN_OR_RETURN(mlir::MlirOp x_sq, BuildMulShlo(input_op, input_op));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp x_cube, BuildMulShlo(x_sq, input_op));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp kKappa_x_cube, BuildMulShlo(kKappa, x_cube));
  // inner_arg = x + kappa * x^3
  TT_ASSIGN_OR_RETURN(mlir::MlirOp inner_arg,
                      BuildAddShlo(input_op, kKappa_x_cube));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp inner, BuildMulShlo(kBeta, inner_arg));  // k
  mlir::MlirOp tanh_inner = stablehlo::Tanh(inner);  // tanh(k)

  // 2. Calculate Term 1 (without 0.5): 1 + tanh(k)
  TT_ASSIGN_OR_RETURN(mlir::MlirOp left_deriv_no_half,
                      BuildAddShlo(k_1_0, tanh_inner));

  // 3. Calculate Term 2 (without 0.5): x * sech^2(k) * k'
  // sech^2(k) = 1 - tanh^2(k)
  TT_ASSIGN_OR_RETURN(mlir::MlirOp tanh_inner_sq,
                      BuildMulShlo(tanh_inner, tanh_inner));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp tanh_derivative,
                      BuildSubShlo(k_1_0, tanh_inner_sq));

  // Calculate k' = sqrt(2/pi) * (1 + 3 * kappa * x^2)
  TT_ASSIGN_OR_RETURN(mlir::MlirOp k3Kappa_x_sq, BuildMulShlo(k3Kappa, x_sq));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp inner_derivative_arg,
                      BuildAddShlo(k_1_0, k3Kappa_x_sq));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp inner_derivative,
                      BuildMulShlo(kBeta, inner_derivative_arg));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp inner_tanh_derivative,
                      BuildMulShlo(tanh_derivative, inner_derivative));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp right_deriv_no_half,
                      BuildMulShlo(input_op, inner_tanh_derivative));

  // 4. Final Gradient: grad_output * 0.5 * (Term 1 + Term 2)
  TT_ASSIGN_OR_RETURN(mlir::MlirOp sum_deriv,
                      BuildAddShlo(left_deriv_no_half, right_deriv_no_half));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp half_sum_deriv,
                      BuildMulShlo(k_0_5, sum_deriv));
  return BuildMulShlo(grad_output_op, half_sum_deriv);
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
  auto element_type = GetElementTypeOrDie(input_op);

  auto k_alpha = MakeScalarConstant(builder, 0.5 * std::numbers::sqrt2,
                                    element_type);  // 1/sqrt(2) == sqrt(2)/2
  // kBeta = 1 / sqrt(2 * pi)
  auto k_beta = MakeScalarConstant(
      builder, 1.0 / std::sqrt(2.0 * std::numbers::pi), element_type);
  auto k_0_5 = MakeScalarConstant(builder, 0.5, element_type);
  auto k_1_0 = MakeScalarConstant(builder, 1.0, element_type);
  auto k_neg_0_5 = MakeScalarConstant(builder, -0.5, element_type);

  // 1. Calculate CDF(x) = 0.5 * (1 + erf(x / sqrt(2)))
  TT_ASSIGN_OR_RETURN(mlir::MlirOp self_alpha, BuildMulShlo(input_op, k_alpha));
  mlir::MlirOp erf_out = chlo::Erf(self_alpha);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp one_plus_erf, BuildAddShlo(k_1_0, erf_out));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp cdf, BuildMulShlo(k_0_5, one_plus_erf));

  // 2. Calculate PDF(x) = (1 / sqrt(2*pi)) * exp(-x^2 / 2)
  TT_ASSIGN_OR_RETURN(mlir::MlirOp self_sq, BuildMulShlo(input_op, input_op));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp self_sq_neg_0_5,
                      BuildMulShlo(self_sq, k_neg_0_5));
  mlir::MlirOp exp_out = stablehlo::Exp(self_sq_neg_0_5);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp pdf, BuildMulShlo(k_beta, exp_out));

  // 3. Calculate dy/dx = CDF(x) + x * PDF(x)
  TT_ASSIGN_OR_RETURN(mlir::MlirOp self_pdf, BuildMulShlo(input_op, pdf));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp self_pdf_plus_cdf,
                      BuildAddShlo(self_pdf, cdf));

  // 4. Final Gradient: grad_output * dy/dx
  return BuildMulShlo(grad_output_op, self_pdf_plus_cdf);
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
