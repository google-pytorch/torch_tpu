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
#include "csrc/ops/unary.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {
// TODO:(gleasonk) Make builder API allow inline exprs
// Currently `stablehlo::Abs(stablehlo::Abs(x))` is not allowed since we use
// non-const references.
absl::StatusOr<mlir::MlirOp> BuildGeluApproximateShlo(mlir::MlirOp input_op) {
  mlir::MlirOp k_0_5 = MakeConstantLike(input_op, 0.5);
  mlir::MlirOp k_1_0 = MakeConstantLike(input_op, 1.0);
  mlir::MlirOp k_sqrt_2_div_pi =
      MakeConstantLike(input_op, std::sqrt(2.0 / std::numbers::pi));
  mlir::MlirOp k_0_044715 = MakeConstantLike(input_op, 0.044715);

  mlir::MlirOp x_sq = stablehlo::Mul(input_op, input_op);
  mlir::MlirOp x_cubed = stablehlo::Mul(x_sq, input_op);
  mlir::MlirOp x_mul = stablehlo::Mul(x_cubed, k_0_044715);
  mlir::MlirOp inner_term = stablehlo::Add(input_op, x_mul);
  mlir::MlirOp tanh_arg = stablehlo::Mul(inner_term, k_sqrt_2_div_pi);

  // gelu(x) = 0.5 * x * (1 + tanh(z))
  mlir::MlirOp tanh_out = stablehlo::Tanh(tanh_arg);
  mlir::MlirOp one_plus_tanh = stablehlo::Add(k_1_0, tanh_out);
  mlir::MlirOp half_x = stablehlo::Mul(k_0_5, input_op);
  return stablehlo::Mul(half_x, one_plus_tanh);
}

// Exact gelu: y = 0.5 * x * (1 + erf(x / sqrt(2)))
//               = 0.5 * x * erfc(-x / sqrt(2)).
// The erfc form avoids the cancellation in 1 + erf(...) for negative x, where
// erf(...) is close to -1.
absl::StatusOr<mlir::MlirOp> BuildGeluNoneShlo(
    mlir::MlirOp input_op, const mlir::ElementType output_dtype) {
  mlir::MlirOp k_0_5 = MakeConstantLike(input_op, 0.5);
  mlir::MlirOp k_neg_alpha = MakeConstantLike(
      input_op, -0.5 * std::numbers::sqrt2);  // -1/sqrt(2) == -sqrt(2)/2

  mlir::MlirOp self_neg_alpha = stablehlo::Mul(input_op, k_neg_alpha);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp erfc_out, BuildErfcForTargetDtypeShlo(
                                                 self_neg_alpha, output_dtype));
  mlir::MlirOp half_x = stablehlo::Mul(k_0_5, input_op);
  return stablehlo::Mul(half_x, erfc_out);
}

// When approximation_type == "tanh", we compute the approximate gelu formula.
// NOTE: 'output_op' is treated here as the input 'x' to the forward Gelu.
//
// Forward approximate Formula:
//   k = sqrt(2/pi) * (x + 0.044715 * x^3)
//   t = tanh(k)
//   y = 0.5 * x * (1 + t)
//
// Backward Derivative (dy/dx):
//   dy/dx = 0.5 * ((1 + t) + x * (1 - t^2) * (dk/dx))
//   dk/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
absl::StatusOr<mlir::MlirOp> BuildGeluBackwardGradInputApproximateShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp input_op) {
  mlir::MlirOp k_0_5 = MakeConstantLike(input_op, 0.5);
  mlir::MlirOp k_1_0 = MakeConstantLike(input_op, 1.0);
  mlir::MlirOp k_3_0 = MakeConstantLike(input_op, 3.0);
  // kBeta = sqrt(2/pi)
  mlir::MlirOp k_beta =
      MakeConstantLike(input_op, std::sqrt(2.0 / std::numbers::pi));
  mlir::MlirOp k_kappa = MakeConstantLike(input_op, 0.044715);

  // 1. Recompute 'k' and t = tanh(k) using a single native VPU op.
  mlir::MlirOp x_sq = stablehlo::Mul(input_op, input_op);
  mlir::MlirOp x_cube = stablehlo::Mul(x_sq, input_op);
  mlir::MlirOp k_kappa_x_cube = stablehlo::Mul(k_kappa, x_cube);
  mlir::MlirOp inner_arg = stablehlo::Add(input_op, k_kappa_x_cube);
  mlir::MlirOp inner = stablehlo::Mul(k_beta, inner_arg);
  mlir::MlirOp tanh_out = stablehlo::Tanh(inner);

  // 2. sech^2(k) = (1 - t) * (1 + t). Factoring avoids losing low bits of
  //    (1 - t^2) when |t| is close to 1.
  mlir::MlirOp one_plus_t = stablehlo::Add(k_1_0, tanh_out);
  mlir::MlirOp one_minus_t = stablehlo::Subtract(k_1_0, tanh_out);
  mlir::MlirOp sech2 = stablehlo::Mul(one_minus_t, one_plus_t);

  // 3. Calculate k' = sqrt(2/pi) * (1 + 3 * kappa * x^2)
  mlir::MlirOp k3_kappa = stablehlo::Mul(k_3_0, k_kappa);
  mlir::MlirOp k3_kappa_x_sq = stablehlo::Mul(k3_kappa, x_sq);
  mlir::MlirOp inner_derivative_arg = stablehlo::Add(k_1_0, k3_kappa_x_sq);
  mlir::MlirOp inner_derivative = stablehlo::Mul(k_beta, inner_derivative_arg);

  // 4. dy/dx = 0.5 * ((1 + t) + x * sech^2(k) * k')
  mlir::MlirOp x_sech2 = stablehlo::Mul(input_op, sech2);
  mlir::MlirOp right_term = stablehlo::Mul(x_sech2, inner_derivative);
  mlir::MlirOp total_sum = stablehlo::Add(one_plus_t, right_term);
  mlir::MlirOp dy_dx = stablehlo::Mul(k_0_5, total_sum);
  return stablehlo::Mul(grad_output_op, dy_dx);
}

// When approximation_type == "none", we compute the exact gelu formula.
// NOTE: 'output_op' is treated here as the input 'x' to the forward Gelu.
//
// Forward Exact Formula:
//   y = x * CDF(x)
//   CDF(x) = 0.5 * (1 + erf(x / sqrt(2))) = 0.5 * erfc(-x / sqrt(2))
//
// Backward Derivative (dy/dx):
//   dy/dx = CDF(x) + x * PDF(x)
//   PDF(x) = (1 / sqrt(2*pi)) * exp(-x^2 / 2)
absl::StatusOr<mlir::MlirOp> BuildGeluBackwardGradInputNoneShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp input_op,
    const mlir::ElementType output_dtype) {
  mlir::MlirOp k_neg_alpha = MakeConstantLike(
      input_op, -0.5 * std::numbers::sqrt2);  // -1/sqrt(2) == -sqrt(2)/2
  // kBeta = 1 / sqrt(2 * pi)
  auto k_beta =
      MakeConstantLike(input_op, 1.0 / std::sqrt(2.0 * std::numbers::pi));
  auto k_0_5 = MakeConstantLike(input_op, 0.5);
  auto k_neg_0_5 = MakeConstantLike(input_op, -0.5);

  // 1. Calculate CDF(x) = 0.5 * erfc(-x / sqrt(2)). This is equal to
  //    0.5 * (1 + erf(x / sqrt(2))) but avoids its cancellation for negative x.
  mlir::MlirOp self_neg_alpha = stablehlo::Mul(input_op, k_neg_alpha);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp erfc_out, BuildErfcForTargetDtypeShlo(
                                                 self_neg_alpha, output_dtype));
  mlir::MlirOp cdf = stablehlo::Mul(k_0_5, erfc_out);

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
    result_op = BuildGeluNoneShlo(computation_input_op, output_dtype);
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
    TT_ASSIGN_OR_RETURN(result_op_val, BuildGeluBackwardGradInputNoneShlo(
                                           computation_grad_output_op,
                                           computation_input_op, output_dtype));
  } else {  // approximation_type == "tanh"
    TT_ASSIGN_OR_RETURN(result_op_val,
                        BuildGeluBackwardGradInputApproximateShlo(
                            computation_grad_output_op, computation_input_op));
  }

  return mlir::stablehlo::ConvertElementType(result_op_val, output_dtype);
}

}  // namespace torch_tpu
