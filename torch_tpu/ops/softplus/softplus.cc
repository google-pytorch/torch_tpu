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

#include "torch_tpu/ops/softplus/softplus.h"

#include <utility>

#include "absl/status/statusor.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildSoftplusShlo(mlir::MlirOp input,
                                               mlir::MlirOp beta,
                                               mlir::MlirOp threshold) {
  // result = 1/beta * log(1 + exp(beta * input)), if beta * input <= threshold
  // result = input, if beta * input > threshold
  auto beta_input = stablehlo::Mul(beta, input);
  auto exp_beta_input = stablehlo::Exp(beta_input);
  auto one = MakeConstantLike(exp_beta_input, 1.0);
  auto one_plus_exp = stablehlo::Add(one, exp_beta_input);
  auto log_one_plus_exp = stablehlo::Log(one_plus_exp);
  auto softplus_result = stablehlo::Div(log_one_plus_exp, beta);

  auto condition = stablehlo::Compare(beta_input, threshold,
                                      stablehlo::ComparisonDirection::LE);
  return stablehlo::Select(condition, softplus_result, input);
}

absl::StatusOr<mlir::MlirOp> BuildSoftplusBackwardShlo(mlir::MlirOp grad_output,
                                                       mlir::MlirOp input,
                                                       mlir::MlirOp beta,
                                                       mlir::MlirOp threshold) {
  // grad_input = grad_output * exp(beta * input) / (1 + exp(beta * input)),
  // if beta * input <= threshold
  // grad_input = grad_output, if beta * input > threshold
  auto beta_input = stablehlo::Mul(beta, input);
  auto exp_beta_input = stablehlo::Exp(beta_input);
  auto one = MakeConstantLike(exp_beta_input, 1.0);
  auto one_plus_exp = stablehlo::Add(one, exp_beta_input);
  auto softplus_grad = stablehlo::Div(exp_beta_input, one_plus_exp);
  auto grad_input_result = stablehlo::Mul(grad_output, softplus_grad);

  auto condition = stablehlo::Compare(beta_input, threshold,
                                      stablehlo::ComparisonDirection::LE);
  return stablehlo::Select(condition, grad_input_result, grad_output);
}

}  // namespace torch_tpu
