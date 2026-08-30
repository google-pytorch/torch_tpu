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

#include "csrc/ops/dropout/dropout.h"

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/uniform/uniform.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

absl::StatusOr<MlirOpResults<2>> BuildDropoutTrainShlo(
    mlir::MlirOp rng_input_state, mlir::MlirOp input, double p) {
  ABSL_VLOG(1) << "[BuildDropoutTrainShlo] input: "
               << mlir::debugString(input.getValue()) << ", p: " << p;
  ABSL_CHECK(p > 0 && p < 1.0)  // CRASH_OK=Caller validates p.
      << "expected p to be in the exclusive range (0, 1), got " << p;
  mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  TT_ASSIGN_OR_RETURN(
      auto rand_op,
      BuildUniformShlo(rng_input_state, /*from=*/0.0, /*to=*/1.0,
                       input_type.getShape(), GetElementTypeOrDie(input)));

  auto p_const = MakeConstantLike(rand_op, p);
  auto mask_op = mlir::stablehlo::Compare(
      rand_op, p_const, mlir::stablehlo::ComparisonDirection::GE);
  auto zero_const = MakeConstantLike(input, 0.0);
  auto masked_input_op = mlir::stablehlo::Select(mask_op, input, zero_const);

  // p is guaranteed to be between 0 and 1 exclusive
  // via early returns in the caller for p == 0 and p >= 1.
  double scale = 1.0 / (1.0 - p);
  auto scale_const = MakeConstantLike(input, scale);
  auto output = mlir::stablehlo::Mul(masked_input_op, scale_const);

  return {{output, mask_op}};
}

absl::StatusOr<MlirOpResults<1>> BuildDropoutBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp mask, double scale) {
  ABSL_VLOG(1) << "[BuildDropoutBackwardShlo] grad_output: "
               << mlir::debugString(grad_output.getValue())
               << ", mask: " << mlir::debugString(mask.getValue())
               << ", scale: " << scale;

  mlir::MlirBuilder& builder = grad_output.getBuilder();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();

  // Convert grad_output tensor to float type if it is not already a float or
  // complex type.
  const mlir::RankedTensorType grad_output_type =
      GetTensorTypeOrDie(grad_output);
  if (!IsFloatType(grad_output_type) && !IsComplexType(grad_output_type)) {
    grad_output = mlir::stablehlo::ConvertElementType(grad_output,
                                                      op_builder.getF32Type());
  }

  auto zero_const = MakeConstantLike(grad_output, 0.0);
  auto masked_grad_op = mlir::stablehlo::Select(mask, grad_output, zero_const);
  auto scale_const = MakeConstantLike(masked_grad_op, scale);
  return mlir::stablehlo::Mul(masked_grad_op, scale_const);
}

}  // namespace torch_tpu
