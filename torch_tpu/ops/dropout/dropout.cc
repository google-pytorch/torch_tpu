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

#include "torch_tpu/ops/dropout/dropout.h"

#include <limits>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

absl::StatusOr<MlirOpResults<3>> BuildDropoutTrainShlo(
    mlir::MlirOp rng_input_state, mlir::MlirOp input, double p) {
  ABSL_VLOG(1) << "[BuildDropoutTrainShlo] input: "
               << mlir::debugString(input.getValue()) << ", p: " << p;
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Error is caught in `AtenDropout()`
                 // function (caller).
      p > 0 && p < 1.0, error::kInvalidArgument)
      << "expected p to be in the exclusive range (0, 1), got " << p;
  mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const auto rng_input_state_type = GetTensorTypeOrDie(rng_input_state);
  mlir::MlirBuilder& builder = input.getBuilder();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();

  // Call rng bit generator,
  auto input_type_float = input_type.clone(op_builder.getF64Type());
  auto input_type_uint64 =
      input_type.clone(op_builder.getIntegerType(64, false));

  auto algorithm = mlir::stablehlo::RngAlgorithmAttr::get(
      op_builder.getContext(), mlir::stablehlo::RngAlgorithm::PHILOX);
  auto rng_op = mlir::stablehlo::RngBitGeneratorOp::create(
      op_builder, input.getValue().getLoc(), rng_input_state_type,
      input_type_uint64, algorithm, rng_input_state.getValue());

  auto rng_output_state = mlir::MlirOp(builder, rng_op.getOutputState());
  mlir::MlirOp rng_output_op = mlir::MlirOp(builder, rng_op.getOutput());
  const mlir::RankedTensorType rng_output_op_type =
      GetTensorTypeOrDie(rng_output_op);
  ABSL_VLOG(1) << "[BuildDropoutTrainShlo] rng_output_state_type: "
               << mlir::debugString(rng_output_op_type);
  // Need to convert this tensor of u64s to a tensor of f64s with exponent bits
  // set to 1.
  // TODO(unda): extract this to a common util, probably part of some other rng
  // op, and make the random bit usage more efficient.

  // Clear the exponent and sign bits
  mlir::MlirOp clear_exponent_mask =
      MakeConstantLike(rng_output_op, 0x000FFFFFFFFFFFFFUL);
  rng_output_op = mlir::stablehlo::And(rng_output_op, clear_exponent_mask);
  // Set the exponent bits to 0 (f64 bias is 1023, so we store 0 + 1023 = 0x3FF)
  mlir::MlirOp set_exp_to_one_mask =
      MakeConstantLike(rng_output_op, 0x3FF0000000000000UL);
  rng_output_op = mlir::stablehlo::Or(rng_output_op, set_exp_to_one_mask);
  // Interpret as f64s
  mlir::MlirOp rand_op =
      mlir::stablehlo::BitcastConvert(input_type_float, rng_output_op);
  // Subtract 1.0
  mlir::MlirOp one_const = MakeConstantLike(rand_op, 1.0);
  rand_op = mlir::stablehlo::Subtract(rand_op, one_const);
  auto p_const = MakeConstantLike(rand_op, p);
  auto mask_op = mlir::stablehlo::Compare(
      rand_op, p_const, mlir::stablehlo::ComparisonDirection::GE);
  auto zero_const = MakeConstantLike(input, 0.0);
  auto masked_input_op = mlir::stablehlo::Select(mask_op, input, zero_const);

  // p is guaranteed to be between 0 and 1 exclusive
  // but we add 1.e-10 to avoid numerical issues when the denominator is tiny.
  double scale = 1.0 / (1.0 - p + std::numeric_limits<double>::epsilon());
  auto scale_const = MakeConstantLike(input, scale);
  auto output = mlir::stablehlo::Mul(masked_input_op, scale_const);

  return {{rng_output_state, output, mask_op}};
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
