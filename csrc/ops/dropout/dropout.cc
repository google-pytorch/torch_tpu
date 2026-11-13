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
#include "csrc/common/aten_utils.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/uniform/uniform.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

absl::StatusOr<MlirOpResults<2>> BuildDropoutTrainShlo(
    mlir::MlirOp rng_input_state, mlir::MlirOp input, double p,
    DropoutProbabilityType prob_type, DropoutMaskType mask_type) {
  ABSL_VLOG(1) << "[BuildDropoutTrainShlo] input: "
               << mlir::debugString(input.getValue()) << ", p: " << p;
  ABSL_CHECK(p >= 0.0 && p <= 1.0)  // CRASH_OK=Caller validates p.
      << "expected p to be in the range [0, 1], got " << p;

  const mlir::ElementType mask_elem_type =
      (mask_type == DropoutMaskType::kUInt8) ? mlir::ElementType::UI8
                                             : mlir::ElementType::PRED;

  const bool keep_all = (prob_type == DropoutProbabilityType::kDropProbability)
                            ? (p <= 0.0)
                            : (p >= 1.0);
  const bool drop_all = (prob_type == DropoutProbabilityType::kDropProbability)
                            ? (p >= 1.0)
                            : (p <= 0.0);

  if (keep_all) {
    auto mask_op = MakeConstantLike(input, true, mask_elem_type);
    return {{input, mask_op}};
  }
  if (drop_all) {
    auto zero_const = MakeConstantLike(input, 0.0);
    auto mask_op = MakeConstantLike(input, false, mask_elem_type);
    return {{zero_const, mask_op}};
  }

  mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  TT_ASSIGN_OR_RETURN(
      const mlir::ElementType elem_type,
      ConvertTo<mlir::ElementType>(input_type.getElementType()));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_elem_type,
                      InferComputationDtype(elem_type));
  const mlir::ElementType rand_elem_type = RealComponentOf(compute_elem_type);
  const mlir::Type rand_mlir_type =
      mlir::getElementType(input.getContext(), rand_elem_type);
  const mlir::RankedTensorType rand_type = input_type.clone(rand_mlir_type);
  TT_ASSIGN_OR_RETURN(
      auto rand_op, BuildUniformShlo(rng_input_state, /*from=*/0.0, /*to=*/1.0,
                                     rand_type, input));

  auto p_const = MakeConstantLike(rand_op, p);
  const auto comparison_dir =
      (prob_type == DropoutProbabilityType::kKeepProbability)
          ? mlir::stablehlo::ComparisonDirection::LT
          : mlir::stablehlo::ComparisonDirection::GE;
  auto mask_op = mlir::stablehlo::Compare(rand_op, p_const, comparison_dir);

  TT_ASSIGN_OR_RETURN(mlir::MlirOp scaled_input,
                      CastIfNeeded(input, compute_elem_type));

  const double scale = (prob_type == DropoutProbabilityType::kKeepProbability)
                           ? (1.0 / p)
                           : (1.0 / (1.0 - p));
  auto scale_const = MakeConstantLike(scaled_input, scale);
  auto scaled = mlir::stablehlo::Mul(scaled_input, scale_const);
  auto zero_const = MakeConstantLike(scaled, 0.0);
  auto output = mlir::stablehlo::Select(mask_op, scaled, zero_const);

  if (compute_elem_type != elem_type) {
    TT_ASSIGN_OR_RETURN(output, CastIfNeeded(output, elem_type));
  }
  if (mask_type == DropoutMaskType::kUInt8) {
    mask_op =
        mlir::stablehlo::ConvertElementType(mask_op, mlir::ElementType::UI8);
  }

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

  if (!IsBooleanType(GetTensorTypeOrDie(mask))) {
    auto zero_mask = MakeConstantLike(mask, 0);
    mask = mlir::stablehlo::Compare(mask, zero_mask,
                                    mlir::stablehlo::ComparisonDirection::NE);
  }

  TT_ASSIGN_OR_RETURN(mlir::MlirOp mask_bcst,
                      BroadcastIfNeeded(mask, GetDimensions(grad_output)));

  TT_ASSIGN_OR_RETURN(const mlir::ElementType elem_type,
                      ConvertTo<mlir::ElementType>(
                          GetTensorTypeOrDie(grad_output).getElementType()));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_elem_type,
                      InferComputationDtype(elem_type));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp scaled_grad,
                      CastIfNeeded(grad_output, compute_elem_type));

  auto scale_const = MakeConstantLike(scaled_grad, scale);
  auto scaled = mlir::stablehlo::Mul(scaled_grad, scale_const);
  auto zero_const = MakeConstantLike(scaled, 0.0);
  auto output = mlir::stablehlo::Select(mask_bcst, scaled, zero_const);

  if (compute_elem_type != elem_type) {
    TT_ASSIGN_OR_RETURN(output, CastIfNeeded(output, elem_type));
  }
  return output;
}

}  // namespace torch_tpu
