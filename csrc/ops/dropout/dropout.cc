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
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

absl::StatusOr<MlirOpResults<2>> BuildDropoutTrainShlo(
    mlir::MlirOp rng_input_state, mlir::MlirOp input, double p) {
  ABSL_VLOG(1) << "[BuildDropoutTrainShlo] input: "
               << mlir::debugString(input.getValue()) << ", p: " << p;
  ABSL_CHECK(p >= 0.0 && p <= 1.0)  // CRASH_OK=Caller validates p.
      << "expected p to be in the range [0, 1], got " << p;

  auto& builder = input.getBuilder();
  auto& op_builder = builder.getOpBuilder();

  if (p <= 0.0) {
    auto mask_op = MakeConstantLike(input, true, op_builder.getI1Type());
    return {{input, mask_op}};
  }
  if (p >= 1.0) {
    auto zero_const = MakeConstantLike(input, 0.0);
    auto mask_op = MakeConstantLike(input, false, op_builder.getI1Type());
    return {{zero_const, mask_op}};
  }

  mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  mlir::RankedTensorType rand_type = input_type;
  if (auto complex_type =
          mlir::dyn_cast<mlir::ComplexType>(input_type.getElementType())) {
    rand_type = input_type.clone(complex_type.getElementType());
  }
  TT_ASSIGN_OR_RETURN(
      auto rand_op, BuildUniformShlo(rng_input_state, /*from=*/0.0, /*to=*/1.0,
                                     rand_type, input));

  auto p_const = MakeConstantLike(rand_op, p);
  auto mask_op = mlir::stablehlo::Compare(
      rand_op, p_const, mlir::stablehlo::ComparisonDirection::GE);
  auto zero_const = MakeConstantLike(input, 0.0);
  auto masked_input_op = mlir::stablehlo::Select(mask_op, input, zero_const);

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

  // Treat non-boolean (uint8) masks as binary (mask != 0) rather than integer
  // multipliers (intentional divergence from CUDA masked_scale_kernel;
  // fused_dropout masks are 0/1).
  if (!IsBooleanType(GetTensorTypeOrDie(mask))) {
    mlir::MlirOp zero_mask = MakeConstantLike(mask, 0);
    mask = mlir::stablehlo::Compare(mask, zero_mask,
                                    mlir::stablehlo::ComparisonDirection::NE);
  }

  TT_ASSIGN_OR_RETURN((auto [grad_output_bcst, mask_bcst]),
                      ApplyBroadcastIfNeeded(grad_output, mask));

  TT_ASSIGN_OR_RETURN(
      const mlir::ElementType elem_type,
      ConvertTo<mlir::ElementType>(
          GetTensorTypeOrDie(grad_output_bcst).getElementType()));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_elem_type,
                      InferComputationDtype(elem_type));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp cast_grad,
                      CastIfNeeded(grad_output_bcst, compute_elem_type));

  // Scale real and imaginary parts separately for complex inputs so inf * 0.0
  // in a full complex multiply with (scale, 0.0) does not produce NaN.
  mlir::MlirOp scaled;
  if (IsComplexType(GetTensorTypeOrDie(cast_grad))) {
    mlir::MlirOp real = mlir::stablehlo::Real(cast_grad);
    mlir::MlirOp imag = mlir::stablehlo::Imag(cast_grad);
    mlir::MlirOp real_scale = MakeConstantLike(real, scale);
    mlir::MlirOp scaled_real = mlir::stablehlo::Mul(real, real_scale);
    mlir::MlirOp scaled_imag = mlir::stablehlo::Mul(imag, real_scale);
    scaled = mlir::stablehlo::Complex(scaled_real, scaled_imag);
  } else {
    mlir::MlirOp scale_const = MakeConstantLike(cast_grad, scale);
    scaled = mlir::stablehlo::Mul(cast_grad, scale_const);
  }
  // Select 0.0 for masked-out elements even when scale is inf (e.g.
  // fused_dropout backward at p=0.0) or grad_output is inf/nan, intentionally
  // avoiding CUDA's 0 * inf -> nan.
  mlir::MlirOp zero_const = MakeConstantLike(scaled, 0.0);
  mlir::MlirOp output = mlir::stablehlo::Select(mask_bcst, scaled, zero_const);

  if (compute_elem_type != elem_type) {
    TT_ASSIGN_OR_RETURN(output, CastIfNeeded(output, elem_type));
  }
  return output;
}

}  // namespace torch_tpu
