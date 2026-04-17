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

#include "torch_tpu/ops/unary.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "ATen/OpMathType.h"
#include "c10/core/ScalarType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {

absl::StatusOr<mlir::MlirOp> LogN(mlir::MlirOp input_op, int32_t n,
                                  mlir::ElementType default_dtype) {
  at::ScalarType default_scalar_type = ConvertTo<at::ScalarType>(default_dtype);
  // Similarly to CUDA, Half and BFloat16 inputs are promoted to to Float for
  // the actual mathematical computation.
  at::ScalarType computation_scalar_type =
      at::toOpMathType(default_scalar_type);

  TT_ASSIGN_OR_RETURN(mlir::ElementType computation_type,
                      ConvertTo<mlir::ElementType>(computation_scalar_type));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp computation_input_op,
                      ConvertIfInteger(input_op, computation_type));
  computation_input_op = mlir::stablehlo::ConvertElementType(
      computation_input_op, computation_type);

  mlir::MlirBuilder& builder = input_op.getBuilder();

  // Defined as LogN(x) = log(x) / log(N):
  mlir::MlirOp log_op = stablehlo::Log(computation_input_op);

  mlir::MlirOp log_of_n;
  if (computation_scalar_type == at::ScalarType::ComplexDouble) {
    log_of_n = MakeScalarConstant(
        builder, std::complex<double>(std::log(n), 0.0), computation_type);
  } else if (computation_scalar_type == at::ScalarType::ComplexFloat) {
    log_of_n = MakeScalarConstant(
        builder, std::complex<float>(std::log(n), 0.0f), computation_type);
  } else if (computation_scalar_type == at::ScalarType::Double) {
    log_of_n = MakeScalarConstant(builder, static_cast<double>(std::log(n)),
                                  computation_type);
  } else {
    log_of_n = MakeScalarConstant(builder, static_cast<float>(std::log(n)),
                                  computation_type);
  }

  mlir::MlirOp result_op = mlir::chlo::BroadcastDiv(log_op, log_of_n);

  return mlir::stablehlo::ConvertElementType(result_op, default_dtype);
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildAbsShlo(mlir::MlirOp input) {
  // Unfortunately shlo.abs() works only on signed ints and floats; hence, this
  // shortcut for unsigned ints.
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  mlir::Type element_type = input_type.getElementType();
  if (element_type.isUnsignedInteger()) {
    return input;
  }

  if (llvm::isa<mlir::ComplexType>(element_type)) {
    // For complex numbers, abs(z) = sqrt(real(z)^2 + imag(z)^2).
    // To avoid overflow and underflow, we use:
    //   abs(z) = max(|x|, |y|) * sqrt(1 + (min(|x|, |y|) / max(|x|, |y|))^2)
    // Similarly to CUDA, we promote to float32 for the computation.
    TT_ASSIGN_OR_RETURN(mlir::ElementType f32_type,
                        ConvertTo<mlir::ElementType>(at::kFloat));
    auto x = mlir::stablehlo::Real(input);
    auto y = mlir::stablehlo::Imag(input);

    x = mlir::stablehlo::ConvertElementType(x, f32_type);
    y = mlir::stablehlo::ConvertElementType(y, f32_type);

    auto abs_x = mlir::stablehlo::Abs(x);
    auto abs_y = mlir::stablehlo::Abs(y);

    auto max_abs = mlir::stablehlo::Max(abs_x, abs_y);
    auto min_abs = mlir::stablehlo::Min(abs_x, abs_y);

    // If max_abs is 0, then the result is 0.
    // We avoid division by zero by using max(max_abs, epsilon).
    auto epsilon = MakeScalarConstant(input.getBuilder(), 1e-30, f32_type);
    TT_ASSIGN_OR_RETURN((auto [max_abs_b, epsilon_b]),
                        ApplyBroadcastIfNeeded(max_abs, epsilon));
    auto safe_max = mlir::stablehlo::Max(max_abs_b, epsilon_b);

    auto ratio = mlir::stablehlo::Div(min_abs, safe_max);
    auto ratio_sq = mlir::stablehlo::Mul(ratio, ratio);
    auto one = MakeScalarConstant(input.getBuilder(), 1.0, f32_type);
    TT_ASSIGN_OR_RETURN((auto [one_b, ratio_sq_b]),
                        ApplyBroadcastIfNeeded(one, ratio_sq));
    auto sum_sq = mlir::stablehlo::Add(one_b, ratio_sq_b);
    auto sqrt_sum_sq = mlir::stablehlo::Sqrt(sum_sq);
    auto res = mlir::stablehlo::Mul(max_abs, sqrt_sum_sq);

    // PyTorch returns real result for abs(complex).
    // The output type should be the base float type.
    mlir::Type base_type =
        llvm::cast<mlir::ComplexType>(element_type).getElementType();
    return mlir::stablehlo::ConvertElementType(res, base_type);
  }

  auto res = mlir::stablehlo::Abs(input);

  return res;
}

absl::StatusOr<mlir::MlirOp> BuildConjPhysicalShlo(mlir::MlirOp input_op) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  TT_ASSIGN_OR_RETURN(c10::ScalarType dtype,
                      ConvertTo<c10::ScalarType>(input_type.getElementType()));
  if (!c10::isComplexType(dtype)) {
    return input_op;
  }
  auto real = mlir::stablehlo::Real(input_op);
  auto imag = mlir::stablehlo::Imag(input_op);
  auto neg_imag = mlir::stablehlo::Neg(imag);
  return mlir::stablehlo::Complex(real, neg_imag);
}

absl::StatusOr<mlir::MlirOp> BuildReciprocalShlo(
    mlir::MlirOp input_op, mlir::ElementType default_mlir_type) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp converted_input_op,
                      ConvertIfInteger(input_op, default_mlir_type));
  mlir::MlirOp one_scalar = MakeConstantLike(converted_input_op, 1.0);
  mlir::MlirOp result_op = stablehlo::Div(one_scalar, converted_input_op);
  return result_op;
}

absl::StatusOr<mlir::MlirOp> BuildReluShlo(mlir::MlirOp input_op) {
  mlir::MlirOp zero_scalar = MakeConstantLike(input_op, 0.0);
  mlir::MlirOp result_op = stablehlo::Max(input_op, zero_scalar);
  return result_op;
}

absl::StatusOr<mlir::MlirOp> BuildSiluShlo(mlir::MlirOp input_op) {
  // Only defined for floating point types
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);

  TT_ASSIGN_OR_RETURN(const at::ScalarType scalar_type,
                      ConvertTo<c10::ScalarType>(input_type.getElementType()));

  TT_RET_CHECK(IsFloatType(input_type), error::kInvalidArgument)
      << "expected the input dtype to be floating point, got "
      << ToString(scalar_type);

  // Defined as silu(x) = x * sigmoid(x):
  mlir::MlirOp sigmoid_op = stablehlo::Logistic(input_op);
  mlir::MlirOp result_op = stablehlo::Mul(input_op, sigmoid_op);
  return result_op;
}

absl::StatusOr<mlir::MlirOp> BuildTruncShlo(mlir::MlirOp input_op) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);

  // For integer inputs, follows the array-api convention of
  // returning a copy of the input tensor.
  if (input_type.getElementType().isInteger()) {
    return input_op;
  }

  // For floating-point division, we floor the absolute value of the result
  // and multiply by the sign of the original result to achieve truncation
  // towards zero.
  auto abs = stablehlo::Abs(input_op);
  auto floor = stablehlo::Floor(abs);
  auto sign = stablehlo::Sign(input_op);
  mlir::MlirOp result_op = stablehlo::Mul(floor, sign);
  return result_op;
}

// TODO(b/442665129): Test the lift_fresh op when torch.compile is ready.
absl::StatusOr<mlir::MlirOp> BuildLiftFreshShlo(mlir::MlirOp input_op) {
  return input_op;
}

absl::StatusOr<mlir::MlirOp> BuildLogShlo(mlir::MlirOp input_op,
                                          mlir::ElementType default_dtype) {
  at::ScalarType default_scalar_type = ConvertTo<at::ScalarType>(default_dtype);
  // Similarly to CUDA, Half and BFloat16 inputs are promoted to to Float for
  // the actual mathematical computation.
  at::ScalarType computation_scalar_type =
      at::toOpMathType(default_scalar_type);

  TT_ASSIGN_OR_RETURN(mlir::ElementType computation_type,
                      ConvertTo<mlir::ElementType>(computation_scalar_type));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp computation_input_op,
                      ConvertIfInteger(input_op, computation_type));
  computation_input_op = mlir::stablehlo::ConvertElementType(
      computation_input_op, computation_type);

  mlir::MlirOp result_op = mlir::stablehlo::Log(computation_input_op);

  return mlir::stablehlo::ConvertElementType(result_op, default_dtype);
}

absl::StatusOr<mlir::MlirOp> BuildLog1pShlo(mlir::MlirOp input_op,
                                            mlir::ElementType default_dtype) {
  at::ScalarType default_scalar_type = ConvertTo<at::ScalarType>(default_dtype);
  // Similarly to CUDA, Half and BFloat16 inputs are promoted to to Float for
  // the actual mathematical computation.
  at::ScalarType computation_scalar_type =
      at::toOpMathType(default_scalar_type);

  TT_ASSIGN_OR_RETURN(mlir::ElementType computation_type,
                      ConvertTo<mlir::ElementType>(computation_scalar_type));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp computation_input_op,
                      ConvertIfInteger(input_op, computation_type));
  computation_input_op = mlir::stablehlo::ConvertElementType(
      computation_input_op, computation_type);

  mlir::MlirOp result_op = mlir::stablehlo::Log1p(computation_input_op);

  return mlir::stablehlo::ConvertElementType(result_op, default_dtype);
}

absl::StatusOr<mlir::MlirOp> BuildLog2Shlo(
    mlir::MlirOp input_op, mlir::ElementType default_mlir_type) {
  return LogN(input_op, 2, default_mlir_type);
}

absl::StatusOr<mlir::MlirOp> BuildLog10Shlo(
    mlir::MlirOp input_op, mlir::ElementType default_mlir_type) {
  return LogN(input_op, 10, default_mlir_type);
}

absl::StatusOr<mlir::MlirOp> BuildSignShlo(mlir::MlirOp input) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  mlir::Type element_type = input_type.getElementType();

  // mlir::stablehlo::Sign requires non-boolean signless integer type.
  auto compute_type = element_type;
  if (element_type.isUnsignedInteger()) {
    // Convert to signless if unsigned.
    compute_type = mlir::IntegerType::get(&input.getContext(),
                                          element_type.getIntOrFloatBitWidth(),
                                          mlir::IntegerType::Signless);
  } else if (element_type.isInteger(1)) {
    // Convert to i8 if boolean.
    compute_type = mlir::IntegerType::get(&input.getContext(), 8,
                                          mlir::IntegerType::Signless);
  }
  if (compute_type != element_type) {
    input = mlir::stablehlo::ConvertElementType(input, compute_type);
    auto output = mlir::stablehlo::Sign(input);
    return mlir::stablehlo::ConvertElementType(output, element_type);
  }
  return mlir::stablehlo::Sign(input);
}

absl::StatusOr<mlir::MlirOp> BuildErfcShlo(mlir::MlirOp input,
                                           mlir::ElementType out_mlir_type) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp op, ConvertIfInteger(input, out_mlir_type));
  return mlir::chlo::Erfc(op);
};

absl::StatusOr<mlir::MlirOp> BuildFracShlo(mlir::MlirOp input) {
  mlir::MlirOp floor_val = mlir::stablehlo::Floor(input);
  return mlir::stablehlo::Subtract(input, floor_val);
}

}  // namespace torch_tpu
