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

#include "ATen/OpMathType.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/op_builder_utils.h"

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
  mlir::MlirOp value_n = MakeScalarConstant(builder, n, computation_type);
  mlir::MlirOp log_of_n = stablehlo::Log(value_n);
  mlir::MlirOp result_op = mlir::chlo::BroadcastDiv(log_op, log_of_n);

  return mlir::stablehlo::ConvertElementType(result_op, default_dtype);
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildAbsShlo(mlir::MlirOp input) {
  // Unfortunately shlo.abs() works only on signed ints and floats; hence, this
  // shortcut for unsigned ints and booleans.
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  mlir::Type element_type = input_type.getElementType();
  if (element_type.isUnsignedInteger() || element_type.isInteger(/*width=*/1)) {
    return input;
  }

  if (llvm::isa<mlir::ComplexType>(element_type)) {
    // For complex numbers, abs(z) = sqrt(real(z)^2 + imag(z)^2).
    // To avoid overflow and underflow, we use:
    //   abs(z) = max(|x|, |y|) * sqrt(1 + (min(|x|, |y|) / max(|x|, |y|))^2)
    mlir::Type base_type =
        llvm::cast<mlir::ComplexType>(element_type).getElementType();
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=complex dtypes with < 32-bit base
                   // (like complex32) are rejected at tensor creation.
        base_type.getIntOrFloatBitWidth() >= 32, error::kInvalidArgument)
        << "expected complex base type bit width to be at least 32, got "
        << base_type.getIntOrFloatBitWidth();
    TT_ASSIGN_OR_RETURN(mlir::ElementType compute_type,
                        ConvertTo<mlir::ElementType>(base_type));

    auto x = mlir::stablehlo::Real(input);
    auto y = mlir::stablehlo::Imag(input);

    x = mlir::stablehlo::ConvertElementType(x, compute_type);
    y = mlir::stablehlo::ConvertElementType(y, compute_type);

    auto abs_x = mlir::stablehlo::Abs(x);
    auto abs_y = mlir::stablehlo::Abs(y);

    auto max_abs = mlir::stablehlo::Max(abs_x, abs_y);
    auto min_abs = mlir::stablehlo::Min(abs_x, abs_y);

    // We avoid division by zero by replacing the divisor with 1.0 if max_abs is
    // 0.0.
    auto zero = MakeScalarConstant(input.getBuilder(), 0.0, compute_type);
    auto one = MakeScalarConstant(input.getBuilder(), 1.0, compute_type);
    TT_ASSIGN_OR_RETURN((auto [max_abs_b, zero_b]),
                        ApplyBroadcastIfNeeded(max_abs, zero));
    auto is_zero = mlir::stablehlo::Compare(
        max_abs_b, zero_b, mlir::stablehlo::ComparisonDirection::EQ);
    TT_ASSIGN_OR_RETURN((auto [one_b, max_abs_b2]),
                        ApplyBroadcastIfNeeded(one, max_abs));
    auto safe_max = mlir::stablehlo::Select(is_zero, one_b, max_abs_b2);

    auto ratio = mlir::stablehlo::Div(min_abs, safe_max);
    auto ratio_sq = mlir::stablehlo::Mul(ratio, ratio);
    TT_ASSIGN_OR_RETURN((auto [one_b_ratio, ratio_sq_b]),
                        ApplyBroadcastIfNeeded(one, ratio_sq));
    auto sum_sq = mlir::stablehlo::Add(one_b_ratio, ratio_sq_b);
    auto sqrt_sum_sq = mlir::stablehlo::Sqrt(sum_sq);
    auto res = mlir::stablehlo::Mul(max_abs, sqrt_sum_sq);

    // PyTorch returns real result for abs(complex).
    // The output type should be the base float type.
    return mlir::stablehlo::ConvertElementType(res, base_type);
  }

  auto res = mlir::stablehlo::Abs(input);

  return res;
}

absl::StatusOr<mlir::MlirOp> BuildCeilShlo(mlir::MlirOp input_op) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  if (input_type.getElementType().isInteger()) {
    return input_op;
  }
  return mlir::stablehlo::Ceil(input_op);
}

absl::StatusOr<mlir::MlirOp> BuildFloorShlo(mlir::MlirOp input_op) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  if (input_type.getElementType().isInteger()) {
    return input_op;
  }
  return mlir::stablehlo::Floor(input_op);
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

absl::StatusOr<mlir::MlirOp> BuildExp2Shlo(
    mlir::MlirOp input_op, mlir::ElementType default_mlir_type) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp op,
                      ConvertIfInteger(input_op, default_mlir_type));
  mlir::MlirBuilder& builder = op.getBuilder();
  mlir::RankedTensorType type = GetTensorTypeOrDie(op);
  mlir::MlirOp value_2 =
      MakeScalarConstant(builder, 2.0, type.getElementType());
  mlir::MlirOp ln2 = mlir::stablehlo::Log(value_2);
  TT_ASSIGN_OR_RETURN((auto [ln2_b, op_b]), ApplyBroadcastIfNeeded(ln2, op));
  mlir::MlirOp mul_op = mlir::stablehlo::Mul(ln2_b, op_b);
  mlir::MlirOp result_op = mlir::stablehlo::Exp(mul_op);
  return mlir::stablehlo::ConvertElementType(result_op, default_mlir_type);
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
  // torch.sign on complex tensors is defined as z / |z|, identical to
  // torch.sgn.
  return BuildSgnShlo(input);
}

absl::StatusOr<mlir::MlirOp> BuildSgnShlo(mlir::MlirOp input) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  mlir::Type element_type = input_type.getElementType();
  TT_ASSIGN_OR_RETURN(c10::ScalarType dtype,
                      ConvertTo<c10::ScalarType>(element_type));

  if (c10::isComplexType(dtype)) {
    // For complex numbers, sgn(z) = z / |z| if z != 0, else 0.
    // stablehlo::Sign can be numerically unstable for small values.
    TT_ASSIGN_OR_RETURN(mlir::MlirOp abs_op, BuildAbsShlo(input));
    // Handle division by zero: if abs_op is 0, we want result to be 0.
    // We can use a small epsilon or just rely on stablehlo::Div behavior if
    // it's defined. Actually, PyTorch's sgn(0) is 0. stablehlo::Div(0, 0) is
    // usually 0 or NaN depending on implementation. To be safe and match Sign
    // behavior, we can use a mask.
    mlir::MlirOp zero_complex = MakeConstantLike(input, 0.0);
    mlir::MlirOp zero_real = MakeConstantLike(abs_op, 0.0);
    mlir::MlirOp is_zero = mlir::stablehlo::Compare(
        abs_op, zero_real, mlir::stablehlo::ComparisonDirection::EQ);

    // We use a safe divisor to avoid NaN.
    mlir::MlirOp one_real = MakeConstantLike(abs_op, 1.0);
    mlir::MlirOp safe_abs = mlir::stablehlo::Select(is_zero, one_real, abs_op);

    // stablehlo::Div requires same types for all operands.
    // Convert safe_abs (real) to complex.
    mlir::MlirOp zero_imag = MakeConstantLike(safe_abs, 0.0);
    mlir::MlirOp safe_abs_complex =
        mlir::stablehlo::Complex(safe_abs, zero_imag);

    mlir::MlirOp div_res = mlir::stablehlo::Div(input, safe_abs_complex);

    return mlir::stablehlo::Select(is_zero, zero_complex, div_res);
  }

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
