/*
 * Copyright 2026 Google LLC
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

#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

absl::StatusOr<mlir::MlirOp> BuildLdexpShlo(mlir::MlirOp input_op,
                                            mlir::MlirOp other_op) {
  TT_ASSIGN_OR_RETURN((auto [input, other]),
                      ApplyBroadcastIfNeeded(input_op, other_op));
  auto input_elty = GetTensorTypeOrDie(input).getElementType();
  auto other_elty = GetTensorTypeOrDie(other).getElementType();

  if (mlir::isa<mlir::ComplexType>(input_elty) ||
      mlir::isa<mlir::ComplexType>(other_elty)) {
    return TT_ERROR(error::kInvalidArgument)
           << absl::StrCat("ldexp not supported for input types (",
                           mlir::debugString(input_elty), ", ",
                           mlir::debugString(other_elty), ")");
  }

  mlir::MlirBuilder& builder = input.getBuilder();

  // Integer types will get cast to f32. Also cast float types with less than 32
  // bits to f32 for the intermediate math to prevent catastrophic precision
  // loss.
  if (!input_elty.isFloat() ||
      mlir::cast<mlir::FloatType>(input_elty).getWidth() < 32) {
    input_elty = builder.getOpBuilder().getF32Type();
    input = mlir::stablehlo::ConvertElementType(input, input_elty);
  }

  if (other_elty != input_elty) {
    other = mlir::stablehlo::ConvertElementType(other, input_elty);
  }

  // Clamp exponent to avoid over/underflow on extreme values.
  // Use 2*1000 for f64 and 2*150 for f32 since we split the exponent later as
  // safe values.
  double clamp_bound = input_elty.isF64() ? 2000.0 : 300.0;
  auto min_clamp = MakeConstantLike(other, -clamp_bound, input_elty);
  auto max_clamp = MakeConstantLike(other, clamp_bound, input_elty);
  auto clamped_exp = mlir::stablehlo::Clamp(min_clamp, other, max_clamp);

  // Split the exponent to prevent intermediate overflow when computing Pow(2.0,
  // e) directly.
  auto two_float = MakeConstantLike(clamped_exp, 2.0, input_elty);
  auto exp_half_raw = mlir::stablehlo::Div(clamped_exp, two_float);
  auto e1 = mlir::stablehlo::Floor(exp_half_raw);
  auto e2 = mlir::stablehlo::Subtract(clamped_exp, e1);

  auto pow_e1 = mlir::stablehlo::Pow(two_float, e1);
  auto pow_e2 = mlir::stablehlo::Pow(two_float, e2);

  auto res_step1 = mlir::stablehlo::Mul(input, pow_e1);
  auto result = mlir::stablehlo::Mul(res_step1, pow_e2);

  auto is_nan = mlir::stablehlo::Compare(
      input, input, mlir::stablehlo::ComparisonDirection::NE);
  auto is_inf = mlir::chlo::IsInf(input);
  auto zero_float = MakeConstantLike(input, 0.0, input_elty);
  auto is_zero = mlir::stablehlo::Compare(
      input, zero_float, mlir::stablehlo::ComparisonDirection::EQ);

  auto is_nan_or_inf = mlir::stablehlo::Or(is_nan, is_inf);
  auto is_special = mlir::stablehlo::Or(is_nan_or_inf, is_zero);

  return mlir::stablehlo::Select(is_special, input, result);
}

}  // namespace torch_tpu
