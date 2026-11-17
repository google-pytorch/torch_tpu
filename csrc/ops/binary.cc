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

#include "csrc/ops/binary.h"

#include "absl/status/statusor.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildAddShlo(mlir::MlirOp lhs_op,
                                          mlir::MlirOp rhs_op) {
  TT_ASSIGN_OR_RETURN((auto [lhs, rhs]),
                      ApplyBroadcastIfNeeded(lhs_op, rhs_op));
  const mlir::RankedTensorType lhs_type = GetTensorTypeOrDie(lhs);
  if (IsBooleanType(lhs_type)) {
    // TODO(gleasonk): This shouldn't be needed, there is legacy code in XLA
    // that converts i1 add to xor which only applied to a now-deleted
    // compilation pipeline.
    // Attempting to remove, if this can't be removed we should push this logic
    // into the `stablehlo::Add` API.
    return stablehlo::Or(lhs, rhs);
  }
  return stablehlo::Add(lhs, rhs);
}

absl::StatusOr<mlir::MlirOp> BuildComplexShlo(mlir::MlirOp real_op,
                                              mlir::MlirOp imag_op) {
  const mlir::RankedTensorType real_type = GetTensorTypeOrDie(real_op);
  const mlir::RankedTensorType imag_type = GetTensorTypeOrDie(imag_op);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Input dtypes are checked before Mlir
                 // lowering.
      real_type.getElementType().isFloat(32) ||
          real_type.getElementType().isFloat(64),
      error::kInvalidArgument)
      << "real must be a float32 or float64 tensor, "
      << "got value type " << debugString(real_type);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Input dtypes are checked before Mlir
                 // lowering.
      imag_type.getElementType().isFloat(32) ||
          imag_type.getElementType().isFloat(64),
      error::kInvalidArgument)
      << "imag must be a float32 or float64 tensor, "
      << "got value type " << debugString(imag_type);
  TT_ASSIGN_OR_RETURN((auto [broadcasted_real_op, broadcasted_imag_op]),
                      ApplyBroadcastIfNeeded(real_op, imag_op),
                      _.SetPrepend() << "failed to broadcast real and imag: ");
  return stablehlo::Complex(broadcasted_real_op, broadcasted_imag_op);
}

absl::StatusOr<mlir::MlirOp> BuildPolarShlo(mlir::MlirOp abs_op,
                                            mlir::MlirOp angle_op) {
  const mlir::RankedTensorType abs_type = GetTensorTypeOrDie(abs_op);
  const mlir::RankedTensorType angle_type = GetTensorTypeOrDie(angle_op);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Input dtypes are checked before Mlir
                 // lowering.
      IsFloatType(abs_type), error::kInvalidArgument)
      << "abs must be a float32 or float64 tensor, "
      << "got value type " << debugString(abs_type);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Input dtypes are checked before Mlir
                 // lowering.
      IsFloatType(angle_type), error::kInvalidArgument)
      << "angle must be a float32 or float64 tensor, "
      << "got value type " << debugString(angle_type);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Inputs are promoted to a common dtype
                 // before reaching this function.
      abs_type.getElementType() == angle_type.getElementType(),
      error::kInvalidArgument)
      << "abs and angle must have the same element type, "
      << "got abs type " << debugString(abs_type) << " and angle type "
      << debugString(angle_type);
  TT_ASSIGN_OR_RETURN((auto [broadcasted_abs_op, broadcasted_angle_op]),
                      ApplyBroadcastIfNeeded(abs_op, angle_op),
                      _.SetPrepend() << "failed to broadcast abs and angle: ");

  mlir::MlirOp cos_op = stablehlo::Cosine(broadcasted_angle_op);
  mlir::MlirOp sin_op = stablehlo::Sine(broadcasted_angle_op);
  TT_ASSIGN_OR_RETURN(auto x_op, BuildMulShlo(broadcasted_abs_op, cos_op),
                      _ << " - could not build pointwise abs * cos(angle)");
  TT_ASSIGN_OR_RETURN(auto y_op, BuildMulShlo(broadcasted_abs_op, sin_op),
                      _ << " - could not build pointwise abs *sin(angle)");

  return stablehlo::Complex(x_op, y_op);
}

absl::StatusOr<mlir::MlirOp> BuildPowShlo(mlir::MlirOp base_op,
                                          mlir::MlirOp exp_op) {
  const mlir::RankedTensorType base_type = GetTensorTypeOrDie(base_op);
  const mlir::RankedTensorType exp_type = GetTensorTypeOrDie(exp_op);
  TT_ASSIGN_OR_RETURN((auto [broadcasted_base_op, broadcasted_exp_op]),
                      ApplyBroadcastIfNeeded(base_op, exp_op),
                      _.SetPrepend() << "failed to broadcast base and exp: ");

  // Treat true as 1 and false as 0 for boolean inputs.
  // Equivalent to base_op or not(exp_op).
  if (IsBooleanType(base_type) && IsBooleanType(exp_type)) {
    broadcasted_exp_op = stablehlo::Not(broadcasted_exp_op);
    return stablehlo::Or(broadcasted_base_op, broadcasted_exp_op);
  }

  // Implements PyTorch powi reference semantics for integer exponentiation with
  // negative exponents: 1 when base == 1, ±1 when base == -1 (depending on
  // whether the exponent is odd or even), and 0 otherwise.
  // The exponent is clamped to 0 via stablehlo::Max(broadcasted_exp_op,
  // zero_op) before calling stablehlo::Pow to bypass the XLA TPU scalar
  // lowering omission (SpowI32) that would otherwise return 0 for base == -1.
  if (base_type.getElementType().isSignlessInteger() &&
      !IsBooleanType(base_type)) {
    mlir::MlirOp zero_op = MakeConstantLike(broadcasted_base_op, 0);
    mlir::MlirOp one_op = MakeConstantLike(broadcasted_base_op, 1);
    mlir::MlirOp neg_one_op = MakeConstantLike(broadcasted_base_op, -1);
    mlir::MlirOp safe_exp_op = stablehlo::Max(broadcasted_exp_op, zero_op);
    mlir::MlirOp pow_op = stablehlo::Pow(broadcasted_base_op, safe_exp_op);
    mlir::MlirOp exp_lt_zero = stablehlo::Compare(
        broadcasted_exp_op, zero_op, stablehlo::ComparisonDirection::LT);
    mlir::MlirOp base_eq_one = stablehlo::Compare(
        broadcasted_base_op, one_op, stablehlo::ComparisonDirection::EQ);
    mlir::MlirOp base_eq_neg_one = stablehlo::Compare(
        broadcasted_base_op, neg_one_op, stablehlo::ComparisonDirection::EQ);
    mlir::MlirOp exp_low_bit_op = stablehlo::And(broadcasted_exp_op, one_op);
    mlir::MlirOp exp_is_odd = stablehlo::Compare(
        exp_low_bit_op, one_op, stablehlo::ComparisonDirection::EQ);
    mlir::MlirOp neg_one_pow_op =
        stablehlo::Select(exp_is_odd, neg_one_op, one_op);
    mlir::MlirOp non_unit_or_neg_one_op =
        stablehlo::Select(base_eq_neg_one, neg_one_pow_op, zero_op);
    mlir::MlirOp neg_exp_result_op =
        stablehlo::Select(base_eq_one, one_op, non_unit_or_neg_one_op);
    return stablehlo::Select(exp_lt_zero, neg_exp_result_op, pow_op);
  }

  return stablehlo::Pow(broadcasted_base_op, broadcasted_exp_op);
}

absl::StatusOr<mlir::MlirOp> BuildBitwiseRightShiftShlo(mlir::MlirOp lhs_op,
                                                        mlir::MlirOp rhs_op) {
  TT_ASSIGN_OR_RETURN((auto [lhs, rhs]),
                      ApplyBroadcastIfNeeded(lhs_op, rhs_op));
  const mlir::RankedTensorType lhs_type = GetTensorTypeOrDie(lhs);
  if (lhs_type.getElementType().isUnsignedInteger()) {
    return stablehlo::ShiftRightLogical(lhs, rhs);
  }
  return stablehlo::ShiftRightArithmetic(lhs, rhs);
}

absl::StatusOr<mlir::MlirOp> BuildRemainderShlo(mlir::MlirOp lhs_op,
                                                mlir::MlirOp rhs_op) {
  TT_ASSIGN_OR_RETURN((auto [lhs, rhs]),
                      ApplyBroadcastIfNeeded(lhs_op, rhs_op));
  const mlir::ElementType orig_dtype = GetElementTypeOrDie(lhs);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType comp_dtype,
                      InferComputationDtype(orig_dtype));
  TT_ASSIGN_OR_RETURN(lhs, CastIfNeeded(lhs, comp_dtype));
  TT_ASSIGN_OR_RETURN(rhs, CastIfNeeded(rhs, comp_dtype));

  mlir::MlirOp rem_op = stablehlo::Rem(lhs, rhs);
  if (mlir::IsFloat(comp_dtype)) {
    mlir::MlirOp is_inf_rhs = mlir::chlo::IsInf(rhs);
    mlir::MlirOp is_finite_lhs = stablehlo::IsFinite(lhs);
    mlir::MlirOp is_inf_rhs_and_finite_lhs =
        stablehlo::And(is_inf_rhs, is_finite_lhs);
    rem_op = stablehlo::Select(is_inf_rhs_and_finite_lhs, lhs, rem_op);
  }
  mlir::MlirOp zero_op = MakeConstantLike(rem_op, 0);
  mlir::MlirOp rem_ne_zero =
      stablehlo::Compare(rem_op, zero_op, stablehlo::ComparisonDirection::NE);
  mlir::MlirOp rem_lt_zero =
      stablehlo::Compare(rem_op, zero_op, stablehlo::ComparisonDirection::LT);
  mlir::MlirOp rhs_lt_zero =
      stablehlo::Compare(rhs, zero_op, stablehlo::ComparisonDirection::LT);
  mlir::MlirOp signs_differ = stablehlo::Compare(
      rem_lt_zero, rhs_lt_zero, stablehlo::ComparisonDirection::NE);
  mlir::MlirOp should_adjust = stablehlo::And(rem_ne_zero, signs_differ);
  mlir::MlirOp adjusted_op = stablehlo::Add(rem_op, rhs);
  mlir::MlirOp result_op =
      stablehlo::Select(should_adjust, adjusted_op, rem_op);
  return CastIfNeeded(result_op, orig_dtype);
}

}  // namespace torch_tpu
