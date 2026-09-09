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

#include "torch_tpu/csrc/ops/binary.h"

#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/csrc/common/aten_utils.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"

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
                      _.SetPrepend() << "could not broadcast real and imag: ");
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
                      _.SetPrepend() << "could not broadcast abs and angle: ");

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
                      _.SetPrepend() << "could not broadcast base and exp: ");

  // Treat true as 1 and false as 0 for boolean inputs.
  // Equivalent to base_op or not(exp_op).
  if (IsBooleanType(base_type) && IsBooleanType(exp_type)) {
    broadcasted_exp_op = stablehlo::Not(broadcasted_exp_op);
    return stablehlo::Or(broadcasted_base_op, broadcasted_exp_op);
  }

  const mlir::ElementType base_elem = GetElementTypeOrDie(broadcasted_base_op);
  const mlir::ElementType exp_elem = GetElementTypeOrDie(broadcasted_exp_op);
  mlir::ElementType common_elem = base_elem;
  if (base_elem != exp_elem) {
    at::ScalarType base_st = ConvertTo<at::ScalarType>(base_elem);
    at::ScalarType exp_st = ConvertTo<at::ScalarType>(exp_elem);
    at::ScalarType promoted_st = c10::promoteTypes(base_st, exp_st);
    TT_ASSIGN_OR_RETURN(common_elem, ConvertTo<mlir::ElementType>(promoted_st));
  }
  TT_ASSIGN_OR_RETURN(const mlir::ElementType comp_dtype,
                      InferComputationDtype(common_elem));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp base_comp,
                      CastIfNeeded(broadcasted_base_op, comp_dtype));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp exp_comp,
                      CastIfNeeded(broadcasted_exp_op, comp_dtype));

  mlir::MlirOp pow_comp = stablehlo::Pow(base_comp, exp_comp);
  return CastIfNeeded(pow_comp, common_elem);
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

absl::StatusOr<mlir::MlirOp> BuildRemainderShlo(mlir::MlirOp self_op,
                                                mlir::MlirOp other_op) {
  TT_ASSIGN_OR_RETURN((auto [cast_self_op, cast_other_op]),
                      ApplyBroadcastIfNeeded(self_op, other_op));
  const mlir::ElementType self_elem = GetElementTypeOrDie(cast_self_op);
  const mlir::ElementType other_elem = GetElementTypeOrDie(cast_other_op);
  mlir::ElementType common_elem = self_elem;
  if (self_elem != other_elem) {
    at::ScalarType self_st = ConvertTo<at::ScalarType>(self_elem);
    at::ScalarType other_st = ConvertTo<at::ScalarType>(other_elem);
    at::ScalarType promoted_st = c10::promoteTypes(self_st, other_st);
    TT_ASSIGN_OR_RETURN(common_elem, ConvertTo<mlir::ElementType>(promoted_st));
  }
  TT_ASSIGN_OR_RETURN(const mlir::ElementType comp_dtype,
                      InferComputationDtype(common_elem));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp self_comp,
                      CastIfNeeded(cast_self_op, comp_dtype));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp other_comp,
                      CastIfNeeded(cast_other_op, comp_dtype));

  // stablehlo::Rem computes IEEE-754 round-to-zero remainder (fmod)
  mlir::MlirOp mod = mlir::stablehlo::Rem(self_comp, other_comp);

  // If mod != 0 and signs_differ(other, mod), mod = mod + other
  auto zero = GetTensorTypeOrDie(mod).getElementType().isInteger()
                  ? MakeConstantLike(mod, 0)
                  : MakeConstantLike(mod, 0.0);
  auto mod_ne_zero = mlir::stablehlo::Compare(
      mod, zero, mlir::stablehlo::ComparisonDirection::NE);
  auto mod_lt_zero = mlir::stablehlo::Compare(
      mod, zero, mlir::stablehlo::ComparisonDirection::LT);
  auto other_lt_zero = mlir::stablehlo::Compare(
      other_comp, zero, mlir::stablehlo::ComparisonDirection::LT);
  auto signs_differ = mlir::stablehlo::Compare(
      mod_lt_zero, other_lt_zero, mlir::stablehlo::ComparisonDirection::NE);
  auto need_adjust = mlir::stablehlo::And(mod_ne_zero, signs_differ);
  auto adjusted = mlir::stablehlo::Add(mod, other_comp);
  auto result_comp = mlir::stablehlo::Select(need_adjust, adjusted, mod);

  return CastIfNeeded(result_comp, common_elem);
}

}  // namespace torch_tpu
