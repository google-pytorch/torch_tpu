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

#include "torch_tpu/ops/binary.h"

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

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

}  // namespace torch_tpu
