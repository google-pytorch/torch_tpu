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

#include "torch_tpu/ops/arange/arange.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

namespace {
absl::StatusOr<mlir::MlirOp> PrepareScalar(
    mlir::MlirOp scalar, const mlir::RankedTensorType output_type) {
  const mlir::RankedTensorType scalar_type = GetTensorTypeOrDie(scalar);

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=scalar always come from a Constant,
                 // which comes from an at::Scalar.
      scalar_type.getNumElements() == 1, error::kInvalidArgument)
      << "expected scalar, but got shape: " << mlir::debugString(scalar_type);

  if (scalar_type.getRank() > 0) {
    scalar = mlir::stablehlo::Reshape(scalar, {});
  }
  if (scalar_type.getElementType() != output_type.getElementType()) {
    scalar = mlir::stablehlo::ConvertElementType(scalar,
                                                 output_type.getElementType());
  }
  return mlir::stablehlo::BroadcastInDim(output_type, scalar, {});
}
}  // namespace

absl::StatusOr<mlir::MlirOp> BuildArangeOp(const int64_t num_elements,
                                           const mlir::Type dtype,
                                           mlir::MlirOp start,
                                           mlir::MlirOp step) {
  mlir::MlirBuilder& builder = start.getBuilder();
  mlir::RankedTensorType output_type =
      mlir::RankedTensorType::get({num_elements}, dtype);

  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=inner checks are all infeasible.
      start, PrepareScalar(start, output_type),
      _.SetPrepend() << "cannot prepare start: ");
  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=inner checks are all infeasible.
      step, PrepareScalar(step, output_type),
      _.SetPrepend() << "cannot prepare step: ");

  auto iota = mlir::stablehlo::Iota(builder, output_type, 0);
  auto iota_step = mlir::stablehlo::Mul(iota, step);
  return mlir::stablehlo::Add(iota_step, start);
}

}  // namespace torch_tpu
