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

#include "torch_tpu/ops/where/where.h"

#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

absl::StatusOr<mlir::MlirOp> BuildWhereShlo(
    mlir::MlirOp condition, mlir::MlirOp input, mlir::MlirOp other,
    mlir::ElementType output_element_type) {
  mlir::MLIRContext& ctx = condition.getContext();
  const mlir::RankedTensorType condition_type = GetTensorTypeOrDie(condition);
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const mlir::RankedTensorType other_type = GetTensorTypeOrDie(other);

  // Apply element type conversion before broadcasting to minimize the amount of
  // data that needs to be copied around.
  if (!IsBooleanType(condition_type)) {
    condition =
        mlir::stablehlo::ConvertElementType(condition, mlir::ElementType::PRED);
  }
  mlir::Type output_type = mlir::getElementType(ctx, output_element_type);
  if (input_type.getElementType() != output_type) {
    input = mlir::stablehlo::ConvertElementType(input, output_type);
  }
  if (other_type.getElementType() != output_type) {
    other = mlir::stablehlo::ConvertElementType(other, output_type);
  }

  TT_ASSIGN_OR_RETURN((auto [bcast_condition, bcast_input, bcast_other]),
                      ApplyBroadcastIfNeeded(condition, input, other));

  return mlir::stablehlo::Select(bcast_condition, bcast_input, bcast_other);
}

}  // namespace torch_tpu
