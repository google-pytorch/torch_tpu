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

#include "csrc/ops/where/where.h"

#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

absl::StatusOr<mlir::MlirOp> BuildWhereShlo(
    mlir::MlirOp condition, mlir::MlirOp input, mlir::MlirOp other,
    mlir::ElementType output_element_type) {
  const mlir::RankedTensorType condition_type = GetTensorTypeOrDie(condition);

  // Apply element type conversion before broadcasting to minimize the amount of
  // data that needs to be copied around.
  if (!IsBooleanType(condition_type)) {
    condition =
        mlir::stablehlo::ConvertElementType(condition, mlir::ElementType::PRED);
  }
  TT_ASSIGN_OR_RETURN(
      input, ComplexSafeConvertElementType(input, output_element_type));
  TT_ASSIGN_OR_RETURN(
      other, ComplexSafeConvertElementType(other, output_element_type));

  TT_ASSIGN_OR_RETURN((auto [bcast_condition, bcast_input, bcast_other]),
                      ApplyBroadcastIfNeeded(condition, input, other));

  return mlir::stablehlo::Select(bcast_condition, bcast_input, bcast_other);
}

}  // namespace torch_tpu
