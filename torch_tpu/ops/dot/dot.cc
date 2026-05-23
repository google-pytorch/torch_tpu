// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/ops/dot/dot.h"

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

absl::StatusOr<mlir::MlirOp> BuildDotShlo(
    mlir::MlirOp lhs, mlir::MlirOp rhs, mlir::stablehlo::Precision precision) {
  const mlir::RankedTensorType lhs_type = GetTensorTypeOrDie(lhs);
  const mlir::RankedTensorType rhs_type = GetTensorTypeOrDie(rhs);
  bool is_any_i64 = lhs_type.getElementType().isInteger(64) ||
                    rhs_type.getElementType().isInteger(64);
  if (is_any_i64) {
    lhs = mlir::stablehlo::ConvertElementType(lhs, mlir::ElementType::F64);
    rhs = mlir::stablehlo::ConvertElementType(rhs, mlir::ElementType::F64);
  }
  auto precision_config = mlir::stablehlo::PrecisionConfigAttr::get(
      &lhs.getContext(), {precision, precision});
  auto dot_dimension_numbers =
      mlir::stablehlo::getDefaultDotDimensionNumbers(lhs.getValue());
  auto dot_result = mlir::stablehlo::DotGeneral(lhs, rhs, dot_dimension_numbers,
                                                precision_config);
  if (is_any_i64) {
    return mlir::stablehlo::ConvertElementType(dot_result,
                                               lhs_type.getElementType());
  }
  return dot_result;
}

}  // namespace torch_tpu
