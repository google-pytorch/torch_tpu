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

#include <optional>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

absl::StatusOr<mlir::MlirOp> BuildDotShlo(
    mlir::MlirOp lhs, mlir::MlirOp rhs, mlir::stablehlo::Precision precision,
    std::optional<DotAlgorithmConfig> dot_algorithm, mlir::Type result_type) {
  const mlir::RankedTensorType lhs_type = GetTensorTypeOrDie(lhs);
  const mlir::RankedTensorType rhs_type = GetTensorTypeOrDie(rhs);
  bool is_any_i64 = lhs_type.getElementType().isInteger(64) ||
                    rhs_type.getElementType().isInteger(64);
  if (is_any_i64) {
    lhs = mlir::stablehlo::ConvertElementType(lhs, mlir::ElementType::F64);
    rhs = mlir::stablehlo::ConvertElementType(rhs, mlir::ElementType::F64);
  }
  auto& ctx = lhs.getContext();
  auto precision_config =
      mlir::stablehlo::PrecisionConfigAttr::get(&ctx, {precision, precision});
  mlir::stablehlo::DotAlgorithmAttr algorithm;

  mlir::Type lhs_prec_type = GetTensorTypeOrDie(lhs).getElementType();
  mlir::Type rhs_prec_type = GetTensorTypeOrDie(rhs).getElementType();
  if (dot_algorithm.has_value()) {
    TT_ASSIGN_OR_RETURN(auto elem_type, ConvertTo<mlir::ElementType>(
                                            dot_algorithm->accumulation_type));
    TT_ASSIGN_OR_RETURN(auto accum_type, GetMlirType(ctx, elem_type));

    algorithm = mlir::stablehlo::DotAlgorithmAttr::get(
        &ctx, lhs_prec_type, rhs_prec_type, accum_type,
        dot_algorithm->lhs_component_count, dot_algorithm->rhs_component_count,
        dot_algorithm->num_primitive_operations,
        dot_algorithm->allow_imprecise_accumulation);
  }

  auto dot_dimension_numbers =
      mlir::stablehlo::getDefaultDotDimensionNumbers(lhs.getValue());

  mlir::MlirOp dot_result =
      mlir::stablehlo::DotGeneral(result_type, lhs, rhs, dot_dimension_numbers,
                                  precision_config, algorithm);

  if (is_any_i64) {
    return mlir::stablehlo::ConvertElementType(dot_result,
                                               lhs_type.getElementType());
  }
  return dot_result;
}

}  // namespace torch_tpu
