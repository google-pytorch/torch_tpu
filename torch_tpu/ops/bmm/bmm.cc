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

#include "torch_tpu/ops/bmm/bmm.h"

#include <optional>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildBmmShlo(
    mlir::MlirOp self_op, mlir::MlirOp mat2_op,
    mlir::stablehlo::Precision precision,
    std::optional<DotAlgorithmConfig> dot_algorithm, mlir::Type result_type) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
  const mlir::RankedTensorType mat2_type = GetTensorTypeOrDie(mat2_op);

  // TODO: XLA doesn't support matmul with i64, so we convert them to f64.
  bool is_any_i64 = self_type.getElementType().isInteger(64) ||
                    mat2_type.getElementType().isInteger(64);
  if (is_any_i64) {
    self_op = stablehlo::ConvertElementType(self_op, mlir::ElementType::F64);
    mat2_op = stablehlo::ConvertElementType(mat2_op, mlir::ElementType::F64);
  }

  const mlir::RankedTensorType self_converted_type =
      GetTensorTypeOrDie(self_op);
  const mlir::RankedTensorType mat2_converted_type =
      GetTensorTypeOrDie(mat2_op);

  mlir::MLIRContext& ctx = self_op.getContext();

  mlir::Type dot_result_type = result_type;
  if (is_any_i64) {
    auto ranked_result_type = mlir::cast<mlir::RankedTensorType>(result_type);
    TT_ASSIGN_OR_RETURN(auto f64_type,
                        GetMlirType(ctx, mlir::ElementType::F64));
    dot_result_type =
        mlir::RankedTensorType::get(ranked_result_type.getShape(), f64_type);
  }

  const auto precision_config =
      mlir::stablehlo::PrecisionConfigAttr::get(&ctx, {precision, precision});
  auto dot_dimension_numbers = stablehlo::DotDimensionNumbersAttr::get(
      &ctx,
      /*lhs_batch_dimensions=*/{0},
      /*rhs_batch_dimensions=*/{0},
      /*lhs_contracting_dimensions=*/{2},
      /*rhs_contracting_dimensions=*/{1});

  mlir::stablehlo::DotAlgorithmAttr algorithm;
  if (dot_algorithm.has_value()) {
    TT_ASSIGN_OR_RETURN(
        auto elem_type,
        ConvertTo<mlir::ElementType>(dot_algorithm->accumulation_type()));
    TT_ASSIGN_OR_RETURN(auto accum_type, GetMlirType(ctx, elem_type));

    algorithm = mlir::stablehlo::DotAlgorithmAttr::get(
        &ctx, self_converted_type.getElementType(),
        mat2_converted_type.getElementType(), accum_type,
        dot_algorithm->lhs_component_count(),
        dot_algorithm->rhs_component_count(),
        dot_algorithm->num_primitive_operations(),
        dot_algorithm->allow_imprecise_accumulation());
  }

  mlir::MlirOp dot_result =
      stablehlo::DotGeneral(dot_result_type, self_op, mat2_op,
                            dot_dimension_numbers, precision_config, algorithm);
  if (is_any_i64) {
    auto out_type =
        mlir::cast<mlir::RankedTensorType>(result_type).getElementType();
    return stablehlo::ConvertElementType(dot_result, out_type);
  }
  return dot_result;
}

}  // namespace torch_tpu
