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

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildBmmShlo(
    mlir::MlirOp self_op, mlir::MlirOp mat2_op, mlir::ElementType out_dtype,
    mlir::stablehlo::Precision precision) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
  const mlir::RankedTensorType mat2_type = GetTensorTypeOrDie(mat2_op);

  // TODO: XLA doesn't support matmul with i64, so we convert them to f64.
  bool is_any_i64 = self_type.getElementType().isInteger(64) ||
                    mat2_type.getElementType().isInteger(64);
  if (is_any_i64) {
    self_op = stablehlo::ConvertElementType(self_op, mlir::ElementType::F64);
    mat2_op = stablehlo::ConvertElementType(mat2_op, mlir::ElementType::F64);
  }

  const auto precision_config = mlir::stablehlo::PrecisionConfigAttr::get(
      &self_op.getContext(), {precision, precision});
  auto dot_dimension_numbers = stablehlo::DotDimensionNumbersAttr::get(
      &self_op.getContext(),
      /*lhs_batch_dimensions=*/{0},
      /*rhs_batch_dimensions=*/{0},
      /*lhs_contracting_dimensions=*/{2},
      /*rhs_contracting_dimensions=*/{1});
  mlir::MlirOp dot_result = stablehlo::DotGeneral(
      self_op, mat2_op, dot_dimension_numbers, precision_config);
  if (is_any_i64) {
    return stablehlo::ConvertElementType(dot_result, out_dtype);
  }
  return dot_result;
}

}  // namespace torch_tpu
