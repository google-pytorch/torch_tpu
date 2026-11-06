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

#include "csrc/ops/sort/sort.h"

#include <cstdint>
#include <limits>
#include <optional>

#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

SortShloOutputs BuildSortShlo(mlir::MlirOp input_op, bool stable, int64_t dim,
                              bool descending) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  mlir::MlirBuilder& builder = input_op.getBuilder();

  mlir::MlirOp sort_key = input_op;
  mlir::Type sort_key_elem_type = input_type.getElementType();
  bool convert_values_back = false;
  if (auto convert_op =
          input_op.getValue().getDefiningOp<stablehlo::ConvertOp>()) {
    mlir::Value operand = convert_op.getOperand();
    if (auto operand_type =
            mlir::dyn_cast<mlir::RankedTensorType>(operand.getType())) {
      mlir::Type elem_type = operand_type.getElementType();
      if (elem_type.isIntOrIndex() && elem_type.getIntOrFloatBitWidth() <= 32 &&
          sort_key_elem_type.isIntOrIndex() &&
          sort_key_elem_type.getIntOrFloatBitWidth() == 64) {
        sort_key = mlir::MlirOp(builder, operand);
        sort_key_elem_type = elem_type;
        convert_values_back = true;
      }
    }
  }

  const int64_t dim_size = input_type.getShape()[dim];
  const bool use_i32_indices = !input_type.isDynamicDim(dim) &&
                               dim_size <= std::numeric_limits<int32_t>::max();
  const mlir::Type index_elem_type = use_i32_indices
                                         ? builder.getOpBuilder().getI32Type()
                                         : builder.getOpBuilder().getI64Type();
  mlir::MlirOp indices = stablehlo::Iota(
      builder,
      makeTensorType(
          builder.getContext(), input_type.getShape(),
          use_i32_indices ? mlir::ElementType::I32 : mlir::ElementType::I64),
      dim);
  auto comparator = [sort_key_elem_type, index_elem_type,
                     descending](mlir::RegionBuilder& rb) {
    mlir::OpBuilder& op_builder = rb.getOpBuilder();
    std::optional<llvm::StringRef> compare_type = std::nullopt;
    if (mlir::isa<mlir::FloatType>(sort_key_elem_type)) {
      compare_type = "TOTALORDER";
    }
    stablehlo::buildSortComparisonBody(
        {sort_key_elem_type, index_elem_type},
        descending ? stablehlo::ComparisonDirection::GT
                   : stablehlo::ComparisonDirection::LT,
        compare_type, &rb.getRegion(), &op_builder);
  };
  auto outputs =
      stablehlo::Sort(builder, {sort_key, indices}, comparator, dim, stable);
  mlir::MlirOp values = outputs[0];
  if (convert_values_back) {
    values = stablehlo::ConvertElementType(values, input_type.getElementType());
  }
  mlir::MlirOp result_indices = outputs[1];
  if (use_i32_indices) {
    result_indices =
        stablehlo::ConvertElementType(result_indices, mlir::ElementType::I64);
  }
  return SortShloOutputs{.values = values, .indices = result_indices};
}

}  // namespace torch_tpu
