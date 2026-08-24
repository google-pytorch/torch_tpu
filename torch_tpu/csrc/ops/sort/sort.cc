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

#include "torch_tpu/csrc/ops/sort/sort.h"

#include <cstdint>
#include <optional>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

SortShloOutputs BuildSortShlo(mlir::MlirOp input_op, bool stable, int64_t dim,
                              bool descending) {
  const mlir::RankedTensorType inputType = GetTensorTypeOrDie(input_op);
  mlir::MlirBuilder& builder = input_op.getBuilder();
  mlir::MlirOp indices = stablehlo::Iota(
      builder,
      makeTensorType(
          builder.getContext(), inputType.getShape(),
          // Per https://pytorch.org/docs/stable/generated/torch.sort.html, the
          // output indices are of type LongTensor, i.e. 64-bit integer tensor.
          mlir::ElementType::I64),
      dim);
  auto comparator = [inputType, descending](mlir::RegionBuilder& rb) {
    mlir::OpBuilder& op_builder = rb.getOpBuilder();
    std::optional<llvm::StringRef> compare_type = std::nullopt;
    if (mlir::isa<mlir::FloatType>(inputType.getElementType())) {
      compare_type = "TOTALORDER";
    }
    stablehlo::buildSortComparisonBody(
        {inputType.getElementType(), op_builder.getI64Type()},
        descending ? stablehlo::ComparisonDirection::GT
                   : stablehlo::ComparisonDirection::LT,
        compare_type, &rb.getRegion(), &op_builder);
  };
  auto outputs =
      stablehlo::Sort(builder, {input_op, indices}, comparator, dim, stable);
  return SortShloOutputs{.values = outputs[0], .indices = outputs[1]};
}

}  // namespace torch_tpu
