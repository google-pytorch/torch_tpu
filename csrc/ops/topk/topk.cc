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

#include "csrc/ops/topk/topk.h"

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/utils.h"
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

absl::StatusOr<TopKOutputs> BuildTopKShlo(mlir::MlirOp input_op, int64_t k,
                                          int64_t dim, TopKMode topk_mode) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  // TODO(b/436289835): Update chlo topk implementation and use it here.
  mlir::MlirBuilder& builder = input_op.getBuilder();
  mlir::MlirOp indices = stablehlo::Iota(
      builder,
      makeTensorType(builder.getContext(), input_type.getShape(),
                     mlir::ElementType::I64),
      dim);
  auto comparator = [input_type, topk_mode](mlir::RegionBuilder& rb) {
    mlir::OpBuilder& op_builder = rb.getOpBuilder();
    std::optional<llvm::StringRef> compare_type = std::nullopt;
    if (mlir::isa<mlir::FloatType>(input_type.getElementType())) {
      compare_type = "TOTALORDER";
    }
    stablehlo::buildSortComparisonBody(
        {input_type.getElementType(), op_builder.getI64Type()},
        topk_mode == TopKMode::kLargest ? stablehlo::ComparisonDirection::GT
                                        : stablehlo::ComparisonDirection::LT,
        compare_type, &rb.getRegion(), &op_builder);
  };
  llvm::SmallVector<mlir::MlirOp> outputs =
      stablehlo::Sort(builder, {input_op, indices}, comparator, dim,
                      /*is_stable=*/false);

  Indices start_indices(input_type.getRank(), 0);
  Indices limit_indices = CopyIntVector(input_type.getShape());
  limit_indices[dim] = k;
  Indices stride_indices(input_type.getRank(), 1);
  mlir::MlirOp topk_values = stablehlo::Slice(outputs[0], start_indices,
                                              limit_indices, stride_indices);
  mlir::MlirOp topk_indices = stablehlo::Slice(outputs[1], start_indices,
                                               limit_indices, stride_indices);
  return TopKOutputs{.values = topk_values, .indices = topk_indices};
}

}  // namespace torch_tpu
