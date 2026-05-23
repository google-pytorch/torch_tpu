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

#include "torch_tpu/ops/topk/topk.h"

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {
void BuildSortComparisonBody(const llvm::ArrayRef<mlir::Type>& elementTypes,
                             stablehlo::ComparisonDirection direction,
                             mlir::RegionBuilder& rb) {
  // Add two arguments for each element type.
  llvm::SmallVector<mlir::MlirOp> args;
  for (auto elementType : elementTypes) {
    mlir::Type shapedType = mlir::RankedTensorType::get({}, elementType);
    args.push_back(mlir::Argument(rb, shapedType));
    args.push_back(mlir::Argument(rb, shapedType));
  }
  mlir::MlirOp compare = stablehlo::Compare(args[0], args[1], direction);
  stablehlo::Return(rb, {compare});
}
}  // namespace

absl::StatusOr<TopKOutputs> BuildTopKShlo(
    mlir::MlirOp input_op, int64_t k, int64_t dim, TopKMode topk_mode,
    std::optional<TopKStableMode> topk_stable_mode =
        TopKStableMode::kUnstable) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  // TODO(b/436289835): Update chlo topk implementation and use it here.
  mlir::MlirBuilder& builder = input_op.getBuilder();
  mlir::MlirOp indices = stablehlo::Iota(
      builder,
      makeTensorType(builder.getContext(), input_type.getShape(),
                     mlir::ElementType::I64),
      dim);
  auto comparator = [input_type, topk_mode](mlir::RegionBuilder& rb) {
    BuildSortComparisonBody(
        {input_type.getElementType(), rb.getOpBuilder().getI64Type()},
        topk_mode == TopKMode::kLargest ? stablehlo::ComparisonDirection::GT
                                        : stablehlo::ComparisonDirection::LT,
        rb);
  };
  llvm::SmallVector<mlir::MlirOp> outputs = stablehlo::Sort(
      builder, {input_op, indices}, comparator, dim,
      /*is_stable=*/topk_stable_mode == TopKStableMode::kStable);

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
