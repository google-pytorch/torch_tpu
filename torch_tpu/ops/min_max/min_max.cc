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

#include "torch_tpu/ops/min_max/min_max.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {

void BuildReduceBody(MinMaxOp op, mlir::Type value_type, mlir::Type index_type,
                     mlir::RegionBuilder& rb) {
  mlir::Type ranked_value_type =
      mlir::RankedTensorType::get(/*shape=*/{}, value_type);
  mlir::Type ranked_index_type =
      mlir::RankedTensorType::get(/*shape=*/{}, index_type);
  auto input_value0 = mlir::Argument(rb, ranked_value_type);
  auto index_value0 = mlir::Argument(rb, ranked_index_type);
  auto input_value1 = mlir::Argument(rb, ranked_value_type);
  auto index_value1 = mlir::Argument(rb, ranked_index_type);

  // Compare the inputs to the block.
  auto comparison_direction = op == MinMaxOp::kMax
                                  ? stablehlo::ComparisonDirection::GE
                                  : stablehlo::ComparisonDirection::LE;
  // Use compare instead of min / max op since we need to select the indices
  // corresponding to the min / max value.
  auto comparison_value =
      stablehlo::Compare(input_value0, input_value1, comparison_direction);
  auto min_max_value =
      stablehlo::Select(comparison_value, input_value0, input_value1);
  // Select the index corresponding to the max value.
  auto arg_min_max_value =
      stablehlo::Select(comparison_value, index_value0, index_value1);
  // If the values are equal, select the minimum index.
  auto is_equal = stablehlo::Compare(input_value0, input_value1,
                                     stablehlo::ComparisonDirection::EQ);
  auto smaller_index = stablehlo::Min(index_value0, index_value1);
  auto final_argmax_value =
      stablehlo::Select(is_equal, smaller_index, arg_min_max_value);
  stablehlo::Return(rb, {min_max_value, final_argmax_value});
}

}  // namespace

absl::StatusOr<MinMaxOutputs> BuildMinMaxShlo(int64_t dim, MinMaxOp op,
                                              ReductionMode mode,
                                              mlir::MlirOp input_op) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  const mlir::Type input_element_type = input_type.getElementType();
  mlir::MlirBuilder& builder = input_op.getBuilder();
  mlir::MlirOp indices =
      stablehlo::IotaLike(input_op, dim, mlir::ElementType::I64);
  const mlir::Type indices_element_type =
      GetTensorTypeOrDie(indices).getElementType();

  // Init stablehlo consts for initial values to the reduction.
  mlir::MlirOp index_init =
      MakeScalarConstant(builder, 0, mlir::ElementType::I64);

  mlir::Attribute min_max_init_attr =
      op == MinMaxOp::kMax
          ? GetMinFiniteValueAttr(input_element_type, builder.getOpBuilder())
          : GetMaxFiniteValueAttr(input_element_type, builder.getOpBuilder());
  mlir::DenseElementsAttr value_init_attr = mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get({}, input_element_type), min_max_init_attr);
  mlir::MlirOp value_init = stablehlo::Constant(builder, value_init_attr);

  auto reduce_body = [input_element_type, indices_element_type,
                      op](mlir::RegionBuilder& rb) {
    BuildReduceBody(op, input_element_type, indices_element_type, rb);
  };
  // Run a reduction over the input elements to get the min/max value and the
  // corresponding index. If tensor is rank 0, reduction dim must be [].
  mlir::SmallVector<int64_t> reduce_dims;
  if (input_type.getRank() > 0) {
    reduce_dims.push_back(dim);
  }
  auto arg_min_max =
      stablehlo::Reduce(builder, {input_op, indices}, {value_init, index_init},
                        reduce_body, reduce_dims);
  if (mode == ReductionMode::kKeepDims) {
    arg_min_max[0] = BuildKeepDimsShlo(input_op, arg_min_max[0], {dim});
    arg_min_max[1] = BuildKeepDimsShlo(input_op, arg_min_max[1], {dim});
  }
  return MinMaxOutputs{.values = arg_min_max[0], .indices = arg_min_max[1]};
}

}  // namespace torch_tpu
