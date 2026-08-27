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

#include "torch_tpu/csrc/ops/min_max/min_max.h"

#include <cstdint>
#include <numeric>

#include "absl/status/statusor.h"
#include "c10/util/Optional.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/reductions/reductions.h"

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

struct ReductionInput {
  mlir::MlirOp op;
  int64_t dim;
  bool was_flattened;
};

// Flattens the input tensor to 1D if no reduction dimension is specified,
// preparing it for global reduction. Otherwise, returns the input as is.
absl::StatusOr<ReductionInput> PrepareReductionInput(
    mlir::MlirOp input_op, c10::optional<int64_t> dim) {
  if (dim.has_value()) {
    return ReductionInput{
        .op = input_op, .dim = dim.value(), .was_flattened = false};
  }

  mlir::MlirOp flattened = Flatten(input_op);
  return ReductionInput{.op = flattened, .dim = 0, .was_flattened = true};
}

// Returns the initial value for reduction based on the MinMaxOp type
// (lowest finite value for Max, highest for Min).
mlir::MlirOp GetMinMaxInitValue(mlir::MlirBuilder& builder, MinMaxOp op,
                                mlir::Type element_type) {
  mlir::Attribute init_attr =
      op == MinMaxOp::kMax
          ? GetMinFiniteValueAttr(element_type, builder.getOpBuilder())
          : GetMaxFiniteValueAttr(element_type, builder.getOpBuilder());
  mlir::DenseElementsAttr init_elements_attr = mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get({}, element_type), init_attr);
  return stablehlo::Constant(builder, init_elements_attr);
}

// Restores the reduced dimensions to size 1 when keepdim is True.
// We use BuildKeepDimsShlo to reshape the output:
// - For global reduction (flattened), we set all dimensions of the original
//   shape to 1 (e.g., [A, B, C] -> [1, 1, 1]).
// - For single-dimension reduction, we only set the reduced dimension to 1
//   (e.g., [A, B, C] reduced along 1 -> [A, 1, C]).
absl::StatusOr<MinMaxOutputs> RestoreReducedDims(mlir::MlirOp original_input,
                                                 MinMaxOutputs outputs,
                                                 int64_t reduction_dim,
                                                 bool was_flattened) {
  const mlir::RankedTensorType original_type =
      GetTensorTypeOrDie(original_input);
  llvm::SmallVector<int64_t> dims;
  if (was_flattened) {
    dims.resize(original_type.getRank());
    std::iota(dims.begin(), dims.end(), 0);
  } else {
    dims.push_back(reduction_dim);
  }
  outputs.values = BuildKeepDimsShlo(original_input, outputs.values, dims);
  outputs.indices = BuildKeepDimsShlo(original_input, outputs.indices, dims);
  return outputs;
}

}  // namespace

absl::StatusOr<MinMaxOutputs> BuildMinMaxShlo(c10::optional<int64_t> dim,
                                              MinMaxOp op, ReductionMode mode,
                                              mlir::MlirOp input_op) {
  mlir::MlirBuilder& builder = input_op.getBuilder();

  TT_ASSIGN_OR_RETURN(ReductionInput prep,
                      PrepareReductionInput(input_op, dim));
  mlir::MlirOp indices =
      stablehlo::IotaLike(prep.op, prep.dim, mlir::ElementType::I64);

  const mlir::Type input_element_type =
      GetTensorTypeOrDie(input_op).getElementType();
  const mlir::Type indices_element_type =
      GetTensorTypeOrDie(indices).getElementType();
  auto reduce_body = [input_element_type, indices_element_type,
                      op](mlir::RegionBuilder& rb) {
    BuildReduceBody(op, input_element_type, indices_element_type, rb);
  };
  // A rank-0 tensor (scalar) cannot be reduced along any dimension, so the
  // reduction dims must be empty.
  mlir::SmallVector<int64_t> reduce_dims;
  if (GetTensorTypeOrDie(prep.op).getRank() > 0) {
    reduce_dims.push_back(prep.dim);
  }

  mlir::MlirOp value_init = GetMinMaxInitValue(builder, op, input_element_type);
  mlir::MlirOp index_init =
      MakeScalarConstant(builder, 0, mlir::ElementType::I64);

  auto arg_min_max =
      stablehlo::Reduce(builder, {prep.op, indices}, {value_init, index_init},
                        reduce_body, reduce_dims);
  MinMaxOutputs outputs{.values = arg_min_max[0], .indices = arg_min_max[1]};

  if (mode == ReductionMode::kKeepDims) {
    TT_ASSIGN_OR_RETURN(outputs, RestoreReducedDims(input_op, outputs, prep.dim,
                                                    prep.was_flattened));
  }

  return outputs;
}

}  // namespace torch_tpu
