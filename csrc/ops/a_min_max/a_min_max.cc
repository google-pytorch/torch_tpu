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

#include "csrc/ops/a_min_max/a_min_max.h"

#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/reductions/reductions.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildAMinMaxShlo(Dimensions dims,
                                              ReductionMode mode, AMinMaxOp op,
                                              mlir::MlirOp input_op) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  if (input_type.getRank() == 0) {
    return input_op;
  }
  mlir::Type input_element_type = input_type.getElementType();
  mlir::MlirBuilder& builder = input_op.getBuilder();
  mlir::Attribute min_max_init_attr =
      op == AMinMaxOp::kAmax
          ? GetMinFiniteValueAttr(input_element_type, builder.getOpBuilder())
          : GetMaxFiniteValueAttr(input_element_type, builder.getOpBuilder());
  mlir::DenseElementsAttr value_init_attr = mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get({}, input_element_type), min_max_init_attr);
  mlir::MlirOp value_init = stablehlo::Constant(builder, value_init_attr);

  auto reduce_builder = [input_element_type, op](mlir::RegionBuilder& rb) {
    if (op == AMinMaxOp::kAmax) {
      mlir::stablehlo::buildReduceBody<stablehlo::MaxOp>(
          input_element_type, rb.getRegion(), rb.getOpBuilder());
      return;
    }
    mlir::stablehlo::buildReduceBody<stablehlo::MinOp>(
        input_element_type, rb.getRegion(), rb.getOpBuilder());
  };
  // Run a reduction over the input elements to get the min or max value.
  auto min_max_value =
      stablehlo::Reduce(builder, input_op, value_init, reduce_builder, dims)[0];
  if (mode == ReductionMode::kKeepDims) {
    min_max_value = BuildKeepDimsShlo(input_op, min_max_value, dims);
  }
  return min_max_value;
}

absl::StatusOr<MlirOpResults<2>> BuildFusedAMinMaxShlo(Dimensions dims,
                                                       ReductionMode mode,
                                                       mlir::MlirOp input_op) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  mlir::Type input_element_type = input_type.getElementType();
  mlir::MlirBuilder& builder = input_op.getBuilder();

  if (input_type.getRank() == 0) {
    return MlirOpResults<2>{input_op, input_op};
  }

  mlir::Attribute min_init_attr =
      GetMaxFiniteValueAttr(input_element_type, builder.getOpBuilder());
  mlir::Attribute max_init_attr =
      GetMinFiniteValueAttr(input_element_type, builder.getOpBuilder());

  mlir::DenseElementsAttr min_value_init_attr = mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get({}, input_element_type), min_init_attr);
  mlir::DenseElementsAttr max_value_init_attr = mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get({}, input_element_type), max_init_attr);

  mlir::MlirOp min_value_init =
      stablehlo::Constant(builder, min_value_init_attr);
  mlir::MlirOp max_value_init =
      stablehlo::Constant(builder, max_value_init_attr);

  mlir::Type ranked_element_type =
      mlir::RankedTensorType::get(/*shape=*/{}, input_element_type);
  auto reduce_builder = [ranked_element_type](mlir::RegionBuilder& rb) {
    auto acc_min = mlir::Argument(rb, ranked_element_type);
    auto acc_max = mlir::Argument(rb, ranked_element_type);
    auto val_min = mlir::Argument(rb, ranked_element_type);
    auto val_max = mlir::Argument(rb, ranked_element_type);

    auto new_min = stablehlo::Min(acc_min, val_min);
    auto new_max = stablehlo::Max(acc_max, val_max);

    stablehlo::Return(rb, {new_min, new_max});
  };

  auto results =
      stablehlo::Reduce(builder, {input_op, input_op},
                        {min_value_init, max_value_init}, reduce_builder, dims);

  mlir::MlirOp min_val = results[0];
  mlir::MlirOp max_val = results[1];

  if (mode == ReductionMode::kKeepDims) {
    min_val = BuildKeepDimsShlo(input_op, min_val, dims);
    max_val = BuildKeepDimsShlo(input_op, max_val, dims);
  }

  return MlirOpResults<2>{min_val, max_val};
}

}  // namespace torch_tpu
