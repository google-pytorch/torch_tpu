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

#include "torch_tpu/ops/a_min_max/a_min_max.h"

#include "absl/status/statusor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildAMinMaxShlo(Dimensions dims,
                                              ReductionMode mode, AMinMaxOp op,
                                              mlir::MlirOp input_op) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
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

}  // namespace torch_tpu
