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
  // Per https://pytorch.org/docs/stable/generated/torch.sort.html, the
  // output indices are of type LongTensor, i.e. 64-bit integer tensor.
  mlir::MlirOp indices =
      stablehlo::IotaLike(input_op, dim, mlir::ElementType::I64);
  auto comparator = [input_type, descending](mlir::RegionBuilder& rb) {
    mlir::OpBuilder& op_builder = rb.getOpBuilder();
    const mlir::Type elem_type = input_type.getElementType();
    if (!mlir::isa<mlir::FloatType>(elem_type)) {
      stablehlo::buildSortComparisonBody(
          {elem_type, op_builder.getI64Type()},
          descending ? stablehlo::ComparisonDirection::GT
                     : stablehlo::ComparisonDirection::LT,
          /*compare_type=*/std::nullopt, &rb.getRegion(), &op_builder);
      return;
    }
    // Use IEEE FLOAT comparison (so -0.0 == +0.0 for stable sort tie-breaking)
    // and order all NaNs (regardless of sign bit) after +inf, matching PyTorch
    // CPU and CUDA semantics.
    const mlir::RankedTensorType val_type =
        mlir::RankedTensorType::get({}, elem_type);
    const mlir::RankedTensorType idx_type =
        mlir::RankedTensorType::get({}, op_builder.getI64Type());
    mlir::MlirOp lhs = mlir::Argument(rb, val_type);
    mlir::MlirOp rhs = mlir::Argument(rb, val_type);
    // Index arguments are required by the Sort comparator signature, though
    // comparison only inspects the value arguments.
    mlir::Argument(rb, idx_type);
    mlir::Argument(rb, idx_type);

    mlir::MlirOp cmp =
        stablehlo::Compare(lhs, rhs,
                           descending ? stablehlo::ComparisonDirection::GT
                                      : stablehlo::ComparisonDirection::LT);
    mlir::MlirOp lhs_is_nan =
        stablehlo::Compare(lhs, lhs, stablehlo::ComparisonDirection::NE);
    mlir::MlirOp rhs_is_nan =
        stablehlo::Compare(rhs, rhs, stablehlo::ComparisonDirection::NE);
    mlir::MlirOp nan_wins;
    if (descending) {
      mlir::MlirOp not_rhs_is_nan = stablehlo::Not(rhs_is_nan);
      nan_wins = stablehlo::And(lhs_is_nan, not_rhs_is_nan);
    } else {
      mlir::MlirOp not_lhs_is_nan = stablehlo::Not(lhs_is_nan);
      nan_wins = stablehlo::And(not_lhs_is_nan, rhs_is_nan);
    }
    mlir::MlirOp result = stablehlo::Or(cmp, nan_wins);
    stablehlo::Return(rb, {result});
  };
  auto outputs =
      stablehlo::Sort(builder, {input_op, indices}, comparator, dim, stable);
  return SortShloOutputs{.values = outputs[0], .indices = outputs[1]};
}

}  // namespace torch_tpu
