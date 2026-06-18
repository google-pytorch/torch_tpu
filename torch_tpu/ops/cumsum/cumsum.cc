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

#include "torch_tpu/ops/cumsum/cumsum.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/scan_builder.h"

namespace torch_tpu {

absl::StatusOr<mlir::MlirOp> BuildCumsumShlo(
    const int64_t normalized_dim,
    const std::optional<mlir::ElementType> out_dtype, mlir::MlirOp input) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  mlir::Type element_type = input_type.getElementType();

  mlir::MlirBuilder& builder = input.getBuilder();
  if (out_dtype.has_value()) {
    element_type = getElementType(builder.getContext(), out_dtype.value());
  } else if (input_type.getElementType().isInteger()) {
    element_type = builder.getOpBuilder().getI64Type();
  }

  if (input_type.getElementType() != element_type) {
    input = mlir::stablehlo::ConvertElementType(input, element_type);
  }

  const mlir::RankedTensorType promoted_type = GetTensorTypeOrDie(input);
  // chlo.ScanOp carries are rank-reduced (the scan dimension is erased).
  llvm::SmallVector<int64_t> carry_shape(promoted_type.getShape().begin(),
                                         promoted_type.getShape().end());
  carry_shape.erase(carry_shape.begin() + normalized_dim);

  const mlir::MlirOp init_value = MakeScalarConstant(builder, 0, element_type);
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp carry_init,
                      BroadcastIfNeeded(init_value, carry_shape));

  MultiInputScanBodyBuilder body_builder =
      [](mlir::OpBuilder& op_builder, mlir::Location loc,
         mlir::ValueRange input_slices, mlir::Value /*index*/,
         mlir::ValueRange carries) -> absl::StatusOr<ScanBodyResults> {
    llvm::SmallVector<mlir::Value> sum = {
        mlir::stablehlo::AddOp::create(op_builder, loc, input_slices[0],
                                       carries[0])
            .getResult()};
    // For a cumulative sum the per-position output is the running carry.
    return ScanBodyResults{sum, sum};
  };

  // Associative scan -> chlo.ScanOp (native scan emitter). Results are
  // [carries..., outputs...]; the prefix-scan output is the single output.
  TT_ASSIGN_OR_RETURN(
      const DynamicMlirOpResults results,
      BuildScanShlo(
          builder, {input}, normalized_dim, /*num_scan_inputs=*/1,
          /*carry_inits=*/{carry_init}, /*output_inits=*/{input},
          std::move(body_builder),
          ScanOptions{.should_squeeze = true, .is_associative = true}));
  return results[/*num_carries=*/1];
}

}  // namespace torch_tpu
