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

#include "torch_tpu/ops/mm/mm.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildMmShlo(mlir::MlirOp lhs, mlir::MlirOp rhs,
                                         mlir::stablehlo::Precision precision) {
  mlir::MlirBuilder& builder = lhs.getBuilder();
  mlir::MLIRContext& ctx = builder.getContext();
  const mlir::RankedTensorType lhs_tensor_type = GetTensorTypeOrDie(lhs);
  auto lhs_shape = lhs_tensor_type.getShape();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Rank constraint is checked before Mlir
                 // lowering.
      lhs_shape.size() == 2, error::kInvalidArgument)
      << "BuildMmShlo: lhs must be a 2D tensor, got rank "
      << lhs_tensor_type.getRank();

  const mlir::RankedTensorType rhs_tensor_type = GetTensorTypeOrDie(rhs);
  auto rhs_shape = rhs_tensor_type.getShape();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Rank constraint is checked before Mlir
                 // lowering.
      rhs_shape.size() == 2, error::kInvalidArgument)
      << "BuildMmShlo: rhs must be a 2D tensor, got rank "
      << rhs_tensor_type.getRank();

  auto dot_general_element_type = lhs_tensor_type.getElementType();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Type constraint is checked before Mlir
                 // lowering.
      rhs_tensor_type.getElementType() == dot_general_element_type,
      error::kInvalidArgument)
      << "BuildMmShlo: lhs and rhs must have the same element type, got "
      << mlir::debugString(lhs_tensor_type.getElementType()) << " and "
      << mlir::debugString(rhs_tensor_type.getElementType());
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Dimension constraint is checked before
                 // Mlir lowering.
      lhs_shape[1] == rhs_shape[0], error::kInvalidArgument)
      << "BuildMmShlo: lhs and rhs must have the same contracting dimension, "
         "got "
      << lhs_shape[1] << " and " << rhs_shape[0];

  // Explanation: (m , n) @ (n, p) -> (m, p)
  // n is a contracting dimension, m and p are result dimensions.
  // There are no batch dimensions (like (b, m, n) @ (b, n, p) -> (b, m, p))).
  int64_t lhs_contracting_dimensions[1] = {1};
  int64_t rhs_contracting_dimensions[1] = {0};
  stablehlo::DotDimensionNumbersAttr dot_dimension_numbers =
      stablehlo::DotDimensionNumbersAttr::get(
          &ctx, /*lhs_batching_dimensions=*/{}, /*rhs_batching_dimensions=*/{},
          lhs_contracting_dimensions, rhs_contracting_dimensions);

  // TODO(bawilson): Enable precision config once StableHLO lowering is fixed.
  // Update: Enabling precision config via context manager.
  stablehlo::Precision precisions[2] = {precision, precision};
  auto precision_config_attr =
      stablehlo::PrecisionConfigAttr::get(&ctx, precisions);

  // NB: deliberately using DotGeneral instead of Dot because Dot is scheduled
  // for removal from StableHLO.
  return stablehlo::DotGeneral(lhs, rhs, dot_dimension_numbers,
                               /*precision_config=*/precision_config_attr);
}

}  // namespace torch_tpu
