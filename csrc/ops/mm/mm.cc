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

#include "csrc/ops/mm/mm.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildMmShlo(mlir::MlirOp lhs, mlir::MlirOp rhs,
                                         mlir::stablehlo::Precision precision) {
  mlir::MlirBuilder& builder = lhs.getBuilder();
  mlir::MLIRContext& ctx = builder.getContext();
  const mlir::RankedTensorType lhs_tensor_type = GetTensorTypeOrDie(lhs);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Rank constraint is checked before Mlir
                 // lowering.
      lhs_tensor_type.getRank() == 2, error::kInvalidArgument)
      << "BuildMmShlo: lhs must be a 2D tensor, got rank "
      << lhs_tensor_type.getRank();

  const mlir::RankedTensorType rhs_tensor_type = GetTensorTypeOrDie(rhs);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Rank constraint is checked before Mlir
                 // lowering.
      rhs_tensor_type.getRank() == 2, error::kInvalidArgument)
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
  // The contracting dimension sizes are validated on the ATen tensors by the
  // caller (`ValidateMmOutInputs`); they are not compared here because a
  // bounded dynamic dimension is `mlir::ShapedType::kDynamic` in the MLIR type.

  // XLA doesn't support matmul with i64, so we convert them to f64.
  const bool is_any_i64 = lhs_tensor_type.getElementType().isInteger(64) ||
                          rhs_tensor_type.getElementType().isInteger(64);
  if (is_any_i64) {
    lhs = stablehlo::ConvertElementType(lhs, mlir::ElementType::F64);
    rhs = stablehlo::ConvertElementType(rhs, mlir::ElementType::F64);
  }

  // XLA requires the contracting dimensions to have the same upper bound, so
  // when only one of them is bounded dynamic, bound the other one like it.
  TT_ASSIGN_OR_RETURN(lhs,
                      BoundDimensionLike(lhs, /*dim=*/1, rhs, /*like_dim=*/0));
  TT_ASSIGN_OR_RETURN(rhs,
                      BoundDimensionLike(rhs, /*dim=*/0, lhs, /*like_dim=*/1));

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
  auto dot_result =
      stablehlo::DotGeneral(lhs, rhs, dot_dimension_numbers,
                            /*precision_config=*/precision_config_attr);
  return is_any_i64 ? stablehlo::ConvertElementType(dot_result,
                                                    dot_general_element_type)
                    : dot_result;
}

}  // namespace torch_tpu
