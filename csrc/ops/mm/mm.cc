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
#include <optional>

#include "absl/status/statusor.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/contraction_utils.h"
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

absl::StatusOr<mlir::MlirOp> BuildMmShlo(
    mlir::MlirOp lhs, mlir::MlirOp rhs, mlir::stablehlo::Precision precision,
    std::optional<mlir::ElementType> out_dtype) {
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

  TT_ASSIGN_OR_RETURN(
      const mlir::ElementType default_out_dtype,
      ConvertTo<mlir::ElementType>(lhs_tensor_type.getElementType()));
  const mlir::ElementType target_out_dtype =
      out_dtype.value_or(default_out_dtype);

  // Explanation: (m , n) @ (n, p) -> (m, p)
  // n is a contracting dimension, m and p are result dimensions.
  // There are no batch dimensions (like (b, m, n) @ (b, n, p) -> (b, m, p))).
  int64_t lhs_contracting_dimensions[1] = {1};
  int64_t rhs_contracting_dimensions[1] = {0};
  stablehlo::DotDimensionNumbersAttr dot_dimension_numbers =
      stablehlo::DotDimensionNumbersAttr::get(
          &ctx, /*lhs_batching_dimensions=*/{}, /*rhs_batching_dimensions=*/{},
          lhs_contracting_dimensions, rhs_contracting_dimensions);

  return BuildAccurateDotGeneral(lhs, rhs, dot_dimension_numbers,
                                 target_out_dtype, precision);
}

}  // namespace torch_tpu
