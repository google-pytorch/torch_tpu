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

#include "csrc/ops/triangular/triangular.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {

// Creates a boolean mask for the upper/lower triangle of a matrix.
//
// Args:
//   input_op: The input tensor op.
//   diagonal: The diagonal offset.
//   triangle_type: Whether to create the mask for the upper or lower triangle.
//
// Returns:
//   A boolean mask tensor matching the shape of `input_op`.
// The mask is constructed as follows for a matrix with last two dimensions
// [m, n]:
//   1. Create a coordinate tensor a matching `input_op`'s shape containing the
//      column indices 0, 1, ..., n-1 along the last dimension.
//   2. Create a coordinate tensor b matching `input_op`'s shape containing the
//      row indices 0, 1, ..., m-1 along the second-to-last dimension.
//   3. Add the diagonal offset to the row indices in b.
//   4. Compare the values in the (b + diagonal) and a tensors using '<=' for
//      the upper triangle and '>=' for the lower triangle.
//   For a matrix of shape [3, 2], diagonal = 0, the upper-triangle mask is
//   constructed as:
//     [[0, 0],       [[0, 1],         [[1, 1],
//      [1, 1],  <=    [0, 1],    =     [0, 1],
//      [2, 2]]        [0, 1]]          [0, 0]]
mlir::MlirOp TriangularMask(mlir::MlirOp input_op, int64_t diagonal,
                            TriangleType triangle_type) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  const int64_t n_dims = input_type.getRank();
  auto mlir_element_type = mlir::ElementType::I64;
  mlir::MlirOp a = stablehlo::IotaLike(input_op, n_dims - 1, mlir_element_type);
  mlir::MlirOp b = stablehlo::IotaLike(input_op, n_dims - 2, mlir_element_type);
  mlir::MlirOp diagonal_shlo_broadcast = MakeConstantLike(b, diagonal);
  mlir::MlirOp b_plus_diagonal = stablehlo::Add(b, diagonal_shlo_broadcast);
  auto comparison_direction = triangle_type == TriangleType::kUpper
                                  ? stablehlo::ComparisonDirection::LE
                                  : stablehlo::ComparisonDirection::GE;
  return stablehlo::Compare(b_plus_diagonal, a, comparison_direction);
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildTriangularShlo(
    const int64_t diagonal, const TriangleType triangle_type,
    mlir::MlirOp input_op) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=pytorch catches this error first.
      input_type.getRank() >= 2, error::kInvalidArgument)
      << "input shape must have at least 2 dimensions, got rank "
      << input_type.getRank();
  mlir::MlirOp triangular_mask =
      TriangularMask(input_op, diagonal, triangle_type);
  mlir::MlirOp k_zero = MakeConstantLike(input_op, 0.0);
  return stablehlo::Select(triangular_mask, input_op, k_zero);
}

}  // namespace torch_tpu
