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

#include "torch_tpu/ops/triangular/triangular.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {

// Creates a boolean mask for the upper/lower triangle of a matrix.
//
// Args:
//   builder: The MLIR builder.
//   input_shape: The shape of the input tensor.
//   diagonal: The diagonal offset.
//   upper_triangle: Whether to create the mask for the upper triangle.
//
// Returns:
//   A boolean mask tensor.
// The mask is constructed as follows for a matrix of shape [m, n].
//   1. Create a tensor a of shape [n] containing the integers 0, 1, ..., n-1.
//   2. Create a tensor b of shape [m] containing the integers 0, 1, ..., m-1.
//   3. Add the diagonal offset to the integers in the second tensor b.
//   4. Broadcast both the tensors a and b to the shape of the input tensor.
//   5. Compare the values in the broadcasted a and b tensors. The comparison is
//   done using the '<=' for upper triangle and '>=' for
//   lower triangle.
//   6. Broadcast the mask to the shape of the input tensor.
//   For a matrix of shape [3, 2], diagonal = 0, the mask is constructed as:
//     [[0, 0],       [[0, 1],         [[1, 1],
//      [1, 1],  <=    [0, 1],    =     [0, 1],
//      [2, 2]]        [0, 1]]          [0, 0]]
mlir::MlirOp TriangularMask(mlir::MlirBuilder& builder,
                            const Dimensions& input_shape, int64_t diagonal,
                            TriangleType triangle_type) {
  const int64_t n_dims = input_shape.size();
  const int64_t m = input_shape[n_dims - 2];
  const int64_t n = input_shape[n_dims - 1];
  Dimensions major_dims(input_shape.begin(), input_shape.end() - 2);
  auto mlir_element_type = mlir::ElementType::I64;
  mlir::MlirOp a = stablehlo::Iota(
      builder, makeTensorType(builder.getContext(), {n}, mlir_element_type), 0);
  mlir::MlirOp b = stablehlo::Iota(
      builder, makeTensorType(builder.getContext(), {m}, mlir_element_type), 0);
  mlir::MlirOp diagonal_shlo = stablehlo::Constant(builder, diagonal);
  mlir::MlirOp diagonal_shlo_broadcast =
      stablehlo::Broadcast(diagonal_shlo, {m});
  mlir::MlirOp b_plus_diagonal = stablehlo::Add(b, diagonal_shlo_broadcast);
  mlir::MlirOp b_plus_diagonal_broadcast = stablehlo::BroadcastInDim(
      makeTensorType(builder.getContext(), {m, n}, mlir_element_type),
      b_plus_diagonal, {0});
  mlir::MlirOp a_broadcast = stablehlo::Broadcast(a, {m});
  auto comparison_direction = triangle_type == TriangleType::kUpper
                                  ? stablehlo::ComparisonDirection::LE
                                  : stablehlo::ComparisonDirection::GE;
  mlir::MlirOp mask = stablehlo::Compare(b_plus_diagonal_broadcast, a_broadcast,
                                         comparison_direction);
  if (!major_dims.empty()) {
    mlir::MlirOp mask_broadcast = stablehlo::Broadcast(mask, major_dims);
    return mask_broadcast;
  }
  return mask;
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildTriangularShlo(
    const int64_t diagonal, const TriangleType triangle_type,
    mlir::MlirOp input_op) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  Dimensions input_shape = CopyIntVector(input_type.getShape());
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=pytorch catches this error first.
      input_shape.size() >= 2, error::kInvalidArgument)
      << "input shape must have at least 2 dimensions, got shape ("
      << absl::StrJoin(input_shape, ", ") << ")";
  mlir::MlirOp triangular_mask =
      TriangularMask(builder, input_shape, diagonal, triangle_type);
  mlir::MlirOp k_zero = MakeConstantLike(input_op, 0.0);
  return stablehlo::Select(triangular_mask, input_op, k_zero);
}

}  // namespace torch_tpu
