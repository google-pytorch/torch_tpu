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

#include "csrc/ops/triangular/triangular_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/functional/bind_front.h"
#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

enum class TriangleType {
  kUpper,
  kLower,
};

namespace stablehlo = mlir::stablehlo;

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
//   1. Create a coordinate tensor `col` matching `input_op`'s shape containing
//      the column indices 0, 1, ..., n-1 along the last dimension.
//   2. Create a coordinate tensor `row` matching `input_op`'s shape containing
//      the row indices 0, 1, ..., m-1 along the second-to-last dimension.
//   3. Compute `col - row`, which lies in [-(m - 1), n - 1] and avoids signed
//      int64 overflow for extreme `diagonal` values near INT64_MAX/INT64_MIN.
//   4. Compare `(col - row)` against `diagonal` using '>=' for the upper
//      triangle and '<=' for the lower triangle.
//   For a matrix of shape [3, 2], diagonal = 0, the upper-triangle mask is
//   constructed as:
//     [[ 0,  1],       [[0, 0],         [[1, 1],
//      [-1,  0],  >=    [0, 0],    =     [0, 1],
//      [-2, -1]]        [0, 0]]          [0, 0]]
mlir::MlirOp TriangularMask(mlir::MlirOp input_op, const int64_t diagonal,
                            const TriangleType triangle_type) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  const int64_t n_dims = input_type.getRank();
  const auto mlir_element_type = mlir::ElementType::I64;
  mlir::MlirOp col =
      stablehlo::IotaLike(input_op, n_dims - 1, mlir_element_type);
  mlir::MlirOp row =
      stablehlo::IotaLike(input_op, n_dims - 2, mlir_element_type);
  mlir::MlirOp col_minus_row = stablehlo::Subtract(col, row);
  mlir::MlirOp diagonal_shlo_broadcast =
      MakeConstantLike(col_minus_row, diagonal);
  const auto comparison_direction = triangle_type == TriangleType::kUpper
                                        ? stablehlo::ComparisonDirection::GE
                                        : stablehlo::ComparisonDirection::LE;
  return stablehlo::Compare(col_minus_row, diagonal_shlo_broadcast,
                            comparison_direction);
}

absl::StatusOr<mlir::MlirOp> BuildTriangularShlo(
    const int64_t diagonal, const TriangleType triangle_type,
    mlir::MlirOp input_op) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=checked by AtenTriuOut/AtenTrilOut.
      input_type.getRank() >= 2, error::kInvalidArgument)
      << "input shape must have at least 2 dimensions, got rank "
      << input_type.getRank();
  mlir::MlirOp triangular_mask =
      TriangularMask(input_op, diagonal, triangle_type);
  mlir::MlirOp k_zero = MakeConstantLike(input_op, 0.0);
  return stablehlo::Select(triangular_mask, input_op, k_zero);
}

}  // namespace

at::Tensor& AtenTriuOut(const at::Tensor& self, int64_t diagonal,
                        at::Tensor& out) {
  TT_KERNEL(OpName::kTriuOut, param_keys, (self, diagonal, out), {
    TT_CHECK_THROW(self.dim() >= 2, error::kInvalidArgument)
        << "expected input to have at least 2 dimensions, got " << self.dim();
    TT_THROW_IF_ERROR(UnaryOpOut(
        self, out,
        absl::bind_front(BuildTriangularShlo, diagonal, TriangleType::kUpper),
        {.op_param_cache_keys = std::move(param_keys),
         .allow_out_dtype_cast = false}));
    return out;
  });
}

at::Tensor& AtenTrilOut(const at::Tensor& self, int64_t diagonal,
                        at::Tensor& out) {
  TT_KERNEL(OpName::kTrilOut, param_keys, (self, diagonal, out), {
    TT_CHECK_THROW(self.dim() >= 2, error::kInvalidArgument)
        << "expected input to have at least 2 dimensions, got " << self.dim();
    TT_THROW_IF_ERROR(UnaryOpOut(
        self, out,
        absl::bind_front(BuildTriangularShlo, diagonal, TriangleType::kLower),
        {.op_param_cache_keys = std::move(param_keys),
         .allow_out_dtype_cast = false}));
    return out;
  });
}

}  // namespace torch_tpu
