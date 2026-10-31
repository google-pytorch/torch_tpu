/*
 * Copyright 2026 Google LLC
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

#include "csrc/ops/linalg/cross/linalg_cross.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

mlir::MlirOp SliceComponent(mlir::MlirOp op, const Dimensions& static_shape,
                            int64_t dim, int64_t index) {
  Indices start_indices(static_shape.size(), 0);
  Indices limit_indices = static_shape;
  Strides strides(static_shape.size(), 1);
  start_indices[dim] = index;
  limit_indices[dim] = index + 1;
  return mlir::stablehlo::Slice(op, start_indices, limit_indices, strides);
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildLinalgCrossShlo(mlir::MlirOp lhs,
                                                  mlir::MlirOp rhs,
                                                  int64_t dim) {
  TT_ASSIGN_OR_RETURN((auto [lhs_bcast, rhs_bcast]),
                      ApplyBroadcastIfNeeded(lhs, rhs));
  mlir::MlirBuilder& builder = lhs_bcast.getBuilder();
  const mlir::stablehlo::Dimensions result_dims = GetDimensions(lhs_bcast);
  const Dimensions static_shape = GetStaticShape(result_dims);

  if (!GetTensorTypeOrDie(lhs_bcast).hasStaticShape()) {
    const mlir::Type i32_type = builder.getOpBuilder().getI32Type();
    for (int64_t d = 0; d < static_cast<int64_t>(result_dims.size()); ++d) {
      if (!result_dims[d].boundOp.has_value()) continue;
      mlir::MlirOp bound =
          MakeScalarConstant(builder, result_dims[d].size, i32_type);
      lhs_bcast = mlir::stablehlo::SetDimensionSize(lhs_bcast, bound, d);
      rhs_bcast = mlir::stablehlo::SetDimensionSize(rhs_bcast, bound, d);
    }
  }

  mlir::MlirOp u0 = SliceComponent(lhs_bcast, static_shape, dim, 0);
  mlir::MlirOp u1 = SliceComponent(lhs_bcast, static_shape, dim, 1);
  mlir::MlirOp u2 = SliceComponent(lhs_bcast, static_shape, dim, 2);
  mlir::MlirOp v0 = SliceComponent(rhs_bcast, static_shape, dim, 0);
  mlir::MlirOp v1 = SliceComponent(rhs_bcast, static_shape, dim, 1);
  mlir::MlirOp v2 = SliceComponent(rhs_bcast, static_shape, dim, 2);

  mlir::MlirOp u1_v2 = mlir::stablehlo::Mul(u1, v2);
  mlir::MlirOp u2_v1 = mlir::stablehlo::Mul(u2, v1);
  mlir::MlirOp u2_v0 = mlir::stablehlo::Mul(u2, v0);
  mlir::MlirOp u0_v2 = mlir::stablehlo::Mul(u0, v2);
  mlir::MlirOp u0_v1 = mlir::stablehlo::Mul(u0, v1);
  mlir::MlirOp u1_v0 = mlir::stablehlo::Mul(u1, v0);

  mlir::MlirOp c0 = mlir::stablehlo::Subtract(u1_v2, u2_v1);
  mlir::MlirOp c1 = mlir::stablehlo::Subtract(u2_v0, u0_v2);
  mlir::MlirOp c2 = mlir::stablehlo::Subtract(u0_v1, u1_v0);

  mlir::MlirOp result =
      mlir::stablehlo::Concatenate(builder, {c0, c1, c2}, dim);
  for (int64_t d = 0; d < static_cast<int64_t>(result_dims.size()); ++d) {
    if (!result_dims[d].boundOp.has_value()) continue;
    mlir::MlirOp bound_op(builder, *result_dims[d].boundOp);
    mlir::MlirOp dim_size =
        mlir::stablehlo::GetDimensionSize(bound_op, result_dims[d].boundOpDim);
    result = mlir::stablehlo::SetDimensionSize(result, dim_size, d);
  }
  return result;
}

}  // namespace torch_tpu
