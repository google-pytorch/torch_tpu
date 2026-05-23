
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

#include "torch_tpu/ops/tril_indices/tril_indices.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/nonzero/nonzero.h"

namespace torch_tpu {
absl::StatusOr<mlir::MlirOp> BuildTrilIndicesShlo(mlir::MlirBuilder& builder,
                                                  int64_t row, int64_t col,
                                                  int64_t offset,
                                                  int64_t tril_size,
                                                  mlir::Type element_type) {
  // Based on jnp.tril_indices StableHLO dump, should be good for TPU
  // Create a grid of row indices (i) of shape (row, col).
  auto iota_row = mlir::stablehlo::Iota(
      builder,
      mlir::RankedTensorType::get({row, col},
                                  builder.getOpBuilder().getI64Type()),
      /*iota_dimension=*/0);

  // Create a grid of column indices (j) of shape (row, col).
  auto iota_col = mlir::stablehlo::Iota(
      builder,
      mlir::RankedTensorType::get({row, col},
                                  builder.getOpBuilder().getI64Type()),
      /*iota_dimension=*/1);

  // Create a constant for the offset and broadcast it to (row, col).
  mlir::MlirOp offset_op = mlir::stablehlo::Constant(builder, offset);
  offset_op = mlir::stablehlo::BroadcastInDim(iota_row.getType(), offset_op,
                                              /*broadcast_dimensions=*/{});

  // Calculate `i + offset` and compare `j <= i + offset` to get the boolean
  // mask.
  auto i_plus_offset = mlir::stablehlo::Add(iota_row, offset_op);
  auto mask = mlir::stablehlo::Compare(
      iota_col, i_plus_offset, mlir::stablehlo::ComparisonDirection::LE);

  // Reuse the BuildNonzeroShlo function to get the coordinates of non-zero
  // values.
  TT_ASSIGN_OR_RETURN(mlir::MlirOp nonzero_op,
                      BuildNonzeroShlo(tril_size, mask));

  // The result from BuildNonzeroShlo is shape (tril_size, 2), expected
  // tril_indices is (2, tril_size). So we need to transpose it.
  auto transposed_indices =
      mlir::stablehlo::Transpose(nonzero_op, /*permutation=*/{1, 0});

  return mlir::stablehlo::ConvertElementType(transposed_indices, element_type);
}
}  // namespace torch_tpu
