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

#include "torch_tpu/ops/scatter/scatter.h"

#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildScatterShlo(
    mlir::MlirOp self, int64_t dim, mlir::MlirOp index, mlir::MlirOp src,
    ScatterOp scatter_op, mlir::ElementType computation_element_type) {
  mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  mlir::RankedTensorType src_type = GetTensorTypeOrDie(src);
  mlir::RankedTensorType index_type = GetTensorTypeOrDie(index);
  ABSL_VLOG(2) << "BuildScatterShlo:"
               << ", dim: " << dim
               << ", self_type: " << mlir::debugString(self_type)
               << ", src_type: " << mlir::debugString(src_type)
               << ", index_type: " << mlir::debugString(index_type)
               << ", scatter_op: " << FormatParamCacheKey(scatter_op);

  TT_RET_CHECK(self_type.getRank() == index_type.getRank(),
               error::kInvalidArgument)
      << "self and index must have the same rank";

  if (src_type.getRank() == 0) {
    TT_ASSIGN_OR_RETURN(src, BroadcastIfNeeded(src, index_type.getShape()));
    src_type = GetTensorTypeOrDie(src);
  }

  TT_RET_CHECK(self_type.getRank() == src_type.getRank(),
               error::kInvalidArgument)
      << "self and src must have the same rank";

  // If arguments are scalars, temporarily reshape them to rank 1.
  bool are_scalars = self_type.getRank() == 0;
  if (are_scalars) {
    self = mlir::stablehlo::Reshape(self, {1});
    self_type = GetTensorTypeOrDie(self);
    src = mlir::stablehlo::Reshape(src, {1});
    src_type = GetTensorTypeOrDie(src);
    index = mlir::stablehlo::Reshape(index, {1});
    index_type = GetTensorTypeOrDie(index);
  }

  TT_RET_CHECK(dim >= 0 && dim < self_type.getRank(), error::kInvalidArgument)
      << "dim must be in the range [0, self.getRank())";

  // Convert arguments to the computation type if necessary.
  mlir::Type computation_type =
      mlir::getElementType(self.getContext(), computation_element_type);
  ABSL_VLOG(2) << "computation_type: " << mlir::debugString(computation_type);
  if (self_type.getElementType() != computation_type) {
    self = mlir::stablehlo::ConvertElementType(self, computation_type);
  }
  if (src_type.getElementType() != computation_type) {
    src = mlir::stablehlo::ConvertElementType(src, computation_type);
  }

  mlir::MlirBuilder& builder = self.getBuilder();
  Dimensions all_dimensions(self_type.getRank());
  absl::c_iota(all_dimensions, 0);

  Dimensions expanded_index_shape = CopyIntVector(index_type.getShape());
  expanded_index_shape.push_back(1);
  mlir::Type index_element_type = index_type.getElementType();

  // scatter_indices has one more dimension that src/index, and contains
  // [i_1, i_2, ..., i_{dim-1}, index[i_1, i_2, ..., i_k], i_{dim+1}, ...,  i_k]
  // for each element [i_1, i_2, ..., i_k] from the index space of src/index.
  std::vector<mlir::MlirOp> parts;
  for (int d = 0; d < self_type.getRank(); ++d) {
    if (d == dim) {
      mlir::MlirOp expanded_index =
          mlir::stablehlo::Reshape(index, expanded_index_shape);
      parts.push_back(expanded_index);
    } else {
      mlir::MlirOp j_part = stablehlo::Iota(
          builder,
          makeTensorType(builder.getContext(), index_type.getShape(),
                         index_element_type),
          /*iota_dimension=*/d);
      j_part = mlir::stablehlo::Reshape(j_part, expanded_index_shape);
      parts.push_back(j_part);
    }
  }
  mlir::MlirOp scatter_indices =
      stablehlo::Concatenate(builder, parts, /*dimension=*/self_type.getRank());

  // Slice src to match index shape if necessary. PyTorch allows src to be
  // larger than index, but StableHLO requires exact match for updates.
  if (src_type.getShape() != index_type.getShape()) {
    Indices start_indices(src_type.getRank(), 0);
    Indices limit_indices = CopyIntVector(index_type.getShape());
    Indices strides(src_type.getRank(), 1);
    src = stablehlo::Slice(src, start_indices, limit_indices, strides);
  }

  stablehlo::ScatterDimensionNumbersAttr scatter_dimension_numbers =
      stablehlo::ScatterDimensionNumbersAttr::get(
          &self.getContext(),
          /*update_window_dims=*/{},
          /*inserted_window_dims=*/all_dimensions,
          /*input_batching_dims=*/{},
          /*scatter_indices_batching_dims=*/{},
          /*scatter_dims_to_operand_dims=*/all_dimensions,
          /*index_vector_dim=*/index_type.getRank());
  ABSL_VLOG(2) << "BuildScatterShlo: ScatterDimensionNumbers = "
               << mlir::debugString(scatter_dimension_numbers);

  // Create a region builder callback depending on the scatter op.
  auto block_type = self_type.clone({}, computation_type);
  mlir::RegionBuilderCallback region_builder;
  region_builder = [block_type, scatter_op](mlir::RegionBuilder& builder) {
    auto arg0 = mlir::Argument(builder, block_type);
    auto arg1 = mlir::Argument(builder, block_type);
    if (scatter_op == ScatterOp::kAdd) {
      mlir::MlirOp result = stablehlo::Add(arg0, arg1);
      stablehlo::Return(builder, {result});
    } else if (scatter_op == ScatterOp::kMul) {
      mlir::MlirOp result = stablehlo::Mul(arg0, arg1);
      stablehlo::Return(builder, {result});
    } else {
      stablehlo::Return(builder, {arg1});
    }
  };

  auto result = stablehlo::Scatter(
      {self}, scatter_indices, {src}, region_builder, scatter_dimension_numbers,
      /*indices_are_sorted=*/false, /*unique_indices=*/false)[0];
  result =
      mlir::stablehlo::ConvertElementType(result, self_type.getElementType());
  if (are_scalars) {
    result = mlir::stablehlo::Reshape(result, {});
  }
  return result;
}

}  // namespace torch_tpu
