// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/ops/index_put/index_put.h"

#include <cstdint>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {

absl::StatusOr<mlir::MlirOp> ApplyScatter(
    mlir::MlirOp self, mlir::ArrayRef<mlir::MlirOp> indices,
    const int64_t index_start_dim, const int64_t index_end_dim,
    const Dimensions& index_broadcast_shape, mlir::MlirOp values,
    const Dimensions& values_broadcast_shape, const bool accumulate) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  ABSL_VLOG(1) << "[BuildIndexPutShlo] self: " << debugString(self_type);

  // Broadcast indices to the broadcast shape.
  std::vector<mlir::MlirOp> parts;
  parts.reserve(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    mlir::MlirOp index = indices[i];
    mlir::RankedTensorType index_type = GetTensorTypeOrDie(index);
    ABSL_VLOG(1) << "[BuildIndexPutShlo] index dim:" << index_start_dim + i
                 << " before broadcast: " << debugString(index_type);
    TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=the op catches this error
                          // first before dispatching.
        index, BroadcastIfNeeded(index, index_broadcast_shape));
    index_type = GetTensorTypeOrDie(index);
    ABSL_VLOG(1) << "[BuildIndexPutShlo] index " << index_start_dim + i
                 << " after broadcast: " << debugString(index_type);

    // Expand the last dimension to 1 for concatenation to create
    // scatter_indices.
    Dimensions reshaped_index_shape = index_broadcast_shape;
    reshaped_index_shape.push_back(1);

    mlir::MlirOp reshaped_index =
        stablehlo::Reshape(index, reshaped_index_shape);
    parts.push_back(reshaped_index);
  }

  // Concatenate the reshaped indices to create scatter_indices.
  mlir::MlirOp scatter_indices = stablehlo::Concatenate(
      self.getBuilder(), parts, index_broadcast_shape.size());
  const mlir::RankedTensorType scatter_indices_type =
      GetTensorTypeOrDie(scatter_indices);
  ABSL_VLOG(1) << "[BuildIndexPutShlo] scatter_indices: "
               << debugString(scatter_indices_type);

  // Calculate update_window_dims, inserted_window_dims and
  // scatter_dims_to_operand_dims.
  //
  // update_window_dims: dimensions in values corresponding to the slice
  // The shape of values is expected to be broadcastable to the shape:
  // slice shape before indexing (S_before) + indexing broadcast shape (B) +
  // slice shape after indexing (S_after) = (S_before + B + S_after)
  //
  // inserted_window_dims: dimensions in self not part of the update window
  // and used for indexing into self.
  //
  // scatter_dims_to_operand_dims: this maps the elements of the scatter_indices
  // tensor to the dimensions in self used for indexing into self. These will be
  // indexed dims as there is a 1:1 mapping between the elements of the
  // scatter_indices tensor and the dimensions in self used for indexing into
  // self.
  //
  // Example 1: (no slice before indexing)
  //   self: tensor<5x5xi64>, indexed dims: [0]
  //   scatter_indices: tensor<2x1xi64> B: (2,)
  //   values: tensor<2x5xi64>.  B: (2,), S_after: (5)
  //
  //   update_window_dims: [1]
  //   inserted_window_dims: [0]
  //   scatter_dims_to_operand_dims: [0]
  //
  // Example 2: (slice before and after indexing)
  //   self: tensor<5x5x5xi64>, indexed dims: [1]
  //   scatter_indices: tensor<2x1xi64> B: (2,)
  //   values: tensor<5x2x5xi64>.  S_before: (5,), B: (2,), S_after: (5,)
  //
  //   update_window_dims: [0, 2]
  //   inserted_window_dims: [1]
  //   scatter_dims_to_operand_dims: [1]
  //

  Dimensions update_window_dims;  // dimensions in values corresponding to the
                                  // slice shape
  // Sliced dimensions before indexed dimensions
  for (int64_t dim = 0; dim < index_start_dim; ++dim) {
    update_window_dims.push_back(dim);
  }
  // Sliced dimensions after indexed dimensions
  int64_t dim_after_indexing = index_start_dim + index_broadcast_shape.size();
  for (int64_t dim = index_end_dim + 1; dim < self_type.getRank(); ++dim) {
    update_window_dims.push_back(dim_after_indexing++);
  }
  for (int64_t i = 0; i < update_window_dims.size(); ++i) {
    ABSL_VLOG(1) << "[BuildIndexPutShlo] update_window_dims[" << i
                 << "]: " << update_window_dims[i];
  }

  Dimensions inserted_window_dims;  // dimensions in self that are indexed into
  for (int64_t dim = index_start_dim; dim <= index_end_dim; ++dim) {
    inserted_window_dims.push_back(dim);
  }

  // scatter_dims_to_operand_dims is the same as inserted_window_dims
  Dimensions scatter_dims_to_operand_dims = inserted_window_dims;

  mlir::RankedTensorType values_type = GetTensorTypeOrDie(values);
  ABSL_VLOG(1) << "[BuildIndexPutShlo] values: " << debugString(values_type);

  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=the op catches this error
                        // first before dispatching.
      values, BroadcastIfNeeded(values, values_broadcast_shape));

  values_type = GetTensorTypeOrDie(values);
  ABSL_VLOG(1) << "[BuildIndexPutShlo] values after broadcast: "
               << debugString(values_type);

  stablehlo::ScatterDimensionNumbersAttr scatter_dimension_numbers =
      stablehlo::ScatterDimensionNumbersAttr::get(
          &self.getContext(), update_window_dims, inserted_window_dims,
          /*input_batching_dims=*/{},
          /*scatter_indices_batching_dims=*/{}, scatter_dims_to_operand_dims,
          /*index_vector_dim=*/index_broadcast_shape.size());

  auto element_type = self_type.getElementType();
  auto region_tensor_type = mlir::RankedTensorType::get({}, element_type);
  auto region_builder = [region_tensor_type,
                         accumulate](mlir::RegionBuilder& builder) {
    mlir::MlirOp arg0 = mlir::Argument(builder, region_tensor_type);
    mlir::MlirOp arg1 = mlir::Argument(builder, region_tensor_type);
    if (accumulate) {
      mlir::MlirOp result_val = stablehlo::Add(arg0, arg1);
      stablehlo::Return(builder, {result_val});
    } else {
      stablehlo::Return(builder, {arg1});
    }
  };

  return stablehlo::Scatter({self}, scatter_indices, {values}, region_builder,
                            scatter_dimension_numbers,
                            /*indices_are_sorted=*/false,
                            /*unique_indices=*/false)[0];
}

absl::StatusOr<mlir::MlirOp> ApplySelect(mlir::MlirOp self, mlir::MlirOp mask,
                                         const int64_t index_start_dim,
                                         mlir::MlirOp values, bool accumulate) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  const mlir::RankedTensorType mask_type = GetTensorTypeOrDie(mask);
  const mlir::RankedTensorType values_type = GetTensorTypeOrDie(values);
  ABSL_CHECK_EQ(values_type.getRank(), 0);  // CRASH_OK=values is a scalar

  // Reshape mask to:
  // [1, 1, ... , 1, D(dim0), D(dim1), ... , D(dim(maskRank-1)), 1, 1, ... , 1]
  Indices mask_shape(self_type.getRank(), 1);
  int64_t start_dim = index_start_dim;
  for (int64_t mask_dim = 0; mask_dim < mask_type.getRank(); ++mask_dim) {
    mask_shape[start_dim++] = mask_type.getDimSize(mask_dim);
  }

  mlir::MlirOp reshaped_mask = stablehlo::Reshape(mask, mask_shape);
  const mlir::RankedTensorType reshaped_mask_type =
      GetTensorTypeOrDie(reshaped_mask);
  ABSL_VLOG(1) << "[BuildIndexPutSelectShlo] reshaped mask: "
               << reshaped_mask.ToString();

  // Broadcast everything to a common shape.
  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=mask is reshaped to a shape
                        // that is broadcastable to self. values is a scalar
                        // and hence is broadcastable to self.
      const Dimensions common_broadcast_shape,
      InferSize(reshaped_mask_type.getShape(), self_type.getShape(),
                values_type.getShape()));

  TT_ASSIGN_OR_CRASH(  // CRASH_OK=self is being broadcasted to
                       // common_broadcast_shape, hence should always succeeds.
      self, BroadcastIfNeeded(self, common_broadcast_shape));

  TT_ASSIGN_OR_CRASH(  // CRASH_OK=mask is being broadcasted to
                       // common_broadcast_shape, hence should always succeeds.
      mask, BroadcastIfNeeded(reshaped_mask, common_broadcast_shape));

  TT_ASSIGN_OR_CRASH(  // CRASH_OK=values is being broadcasted to
                       // common_broadcast_shape, hence should always succeeds.
      values, BroadcastIfNeeded(values, common_broadcast_shape));

  ABSL_VLOG(1) << "[BuildIndexPutSelectShlo] self: " << self.ToString();
  ABSL_VLOG(1) << "[BuildIndexPutSelectShlo] mask: " << mask.ToString();
  ABSL_VLOG(1) << "[BuildIndexPutSelectShlo] values: " << values.ToString();

  if (accumulate) {
    // Accumulate: self + values
    values = stablehlo::Add(self, values);
  }
  return mlir::stablehlo::Select(mask, values, self);
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildIndexPutShlo(
    mlir::MlirOp self, mlir::ArrayRef<mlir::MlirOp> indices,
    const int64_t index_start_dim, const int64_t index_end_dim,
    const Dimensions& index_broadcast_shape, mlir::MlirOp values,
    const Dimensions& values_broadcast_shape, const bool accumulate) {
  return ApplyScatter(self, indices, index_start_dim, index_end_dim,
                      index_broadcast_shape, values, values_broadcast_shape,
                      accumulate);
}

absl::StatusOr<mlir::MlirOp> BuildIndexPutSelectShlo(
    mlir::MlirOp self, mlir::MlirOp mask, const int64_t index_start_dim,
    mlir::MlirOp values, const bool accumulate) {
  return ApplySelect(self, mask, index_start_dim, values, accumulate);
}

}  // namespace torch_tpu
