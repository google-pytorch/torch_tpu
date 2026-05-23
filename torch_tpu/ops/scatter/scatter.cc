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
#include <limits>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {
namespace {

// Returns the identity value for the given scatter op and computation type
// only when include_self is kNo.
mlir::MlirOp GetIdentityValue(mlir::MlirBuilder& builder, mlir::Type type,
                              ScatterOp op) {
  if (op == ScatterOp::kProd || op == ScatterOp::kMul) {
    return MakeScalarConstant(builder, 1.0, type);
  }
  if (op == ScatterOp::kAmax || op == ScatterOp::kAmin) {
    const bool is_max = (op == ScatterOp::kAmax);
    if (mlir::isa<mlir::FloatType>(type)) {
      double val = is_max ? -std::numeric_limits<double>::infinity()
                          : std::numeric_limits<double>::infinity();
      return MakeScalarConstant(builder, val, type);
    }
    mlir::Attribute attr =
        is_max ? GetMinFiniteValueAttr(type, builder.getOpBuilder())
               : GetMaxFiniteValueAttr(type, builder.getOpBuilder());
    return mlir::stablehlo::Constant(
        builder, mlir::DenseElementsAttr::get(
                     mlir::RankedTensorType::get({}, type), attr));
  }
  // kSum, kAdd, kMean, kReplace have 0.0 as identity value.
  // For kMean, sum and count are computed separately, so identity should be
  // 0.0 when include_self is kNo.
  return MakeScalarConstant(builder, 0.0, type);
}

// Builds the scatter_indices tensor for the given index tensor and dimension
// to be used in the StableHLO scatter op.
mlir::MlirOp BuildScatterIndices(mlir::MlirOp index, int64_t dim) {
  mlir::MlirBuilder& builder = index.getBuilder();
  mlir::RankedTensorType index_type = GetTensorTypeOrDie(index);
  int64_t rank = index_type.getRank();
  Dimensions expanded_index_shape = CopyIntVector(index_type.getShape());
  expanded_index_shape.push_back(1);
  mlir::Type index_element_type = index_type.getElementType();

  // scatter_indices has one more dimension that src/index, and contains
  // [i_1, i_2, ..., i_{dim-1}, index[i_1, i_2, ..., i_k], i_{dim+1}, ...,  i_k]
  // for each element [i_1, i_2, ..., i_k] from the index space of src/index.
  std::vector<mlir::MlirOp> parts;
  parts.reserve(rank);
  for (int d = 0; d < rank; ++d) {
    if (d == dim) {
      mlir::MlirOp expanded_index =
          mlir::stablehlo::Reshape(index, expanded_index_shape);
      parts.push_back(expanded_index);
    } else {
      mlir::MlirOp j_part = mlir::stablehlo::Iota(
          builder,
          makeTensorType(builder.getContext(), index_type.getShape(),
                         index_element_type),
          /*iota_dimension=*/d);
      j_part = mlir::stablehlo::Reshape(j_part, expanded_index_shape);
      parts.push_back(j_part);
    }
  }
  return mlir::stablehlo::Concatenate(builder, parts, /*dimension=*/rank);
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildScatterShlo(
    mlir::MlirOp self, int64_t dim, mlir::MlirOp index, mlir::MlirOp src,
    ScatterOp scatter_op, mlir::ElementType computation_element_type,
    ScatterIncludeSelf include_self) {
  mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  mlir::RankedTensorType src_type = GetTensorTypeOrDie(src);
  mlir::RankedTensorType index_type = GetTensorTypeOrDie(index);
  ABSL_VLOG(2) << "BuildScatterShlo:"
               << ", dim: " << dim
               << ", self_type: " << mlir::debugString(self_type)
               << ", src_type: " << mlir::debugString(src_type)
               << ", index_type: " << mlir::debugString(index_type)
               << ", scatter_op: " << FormatParamCacheKey(scatter_op)
               << ", include_self: " << FormatParamCacheKey(include_self);

  // If arguments are scalars, temporarily reshape them to rank 1.
  bool are_scalars = self_type.getRank() == 0 || src_type.getRank() == 0 ||
                     index_type.getRank() == 0;
  if (are_scalars) {
    if (self_type.getRank() == 0) {
      self = mlir::stablehlo::Reshape(self, {1});
      self_type = GetTensorTypeOrDie(self);
    }
    if (src_type.getRank() == 0) {
      src = mlir::stablehlo::Reshape(src, {1});
      src_type = GetTensorTypeOrDie(src);
    }
    if (index_type.getRank() == 0) {
      index = mlir::stablehlo::Reshape(index, {1});
      index_type = GetTensorTypeOrDie(index);
    }
  }

  TT_RET_CHECK(self_type.getRank() == src_type.getRank(),
               error::kInvalidArgument)
      << "expected the self tensor of shape " << ToString(self_type.getShape())
      << " to have the same rank as the src tensor of shape "
      << ToString(src_type.getShape()) << ", got " << self_type.getRank()
      << " vs. " << src_type.getRank();
  TT_RET_CHECK(self_type.getRank() == index_type.getRank(),
               error::kInvalidArgument)
      << "expected the self tensor of shape " << ToString(self_type.getShape())
      << " to have the same rank as the index tensor of shape "
      << ToString(index_type.getShape()) << ", got " << self_type.getRank()
      << " vs. " << index_type.getRank();

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Caught by the `SafeWrapDim()` function
                 // call in the caller.
      dim >= 0 && dim < self_type.getRank(), error::kInvalidArgument)
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

  mlir::MlirOp scatter_indices = BuildScatterIndices(index, dim);

  // Slice src to match index shape if necessary. PyTorch allows src to be
  // larger than index, but StableHLO requires exact match for updates.
  if (src_type.getShape() != index_type.getShape()) {
    Indices start_indices(src_type.getRank(), 0);
    Indices limit_indices = CopyIntVector(index_type.getShape());
    Indices strides(src_type.getRank(), 1);
    src = mlir::stablehlo::Slice(src, start_indices, limit_indices, strides);
  }

  mlir::stablehlo::ScatterDimensionNumbersAttr scatter_dimension_numbers =
      mlir::stablehlo::ScatterDimensionNumbersAttr::get(
          &self.getContext(),
          /*update_window_dims=*/{},
          /*inserted_window_dims=*/all_dimensions,
          /*input_batching_dims=*/{},
          /*scatter_indices_batching_dims=*/{},
          /*scatter_dims_to_operand_dims=*/all_dimensions,
          /*index_vector_dim=*/index_type.getRank());
  ABSL_VLOG(2) << "BuildScatterShlo: ScatterDimensionNumbers = "
               << mlir::debugString(scatter_dimension_numbers);

  // Handle include_self=false by resetting scattered locations to identity
  auto block_type = mlir::RankedTensorType::get({}, computation_type);

  if (include_self == ScatterIncludeSelf::kNo) {
    mlir::MlirOp identity =
        GetIdentityValue(builder, computation_type, scatter_op);
    auto identity_updates_type = index_type.clone(computation_type);
    mlir::MlirOp identity_updates =
        mlir::stablehlo::BroadcastInDim(identity_updates_type, identity, {});

    self = mlir::stablehlo::Scatter(
        {self}, scatter_indices, {identity_updates},
        [block_type](mlir::RegionBuilder& rb) {
          // Ignore current value and return a1 (identity)
          mlir::Argument(rb, block_type);
          auto arg1 = mlir::Argument(rb, block_type);
          mlir::stablehlo::Return(rb, {arg1});
        },
        scatter_dimension_numbers, /*indices_are_sorted=*/false,
        /*unique_indices=*/false)[0];
  }

  // Create a region builder callback depending on the scatter op.
  mlir::RegionBuilderCallback region_builder;
  region_builder = [block_type, scatter_op](mlir::RegionBuilder& builder) {
    auto arg0 = mlir::Argument(builder, block_type);  // current
    auto arg1 = mlir::Argument(builder, block_type);  // update
    mlir::MlirOp result;
    switch (scatter_op) {
      case ScatterOp::kAdd:
      case ScatterOp::kSum:
      case ScatterOp::kMean:  // Compute sum separately first
        result = mlir::stablehlo::Add(arg0, arg1);
        break;
      case ScatterOp::kMul:
      case ScatterOp::kProd:
        result = mlir::stablehlo::Mul(arg0, arg1);
        break;
      case ScatterOp::kAmax:
        result = mlir::stablehlo::Max(arg0, arg1);
        break;
      case ScatterOp::kAmin:
        result = mlir::stablehlo::Min(arg0, arg1);
        break;
      case ScatterOp::kReplace:
        result = arg1;
        break;
    }
    mlir::stablehlo::Return(builder, {result});
  };

  auto result = mlir::stablehlo::Scatter(
      {self}, scatter_indices, {src}, region_builder, scatter_dimension_numbers,
      /*indices_are_sorted=*/false, /*unique_indices=*/false)[0];

  // Handle mean separately as it requires division by the number of updates
  if (scatter_op == ScatterOp::kMean) {
    mlir::MlirOp one = MakeScalarConstant(builder, 1.0, computation_type);
    mlir::MlirOp zero = MakeScalarConstant(builder, 0.0, computation_type);

    auto updates_count_type = index_type.clone(computation_type);
    mlir::MlirOp ones_updates =
        mlir::stablehlo::BroadcastInDim(updates_count_type, one, {});

    auto total_count_type = self_type.clone(computation_type);
    mlir::MlirOp initial_count = mlir::stablehlo::BroadcastInDim(
        total_count_type, include_self == ScatterIncludeSelf::kYes ? one : zero,
        {});

    // Count the number of updates for each element
    auto count_reducer = [block_type](mlir::RegionBuilder& rb) {
      auto a0 = mlir::Argument(rb, block_type);
      auto a1 = mlir::Argument(rb, block_type);
      mlir::stablehlo::Return(rb, {mlir::stablehlo::Add(a0, a1)});
    };
    mlir::MlirOp total_count = mlir::stablehlo::Scatter(
        {initial_count}, scatter_indices, {ones_updates}, count_reducer,
        scatter_dimension_numbers, false, false)[0];

    // To avoid division by zero (for include_self=false and unindexed
    // locations), use the original self for selection.
    mlir::MlirOp div_res = mlir::stablehlo::Div(result, total_count);
    mlir::MlirOp zero_bcast =
        mlir::stablehlo::BroadcastInDim(total_count_type, zero, {});
    mlir::MlirOp is_hit = mlir::stablehlo::Compare(
        total_count, zero_bcast, mlir::stablehlo::ComparisonDirection::GT);
    result = mlir::stablehlo::Select(is_hit, div_res, self);
  }

  result =
      mlir::stablehlo::ConvertElementType(result, self_type.getElementType());
  if (are_scalars) {
    result = mlir::stablehlo::Reshape(result, {});
  }
  return result;
}

}  // namespace torch_tpu
