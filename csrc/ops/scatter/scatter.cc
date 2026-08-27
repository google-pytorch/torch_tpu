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

#include "torch_tpu/csrc/ops/scatter/scatter.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/csrc/common/dimension_types.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/common/utils.h"
#include "torch_tpu/csrc/ops/index_add/index_add.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"

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

// If `index` is a broadcast of a 1-D tensor that is constant along every
// dimension except `dim` -- i.e. the `ids[:, None].expand_as(src)` form emitted
// by the idiomatic top-k MoE token-combine -- returns that original 1-D tensor,
// else nullopt. For such an index the per-element scatter update
// `out[index[i][j], j] += src[i][j]` is invariant in `j`, so a scatter-add
// collapses exactly to an index_add over the 1-D tensor. Both eager and
// compiled execution hand this builder the broadcast, so detecting it here
// fixes the lowering for every execution mode.
std::optional<mlir::MlirOp> GetRowBroadcastIndex(mlir::MlirOp index,
                                                 int64_t dim) {
  auto broadcast = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
      index.getValue().getDefiningOp());
  if (!broadcast) {
    return std::nullopt;
  }
  auto operand_type =
      mlir::dyn_cast<mlir::RankedTensorType>(broadcast.getOperand().getType());
  if (!operand_type || operand_type.getRank() != 1) {
    return std::nullopt;
  }
  // The single input dim must map only to the scatter dim, so the broadcast
  // introduces constant values along every other dim.
  llvm::ArrayRef<int64_t> broadcast_dims = broadcast.getBroadcastDimensions();
  if (broadcast_dims.size() != 1 || broadcast_dims[0] != dim) {
    return std::nullopt;
  }
  // The pre-broadcast operand is the original 1-D index (length
  // index.size(dim)).
  return mlir::MlirOp(index.getBuilder(), broadcast.getOperand());
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
               << ", scatter_op: " << EncodeParamCacheKey(scatter_op)
               << ", include_self: " << EncodeParamCacheKey(include_self);

  if (src_type.getRank() == 0) {
    TT_ASSIGN_OR_RETURN(src, BroadcastIfNeeded(src, index));
    src_type = GetTensorTypeOrDie(src);
  }

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

  // The fast path rewrites the scatter as an index_add, which adds each `src`
  // slice onto a same-shaped slice of `self`, so it requires `self` and `src`
  // to have equal extents on every dim except `dim`. Plain scatter_add allows
  // `self` to be larger on those dims (writing only a sub-slice of each row),
  // which index_add cannot express -- so restrict the fast path to the case
  // where the shapes match off the scatter dim.
  bool self_src_match_off_scatter_dim = true;
  for (int d = 0; d < self_type.getRank(); ++d) {
    if (d != dim && self_type.getDimSize(d) != src_type.getDimSize(d)) {
      self_src_match_off_scatter_dim = false;
      break;
    }
  }

  // Fast path for a broadcast index (`ids[:, None].expand_as(src)`, the top-k
  // MoE token-combine idiom). When the index is constant along every dim except
  // `dim`, scatter_add is exactly index_add over the original 1-D index, which
  // lowers to an efficient row (window) scatter -- far faster on TPU than the
  // per-element scalar scatter built below, which materializes a full index
  // tensor. Restricted to what index_add can express: add/sum with include_self
  // and matching shapes.
  const mlir::Type original_element_type = self_type.getElementType();
  if ((scatter_op == ScatterOp::kAdd || scatter_op == ScatterOp::kSum) &&
      include_self == ScatterIncludeSelf::kYes &&
      self_src_match_off_scatter_dim &&
      src_type.getShape() == index_type.getShape()) {
    if (std::optional<mlir::MlirOp> index_1d =
            GetRowBroadcastIndex(index, dim)) {
      mlir::MlirBuilder& fast_builder = self.getBuilder();
      mlir::Type fast_computation_type =
          mlir::getElementType(self.getContext(), computation_element_type);
      mlir::MlirOp alpha =
          MakeScalarConstant(fast_builder, 1.0, fast_computation_type);
      TT_ASSIGN_OR_RETURN(mlir::MlirOp result,
                          BuildIndexAddShlo(self, dim, *index_1d, src, alpha,
                                            computation_element_type));
      if (GetTensorTypeOrDie(result).getElementType() !=
          original_element_type) {
        result =
            mlir::stablehlo::ConvertElementType(result, original_element_type);
      }
      return result;
    }
  }

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

  // Slice src to match index shape if necessary. PyTorch allows src to be
  // larger than index, but StableHLO requires exact match for updates.
  if (src_type.getShape() != index_type.getShape()) {
    Indices start_indices(src_type.getRank(), 0);
    Indices limit_indices = CopyIntVector(index_type.getShape());
    Indices strides(src_type.getRank(), 1);
    src = mlir::stablehlo::Slice(src, start_indices, limit_indices, strides);
  }

  mlir::MlirOp full_self = self;
  self = SliceBatchDimensions(self, dim, index_type);
  self_type = GetTensorTypeOrDie(self);
  const int64_t rank = self_type.getRank();
  TT_ASSIGN_OR_RETURN(mlir::MlirOp scatter_indices, Unsqueeze(index, -1));
  const Dimensions batching_dims = GetBatchDimensions(rank, dim);

  const mlir::stablehlo::ScatterDimensionNumbersAttr scatter_dimension_numbers =
      mlir::stablehlo::ScatterDimensionNumbersAttr::get(
          &self.getContext(),
          /*update_window_dims=*/{},
          /*inserted_window_dims=*/{dim},
          /*input_batching_dims=*/batching_dims,
          /*scatter_indices_batching_dims=*/batching_dims,
          /*scatter_dims_to_operand_dims=*/{dim},
          /*index_vector_dim=*/rank);
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

    auto total_count_type = GetTensorTypeOrDie(self).clone(computation_type);
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
        scatter_dimension_numbers, /*indices_are_sorted=*/false,
        /*unique_indices=*/false)[0];

    // To avoid division by zero (for include_self=false and unindexed
    // locations), use the original self for selection.
    mlir::MlirOp div_res = mlir::stablehlo::Div(result, total_count);
    mlir::MlirOp zero_bcast =
        mlir::stablehlo::BroadcastInDim(total_count_type, zero, {});
    mlir::MlirOp is_hit = mlir::stablehlo::Compare(
        total_count, zero_bcast, mlir::stablehlo::ComparisonDirection::GT);
    result = mlir::stablehlo::Select(is_hit, div_res, self);
  }

  const bool is_sliced = GetTensorTypeOrDie(self).getShape() !=
                         GetTensorTypeOrDie(full_self).getShape();
  if (is_sliced) {
    const mlir::MlirOp zero =
        MakeScalarConstant(builder, 0, mlir::ElementType::I64);
    result = mlir::stablehlo::DynamicUpdateSlice(
        full_self, result, std::vector<mlir::MlirOp>(rank, zero));
  }

  result = mlir::stablehlo::ConvertElementType(result, original_element_type);

  if (are_scalars) {
    result = mlir::stablehlo::Reshape(result, {});
  }
  return result;
}

}  // namespace torch_tpu
