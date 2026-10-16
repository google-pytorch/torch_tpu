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

#include "csrc/ops/gather/gather.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {

struct BroadcastGatherPlan {
  mlir::MlirOp base_op;
  mlir::RankedTensorType base_type;
  Dimensions batch_dims;
  Dimensions rhs_batch_dims;
  Dimensions free_dims;
};

mlir::MlirOp HandleScalarGather(mlir::MlirOp self,
                                mlir::RankedTensorType self_type,
                                mlir::RankedTensorType index_type) {
  if (self_type.getRank() == index_type.getRank()) {
    return self;
  }
  if (self_type.getRank() == 0 && index_type.getRank() == 1 &&
      index_type.getShape()[0] == 0) {
    // Gathering from a scalar with an empty index results in an empty tensor.
    mlir::MlirBuilder& builder = self.getBuilder();
    mlir::Type element_type = self_type.getElementType();
    auto result_type = mlir::RankedTensorType::get({0}, element_type);
    auto empty_attr = mlir::DenseElementsAttr::get(
        result_type, llvm::ArrayRef<mlir::Attribute>());
    return stablehlo::Constant(builder, empty_attr);
  }
  return stablehlo::Reshape(self, index_type.getShape());
}

bool ParseBroadcastGatherAxes(mlir::RankedTensorType base_type,
                              mlir::RankedTensorType index_type,
                              llvm::ArrayRef<int64_t> bcast_dims,
                              Dimensions& base_pos_for_dim,
                              Dimensions& squeezed_shape,
                              bool& has_size1_expand) {
  const int64_t rank = index_type.getRank();
  for (int64_t i = 0; i < static_cast<int64_t>(bcast_dims.size()); ++i) {
    const int64_t d = bcast_dims[i];
    if (d < 0 || d >= rank || base_pos_for_dim[d] != -1) {
      return false;
    }
    const int64_t base_dim = base_type.getDimSize(i);
    const int64_t out_dim = index_type.getDimSize(d);
    if (base_dim == out_dim) {
      base_pos_for_dim[d] = static_cast<int64_t>(squeezed_shape.size());
      squeezed_shape.push_back(base_dim);
    } else if (base_dim == 1 && out_dim > 1) {
      has_size1_expand = true;
    } else {
      return false;
    }
  }
  return true;
}

bool ShapesMatchOffGatherDim(mlir::RankedTensorType self_type,
                             mlir::RankedTensorType index_type, int64_t dim) {
  const int64_t rank = self_type.getRank();
  for (int64_t d = 0; d < rank; ++d) {
    if (d != dim && self_type.getDimSize(d) != index_type.getDimSize(d)) {
      return false;
    }
  }
  return true;
}

void PartitionBroadcastGatherAxes(int64_t rank, int64_t dim,
                                  absl::Span<const int64_t> base_pos_for_dim,
                                  BroadcastGatherPlan& plan) {
  for (int64_t d = 0; d < rank; ++d) {
    if (d == dim) continue;
    if (base_pos_for_dim[d] != -1) {
      plan.batch_dims.push_back(d);
      plan.rhs_batch_dims.push_back(base_pos_for_dim[d]);
    } else {
      plan.free_dims.push_back(d);
    }
  }
}

std::optional<BroadcastGatherPlan> MatchBroadcastGatherIndex(
    mlir::MlirBuilder& builder, mlir::RankedTensorType self_type, int64_t dim,
    mlir::MlirOp index, mlir::RankedTensorType index_type) {
  mlir::Value raw_index_val = index.getValue();
  while (auto conv = mlir::dyn_cast_or_null<stablehlo::ConvertOp>(
             raw_index_val.getDefiningOp())) {
    raw_index_val = conv.getOperand();
  }
  auto broadcast = mlir::dyn_cast_or_null<stablehlo::BroadcastInDimOp>(
      raw_index_val.getDefiningOp());
  if (!broadcast) {
    return std::nullopt;
  }
  auto base_type =
      mlir::dyn_cast<mlir::RankedTensorType>(broadcast.getOperand().getType());
  if (!base_type || base_type.getRank() < 2 || !base_type.hasStaticShape() ||
      base_type.getNumElements() <= 0) {
    return std::nullopt;
  }
  const int64_t rank = self_type.getRank();
  Dimensions base_pos_for_dim(rank, -1);
  Dimensions squeezed_shape;
  bool has_size1_expand = false;
  if (!ParseBroadcastGatherAxes(
          base_type, index_type, broadcast.getBroadcastDimensions(),
          base_pos_for_dim, squeezed_shape, has_size1_expand) ||
      base_pos_for_dim[dim] == -1 ||
      !ShapesMatchOffGatherDim(self_type, index_type, dim)) {
    return std::nullopt;
  }
  const int64_t vocab_len = self_type.getDimSize(dim);
  const int64_t out_len = index_type.getDimSize(dim);
  if (vocab_len <= 0 || out_len <= 0 || vocab_len > 1024 || out_len > 1024 ||
      base_type.getNumElements() * vocab_len > 268435456) {
    return std::nullopt;
  }
  mlir::MlirOp base_op(builder, broadcast.getOperand());
  if (has_size1_expand) {
    base_op = stablehlo::Reshape(base_op, squeezed_shape);
    base_type = GetTensorTypeOrDie(base_op);
  }
  BroadcastGatherPlan plan{
      .base_op = base_op,
      .base_type = base_type,
  };
  PartitionBroadcastGatherAxes(rank, dim, base_pos_for_dim, plan);
  if (plan.free_dims.empty()) {
    return std::nullopt;
  }
  return plan;
}

mlir::MlirOp EmitBroadcastGatherDot(mlir::MlirOp self, int64_t dim,
                                    int64_t rank, int64_t vocab_len,
                                    mlir::Type computation_type,
                                    mlir::Type original_element_type,
                                    const BroadcastGatherPlan& plan) {
  mlir::MlirBuilder& builder = self.getBuilder();
  const mlir::Type i32_type =
      mlir::getElementType(self.getContext(), mlir::ElementType::I32);
  mlir::MlirOp base_i32 = stablehlo::ConvertElementType(plan.base_op, i32_type);
  const int64_t base_rank = plan.base_type.getRank();
  Dimensions ind_shape(plan.base_type.getShape().begin(),
                       plan.base_type.getShape().end());
  ind_shape.push_back(vocab_len);
  Dimensions bcast_to_ind(base_rank);
  for (int64_t i = 0; i < base_rank; ++i) {
    bcast_to_ind[i] = i;
  }
  auto ind_i32_type = mlir::RankedTensorType::get(ind_shape, i32_type);
  mlir::MlirOp idx_expanded =
      stablehlo::BroadcastInDim(ind_i32_type, base_i32, bcast_to_ind);
  mlir::MlirOp iota_p =
      stablehlo::Iota(builder, ind_i32_type, /*iota_dimension=*/base_rank);
  mlir::MlirOp mask = stablehlo::Compare(idx_expanded, iota_p,
                                         stablehlo::ComparisonDirection::EQ);
  auto ind_comp_type = mlir::RankedTensorType::get(ind_shape, computation_type);
  mlir::MlirOp one_scalar = MakeScalarConstant(builder, 1.0, computation_type);
  mlir::MlirOp zero_scalar = MakeScalarConstant(builder, 0.0, computation_type);
  mlir::MlirOp one = stablehlo::BroadcastInDim(ind_comp_type, one_scalar, {});
  mlir::MlirOp zero = stablehlo::BroadcastInDim(ind_comp_type, zero_scalar, {});
  mlir::MlirOp weights = stablehlo::Select(mask, one, zero);
  auto dot_dims = stablehlo::DotDimensionNumbersAttr::get(
      &self.getContext(),
      /*lhsBatchingDimensions=*/plan.batch_dims,
      /*rhsBatchingDimensions=*/plan.rhs_batch_dims,
      /*lhsContractingDimensions=*/{dim},
      /*rhsContractingDimensions=*/{base_rank});
  mlir::ArrayAttr precision_config;
  if (computation_type.isF32()) {
    const mlir::stablehlo::Precision precisions[2] = {
        mlir::stablehlo::Precision::
            HIGHEST,  // EXPLICIT_PRECISION_OK=f32_gather
        mlir::stablehlo::Precision::
            DEFAULT,  // EXPLICIT_PRECISION_OK=one_hot_rhs
    };
    precision_config =
        stablehlo::PrecisionConfigAttr::get(&self.getContext(), precisions);
  }
  mlir::MlirOp dot_out = stablehlo::DotGeneral(
      self, weights, dot_dims, /*precisionConfig=*/precision_config);
  Dimensions dot_out_axes;
  dot_out_axes.insert(dot_out_axes.end(), plan.batch_dims.begin(),
                      plan.batch_dims.end());
  dot_out_axes.insert(dot_out_axes.end(), plan.free_dims.begin(),
                      plan.free_dims.end());
  dot_out_axes.push_back(dim);
  Dimensions perm(rank);
  for (int64_t pos = 0; pos < rank; ++pos) {
    perm[dot_out_axes[pos]] = pos;
  }
  mlir::MlirOp result = stablehlo::Transpose(dot_out, perm);
  return stablehlo::ConvertElementType(result, original_element_type);
}

}  // namespace

absl::Status ValidateGatherInputs(absl::Span<const int64_t> self_dims,
                                  int64_t dim,
                                  absl::Span<const int64_t> index_dims,
                                  bool sparse_grad) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Error is caught by caller `Gather`
                 // function.
      sparse_grad == false, error::kPythonNotImplementedError)
      << "sparse_grad is not yet supported";

  const int64_t self_rank = self_dims.size();
  const int64_t index_rank = index_dims.size();

  // self and index should have the same rank, except when one of them is a
  // scalar and the other is a vector.
  if (self_rank == 0 || index_rank == 0) {
    // PyTorch allows 0D input with 1D index (if index size is 1)
    // and vice versa.
    if (self_rank == 0 && index_rank > 0) {
      TT_RET_CHECK(index_rank == 1 && index_dims[0] <= 1,
                   error::kInvalidArgument)
          << "expected the input and the index tensor to have the same number "
          << "of dimensions, got 0D vs " << index_rank << "D";
    } else if (index_rank == 0 && self_rank > 0) {
      TT_RET_CHECK(self_rank == 1 && self_dims[0] <= 1, error::kInvalidArgument)
          << "expected the input to be a 1D tensor with size at most 1 when "
             "index is 0D, got "
          << self_rank << "D with shape {" << self_dims[0] << "}";
    }
  } else {
    TT_RET_CHECK(self_rank == index_rank, error::kInvalidArgument)
        << "expected the input and the index tensor to have the same number of "
           "dimensions, got "
        << self_rank << "D vs " << index_rank << "D";

    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=SafeWrapDim function (called before
                   // this one) catches this error first.
        dim >= 0 && dim < self_rank, error::kInvalidArgument)
        << "expected the dim argument to be in the range [0, " << self_rank
        << "), got " << dim;

    for (auto d = 0; d < self_rank; ++d) {
      if (d != dim) {
        TT_RET_CHECK(index_dims[d] <= self_dims[d], error::kInvalidArgument)
            << "expected the index tensor to have size less than or equal to "
               "the input tensor at dimension "
            << d << ", got " << index_dims[d] << " vs " << self_dims[d];
      }
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<mlir::MlirOp> BuildGatherShlo(
    mlir::MlirOp self, int64_t dim, mlir::MlirOp index, bool sparse_grad,
    mlir::ElementType computation_element_type) {
  mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  const mlir::RankedTensorType index_type = GetTensorTypeOrDie(index);

  ABSL_VLOG(2) << "BuildGatherShlo:"
               << ", dim: " << dim
               << ", self_type: " << mlir::debugString(self_type)
               << ", index_type: " << mlir::debugString(index_type)
               << ", sparse_grad: " << (sparse_grad ? "true" : "false");

  TT_RETURN_IF_ERROR(ValidateGatherInputs(self_type.getShape(), dim,
                                          index_type.getShape(), sparse_grad));

  // self and index should have the same rank, except when one of them is a
  // scalar and the other is a vector.
  if (self_type.getRank() == 0 || index_type.getRank() == 0) {
    return HandleScalarGather(self, self_type, index_type);
  }

  const mlir::Type original_element_type = self_type.getElementType();

  // Convert arguments to the computation type if necessary.
  mlir::Type computation_type =
      mlir::getElementType(self.getContext(), computation_element_type);
  ABSL_VLOG(2) << "computation_type: " << mlir::debugString(computation_type);
  if (self_type.getElementType() != computation_type) {
    self = stablehlo::ConvertElementType(self, computation_type);
  }

  self = SliceBatchDimensions(self, dim, index_type);
  self_type = GetTensorTypeOrDie(self);
  const int64_t rank = self_type.getRank();

  // Fast path: when `index` is broadcast along one or more non-gather
  // dimensions `free_dims` (either via rank-expanding `BroadcastInDim` or via
  // size-1 axis broadcast `[B, 1, Q, K] -> [B, H, Q, K]`) and the gather axis
  // is bounded, contract `self` against the shared 1-bit indicator matrix on
  // the MXU instead of materializing a full-rank per-element gather index
  // tensor.
  if ((computation_type.isF32() || computation_type.isBF16()) && rank >= 3 &&
      self_type.hasStaticShape() && index_type.hasStaticShape()) {
    if (std::optional<BroadcastGatherPlan> plan = MatchBroadcastGatherIndex(
            self.getBuilder(), self_type, dim, index, index_type)) {
      return EmitBroadcastGatherDot(self, dim, rank, self_type.getDimSize(dim),
                                    computation_type, original_element_type,
                                    *plan);
    }
  }

  TT_ASSIGN_OR_RETURN(mlir::MlirOp gather_indices, Unsqueeze(index, -1));
  const Dimensions batching_dims = GetBatchDimensions(rank, dim);
  const stablehlo::GatherDimensionNumbersAttr gather_dimension_numbers =
      stablehlo::GatherDimensionNumbersAttr::get(
          &self.getContext(),
          /*offset_dims=*/{},
          /*collapsed_slice_dims=*/{dim},
          /*operand_batching_dims=*/batching_dims,
          /*start_indices_batching_dims=*/batching_dims,
          /*start_index_map=*/{dim},
          /*index_vector_dim=*/rank);
  ABSL_VLOG(2) << "BuildGatherShlo: GatherDimensionNumbers = "
               << mlir::debugString(gather_dimension_numbers);

  auto slice_sizes = std::vector<int64_t>(rank, 1);  // INT_VEC_OK
  for (int64_t d = 0; d < rank; ++d) {
    if (self_type.getShape()[d] == 0) {
      slice_sizes[d] = 0;
    }
  }
  auto result = stablehlo::Gather(self, gather_indices,
                                  gather_dimension_numbers, slice_sizes,
                                  /*indices_are_sorted=*/false);
  return stablehlo::ConvertElementType(result, original_element_type);
}

}  // namespace torch_tpu
