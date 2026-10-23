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

#include "csrc/ops/index/index.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/nonzero/nonzero.h"
#include "csrc/ops/op_builder_utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
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
#include "stablehlo/transforms/StablehloBroadcastLowering.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {

constexpr int64_t kMinUniformRowWidth = 256;
constexpr int64_t kMinUniformElements = 4096;
constexpr int64_t kMinUniformRows = 4;

struct DynamicUnindexedDim {
  mlir::stablehlo::DimensionInfo dim_info;
  int64_t out_dim;
};

bool AreConsecutive(const Indices& indexed_dims) {
  for (size_t i = 1; i < indexed_dims.size(); ++i) {
    if (indexed_dims[i] != indexed_dims[i - 1] + 1) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<mlir::MlirOp> BuildRowUniformMaskIndexTrueBranch(
    mlir::RegionBuilder& true_rb, mlir::MlirOp self, mlir::MlirOp col0,
    int64_t dim_indexed, int64_t mask_rank, int64_t num_rows, int64_t row_width,
    int64_t selected_rows, mlir::RankedTensorType out_type) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  const mlir::ArrayRef<int64_t> self_shape = self_type.getShape();
  const mlir::Type i1_type = true_rb.getOpBuilder().getI1Type();
  mlir::MlirOp true_col0(true_rb, col0.getValue());
  mlir::MlirOp col0_1d = stablehlo::Reshape(
      mlir::makeTensorType(true_rb.getContext(), {num_rows}, i1_type),
      true_col0);
  mlir::MlirOp iota_r =
      stablehlo::Iota(true_rb,
                      mlir::makeTensorType(true_rb.getContext(), {num_rows},
                                           mlir::ElementType::I32),
                      /*dim=*/0);
  auto comparator = [i1_type](mlir::RegionBuilder& rb) {
    mlir::OpBuilder& op_builder = rb.getOpBuilder();
    stablehlo::buildSortComparisonBody(
        {i1_type, op_builder.getI32Type()}, stablehlo::ComparisonDirection::GT,
        /*compare_type=*/std::nullopt, &rb.getRegion(), &op_builder);
  };
  auto sort_outputs = stablehlo::Sort(true_rb, {col0_1d, iota_r}, comparator,
                                      /*dim=*/0, /*is_stable=*/true);
  mlir::MlirOp selected_row_indices = stablehlo::Slice(
      sort_outputs[1], /*start_indices=*/{0},
      /*limit_indices=*/{selected_rows}, /*stride_indices=*/{1});

  Dimensions self_rows_shape;
  self_rows_shape.reserve(self_shape.size() - mask_rank + 2);
  for (int64_t i = 0; i < dim_indexed; ++i) {
    self_rows_shape.push_back(self_shape[i]);
  }
  self_rows_shape.push_back(num_rows);
  self_rows_shape.push_back(row_width);
  for (size_t i = dim_indexed + mask_rank; i < self_shape.size(); ++i) {
    self_rows_shape.push_back(self_shape[i]);
  }

  mlir::MlirOp true_self(true_rb, self.getValue());
  mlir::MlirOp self_rows = stablehlo::Reshape(
      mlir::makeTensorType(true_rb.getContext(), self_rows_shape,
                           self_type.getElementType()),
      true_self);
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp gathered_rows,
      BuildIndexShlo({self_rows, selected_row_indices}, {dim_indexed}));
  return stablehlo::Reshape(out_type, gathered_rows);
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildIndexShlo(
    mlir::ArrayRef<mlir::MlirOp> input_ops, Indices indexed_dims) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=AtenIndexTensorOut (caller) makes sure
                 // `input_ops` has the `self` input and, at least, one index.
      input_ops.size() >= 2, error::kInvalidArgument)
      << "[BuildIndexShlo]: requires at least two input ops: an operand and an "
         "indexing tensor";
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=AtenIndexTensorOut (caller) creates
                 // `indexed_dims`, making sure there's always a corresponding
                 // index.
      input_ops.size() == indexed_dims.size() + 1, error::kInvalidArgument)
      << "[BuildIndexShlo]: requires exactly one indexing tensor per indexed "
         "dimension, plus one for the operand";

  mlir::MlirOp self = input_ops[0];
  mlir::ArrayRef<mlir::MlirOp> indices = input_ops.drop_front();
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);

  if (self_type.getRank() == 0) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=AtenIndexTensorOut (caller) errors
                   // before calling this function, when rank == 0.
        indexed_dims.size() == 1 && indexed_dims[0] == 0,
        error::kInvalidArgument)
        << "[BuildIndexShlo]: if the operand is a scalar, the only indexed "
           "dimension must be 0";
    return self;
  }

  // Indexing tensors need to be broadcastable together.
  TT_ASSIGN_OR_RETURN(std::vector<mlir::MlirOp> broadcasted_indices,
                      ApplyBroadcastIfNeeded(indices));

  const int64_t broadcasted_rank =
      GetTensorTypeOrDie(broadcasted_indices[0]).getRank();

  std::vector<mlir::MlirOp> reshaped_indices;
  reshaped_indices.reserve(broadcasted_indices.size());
  for (const mlir::MlirOp& broadcasted_index : broadcasted_indices) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp reshaped_index,
                        Unsqueeze(broadcasted_index, broadcasted_rank));
    reshaped_indices.push_back(reshaped_index);
  }
  mlir::MlirOp index = mlir::stablehlo::Concatenate(
      self.getBuilder(), reshaped_indices, broadcasted_rank);

  // Dynamism in Gather:
  // - `gather_slice_sizes` is a static attribute in StableHLO and must contain
  //   compile-time non-negative integers. Therefore, we use `self_dims[i].size`
  //   (the upper bound / physical dimension size) rather than
  //   `self_type.getShape()[i]` which evaluates to `kDynamic` (-1) for dynamic
  //   dimensions.
  // - For indexed dimensions, the slice size is 1. If `index` has dynamic
  //   dimensions, StableHLO Gather automatically preserves and propagates those
  //   dynamic batch dimensions from `index` (via `start_indices`) into the
  //   result type.
  // - For unindexed dimensions, the slice size extracts the full upper-bound
  //   extent from `self`. If an unindexed dimension is dynamic, we re-attach
  //   its dynamic runtime bound onto the Gather output via `SetDimensionSize`.
  mlir::stablehlo::Dimensions self_dims = GetDimensions(self);
  Dimensions gather_slice_sizes;
  gather_slice_sizes.reserve(self_dims.size());
  Indices offset_dims;

  bool index_dims_consecutive = AreConsecutive(indexed_dims);

  std::vector<DynamicUnindexedDim> dynamic_unindexed_dims;

  for (size_t i = 0, j = 0; i < self_dims.size(); ++i) {
    if (j < indexed_dims.size() && indexed_dims[j] == i) {
      gather_slice_sizes.push_back(1);
      ++j;
    } else {
      gather_slice_sizes.push_back(self_dims[i].size);
      TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=AtenIndexTensorOut (caller) creates
                     // `indexed_dims`, making sure they are always increasing,
                     // non-negative, and distinct.
          j >= indexed_dims.size() || i < indexed_dims[j],
          error::kInvalidArgument)
          << "[BuildIndexShlo]: indexed dimensions must be increasing, "
             "non-negative and distinct";
      // If the indexed dimensions are consecutive, the indexed dimensions will
      // be inserted at the position of the first indexed dimension. Otherwise,
      // they will be inserted at the beginning of the output tensor shape.
      // offset_dims contains the positions of the non-indexed dimensions in the
      // output tensor shape.
      // See
      // https://numpy.org/devdocs/user/basics.indexing.html#combining-advanced-and-basic-indexing
      int64_t out_dim =
          index_dims_consecutive && j == 0 ? i : broadcasted_rank + i - j;
      offset_dims.push_back(out_dim);
      if (self_dims[i].boundOp.has_value()) {
        dynamic_unindexed_dims.push_back({self_dims[i], out_dim});
      }
    }
  }

  ABSL_VLOG(2) << "[BuildIndexShlo]: gather_slice_sizes = "
               << absl::StrJoin(gather_slice_sizes, ",");

  stablehlo::GatherDimensionNumbersAttr gather_dimension_numbers =
      stablehlo::GatherDimensionNumbersAttr::get(
          &self.getContext(),
          /*offset_dims=*/offset_dims,
          /*collapsed_slice_dims=*/indexed_dims,
          /*operand_batching_dims=*/{},
          /*start_indices_batching_dims=*/{},
          /*start_index_map=*/indexed_dims,
          /*index_vector_dim=*/broadcasted_rank);
  ABSL_VLOG(2) << "[BuildIndexShlo]: GatherDimensionNumbers = "
               << mlir::debugString(gather_dimension_numbers);

  auto result = stablehlo::Gather(self, index, gather_dimension_numbers,
                                  gather_slice_sizes,
                                  /*indices_are_sorted=*/false);
  for (const auto& [dim_info, out_dim] : dynamic_unindexed_dims) {
    mlir::MlirOp bound_op_mlir(self.getBuilder(), *dim_info.boundOp);
    mlir::MlirOp dim_size =
        mlir::stablehlo::GetDimensionSize(bound_op_mlir, dim_info.boundOpDim);
    result = mlir::stablehlo::SetDimensionSize(result, dim_size, out_dim);
  }
  return stablehlo::ConvertElementType(result, self_type.getElementType());
}

absl::StatusOr<mlir::MlirOp> BuildBooleanMaskIndexScalarShlo(
    mlir::MlirOp self, mlir::MlirOp mask, int64_t dim_indexed,
    int64_t num_non_zero_elements) {
  const int64_t mask_rank = GetTensorTypeOrDie(mask).getRank();

  if (mask_rank == 0) {
    TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=dim_indexed is guaranteed to
                          // be in [0, self.dim()].
        self, Unsqueeze(self, dim_indexed));
    mask = Flatten(mask);
  } else if (mask_rank > 1) {
    mlir::stablehlo::Dimensions self_dims = GetDimensions(self);
    Dimensions static_shape_before;
    static_shape_before.reserve(self_dims.size());
    for (const auto& dim : self_dims) {
      static_shape_before.push_back(dim.size);
    }

    Dimensions static_shape_after;
    static_shape_after.reserve(self_dims.size() - mask_rank + 1);
    for (int64_t i = 0; i < dim_indexed; ++i) {
      static_shape_after.push_back(static_shape_before[i]);
    }
    int64_t collapsed_dim_size = 1;
    for (int64_t i = dim_indexed; i < dim_indexed + mask_rank; ++i) {
      collapsed_dim_size *= static_shape_before[i];
    }
    static_shape_after.push_back(collapsed_dim_size);
    for (int64_t i = dim_indexed + mask_rank; i < static_shape_before.size();
         ++i) {
      static_shape_after.push_back(static_shape_before[i]);
    }

    TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=collapsing contiguous valid
                          // dimensions always succeeds.
        self, ReshapeFromStaticDimensions(self, static_shape_before,
                                          static_shape_after));
    mask = Flatten(mask);
  }

  if (GetTensorTypeOrDie(self).getDimSize(dim_indexed) == 0) {
    return self;
  }

  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=1D mask nonzero always succeeds.
      mlir::MlirOp nonzero_indices,
      BuildNonzeroShlo(num_non_zero_elements, mask));
  mlir::MlirOp flat_indices = Flatten(nonzero_indices);

  return BuildIndexShlo({self, flat_indices}, {dim_indexed});
}

absl::StatusOr<mlir::MlirOp> BuildBooleanMaskIndexShlo(
    mlir::MlirOp self, mlir::MlirOp mask, int64_t dim_indexed,
    int64_t num_non_zero_elements) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  const mlir::RankedTensorType mask_type = GetTensorTypeOrDie(mask);
  const int64_t mask_rank = mask_type.getRank();
  if (mask_rank < 3 || !self_type.hasStaticShape() ||
      !mask_type.hasStaticShape() ||
      mask_type.getNumElements() < kMinUniformElements ||
      mask_type.getNumElements() > std::numeric_limits<int32_t>::max()) {
    return BuildBooleanMaskIndexScalarShlo(self, mask, dim_indexed,
                                           num_non_zero_elements);
  }

  const mlir::ArrayRef<int64_t> mask_shape = mask_type.getShape();
  const mlir::ArrayRef<int64_t> self_shape = self_type.getShape();
  const int64_t row_width = mask_shape[mask_rank - 1];
  int64_t collapsed_dim_size = 1;
  for (int64_t i = dim_indexed; i < dim_indexed + mask_rank; ++i) {
    collapsed_dim_size *= self_shape[i];
  }
  const int64_t num_rows = collapsed_dim_size / row_width;
  if (row_width < kMinUniformRowWidth || num_rows < kMinUniformRows ||
      self_shape[dim_indexed + mask_rank - 1] != row_width ||
      collapsed_dim_size < row_width || num_non_zero_elements < row_width ||
      num_non_zero_elements % row_width != 0) {
    return BuildBooleanMaskIndexScalarShlo(self, mask, dim_indexed,
                                           num_non_zero_elements);
  }

  const int64_t selected_rows = num_non_zero_elements / row_width;
  mlir::MlirBuilder& builder = self.getBuilder();
  mlir::MlirOp col0;
  mlir::MlirOp is_row_uniform =
      SliceMaskFirstColumnAndCheckRowUniform(mask, col0);

  Dimensions out_shape;
  out_shape.reserve(self_shape.size() - mask_rank + 1);
  for (int64_t i = 0; i < dim_indexed; ++i) {
    out_shape.push_back(self_shape[i]);
  }
  out_shape.push_back(num_non_zero_elements);
  for (size_t i = dim_indexed + mask_rank; i < self_shape.size(); ++i) {
    out_shape.push_back(self_shape[i]);
  }
  const mlir::RankedTensorType out_type = mlir::makeTensorType(
      builder.getContext(), out_shape, self_type.getElementType());

  auto if_op = stablehlo::IfOp::create(builder.getOpBuilder(), builder.getLoc(),
                                       mlir::TypeRange{out_type},
                                       is_row_uniform.getValue());

  {
    mlir::RegionBuilder true_rb(builder, if_op->getRegion(0));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp true_out,
                        BuildRowUniformMaskIndexTrueBranch(
                            true_rb, self, col0, dim_indexed, mask_rank,
                            num_rows, row_width, selected_rows, out_type));
    stablehlo::Return(true_rb, {true_out});
  }

  {
    mlir::RegionBuilder false_rb(builder, if_op->getRegion(1));
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp false_out,
        BuildBooleanMaskIndexScalarShlo(mlir::MlirOp(false_rb, self.getValue()),
                                        mlir::MlirOp(false_rb, mask.getValue()),
                                        dim_indexed, num_non_zero_elements));
    stablehlo::Return(false_rb, {false_out});
  }

  return mlir::MlirOp(builder, if_op.getResult(0));
}

}  // namespace torch_tpu
