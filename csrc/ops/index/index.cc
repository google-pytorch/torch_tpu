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
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"

namespace torch_tpu {

namespace {

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

}  // namespace

namespace stablehlo = mlir::stablehlo;

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

}  // namespace torch_tpu
