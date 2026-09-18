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

#include "csrc/ops/index_select/index_select.h"

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

namespace stablehlo = mlir::stablehlo;

namespace {

struct DynamicUnindexedDim {
  mlir::stablehlo::DimensionInfo dim_info;
  int64_t out_dim;
};

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildIndexSelectShlo(mlir::MlirOp self,
                                                  int64_t dim,
                                                  mlir::MlirOp index) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  const mlir::RankedTensorType index_type = GetTensorTypeOrDie(index);

  TT_ASSIGN_OR_RETURN(dim, SafeWrapDim(dim, self_type.getRank()));

  if (self_type.getRank() == 0) {
    return self;
  }

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch catches index rank != 1 first.
      index_type.getRank() == 1, error::kInvalidArgument)
      << "expected index to be 1D, got rank " << index_type.getRank();

  mlir::stablehlo::Dimensions self_dims = GetDimensions(self);
  Dimensions gather_slice_sizes;
  gather_slice_sizes.reserve(self_dims.size());
  Indices offset_dims;
  std::vector<DynamicUnindexedDim> dynamic_unindexed_dims;

  for (int64_t i = 0; i < self_type.getRank(); ++i) {
    if (i == dim) {
      gather_slice_sizes.push_back(std::min<int64_t>(self_dims[i].size, 1));
    } else {
      gather_slice_sizes.push_back(self_dims[i].size);
      offset_dims.push_back(i);
      if (self_dims[i].boundOp.has_value()) {
        dynamic_unindexed_dims.push_back({self_dims[i], i});
      }
    }
  }

  TT_ASSIGN_OR_RETURN(index, Unsqueeze(index, 1));

  ABSL_VLOG(2) << "BuildIndexSelectShlo: gather_slice_sizes = "
               << absl::StrJoin(gather_slice_sizes, ",");

  stablehlo::GatherDimensionNumbersAttr gather_dimension_numbers =
      stablehlo::GatherDimensionNumbersAttr::get(
          &self.getContext(),
          /*offset_dims=*/offset_dims,
          /*collapsed_slice_dims=*/{dim},
          /*operand_batching_dims=*/{},
          /*start_indices_batching_dims=*/{},
          /*start_index_map=*/{dim},
          /*index_vector_dim=*/1);
  ABSL_VLOG(2) << "BuildIndexSelectShlo: GatherDimensionNumbers = "
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
