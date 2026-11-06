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

#include "csrc/ops/index_copy/index_copy.h"

#include <cstdint>

#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "csrc/common/dimension_types.h"
#include "csrc/ops/op_builder_utils.h"
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

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

mlir::MlirOp BuildIndexCopyShlo(mlir::MlirOp self, int64_t dim,
                                mlir::MlirOp index, mlir::MlirOp source,
                                mlir::ElementType computation_dtype) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  const mlir::RankedTensorType source_type = GetTensorTypeOrDie(source);
  const mlir::RankedTensorType index_type = GetTensorTypeOrDie(index);
  ABSL_VLOG(2) << "BuildIndexCopyShlo:"
               << ", dim: " << dim
               << ", self_type: " << mlir::debugString(self_type)
               << ", source_type: " << mlir::debugString(source_type)
               << ", index_type: " << mlir::debugString(index_type);
  if (self_type.getRank() == 0) {
    return source;
  }

  // Convert arguments to the computation type if necessary.
  mlir::Type computation_type =
      mlir::getElementType(self.getContext(), computation_dtype);
  ABSL_VLOG(2) << "computation_type: " << mlir::debugString(computation_type);
  if (self_type.getElementType() != computation_type) {
    self = mlir::stablehlo::ConvertElementType(self, computation_type);
  }
  if (source_type.getElementType() != computation_type) {
    source = mlir::stablehlo::ConvertElementType(source, computation_type);
  }

  if (index_type.getRank() == 1 && index_type.getDimSize(0) == 1 &&
      self_type.hasStaticShape() && !self_type.isDynamicDim(dim)) {
    const int64_t self_rank = self_type.getRank();
    const int64_t dim_size = self_type.getDimSize(dim);
    mlir::MlirBuilder& builder = self.getBuilder();
    mlir::MlirOp idx_scalar = stablehlo::Reshape(index, {});
    idx_scalar =
        stablehlo::ConvertElementType(idx_scalar, mlir::ElementType::I64);

    mlir::MlirOp dim_size_op =
        MakeScalarConstant(builder, dim_size, mlir::ElementType::I64);
    mlir::MlirOp zero = MakeScalarConstant(builder, 0, mlir::ElementType::I64);
    mlir::MlirOp dim_minus_1 =
        MakeScalarConstant(builder, dim_size - 1, mlir::ElementType::I64);

    mlir::MlirOp is_neg = stablehlo::Compare(
        idx_scalar, zero, stablehlo::ComparisonDirection::LT);
    mlir::MlirOp add_op = stablehlo::Add(idx_scalar, dim_size_op);
    mlir::MlirOp wrapped_idx = stablehlo::Select(is_neg, add_op, idx_scalar);
    mlir::MlirOp max_idx = stablehlo::Max(wrapped_idx, zero);
    mlir::MlirOp clamped_idx = stablehlo::Min(max_idx, dim_minus_1);

    std::vector<mlir::MlirOp> start_indices;
    start_indices.reserve(self_rank);
    for (int64_t i = 0; i < self_rank; ++i) {
      if (i == dim) {
        start_indices.push_back(clamped_idx);
      } else {
        start_indices.push_back(
            MakeScalarConstant(builder, 0, mlir::ElementType::I64));
      }
    }
    return mlir::stablehlo::DynamicUpdateSlice(self, source, start_indices);
  }

  const Dimensions all_other_dims =
      GetBatchDimensions(self_type.getRank(), dim);

  index = mlir::stablehlo::Reshape(index, {index_type.getDimSize(0), 1});
  stablehlo::ScatterDimensionNumbersAttr scatter_dimension_numbers =
      stablehlo::ScatterDimensionNumbersAttr::get(
          &self.getContext(),
          /*update_window_dims=*/all_other_dims,
          /*inserted_window_dims=*/{dim},
          /*input_batching_dims=*/{},
          /*scatter_indices_batching_dims=*/{},
          /*scatter_dims_to_operand_dims=*/{dim},
          /*index_vector_dim=*/1);
  ABSL_VLOG(2) << "BuildScatterShlo: ScatterDimensionNumbers = "
               << mlir::debugString(scatter_dimension_numbers);

  // Create a region builder callback.
  auto block_type = self_type.clone({}, computation_type);
  auto region_builder = [block_type](mlir::RegionBuilder& builder) {
    mlir::MlirOp _ = mlir::Argument(builder, block_type);
    mlir::MlirOp arg1 = mlir::Argument(builder, block_type);
    // Region returns the second argument (from source).
    stablehlo::Return(builder, {arg1});
  };

  return stablehlo::Scatter(self, index, source, region_builder,
                            scatter_dimension_numbers)[0];
}

}  // namespace torch_tpu
