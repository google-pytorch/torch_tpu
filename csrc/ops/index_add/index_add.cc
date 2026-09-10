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

#include "csrc/ops/index_add/index_add.h"

#include <cstdint>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/Builders.h"
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

absl::StatusOr<mlir::MlirOp> BuildIndexAddShlo(
    mlir::MlirOp self, int64_t dim, mlir::MlirOp index, mlir::MlirOp source,
    mlir::MlirOp alpha, mlir::ElementType computation_xla_type) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  const mlir::RankedTensorType source_type = GetTensorTypeOrDie(source);
  const mlir::RankedTensorType index_type = GetTensorTypeOrDie(index);
  const mlir::RankedTensorType alpha_type = GetTensorTypeOrDie(alpha);
  ABSL_VLOG(2) << "BuildIndexAddShlo:"
               << ", dim: " << dim
               << ", self_type: " << mlir::debugString(self_type)
               << ", source_type: " << mlir::debugString(source_type)
               << ", index_type: " << mlir::debugString(index_type)
               << ", alpha_type: " << mlir::debugString(alpha_type);
  if (self_type.getRank() == 0) {
    mlir::MlirOp alpha_mul_source = stablehlo::Mul(alpha, source);
    return stablehlo::Add(self, alpha_mul_source);
  }

  // Convert arguments to the computation type if necessary.
  mlir::Type computation_type =
      mlir::getElementType(self.getContext(), computation_xla_type);
  ABSL_VLOG(2) << "computation_type: " << mlir::debugString(computation_type);
  if (self_type.getElementType() != computation_type) {
    self = mlir::stablehlo::ConvertElementType(self, computation_type);
  }
  if (source_type.getElementType() != computation_type) {
    source = mlir::stablehlo::ConvertElementType(source, computation_type);
  }
  if (alpha_type.getElementType() != computation_type) {
    alpha = mlir::stablehlo::ConvertElementType(alpha, computation_type);
  }

  Dimensions all_other_dims;
  all_other_dims.reserve(self_type.getRank() - 1);
  for (int i = 0; i < self_type.getRank(); ++i) {
    if (i != dim) {
      all_other_dims.push_back(i);
    }
  }

  index = mlir::stablehlo::Reshape(index, {index_type.getDimSize(0), 1});

  TT_ASSIGN_OR_RETURN((auto [broadcasted_source, broadcasted_alpha]),
                      ApplyBroadcastIfNeeded(source, alpha));
  mlir::MlirOp source_scaled =
      stablehlo::Mul(broadcasted_source, broadcasted_alpha);

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
  auto region_builder = [computation_type](mlir::RegionBuilder& builder) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        computation_type, builder.getRegion(), builder.getOpBuilder());
  };

  // Call stablehlo::Scatter
  return stablehlo::Scatter(self, index, source_scaled, region_builder,
                            scatter_dimension_numbers)[0];
}

}  // namespace torch_tpu
