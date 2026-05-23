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

#include "torch_tpu/ops/index_select/index_select.h"

#include <cstdint>

#include "absl/log/absl_log.h"
#include "absl/strings/str_join.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

mlir::MlirOp BuildIndexSelectShlo(mlir::MlirOp self, int64_t dim,
                                  mlir::MlirOp index) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  const mlir::RankedTensorType index_type = GetTensorTypeOrDie(index);

  if (self_type.getRank() == 0) {
    return self;
  }

  Dimensions gather_slice_sizes = CopyIntVector(self_type.getShape());
  gather_slice_sizes[dim] = 1;
  Indices offset_dims;
  for (int i = 0; i < self_type.getRank(); ++i) {
    if (i != dim) {
      offset_dims.push_back(i);
    }
  }

  index = mlir::stablehlo::Reshape(index, {index_type.getDimSize(0), 1});

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
  return stablehlo::ConvertElementType(result, self_type.getElementType());
}

}  // namespace torch_tpu
