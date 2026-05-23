// Copyright 2026 Google LLC
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

#include "torch_tpu/ops/masked_scatter/masked_scatter.h"

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/cumsum/cumsum.h"
#include "torch_tpu/ops/index_select/index_select.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

absl::StatusOr<mlir::MlirOp> BuildMaskedScatterShlo(mlir::MlirOp input,
                                                    mlir::MlirOp mask,
                                                    mlir::MlirOp source) {
  auto& builder = input.getBuilder();
  const auto input_type = GetTensorTypeOrDie(input);
  const auto mask_type = GetTensorTypeOrDie(mask);
  const auto source_type = GetTensorTypeOrDie(source);

  int64_t numel = input_type.getNumElements();
  mlir::Type i64_type = builder.getOpBuilder().getI64Type();

  // 1. Flatten all input tensors to 1D
  mlir::MlirOp flat_input = mlir::stablehlo::Reshape(
      mlir::makeTensorType(builder.getContext(), {numel},
                           input_type.getElementType()),
      input);
  mlir::MlirOp flat_mask = mlir::stablehlo::Reshape(
      mlir::makeTensorType(builder.getContext(), {numel},
                           mask_type.getElementType()),
      mask);
  mlir::MlirOp flat_source = mlir::stablehlo::Reshape(
      mlir::makeTensorType(builder.getContext(), {source_type.getNumElements()},
                           source_type.getElementType()),
      source);

  // 2. Convert flat_mask from bool to i64
  mlir::MlirOp mask_i64 =
      mlir::stablehlo::ConvertElementType(flat_mask, i64_type);

  // 3. Compute prefix sum (cumsum) of mask values on device
  TT_ASSIGN_OR_RETURN(mlir::MlirOp prefix_sum,
                      BuildCumsumShlo(0, std::nullopt, mask_i64));

  // 4. Compute indices for gather: gather_indices = prefix_sum - 1
  mlir::MlirOp constant_1 =
      MakeScalarConstant(builder, static_cast<int64_t>(1), i64_type);
  // Broadcast constant_1 to match prefix_sum rank/size: broadcast to shape
  // {numel}
  mlir::MlirOp broadcasted_1 =
      mlir::stablehlo::Broadcast(constant_1, llvm::ArrayRef<int64_t>{numel});
  mlir::MlirOp gather_indices =
      mlir::stablehlo::Subtract(prefix_sum, broadcasted_1);

  // 5. Perform dynamic gather from source using BuildIndexSelectShlo helper
  // along dim 0 (returns raw MlirOp!)
  mlir::MlirOp gathered_source =
      BuildIndexSelectShlo(flat_source, 0, gather_indices);

  // 6. Elementwise selection: if mask is True, take gathered_source, else keep
  // flat_input
  mlir::MlirOp flat_output =
      mlir::stablehlo::Select(flat_mask, gathered_source, flat_input);

  // 7. Reshape flat_output back to the original ranked shape of input
  return mlir::stablehlo::Reshape(input_type, flat_output);
}

}  // namespace torch_tpu
