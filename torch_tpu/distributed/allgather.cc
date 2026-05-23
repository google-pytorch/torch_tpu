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

#include "torch_tpu/distributed/allgather.h"

#include <cstdint>

#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/distributed/types.h"
#include "torch_tpu/distributed/utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> BuildDistributedAllGatherShlo(
    mlir::MlirOp input, const DeviceGroupList& device_groups,
    AllGatherOutputMode output_mode) {
  ABSL_CHECK(!device_groups.empty())  // CRASH_OK
      << "[BuildDistributedAllGatherShlo]: device_groups is empty";
  const int group_size = device_groups[0].size();

  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  // Inputs on all processes are expected to have the same shape and dtype.
  const auto input_element_type = input_type.getElementType();
  auto input_shape = input_type.getShape();

  // XLA requires tensors for all_gather, so if inputs are scalars wrap them
  // as single-element tensors. The dispatch code will insert a reshape
  // from single-element tensors back to scalars if needed.
  Dimensions scalar_dim_storage;
  if (input_shape.empty()) {
    scalar_dim_storage = {1};
    input_shape = llvm::ArrayRef<int64_t>(scalar_dim_storage);
    input = mlir::stablehlo::Reshape(input, input_shape);
  }

  // Though XLA can perform an allgather along any dimension, using the 0th
  // dimension is sufficient for all PyTorch use cases.
  Dimensions output_shape = CopyIntVector(input_shape);
  output_shape[0] *= group_size;
  auto output_type =
      mlir::RankedTensorType::get(output_shape, input_element_type);

  auto& builder = input.getBuilder();
  auto gather_outputs = mlir::stablehlo::AllGather(
      builder,
      /*resultType=*/output_type,
      /*operands=*/input,
      /*all_gather_dim=*/0,
      /*replica_groups=*/BuildReplicaGroupsAttr(builder, device_groups),
      /*channel_id=*/BuildChannelHandleAttr(builder, /*handle=*/1, /*type=*/0),
      /*use_global_device_ids=*/true);
  ABSL_CHECK_EQ(gather_outputs.size(), 1);  // CRASH_OK
  auto gather_output = gather_outputs[0];

  switch (output_mode) {
    case AllGatherOutputMode::kConcat:
      return mlir::SmallVector<mlir::MlirOp>{gather_output};
    case AllGatherOutputMode::kStack: {
      Dimensions stacked_shape;
      stacked_shape.reserve(input_shape.size() + 1);
      stacked_shape.push_back(group_size);
      for (int64_t dim : input_shape) {
        stacked_shape.push_back(dim);
      }
      return mlir::SmallVector<mlir::MlirOp>{
          mlir::stablehlo::Reshape(gather_output, stacked_shape)};
    }
    case AllGatherOutputMode::kList: {
      mlir::SmallVector<mlir::MlirOp> results;
      results.reserve(group_size);

      int64_t gather_rank = output_type.getRank();
      Indices start_indices(gather_rank, 0);
      Indices limit_indices = CopyIntVector(output_type.getShape());
      Indices stride_indices(gather_rank, 1);

      int64_t dim0_size = input_shape[0];
      for (int i = 0; i < group_size; ++i) {
        start_indices[0] = i * dim0_size;
        limit_indices[0] = (i + 1) * dim0_size;
        results.push_back(mlir::stablehlo::Slice(
            gather_output, start_indices, limit_indices, stride_indices));
      }
      return results;
    }
  }
}

}  // namespace torch_tpu
