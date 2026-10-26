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

#include "csrc/distributed/allgather.h"

#include <cstdint>

#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/utils.h"
#include "csrc/distributed/types.h"
#include "csrc/distributed/utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> BuildDistributedAllGatherShlo(
    mlir::MlirOp input, const DeviceGroupList& device_groups,
    AllGatherOutputMode output_mode, int64_t gather_dim) {
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

  ABSL_CHECK(gather_dim >= 0 &&  // CRASH_OK
             gather_dim < static_cast<int64_t>(input_shape.size()))
      << "[BuildDistributedAllGatherShlo]: invalid gather_dim " << gather_dim;
  Dimensions output_shape = CopyIntVector(input_shape);
  output_shape[gather_dim] *= group_size;
  auto output_type =
      mlir::RankedTensorType::get(output_shape, input_element_type);

  auto& builder = input.getBuilder();
  auto gather_outputs = mlir::stablehlo::AllGather(
      builder,
      /*resultType=*/output_type,
      /*operands=*/input,
      /*all_gather_dim=*/gather_dim,
      /*replica_groups=*/BuildReplicaGroupsAttr(builder, device_groups),
      /*channel_id=*/BuildChannelHandleAttr(builder, /*handle=*/1, /*type=*/0),
      /*use_global_device_ids=*/true);
  ABSL_CHECK_EQ(gather_outputs.size(), 1);  // CRASH_OK
  auto gather_output = gather_outputs[0];

  switch (output_mode) {
    case AllGatherOutputMode::kConcat:
      return mlir::SmallVector<mlir::MlirOp>{gather_output};
    case AllGatherOutputMode::kStack: {
      ABSL_CHECK_EQ(gather_dim, 0)  // CRASH_OK
          << "[BuildDistributedAllGatherShlo]: only kConcat supports a nonzero "
             "gather_dim";
      Dimensions stacked_shape;
      stacked_shape.reserve(input_shape.size() + 1);
      stacked_shape.push_back(group_size);
      stacked_shape.insert(stacked_shape.end(), input_shape.begin(),
                           input_shape.end());
      return mlir::SmallVector<mlir::MlirOp>{
          mlir::stablehlo::Reshape(gather_output, stacked_shape)};
    }
    case AllGatherOutputMode::kList: {
      ABSL_CHECK_EQ(gather_dim, 0)  // CRASH_OK
          << "[BuildDistributedAllGatherShlo]: only kConcat supports a nonzero "
             "gather_dim";
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
