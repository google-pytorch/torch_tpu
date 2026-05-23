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

#include "torch_tpu/distributed/reduce_scatter.h"

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/distributed/types.h"
#include "torch_tpu/distributed/utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

absl::StatusOr<mlir::MlirOp> BuildDistributedReduceScatterShlo(
    mlir::MlirOp input, c10d::ReduceOp::RedOpType reduce_op_type,
    const DeviceGroupList& device_groups) {
  ABSL_CHECK(!device_groups.empty())  // CRASH_OK
      << "[BuildDistributedReduceScatterShlo]: device_groups is empty";
  const int group_size = device_groups[0].size();

  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  auto element_type = input_type.getElementType();

  // Calculate the output shape by splitting on dim=0 (our scatter dimension).
  Dimensions output_shape = CopyIntVector(input_type.getShape());
  ABSL_CHECK_EQ(output_shape[0] % group_size, 0);  // CRASH_OK
  output_shape[0] /= group_size;

  // For AVG, we perform a SUM reduction and then divide by the group size.
  c10d::ReduceOp::RedOpType effective_reduce_op_type =
      reduce_op_type == c10d::ReduceOp::AVG ? c10d::ReduceOp::SUM
                                            : reduce_op_type;

  auto output_type =
      mlir::RankedTensorType::get(output_shape, input_type.getElementType());

  auto reduction_computation = [element_type, effective_reduce_op_type](
                                   mlir::RegionBuilder& rb) {
    auto status = BuildReduceBody(rb, element_type, effective_reduce_op_type);
    if (!status.ok()) {
      ABSL_LOG(FATAL)  // CRASH_OK
          << "[BuildDistributedReduceScatterShlo]: BuildReduceBody failed: "
          << status.message();
    }
  };

  auto& builder = input.getBuilder();
  auto output = mlir::stablehlo::ReduceScatter(
      output_type, input, reduction_computation,
      /*scatter_dimension=*/0,
      /*replica_groups=*/BuildReplicaGroupsAttr(builder, device_groups),
      /*channel_id=*/BuildChannelHandleAttr(builder, /*handle=*/1, /*type=*/0),
      /*use_global_device_ids=*/true);

  if (reduce_op_type == c10d::ReduceOp::AVG) {
    const mlir::RankedTensorType output_type = GetTensorTypeOrDie(output);
    mlir::MlirOp divisor = MakeConstant(builder, group_size, output_type);
    output = mlir::stablehlo::Div(output, divisor);
  }

  return output;
}

}  // namespace torch_tpu
