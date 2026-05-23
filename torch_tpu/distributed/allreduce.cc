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

#include "torch_tpu/distributed/allreduce.h"

#include <cstdint>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/distributed/types.h"
#include "torch_tpu/distributed/utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildDistributedAllReduceShlo(
    mlir::MlirOp input, c10d::ReduceOp::RedOpType reduce_op_type,
    const DeviceGroupList& device_groups) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  auto input_element_type = input_type.getElementType();
  ABSL_VLOG(3) << "[BuildDistributedAllReduceShlo] input: " << input.ToString()
               << " reduce_op_type: " << ToString(reduce_op_type)
               << " device_groups: " << device_groups;

  // For AVG, we perform a SUM reduction and then divide by the group size.
  c10d::ReduceOp::RedOpType effective_reduce_op_type =
      reduce_op_type == c10d::ReduceOp::AVG ? c10d::ReduceOp::SUM
                                            : reduce_op_type;

  auto& builder = input.getBuilder();
  auto all_reduce_results = stablehlo::AllReduce(
      builder, input,
      /*computation=*/
      [input_element_type, effective_reduce_op_type](mlir::RegionBuilder& rb) {
        auto status =
            BuildReduceBody(rb, input_element_type, effective_reduce_op_type);
        if (!status.ok()) {
          ABSL_LOG(FATAL)  // CRASH_OK
              << "[BuildDistributedAllReduceShlo]: BuildReduceBody failed: "
              << status.message();
        }
      },
      /*replica_groups=*/BuildReplicaGroupsAttr(builder, device_groups),
      /*channel_id=*/BuildChannelHandleAttr(builder, /*handle=*/1, /*type=*/0),
      /*use_global_device_ids=*/true);

  ABSL_CHECK_EQ(all_reduce_results.size(), 1);  // CRASH_OK
  mlir::MlirOp result = all_reduce_results[0];

  // If the original op was AVG, divide the SUM result by the number of devices.
  if (reduce_op_type == c10d::ReduceOp::AVG) {
    ABSL_CHECK(!device_groups.empty())  // CRASH_OK
        << "[BuildDistributedAllReduceShlo]: device_groups is empty";
    const int64_t group_size = device_groups[0].size();
    mlir::MlirOp divisor = MakeConstant(builder, group_size, input_type);
    result = stablehlo::Div(result, divisor);
  }

  ABSL_VLOG(3) << "BuildDistributedAllReduceShlo: output: "
               << result.ToString();
  return result;
}

}  // namespace torch_tpu
