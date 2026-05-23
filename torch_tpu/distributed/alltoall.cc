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

#include "torch_tpu/distributed/alltoall.h"

#include <cstdint>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/distributed/types.h"
#include "torch_tpu/distributed/utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildDistributedAllToAllBaseShlo(
    mlir::MlirOp input, const DeviceGroupList& device_groups) {
  ABSL_VLOG(3) << "[BuildDistributedAllToAllBaseShlo] input: "
               << input.ToString();

  mlir::MlirBuilder& builder = input.getBuilder();
  // From PyTorch docs, all_to_all_single splits and concats on dimension 0
  const int64_t split_dimension = 0;
  const int64_t concat_dimension = 0;
  const int64_t split_count = device_groups[0].size();

  auto replica_groups_attr = BuildReplicaGroupsAttr(builder, device_groups);

  auto all_to_all_results = stablehlo::AllToAll(
      builder, input,
      /*split_dimension=*/split_dimension,
      /*concat_dimension=*/concat_dimension,
      /*split_count=*/split_count,
      /*replica_groups=*/replica_groups_attr,
      /*channel_id=*/BuildChannelHandleAttr(builder, /*handle=*/1, /*type=*/0));

  ABSL_CHECK_EQ(all_to_all_results.size(), 1);  // CRASH_OK
  mlir::MlirOp result = all_to_all_results[0];

  ABSL_VLOG(3) << "BuildDistributedAllToAllBaseShlo: output: "
               << result.ToString();
  return result;
}

absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> BuildDistributedAllToAllShlo(
    absl::Span<mlir::MlirOp> inputs, const DeviceGroupList& device_groups) {
  ABSL_CHECK_GT(inputs.size(), 0)  // CRASH_OK
      << "[BuildDistributedAllToAllShlo] No inputs provided";

  mlir::MlirBuilder& builder = inputs[0].getBuilder();

  const mlir::RankedTensorType input_rank_type = GetTensorTypeOrDie(inputs[0]);

  // This step concatenates the list of N input tensors
  // (each of shape S = [d0, d1, ..., dk]) along dimension 0.
  // The result concatenated_input will have a shape of [N * d0, d1, ..., dk]
  mlir::MlirOp concatenated_input =
      stablehlo::Concatenate(builder, inputs, /*dimension=*/0);

  // Perform all_to_all_single on the concatenated input.
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp all_to_all_base_result,
      BuildDistributedAllToAllBaseShlo(concatenated_input, device_groups));

  // Slice to extract the individual tensors received from each rank.
  mlir::SmallVector<mlir::MlirOp> results;
  results.reserve(inputs.size());

  const auto all_to_all_base_result_type =
      GetTensorTypeOrDie(all_to_all_base_result);

  Indices start_indices(all_to_all_base_result_type.getRank(), 0);
  Indices limit_indices = CopyIntVector(all_to_all_base_result_type.getShape());
  Indices stride_indices(all_to_all_base_result_type.getRank(), 1);

  const int64_t dim0 = input_rank_type.getShape()[0];
  for (int i = 0; i < inputs.size(); ++i) {
    const int64_t start_offset = i * dim0;
    const int64_t end_offset = (i + 1) * dim0;

    start_indices[0] = start_offset;
    limit_indices[0] = end_offset;
    mlir::MlirOp sliced_result = stablehlo::Slice(
        all_to_all_base_result, start_indices, limit_indices, stride_indices);

    results.push_back(sliced_result);
  }
  return results;
}

}  // namespace torch_tpu
