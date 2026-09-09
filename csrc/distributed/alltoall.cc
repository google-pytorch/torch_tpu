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

#include "csrc/distributed/alltoall.h"

#include <cstddef>
#include <cstdint>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/utils.h"
#include "csrc/distributed/types.h"
#include "csrc/distributed/utils.h"
#include "csrc/ops/experimental/ragged_all_to_all/ragged_all_to_all_builder.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

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

absl::StatusOr<mlir::MlirOp> BuildDistributedAllToAllBaseUnevenSplitsShlo(
    mlir::MlirOp input, mlir::MlirOp output,
    absl::Span<const int32_t> input_offsets,
    absl::Span<const int32_t> send_sizes,
    absl::Span<const int32_t> output_offsets,
    absl::Span<const int32_t> recv_sizes,
    const DeviceGroupList& device_groups) {
  auto& builder = input.getBuilder();
  auto replica_groups_attr = BuildReplicaGroupsAttr(builder, device_groups);

  auto i32_type =
      mlir::RankedTensorType::get({static_cast<int64_t>(input_offsets.size())},
                                  builder.getOpBuilder().getI32Type());

  auto input_offsets_op = mlir::stablehlo::Constant(
      builder, mlir::makeConstant(llvm::ArrayRef<int32_t>(input_offsets.data(),
                                                          input_offsets.size()),
                                  i32_type));
  auto send_sizes_op = mlir::stablehlo::Constant(
      builder, mlir::makeConstant(llvm::ArrayRef<int32_t>(send_sizes.data(),
                                                          send_sizes.size()),
                                  i32_type));
  auto output_offsets_op = mlir::stablehlo::Constant(
      builder,
      mlir::makeConstant(
          llvm::ArrayRef<int32_t>(output_offsets.data(), output_offsets.size()),
          i32_type));
  auto recv_sizes_op = mlir::stablehlo::Constant(
      builder, mlir::makeConstant(llvm::ArrayRef<int32_t>(recv_sizes.data(),
                                                          recv_sizes.size()),
                                  i32_type));

  return BuildRaggedAllToAllShlo(input, output, input_offsets_op, send_sizes_op,
                                 output_offsets_op, recv_sizes_op,
                                 replica_groups_attr);
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

namespace {

// Slices a tensor along dimension 0 into individual tensors matching
// split_sizes.
mlir::SmallVector<mlir::MlirOp> SliceTensorsAlongDim0(
    mlir::MlirOp tensor, absl::Span<const int32_t> split_sizes) {
  const auto tensor_type = GetTensorTypeOrDie(tensor);
  Indices start_indices(tensor_type.getRank(), 0);
  Indices limit_indices = CopyIntVector(tensor_type.getShape());
  Indices stride_indices(tensor_type.getRank(), 1);

  mlir::SmallVector<mlir::MlirOp> results;
  results.reserve(split_sizes.size());

  int64_t current_offset = 0;
  for (int32_t count : split_sizes) {
    start_indices[0] = current_offset;
    limit_indices[0] = current_offset + count;
    results.push_back(
        stablehlo::Slice(tensor, start_indices, limit_indices, stride_indices));
    current_offset += count;
  }
  return results;
}

}  // namespace

absl::StatusOr<mlir::SmallVector<mlir::MlirOp>>
BuildDistributedAllToAllUnevenSplitsShlo(
    absl::Span<mlir::MlirOp> inputs, mlir::ElementType output_dtype,
    absl::Span<const int32_t> input_offsets,
    absl::Span<const int32_t> send_sizes,
    absl::Span<const int32_t> output_offsets,
    absl::Span<const int32_t> recv_sizes,
    const DeviceGroupList& device_groups) {
  ABSL_CHECK_GT(inputs.size(), 0)  // CRASH_OK
      << "[BuildDistributedAllToAllUnevenSplitsShlo] No inputs provided";

  mlir::MlirBuilder& builder = inputs[0].getBuilder();

  // 1. Concatenate the input ops along dim 0.
  mlir::MlirOp concatenated_input =
      stablehlo::Concatenate(builder, inputs, /*dimension=*/0);

  // 2. Prepare output shape for the ragged all-to-all custom call.
  // Trailing dimensions match inputs[0].
  const mlir::RankedTensorType input_rank_type = GetTensorTypeOrDie(inputs[0]);
  Dimensions total_output_shape = CopyIntVector(input_rank_type.getShape());
  int64_t total_recv_elements = 0;
  for (int32_t sz : recv_sizes) {
    total_recv_elements += sz;
  }
  total_output_shape[0] = total_recv_elements;

  // 3. Create uninitialized buffer for output of ragged all-to-all.
  mlir::MlirOp output_placeholder =
      BuildFillUninitialized(builder, output_dtype, total_output_shape);

  // 4. Build ragged all-to-all MLIR operation.
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp ragged_all_to_all_result,
      BuildDistributedAllToAllBaseUnevenSplitsShlo(
          concatenated_input, output_placeholder, input_offsets, send_sizes,
          output_offsets, recv_sizes, device_groups));

  // 5. Slice ragged_all_to_all_result to extract individual output tensors.
  return SliceTensorsAlongDim0(ragged_all_to_all_result, recv_sizes);
}

}  // namespace torch_tpu
