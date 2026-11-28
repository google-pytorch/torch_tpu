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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {

// TPU vector lane count. Combined with 8 sublanes, one native tile holds
// `k1DRaggedAllToAllTileElements = 1024` elements.
constexpr int64_t kTpuLaneDimension = 128;

// Element widths supported by native `(8, 128)` TPU tile factoring.
constexpr int kMinFactoredElementBits = 8;
constexpr int kMaxFactoredElementBits = 32;

int64_t AccumulateSpanGcd(int64_t gcd_val,
                          const absl::Span<const int32_t> values) {
  // Fold every split size or offset into `gcd_val` so factoring only enables
  // when every slice boundary aligns with a full 1024-element tile.
  for (const int32_t v : values) {
    gcd_val = std::gcd(gcd_val, static_cast<int64_t>(v));
  }
  return gcd_val;
}

std::vector<int32_t> ScaleSpanByFactor(const absl::Span<const int32_t> values,
                                       const int32_t factor) {
  std::vector<int32_t> scaled(values.size());
  // Convert element offsets and counts into 1024-element tile indices so
  // `ragged_all_to_all` indexes dimension 0 of the reshaped 3D operand.
  for (size_t i = 0; i < values.size(); ++i) {
    scaled[i] = values[i] / factor;
  }
  return scaled;
}

mlir::MlirOp BuildConstantI32VectorOp(mlir::MlirBuilder& builder,
                                      const absl::Span<const int32_t> values) {
  const auto i32_type = mlir::makeTensorType(
      builder.getContext(), {static_cast<int64_t>(values.size())},
      mlir::ElementType::I32);
  return stablehlo::Constant(builder,
                             mlir::makeConstant(AsArrayRef(values), i32_type));
}

struct MaybeFactored1DOperands {
  mlir::MlirOp input;
  mlir::MlirOp output;
  bool reshaped_1d = false;
  int64_t original_output_size = 0;
};

// Reshapes 1D `input` and `output` to `[N / flat_factor, sublanes, 128]`
// when `flat_factor > 1` so the XLA:TPU backend uses native `(8, 128)`
// tiling without degenerate-dimension padding.
absl::StatusOr<MaybeFactored1DOperands> MaybeReshape1DOperands(
    mlir::MlirOp input, mlir::MlirOp output, const int64_t flat_factor) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const mlir::RankedTensorType output_type = GetTensorTypeOrDie(output);
  // Skip reshaping for higher-rank tensors, which already have trailing
  // dimensions, and when `flat_factor <= 1` indicates unaligned splits or
  // unsupported element widths.
  if (flat_factor <= 1 || input_type.getRank() != 1 ||
      output_type.getRank() != 1) {
    return MaybeFactored1DOperands{.input = input, .output = output};
  }
  const int64_t input_size = input_type.getShape()[0];
  const int64_t original_output_size = output_type.getShape()[0];
  const int64_t sublanes = flat_factor / kTpuLaneDimension;
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Validated by caller before dispatch.
      input_size % flat_factor == 0 &&
          original_output_size % flat_factor == 0 &&
          flat_factor % kTpuLaneDimension == 0 && sublanes >= 1,
      error::kInternal)
      << "1D all-to-all factor " << flat_factor
      << " is not lane-aligned or does not divide input size " << input_size
      << " or output size " << original_output_size;

  return MaybeFactored1DOperands{
      .input = stablehlo::Reshape(
          input, {input_size / flat_factor, sublanes, kTpuLaneDimension}),
      .output = stablehlo::Reshape(output, {original_output_size / flat_factor,
                                            sublanes, kTpuLaneDimension}),
      .reshaped_1d = true,
      .original_output_size = original_output_size,
  };
}

absl::StatusOr<mlir::MlirOp> EmitRaggedAllToAllAndRestore1D(
    const MaybeFactored1DOperands& factored, mlir::MlirOp input_offsets_op,
    mlir::MlirOp send_sizes_op, mlir::MlirOp output_offsets_op,
    mlir::MlirOp recv_sizes_op,
    const mlir::DenseIntElementsAttr replica_groups_attr) {
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp result,
      BuildRaggedAllToAllShlo(factored.input, factored.output, input_offsets_op,
                              send_sizes_op, output_offsets_op, recv_sizes_op,
                              replica_groups_attr));
  // Restore the flat 1D shape so the collective result matches the caller's
  // original output tensor contract.
  if (factored.reshaped_1d) {
    return stablehlo::Reshape(result, {factored.original_output_size});
  }
  return result;
}

}  // namespace

int64_t Compute1DRaggedAllToAllLocalGcd(
    const int64_t input_size, const int64_t output_size,
    const absl::Span<const int32_t> input_offsets,
    const absl::Span<const int32_t> send_sizes,
    const absl::Span<const int32_t> local_output_offsets,
    const absl::Span<const int32_t> recv_sizes) {
  int64_t gcd_val = std::gcd(std::max<int64_t>(input_size, 0),
                             std::max<int64_t>(output_size, 0));
  gcd_val = AccumulateSpanGcd(gcd_val, input_offsets);
  gcd_val = AccumulateSpanGcd(gcd_val, send_sizes);
  gcd_val = AccumulateSpanGcd(gcd_val, local_output_offsets);
  gcd_val = AccumulateSpanGcd(gcd_val, recv_sizes);
  return gcd_val;
}

int64_t Select1DRaggedAllToAllFactor(const int64_t group_gcd,
                                     const int element_bitwidth) {
  // Factor only when every rank's sizes and offsets divide evenly by 1024 and
  // the element width (8..32 bits) maps natively onto `(8, 128)` TPU tiles.
  // 64-bit, boolean, complex, and packed types use distinct layouts and stay
  // unfactored.
  if (group_gcd > 0 && group_gcd % k1DRaggedAllToAllTileElements == 0 &&
      element_bitwidth >= kMinFactoredElementBits &&
      element_bitwidth <= kMaxFactoredElementBits) {
    return k1DRaggedAllToAllTileElements;
  }
  return 1;
}

absl::StatusOr<mlir::MlirOp> BuildDistributedAllToAllBaseShlo(
    mlir::MlirOp input, const DeviceGroupList& device_groups) {
  ABSL_VLOG(3) << "[BuildDistributedAllToAllBaseShlo] input: "
               << input.ToString();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=ProcessGroupTpu constructs non-empty
                 // replica_groups.
      !device_groups.empty(), error::kInternal)
      << "expected non-empty device_groups";

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
  ABSL_VLOG(3) << "BuildDistributedAllToAllBaseShlo: output: "
               << all_to_all_results[0].ToString();
  return all_to_all_results[0];
}

absl::StatusOr<mlir::MlirOp> BuildDistributedAllToAllBaseUnevenSplitsShlo(
    mlir::MlirOp input, mlir::MlirOp output,
    absl::Span<const int32_t> input_offsets,
    absl::Span<const int32_t> send_sizes,
    absl::Span<const int32_t> output_offsets,
    absl::Span<const int32_t> recv_sizes, const int64_t flat_factor,
    const DeviceGroupList& device_groups) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=ProcessGroupTpu constructs non-empty
                 // replica_groups.
      !device_groups.empty(), error::kInternal)
      << "expected non-empty device_groups";
  auto& builder = input.getBuilder();
  auto replica_groups_attr = BuildReplicaGroupsAttr(builder, device_groups);

  TT_ASSIGN_OR_RETURN(const MaybeFactored1DOperands factored,
                      MaybeReshape1DOperands(input, output, flat_factor));

  std::vector<int32_t> scaled_input_offsets;
  std::vector<int32_t> scaled_send_sizes;
  std::vector<int32_t> scaled_output_offsets;
  std::vector<int32_t> scaled_recv_sizes;
  // Scale all four metadata vectors when 1D factoring reshaped dimension 0
  // from element counts into 1024-element tile counts.
  if (factored.reshaped_1d) {
    const int32_t factor_i32 = static_cast<int32_t>(flat_factor);
    scaled_input_offsets = ScaleSpanByFactor(input_offsets, factor_i32);
    scaled_send_sizes = ScaleSpanByFactor(send_sizes, factor_i32);
    scaled_output_offsets = ScaleSpanByFactor(output_offsets, factor_i32);
    scaled_recv_sizes = ScaleSpanByFactor(recv_sizes, factor_i32);
    input_offsets = scaled_input_offsets;
    send_sizes = scaled_send_sizes;
    output_offsets = scaled_output_offsets;
    recv_sizes = scaled_recv_sizes;
  }

  mlir::MlirOp input_offsets_op =
      BuildConstantI32VectorOp(builder, input_offsets);
  mlir::MlirOp send_sizes_op = BuildConstantI32VectorOp(builder, send_sizes);
  mlir::MlirOp output_offsets_op =
      BuildConstantI32VectorOp(builder, output_offsets);
  mlir::MlirOp recv_sizes_op = BuildConstantI32VectorOp(builder, recv_sizes);

  return EmitRaggedAllToAllAndRestore1D(factored, input_offsets_op,
                                        send_sizes_op, output_offsets_op,
                                        recv_sizes_op, replica_groups_attr);
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

absl::StatusOr<mlir::MlirOp> ConcatenateTensorsAlongDim0(
    absl::Span<mlir::MlirOp> inputs, mlir::ElementType output_dtype,
    absl::Span<const int64_t> total_input_shape, mlir::MlirBuilder& builder) {
  if (inputs.empty()) {
    return BuildFillUninitialized(builder, output_dtype, total_input_shape);
  }
  return stablehlo::Concatenate(builder, inputs, /*dimension=*/0);
}

mlir::SmallVector<mlir::MlirOp> SliceTensorsAlongDim0(
    mlir::MlirOp tensor, absl::Span<const int32_t> split_sizes) {
  const auto tensor_type = GetTensorTypeOrDie(tensor);
  Indices start_indices(tensor_type.getRank(), 0);
  Indices limit_indices = CopyIntVector(tensor_type.getShape());
  Indices stride_indices(tensor_type.getRank(), 1);

  mlir::SmallVector<mlir::MlirOp> results;
  results.reserve(split_sizes.size());

  int64_t current_offset = 0;
  for (const int32_t count : split_sizes) {
    start_indices[0] = current_offset;
    limit_indices[0] = current_offset + count;
    results.push_back(
        stablehlo::Slice(tensor, start_indices, limit_indices, stride_indices));
    current_offset += count;
  }
  return results;
}

}  // namespace torch_tpu
