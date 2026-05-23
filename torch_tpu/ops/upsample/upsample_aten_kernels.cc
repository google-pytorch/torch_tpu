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

#include "torch_tpu/ops/upsample/upsample_aten_kernels.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/binary.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

// Upsample nearest interpolation for 1D, 2D and 3D tensors.
// This implementation is based on creating a gather tensor from scaled iota
// tensors that are broadcast in each dimension. These iota tensors are
// concatenated into a single gather tensor with an index in the last dimension.
// The gather operation maps the input to the output.

namespace torch_tpu {

namespace {

enum class UpsampleMode {
  // Nearest mode uses the floor of the scaled coordinate.
  kNearest,
  // Nearest exact mode aligns the centers of the pixels by adding 0.5 before
  // scaling and applying the floor function.
  kNearestExact,
};

using ScaleVector = absl::InlinedVector<std::optional<double>, 6>;
using MlirOpVector = absl::InlinedVector<mlir::MlirOp, 6>;
using TensorVector = absl::InlinedVector<at::Tensor, 6>;

// Configuration for the gather operation used in upsampling.
// Contains the dimension numbers for the gather op and the slice sizes
// for each dimension of the input tensor.
struct UpsampleGatherConfig {
  mlir::stablehlo::GatherDimensionNumbersAttr gather_dimension_numbers;
  SmallInt64Vector slice_sizes;
};

struct UpsampleScatterConfig {
  mlir::stablehlo::ScatterDimensionNumbersAttr scatter_dimension_numbers;
};

UpsampleScatterConfig GetUpsampleScatterConfig(mlir::MlirOp grad_output,
                                               int64_t spatial_dim_size) {
  int64_t num_output_dimensions =
      GetTensorTypeOrDie(grad_output).getShape().size();
  Dimensions scatter_dims_to_operand_dims;
  scatter_dims_to_operand_dims.reserve(spatial_dim_size);
  for (int i = 0; i < spatial_dim_size; ++i) {
    scatter_dims_to_operand_dims.push_back(i + num_output_dimensions -
                                           spatial_dim_size);
  }
  Dimensions batching_dims;
  batching_dims.reserve(num_output_dimensions - spatial_dim_size);
  for (int i = 0; i < num_output_dimensions - spatial_dim_size; ++i) {
    batching_dims.push_back(i);
  }
  Dimensions inserted_window_dims;
  inserted_window_dims.reserve(spatial_dim_size);
  for (int i = 0; i < spatial_dim_size; ++i) {
    inserted_window_dims.push_back(i + num_output_dimensions -
                                   spatial_dim_size);
  }

  mlir::stablehlo::ScatterDimensionNumbersAttr scatter_dimension_numbers =
      mlir::stablehlo::ScatterDimensionNumbersAttr::get(
          &grad_output.getContext(),
          /*update_window_dims=*/{},
          /*inserted_window_dims=*/
          AsArrayRef<int64_t>(inserted_window_dims),
          /*input_batching_dims=*/AsArrayRef<int64_t>(batching_dims),
          /*scatter_indices_batching_dims=*/AsArrayRef<int64_t>(batching_dims),
          /*scatter_dims_to_operand_dims=*/
          AsArrayRef<int64_t>(scatter_dims_to_operand_dims),
          /*index_vector_dim=*/num_output_dimensions);
  return {scatter_dimension_numbers};
}

UpsampleGatherConfig GetUpsampleGatherConfig(mlir::MlirOp input,
                                             int64_t spatial_dim_size) {
  Dimensions input_shape = CopyIntVector(GetTensorTypeOrDie(input).getShape());
  int64_t offset_dim_size = input_shape.size() - spatial_dim_size;

  // We gather on the spatial dimensions.
  // slice_sizes will be 1 for spatial dims, and unchanged for offset dims.
  auto slice_sizes = input_shape;
  for (int i = 0; i < spatial_dim_size; ++i) {
    slice_sizes[offset_dim_size + i] = 1;
  }

  SmallInt64Vector collapsed_slice_dims;
  SmallInt64Vector start_index_map;
  for (int i = 0; i < spatial_dim_size; ++i) {
    collapsed_slice_dims.push_back(offset_dim_size + i);
    start_index_map.push_back(offset_dim_size + i);
  }

  SmallInt64Vector offset_dims;
  offset_dims.reserve(offset_dim_size);
  for (int i = 0; i < offset_dim_size; ++i) {
    offset_dims.push_back(i);
  }

  mlir::stablehlo::GatherDimensionNumbersAttr gather_dimension_numbers =
      mlir::stablehlo::GatherDimensionNumbersAttr::get(
          &input.getContext(),
          /*offset_dims=*/AsArrayRef<int64_t>(offset_dims),
          /*collapsed_slice_dims=*/AsArrayRef<int64_t>(collapsed_slice_dims),
          /*operand_batching_dims=*/{},
          /*start_indices_batching_dims=*/{},
          /*start_index_map=*/AsArrayRef<int64_t>(start_index_map),
          /*index_vector_dim=*/spatial_dim_size);

  return {gather_dimension_numbers, slice_sizes};
}

// Stores the floor and ceil indices and their corresponding weights (lambda)
// for linear interpolation in a single spatial dimension.
struct DimInfo {
  // Lower bound index (floor(source_index))
  mlir::MlirOp idx_floor;
  // Upper bound index (ceil(source_index))
  mlir::MlirOp idx_ceil;
  // Weight for the lower bound index (1 - lambda)
  mlir::MlirOp weight_floor;
  // Weight for the upper bound index (lambda)
  mlir::MlirOp weight_ceil;
};

// This function calculates where in the input image we should look for each
// pixel in the output image.
//
// Logic: Creates an iota (sequence 0, 1, 2...) for the output dimension.
// Align Corners:
//   - True: Maps the first pixel to first and last to last exactly.
//     src_idx = dst_idx * (in_size - 1) / (out_size - 1)
//   - False: Treats pixels as squares with centers.
//     src_idx = (dst_idx + 0.5) * scale - 0.5
// Result: A tensor of floating point coordinates.
absl::StatusOr<mlir::MlirOp> ComputeSourceIndex(
    mlir::MlirBuilder& builder, int64_t out_size, int64_t in_size,
    std::optional<double> scale_opt, bool align_corners, mlir::Type calc_type) {
  auto iota_type = mlir::RankedTensorType::get({out_size}, calc_type);
  auto iota = mlir::stablehlo::Iota(builder, iota_type, 0);
  mlir::MlirOp src_idx;

  if (align_corners) {
    // align_corners=True: src_idx = scale * dst_idx
    // scale = (in_size - 1) / (out_size - 1)
    double stride = (out_size > 1) ? static_cast<double>(in_size - 1) /
                                         static_cast<double>(out_size - 1)
                                   : 0.0;

    auto stride_const = MakeConstant(builder, stride, calc_type, out_size);
    TT_ASSIGN_OR_RETURN(src_idx, BuildMulShlo(iota, stride_const));
  } else {
    // align_corners=False: src_idx = (dst_idx + 0.5) * scale - 0.5
    // scale = in_size / out_size
    double scale = static_cast<double>(in_size) / static_cast<double>(out_size);
    if (scale_opt.has_value()) {
      scale = 1.0 / (*scale_opt);
    }
    auto half_const = MakeConstant(builder, 0.5, calc_type, out_size);
    TT_ASSIGN_OR_RETURN(auto iota_plus_half, BuildAddShlo(iota, half_const));

    auto scale_const = MakeConstant(builder, scale, calc_type, out_size);
    TT_ASSIGN_OR_RETURN(auto scaled, BuildMulShlo(iota_plus_half, scale_const));
    TT_ASSIGN_OR_RETURN(src_idx, BuildSubShlo(scaled, half_const));
  }

  return src_idx;
}

// Computes the gather indices (floor/ceil) and interpolation weights (lambda)
// for a single spatial dimension. For a single dimension (e.g., Height), this
// breaks the continuous coordinate into discrete lookup indices and
// interpolation weights.
//
// Floor & Ceil:
//   - idx_floor_calc = floor(src_idx)
//   - idx_ceil_calc = floor(src_idx) + 1
//
// Weights (Lambda):
//   - lambda is the fractional part: src_idx - floor.
//   - If src_idx is 3.7, floor is 3, lambda is 0.7.
//   - We trust index 3 (floor) with weight 1 - 0.7 = 0.3.
//   - We trust index 4 (ceil) with weight 0.7.
//
// Result: DimInfo struct containing idx_floor, idx_ceil, weight_floor,
// weight_ceil.
absl::StatusOr<DimInfo> ComputeDimInfo(
    mlir::MlirBuilder& builder, mlir::MlirOp src_idx, int64_t out_size,
    int64_t in_size, mlir::Type calc_type,
    mlir::RankedTensorType broadcast_type,
    mlir::RankedTensorType broadcast_type_calc, int spatial_dim_index) {
  auto idx_floor_calc = mlir::stablehlo::Floor(src_idx);
  auto one_calc = MakeConstant(builder, 1.0, calc_type, out_size);
  TT_ASSIGN_OR_RETURN(auto idx_ceil_calc,
                      BuildAddShlo(idx_floor_calc, one_calc));

  TT_ASSIGN_OR_RETURN(auto lambda, BuildSubShlo(src_idx, idx_floor_calc));
  TT_ASSIGN_OR_RETURN(auto one_minus_lambda, BuildSubShlo(one_calc, lambda));

  // Convert indices to i32 for Gather
  auto idx_floor_i32_unclamped = mlir::stablehlo::ConvertElementType(
      idx_floor_calc, builder.getOpBuilder().getI32Type());
  auto idx_ceil_i32_unclamped = mlir::stablehlo::ConvertElementType(
      idx_ceil_calc, builder.getOpBuilder().getI32Type());

  auto zero_i32 =
      MakeConstant(builder, 0, builder.getOpBuilder().getI32Type(), out_size);
  auto in_size_minus_1_i32 =
      MakeConstant(builder, std::max(0L, in_size - 1),
                   builder.getOpBuilder().getI32Type(), out_size);

  auto idx_floor_i32 = mlir::stablehlo::Clamp(idx_floor_i32_unclamped, zero_i32,
                                              in_size_minus_1_i32);
  auto idx_ceil_i32 = mlir::stablehlo::Clamp(idx_ceil_i32_unclamped, zero_i32,
                                             in_size_minus_1_i32);

  // Broadcast to rank+1 for concatenation
  auto b_idx_floor = mlir::stablehlo::BroadcastInDim(
      broadcast_type, idx_floor_i32, {spatial_dim_index});
  auto b_idx_ceil = mlir::stablehlo::BroadcastInDim(
      broadcast_type, idx_ceil_i32, {spatial_dim_index});

  auto b_lambda = mlir::stablehlo::BroadcastInDim(broadcast_type_calc, lambda,
                                                  {spatial_dim_index});
  auto b_one_minus_lambda = mlir::stablehlo::BroadcastInDim(
      broadcast_type_calc, one_minus_lambda, {spatial_dim_index});

  return DimInfo{b_idx_floor, b_idx_ceil, b_one_minus_lambda, b_lambda};
}

absl::StatusOr<std::pair<std::vector<mlir::MlirOp>, mlir::MlirOp>>
ComputeCornerIndicesAndWeight(int spatial_dim_size, int corner,
                              const std::vector<DimInfo>& dim_infos) {
  std::vector<mlir::MlirOp> current_indices;
  mlir::MlirOp current_weight;

  for (int i = 0; i < spatial_dim_size; ++i) {
    bool use_ceil = (corner >> (spatial_dim_size - 1 - i)) & 1;
    current_indices.push_back(use_ceil ? dim_infos[i].idx_ceil
                                       : dim_infos[i].idx_floor);

    mlir::MlirOp w =
        use_ceil ? dim_infos[i].weight_ceil : dim_infos[i].weight_floor;
    if (i == 0) {
      current_weight = w;
    } else {
      TT_ASSIGN_OR_RETURN(current_weight, BuildMulShlo(current_weight, w));
    }
  }
  return std::make_pair(current_indices, current_weight);
}

// Accumulates the weighted values from all 2^k corners for bilinear
// interpolation. For 2d interpolation, this function constructs the new image
// by summing e.g. 2^2=4 corners.
//
// It iterates 4 times (num_corners = 1 << 2):
//   1. Top-Left (Floor H, Floor W)
//   2. Top-Right (Floor H, Ceil W)
//   3. Bottom-Left (Ceil H, Floor W)
//   4. Bottom-Right (Ceil H, Ceil W)
//
// Inside the loop:
//   - ComputeCornerIndicesAndWeight:
//     - Selects the correct index (floor vs ceil) for H and W based on the
//     current corner iteration.
//     - Multiplies the weights for H and W.
//     - e.g. For Top-Left: Weight = (1 - lambda_h) * (1 - lambda_w)
//   - Concatenate: Combines H and W indices into a coordinate vector for
//   Gather.
//   - Gather: Fetches pixel values from the input image at these coordinates.
//     - Uses gather_config from earlier to know it should slice spatial dims
//     and batch the others.
//   - Weight & Accumulate:
//     - weighted_value = gathered_pixel * corner_weight
//     - accumulated_result += weighted_value
absl::StatusOr<mlir::MlirOp> AccumulateBilinearInterpolation(
    mlir::MlirBuilder& builder, mlir::MlirOp input,
    at::IntArrayRef output_shape, const std::vector<DimInfo>& dim_infos,
    const UpsampleGatherConfig& gather_config, int64_t offset_dim_size,
    int64_t spatial_dim_size, mlir::RankedTensorType broadcast_type,
    mlir::Type calc_type) {
  // Generate 2^kDimensions corners (e.g., 4 for 2D bilinear: top-left,
  // top-right, bottom-left, bottom-right).
  int num_corners = 1 << spatial_dim_size;
  mlir::MlirOp accumulated_result;

  SmallInt64Vector result_broadcast_dims;
  result_broadcast_dims.reserve(spatial_dim_size);
  for (int i = 0; i < spatial_dim_size; ++i) {
    result_broadcast_dims.push_back(offset_dim_size + i);
  }

  for (int corner = 0; corner < num_corners; ++corner) {
    TT_ASSIGN_OR_RETURN(
        auto indices_and_weight,
        ComputeCornerIndicesAndWeight(spatial_dim_size, corner, dim_infos));
    auto current_indices = indices_and_weight.first;
    auto current_weight = indices_and_weight.second;

    // Concatenate indices for the gather operation.
    auto index_tensor = mlir::stablehlo::Concatenate(
        builder, current_indices, output_shape.size() - offset_dim_size);

    auto gathered = mlir::stablehlo::Gather(
        input, index_tensor, gather_config.gather_dimension_numbers,
        gather_config.slice_sizes,
        /*indices_are_sorted=*/false);

    auto weight_type_sq = mlir::RankedTensorType::get(
        broadcast_type.getShape().drop_back(), calc_type);
    auto weight_sq = mlir::stablehlo::Reshape(weight_type_sq, current_weight);

    // Convert gathered to F32/F64 for precision. Use calc_type.
    auto gathered_calc =
        mlir::stablehlo::ConvertElementType(gathered, calc_type);

    auto weight_bcast = mlir::stablehlo::BroadcastInDim(
        gathered_calc.getType(), weight_sq,
        AsArrayRef<int64_t>(result_broadcast_dims));

    TT_ASSIGN_OR_RETURN(auto weighted_value,
                        BuildMulShlo(gathered_calc, weight_bcast));

    if (corner == 0) {
      accumulated_result = weighted_value;
    } else {
      TT_ASSIGN_OR_RETURN(accumulated_result,
                          BuildAddShlo(accumulated_result, weighted_value));
    }
  }
  return accumulated_result;
}

// Prep: Gets input/output shapes and setup types.
// Gather Config: Calls GetUpsampleGatherConfig to determine which dimensions
//   are spatial (gathered) vs offset (preserved).
// Dimension Processing: Loops through each spatial dimension to pre-compute
//   indices and weights.
//   - ComputeSourceIndex: Maps output coordinate -> float input coordinate.
//   - ComputeDimInfo: Converts float coordinate -> integer indices (floor/ceil)
//     and weights.
// Accumulation: Calls AccumulateBilinearInterpolation to combine the results.
// kDimensions: The number of spatial dimensions (2 for bilinear 2d).
absl::StatusOr<mlir::MlirOp> BuildUpsampleBilinearShlo(
    mlir::MlirOp input, int num_dimensions, at::IntArrayRef output_shape,
    bool align_corners, const ScaleVector& scales_opt) {
  auto& builder = input.getBuilder();

  // Input shape: (N, C, H_in, W_in)
  // Output shape: (N, C, H_out, W_out)
  Dimensions input_shape = CopyIntVector(GetTensorTypeOrDie(input).getShape());
  int64_t spatial_dim_size = num_dimensions;
  int64_t offset_dim_size = input_shape.size() - spatial_dim_size;

  UpsampleGatherConfig gather_config =
      GetUpsampleGatherConfig(input, spatial_dim_size);

  // Helper to compute source index and weight for a dimension.
  auto element_type = GetTensorTypeOrDie(input).getElementType();
  mlir::Type calc_type = builder.getOpBuilder().getF32Type();
  if (element_type.isF64()) {
    calc_type = builder.getOpBuilder().getF64Type();
  }

  std::vector<DimInfo> dim_infos;
  dim_infos.reserve(spatial_dim_size);

  // Create broadcast type for index reshaping
  Dimensions broadcast_shape_array(num_dimensions + 1);
  broadcast_shape_array[num_dimensions] = 1;
  for (int i = 0; i < num_dimensions; ++i) {
    broadcast_shape_array[i] = output_shape[offset_dim_size + i];
  }
  auto broadcast_type = mlir::RankedTensorType::get(
      broadcast_shape_array, builder.getOpBuilder().getI32Type());

  auto broadcast_type_calc =
      mlir::RankedTensorType::get(broadcast_shape_array, calc_type);

  for (int i = 0; i < spatial_dim_size; ++i) {
    int64_t out_size = output_shape[offset_dim_size + i];
    int64_t in_size = input_shape[offset_dim_size + i];
    auto scale_opt = scales_opt[i];

    TT_ASSIGN_OR_RETURN(
        auto src_idx, ComputeSourceIndex(builder, out_size, in_size, scale_opt,
                                         align_corners, calc_type));

    TT_ASSIGN_OR_RETURN(
        auto dim_info,
        ComputeDimInfo(builder, src_idx, out_size, in_size, calc_type,
                       broadcast_type, broadcast_type_calc, i));
    dim_infos.push_back(dim_info);
  }

  TT_ASSIGN_OR_RETURN(
      auto accumulated_result,
      AccumulateBilinearInterpolation(
          builder, input, output_shape, dim_infos, gather_config,
          offset_dim_size, spatial_dim_size, broadcast_type, calc_type));

  return mlir::stablehlo::ConvertElementType(accumulated_result, element_type);
}

absl::StatusOr<mlir::MlirOp> ScatterBilinearBackwardForSingleCorner(
    mlir::MlirBuilder& builder, mlir::MlirOp grad_input, int spatial_dim_size,
    int offset_dim_size, int corner, const std::vector<DimInfo>& dim_infos,
    const UpsampleScatterConfig& scatter_config,
    absl::FunctionRef<void(mlir::RegionBuilder&)> scatter_body,
    mlir::MlirOp grad_output, at::IntArrayRef grad_output_shape,
    mlir::MlirOp grad_output_calc, mlir::RankedTensorType broadcast_type,
    mlir::Type calc_type, mlir::Type element_type) {
  std::vector<mlir::MlirOp> current_indices_1d;
  mlir::MlirOp current_weight_1d;
  for (int i = 0; i < spatial_dim_size; ++i) {
    bool use_ceil = (corner >> (spatial_dim_size - 1 - i)) & 1;
    current_indices_1d.push_back(use_ceil ? dim_infos[i].idx_ceil
                                          : dim_infos[i].idx_floor);
    mlir::MlirOp w =
        use_ceil ? dim_infos[i].weight_ceil : dim_infos[i].weight_floor;
    if (i == 0) {
      current_weight_1d = w;
    } else {
      TT_ASSIGN_OR_RETURN(current_weight_1d,
                          BuildMulShlo(current_weight_1d, w));
    }
  }

  auto corner_indices_spatial = mlir::stablehlo::Concatenate(
      builder, current_indices_1d, spatial_dim_size);

  Dimensions scatter_indices_shape_dims = CopyIntVector(grad_output_shape);
  scatter_indices_shape_dims.push_back(spatial_dim_size);
  auto scatter_indices_type = mlir::RankedTensorType::get(
      scatter_indices_shape_dims, builder.getOpBuilder().getI32Type());
  Dimensions indices_bcast_dims;
  for (int i = 0; i < spatial_dim_size; ++i) {
    indices_bcast_dims.push_back(offset_dim_size + i);
  }
  indices_bcast_dims.push_back(grad_output_shape.size());

  auto scatter_indices = mlir::stablehlo::BroadcastInDim(
      scatter_indices_type, corner_indices_spatial, indices_bcast_dims);

  auto weight_bcast_type =
      mlir::RankedTensorType::get(grad_output_shape, calc_type);
  Dimensions weight_bcast_dims;
  for (int i = 0; i < spatial_dim_size; ++i) {
    weight_bcast_dims.push_back(offset_dim_size + i);
  }
  Dimensions squeezed_shape;
  for (int i = 0; i < spatial_dim_size; ++i) {
    squeezed_shape.push_back(grad_output_shape[offset_dim_size + i]);
  }
  auto corner_weight_squeezed_type =
      mlir::RankedTensorType::get(squeezed_shape, calc_type);
  auto corner_weight_squeezed =
      mlir::stablehlo::Reshape(corner_weight_squeezed_type, current_weight_1d);

  auto weight_bcast = mlir::stablehlo::BroadcastInDim(
      weight_bcast_type, corner_weight_squeezed, weight_bcast_dims);

  TT_ASSIGN_OR_RETURN(auto update_calc,
                      BuildMulShlo(grad_output_calc, weight_bcast));
  auto update = mlir::stablehlo::ConvertElementType(update_calc, element_type);

  return mlir::stablehlo::Scatter(grad_input, scatter_indices, update,
                                  scatter_body,
                                  scatter_config.scatter_dimension_numbers,
                                  /*indices_are_sorted=*/false)[0];
}

absl::StatusOr<mlir::MlirOp> BuildUpsampleBilinearBackwardShlo(
    mlir::MlirOp grad_output, int num_dimensions,
    at::IntArrayRef grad_input_shape, bool align_corners,
    const ScaleVector& scales_opt) {
  mlir::MlirBuilder& builder = grad_output.getBuilder();
  at::IntArrayRef grad_output_shape =
      GetTensorTypeOrDie(grad_output).getShape();
  const int64_t spatial_dim_size = num_dimensions;
  const int64_t offset_dim_size = grad_output_shape.size() - spatial_dim_size;

  auto element_type = GetTensorTypeOrDie(grad_output).getElementType();
  mlir::Type calc_type = builder.getOpBuilder().getF32Type();
  if (element_type.isF64()) {
    calc_type = builder.getOpBuilder().getF64Type();
  }

  std::vector<DimInfo> dim_infos;
  dim_infos.reserve(spatial_dim_size);

  Dimensions broadcast_shape_array(num_dimensions + 1);
  broadcast_shape_array[num_dimensions] = 1;
  for (int i = 0; i < num_dimensions; ++i) {
    broadcast_shape_array[i] = grad_output_shape[offset_dim_size + i];
  }
  auto broadcast_type = mlir::RankedTensorType::get(
      broadcast_shape_array, builder.getOpBuilder().getI32Type());
  auto broadcast_type_calc =
      mlir::RankedTensorType::get(broadcast_shape_array, calc_type);

  for (int i = 0; i < spatial_dim_size; ++i) {
    int64_t out_size = grad_output_shape[offset_dim_size + i];
    int64_t in_size = grad_input_shape[offset_dim_size + i];
    auto scale_opt = scales_opt[i];

    TT_ASSIGN_OR_RETURN(
        auto src_idx, ComputeSourceIndex(builder, out_size, in_size, scale_opt,
                                         align_corners, calc_type));
    TT_ASSIGN_OR_RETURN(
        auto dim_info,
        ComputeDimInfo(builder, src_idx, out_size, in_size, calc_type,
                       broadcast_type, broadcast_type_calc, i));
    dim_infos.push_back(dim_info);
  }

  // grad_input = zeros
  mlir::MlirOp grad_input =
      MakeConstant(builder, 0.0, element_type, grad_input_shape);
  mlir::MlirOp grad_output_calc =
      mlir::stablehlo::ConvertElementType(grad_output, calc_type);

  auto scatter_config = GetUpsampleScatterConfig(grad_output, spatial_dim_size);
  auto scatter_body = [element_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        element_type, rb.getRegion(), rb.getOpBuilder());
  };

  const int num_corners = 1 << spatial_dim_size;
  for (int corner = 0; corner < num_corners; ++corner) {
    TT_ASSIGN_OR_RETURN(
        grad_input, ScatterBilinearBackwardForSingleCorner(
                        builder, grad_input, spatial_dim_size, offset_dim_size,
                        corner, dim_infos, scatter_config, scatter_body,
                        grad_output, grad_output_shape, grad_output_calc,
                        broadcast_type, calc_type, element_type));
  }

  return grad_input;
}

absl::StatusOr<mlir::MlirOp> BuildUpsampleNearestSingleDimIndexTensor(
    mlir::MlirBuilder& builder, int num_dimensions, int dimension,
    mlir::MlirOp grad_output, at::IntArrayRef grad_output_shape,
    const MlirOpVector& inverse_scale_factor_array,
    UpsampleMode upsample_mode) {
  auto inverse_scale_factor_float32 =
      mlir::stablehlo::ConvertElementType(inverse_scale_factor_array[dimension],
                                          builder.getOpBuilder().getF32Type());
  auto iota_type = mlir::RankedTensorType::get(
      grad_output_shape[dimension + 2], builder.getOpBuilder().getF32Type());
  auto iota = mlir::stablehlo::Iota(builder, iota_type, 0);
  mlir::MlirOp offset_iota = [&]() {
    if (upsample_mode != UpsampleMode::kNearestExact) {
      return iota;
    }
    float center_offset_val = 0.5f;
    mlir::MlirOp center_offset_op =
        MakeConstant(builder, center_offset_val, mlir::ElementType::F32,
                     grad_output_shape[dimension + 2]);
    return mlir::stablehlo::Add(iota, center_offset_op);
  }();
  TT_ASSIGN_OR_RETURN(mlir::MlirOp scaled_iota,
                      BuildMulShlo(offset_iota, inverse_scale_factor_float32));
  return mlir::stablehlo::ConvertElementType(
      mlir::stablehlo::Floor(scaled_iota), builder.getOpBuilder().getI32Type());
}

absl::StatusOr<mlir::MlirOp> BuildUpsampleNearestIndexTensor(
    mlir::MlirBuilder& builder, int num_dimensions, mlir::MlirOp input,
    at::IntArrayRef output_shape,
    const MlirOpVector& inverse_scale_factor_array,
    UpsampleMode upsample_mode) {
  Dimensions broadcast_shape_array(num_dimensions + 1);
  broadcast_shape_array[num_dimensions] = 1;
  for (int i = 0; i < num_dimensions; ++i) {
    broadcast_shape_array[i] = output_shape[i + 2];
  }
  auto broadcast_type = mlir::RankedTensorType::get(
      broadcast_shape_array, builder.getOpBuilder().getI32Type());

  // Generate the set of reshaped index tensors to form the gather tensor.
  MlirOpVector index_tensor_reshape_array(num_dimensions);
  for (int i = 0; i < num_dimensions; ++i) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp index_tensor,
                        BuildUpsampleNearestSingleDimIndexTensor(
                            builder, num_dimensions, i, input, output_shape,
                            inverse_scale_factor_array, upsample_mode));

    // Reshape and project the index tensors into the same shape for the gather
    // dimensions
    index_tensor_reshape_array[i] =
        mlir::stablehlo::BroadcastInDim(broadcast_type, index_tensor, {i});
  }

  // Concatenate the index tensors into a single index tensor with gather
  // dimension pairs in the third dimension. This creates a tensor where the
  // final dimension is of 1, 2 or 3-element tuples that address the input
  // tensor in its gather dimensions.
  return mlir::stablehlo::Concatenate(
      input.getBuilder(), index_tensor_reshape_array, output_shape.size() - 2);
}

absl::StatusOr<mlir::MlirOp> BuildUpsampleNearestShlo(
    mlir::MlirOp input, at::IntArrayRef output_shape,
    const MlirOpVector& inverse_scale_factor_array,
    UpsampleMode upsample_mode) {
  mlir::MlirBuilder& builder = input.getBuilder();
  const int num_dimensions = inverse_scale_factor_array.size();
  TT_ASSIGN_OR_RETURN(mlir::MlirOp index_tensor,
                      BuildUpsampleNearestIndexTensor(
                          builder, num_dimensions, input, output_shape,
                          inverse_scale_factor_array, upsample_mode));

  UpsampleGatherConfig gather_config =
      GetUpsampleGatherConfig(input, num_dimensions);
  auto& slice_sizes = gather_config.slice_sizes;
  auto& gather_dimension_numbers = gather_config.gather_dimension_numbers;

  return mlir::stablehlo::Gather(input, index_tensor, gather_dimension_numbers,
                                 slice_sizes,
                                 /*indices_are_sorted=*/false);
}

absl::StatusOr<mlir::MlirOp> BuildUpsampleNearestBackwardIndexTensor(
    mlir::MlirBuilder& builder, int num_dimensions, mlir::MlirOp grad_output,
    at::IntArrayRef grad_input_shape,
    const MlirOpVector& inverse_scale_factor_array,
    UpsampleMode upsample_mode) {
  at::IntArrayRef grad_output_shape =
      GetTensorTypeOrDie(grad_output).getShape();
  int64_t rank = grad_output_shape.size();
  auto broadcast_to_indices_type = mlir::RankedTensorType::get(
      grad_output_shape, builder.getOpBuilder().getI32Type());

  // 1. Compute scatter indices
  MlirOpVector index_tensors_to_concat(num_dimensions);
  for (int i = 0; i < num_dimensions; ++i) {
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp index_tensor,
        BuildUpsampleNearestSingleDimIndexTensor(
            builder, num_dimensions, i, grad_output, grad_output_shape,
            inverse_scale_factor_array, upsample_mode));

    auto broadcast_indices = mlir::stablehlo::BroadcastInDim(
        broadcast_to_indices_type, index_tensor, {i + 2});

    SmallInt64Vector reshape_dims = CopyIntVector(grad_output_shape);
    reshape_dims.push_back(1);
    auto reshape_type = mlir::RankedTensorType::get(
        reshape_dims, builder.getOpBuilder().getI32Type());
    index_tensors_to_concat[i] =
        mlir::stablehlo::Reshape(reshape_type, broadcast_indices);
  }
  return mlir::stablehlo::Concatenate(builder, index_tensors_to_concat, rank);
}

absl::StatusOr<mlir::MlirOp> BuildUpsampleNearestBackwardShlo(
    mlir::MlirOp grad_output, at::IntArrayRef grad_input_shape,
    const MlirOpVector& inverse_scale_factor_array,
    UpsampleMode upsample_mode) {
  mlir::MlirBuilder& builder = grad_output.getBuilder();
  const int num_dimensions = inverse_scale_factor_array.size();
  TT_ASSIGN_OR_RETURN(
      auto scatter_indices,
      BuildUpsampleNearestBackwardIndexTensor(
          builder, num_dimensions, grad_output, grad_input_shape,
          inverse_scale_factor_array, upsample_mode));

  // 2. Create zero tensor to accumulate the grad_input into
  auto grad_input_type = mlir::RankedTensorType::get(
      grad_input_shape, GetTensorTypeOrDie(grad_output).getElementType());
  auto element_type = grad_input_type.getElementType();
  auto grad_input_zeros =
      MakeConstant(builder, 0.0, element_type, grad_input_shape);

  // 3. Call scatter
  auto scatter_config = GetUpsampleScatterConfig(grad_output, num_dimensions);

  auto scatter_body = [element_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        element_type, rb.getRegion(), rb.getOpBuilder());
  };

  return mlir::stablehlo::Scatter(grad_input_zeros, scatter_indices,
                                  grad_output, scatter_body,
                                  scatter_config.scatter_dimension_numbers,
                                  /*indices_are_sorted=*/false)[0];
}

absl::InlinedVector<double, 6> StandardizeScaleFactors(
    at::IntArrayRef input_shape, at::IntArrayRef upsample_shape,
    absl::InlinedVector<std::optional<double>, 6> scale_array, bool inverse) {
  // Generate a scale factor from the output shape when scale factor not
  // provided.
  absl::InlinedVector<double, 6> scale_factor_result(upsample_shape.size());
  for (int i = 0; i < upsample_shape.size(); ++i) {
    double scale_factor = scale_array[i].value_or(
        static_cast<double>(upsample_shape[i]) / input_shape[i + 2]);

    float maybe_inverse_scale_factor =
        inverse ? (1.0 / scale_factor) : scale_factor;
    scale_factor_result[i] = maybe_inverse_scale_factor;
  }

  return scale_factor_result;
}

absl::StatusOr<TensorVector> ConstructScaleFactorArray(
    absl::InlinedVector<double, 6> scale_array) {
  // Generate a scale factor from the output shape when scale factor not
  // provided.

  TensorVector scale_factor_result(scale_array.size());
  for (int i = 0; i < scale_factor_result.size(); ++i) {
    at::Scalar maybe_inverse_scale_factor_scalar(scale_array[i]);
    TT_ASSIGN_OR_RETURN(scale_factor_result[i],
                        MakeTensor(maybe_inverse_scale_factor_scalar));
  }

  return scale_factor_result;
}

}  // namespace

at::Tensor& AtenUpsampleNearest1dBackwardGradInput(
    const at::Tensor& grad_output, at::IntArrayRef output_shape_in,
    at::IntArrayRef input_shape, std::optional<double> scale,
    at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kUpsampleNearest1dBackwardGradInput, param_keys,
      (grad_output, output_shape_in, input_shape, scale, grad_input), {
        auto output_shape = CopyIntVector(grad_output.sizes());

        // TODO(lwh): Add check for output_shape and scaled version from
        // input_shape, and check that grad_input and grad_output shapes match
        // those.
        auto preprocessed_scale_factors = StandardizeScaleFactors(
            input_shape, output_shape_in, {scale}, /*inverse=*/true);
        TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<mlir::ElementType>(
                                                  grad_input.scalar_type()));
        auto op_builder =
            [output_shape, grad_input_shape = CopyIntVector(input_shape),
             preprocessed_scale_factors](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [grad_output, inverse_scale_factor_tensor] = inputs;
          TT_ASSIGN_OR_RETURN(
              auto grad_input,
              BuildUpsampleNearestBackwardShlo(grad_output, grad_input_shape,
                                               {inverse_scale_factor_tensor},
                                               UpsampleMode::kNearest));
          return grad_input;
        };
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_scale_factors));

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<2>(std::move(op_builder),
                           {grad_output, inverse_scale_factors[0]},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(input_shape),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(grad_input_buf), grad_input));
        return grad_input;
      });
}

at::Tensor& AtenUpsampleNearest2dBackwardGradInput(
    const at::Tensor& grad_output, at::IntArrayRef output_shape_in,
    at::IntArrayRef input_shape, std::optional<double> scale_h,
    std::optional<double> scale_w, at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kUpsampleNearest2dBackwardGradInput, param_keys,
      (grad_output, output_shape_in, input_shape, scale_h, scale_w, grad_input),
      {
        auto output_shape = CopyIntVector(grad_output.sizes());
        // TODO(lwh): Add check for output_shape and scaled version from
        // input_shape, and check that grad_input and grad_output shapes match
        // those.
        auto preprocessed_scale_factors = StandardizeScaleFactors(
            input_shape, output_shape_in, {scale_h, scale_w}, /*inverse=*/true);
        TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<mlir::ElementType>(
                                                  grad_input.scalar_type()));
        auto op_builder =
            [output_shape, grad_input_shape = CopyIntVector(input_shape),
             preprocessed_scale_factors](FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [grad_output, inverse_scale_factor_tensor_h,
                 inverse_scale_factor_tensor_w] = inputs;
          TT_ASSIGN_OR_RETURN(
              auto grad_input,
              BuildUpsampleNearestBackwardShlo(grad_output, grad_input_shape,
                                               {inverse_scale_factor_tensor_h,
                                                inverse_scale_factor_tensor_w},
                                               UpsampleMode::kNearest));
          return grad_input;
        };
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_scale_factors));

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<3>(std::move(op_builder),
                           {grad_output, inverse_scale_factors[0],
                            inverse_scale_factors[1]},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(input_shape),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(grad_input_buf), grad_input));
        return grad_input;
      });
}

at::Tensor& AtenUpsampleNearest3dBackwardGradInput(
    const at::Tensor& grad_output, at::IntArrayRef output_shape_in,
    at::IntArrayRef input_shape, std::optional<double> scale_h,
    std::optional<double> scale_w, std::optional<double> scale_d,
    at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kUpsampleNearest3dBackwardGradInput, param_keys,
      (grad_output, output_shape_in, input_shape, scale_h, scale_w, scale_d,
       grad_input),
      {
        auto output_shape = CopyIntVector(grad_output.sizes());
        // TODO(lwh): Add check for output_shape and scaled version from
        // input_shape, and check that grad_input and grad_output shapes match
        // those.
        auto preprocessed_scale_factors = StandardizeScaleFactors(
            input_shape, output_shape_in, {scale_h, scale_w, scale_d},
            /*inverse=*/true);
        TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<mlir::ElementType>(
                                                  grad_input.scalar_type()));
        auto op_builder =
            [output_shape, grad_input_shape = CopyIntVector(input_shape),
             preprocessed_scale_factors](FixedSizeSpan<mlir::MlirOp, 4> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [grad_output, inverse_scale_factor_tensor_h,
                 inverse_scale_factor_tensor_w, inverse_scale_factor_tensor_d] =
              inputs;
          TT_ASSIGN_OR_RETURN(
              auto grad_input,
              BuildUpsampleNearestBackwardShlo(
                  grad_output, grad_input_shape,
                  {inverse_scale_factor_tensor_h, inverse_scale_factor_tensor_w,
                   inverse_scale_factor_tensor_d},
                  UpsampleMode::kNearest));
          return grad_input;
        };
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_scale_factors));

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<4>(std::move(op_builder),
                           {grad_output, inverse_scale_factors[0],
                            inverse_scale_factors[1], inverse_scale_factors[2]},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(input_shape),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(grad_input_buf), grad_input));
        return grad_input;
      });
}

at::Tensor& AtenUpsampleNearestExact1dBackwardGradInput(
    const at::Tensor& grad_output, at::IntArrayRef output_shape_in,
    at::IntArrayRef input_shape, std::optional<double> scale,
    at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kUpsampleNearestExact1dBackwardGradInput, param_keys,
      (grad_output, output_shape_in, input_shape, scale, grad_input), {
        auto output_shape = CopyIntVector(grad_output.sizes());
        // TODO(lwh): Add check for output_shape and scaled version from
        // input_shape, and check that grad_input and grad_output shapes match
        // those.
        auto preprocessed_scale_factors = StandardizeScaleFactors(
            input_shape, output_shape_in, {scale}, /*inverse=*/true);
        TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<mlir::ElementType>(
                                                  grad_input.scalar_type()));
        auto op_builder =
            [output_shape, grad_input_shape = CopyIntVector(input_shape),
             preprocessed_scale_factors](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [grad_output, inverse_scale_factor_tensor] = inputs;
          TT_ASSIGN_OR_RETURN(
              auto grad_input,
              BuildUpsampleNearestBackwardShlo(grad_output, grad_input_shape,
                                               {inverse_scale_factor_tensor},
                                               UpsampleMode::kNearestExact));
          return grad_input;
        };
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_scale_factors));

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<2>(std::move(op_builder),
                           {grad_output, inverse_scale_factors[0]},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(input_shape),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(grad_input_buf), grad_input));
        return grad_input;
      });
}

at::Tensor& AtenUpsampleNearestExact2dBackwardGradInput(
    const at::Tensor& grad_output, at::IntArrayRef output_shape_in,
    at::IntArrayRef input_shape, std::optional<double> scale_h,
    std::optional<double> scale_w, at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kUpsampleNearestExact2dBackwardGradInput, param_keys,
      (grad_output, output_shape_in, input_shape, scale_h, scale_w, grad_input),
      {
        auto output_shape = CopyIntVector(grad_output.sizes());
        // TODO(lwh): Add check for output_shape and scaled version from
        // input_shape, and check that grad_input and grad_output shapes match
        // those.
        auto preprocessed_scale_factors = StandardizeScaleFactors(
            input_shape, output_shape_in, {scale_h, scale_w}, /*inverse=*/true);
        TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<mlir::ElementType>(
                                                  grad_input.scalar_type()));
        auto op_builder =
            [output_shape, grad_input_shape = CopyIntVector(input_shape),
             preprocessed_scale_factors](FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [grad_output, inverse_scale_factor_tensor_h,
                 inverse_scale_factor_tensor_w] = inputs;
          TT_ASSIGN_OR_RETURN(
              auto grad_input,
              BuildUpsampleNearestBackwardShlo(grad_output, grad_input_shape,
                                               {inverse_scale_factor_tensor_h,
                                                inverse_scale_factor_tensor_w},
                                               UpsampleMode::kNearestExact));
          return grad_input;
        };
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_scale_factors));

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<3>(std::move(op_builder),
                           {grad_output, inverse_scale_factors[0],
                            inverse_scale_factors[1]},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(input_shape),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(grad_input_buf), grad_input));
        return grad_input;
      });
}

at::Tensor& AtenUpsampleNearestExact3dBackwardGradInput(
    const at::Tensor& grad_output, at::IntArrayRef output_shape_in,
    at::IntArrayRef input_shape, std::optional<double> scale_h,
    std::optional<double> scale_w, std::optional<double> scale_d,
    at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kUpsampleNearestExact3dBackwardGradInput, param_keys,
      (grad_output, output_shape_in, input_shape, scale_h, scale_w, scale_d,
       grad_input),
      {
        auto output_shape = CopyIntVector(grad_output.sizes());
        // TODO(lwh): Add check for output_shape and scaled version from
        // input_shape, and check that grad_input and grad_output shapes match
        // those.
        auto preprocessed_scale_factors = StandardizeScaleFactors(
            input_shape, output_shape_in, {scale_h, scale_w, scale_d},
            /*inverse=*/true);
        TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<mlir::ElementType>(
                                                  grad_input.scalar_type()));
        auto op_builder =
            [output_shape, grad_input_shape = CopyIntVector(input_shape),
             preprocessed_scale_factors](FixedSizeSpan<mlir::MlirOp, 4> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [grad_output, inverse_scale_factor_tensor_h,
                 inverse_scale_factor_tensor_w, inverse_scale_factor_tensor_d] =
              inputs;
          TT_ASSIGN_OR_RETURN(
              auto grad_input,
              BuildUpsampleNearestBackwardShlo(
                  grad_output, grad_input_shape,
                  {inverse_scale_factor_tensor_h, inverse_scale_factor_tensor_w,
                   inverse_scale_factor_tensor_d},
                  UpsampleMode::kNearestExact));
          return grad_input;
        };
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_scale_factors));

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<4>(std::move(op_builder),
                           {grad_output, inverse_scale_factors[0],
                            inverse_scale_factors[1], inverse_scale_factors[2]},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(input_shape),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(grad_input_buf), grad_input));
        return grad_input;
      });
}

// Bilinear interpolation estimates a new pixel value by taking a weighted
// average of the four nearest pixels (top-left, top-right, bottom-left,
// bottom-right) from the source image. Mathematically, for a target coordinate
// (x, y), we map it to a source coordinate (u, v). We then find the integer
// coordinates:
//
// - u_floor = floor(u), u_ceil = ceil(u)
// - v_floor = floor(v), v_ceil = ceil(v)
//
// We calculate weights based on how close (u, v) is to these integer points
// and sum up the 4 weighted corner values.
at::Tensor& AtenUpsampleBilinear2dOut(const at::Tensor& self,
                                      at::IntArrayRef upsample_shape,
                                      bool align_corners,
                                      std::optional<double> scale_h,
                                      std::optional<double> scale_w,
                                      at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleBilinear2dOut, param_keys,
      (self, upsample_shape, align_corners, scale_h, scale_w, out), {
        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        auto op_builder =
            [output_shape = CopyIntVector(out.sizes()), align_corners, scale_h,
             scale_w](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          TT_ASSIGN_OR_RETURN(auto output,
                              BuildUpsampleBilinearShlo(
                                  input, /*num_dimensions=*/2, output_shape,
                                  align_corners, {scale_h, scale_w}));
          return output;
        };

        TT_ASSIGN_OR_THROW(
            auto out_buf,
            (DispatchOp<1>(std::move(op_builder), {self},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(out.sizes()),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(out_buf), out));
        return out;
      });
}

at::Tensor& AtenUpsampleBilinear2dBackwardGradInput(
    const at::Tensor& grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size, bool align_corners,
    std::optional<double> scales_h, std::optional<double> scales_w,
    at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kUpsampleBilinear2dBackwardGradInput, param_keys,
      (grad_output, output_size, input_size, align_corners, scales_h, scales_w,
       grad_input),
      {
        TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<mlir::ElementType>(
                                                  grad_input.scalar_type()));
        auto op_builder =
            [grad_input_shape = CopyIntVector(input_size), align_corners,
             scales_h, scales_w](
                mlir::MlirOp grad_output) -> absl::StatusOr<mlir::MlirOp> {
          TT_ASSIGN_OR_RETURN(
              auto grad_input,
              BuildUpsampleBilinearBackwardShlo(
                  grad_output, /*num_dimensions=*/2, grad_input_shape,
                  align_corners, {scales_h, scales_w}));
          return grad_input;
        };

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<1>(std::move(op_builder), {grad_output},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(input_size),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(grad_input_buf), grad_input));
        return grad_input;
      });
}

at::Tensor& AtenUpsampleNearest1dOut(const at::Tensor& self,
                                     at::IntArrayRef upsample_shape,
                                     std::optional<double> scale,
                                     at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleNearest1dOut, param_keys,
      (self, upsample_shape, scale, out), {
        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        auto op_builder = [output_shape = CopyIntVector(out.sizes())](
                              FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [input, inverse_scale_factor_tensor] = inputs;
          TT_ASSIGN_OR_RETURN(auto output, BuildUpsampleNearestShlo(
                                               input, output_shape,
                                               {inverse_scale_factor_tensor},
                                               UpsampleMode::kNearest));
          return output;
        };
        auto preprocessed_inverse_scale_factors = StandardizeScaleFactors(
            self.sizes(), upsample_shape, {scale}, /*inverse=*/true);
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_inverse_scale_factors));

        TT_ASSIGN_OR_THROW(
            auto out_buf,
            (DispatchOp<2>(std::move(op_builder),
                           {self, inverse_scale_factors[0]},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(out.sizes()),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(out_buf), out));
        return out;
      });
}

at::Tensor& AtenUpsampleNearest2dOut(const at::Tensor& self,
                                     at::IntArrayRef upsample_shape,
                                     std::optional<double> scale_h,
                                     std::optional<double> scale_w,
                                     at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleNearest2dOut, param_keys,
      (self, upsample_shape, scale_h, scale_w, out), {
        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        auto op_builder = [output_shape = CopyIntVector(out.sizes())](
                              FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [input, inverse_scale_factor_tensor_h,
                 inverse_scale_factor_tensor_w] = inputs;
          TT_ASSIGN_OR_RETURN(auto output, BuildUpsampleNearestShlo(
                                               input, output_shape,
                                               {inverse_scale_factor_tensor_h,
                                                inverse_scale_factor_tensor_w},
                                               UpsampleMode::kNearest));
          return output;
        };

        auto preprocessed_inverse_scale_factors = StandardizeScaleFactors(
            self.sizes(), upsample_shape, {scale_h, scale_w},
            /*inverse=*/true);
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_inverse_scale_factors));

        TT_ASSIGN_OR_THROW(
            auto out_buf,
            (DispatchOp<3>(
                std::move(op_builder),
                {self, inverse_scale_factors[0], inverse_scale_factors[1]},
                {.out_dtype = element_type,
                 .out_dims = CopyIntVector(out.sizes()),
                 .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(out_buf), out));
        return out;
      });
}

at::Tensor& AtenUpsampleNearest3dOut(const at::Tensor& self,
                                     at::IntArrayRef upsample_shape,
                                     std::optional<double> scale_h,
                                     std::optional<double> scale_w,
                                     std::optional<double> scale_d,
                                     at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleNearest3dOut, param_keys,
      (self, upsample_shape, scale_h, scale_w, scale_d, out), {
        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        auto op_builder = [output_shape = CopyIntVector(out.sizes())](
                              FixedSizeSpan<mlir::MlirOp, 4> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [input, inverse_scale_factor_tensor_h,
                 inverse_scale_factor_tensor_w, inverse_scale_factor_tensor_d] =
              inputs;
          TT_ASSIGN_OR_RETURN(auto output, BuildUpsampleNearestShlo(
                                               input, output_shape,
                                               {inverse_scale_factor_tensor_h,
                                                inverse_scale_factor_tensor_w,
                                                inverse_scale_factor_tensor_d},
                                               UpsampleMode::kNearest));
          return output;
        };
        auto preprocessed_inverse_scale_factors = StandardizeScaleFactors(
            self.sizes(), upsample_shape, {scale_h, scale_w, scale_d},
            /*inverse=*/true);
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_inverse_scale_factors));
        TT_ASSIGN_OR_THROW(
            auto out_buf,
            (DispatchOp<4>(std::move(op_builder),
                           {self, inverse_scale_factors[0],
                            inverse_scale_factors[1], inverse_scale_factors[2]},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(out.sizes()),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(out_buf), out));

        return out;
      });
}

at::Tensor& AtenUpsampleNearestExact1dOut(const at::Tensor& self,
                                          at::IntArrayRef upsample_shape,
                                          std::optional<double> scale,
                                          at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleNearestExact1dOut, param_keys,
      (self, upsample_shape, scale, out), {
        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        auto op_builder = [output_shape = CopyIntVector(out.sizes())](
                              FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [input, inverse_scale_factor_tensor] = inputs;
          TT_ASSIGN_OR_RETURN(auto output, BuildUpsampleNearestShlo(
                                               input, output_shape,
                                               {inverse_scale_factor_tensor},
                                               UpsampleMode::kNearestExact));
          return output;
        };

        auto preprocessed_inverse_scale_factors = StandardizeScaleFactors(
            self.sizes(), upsample_shape, {scale}, /*inverse=*/true);
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_inverse_scale_factors));

        TT_ASSIGN_OR_THROW(
            auto out_buf,
            (DispatchOp<2>(std::move(op_builder),
                           {self, inverse_scale_factors[0]},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(out.sizes()),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(out_buf), out));
        return out;
      });
}

at::Tensor& AtenUpsampleNearestExact2dOut(const at::Tensor& self,
                                          at::IntArrayRef upsample_shape,
                                          std::optional<double> scale_h,
                                          std::optional<double> scale_w,
                                          at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleNearestExact2dOut, param_keys,
      (self, upsample_shape, scale_h, scale_w, out), {
        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        auto op_builder = [output_shape = CopyIntVector(out.sizes())](
                              FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [input, inverse_scale_factor_tensor_h,
                 inverse_scale_factor_tensor_w] = inputs;
          TT_ASSIGN_OR_RETURN(auto output, BuildUpsampleNearestShlo(
                                               input, output_shape,
                                               {inverse_scale_factor_tensor_h,
                                                inverse_scale_factor_tensor_w},
                                               UpsampleMode::kNearestExact));
          return output;
        };
        auto preprocessed_inverse_scale_factors = StandardizeScaleFactors(
            self.sizes(), upsample_shape, {scale_h, scale_w}, /*inverse=*/true);
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_inverse_scale_factors));
        TT_ASSIGN_OR_THROW(
            auto out_buf,
            (DispatchOp<3>(
                std::move(op_builder),
                {self, inverse_scale_factors[0], inverse_scale_factors[1]},
                {.out_dtype = element_type,
                 .out_dims = CopyIntVector(out.sizes()),
                 .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(out_buf), out));
        return out;
      });
}

at::Tensor& AtenUpsampleNearestExact3dOut(const at::Tensor& self,
                                          at::IntArrayRef upsample_shape,
                                          std::optional<double> scale_h,
                                          std::optional<double> scale_w,
                                          std::optional<double> scale_d,
                                          at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleNearestExact3dOut, param_keys,
      (self, upsample_shape, scale_h, scale_w, scale_d, out), {
        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        auto op_builder = [output_shape = CopyIntVector(out.sizes())](
                              FixedSizeSpan<mlir::MlirOp, 4> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [input, inverse_scale_factor_tensor_h,
                 inverse_scale_factor_tensor_w, inverse_scale_factor_tensor_d] =
              inputs;
          TT_ASSIGN_OR_RETURN(auto output, BuildUpsampleNearestShlo(
                                               input, output_shape,
                                               {inverse_scale_factor_tensor_h,
                                                inverse_scale_factor_tensor_w,
                                                inverse_scale_factor_tensor_d},
                                               UpsampleMode::kNearestExact));
          return output;
        };
        auto preprocessed_inverse_scale_factors = StandardizeScaleFactors(
            self.sizes(), upsample_shape, {scale_h, scale_w, scale_d},
            /*inverse=*/true);
        TT_ASSIGN_OR_THROW(
            auto inverse_scale_factors,
            ConstructScaleFactorArray(preprocessed_inverse_scale_factors));
        TT_ASSIGN_OR_THROW(
            auto out_buf,
            (DispatchOp<4>(std::move(op_builder),
                           {self, inverse_scale_factors[0],
                            inverse_scale_factors[1], inverse_scale_factors[2]},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(out.sizes()),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(out_buf), out));

        return out;
      });
}

}  // namespace torch_tpu
