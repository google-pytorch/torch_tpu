/*
 * Copyright 2026 Google LLC
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

#include "torch_tpu/csrc/ops/upsample/upsample_bicubic2d_aten_kernels.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dimension_types.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/common/utils.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/csrc/ops/binary.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/nullary_aten_kernels.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/csrc/ops/resize/resize_aten_kernels.h"

namespace torch_tpu {
namespace {

using ScaleVector = absl::InlinedVector<std::optional<double>, 2>;

enum class TapIndex {
  kTap0 = 0,
  kTap1 = 1,
  kTap2 = 2,
  kTap3 = 3,
};

struct BicubicGatherConfig {
  mlir::stablehlo::GatherDimensionNumbersAttr gather_dimension_numbers;
  Dimensions slice_sizes;
};

struct BicubicScatterConfig {
  mlir::stablehlo::ScatterDimensionNumbersAttr scatter_dimension_numbers;
};

struct BicubicDimInfo {
  absl::InlinedVector<mlir::MlirOp, 4> indices;
  absl::InlinedVector<mlir::MlirOp, 4> weights;
};

struct BicubicCornerIndicesAndWeight {
  absl::InlinedVector<mlir::MlirOp, 2> indices;
  mlir::MlirOp weight;
};

BicubicScatterConfig GetBicubicScatterConfig(mlir::MlirOp grad_output) {
  const int64_t spatial_dim_size = 2;
  const int64_t num_output_dimensions =
      GetTensorTypeOrDie(grad_output).getShape().size();
  // Map scatter spatial dims to the last dimensions of the operand.
  Dimensions scatter_dims_to_operand_dims;
  scatter_dims_to_operand_dims.reserve(spatial_dim_size);
  for (int i = 0; i < spatial_dim_size; ++i) {
    scatter_dims_to_operand_dims.push_back(i + num_output_dimensions -
                                           spatial_dim_size);
  }
  // Identify leading batch/channel axes to preserve during scatter.
  const int64_t batching_dims_size = num_output_dimensions - spatial_dim_size;
  Dimensions batching_dims;
  batching_dims.reserve(batching_dims_size);
  for (int i = 0; i < num_output_dimensions - spatial_dim_size; ++i) {
    batching_dims.push_back(i);
  }
  // Specify window dimensions inserted into the operand shape during reduction.
  Dimensions inserted_window_dims;
  inserted_window_dims.reserve(spatial_dim_size);
  for (int i = 0; i < spatial_dim_size; ++i) {
    inserted_window_dims.push_back(i + num_output_dimensions -
                                   spatial_dim_size);
  }

  // Configure StableHLO scatter dimension numbers for 2D spatial backprop.
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

BicubicGatherConfig GetBicubicGatherConfig(mlir::MlirOp input) {
  const Dimensions input_shape =
      CopyIntVector(GetTensorTypeOrDie(input).getShape());
  const int64_t spatial_dim_size = 2;
  const int64_t offset_dim_size = input_shape.size() - spatial_dim_size;

  // Set spatial slice sizes to 1 while preserving batch/channel axes.
  Dimensions slice_sizes = input_shape;
  for (int i = 0; i < spatial_dim_size; ++i) {
    slice_sizes[offset_dim_size + i] = 1;
  }

  // Map spatial axes to be collapsed in slices and indexed by start indices.
  Dimensions collapsed_slice_dims;
  collapsed_slice_dims.reserve(spatial_dim_size);
  Dimensions start_index_map;
  start_index_map.reserve(spatial_dim_size);
  for (int i = 0; i < spatial_dim_size; ++i) {
    collapsed_slice_dims.push_back(offset_dim_size + i);
    start_index_map.push_back(offset_dim_size + i);
  }

  // Specify leading batch and channel axes as preserved offset dimensions.
  Dimensions offset_dims;
  offset_dims.reserve(offset_dim_size);
  for (int i = 0; i < offset_dim_size; ++i) {
    offset_dims.push_back(i);
  }

  // Configure StableHLO gather dimension numbers for 2D neighborhood indexing.
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

absl::StatusOr<mlir::MlirOp> ComputeBicubicSourceIndex(
    mlir::MlirBuilder& builder, int64_t out_size, int64_t in_size,
    std::optional<double> scale_opt, bool align_corners, mlir::Type calc_type) {
  auto iota_type = mlir::RankedTensorType::get({out_size}, calc_type);
  auto iota = mlir::stablehlo::Iota(builder, iota_type, 0);
  mlir::MlirOp src_idx;

  if (align_corners) {
    // Map corner pixels proportionally across input/output grid indices.
    double stride = (out_size > 1) ? static_cast<double>(in_size - 1) /
                                         static_cast<double>(out_size - 1)
                                   : 0.0;
    auto stride_const = MakeConstant(builder, stride, calc_type, {out_size});
    TT_ASSIGN_OR_RETURN(src_idx, BuildMulShlo(iota, stride_const));
  } else {
    // Map pixel centers proportionally using scale ratio (OpenCV style).
    double scale = static_cast<double>(in_size) / static_cast<double>(out_size);
    if (scale_opt.has_value()) {
      scale = 1.0 / (*scale_opt);
    }
    auto half_const = MakeConstant(builder, 0.5, calc_type, {out_size});
    TT_ASSIGN_OR_RETURN(auto iota_plus_half, BuildAddShlo(iota, half_const));

    auto scale_const = MakeConstant(builder, scale, calc_type, {out_size});
    TT_ASSIGN_OR_RETURN(auto scaled, BuildMulShlo(iota_plus_half, scale_const));
    TT_ASSIGN_OR_RETURN(src_idx, BuildSubShlo(scaled, half_const));
  }

  return src_idx;
}

absl::StatusOr<mlir::MlirOp> ComputeCubicWeightNear(mlir::MlirBuilder& builder,
                                                    mlir::MlirOp d,
                                                    mlir::Type calc_type,
                                                    int64_t out_size) {
  // W(d) = 1.25 * d^3 - 2.25 * d^2 + 1
  auto c1_25 = MakeConstant(builder, 1.25, calc_type, {out_size});
  auto c2_25 = MakeConstant(builder, 2.25, calc_type, {out_size});
  auto c1_0 = MakeConstant(builder, 1.0, calc_type, {out_size});

  TT_ASSIGN_OR_RETURN(auto t1, BuildMulShlo(d, c1_25));
  TT_ASSIGN_OR_RETURN(auto t2, BuildSubShlo(t1, c2_25));
  TT_ASSIGN_OR_RETURN(auto t3, BuildMulShlo(d, t2));
  TT_ASSIGN_OR_RETURN(auto t4, BuildMulShlo(d, t3));
  return BuildAddShlo(t4, c1_0);
}

absl::StatusOr<mlir::MlirOp> ComputeCubicWeightFar(mlir::MlirBuilder& builder,
                                                   mlir::MlirOp d,
                                                   mlir::Type calc_type,
                                                   int64_t out_size) {
  // W(d) = -0.75 * d^3 + 3.75 * d^2 - 6 * d + 3
  auto c_neg0_75 = MakeConstant(builder, -0.75, calc_type, {out_size});
  auto c3_75 = MakeConstant(builder, 3.75, calc_type, {out_size});
  auto c6_0 = MakeConstant(builder, 6.0, calc_type, {out_size});
  auto c3_0 = MakeConstant(builder, 3.0, calc_type, {out_size});

  TT_ASSIGN_OR_RETURN(auto t1, BuildMulShlo(d, c_neg0_75));
  TT_ASSIGN_OR_RETURN(auto t2, BuildAddShlo(t1, c3_75));
  TT_ASSIGN_OR_RETURN(auto t3, BuildMulShlo(d, t2));
  TT_ASSIGN_OR_RETURN(auto t4, BuildSubShlo(t3, c6_0));
  TT_ASSIGN_OR_RETURN(auto t5, BuildMulShlo(d, t4));
  return BuildAddShlo(t5, c3_0);
}

absl::StatusOr<mlir::MlirOp> ComputeTapDistance(TapIndex tap,
                                                mlir::MlirOp lambda,
                                                mlir::MlirOp one_calc,
                                                mlir::MlirOp two_calc) {
  switch (tap) {
    case TapIndex::kTap0:
      return BuildAddShlo(lambda, one_calc);
    case TapIndex::kTap1:
      return lambda;
    case TapIndex::kTap2:
      return BuildSubShlo(one_calc, lambda);
    case TapIndex::kTap3:
      return BuildSubShlo(two_calc, lambda);
  }
}

absl::StatusOr<mlir::MlirOp> ComputeTapOffset(TapIndex tap,
                                              mlir::MlirOp idx_floor_calc,
                                              mlir::MlirOp one_calc,
                                              mlir::MlirOp two_calc) {
  switch (tap) {
    case TapIndex::kTap0:
      return BuildSubShlo(idx_floor_calc, one_calc);
    case TapIndex::kTap1:
      return idx_floor_calc;
    case TapIndex::kTap2:
      return BuildAddShlo(idx_floor_calc, one_calc);
    case TapIndex::kTap3:
      return BuildAddShlo(idx_floor_calc, two_calc);
  }
}

absl::StatusOr<mlir::MlirOp> ComputeTapWeight(mlir::MlirBuilder& builder,
                                              TapIndex tap,
                                              mlir::MlirOp distance,
                                              mlir::Type calc_type,
                                              int64_t out_size) {
  switch (tap) {
    case TapIndex::kTap0:
    case TapIndex::kTap3:
      return ComputeCubicWeightFar(builder, distance, calc_type, out_size);
    case TapIndex::kTap1:
    case TapIndex::kTap2:
      return ComputeCubicWeightNear(builder, distance, calc_type, out_size);
  }
}

absl::StatusOr<BicubicDimInfo> ComputeBicubicDimInfo(
    mlir::MlirBuilder& builder, mlir::MlirOp src_idx, int64_t out_size,
    int64_t in_size, mlir::Type calc_type,
    mlir::RankedTensorType broadcast_type,
    mlir::RankedTensorType broadcast_type_calc, int spatial_dim_index) {
  // Extract integer floor index and compute fractional distance lambda.
  auto idx_floor_calc = mlir::stablehlo::Floor(src_idx);
  auto one_calc = MakeConstant(builder, 1.0, calc_type, {out_size});
  auto two_calc = MakeConstant(builder, 2.0, calc_type, {out_size});

  TT_ASSIGN_OR_RETURN(auto lambda, BuildSubShlo(src_idx, idx_floor_calc));

  // Prepare boundary limits for clamping indices to valid input grid range.
  auto i32_type = builder.getOpBuilder().getI32Type();
  auto zero_i32 = MakeConstant(builder, 0, i32_type, {out_size});
  auto max_idx_i32 =
      MakeConstant(builder, std::max(0L, in_size - 1), i32_type, {out_size});

  const TapIndex taps[4] = {TapIndex::kTap0, TapIndex::kTap1, TapIndex::kTap2,
                            TapIndex::kTap3};
  BicubicDimInfo info;
  // Compute tap offsets, weights, and clamped coordinates for 4 taps.
  for (int i = 0; i < 4; ++i) {
    TapIndex tap = taps[i];
    TT_ASSIGN_OR_RETURN(auto dist,
                        ComputeTapDistance(tap, lambda, one_calc, two_calc));
    TT_ASSIGN_OR_RETURN(
        auto w, ComputeTapWeight(builder, tap, dist, calc_type, out_size));
    TT_ASSIGN_OR_RETURN(auto raw_idx, ComputeTapOffset(tap, idx_floor_calc,
                                                       one_calc, two_calc));

    // Clamp coordinates to [0, in_size - 1] to prevent out-of-bounds access.
    auto idx_i32_unclamped =
        mlir::stablehlo::ConvertElementType(raw_idx, i32_type);
    auto idx_clamped =
        mlir::stablehlo::Clamp(idx_i32_unclamped, zero_i32, max_idx_i32);
    info.indices.push_back(mlir::stablehlo::BroadcastInDim(
        broadcast_type, idx_clamped, {spatial_dim_index}));
    info.weights.push_back(mlir::stablehlo::BroadcastInDim(
        broadcast_type_calc, w, {spatial_dim_index}));
  }
  return info;
}

absl::StatusOr<BicubicCornerIndicesAndWeight>
ComputeBicubicCornerIndicesAndWeight(
    int corner, absl::Span<const BicubicDimInfo> dim_infos) {
  int h_tap_idx = corner / 4;
  int w_tap_idx = corner % 4;
  absl::InlinedVector<mlir::MlirOp, 2> current_indices = {
      dim_infos[0].indices[h_tap_idx], dim_infos[1].indices[w_tap_idx]};
  TT_ASSIGN_OR_RETURN(mlir::MlirOp current_weight,
                      BuildMulShlo(dim_infos[0].weights[h_tap_idx],
                                   dim_infos[1].weights[w_tap_idx]));
  return BicubicCornerIndicesAndWeight{std::move(current_indices),
                                       current_weight};
}

absl::StatusOr<mlir::MlirOp> GatherAndWeightSingleCorner(
    mlir::MlirBuilder& builder, mlir::MlirOp input,
    const Dimensions& output_shape, int corner,
    absl::Span<const BicubicDimInfo> dim_infos,
    const BicubicGatherConfig& gather_config, int64_t offset_dim_size,
    mlir::RankedTensorType broadcast_type, mlir::Type calc_type) {
  // Combine 1D height/width indices and weights for the specified corner.
  TT_ASSIGN_OR_RETURN(auto indices_and_weight,
                      ComputeBicubicCornerIndicesAndWeight(corner, dim_infos));
  auto current_indices = indices_and_weight.indices;
  auto current_weight = indices_and_weight.weight;

  // Concatenate spatial indices into a 2D coordinate vector along last axis.
  auto index_tensor = mlir::stablehlo::Concatenate(
      builder, current_indices, output_shape.size() - offset_dim_size);

  // Gather pixel values from the input tensor at the corner spatial indices.
  auto gathered = mlir::stablehlo::Gather(
      input, index_tensor, gather_config.gather_dimension_numbers,
      AsArrayRef<int64_t>(gather_config.slice_sizes),
      /*indices_are_sorted=*/false);

  // Reshape and broadcast corner weights to match gathered pixel dimensions.
  auto weight_type_sq = mlir::RankedTensorType::get(
      broadcast_type.getShape().drop_back(), calc_type);
  auto weight_sq = mlir::stablehlo::Reshape(weight_type_sq, current_weight);

  auto gathered_calc = mlir::stablehlo::ConvertElementType(gathered, calc_type);

  Dimensions result_broadcast_dims = {offset_dim_size, offset_dim_size + 1};
  auto weight_bcast = mlir::stablehlo::BroadcastInDim(
      gathered_calc.getType(), weight_sq,
      AsArrayRef<int64_t>(result_broadcast_dims));

  // Multiply gathered pixel values by the combined 2D bicubic corner weight.
  return BuildMulShlo(gathered_calc, weight_bcast);
}

absl::StatusOr<mlir::MlirOp> AccumulateBicubicInterpolation(
    mlir::MlirBuilder& builder, mlir::MlirOp input,
    const Dimensions& output_shape, absl::Span<const BicubicDimInfo> dim_infos,
    const BicubicGatherConfig& gather_config, int64_t offset_dim_size,
    mlir::RankedTensorType broadcast_type, mlir::Type calc_type) {
  mlir::MlirOp accumulated_result;
  // Loop across all 16 4x4 neighborhood corners and sum weighted values.
  for (int corner = 0; corner < 16; ++corner) {
    TT_ASSIGN_OR_RETURN(
        auto weighted_value,
        GatherAndWeightSingleCorner(builder, input, output_shape, corner,
                                    dim_infos, gather_config, offset_dim_size,
                                    broadcast_type, calc_type));
    if (corner == 0) {
      accumulated_result = weighted_value;
    } else {
      TT_ASSIGN_OR_RETURN(accumulated_result,
                          BuildAddShlo(accumulated_result, weighted_value));
    }
  }
  return accumulated_result;
}

absl::StatusOr<mlir::MlirOp> BuildUpsampleBicubic2dShlo(
    mlir::MlirOp input, const Dimensions& output_shape, bool align_corners,
    const ScaleVector& scales_opt, mlir::Type calc_type) {
  auto& builder = input.getBuilder();
  Dimensions input_shape = CopyIntVector(GetTensorTypeOrDie(input).getShape());
  const int64_t spatial_dim_size = 2;
  const int64_t offset_dim_size = input_shape.size() - spatial_dim_size;

  BicubicGatherConfig gather_config = GetBicubicGatherConfig(input);
  auto element_type = GetTensorTypeOrDie(input).getElementType();

  // Prepare broadcast and calc types for spatial index/weight vectors.
  absl::InlinedVector<BicubicDimInfo, 2> dim_infos;
  dim_infos.reserve(spatial_dim_size);

  Dimensions broadcast_shape_array(spatial_dim_size + 1);
  broadcast_shape_array[spatial_dim_size] = 1;
  for (int i = 0; i < spatial_dim_size; ++i) {
    broadcast_shape_array[i] = output_shape[offset_dim_size + i];
  }
  auto broadcast_type = mlir::RankedTensorType::get(
      broadcast_shape_array, builder.getOpBuilder().getI32Type());
  auto broadcast_type_calc =
      mlir::RankedTensorType::get(broadcast_shape_array, calc_type);

  // 1. Compute source coordinates and bicubic polynomial weights (Horner
  // evaluation) for spatial dimensions.
  for (int i = 0; i < spatial_dim_size; ++i) {
    int64_t out_size = output_shape[offset_dim_size + i];
    int64_t in_size = input_shape[offset_dim_size + i];
    auto scale_opt = scales_opt[i];

    TT_ASSIGN_OR_RETURN(auto src_idx, ComputeBicubicSourceIndex(
                                          builder, out_size, in_size, scale_opt,
                                          align_corners, calc_type));
    TT_ASSIGN_OR_RETURN(
        auto dim_info,
        ComputeBicubicDimInfo(builder, src_idx, out_size, in_size, calc_type,
                              broadcast_type, broadcast_type_calc, i));
    dim_infos.push_back(dim_info);
  }

  // 2. Gather values across all 16 neighborhood corners and accumulate weighted
  // contributions.
  TT_ASSIGN_OR_RETURN(
      auto accumulated_result,
      AccumulateBicubicInterpolation(builder, input, output_shape, dim_infos,
                                     gather_config, offset_dim_size,
                                     broadcast_type, calc_type));

  // 3. Convert accumulated calculation precision back to input element type.
  return mlir::stablehlo::ConvertElementType(accumulated_result, element_type);
}

absl::StatusOr<mlir::MlirOp> ScatterBicubic2dBackwardForSingleCorner(
    mlir::MlirBuilder& builder, mlir::MlirOp grad_input, int offset_dim_size,
    int corner, absl::Span<const BicubicDimInfo> dim_infos,
    const BicubicScatterConfig& scatter_config,
    absl::FunctionRef<void(mlir::RegionBuilder&)> scatter_body,
    const Dimensions& grad_output_shape, mlir::MlirOp grad_output_calc,
    mlir::Type calc_type, mlir::Type element_type) {
  const int spatial_dim_size = 2;
  // Compute 1D tap indices and combined 2D bicubic weight for this corner.
  TT_ASSIGN_OR_RETURN(auto indices_and_weight,
                      ComputeBicubicCornerIndicesAndWeight(corner, dim_infos));
  auto current_indices_1d = indices_and_weight.indices;
  auto current_weight_1d = indices_and_weight.weight;

  // Concatenate spatial indices into a 2D coordinate vector for scattering.
  auto corner_indices_spatial = mlir::stablehlo::Concatenate(
      builder, current_indices_1d, spatial_dim_size);

  // Broadcast corner spatial indices across batch/channel axes of grad_output.
  Dimensions scatter_indices_shape_dims = grad_output_shape;
  scatter_indices_shape_dims.push_back(spatial_dim_size);
  auto scatter_indices_type = mlir::RankedTensorType::get(
      scatter_indices_shape_dims, builder.getOpBuilder().getI32Type());

  Dimensions indices_bcast_dims = {
      offset_dim_size, offset_dim_size + 1,
      static_cast<int64_t>(grad_output_shape.size())};
  auto scatter_indices = mlir::stablehlo::BroadcastInDim(
      scatter_indices_type, corner_indices_spatial,
      AsArrayRef<int64_t>(indices_bcast_dims));

  // Reshape and broadcast corner weight to match grad_output shape.
  auto weight_bcast_type =
      mlir::RankedTensorType::get(grad_output_shape, calc_type);
  Dimensions weight_bcast_dims = {offset_dim_size, offset_dim_size + 1};
  Dimensions squeezed_shape = {grad_output_shape[offset_dim_size],
                               grad_output_shape[offset_dim_size + 1]};
  auto corner_weight_squeezed_type =
      mlir::RankedTensorType::get(squeezed_shape, calc_type);
  auto corner_weight_squeezed =
      mlir::stablehlo::Reshape(corner_weight_squeezed_type, current_weight_1d);

  auto weight_bcast =
      mlir::stablehlo::BroadcastInDim(weight_bcast_type, corner_weight_squeezed,
                                      AsArrayRef<int64_t>(weight_bcast_dims));

  // Scale incoming gradient output values by the corner polynomial weight.
  TT_ASSIGN_OR_RETURN(auto update_calc,
                      BuildMulShlo(grad_output_calc, weight_bcast));
  auto update = mlir::stablehlo::ConvertElementType(update_calc, element_type);

  // Scatter weighted gradient updates into grad_input using additive reduction.
  return mlir::stablehlo::Scatter(grad_input, scatter_indices, update,
                                  scatter_body,
                                  scatter_config.scatter_dimension_numbers,
                                  /*indices_are_sorted=*/false)[0];
}

absl::StatusOr<mlir::MlirOp> BuildUpsampleBicubic2dBackwardShlo(
    mlir::MlirOp grad_output, const Dimensions& grad_input_shape,
    bool align_corners, const ScaleVector& scales_opt, mlir::Type calc_type) {
  mlir::MlirBuilder& builder = grad_output.getBuilder();
  Dimensions grad_output_shape =
      CopyIntVector(GetTensorTypeOrDie(grad_output).getShape());
  const int64_t spatial_dim_size = 2;
  const int64_t offset_dim_size = grad_output_shape.size() - spatial_dim_size;

  auto element_type = GetTensorTypeOrDie(grad_output).getElementType();

  // Prepare broadcast and calc types for spatial coordinate vectors.
  absl::InlinedVector<BicubicDimInfo, 2> dim_infos;
  dim_infos.reserve(spatial_dim_size);

  Dimensions broadcast_shape_array(spatial_dim_size + 1);
  broadcast_shape_array[spatial_dim_size] = 1;
  for (int i = 0; i < spatial_dim_size; ++i) {
    broadcast_shape_array[i] = grad_output_shape[offset_dim_size + i];
  }
  auto broadcast_type = mlir::RankedTensorType::get(
      broadcast_shape_array, builder.getOpBuilder().getI32Type());
  auto broadcast_type_calc =
      mlir::RankedTensorType::get(broadcast_shape_array, calc_type);

  // 1. Compute source coordinates and bicubic interpolation weights for
  // backward projection.
  for (int i = 0; i < spatial_dim_size; ++i) {
    int64_t out_size = grad_output_shape[offset_dim_size + i];
    int64_t in_size = grad_input_shape[offset_dim_size + i];
    auto scale_opt = scales_opt[i];

    TT_ASSIGN_OR_RETURN(auto src_idx, ComputeBicubicSourceIndex(
                                          builder, out_size, in_size, scale_opt,
                                          align_corners, calc_type));
    TT_ASSIGN_OR_RETURN(
        auto dim_info,
        ComputeBicubicDimInfo(builder, src_idx, out_size, in_size, calc_type,
                              broadcast_type, broadcast_type_calc, i));
    dim_infos.push_back(dim_info);
  }

  // 2. Initialize grad_input buffer to zero and configure StableHLO scatter
  // reduction body.
  mlir::MlirOp grad_input = MakeConstant(builder, 0.0, element_type,
                                         AsArrayRef<int64_t>(grad_input_shape));
  mlir::MlirOp grad_output_calc =
      mlir::stablehlo::ConvertElementType(grad_output, calc_type);

  auto scatter_config = GetBicubicScatterConfig(grad_output);
  auto scatter_body = [element_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        element_type, rb.getRegion(), rb.getOpBuilder());
  };

  // 3. Scatter weighted gradient contributions across all 16 neighborhood
  // corners back into grad_input.
  for (int corner = 0; corner < 16; ++corner) {
    TT_ASSIGN_OR_RETURN(
        grad_input, ScatterBicubic2dBackwardForSingleCorner(
                        builder, grad_input, offset_dim_size, corner, dim_infos,
                        scatter_config, scatter_body, grad_output_shape,
                        grad_output_calc, calc_type, element_type));
  }

  return grad_input;
}

void CheckUpsampleDtypes(const at::Tensor& tensor) {
  TT_CHECK_THROW(tensor.scalar_type() != at::kBool &&
                     tensor.scalar_type() != at::kInt &&
                     tensor.scalar_type() != at::kLong,
                 error::kPythonNotImplementedError)
      << "not implemented for " << ToString(tensor.scalar_type());
}

}  // namespace

at::Tensor& AtenUpsampleBicubic2dOut(const at::Tensor& self,
                                     at::IntArrayRef output_size,
                                     bool align_corners,
                                     std::optional<double> scale_h,
                                     std::optional<double> scale_w,
                                     at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleBicubic2dOut, param_keys,
      (self, output_size, align_corners, scale_h, scale_w, out), {
        Dimensions expected_output_shape = {self.size(0), self.size(1),
                                            output_size[0], output_size[1]};
        TT_THROW_IF_ERROR(
            ResizeTensorIfShapeDiffers(out, expected_output_shape));

        CheckUpsampleDtypes(self);

        TT_CHECK_THROW(self.scalar_type() == out.scalar_type(),
                       error::kInvalidArgument)
            << "expected out dtype " << ToString(self.scalar_type()) << ", got "
            << ToString(out.scalar_type());

        if (self.numel() == 0 || out.numel() == 0) {
          if (out.numel() > 0) {
            out.copy_(AtenEfficientZeroTensor(expected_output_shape,
                                              out.scalar_type(), std::nullopt,
                                              out.device(), std::nullopt));
          }
          return out;
        }

        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        const at::ScalarType real_dtype =
            c10::toRealValueType(self.scalar_type());

        // Perform lowering using higher precision (f32/f64) to maintain
        // numerical accuracy.
        auto op_builder =
            [out_shape = CopyIntVector(out.sizes()), align_corners, scale_h,
             scale_w,
             real_dtype](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          mlir::Type calc_type =
              (real_dtype == at::kDouble)
                  ? input.getBuilder().getOpBuilder().getF64Type()
                  : input.getBuilder().getOpBuilder().getF32Type();
          return BuildUpsampleBicubic2dShlo(input, out_shape, align_corners,
                                            {scale_h, scale_w}, calc_type);
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

at::Tensor& AtenUpsampleBicubic2dBackwardGradInput(
    const at::Tensor& grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size, bool align_corners,
    std::optional<double> scales_h, std::optional<double> scales_w,
    at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kUpsampleBicubic2dBackwardGradInput, param_keys,
      (grad_output, IgnoreInCacheKey(output_size, "Unused"), input_size,
       align_corners, scales_h, scales_w, grad_input),
      {
        Dimensions expected_grad_input_shape = CopyIntVector(input_size);
        TT_THROW_IF_ERROR(
            ResizeTensorIfShapeDiffers(grad_input, expected_grad_input_shape));

        CheckUpsampleDtypes(grad_output);

        TT_CHECK_THROW(grad_output.scalar_type() == grad_input.scalar_type(),
                       error::kInvalidArgument)
            << "expected grad_input dtype "
            << ToString(grad_output.scalar_type()) << ", got "
            << ToString(grad_input.scalar_type());

        if (grad_output.numel() == 0 || grad_input.numel() == 0) {
          if (grad_input.numel() > 0) {
            grad_input.copy_(AtenEfficientZeroTensor(
                expected_grad_input_shape, grad_input.scalar_type(),
                std::nullopt, grad_input.device(), std::nullopt));
          }
          return grad_input;
        }

        TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<mlir::ElementType>(
                                                  grad_input.scalar_type()));
        const at::ScalarType real_dtype =
            c10::toRealValueType(grad_input.scalar_type());

        auto op_builder =
            [grad_in_shape = CopyIntVector(grad_input.sizes()), align_corners,
             scales_h, scales_w, real_dtype](
                mlir::MlirOp grad_out) -> absl::StatusOr<mlir::MlirOp> {
          mlir::Type calc_type =
              (real_dtype == at::kDouble)
                  ? grad_out.getBuilder().getOpBuilder().getF64Type()
                  : grad_out.getBuilder().getOpBuilder().getF32Type();
          return BuildUpsampleBicubic2dBackwardShlo(
              grad_out, grad_in_shape, align_corners, {scales_h, scales_w},
              calc_type);
        };

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<1>(std::move(op_builder), {grad_output},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(grad_input.sizes()),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(grad_input_buf), grad_input));
        return grad_input;
      });
}

}  // namespace torch_tpu
