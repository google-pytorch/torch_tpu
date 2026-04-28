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

#include "torch_tpu/ops/grid_sampler/grid_sampler_aten_kernels.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/binary.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

struct GridSamplerGatherConfig {
  mlir::stablehlo::GatherDimensionNumbersAttr gather_dimension_numbers;
  Dimensions slice_sizes;
};

GridSamplerGatherConfig GetGridSamplerGatherConfig(mlir::MlirOp input,
                                                   int64_t spatial_dim_size) {
  const Dimensions input_shape =
      CopyIntVector(GetTensorTypeOrDie(input).getShape());
  const int64_t offset_dim_size = input_shape.size() - spatial_dim_size;

  auto slice_sizes = input_shape;
  slice_sizes[0] = 1;  // Batch dimension
  for (int i = 0; i < spatial_dim_size; ++i) {
    slice_sizes[offset_dim_size + i] = 1;
  }
  // N appears in the input and in the index tensor.
  Dimensions batch_dims;
  batch_dims.push_back(0);
  Dimensions collapsed_slice_dims;
  Dimensions start_index_map;
  for (int i = 0; i < spatial_dim_size; ++i) {
    collapsed_slice_dims.push_back(offset_dim_size + i);
    start_index_map.push_back(offset_dim_size + spatial_dim_size - 1 - i);
  }
  // Only the channel dimension is offset in the input.
  Dimensions offset_dims;
  offset_dims.push_back(1);

  mlir::stablehlo::GatherDimensionNumbersAttr gather_dimension_numbers =
      mlir::stablehlo::GatherDimensionNumbersAttr::get(
          &input.getContext(),
          /*offset_dims=*/AsArrayRef<int64_t>(offset_dims),
          /*collapsed_slice_dims=*/AsArrayRef<int64_t>(collapsed_slice_dims),
          /*operand_batching_dims=*/AsArrayRef<int64_t>(batch_dims),
          /*start_indices_batching_dims=*/AsArrayRef<int64_t>(batch_dims),
          /*start_index_map=*/AsArrayRef<int64_t>(start_index_map),
          /*index_vector_dim=*/spatial_dim_size + 1);

  return {gather_dimension_numbers, slice_sizes};
}

struct DimInfo {
  mlir::MlirOp idx_floor;
  mlir::MlirOp idx_ceil;
  mlir::MlirOp weight_floor;
  mlir::MlirOp weight_ceil;
  mlir::MlirOp idx_floor_unclamped;
  mlir::MlirOp idx_ceil_unclamped;
};

enum class PaddingMode { kZeros = 0, kBorder = 1, kReflection = 2 };

absl::StatusOr<mlir::MlirOp> ComputeSourceCoord(mlir::MlirOp grid_coord,
                                                int64_t in_size,
                                                bool align_corners,
                                                PaddingMode padding_mode,
                                                mlir::Type calc_type) {
  const auto half = MakeConstantLike(grid_coord, 0.5, calc_type);
  const auto one = MakeConstantLike(grid_coord, 1.0, calc_type);
  TT_ASSIGN_OR_RETURN(auto grid_coord_plus_1, BuildAddShlo(grid_coord, one));
  TT_ASSIGN_OR_RETURN(auto grid_coord_plus_1_div_2,
                      BuildMulShlo(grid_coord_plus_1, half));

  mlir::MlirOp src_coord;
  if (align_corners) {
    auto in_size_minus_1 = MakeConstantLike(
        grid_coord, static_cast<double>(in_size - 1), calc_type);
    TT_ASSIGN_OR_RETURN(src_coord,
                        BuildMulShlo(grid_coord_plus_1_div_2, in_size_minus_1));
  } else {
    auto in_size_val =
        MakeConstantLike(grid_coord, static_cast<double>(in_size), calc_type);
    TT_ASSIGN_OR_RETURN(auto scaled,
                        BuildMulShlo(grid_coord_plus_1_div_2, in_size_val));
    TT_ASSIGN_OR_RETURN(src_coord, BuildSubShlo(scaled, half));
  }

  if (padding_mode == PaddingMode::kReflection) {
    if (align_corners) {
      if (in_size > 1) {
        auto double_range = MakeConstantLike(
            grid_coord, static_cast<double>(2 * (in_size - 1)), calc_type);
        auto abs_src_coord = mlir::stablehlo::Abs(src_coord);
        TT_ASSIGN_OR_RETURN(auto mod_src_coord,
                            BuildFmodTensorShlo(abs_src_coord, double_range));
        TT_ASSIGN_OR_RETURN(auto reflected_src_coord,
                            BuildSubShlo(double_range, mod_src_coord));
        TT_ASSIGN_OR_RETURN(
            src_coord, BuildMinimumShlo(mod_src_coord, reflected_src_coord));
      } else {
        src_coord = MakeConstantLike(grid_coord, 0.0, calc_type);
      }
    } else {
      // For align_corners=false, the valid range is [-0.5, in_size - 0.5]
      // reflected about the boundaries -0.5 and in_size - 0.5
      // This is equivalent to reflecting x + 0.5 about 0 and in_size
      const auto half_calc = MakeConstantLike(grid_coord, 0.5, calc_type);
      TT_ASSIGN_OR_RETURN(auto src_coord_shifted,
                          BuildAddShlo(src_coord, half_calc));
      const auto double_range = MakeConstantLike(
          grid_coord, static_cast<double>(2 * in_size), calc_type);
      const auto abs_src_coord = mlir::stablehlo::Abs(src_coord_shifted);
      TT_ASSIGN_OR_RETURN(auto mod_src_coord,
                          BuildFmodTensorShlo(abs_src_coord, double_range));
      TT_ASSIGN_OR_RETURN(auto reflected_src_coord,
                          BuildSubShlo(double_range, mod_src_coord));
      TT_ASSIGN_OR_RETURN(auto min_src_coord,
                          BuildMinimumShlo(mod_src_coord, reflected_src_coord));
      TT_ASSIGN_OR_RETURN(src_coord, BuildSubShlo(min_src_coord, half_calc));
    }
  }

  return src_coord;
}

absl::StatusOr<DimInfo> ComputeDimInfo(mlir::MlirBuilder& builder,
                                       mlir::MlirOp src_idx, int64_t in_size,
                                       mlir::Type calc_type) {
  const auto idx_floor_calc = mlir::stablehlo::Floor(src_idx);
  const auto one_calc = MakeConstantLike(src_idx, 1.0, calc_type);
  TT_ASSIGN_OR_RETURN(const auto idx_ceil_calc,
                      BuildAddShlo(idx_floor_calc, one_calc));

  TT_ASSIGN_OR_RETURN(const auto d_floor,
                      BuildSubShlo(src_idx, idx_floor_calc));
  TT_ASSIGN_OR_RETURN(const auto d_ceil, BuildSubShlo(idx_ceil_calc, src_idx));

  auto idx_floor_i32_unclamped = mlir::stablehlo::ConvertElementType(
      idx_floor_calc, builder.getOpBuilder().getI32Type());
  auto idx_ceil_i32_unclamped = mlir::stablehlo::ConvertElementType(
      idx_ceil_calc, builder.getOpBuilder().getI32Type());

  auto zero_i32 =
      MakeConstantLike(src_idx, 0, builder.getOpBuilder().getI32Type());
  auto in_size_minus_1_i32 = MakeConstantLike(
      src_idx, std::max(0L, in_size - 1), builder.getOpBuilder().getI32Type());

  auto idx_floor_i32 = mlir::stablehlo::Clamp(idx_floor_i32_unclamped, zero_i32,
                                              in_size_minus_1_i32);
  auto idx_ceil_i32 = mlir::stablehlo::Clamp(idx_ceil_i32_unclamped, zero_i32,
                                             in_size_minus_1_i32);

  return DimInfo{
      idx_floor_i32,           idx_ceil_i32,          d_ceil, d_floor,
      idx_floor_i32_unclamped, idx_ceil_i32_unclamped};
}

absl::StatusOr<mlir::MlirOp> GatherInterpolatedValue(
    mlir::MlirBuilder& builder, mlir::MlirOp input_calc,
    const std::vector<mlir::MlirOp>& current_indices,
    const GridSamplerGatherConfig& gather_config) {
  std::vector<mlir::MlirOp> reshaped_indices;
  reshaped_indices.reserve(current_indices.size());
  for (auto index : current_indices) {
    const auto ty = GetTensorTypeOrDie(index);
    const auto shape = ty.getShape();
    Dimensions new_shape(shape.begin(), shape.end());
    new_shape.push_back(1);
    auto reshaped_ty =
        mlir::RankedTensorType::get(new_shape, ty.getElementType());
    reshaped_indices.push_back(mlir::stablehlo::Reshape(reshaped_ty, index));
  }

  auto index_tensor = mlir::stablehlo::Concatenate(
      builder, reshaped_indices,
      GetTensorTypeOrDie(reshaped_indices[0]).getRank() - 1);

  return mlir::stablehlo::Gather(input_calc, index_tensor,
                                 gather_config.gather_dimension_numbers,
                                 gather_config.slice_sizes,
                                 /*indices_are_sorted=*/false);
}

absl::StatusOr<mlir::MlirOp> ApplyZeroPadding(
    mlir::MlirBuilder& builder, mlir::MlirOp gathered,
    const std::vector<mlir::MlirOp>& indices_unclamped,
    const Dimensions& input_shape, int64_t spatial_dim_size,
    mlir::Type calc_type, const Dimensions& bcast_dims) {
  auto is_in_bounds = [&](mlir::MlirOp idx, int64_t in_size) {
    auto zero = MakeConstantLike(idx, 0, builder.getOpBuilder().getI32Type());
    auto limit = MakeConstantLike(idx, static_cast<int32_t>(in_size),
                                  builder.getOpBuilder().getI32Type());
    auto lt_zero = mlir::stablehlo::Compare(
        idx, zero, mlir::stablehlo::ComparisonDirection::LT);
    auto ge_limit = mlir::stablehlo::Compare(
        idx, limit, mlir::stablehlo::ComparisonDirection::GE);
    auto out_of_bounds = mlir::stablehlo::Or(lt_zero, ge_limit);
    return mlir::stablehlo::Not(out_of_bounds);
  };

  int64_t offset_dim_size = input_shape.size() - spatial_dim_size;
  mlir::MlirOp pixel_ok;

  if (spatial_dim_size == 2) {
    auto in_h = input_shape[offset_dim_size];
    auto in_w = input_shape[offset_dim_size + 1];
    auto x_ok = is_in_bounds(indices_unclamped[0], in_w);
    auto y_ok = is_in_bounds(indices_unclamped[1], in_h);
    pixel_ok = mlir::stablehlo::And(x_ok, y_ok);
  } else {  // spatial_dim_size == 3
    auto in_d = input_shape[offset_dim_size];
    auto in_h = input_shape[offset_dim_size + 1];
    auto in_w = input_shape[offset_dim_size + 2];
    auto x_ok = is_in_bounds(indices_unclamped[0], in_w);
    auto y_ok = is_in_bounds(indices_unclamped[1], in_h);
    auto z_ok = is_in_bounds(indices_unclamped[2], in_d);
    auto yz_ok = mlir::stablehlo::And(y_ok, z_ok);
    pixel_ok = mlir::stablehlo::And(x_ok, yz_ok);
  }

  auto gathered_ty = GetTensorTypeOrDie(gathered);
  auto pixel_ok_bcast_ty = mlir::RankedTensorType::get(
      gathered_ty.getShape(), builder.getOpBuilder().getI1Type());
  auto pixel_ok_bcast =
      mlir::stablehlo::BroadcastInDim(pixel_ok_bcast_ty, pixel_ok, bcast_dims);
  auto zero_pixel = MakeConstantLike(gathered, 0.0, calc_type);
  return mlir::stablehlo::Select(pixel_ok_bcast, gathered, zero_pixel);
}

struct BilinearDimInfo {
  mlir::MlirOp idx;
  mlir::MlirOp weight;
  mlir::MlirOp idx_unclamped;
};

BilinearDimInfo GetBilinearDimInfo(DimInfo dim_info, int bit) {
  if (bit) {
    return BilinearDimInfo{
        .idx = dim_info.idx_ceil,
        .weight = dim_info.weight_ceil,
        .idx_unclamped = dim_info.idx_ceil_unclamped,
    };
  }
  return BilinearDimInfo{
      .idx = dim_info.idx_floor,
      .weight = dim_info.weight_floor,
      .idx_unclamped = dim_info.idx_floor_unclamped,
  };
}

absl::StatusOr<mlir::MlirOp> AccumulateBilinearInterpolation2D(
    mlir::MlirBuilder& builder, mlir::MlirOp input,
    const std::vector<DimInfo>& dim_infos,
    const GridSamplerGatherConfig& gather_config, PaddingMode padding_mode,
    mlir::Type calc_type) {
  mlir::MlirOp accumulated_result = MakeConstant(builder, 0, calc_type, {1});
  const auto input_calc = mlir::stablehlo::ConvertElementType(input, calc_type);
  const auto bcast_dims = Dimensions{0, 2, 3};
  const Dimensions input_shape =
      CopyIntVector(GetTensorTypeOrDie(input).getShape());
  const int64_t spatial_dim_size = 2;

  for (int y_bit = 0; y_bit < 2; ++y_bit) {
    for (int x_bit = 0; x_bit < 2; ++x_bit) {
      // x is dim_infos[0], y is dim_infos[1]
      BilinearDimInfo x_dim_info = GetBilinearDimInfo(dim_infos[0], x_bit);
      BilinearDimInfo y_dim_info = GetBilinearDimInfo(dim_infos[1], y_bit);

      TT_ASSIGN_OR_RETURN(auto current_weight,
                          BuildMulShlo(x_dim_info.weight, y_dim_info.weight));

      std::vector<mlir::MlirOp> current_indices = {x_dim_info.idx,
                                                   y_dim_info.idx};
      TT_ASSIGN_OR_RETURN(auto gathered, GatherInterpolatedValue(
                                             builder, input_calc,
                                             current_indices, gather_config));

      if (padding_mode == PaddingMode::kZeros) {
        std::vector<mlir::MlirOp> indices_unclamped = {
            x_dim_info.idx_unclamped, y_dim_info.idx_unclamped};
        TT_ASSIGN_OR_RETURN(
            gathered,
            ApplyZeroPadding(builder, gathered, indices_unclamped, input_shape,
                             spatial_dim_size, calc_type, bcast_dims));
      }

      auto weight_bcast = mlir::stablehlo::BroadcastInDim(
          gathered.getType(), current_weight, bcast_dims);

      TT_ASSIGN_OR_RETURN(auto weighted_value,
                          BuildMulShlo(gathered, weight_bcast));

      TT_ASSIGN_OR_RETURN(accumulated_result,
                          BuildAddShlo(accumulated_result, weighted_value));
    }
  }
  return accumulated_result;
}

absl::StatusOr<mlir::MlirOp> AccumulateBilinearInterpolation3D(
    mlir::MlirBuilder& builder, mlir::MlirOp input,
    const std::vector<DimInfo>& dim_infos,
    const GridSamplerGatherConfig& gather_config, PaddingMode padding_mode,
    mlir::Type calc_type) {
  mlir::MlirOp accumulated_result = MakeConstant(builder, 0, calc_type, {1});

  const auto input_calc = mlir::stablehlo::ConvertElementType(input, calc_type);
  const auto bcast_dims = Dimensions{0, 2, 3, 4};
  const Dimensions input_shape =
      CopyIntVector(GetTensorTypeOrDie(input).getShape());
  const int64_t spatial_dim_size = 3;

  for (int z_bit = 0; z_bit < 2; ++z_bit) {
    for (int y_bit = 0; y_bit < 2; ++y_bit) {
      for (int x_bit = 0; x_bit < 2; ++x_bit) {
        // x is 0, y is 1, z is 2
        BilinearDimInfo x_dim_info = GetBilinearDimInfo(dim_infos[0], x_bit);
        BilinearDimInfo y_dim_info = GetBilinearDimInfo(dim_infos[1], y_bit);
        BilinearDimInfo z_dim_info = GetBilinearDimInfo(dim_infos[2], z_bit);

        TT_ASSIGN_OR_RETURN(auto current_weight,
                            BuildMulShlo(x_dim_info.weight, y_dim_info.weight));
        TT_ASSIGN_OR_RETURN(current_weight,
                            BuildMulShlo(current_weight, z_dim_info.weight));

        std::vector<mlir::MlirOp> current_indices = {
            x_dim_info.idx, y_dim_info.idx, z_dim_info.idx};
        TT_ASSIGN_OR_RETURN(auto gathered, GatherInterpolatedValue(
                                               builder, input_calc,
                                               current_indices, gather_config));

        if (padding_mode == PaddingMode::kZeros) {
          std::vector<mlir::MlirOp> indices_unclamped = {
              x_dim_info.idx_unclamped, y_dim_info.idx_unclamped,
              z_dim_info.idx_unclamped};
          TT_ASSIGN_OR_RETURN(
              gathered, ApplyZeroPadding(builder, gathered, indices_unclamped,
                                         input_shape, spatial_dim_size,
                                         calc_type, bcast_dims));
        }

        auto weight_bcast = mlir::stablehlo::BroadcastInDim(
            gathered.getType(), current_weight, bcast_dims);

        TT_ASSIGN_OR_RETURN(auto weighted_value,
                            BuildMulShlo(gathered, weight_bcast));

        TT_ASSIGN_OR_RETURN(accumulated_result,
                            BuildAddShlo(accumulated_result, weighted_value));
      }
    }
  }
  return accumulated_result;
}

absl::StatusOr<mlir::MlirOp> AccumulateBilinearInterpolation(
    mlir::MlirBuilder& builder, const mlir::MlirOp input,
    const std::vector<DimInfo>& dim_infos,
    const GridSamplerGatherConfig& gather_config,
    const int64_t spatial_dim_size, PaddingMode padding_mode,
    mlir::Type calc_type) {
  if (spatial_dim_size == 2) {
    return AccumulateBilinearInterpolation2D(
        builder, input, dim_infos, gather_config, padding_mode, calc_type);
  } else if (spatial_dim_size == 3) {
    return AccumulateBilinearInterpolation3D(
        builder, input, dim_infos, gather_config, padding_mode, calc_type);
  } else {
    return TT_ERROR(error::kInvalidArgument)
           << "expected 2 or 3 spatial dimensions for bilinear interpolation, "
              "got "
           << spatial_dim_size;
  }
}

struct CubicDimInfo {
  std::vector<mlir::MlirOp> indices;
  std::vector<mlir::MlirOp> weights;
  std::vector<mlir::MlirOp> indices_unclamped;
};

absl::Status UpdateIndicesForPaddingMode(
    mlir::MlirBuilder& builder, const PaddingMode padding_mode,
    CubicDimInfo& info, mlir::MlirOp src_idx, int64_t in_size,
    mlir::MlirOp in_size_minus_1_i32, bool align_corners, mlir::MlirOp one_i32,
    mlir::MlirOp zero_i32) {
  auto i32_type = builder.getOpBuilder().getI32Type();

  // Default update operation for non-reflection padding modes.
  absl::AnyInvocable<absl::Status(mlir::MlirOp & idx)> update_index_fn =
      [&](mlir::MlirOp& idx) mutable -> absl::Status {
    idx = mlir::stablehlo::Clamp(zero_i32, idx, in_size_minus_1_i32);
    return absl::OkStatus();
  };

  // Index update for reflection padding mode.
  if (padding_mode == PaddingMode::kReflection) {
    if (in_size > 1) {
      if (align_corners) {
        auto double_size_minus_2 = MakeConstantLike(
            src_idx, static_cast<int32_t>(2 * in_size - 2), i32_type);
        update_index_fn =
            [&, double_size_minus_2](mlir::MlirOp& idx) -> absl::Status {
          auto abs_idx = mlir::stablehlo::Abs(idx);
          TT_ASSIGN_OR_RETURN(
              auto mod_idx, BuildFmodTensorShlo(abs_idx, double_size_minus_2));
          TT_ASSIGN_OR_RETURN(auto reflected_idx,
                              BuildSubShlo(double_size_minus_2, mod_idx));
          TT_ASSIGN_OR_RETURN(idx, BuildMinimumShlo(mod_idx, reflected_idx));
          return absl::OkStatus();
        };
      } else {
        auto four_size = MakeConstantLike(
            src_idx, static_cast<int32_t>(4 * in_size), i32_type);
        auto two_i32 = MakeConstantLike(src_idx, 2, i32_type);
        update_index_fn = [&, four_size,
                           two_i32](mlir::MlirOp& idx) -> absl::Status {
          // shifted = 2 * idx + 1
          TT_ASSIGN_OR_RETURN(auto temp1, BuildMulShlo(idx, two_i32));
          TT_ASSIGN_OR_RETURN(auto shifted, BuildAddShlo(temp1, one_i32));

          auto abs_shifted = mlir::stablehlo::Abs(shifted);
          TT_ASSIGN_OR_RETURN(auto mod_shifted,
                              BuildFmodTensorShlo(abs_shifted, four_size));
          TT_ASSIGN_OR_RETURN(auto reflected_shifted,
                              BuildSubShlo(four_size, mod_shifted));
          TT_ASSIGN_OR_RETURN(auto min_shifted,
                              BuildMinimumShlo(mod_shifted, reflected_shifted));

          // idx = (min_shifted - 1) / 2
          TT_ASSIGN_OR_RETURN(auto temp2, BuildSubShlo(min_shifted, one_i32));
          TT_ASSIGN_OR_RETURN(idx, BuildDivShlo(temp2, two_i32));
          idx = mlir::stablehlo::Clamp(zero_i32, idx, in_size_minus_1_i32);
          return absl::OkStatus();
        };
      }
    } else {
      update_index_fn = [&](mlir::MlirOp& idx) -> absl::Status {
        idx = zero_i32;
        return absl::OkStatus();
      };
    }
  }

  for (auto& idx : info.indices) {
    TT_RETURN_IF_ERROR(update_index_fn(idx));
  }
  return absl::OkStatus();
}

absl::StatusOr<CubicDimInfo> ComputeCubicDimInfo(
    mlir::MlirBuilder& builder, mlir::MlirOp src_idx, int64_t in_size,
    PaddingMode padding_mode, bool align_corners, mlir::Type calc_type) {
  const auto idx_floor_calc = mlir::stablehlo::Floor(src_idx);
  TT_ASSIGN_OR_RETURN(auto t, BuildSubShlo(src_idx, idx_floor_calc));

  const auto one = MakeConstantLike(src_idx, 1.0, calc_type);
  const auto two = MakeConstantLike(src_idx, 2.0, calc_type);

  TT_ASSIGN_OR_RETURN(auto x1, BuildAddShlo(t, one));
  auto x2 = t;
  TT_ASSIGN_OR_RETURN(auto x3, BuildSubShlo(one, t));
  TT_ASSIGN_OR_RETURN(auto x4, BuildSubShlo(two, t));

  const auto A_val = -0.75;
  const auto A = MakeConstantLike(src_idx, A_val, calc_type);
  const auto A_plus_2 = MakeConstantLike(src_idx, A_val + 2.0, calc_type);
  const auto A_plus_3 = MakeConstantLike(src_idx, A_val + 3.0, calc_type);

  auto conv1 = [&](mlir::MlirOp x) -> absl::StatusOr<mlir::MlirOp> {
    // ((A + 2) * x - (A + 3)) * x * x + 1
    TT_ASSIGN_OR_RETURN(auto term1, BuildMulShlo(A_plus_2, x));
    TT_ASSIGN_OR_RETURN(auto term2, BuildSubShlo(term1, A_plus_3));
    TT_ASSIGN_OR_RETURN(auto term3, BuildMulShlo(term2, x));
    TT_ASSIGN_OR_RETURN(auto term4, BuildMulShlo(term3, x));
    return BuildAddShlo(term4, one);
  };

  auto conv2 = [&](mlir::MlirOp x) -> absl::StatusOr<mlir::MlirOp> {
    // ((A * x - 5 * A) * x + 8 * A) * x - 4 * A
    const auto five_A = MakeConstantLike(src_idx, 5.0 * A_val, calc_type);
    const auto eight_A = MakeConstantLike(src_idx, 8.0 * A_val, calc_type);
    const auto four_A = MakeConstantLike(src_idx, 4.0 * A_val, calc_type);

    TT_ASSIGN_OR_RETURN(auto term1, BuildMulShlo(A, x));
    TT_ASSIGN_OR_RETURN(auto term2, BuildSubShlo(term1, five_A));
    TT_ASSIGN_OR_RETURN(auto term3, BuildMulShlo(term2, x));
    TT_ASSIGN_OR_RETURN(auto term4, BuildAddShlo(term3, eight_A));
    TT_ASSIGN_OR_RETURN(auto term5, BuildMulShlo(term4, x));
    return BuildSubShlo(term5, four_A);
  };

  CubicDimInfo info;
  TT_ASSIGN_OR_RETURN(auto w0, conv2(x1));
  TT_ASSIGN_OR_RETURN(auto w1, conv1(x2));
  TT_ASSIGN_OR_RETURN(auto w2, conv1(x3));
  TT_ASSIGN_OR_RETURN(auto w3, conv2(x4));
  info.weights = {w0, w1, w2, w3};

  const auto i32_type = builder.getOpBuilder().getI32Type();
  const auto idx_floor_i32 =
      mlir::stablehlo::ConvertElementType(idx_floor_calc, i32_type);

  const auto minus_one_i32 = MakeConstantLike(src_idx, -1, i32_type);
  const auto zero_i32 = MakeConstantLike(src_idx, 0, i32_type);
  const auto one_i32 = MakeConstantLike(src_idx, 1, i32_type);
  const auto two_i32 = MakeConstantLike(src_idx, 2, i32_type);

  TT_ASSIGN_OR_RETURN(auto i0, BuildAddShlo(idx_floor_i32, minus_one_i32));
  auto i1 = idx_floor_i32;
  TT_ASSIGN_OR_RETURN(auto i2, BuildAddShlo(idx_floor_i32, one_i32));
  TT_ASSIGN_OR_RETURN(auto i3, BuildAddShlo(idx_floor_i32, two_i32));

  info.indices_unclamped = {i0, i1, i2, i3};
  info.indices = info.indices_unclamped;

  const auto in_size_minus_1_i32 =
      MakeConstantLike(src_idx, static_cast<int32_t>(in_size - 1), i32_type);

  TT_RETURN_IF_ERROR(UpdateIndicesForPaddingMode(
      builder, padding_mode, info, src_idx, in_size, in_size_minus_1_i32,
      align_corners, one_i32, zero_i32));

  return info;
}

absl::StatusOr<mlir::MlirOp> AccumulateBicubicInterpolation(
    mlir::MlirBuilder& builder, mlir::MlirOp input,
    const std::vector<CubicDimInfo>& dim_infos,
    const std::vector<mlir::MlirOp>& grid_coords,
    const GridSamplerGatherConfig& gather_config, int64_t spatial_dim_size,
    PaddingMode padding_mode, mlir::Type calc_type) {
  mlir::MlirOp accumulated_result = MakeConstant(builder, 0, calc_type, {1});
  auto input_calc = mlir::stablehlo::ConvertElementType(input, calc_type);
  const auto bcast_dims = Dimensions{0, 2, 3};

  for (int y_bit = 0; y_bit < 4; ++y_bit) {
    for (int x_bit = 0; x_bit < 4; ++x_bit) {
      auto x_idx = dim_infos[0].indices[x_bit];
      auto y_idx = dim_infos[1].indices[y_bit];
      auto x_w = dim_infos[0].weights[x_bit];
      auto y_w = dim_infos[1].weights[y_bit];

      auto x_idx_unclamped = dim_infos[0].indices_unclamped[x_bit];
      auto y_idx_unclamped = dim_infos[1].indices_unclamped[y_bit];

      TT_ASSIGN_OR_RETURN(auto current_weight, BuildMulShlo(x_w, y_w));

      std::vector<mlir::MlirOp> current_indices = {x_idx, y_idx};
      std::vector<mlir::MlirOp> reshaped_indices;
      for (auto index : current_indices) {
        auto ty = GetTensorTypeOrDie(index);
        auto shape = ty.getShape();
        Dimensions new_shape(shape.begin(), shape.end());
        new_shape.push_back(1);
        auto reshaped_ty =
            mlir::RankedTensorType::get(new_shape, ty.getElementType());
        reshaped_indices.push_back(
            mlir::stablehlo::Reshape(reshaped_ty, index));
      }

      auto index_tensor = mlir::stablehlo::Concatenate(
          builder, reshaped_indices,
          GetTensorTypeOrDie(reshaped_indices[0]).getRank() - 1);

      auto gathered = mlir::stablehlo::Gather(
          input_calc, index_tensor, gather_config.gather_dimension_numbers,
          gather_config.slice_sizes,
          /*indices_are_sorted=*/false);

      if (padding_mode == PaddingMode::kZeros) {
        auto is_in_bounds = [&](mlir::MlirOp idx, int64_t in_size) {
          auto zero =
              MakeConstantLike(idx, 0, builder.getOpBuilder().getI32Type());
          auto limit = MakeConstantLike(idx, static_cast<int32_t>(in_size),
                                        builder.getOpBuilder().getI32Type());
          auto lt_zero = mlir::stablehlo::Compare(
              idx, zero, mlir::stablehlo::ComparisonDirection::LT);
          auto ge_limit = mlir::stablehlo::Compare(
              idx, limit, mlir::stablehlo::ComparisonDirection::GE);
          auto out_of_bounds = mlir::stablehlo::Or(lt_zero, ge_limit);
          return mlir::stablehlo::Not(out_of_bounds);
        };

        // TODO: lwh - Add support for dynamic shape.
        auto input_ty = GetTensorTypeOrDie(input);
        Dimensions input_shape = CopyIntVector(input_ty.getShape());
        int64_t offset_dim_size = input_shape.size() - spatial_dim_size;
        auto in_w = input_shape[offset_dim_size + 1];
        auto in_h = input_shape[offset_dim_size];

        auto x_ok = is_in_bounds(x_idx_unclamped, in_w);
        auto y_ok = is_in_bounds(y_idx_unclamped, in_h);
        auto pixel_ok = mlir::stablehlo::And(x_ok, y_ok);

        const auto gathered_ty = GetTensorTypeOrDie(gathered);
        auto pixel_ok_bcast_ty = mlir::RankedTensorType::get(
            gathered_ty.getShape(), builder.getOpBuilder().getI1Type());
        auto pixel_ok_bcast = mlir::stablehlo::BroadcastInDim(
            pixel_ok_bcast_ty, pixel_ok, bcast_dims);
        auto zero_pixel = MakeConstantLike(gathered, 0.0, calc_type);
        gathered =
            mlir::stablehlo::Select(pixel_ok_bcast, gathered, zero_pixel);
      }

      const auto weight_bcast = mlir::stablehlo::BroadcastInDim(
          gathered.getType(), current_weight, bcast_dims);

      TT_ASSIGN_OR_RETURN(auto weighted_value,
                          BuildMulShlo(gathered, weight_bcast));
      TT_ASSIGN_OR_RETURN(accumulated_result,
                          BuildAddShlo(accumulated_result, weighted_value));
    }
  }

  return accumulated_result;
}

absl::StatusOr<mlir::MlirOp> BuildGridSamplerBicubicShlo(
    mlir::MlirOp input, mlir::MlirOp grid, bool align_corners,
    PaddingMode padding_mode) {
  mlir::MlirBuilder& builder = input.getBuilder();
  const auto input_ty = GetTensorTypeOrDie(input);
  Dimensions input_shape = CopyIntVector(input_ty.getShape());
  auto grid_ty = GetTensorTypeOrDie(grid);
  Dimensions grid_shape = CopyIntVector(grid_ty.getShape());
  int64_t spatial_dim_size = grid_shape.back();
  int64_t rank = input_shape.size();
  int64_t offset_dim_size = rank - spatial_dim_size;

  if (spatial_dim_size != 2) {
    return TT_ERROR(error::kInvalidArgument)
           << "expected 2 spatial dimensions for bicubic interpolation, got "
           << spatial_dim_size;
  }

  GridSamplerGatherConfig gather_config =
      GetGridSamplerGatherConfig(input, spatial_dim_size);

  const auto element_type = input_ty.getElementType();
  // Use f64 for bicubic calculation to match PyTorch accscalar_t.
  const mlir::Type calc_type = builder.getOpBuilder().getF64Type();
  auto grid_calc = mlir::stablehlo::ConvertElementType(grid, calc_type);

  std::vector<CubicDimInfo> dim_infos;
  dim_infos.reserve(spatial_dim_size);
  std::vector<mlir::MlirOp> grid_coords;
  grid_coords.reserve(spatial_dim_size);

  for (int i = 0; i < spatial_dim_size; ++i) {
    Dimensions offsets(grid_shape.size(), 0);
    offsets.back() = i;
    Dimensions limits = grid_shape;
    limits.back() = i + 1;
    Dimensions strides(grid_shape.size(), 1);
    auto grid_coord_sliced =
        mlir::stablehlo::Slice(grid_calc, offsets, limits, strides);
    Dimensions coord_shape = grid_shape;
    coord_shape.pop_back();
    auto grid_coord_ty = mlir::RankedTensorType::get(coord_shape, calc_type);
    auto grid_coord_raw =
        mlir::stablehlo::Reshape(grid_coord_ty, grid_coord_sliced);
    grid_coords.push_back(grid_coord_raw);

    int64_t in_size = input_shape[offset_dim_size + spatial_dim_size - 1 - i];
    TT_ASSIGN_OR_RETURN(
        auto src_coord,
        ComputeSourceCoord(grid_coord_raw, in_size, align_corners, padding_mode,
                           calc_type));
    TT_ASSIGN_OR_RETURN(
        auto dim_info,
        ComputeCubicDimInfo(builder, src_coord, in_size, padding_mode,
                            align_corners, calc_type));
    dim_infos.push_back(dim_info);
  }

  TT_ASSIGN_OR_RETURN(auto accumulated_result,
                      AccumulateBicubicInterpolation(
                          builder, input, dim_infos, grid_coords, gather_config,
                          spatial_dim_size, padding_mode, calc_type));

  return mlir::stablehlo::ConvertElementType(accumulated_result, element_type);
}

absl::StatusOr<mlir::MlirOp> BuildGridSamplerBilinearShlo(
    mlir::MlirOp input, mlir::MlirOp grid, bool align_corners,
    PaddingMode padding_mode) {
  mlir::MlirBuilder& builder = input.getBuilder();
  const auto input_ty = GetTensorTypeOrDie(input);
  // TODO: lwh - Add support for dynamic shape.
  Dimensions input_shape = CopyIntVector(input_ty.getShape());
  const auto grid_ty = GetTensorTypeOrDie(grid);
  Dimensions grid_shape = CopyIntVector(grid_ty.getShape());
  int64_t spatial_dim_size = grid_shape.back();
  int64_t rank = input_shape.size();
  int64_t offset_dim_size = rank - spatial_dim_size;

  GridSamplerGatherConfig gather_config =
      GetGridSamplerGatherConfig(input, spatial_dim_size);

  const auto element_type = input_ty.getElementType();
  // Use f64 for intermediate calculation to match PyTorch accscalar_t.
  const mlir::Type calc_type = builder.getOpBuilder().getF64Type();
  auto grid_calc = mlir::stablehlo::ConvertElementType(grid, calc_type);

  std::vector<DimInfo> dim_infos;
  dim_infos.reserve(spatial_dim_size);
  std::vector<mlir::MlirOp> grid_coords;
  grid_coords.reserve(spatial_dim_size);

  for (int i = 0; i < spatial_dim_size; ++i) {
    Dimensions offsets(grid_shape.size(), 0);
    offsets.back() = i;
    Dimensions limits = grid_shape;
    limits.back() = i + 1;
    Dimensions strides(grid_shape.size(), 1);
    auto grid_coord_sliced =
        mlir::stablehlo::Slice(grid_calc, offsets, limits, strides);
    Dimensions coord_shape = grid_shape;
    coord_shape.pop_back();
    auto grid_coord_ty = mlir::RankedTensorType::get(coord_shape, calc_type);
    auto grid_coord_raw =
        mlir::stablehlo::Reshape(grid_coord_ty, grid_coord_sliced);
    grid_coords.push_back(grid_coord_raw);

    const int64_t in_size =
        input_shape[offset_dim_size + spatial_dim_size - 1 - i];
    TT_ASSIGN_OR_RETURN(
        const auto src_coord,
        ComputeSourceCoord(grid_coord_raw, in_size, align_corners, padding_mode,
                           calc_type));
    TT_ASSIGN_OR_RETURN(auto dim_info,
                        ComputeDimInfo(builder, src_coord, in_size, calc_type));
    dim_infos.push_back(std::move(dim_info));
  }

  TT_ASSIGN_OR_RETURN(auto accumulated_result,
                      AccumulateBilinearInterpolation(
                          builder, input, dim_infos, gather_config,
                          spatial_dim_size, padding_mode, calc_type));

  return mlir::stablehlo::ConvertElementType(accumulated_result, element_type);
}

absl::StatusOr<mlir::MlirOp> BuildGridSamplerNearestShlo(
    mlir::MlirOp input, mlir::MlirOp grid, bool align_corners,
    PaddingMode padding_mode) {
  mlir::MlirBuilder& builder = input.getBuilder();
  auto input_ty = GetTensorTypeOrDie(input);
  // TODO: lwh - Add support for dynamic shape.
  Dimensions input_shape = CopyIntVector(input_ty.getShape());
  auto grid_ty = GetTensorTypeOrDie(grid);
  Dimensions grid_shape = CopyIntVector(grid_ty.getShape());
  int64_t spatial_dim_size = grid_shape.back();
  int64_t rank = input_shape.size();
  int64_t offset_dim_size = rank - spatial_dim_size;

  GridSamplerGatherConfig gather_config =
      GetGridSamplerGatherConfig(input, spatial_dim_size);

  // Use f64 for intermediate calculation to match PyTorch accscalar_t.
  mlir::Type calc_type = builder.getOpBuilder().getF64Type();
  auto grid_calc = mlir::stablehlo::ConvertElementType(grid, calc_type);

  std::vector<mlir::MlirOp> indices;
  indices.reserve(spatial_dim_size);
  std::vector<mlir::MlirOp> grid_coords;
  grid_coords.reserve(spatial_dim_size);
  std::vector<mlir::MlirOp> indices_ok;
  indices_ok.reserve(spatial_dim_size);

  for (int i = 0; i < spatial_dim_size; ++i) {
    Dimensions offsets(grid_shape.size(), 0);
    offsets.back() = i;
    Dimensions limits = grid_shape;
    limits.back() = i + 1;
    Dimensions strides(grid_shape.size(), 1);
    auto grid_coord_sliced =
        mlir::stablehlo::Slice(grid_calc, offsets, limits, strides);
    Dimensions coord_shape = grid_shape;
    coord_shape.pop_back();
    auto grid_coord_ty = mlir::RankedTensorType::get(coord_shape, calc_type);
    auto grid_coord_raw =
        mlir::stablehlo::Reshape(grid_coord_ty, grid_coord_sliced);
    grid_coords.push_back(grid_coord_raw);

    int64_t in_size = input_shape[offset_dim_size + spatial_dim_size - 1 - i];
    TT_ASSIGN_OR_RETURN(
        auto src_coord,
        ComputeSourceCoord(grid_coord_raw, in_size, align_corners, padding_mode,
                           calc_type));

    auto round = mlir::stablehlo::RoundNearestEven(src_coord);
    auto indices_i32_unclamped = mlir::stablehlo::ConvertElementType(
        round, builder.getOpBuilder().getI32Type());

    auto zero_i32 = MakeConstantLike(grid_coord_raw, 0,
                                     builder.getOpBuilder().getI32Type());
    auto in_size_minus_1_i32 =
        MakeConstantLike(grid_coord_raw, std::max(0L, in_size - 1),
                         builder.getOpBuilder().getI32Type());
    auto indices_i32 = mlir::stablehlo::Clamp(indices_i32_unclamped, zero_i32,
                                              in_size_minus_1_i32);
    indices.push_back(indices_i32);

    if (padding_mode == PaddingMode::kZeros) {
      auto is_in_bounds = [&](mlir::MlirOp idx, int64_t in_size) {
        auto zero =
            MakeConstantLike(idx, 0, builder.getOpBuilder().getI32Type());
        auto limit = MakeConstantLike(idx, static_cast<int32_t>(in_size),
                                      builder.getOpBuilder().getI32Type());
        auto lt_zero = mlir::stablehlo::Compare(
            idx, zero, mlir::stablehlo::ComparisonDirection::LT);
        auto ge_limit = mlir::stablehlo::Compare(
            idx, limit, mlir::stablehlo::ComparisonDirection::GE);
        auto out_of_bounds = mlir::stablehlo::Or(lt_zero, ge_limit);
        return mlir::stablehlo::Not(out_of_bounds);
      };
      indices_ok.push_back(is_in_bounds(indices_i32_unclamped, in_size));
    }
  }

  std::vector<mlir::MlirOp> reshaped_indices;
  for (auto index : indices) {
    auto ty = GetTensorTypeOrDie(index);
    auto shape = ty.getShape();
    Dimensions new_shape(shape.begin(), shape.end());
    new_shape.push_back(1);
    auto reshaped_ty =
        mlir::RankedTensorType::get(new_shape, ty.getElementType());
    reshaped_indices.push_back(mlir::stablehlo::Reshape(reshaped_ty, index));
  }

  auto index_tensor = mlir::stablehlo::Concatenate(
      builder, reshaped_indices,
      GetTensorTypeOrDie(reshaped_indices[0]).getRank() - 1);

  auto gathered = mlir::stablehlo::Gather(
      input, index_tensor, gather_config.gather_dimension_numbers,
      gather_config.slice_sizes,
      /*indices_are_sorted=*/false);

  if (padding_mode == PaddingMode::kZeros) {
    mlir::MlirOp all_indices_ok;
    for (int i = 0; i < spatial_dim_size; ++i) {
      if (i == 0) {
        all_indices_ok = indices_ok[i];
      } else {
        all_indices_ok = mlir::stablehlo::And(all_indices_ok, indices_ok[i]);
      }
    }
    auto bcast_dims =
        spatial_dim_size == 2 ? Dimensions{0, 2, 3} : Dimensions{0, 2, 3, 4};
    Dimensions all_indices_ok_bcast_shape =
        CopyIntVector(GetTensorTypeOrDie(gathered).getShape());
    auto all_indices_ok_bcast_ty = mlir::RankedTensorType::get(
        all_indices_ok_bcast_shape,
        GetTensorTypeOrDie(all_indices_ok).getElementType());

    auto all_indices_ok_bcast = mlir::stablehlo::BroadcastInDim(
        all_indices_ok_bcast_ty, all_indices_ok, bcast_dims);
    auto zero_result = MakeConstantLike(
        gathered, 0.0, GetTensorTypeOrDie(gathered).getElementType());
    gathered =
        mlir::stablehlo::Select(all_indices_ok_bcast, gathered, zero_result);
  }

  return gathered;
}

// Checks for the `input` and `padding_mode` arguments.
// These are common for both 2D and 3D grid sampler implementations.
absl::Status CheckInputAndPaddingMode(const at::Tensor& input,
                                      const int64_t padding_mode) {
  TT_RET_CHECK(padding_mode == 0 || padding_mode == 1 || padding_mode == 2,
               error::kUnimplemented)
      << "expected the padding mode to be 0 (zeros), 1 (border), or 2 "
         "(reflection), got "
      << padding_mode;

  TT_RET_CHECK(IsFloatingPoint(input), error::kInvalidArgument)
      << "expected the input dtype to be floating point, got "
      << ToString(input.scalar_type());

  return absl::OkStatus();
}

// Checks for the `interpolation_mode` argument for 2D grid sampler.
absl::Status Check2DInterpolationMode(const int64_t interpolation_mode) {
  TT_RET_CHECK(interpolation_mode == 0 || interpolation_mode == 1 ||
                   interpolation_mode == 2,
               error::kUnimplemented)
      << "expected the interpolation mode to be 0 (bilinear), 1 (nearest), or "
         "2 (bicubic), got "
      << interpolation_mode;

  return absl::OkStatus();
}

// Checks for the `interpolation_mode` argument for 3D grid sampler.
absl::Status Check3DInterpolationMode(const int64_t interpolation_mode) {
  TT_RET_CHECK(interpolation_mode == 0 || interpolation_mode == 1,
               error::kUnimplemented)
      << "expected the interpolation mode to be 0 (bilinear) or 1 (nearest), "
         "got "
      << interpolation_mode;

  return absl::OkStatus();
}

}  // namespace

at::Tensor AtenGridSampler2d(const at::Tensor& input, const at::Tensor& grid,
                             int64_t interpolation_mode, int64_t padding_mode,
                             bool align_corners) {
  TT_KERNEL(
      OpName::kGridSampler2d, param_keys,
      (input, grid, interpolation_mode, padding_mode, align_corners), {
        TT_THROW_IF_ERROR(CheckInputAndPaddingMode(input, padding_mode));
        TT_THROW_IF_ERROR(Check2DInterpolationMode(interpolation_mode));

        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(input.scalar_type()));
        auto grid_shape = grid.sizes();
        Dimensions output_dims = {grid_shape[0], input.size(1), grid_shape[1],
                                  grid_shape[2]};

        PaddingMode pm = static_cast<PaddingMode>(padding_mode);
        auto op_builder = [&]() -> NAryMlirOpBuilder<2, 1> {
          if (interpolation_mode == 0) {  // bilinear
            return [align_corners, pm](FixedSizeSpan<mlir::MlirOp, 2> inputs)
                       -> absl::StatusOr<mlir::MlirOp> {
              return BuildGridSamplerBilinearShlo(inputs[0], inputs[1],
                                                  align_corners, pm);
            };
          } else if (interpolation_mode == 1) {  // nearest
            return [align_corners, pm](FixedSizeSpan<mlir::MlirOp, 2> inputs)
                       -> absl::StatusOr<mlir::MlirOp> {
              return BuildGridSamplerNearestShlo(inputs[0], inputs[1],
                                                 align_corners, pm);
            };
          } else {  // bicubic
            return [align_corners, pm](FixedSizeSpan<mlir::MlirOp, 2> inputs)
                       -> absl::StatusOr<mlir::MlirOp> {
              return BuildGridSamplerBicubicShlo(inputs[0], inputs[1],
                                                 align_corners, pm);
            };
          }
        }();

        TT_ASSIGN_OR_THROW(
            auto out_buf,
            (DispatchOp<2>(std::move(op_builder), {input, grid},
                           {.out_dtype = element_type,
                            .out_dims = output_dims,
                            .op_param_cache_keys = std::move(param_keys)})));
        at::Tensor out = at::empty(output_dims, input.options());
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(out_buf), out));
        return out;
      });
}

at::Tensor AtenGridSampler3d(const at::Tensor& input, const at::Tensor& grid,
                             int64_t interpolation_mode, int64_t padding_mode,
                             bool align_corners) {
  TT_KERNEL(
      OpName::kGridSampler3d, param_keys,
      (input, grid, interpolation_mode, padding_mode, align_corners), {
        TT_THROW_IF_ERROR(CheckInputAndPaddingMode(input, padding_mode));
        TT_THROW_IF_ERROR(Check3DInterpolationMode(interpolation_mode));

        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(input.scalar_type()));
        auto grid_shape = grid.sizes();
        Dimensions output_dims = {grid_shape[0], input.size(1), grid_shape[1],
                                  grid_shape[2], grid_shape[3]};

        PaddingMode pm = static_cast<PaddingMode>(padding_mode);

        auto op_builder = [&]() -> NAryMlirOpBuilder<2, 1> {
          if (interpolation_mode == 0) {  // trilinear
            return [align_corners, pm](FixedSizeSpan<mlir::MlirOp, 2> inputs)
                       -> absl::StatusOr<mlir::MlirOp> {
              return BuildGridSamplerBilinearShlo(inputs[0], inputs[1],
                                                  align_corners, pm);
            };

          } else {  // nearest
            return [align_corners, pm](FixedSizeSpan<mlir::MlirOp, 2> inputs)
                       -> absl::StatusOr<mlir::MlirOp> {
              return BuildGridSamplerNearestShlo(inputs[0], inputs[1],
                                                 align_corners, pm);
            };
          }
        }();
        TT_ASSIGN_OR_THROW(
            auto out_buf,
            (DispatchOp<2>(std::move(op_builder), {input, grid},
                           {.out_dtype = element_type,
                            .out_dims = output_dims,
                            .op_param_cache_keys = std::move(param_keys)})));
        at::Tensor out = at::empty(output_dims, input.options());
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(out_buf), out));
        return out;
      });
}

}  // namespace torch_tpu
