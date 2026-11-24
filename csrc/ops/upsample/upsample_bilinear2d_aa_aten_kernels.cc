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

#include "csrc/ops/upsample/upsample_bilinear2d_aa_aten_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/binary.h"
#include "csrc/ops/einsum/einsum_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/precision_context.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/reductions/sum.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

constexpr int64_t kMaxMatrixSpatialDim = 4096;

enum class AlignCorners : bool {
  kNo = false,
  kYes = true,
};

enum class SpatialAxis : int64_t {
  kHeight = 2,
  kWidth = 3,
};

int64_t SpatialAxisDim(const SpatialAxis axis) {
  switch (axis) {
    case SpatialAxis::kHeight:
      return 2;
    case SpatialAxis::kWidth:
      return 3;
  }
}

std::string_view SpatialAxisEinsumEquation(const SpatialAxis axis) {
  switch (axis) {
    case SpatialAxis::kHeight:
      return "nchw,oh->ncow";
    case SpatialAxis::kWidth:
      return "nchw,ow->ncho";
  }
}

struct BilinearAa1dGatherConfig {
  mlir::stablehlo::GatherDimensionNumbersAttr gather_dimension_numbers;
  Dimensions slice_sizes;
};

struct BilinearAaFilterParams {
  double scale;
  double support;
  double invscale;
  int64_t interp_size;
};

struct BilinearAaWindowBounds {
  mlir::MlirOp iota;
  mlir::MlirOp zero_calc;
  mlir::MlirOp invscale_calc;
  mlir::MlirOp xmin_floor_calc;
  mlir::MlirOp xsize_calc;
  mlir::MlirOp max_idx_calc;
  mlir::MlirOp xmin_m_center;
};

struct BilinearAa1dTaps {
  std::vector<mlir::MlirOp> clamped_indices_1d;
  std::vector<mlir::MlirOp> norm_weights;
};

BilinearAa1dGatherConfig GetBilinearAa1dGatherConfig(
    mlir::MlirOp input, const int64_t spatial_dim) {
  const Dimensions input_shape =
      CopyIntVector(GetTensorTypeOrDie(input).getShape());
  Dimensions slice_sizes = input_shape;
  slice_sizes[spatial_dim] = 1;

  Dimensions offset_dims;
  offset_dims.reserve(input_shape.size() - 1);
  for (int64_t d = 0; d < static_cast<int64_t>(input_shape.size()); ++d) {
    if (d != spatial_dim) {
      offset_dims.push_back(d);
    }
  }
  const Dimensions collapsed_slice_dims = {spatial_dim};
  const Dimensions start_index_map = {spatial_dim};

  const mlir::stablehlo::GatherDimensionNumbersAttr gather_dimension_numbers =
      mlir::stablehlo::GatherDimensionNumbersAttr::get(
          &input.getContext(),
          /*offset_dims=*/AsArrayRef<int64_t>(offset_dims),
          /*collapsed_slice_dims=*/AsArrayRef<int64_t>(collapsed_slice_dims),
          /*operand_batching_dims=*/{},
          /*start_indices_batching_dims=*/{},
          /*start_index_map=*/AsArrayRef<int64_t>(start_index_map),
          /*index_vector_dim=*/1);

  return {gather_dimension_numbers, slice_sizes};
}

double DivideInCalcPrecision(const double num, const double den,
                             const bool is_f64) {
  if (is_f64) {
    return num / den;
  }
  return static_cast<double>(static_cast<float>(num) / static_cast<float>(den));
}

double ComputeAxisScale(const int64_t in_size, const int64_t out_size,
                        const AlignCorners align_corners,
                        const std::optional<double> scale_opt,
                        const bool is_f64) {
  switch (align_corners) {
    case AlignCorners::kYes:
      if (out_size > 1) {
        return DivideInCalcPrecision(static_cast<double>(in_size - 1),
                                     static_cast<double>(out_size - 1), is_f64);
      }
      return 0.0;
    case AlignCorners::kNo:
      if (scale_opt.has_value() && *scale_opt > 0.0) {
        return DivideInCalcPrecision(1.0, *scale_opt, is_f64);
      }
      return DivideInCalcPrecision(static_cast<double>(in_size),
                                   static_cast<double>(out_size), is_f64);
  }
}

BilinearAaFilterParams ComputeBilinearAaFilterParams(
    const int64_t in_size, const int64_t out_size,
    const AlignCorners align_corners, const std::optional<double> scale_opt,
    const bool is_f64) {
  const double scale =
      ComputeAxisScale(in_size, out_size, align_corners, scale_opt, is_f64);
  constexpr double kHalfWidth = 1.0;
  if (scale >= 1.0) {
    const double support =
        is_f64 ? (kHalfWidth * scale)
               : static_cast<double>(static_cast<float>(kHalfWidth) *
                                     static_cast<float>(scale));
    const double invscale = DivideInCalcPrecision(1.0, scale, is_f64);
    const int64_t interp_size = std::min<int64_t>(
        static_cast<int64_t>(std::ceil(support)) * 2 + 1, in_size);
    return {scale, support, invscale, interp_size};
  }
  const int64_t interp_size = std::min<int64_t>(
      static_cast<int64_t>(std::ceil(kHalfWidth)) * 2 + 1, in_size);
  return {scale, kHalfWidth, 1.0, interp_size};
}

// Evaluates the bilinear (triangle) filter for x >= 0:
//   w(x) = max(0.0, 1.0 - x)
absl::StatusOr<mlir::MlirOp> EvaluateBilinearFilter(mlir::MlirOp x) {
  mlir::MlirOp zero = MakeConstantLike(x, 0.0);
  mlir::MlirOp one = MakeConstantLike(x, 1.0);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp one_minus_x, BuildSubShlo(one, x));
  return BuildMaximumShlo(zero, one_minus_x);
}

absl::StatusOr<BilinearAaWindowBounds> ComputeBilinearAaWindowBounds(
    mlir::MlirBuilder& builder, const int64_t in_size, const int64_t out_size,
    const BilinearAaFilterParams& params, const mlir::Type calc_type) {
  const mlir::RankedTensorType iota_type =
      mlir::RankedTensorType::get(Dimensions{out_size, 1}, calc_type);
  mlir::MlirOp iota = mlir::stablehlo::Iota(builder, iota_type, 0);
  mlir::MlirOp half_calc = MakeConstantLike(iota, 0.5);
  mlir::MlirOp zero_calc = MakeConstantLike(iota, 0.0);
  mlir::MlirOp scale_calc = MakeConstantLike(iota, params.scale);
  mlir::MlirOp support_calc = MakeConstantLike(iota, params.support);
  mlir::MlirOp invscale_calc = MakeConstantLike(iota, params.invscale);
  mlir::MlirOp in_size_calc =
      MakeConstantLike(iota, static_cast<double>(in_size));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp iota_plus_half,
                      BuildAddShlo(iota, half_calc));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp center,
                      BuildMulShlo(iota_plus_half, scale_calc));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp center_minus_support,
                      BuildSubShlo(center, support_calc));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp xmin_raw,
                      BuildAddShlo(center_minus_support, half_calc));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp xmin_clamped_calc,
                      BuildMaximumShlo(xmin_raw, zero_calc));
  mlir::MlirOp xmin_floor_calc = mlir::stablehlo::Floor(xmin_clamped_calc);

  TT_ASSIGN_OR_RETURN(mlir::MlirOp center_plus_support,
                      BuildAddShlo(center, support_calc));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp xmax_raw,
                      BuildAddShlo(center_plus_support, half_calc));
  mlir::MlirOp xmax_floor_calc = mlir::stablehlo::Floor(xmax_raw);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp xmax_clamped_calc,
                      BuildMinimumShlo(xmax_floor_calc, in_size_calc));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp xsize_unclamped_calc,
                      BuildSubShlo(xmax_clamped_calc, xmin_floor_calc));

  mlir::MlirOp interp_size_calc =
      MakeConstantLike(iota, static_cast<double>(params.interp_size));
  mlir::MlirOp max_idx_calc = MakeConstantLike(
      iota, static_cast<double>(std::max<int64_t>(0, in_size - 1)));
  mlir::MlirOp xsize_calc =
      mlir::stablehlo::Clamp(zero_calc, xsize_unclamped_calc, interp_size_calc);

  TT_ASSIGN_OR_RETURN(mlir::MlirOp xmin_m_center,
                      BuildSubShlo(xmin_floor_calc, center));

  return BilinearAaWindowBounds{
      iota,       zero_calc,    invscale_calc, xmin_floor_calc,
      xsize_calc, max_idx_calc, xmin_m_center};
}

absl::StatusOr<BilinearAa1dTaps> ComputeBilinearAa1dTaps(
    mlir::MlirBuilder& builder, const int64_t in_size, const int64_t out_size,
    const BilinearAaFilterParams& params, const mlir::Type calc_type) {
  TT_ASSIGN_OR_RETURN(BilinearAaWindowBounds bounds,
                      ComputeBilinearAaWindowBounds(builder, in_size, out_size,
                                                    params, calc_type));

  std::vector<mlir::MlirOp> clamped_indices_1d;
  std::vector<mlir::MlirOp> raw_weights;
  clamped_indices_1d.reserve(params.interp_size);
  raw_weights.reserve(params.interp_size);

  mlir::MlirOp total_w = bounds.zero_calc;
  for (int64_t j = 0; j < params.interp_size; ++j) {
    mlir::MlirOp j_calc = MakeConstantLike(bounds.iota, static_cast<double>(j));
    mlir::MlirOp active_mask = mlir::stablehlo::Compare(
        j_calc, bounds.xsize_calc, mlir::stablehlo::ComparisonDirection::LT);

    TT_ASSIGN_OR_RETURN(mlir::MlirOp raw_idx_calc,
                        BuildAddShlo(bounds.xmin_floor_calc, j_calc));
    mlir::MlirOp clamped_idx_calc = mlir::stablehlo::Clamp(
        bounds.zero_calc, raw_idx_calc, bounds.max_idx_calc);
    clamped_indices_1d.push_back(clamped_idx_calc);

    mlir::MlirOp j_plus_half_calc =
        MakeConstantLike(bounds.iota, static_cast<double>(j) + 0.5);
    TT_ASSIGN_OR_RETURN(mlir::MlirOp dist_unscaled,
                        BuildAddShlo(bounds.xmin_m_center, j_plus_half_calc));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp dist_scaled,
                        BuildMulShlo(dist_unscaled, bounds.invscale_calc));
    mlir::MlirOp abs_dist = mlir::stablehlo::Abs(dist_scaled);
    TT_ASSIGN_OR_RETURN(mlir::MlirOp filter_w,
                        EvaluateBilinearFilter(abs_dist));
    mlir::MlirOp masked_w =
        mlir::stablehlo::Select(active_mask, filter_w, bounds.zero_calc);
    raw_weights.push_back(masked_w);
    TT_ASSIGN_OR_RETURN(total_w, BuildAddShlo(total_w, masked_w));
  }

  mlir::MlirOp nonzero_total_mask = mlir::stablehlo::Compare(
      total_w, bounds.zero_calc, mlir::stablehlo::ComparisonDirection::NE);
  std::vector<mlir::MlirOp> norm_weights;
  norm_weights.reserve(params.interp_size);
  for (int64_t j = 0; j < params.interp_size; ++j) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp div_w,
                        BuildDivShlo(raw_weights[j], total_w));
    norm_weights.push_back(
        mlir::stablehlo::Select(nonzero_total_mask, div_w, bounds.zero_calc));
  }

  return BilinearAa1dTaps{std::move(clamped_indices_1d),
                          std::move(norm_weights)};
}

absl::StatusOr<mlir::MlirOp> BuildBilinearAa1dWeightMatrix(
    mlir::MlirBuilder& builder, const int64_t in_size, const int64_t out_size,
    const BilinearAaFilterParams& params, const mlir::Type calc_type) {
  TT_ASSIGN_OR_RETURN(BilinearAaWindowBounds bounds,
                      ComputeBilinearAaWindowBounds(builder, in_size, out_size,
                                                    params, calc_type));
  const mlir::RankedTensorType mat_calc_type =
      mlir::RankedTensorType::get(Dimensions{out_size, in_size}, calc_type);
  const Dimensions bcast_dims = {0, 1};
  mlir::MlirOp xmin_mat = mlir::stablehlo::BroadcastInDim(
      mat_calc_type, bounds.xmin_floor_calc, AsArrayRef<int64_t>(bcast_dims));
  mlir::MlirOp xsize_mat = mlir::stablehlo::BroadcastInDim(
      mat_calc_type, bounds.xsize_calc, AsArrayRef<int64_t>(bcast_dims));
  mlir::MlirOp xmin_m_center_mat = mlir::stablehlo::BroadcastInDim(
      mat_calc_type, bounds.xmin_m_center, AsArrayRef<int64_t>(bcast_dims));
  mlir::MlirOp invscale_mat = mlir::stablehlo::BroadcastInDim(
      mat_calc_type, bounds.invscale_calc, AsArrayRef<int64_t>(bcast_dims));

  mlir::MlirOp col_iota = mlir::stablehlo::Iota(builder, mat_calc_type, 1);
  mlir::MlirOp zero_mat = MakeConstantLike(col_iota, 0.0);
  mlir::MlirOp half_mat = MakeConstantLike(col_iota, 0.5);

  TT_ASSIGN_OR_RETURN(mlir::MlirOp j_mat, BuildSubShlo(col_iota, xmin_mat));
  mlir::MlirOp ge_zero = mlir::stablehlo::Compare(
      j_mat, zero_mat, mlir::stablehlo::ComparisonDirection::GE);
  mlir::MlirOp lt_xsize = mlir::stablehlo::Compare(
      j_mat, xsize_mat, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp active_mask = mlir::stablehlo::And(ge_zero, lt_xsize);

  TT_ASSIGN_OR_RETURN(mlir::MlirOp j_plus_half, BuildAddShlo(j_mat, half_mat));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp dist_unscaled,
                      BuildAddShlo(xmin_m_center_mat, j_plus_half));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp dist_scaled,
                      BuildMulShlo(dist_unscaled, invscale_mat));
  mlir::MlirOp abs_dist = mlir::stablehlo::Abs(dist_scaled);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp filter_w, EvaluateBilinearFilter(abs_dist));
  mlir::MlirOp raw_w = mlir::stablehlo::Select(active_mask, filter_w, zero_mat);

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp total_w,
      BuildSumShlo(raw_w, Dimensions{1}, ReductionMode::kKeepDims));
  mlir::MlirOp total_w_mat = mlir::stablehlo::BroadcastInDim(
      mat_calc_type, total_w, AsArrayRef<int64_t>(bcast_dims));
  mlir::MlirOp nonzero_total_mask = mlir::stablehlo::Compare(
      total_w_mat, zero_mat, mlir::stablehlo::ComparisonDirection::NE);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp norm_w, BuildDivShlo(raw_w, total_w_mat));
  return mlir::stablehlo::Select(nonzero_total_mask, norm_w, zero_mat);
}

absl::StatusOr<mlir::MlirOp> InterpolateBilinearAa1dMatrix(
    mlir::MlirBuilder& builder, mlir::MlirOp input, const SpatialAxis axis,
    const int64_t in_size, const int64_t out_size,
    const BilinearAaFilterParams& params, const mlir::Type calc_type,
    const mlir::stablehlo::Precision precision) {
  const mlir::Type element_type = GetTensorTypeOrDie(input).getElementType();
  TT_ASSIGN_OR_RETURN(mlir::MlirOp weight_mat,
                      BuildBilinearAa1dWeightMatrix(builder, in_size, out_size,
                                                    params, calc_type));
  mlir::MlirOp input_calc =
      (element_type == calc_type)
          ? input
          : mlir::stablehlo::ConvertElementType(input, calc_type);
  const mlir::MlirOp contract_ops[] = {input_calc, weight_mat};
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp contracted,
      BuildEinsumShlo(contract_ops, SpatialAxisEinsumEquation(axis),
                      precision));
  if (element_type == calc_type) {
    return contracted;
  }
  return mlir::stablehlo::ConvertElementType(contracted, element_type);
}

absl::StatusOr<mlir::MlirOp> InterpolateBilinearAa1dGather(
    mlir::MlirBuilder& builder, mlir::MlirOp input, const int64_t spatial_dim,
    const int64_t in_size, const int64_t out_size,
    const BilinearAaFilterParams& params, const mlir::Type calc_type) {
  TT_ASSIGN_OR_RETURN(
      BilinearAa1dTaps taps,
      ComputeBilinearAa1dTaps(builder, in_size, out_size, params, calc_type));
  const Dimensions input_shape =
      CopyIntVector(GetTensorTypeOrDie(input).getShape());
  const mlir::Type element_type = GetTensorTypeOrDie(input).getElementType();
  const mlir::Type i32_type = builder.getOpBuilder().getI32Type();
  const mlir::RankedTensorType vec_calc_type =
      mlir::RankedTensorType::get(Dimensions{out_size}, calc_type);

  Dimensions pass_output_shape = input_shape;
  pass_output_shape[spatial_dim] = out_size;
  const mlir::RankedTensorType pass_calc_type = mlir::RankedTensorType::get(
      AsArrayRef<int64_t>(pass_output_shape), calc_type);
  const BilinearAa1dGatherConfig gather_config =
      GetBilinearAa1dGatherConfig(input, spatial_dim);
  const Dimensions spatial_bcast_dims = {spatial_dim};

  mlir::MlirOp accumulated;
  for (size_t j = 0; j < taps.norm_weights.size(); ++j) {
    mlir::MlirOp clamped_idx_2d = mlir::stablehlo::ConvertElementType(
        taps.clamped_indices_1d[j], i32_type);
    mlir::MlirOp gathered = mlir::stablehlo::Gather(
        input, clamped_idx_2d, gather_config.gather_dimension_numbers,
        AsArrayRef<int64_t>(gather_config.slice_sizes),
        /*indices_are_sorted=*/false);
    mlir::MlirOp gathered_calc =
        (element_type == calc_type)
            ? gathered
            : mlir::stablehlo::ConvertElementType(gathered, calc_type);

    mlir::MlirOp weight_1d =
        mlir::stablehlo::Reshape(vec_calc_type, taps.norm_weights[j]);
    mlir::MlirOp weight_bcast = mlir::stablehlo::BroadcastInDim(
        pass_calc_type, weight_1d, AsArrayRef<int64_t>(spatial_bcast_dims));

    TT_ASSIGN_OR_RETURN(mlir::MlirOp weighted_tap,
                        BuildMulShlo(gathered_calc, weight_bcast));

    if (j == 0) {
      accumulated = weighted_tap;
    } else {
      TT_ASSIGN_OR_RETURN(accumulated, BuildAddShlo(accumulated, weighted_tap));
    }
  }

  if (element_type == calc_type) {
    return accumulated;
  }
  return mlir::stablehlo::ConvertElementType(accumulated, element_type);
}

absl::StatusOr<mlir::MlirOp> InterpolateBilinearAa1d(
    mlir::MlirBuilder& builder, mlir::MlirOp input, const SpatialAxis axis,
    const int64_t out_size, const AlignCorners align_corners,
    const std::optional<double> scale_opt, const mlir::Type calc_type,
    const mlir::stablehlo::Precision precision) {
  const int64_t spatial_dim = SpatialAxisDim(axis);
  const Dimensions input_shape =
      CopyIntVector(GetTensorTypeOrDie(input).getShape());
  const int64_t in_size = input_shape[spatial_dim];
  if (in_size == out_size && (align_corners == AlignCorners::kYes ||
                              !scale_opt.has_value() || *scale_opt == 1.0)) {
    return input;
  }

  const BilinearAaFilterParams params = ComputeBilinearAaFilterParams(
      in_size, out_size, align_corners, scale_opt, calc_type.isF64());
  if (in_size <= kMaxMatrixSpatialDim && out_size <= kMaxMatrixSpatialDim) {
    return InterpolateBilinearAa1dMatrix(
        builder, input, axis, in_size, out_size, params, calc_type, precision);
  }
  return InterpolateBilinearAa1dGather(builder, input, spatial_dim, in_size,
                                       out_size, params, calc_type);
}

absl::StatusOr<mlir::MlirOp> BuildUpsampleBilinear2dAaShlo(
    mlir::MlirOp input, const Dimensions& output_shape,
    const AlignCorners align_corners, const std::optional<double> scales_h,
    const std::optional<double> scales_w, const mlir::Type calc_type,
    const mlir::stablehlo::Precision precision) {
  mlir::MlirBuilder& builder = input.getBuilder();
  // Separable 2D anti-aliased bilinear interpolation:
  // 1. Horizontal pass along W (dim 3):
  //    [N, C, H_in, W_in] -> [N, C, H_in, W_out]
  TT_ASSIGN_OR_RETURN(mlir::MlirOp horiz,
                      InterpolateBilinearAa1d(
                          builder, input, SpatialAxis::kWidth, output_shape[3],
                          align_corners, scales_w, calc_type, precision));
  // 2. Vertical pass along H (dim 2):
  //    [N, C, H_in, W_out] -> [N, C, H_out, W_out]
  return InterpolateBilinearAa1d(builder, horiz, SpatialAxis::kHeight,
                                 output_shape[2], align_corners, scales_h,
                                 calc_type, precision);
}

void CheckUpsampleBilinear2dAaDtypes(const at::Tensor& tensor) {
  TT_CHECK_THROW(at::isFloatingType(tensor.scalar_type()),
                 error::kPythonNotImplementedError)
      << "not implemented for " << ToString(tensor.scalar_type());
}

void ValidateUpsampleBilinear2dAaInputs(const at::Tensor& self,
                                        const at::IntArrayRef output_size,
                                        const at::Tensor& out) {
  TT_CHECK_THROW(output_size.size() == 2, error::kInvalidArgument)
      << "expected output_size to have 2 elements, got "
      << ToString(output_size.size());

  TT_CHECK_THROW(self.dim() == 4, error::kInvalidArgument)
      << "expected 4D input tensor, got " << ToString(self.dim()) << "D tensor";

  TT_CHECK_THROW(self.size(1) != 0, error::kInvalidArgument)
      << "expected non-empty channel dimension, got input: "
      << ToString(self.sizes());

  TT_CHECK_THROW(self.size(2) > 0 && self.size(3) > 0 && output_size[0] > 0 &&
                     output_size[1] > 0,
                 error::kInvalidArgument)
      << "expected input and output spatial sizes to be greater than 0, got "
         "input: "
      << ToString(self.sizes()) << ", output_size: " << ToString(output_size);

  CheckUpsampleBilinear2dAaDtypes(self);

  TT_CHECK_THROW(self.scalar_type() == out.scalar_type(),
                 error::kInvalidArgument)
      << "expected out dtype " << ToString(self.scalar_type()) << ", got "
      << ToString(out.scalar_type());
}

mlir::stablehlo::Precision GetBilinearAaPrecision(
    OpParamCacheKeys& param_keys) {
  return GetAndAddPrecisionWithDefaultTo(
      param_keys,
      mlir::stablehlo::Precision::HIGHEST);  // EXPLICIT_PRECISION_OK=bilinear
}

}  // namespace

at::Tensor& AtenUpsampleBilinear2dAaOut(const at::Tensor& self,
                                        const at::IntArrayRef output_size,
                                        const bool align_corners,
                                        const std::optional<double> scales_h,
                                        const std::optional<double> scales_w,
                                        at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleBilinear2dAaOut, param_keys,
      (self, output_size, align_corners, scales_h, scales_w, out), {
        ValidateUpsampleBilinear2dAaInputs(self, output_size, out);

        const Dimensions expected_output_shape = {
            self.size(0), self.size(1), output_size[0], output_size[1]};
        TT_THROW_IF_ERROR(
            ResizeTensorIfShapeDiffers(out, expected_output_shape));

        if (out.numel() == 0) {
          return out;
        }

        TT_ASSIGN_OR_THROW(const mlir::ElementType element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        const at::ScalarType real_dtype =
            c10::toRealValueType(self.scalar_type());
        const mlir::stablehlo::Precision precision =
            GetBilinearAaPrecision(param_keys);

        const auto op_builder =
            [out_shape = CopyIntVector(out.sizes()),
             align_corners =
                 align_corners ? AlignCorners::kYes : AlignCorners::kNo,
             scales_h, scales_w, real_dtype,
             precision](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          const mlir::Type calc_type =
              (real_dtype == at::kDouble)
                  ? input.getBuilder().getOpBuilder().getF64Type()
                  : input.getBuilder().getOpBuilder().getF32Type();
          return BuildUpsampleBilinear2dAaShlo(input, out_shape, align_corners,
                                               scales_h, scales_w, calc_type,
                                               precision);
        };

        TT_THROW_IF_ERROR(
            (DispatchOpOut<1>(std::move(op_builder), {self}, out,
                              {.out_dtype = element_type,
                               .out_dims = CopyIntVector(out.sizes()),
                               .op_param_cache_keys = std::move(param_keys)})));
        return out;
      });
}

}  // namespace torch_tpu
