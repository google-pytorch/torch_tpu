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
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/binary.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

enum class AlignCorners : bool {
  kNo = false,
  kYes = true,
};

enum class SpatialAxis : int64_t {
  kHeight = 2,
  kWidth = 3,
};

struct Aa1dFilterParams {
  double scale;
  double support;
  double invscale;
  int64_t interp_size;
};

struct Aa1dSpanInfo {
  mlir::MlirOp center;
  mlir::MlirOp xmin;
  mlir::MlirOp xsize;
};

struct Aa1dTapInfo {
  mlir::MlirOp clamped_indices;
  mlir::MlirOp weight;
};

struct Aa1dGatherConfig {
  mlir::stablehlo::GatherDimensionNumbersAttr gather_dimension_numbers;
  Dimensions slice_sizes;
};

mlir::MlirOp Clamp(mlir::MlirOp min, mlir::MlirOp operand, mlir::MlirOp max) {
  return mlir::stablehlo::Clamp(min, operand, max);
}

mlir::MlirOp Gather(
    mlir::MlirOp operand, mlir::MlirOp start_indices,
    mlir::stablehlo::GatherDimensionNumbersAttr dimension_numbers,
    llvm::ArrayRef<int64_t> slice_sizes) {
  return mlir::stablehlo::Gather(operand, start_indices, dimension_numbers,
                                 slice_sizes, /*indices_are_sorted=*/false);
}

mlir::MlirOp Reshape(mlir::Type result_type, mlir::MlirOp operand) {
  return mlir::stablehlo::Reshape(result_type, operand);
}

mlir::MlirOp BroadcastInDim(mlir::Type result_type, mlir::MlirOp operand,
                            llvm::ArrayRef<int64_t> broadcast_dimensions) {
  return mlir::stablehlo::BroadcastInDim(result_type, operand,
                                         broadcast_dimensions);
}

Aa1dGatherConfig GetAa1dGatherConfig(mlir::MlirOp input, SpatialAxis axis) {
  const Dimensions input_shape =
      CopyIntVector(GetTensorTypeOrDie(input).getShape());
  const auto axis_dim = static_cast<int64_t>(axis);
  Dimensions slice_sizes = input_shape;
  slice_sizes[axis_dim] = 1;

  const Dimensions offset_dims =
      (axis == SpatialAxis::kWidth) ? Dimensions{0, 1, 2} : Dimensions{0, 1, 3};
  const Dimensions collapsed_slice_dims = {axis_dim};
  const Dimensions start_index_map = {axis_dim};

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

double ComputeAreaPixelScale(int64_t in_size, int64_t out_size,
                             AlignCorners align_corners,
                             std::optional<double> scale_opt,
                             mlir::Type calc_type) {
  double scale = 0.0;
  if (align_corners == AlignCorners::kYes) {
    if (out_size > 1) {
      scale =
          static_cast<double>(in_size - 1) / static_cast<double>(out_size - 1);
    }
  } else if (scale_opt.has_value() && *scale_opt > 0.0) {
    scale = 1.0 / *scale_opt;
  } else {
    scale = static_cast<double>(in_size) / static_cast<double>(out_size);
  }
  if (!calc_type.isF64()) {
    scale = static_cast<double>(static_cast<float>(scale));
  }
  return scale;
}

Aa1dFilterParams ComputeAa1dFilterParams(int64_t in_size, int64_t out_size,
                                         AlignCorners align_corners,
                                         std::optional<double> scale_opt,
                                         mlir::Type calc_type) {
  const double scale = ComputeAreaPixelScale(in_size, out_size, align_corners,
                                             scale_opt, calc_type);
  const double support = (scale >= 1.0) ? scale : 1.0;
  double invscale = 1.0;
  if (scale >= 1.0) {
    invscale = calc_type.isF64()
                   ? (1.0 / scale)
                   : static_cast<double>(1.0f / static_cast<float>(scale));
  }
  const int64_t interp_size = 2 * static_cast<int64_t>(std::ceil(support)) + 1;
  return {scale, support, invscale, interp_size};
}

absl::StatusOr<Aa1dSpanInfo> ComputeAa1dSpanShlo(mlir::MlirBuilder& builder,
                                                 int64_t in_size,
                                                 int64_t out_size,
                                                 const Aa1dFilterParams& params,
                                                 mlir::Type calc_type) {
  const auto iota_type = mlir::RankedTensorType::get({out_size}, calc_type);
  const auto iota = mlir::stablehlo::Iota(builder, iota_type, 0);
  const auto half_const = MakeConstantLike(iota, 0.5);
  const auto scale_const = MakeConstantLike(iota, params.scale);
  const auto support_const = MakeConstantLike(iota, params.support);

  TT_ASSIGN_OR_RETURN(const auto iota_plus_half,
                      BuildAddShlo(iota, half_const));
  TT_ASSIGN_OR_RETURN(const auto center,
                      BuildMulShlo(iota_plus_half, scale_const));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp center_minus_support,
                      BuildSubShlo(center, support_const));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp xmin_shifted,
                      BuildAddShlo(center_minus_support, half_const));
  mlir::MlirOp xmin_floor = mlir::stablehlo::Floor(xmin_shifted);

  TT_ASSIGN_OR_RETURN(mlir::MlirOp center_plus_support,
                      BuildAddShlo(center, support_const));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp xmax_shifted,
                      BuildAddShlo(center_plus_support, half_const));
  mlir::MlirOp xmax_floor = mlir::stablehlo::Floor(xmax_shifted);

  const auto i32_type = builder.getOpBuilder().getI32Type();
  mlir::MlirOp xmin_raw =
      mlir::stablehlo::ConvertElementType(xmin_floor, i32_type);
  mlir::MlirOp xmax_raw =
      mlir::stablehlo::ConvertElementType(xmax_floor, i32_type);

  mlir::MlirOp zero_i32 = MakeConstantLike(xmin_raw, 0);
  mlir::MlirOp in_size_i32 =
      MakeConstantLike(xmin_raw, static_cast<int32_t>(in_size));
  mlir::MlirOp interp_size_i32 =
      MakeConstantLike(xmin_raw, static_cast<int32_t>(params.interp_size));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp xmin, BuildMaximumShlo(xmin_raw, zero_i32));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp xmax,
                      BuildMinimumShlo(xmax_raw, in_size_i32));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp raw_xsize, BuildSubShlo(xmax, xmin));
  mlir::MlirOp xsize = Clamp(zero_i32, raw_xsize, interp_size_i32);

  return Aa1dSpanInfo{center, xmin, xsize};
}

absl::StatusOr<Aa1dTapInfo> ComputeAa1dTapInfo(int64_t tap_idx, int64_t in_size,
                                               int64_t out_size,
                                               const Aa1dFilterParams& params,
                                               Aa1dSpanInfo span) {
  mlir::MlirOp j_i32 =
      MakeConstantLike(span.xmin, static_cast<int32_t>(tap_idx));
  mlir::MlirOp xsize = span.xsize;
  mlir::MlirOp is_valid = mlir::stablehlo::Compare(
      j_i32, xsize, mlir::stablehlo::ComparisonDirection::LT);

  TT_ASSIGN_OR_RETURN(mlir::MlirOp raw_idx_i32, BuildAddShlo(span.xmin, j_i32));
  mlir::MlirOp zero_i32 = MakeConstantLike(span.xmin, 0);
  mlir::MlirOp max_idx_i32 = MakeConstantLike(
      span.xmin, static_cast<int32_t>(std::max<int64_t>(0, in_size - 1)));
  mlir::MlirOp clamped_idx_i32 = Clamp(zero_i32, raw_idx_i32, max_idx_i32);

  const auto i32_type = GetTensorTypeOrDie(span.xmin).getElementType();
  const auto index_tensor_type =
      mlir::RankedTensorType::get({out_size, 1}, i32_type);
  mlir::MlirOp clamped_indices = Reshape(index_tensor_type, clamped_idx_i32);

  const auto calc_type = GetTensorTypeOrDie(span.center).getElementType();
  mlir::MlirOp j_plus_xmin_calc =
      mlir::stablehlo::ConvertElementType(raw_idx_i32, calc_type);
  mlir::MlirOp half_calc = MakeConstantLike(span.center, 0.5);
  mlir::MlirOp one_calc = MakeConstantLike(span.center, 1.0);
  mlir::MlirOp zero_calc = MakeConstantLike(span.center, 0.0);
  mlir::MlirOp invscale_calc = MakeConstantLike(span.center, params.invscale);

  TT_ASSIGN_OR_RETURN(mlir::MlirOp diff,
                      BuildSubShlo(j_plus_xmin_calc, span.center));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp shifted, BuildAddShlo(diff, half_calc));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp x, BuildMulShlo(shifted, invscale_calc));

  mlir::MlirOp abs_x = mlir::stablehlo::Abs(x);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp one_minus_abs_x,
                      BuildSubShlo(one_calc, abs_x));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp w_raw,
                      BuildMaximumShlo(zero_calc, one_minus_abs_x));
  mlir::MlirOp weight = mlir::stablehlo::Select(is_valid, w_raw, zero_calc);

  return Aa1dTapInfo{clamped_indices, weight};
}

absl::StatusOr<mlir::MlirOp> ComputeWeightedAa1dTapSlice(
    mlir::MlirOp input_calc, const Aa1dTapInfo& tap, mlir::MlirOp safe_total_w,
    const Aa1dGatherConfig& gather_config,
    mlir::RankedTensorType pass_calc_type, SpatialAxis axis) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp norm_w,
                      BuildDivShlo(tap.weight, safe_total_w));
  const auto axis_dim = static_cast<int64_t>(axis);
  mlir::MlirOp norm_w_bcast =
      BroadcastInDim(pass_calc_type, norm_w, {axis_dim});
  mlir::MlirOp gathered = Gather(
      input_calc, tap.clamped_indices, gather_config.gather_dimension_numbers,
      AsArrayRef<int64_t>(gather_config.slice_sizes));
  return BuildMulShlo(gathered, norm_w_bcast);
}

absl::StatusOr<mlir::MlirOp> InterpolateAa1dShlo(
    mlir::MlirOp input, int64_t out_size, SpatialAxis axis,
    AlignCorners align_corners, std::optional<double> scale_opt,
    mlir::Type calc_type) {
  mlir::MlirBuilder& builder = input.getBuilder();
  const auto input_type = GetTensorTypeOrDie(input);
  const Dimensions input_shape = CopyIntVector(input_type.getShape());
  const auto axis_dim = static_cast<int64_t>(axis);
  const int64_t in_size = input_shape[axis_dim];
  const auto element_type = input_type.getElementType();

  const Aa1dFilterParams params = ComputeAa1dFilterParams(
      in_size, out_size, align_corners, scale_opt, calc_type);
  TT_ASSIGN_OR_RETURN(
      const Aa1dSpanInfo span,
      ComputeAa1dSpanShlo(builder, in_size, out_size, params, calc_type));

  absl::InlinedVector<Aa1dTapInfo, 8> taps;
  taps.reserve(params.interp_size);
  mlir::MlirOp total_w = MakeConstantLike(span.center, 0.0);
  for (int64_t j = 0; j < params.interp_size; ++j) {
    TT_ASSIGN_OR_RETURN(const Aa1dTapInfo tap,
                        ComputeAa1dTapInfo(j, in_size, out_size, params, span));
    TT_ASSIGN_OR_RETURN(total_w, BuildAddShlo(total_w, tap.weight));
    taps.push_back(tap);
  }

  mlir::MlirOp zero_calc = MakeConstantLike(span.center, 0.0);
  mlir::MlirOp one_calc = MakeConstantLike(span.center, 1.0);
  mlir::MlirOp is_nonzero = mlir::stablehlo::Compare(
      total_w, zero_calc, mlir::stablehlo::ComparisonDirection::NE);
  mlir::MlirOp safe_total_w =
      mlir::stablehlo::Select(is_nonzero, total_w, one_calc);

  const auto input_calc = mlir::stablehlo::ConvertElementType(input, calc_type);
  Dimensions pass_output_shape = input_shape;
  pass_output_shape[axis_dim] = out_size;
  const auto pass_calc_type = mlir::RankedTensorType::get(
      AsArrayRef<int64_t>(pass_output_shape), calc_type);
  const Aa1dGatherConfig gather_config = GetAa1dGatherConfig(input, axis);

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp accumulated,
      ComputeWeightedAa1dTapSlice(input_calc, taps[0], safe_total_w,
                                  gather_config, pass_calc_type, axis));
  for (size_t j = 1; j < taps.size(); ++j) {
    TT_ASSIGN_OR_RETURN(
        const auto weighted_slice,
        ComputeWeightedAa1dTapSlice(input_calc, taps[j], safe_total_w,
                                    gather_config, pass_calc_type, axis));
    TT_ASSIGN_OR_RETURN(accumulated, BuildAddShlo(accumulated, weighted_slice));
  }

  return mlir::stablehlo::ConvertElementType(accumulated, element_type);
}

absl::StatusOr<mlir::MlirOp> BuildUpsampleBilinear2dAaShlo(
    mlir::MlirOp input, const Dimensions& output_shape,
    AlignCorners align_corners, std::optional<double> scales_h,
    std::optional<double> scales_w, mlir::Type calc_type) {
  const int64_t out_height = output_shape[2];
  const int64_t out_width = output_shape[3];

  TT_ASSIGN_OR_RETURN(const auto width_interpolated,
                      InterpolateAa1dShlo(input, out_width, SpatialAxis::kWidth,
                                          align_corners, scales_w, calc_type));
  return InterpolateAa1dShlo(width_interpolated, out_height,
                             SpatialAxis::kHeight, align_corners, scales_h,
                             calc_type);
}

void CheckUpsampleBilinear2dAaDtypes(const at::Tensor& tensor) {
  const at::ScalarType dtype = tensor.scalar_type();
  TT_CHECK_THROW(dtype == at::kFloat || dtype == at::kDouble ||
                     dtype == at::kBFloat16 || dtype == at::kHalf,
                 error::kPythonNotImplementedError)
      << "not implemented for " << ToString(dtype);
}

void ValidateUpsampleBilinear2dAaInputs(const at::Tensor& self,
                                        at::IntArrayRef output_size,
                                        const at::Tensor& out) {
  TT_CHECK_THROW(self.dim() == 4, error::kInvalidArgument)
      << "expected input tensor to be 4D (NCHW), got " << ToString(self.dim())
      << "D tensor with shape " << ToString(self.sizes());

  TT_CHECK_THROW(output_size.size() == 2, error::kInvalidArgument)
      << "expected output_size to have 2 elements, got "
      << ToString(output_size.size());

  TT_CHECK_THROW(output_size[0] > 0 && output_size[1] > 0,
                 error::kInvalidArgument)
      << "expected output_size dimensions to be positive, got "
      << ToString(output_size);

  TT_CHECK_THROW(self.size(1) > 0 && self.size(2) > 0 && self.size(3) > 0,
                 error::kInvalidArgument)
      << "expected input tensor to have non-zero size for non-batch "
         "dimensions, got "
      << ToString(self.sizes());

  CheckUpsampleBilinear2dAaDtypes(self);

  TT_CHECK_THROW(self.scalar_type() == out.scalar_type(),
                 error::kInvalidArgument)
      << "expected out tensor to have the same dtype as self ("
      << ToString(self.scalar_type()) << "), got "
      << ToString(out.scalar_type());
}

}  // namespace

at::Tensor& AtenUpsampleBilinear2dAaOut(const at::Tensor& self,
                                        at::IntArrayRef output_size,
                                        bool align_corners,
                                        std::optional<double> scales_h,
                                        std::optional<double> scales_w,
                                        at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleBilinear2dAaOut, param_keys,
      (self, output_size, align_corners, scales_h, scales_w, out), {
        ValidateUpsampleBilinear2dAaInputs(self, output_size, out);

        const Dimensions expected_output_shape = {
            self.size(0), self.size(1), output_size[0], output_size[1]};
        TT_THROW_IF_ERROR(
            ResizeTensorIfShapeDiffers(out, expected_output_shape));

        if (self.numel() == 0) {
          return out;
        }

        TT_ASSIGN_OR_THROW(const auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        const at::ScalarType real_dtype =
            c10::toRealValueType(self.scalar_type());

        const auto op_builder =
            [out_shape = CopyIntVector(out.sizes()),
             align_corners =
                 align_corners ? AlignCorners::kYes : AlignCorners::kNo,
             scales_h, scales_w,
             real_dtype](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          const mlir::Type calc_type =
              (real_dtype == at::kDouble)
                  ? input.getBuilder().getOpBuilder().getF64Type()
                  : input.getBuilder().getOpBuilder().getF32Type();
          return BuildUpsampleBilinear2dAaShlo(input, out_shape, align_corners,
                                               scales_h, scales_w, calc_type);
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
