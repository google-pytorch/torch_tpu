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

#include "csrc/ops/upsample/upsample_aten_kernels.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/native/UpSample.h"
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "c10/util/string_view.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/einsum/einsum_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/precision_context.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

enum class UpsampleMode {
  // Nearest mode uses the floor of the scaled coordinate.
  kNearest,
  // Nearest exact mode aligns the centers of the pixels by adding 0.5 before
  // scaling and applying the floor function.
  kNearestExact,
};

constexpr int64_t kBilinearHeightDim = 2;
constexpr int64_t kBilinearWidthDim = 3;

c10::string_view GetSpatialEinsumEquation(int64_t rank, int64_t spatial_dim,
                                          bool is_backward) {
  static constexpr c10::string_view kEquations[3][3][2] = {
      // rank 3 (N, C, W)
      {{"ncw,ow->nco", "nco,ow->ncw"}, {}, {}},
      // rank 4 (N, C, H, W)
      {{"nchw,oh->ncow", "ncow,oh->nchw"},
       {"nchw,ow->ncho", "ncho,ow->nchw"},
       {}},
      // rank 5 (N, C, D, H, W)
      {{"ncdhw,od->ncohw", "ncohw,od->ncdhw"},
       {"ncdhw,oh->ncdow", "ncdow,oh->ncdhw"},
       {"ncdhw,ow->ncdho", "ncdho,ow->ncdhw"}},
  };
  ABSL_DCHECK(rank >= 3 && rank <= 5);
  ABSL_DCHECK(spatial_dim >= 2 && spatial_dim < rank);
  return kEquations[rank - 3][spatial_dim - 2][is_backward ? 1 : 0];
}

// Inspired by `jax.image.scale` (`jax/_src/image/scale.py`),
// builds the [out_size, in_size] 1D linear interpolation weight matrix
//   A[out_idx, in_idx] =
//       lambda0[out_idx] * (in_idx == index0[out_idx]) +
//       lambda1[out_idx] * (in_idx == index1[out_idx])
// directly in StableHLO from `Iota` and scalar splat constants.
mlir::MlirOp BuildInterpolationMatrix(mlir::MlirBuilder& builder,
                                      int64_t in_size, int64_t out_size,
                                      bool align_corners,
                                      std::optional<double> scale,
                                      mlir::Type calc_type) {
  const double ratio =
      calc_type.isF64()
          ? at::native::area_pixel_compute_scale<double>(in_size, out_size,
                                                         align_corners, scale)
          : static_cast<double>(at::native::area_pixel_compute_scale<float>(
                in_size, out_size, align_corners, scale));

  const mlir::Type i32_type = builder.getOpBuilder().getI32Type();
  const mlir::RankedTensorType weight_vec_type =
      mlir::RankedTensorType::get({out_size}, calc_type);
  const mlir::RankedTensorType index_mat_type =
      mlir::RankedTensorType::get({out_size, in_size}, i32_type);
  const mlir::RankedTensorType weight_mat_type =
      mlir::RankedTensorType::get({out_size, in_size}, calc_type);

  mlir::MlirOp dst_idx =
      mlir::stablehlo::Iota(builder, weight_vec_type, /*iotaDimension=*/0);
  mlir::MlirOp ratio_vec = MakeConstant(builder, ratio, calc_type, out_size);
  mlir::MlirOp zero_vec = MakeConstant(builder, 0.0, calc_type, out_size);
  mlir::MlirOp one_vec = MakeConstant(builder, 1.0, calc_type, out_size);
  mlir::MlirOp src_idx;
  if (align_corners) {
    src_idx = mlir::stablehlo::Mul(dst_idx, ratio_vec);
  } else {
    mlir::MlirOp half_vec = MakeConstant(builder, 0.5, calc_type, out_size);
    mlir::MlirOp centered = mlir::stablehlo::Add(dst_idx, half_vec);
    mlir::MlirOp scaled = mlir::stablehlo::Mul(centered, ratio_vec);
    mlir::MlirOp unshifted = mlir::stablehlo::Subtract(scaled, half_vec);
    src_idx = mlir::stablehlo::Max(unshifted, zero_vec);
  }

  const int64_t max_in_idx = in_size - 1;
  mlir::MlirOp max_in_float = MakeConstant(
      builder, static_cast<double>(max_in_idx), calc_type, out_size);
  mlir::MlirOp floor_src = mlir::stablehlo::Floor(src_idx);
  mlir::MlirOp idx0_float = mlir::stablehlo::Min(floor_src, max_in_float);
  mlir::MlirOp frac = mlir::stablehlo::Subtract(src_idx, idx0_float);
  mlir::MlirOp lambda1 = mlir::stablehlo::Clamp(zero_vec, frac, one_vec);
  mlir::MlirOp lambda0 = mlir::stablehlo::Subtract(one_vec, lambda1);

  mlir::MlirOp index0 =
      mlir::stablehlo::ConvertElementType(idx0_float, i32_type);
  mlir::MlirOp one_i32 = MakeConstant(builder, 1, i32_type, out_size);
  mlir::MlirOp max_in_i32 =
      MakeConstant(builder, max_in_idx, i32_type, out_size);
  mlir::MlirOp index0_plus_one = mlir::stablehlo::Add(index0, one_i32);
  mlir::MlirOp index1 = mlir::stablehlo::Min(index0_plus_one, max_in_i32);

  mlir::MlirOp col_indices =
      mlir::stablehlo::Iota(builder, index_mat_type, /*iotaDimension=*/1);
  mlir::MlirOp zeros =
      MakeConstant(builder, 0.0, calc_type, {out_size, in_size});

  auto build_term = [col_indices, zeros, index_mat_type, weight_mat_type](
                        mlir::MlirOp index_vec,
                        mlir::MlirOp weight_vec) mutable {
    mlir::MlirOp index_mat =
        mlir::stablehlo::BroadcastInDim(index_mat_type, index_vec, {0});
    mlir::MlirOp weight_mat =
        mlir::stablehlo::BroadcastInDim(weight_mat_type, weight_vec, {0});
    mlir::MlirOp is_tap = mlir::stablehlo::Compare(
        col_indices, index_mat, mlir::stablehlo::ComparisonDirection::EQ);
    return mlir::stablehlo::Select(is_tap, weight_mat, zeros);
  };

  mlir::MlirOp term0 = build_term(index0, lambda0);
  mlir::MlirOp term1 = build_term(index1, lambda1);
  return mlir::stablehlo::Add(term0, term1);
}

// Interpolates `operand` along `spatial_dim` (H=2 or W=3) by contracting it
// with the [out_size, in_size] 1D weight matrix (or its transpose when
// `is_backward` is true).
absl::StatusOr<mlir::MlirOp> InterpolateSpatialDim(
    mlir::MlirOp operand, int64_t in_size, int64_t out_size, bool align_corners,
    std::optional<double> scale, int64_t spatial_dim,
    mlir::stablehlo::Precision precision, bool is_backward) {
  ABSL_DCHECK(spatial_dim == kBilinearHeightDim ||
              spatial_dim == kBilinearWidthDim);
  ABSL_DCHECK_EQ(GetTensorTypeOrDie(operand).getRank(), 4);
  if (in_size == out_size) {
    return operand;
  }
  mlir::MlirBuilder& builder = operand.getBuilder();
  const mlir::Type calc_type = GetTensorTypeOrDie(operand).getElementType();
  mlir::MlirOp matrix = BuildInterpolationMatrix(
      builder, in_size, out_size, align_corners, scale, calc_type);
  const c10::string_view equation =
      GetSpatialEinsumEquation(/*rank=*/4, spatial_dim, is_backward);
  return BuildEinsumShlo({operand, matrix}, equation, precision);
}

// Builds the 2D bilinear interpolation (or its adjoint when `is_backward` is
// true) of 4D NCHW `operand`. `in_h`/`in_w` and `out_h`/`out_w` are always the
// forward input and output spatial sizes.
absl::StatusOr<mlir::MlirOp> BuildBilinear2d(
    mlir::MlirOp operand, int64_t in_h, int64_t in_w, int64_t out_h,
    int64_t out_w, bool align_corners, std::optional<double> scale_h,
    std::optional<double> scale_w, mlir::stablehlo::Precision precision,
    bool is_backward) {
  if (in_h == out_h && in_w == out_w) {
    return operand;
  }

  mlir::MlirBuilder& builder = operand.getBuilder();
  const mlir::Type elem_type = GetTensorTypeOrDie(operand).getElementType();
  // Matches ATen's `at::opmath_type<scalar_t>` (`ATen/OpMathType.h`) and
  // `UpSampleKernel.cpp`, which promote narrow floating-point types (f16,
  // bf16, f8) and integer types (uint8) to f32 (and keep f64 as f64) for
  // intermediate interpolation and backward gradient accumulation.
  const mlir::Type calc_type = elem_type.isF64()
                                   ? builder.getOpBuilder().getF64Type()
                                   : builder.getOpBuilder().getF32Type();
  mlir::MlirOp result =
      elem_type == calc_type
          ? operand
          : mlir::stablehlo::ConvertElementType(operand, calc_type);

  TT_ASSIGN_OR_RETURN(result, InterpolateSpatialDim(
                                  result, in_h, out_h, align_corners, scale_h,
                                  kBilinearHeightDim, precision, is_backward));
  TT_ASSIGN_OR_RETURN(
      result, InterpolateSpatialDim(result, in_w, out_w, align_corners, scale_w,
                                    kBilinearWidthDim, precision, is_backward));

  return elem_type == calc_type
             ? result
             : mlir::stablehlo::ConvertElementType(result, elem_type);
}

bool IsIdentityNearestAxis(int64_t in_size, int64_t out_size,
                           std::optional<double> scale) {
  return in_size == out_size && (!scale.has_value() || *scale == 1.0);
}

// Builds the [out_size, in_size] 1D 0/1 indicator matrix
//   W[out_idx, in_idx] = (in_idx == src_idx(out_idx))
// for nearest or nearest-exact interpolation directly in StableHLO.
mlir::MlirOp BuildNearestInterpolationMatrix(mlir::MlirBuilder& builder,
                                             int64_t in_size, int64_t out_size,
                                             std::optional<double> scale,
                                             UpsampleMode mode,
                                             mlir::Type calc_type) {
  const float inv_scale =
      at::native::compute_scales_value<float>(scale, in_size, out_size);

  const mlir::Type f32_type = builder.getOpBuilder().getF32Type();
  const mlir::Type i32_type = builder.getOpBuilder().getI32Type();
  const mlir::RankedTensorType f32_vec_type =
      mlir::RankedTensorType::get({out_size}, f32_type);
  const mlir::RankedTensorType index_mat_type =
      mlir::RankedTensorType::get({out_size, in_size}, i32_type);

  mlir::MlirOp dst_idx =
      mlir::stablehlo::Iota(builder, f32_vec_type, /*iotaDimension=*/0);
  if (mode == UpsampleMode::kNearestExact) {
    mlir::MlirOp half_vec = MakeConstant(builder, 0.5f, f32_type, out_size);
    dst_idx = mlir::stablehlo::Add(dst_idx, half_vec);
  }
  mlir::MlirOp inv_scale_vec =
      MakeConstant(builder, inv_scale, f32_type, out_size);
  mlir::MlirOp scaled_idx = mlir::stablehlo::Mul(dst_idx, inv_scale_vec);
  mlir::MlirOp floor_idx = mlir::stablehlo::Floor(scaled_idx);
  mlir::MlirOp raw_idx =
      mlir::stablehlo::ConvertElementType(floor_idx, i32_type);

  mlir::MlirOp zero_i32 = MakeConstant(builder, 0, i32_type, out_size);
  mlir::MlirOp max_in_i32 =
      MakeConstant(builder, in_size - 1, i32_type, out_size);
  mlir::MlirOp src_idx = mlir::stablehlo::Clamp(zero_i32, raw_idx, max_in_i32);

  mlir::MlirOp index_mat =
      mlir::stablehlo::BroadcastInDim(index_mat_type, src_idx, {0});
  mlir::MlirOp col_indices =
      mlir::stablehlo::Iota(builder, index_mat_type, /*iotaDimension=*/1);
  mlir::MlirOp is_match = mlir::stablehlo::Compare(
      col_indices, index_mat, mlir::stablehlo::ComparisonDirection::EQ);
  return mlir::stablehlo::ConvertElementType(is_match, calc_type);
}

absl::StatusOr<mlir::MlirOp> InterpolateNearestSpatialDim(
    mlir::MlirOp operand, int64_t in_size, int64_t out_size,
    std::optional<double> scale, int64_t spatial_dim, int64_t rank,
    UpsampleMode mode, mlir::stablehlo::Precision precision, bool is_backward) {
  if (IsIdentityNearestAxis(in_size, out_size, scale)) {
    return operand;
  }
  mlir::MlirBuilder& builder = operand.getBuilder();
  const mlir::Type calc_type = GetTensorTypeOrDie(operand).getElementType();
  mlir::MlirOp matrix = BuildNearestInterpolationMatrix(
      builder, in_size, out_size, scale, mode, calc_type);
  const c10::string_view equation =
      GetSpatialEinsumEquation(rank, spatial_dim, is_backward);
  return BuildEinsumShlo({operand, matrix}, equation, precision);
}

// Builds 1D, 2D, or 3D nearest / nearest-exact interpolation (or its backward
// adjoint when `is_backward` is true) via separable 1D contractions.
absl::StatusOr<mlir::MlirOp> BuildNearest(
    mlir::MlirOp operand, llvm::ArrayRef<int64_t> in_spatial,
    llvm::ArrayRef<int64_t> out_spatial,
    llvm::ArrayRef<std::optional<double>> scales, UpsampleMode mode,
    mlir::stablehlo::Precision precision, bool is_backward) {
  ABSL_DCHECK_EQ(in_spatial.size(), out_spatial.size());
  ABSL_DCHECK_EQ(in_spatial.size(), scales.size());

  bool all_identity = true;
  for (size_t i = 0; i < in_spatial.size(); ++i) {
    if (!IsIdentityNearestAxis(in_spatial[i], out_spatial[i], scales[i])) {
      all_identity = false;
      break;
    }
  }
  if (all_identity) {
    return operand;
  }

  mlir::MlirBuilder& builder = operand.getBuilder();
  const mlir::Type elem_type = GetTensorTypeOrDie(operand).getElementType();
  const mlir::Type calc_type = elem_type.isF64()
                                   ? builder.getOpBuilder().getF64Type()
                                   : builder.getOpBuilder().getF32Type();
  mlir::MlirOp result =
      elem_type == calc_type
          ? operand
          : mlir::stablehlo::ConvertElementType(operand, calc_type);

  const int64_t rank = GetTensorTypeOrDie(operand).getRank();
  for (size_t i = 0; i < in_spatial.size(); ++i) {
    TT_ASSIGN_OR_RETURN(
        result, InterpolateNearestSpatialDim(
                    result, in_spatial[i], out_spatial[i], scales[i],
                    /*spatial_dim=*/static_cast<int64_t>(i + 2), rank, mode,
                    precision, is_backward));
  }

  return elem_type == calc_type
             ? result
             : mlir::stablehlo::ConvertElementType(result, elem_type);
}

void CheckUpsampleDtypes(const at::Tensor& tensor) {
  TT_CHECK_THROW(tensor.scalar_type() != at::kBool &&
                     tensor.scalar_type() != at::kInt &&
                     tensor.scalar_type() != at::kLong,
                 error::kPythonNotImplementedError)
      << "not implemented for " << ToString(tensor.scalar_type());
}

}  // namespace

at::Tensor& AtenUpsampleNearest1dBackwardGradInput(
    const at::Tensor& grad_output, at::IntArrayRef output_shape_in,
    at::IntArrayRef input_shape, std::optional<double> scale,
    at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kUpsampleNearest1dBackwardGradInput, param_keys,
      (grad_output, output_shape_in, input_shape, scale, grad_input), {
        TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<mlir::ElementType>(
                                                  grad_input.scalar_type()));
        const mlir::stablehlo::Precision precision =
            GetAndAddPrecisionWithDefaultTo(
                param_keys,
                mlir::stablehlo::Precision::
                    HIGHEST);  // EXPLICIT_PRECISION_OK=nearest

        auto op_builder =
            [in_w = input_shape[2], out_w = output_shape_in[0], scale,
             precision](mlir::MlirOp grad_out) -> absl::StatusOr<mlir::MlirOp> {
          return BuildNearest(grad_out, {in_w}, {out_w}, {scale},
                              UpsampleMode::kNearest, precision,
                              /*is_backward=*/true);
        };

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<1>(std::move(op_builder), {grad_output},
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
        TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<mlir::ElementType>(
                                                  grad_input.scalar_type()));
        const mlir::stablehlo::Precision precision =
            GetAndAddPrecisionWithDefaultTo(
                param_keys,
                mlir::stablehlo::Precision::
                    HIGHEST);  // EXPLICIT_PRECISION_OK=nearest

        auto op_builder =
            [in_h = input_shape[2], in_w = input_shape[3],
             out_h = output_shape_in[0], out_w = output_shape_in[1], scale_h,
             scale_w,
             precision](mlir::MlirOp grad_out) -> absl::StatusOr<mlir::MlirOp> {
          return BuildNearest(grad_out, {in_h, in_w}, {out_h, out_w},
                              {scale_h, scale_w}, UpsampleMode::kNearest,
                              precision, /*is_backward=*/true);
        };

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<1>(std::move(op_builder), {grad_output},
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
        TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<mlir::ElementType>(
                                                  grad_input.scalar_type()));
        const mlir::stablehlo::Precision precision =
            GetAndAddPrecisionWithDefaultTo(
                param_keys,
                mlir::stablehlo::Precision::
                    HIGHEST);  // EXPLICIT_PRECISION_OK=nearest

        auto op_builder =
            [in_d = input_shape[2], in_h = input_shape[3],
             in_w = input_shape[4], out_d = output_shape_in[0],
             out_h = output_shape_in[1], out_w = output_shape_in[2], scale_h,
             scale_w, scale_d,
             precision](mlir::MlirOp grad_out) -> absl::StatusOr<mlir::MlirOp> {
          return BuildNearest(
              grad_out, {in_d, in_h, in_w}, {out_d, out_h, out_w},
              {scale_h, scale_w, scale_d}, UpsampleMode::kNearest, precision,
              /*is_backward=*/true);
        };

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<1>(std::move(op_builder), {grad_output},
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
        TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<mlir::ElementType>(
                                                  grad_input.scalar_type()));
        const mlir::stablehlo::Precision precision =
            GetAndAddPrecisionWithDefaultTo(
                param_keys,
                mlir::stablehlo::Precision::
                    HIGHEST);  // EXPLICIT_PRECISION_OK=nearest_exact

        auto op_builder =
            [in_w = input_shape[2], out_w = output_shape_in[0], scale,
             precision](mlir::MlirOp grad_out) -> absl::StatusOr<mlir::MlirOp> {
          return BuildNearest(grad_out, {in_w}, {out_w}, {scale},
                              UpsampleMode::kNearestExact, precision,
                              /*is_backward=*/true);
        };

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<1>(std::move(op_builder), {grad_output},
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
        TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<mlir::ElementType>(
                                                  grad_input.scalar_type()));
        const mlir::stablehlo::Precision precision =
            GetAndAddPrecisionWithDefaultTo(
                param_keys,
                mlir::stablehlo::Precision::
                    HIGHEST);  // EXPLICIT_PRECISION_OK=nearest_exact

        auto op_builder =
            [in_h = input_shape[2], in_w = input_shape[3],
             out_h = output_shape_in[0], out_w = output_shape_in[1], scale_h,
             scale_w,
             precision](mlir::MlirOp grad_out) -> absl::StatusOr<mlir::MlirOp> {
          return BuildNearest(grad_out, {in_h, in_w}, {out_h, out_w},
                              {scale_h, scale_w}, UpsampleMode::kNearestExact,
                              precision, /*is_backward=*/true);
        };

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<1>(std::move(op_builder), {grad_output},
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
        TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<mlir::ElementType>(
                                                  grad_input.scalar_type()));
        const mlir::stablehlo::Precision precision =
            GetAndAddPrecisionWithDefaultTo(
                param_keys,
                mlir::stablehlo::Precision::
                    HIGHEST);  // EXPLICIT_PRECISION_OK=nearest_exact

        auto op_builder =
            [in_d = input_shape[2], in_h = input_shape[3],
             in_w = input_shape[4], out_d = output_shape_in[0],
             out_h = output_shape_in[1], out_w = output_shape_in[2], scale_h,
             scale_w, scale_d,
             precision](mlir::MlirOp grad_out) -> absl::StatusOr<mlir::MlirOp> {
          return BuildNearest(grad_out, {in_d, in_h, in_w},
                              {out_d, out_h, out_w},
                              {scale_h, scale_w, scale_d},
                              UpsampleMode::kNearestExact, precision,
                              /*is_backward=*/true);
        };

        TT_ASSIGN_OR_THROW(
            auto grad_input_buf,
            (DispatchOp<1>(std::move(op_builder), {grad_output},
                           {.out_dtype = element_type,
                            .out_dims = CopyIntVector(input_shape),
                            .op_param_cache_keys = std::move(param_keys)})));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(grad_input_buf), grad_input));
        return grad_input;
      });
}

// Lowers 2D bilinear interpolation as separable contractions along H and W:
// Y = A_h * X * A_w^T.
at::Tensor& AtenUpsampleBilinear2dOut(const at::Tensor& self,
                                      at::IntArrayRef upsample_shape,
                                      bool align_corners,
                                      std::optional<double> scale_h,
                                      std::optional<double> scale_w,
                                      at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleBilinear2dOut, param_keys,
      (self, upsample_shape, align_corners, scale_h, scale_w, out), {
        CheckUpsampleDtypes(self);
        TT_CHECK_THROW(self.dim() == 4 && out.dim() == 4,
                       error::kInvalidArgument)
            << "expected 4D input and out tensors, got input: "
            << ToString(self.dim()) << "D, out: " << ToString(out.dim()) << "D";
        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        const mlir::stablehlo::Precision precision =
            GetAndAddPrecisionWithDefaultTo(
                param_keys,
                mlir::stablehlo::Precision::
                    HIGHEST);  // EXPLICIT_PRECISION_OK=bilinear

        auto op_builder =
            [in_h = self.size(kBilinearHeightDim),
             in_w = self.size(kBilinearWidthDim),
             out_h = out.size(kBilinearHeightDim),
             out_w = out.size(kBilinearWidthDim), align_corners, scale_h,
             scale_w,
             precision](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildBilinear2d(input, in_h, in_w, out_h, out_w, align_corners,
                                 scale_h, scale_w, precision,
                                 /*is_backward=*/false);
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

// Lowers 2D bilinear interpolation backward as the adjoint separable
// contractions along H and W: dX = A_h^T * dY * A_w.
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
        TT_CHECK_THROW(grad_output.dim() == 4 && input_size.size() == 4,
                       error::kInvalidArgument)
            << "expected 4D grad_output tensor and 4-element input_size, got "
               "grad_output: "
            << ToString(grad_output.dim())
            << "D, input_size: " << ToString(input_size.size()) << " elements";
        TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<mlir::ElementType>(
                                                  grad_input.scalar_type()));
        const mlir::stablehlo::Precision precision =
            GetAndAddPrecisionWithDefaultTo(
                param_keys,
                mlir::stablehlo::Precision::
                    HIGHEST);  // EXPLICIT_PRECISION_OK=bilinear
        auto op_builder =
            [in_h = input_size[kBilinearHeightDim],
             in_w = input_size[kBilinearWidthDim],
             out_h = grad_output.size(kBilinearHeightDim),
             out_w = grad_output.size(kBilinearWidthDim), align_corners,
             scales_h, scales_w,
             precision](mlir::MlirOp grad_out) -> absl::StatusOr<mlir::MlirOp> {
          return BuildBilinear2d(grad_out, in_h, in_w, out_h, out_w,
                                 align_corners, scales_h, scales_w, precision,
                                 /*is_backward=*/true);
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
        CheckUpsampleDtypes(self);
        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        const mlir::stablehlo::Precision precision =
            GetAndAddPrecisionWithDefaultTo(
                param_keys,
                mlir::stablehlo::Precision::
                    HIGHEST);  // EXPLICIT_PRECISION_OK=nearest

        auto op_builder =
            [in_w = self.size(2), out_w = upsample_shape[0], scale,
             precision](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildNearest(input, {in_w}, {out_w}, {scale},
                              UpsampleMode::kNearest, precision,
                              /*is_backward=*/false);
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

at::Tensor& AtenUpsampleNearest2dOut(const at::Tensor& self,
                                     at::IntArrayRef upsample_shape,
                                     std::optional<double> scale_h,
                                     std::optional<double> scale_w,
                                     at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleNearest2dOut, param_keys,
      (self, upsample_shape, scale_h, scale_w, out), {
        CheckUpsampleDtypes(self);
        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        const mlir::stablehlo::Precision precision =
            GetAndAddPrecisionWithDefaultTo(
                param_keys,
                mlir::stablehlo::Precision::
                    HIGHEST);  // EXPLICIT_PRECISION_OK=nearest

        auto op_builder =
            [in_h = self.size(2), in_w = self.size(3),
             out_h = upsample_shape[0], out_w = upsample_shape[1], scale_h,
             scale_w,
             precision](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildNearest(input, {in_h, in_w}, {out_h, out_w},
                              {scale_h, scale_w}, UpsampleMode::kNearest,
                              precision, /*is_backward=*/false);
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

at::Tensor& AtenUpsampleNearest3dOut(const at::Tensor& self,
                                     at::IntArrayRef upsample_shape,
                                     std::optional<double> scale_h,
                                     std::optional<double> scale_w,
                                     std::optional<double> scale_d,
                                     at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleNearest3dOut, param_keys,
      (self, upsample_shape, scale_h, scale_w, scale_d, out), {
        CheckUpsampleDtypes(self);
        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        const mlir::stablehlo::Precision precision =
            GetAndAddPrecisionWithDefaultTo(
                param_keys,
                mlir::stablehlo::Precision::
                    HIGHEST);  // EXPLICIT_PRECISION_OK=nearest

        auto op_builder =
            [in_d = self.size(2), in_h = self.size(3), in_w = self.size(4),
             out_d = upsample_shape[0], out_h = upsample_shape[1],
             out_w = upsample_shape[2], scale_h, scale_w, scale_d,
             precision](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildNearest(input, {in_d, in_h, in_w}, {out_d, out_h, out_w},
                              {scale_h, scale_w, scale_d},
                              UpsampleMode::kNearest, precision,
                              /*is_backward=*/false);
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

at::Tensor& AtenUpsampleNearestExact1dOut(const at::Tensor& self,
                                          at::IntArrayRef upsample_shape,
                                          std::optional<double> scale,
                                          at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleNearestExact1dOut, param_keys,
      (self, upsample_shape, scale, out), {
        CheckUpsampleDtypes(self);
        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        const mlir::stablehlo::Precision precision =
            GetAndAddPrecisionWithDefaultTo(
                param_keys,
                mlir::stablehlo::Precision::
                    HIGHEST);  // EXPLICIT_PRECISION_OK=nearest_exact

        auto op_builder =
            [in_w = self.size(2), out_w = upsample_shape[0], scale,
             precision](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildNearest(input, {in_w}, {out_w}, {scale},
                              UpsampleMode::kNearestExact, precision,
                              /*is_backward=*/false);
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

at::Tensor& AtenUpsampleNearestExact2dOut(const at::Tensor& self,
                                          at::IntArrayRef upsample_shape,
                                          std::optional<double> scale_h,
                                          std::optional<double> scale_w,
                                          at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleNearestExact2dOut, param_keys,
      (self, upsample_shape, scale_h, scale_w, out), {
        CheckUpsampleDtypes(self);
        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        const mlir::stablehlo::Precision precision =
            GetAndAddPrecisionWithDefaultTo(
                param_keys,
                mlir::stablehlo::Precision::
                    HIGHEST);  // EXPLICIT_PRECISION_OK=nearest_exact

        auto op_builder =
            [in_h = self.size(2), in_w = self.size(3),
             out_h = upsample_shape[0], out_w = upsample_shape[1], scale_h,
             scale_w,
             precision](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildNearest(input, {in_h, in_w}, {out_h, out_w},
                              {scale_h, scale_w}, UpsampleMode::kNearestExact,
                              precision, /*is_backward=*/false);
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

at::Tensor& AtenUpsampleNearestExact3dOut(const at::Tensor& self,
                                          at::IntArrayRef upsample_shape,
                                          std::optional<double> scale_h,
                                          std::optional<double> scale_w,
                                          std::optional<double> scale_d,
                                          at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleNearestExact3dOut, param_keys,
      (self, upsample_shape, scale_h, scale_w, scale_d, out), {
        CheckUpsampleDtypes(self);
        TT_ASSIGN_OR_THROW(auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        const mlir::stablehlo::Precision precision =
            GetAndAddPrecisionWithDefaultTo(
                param_keys,
                mlir::stablehlo::Precision::
                    HIGHEST);  // EXPLICIT_PRECISION_OK=nearest_exact

        auto op_builder =
            [in_d = self.size(2), in_h = self.size(3), in_w = self.size(4),
             out_d = upsample_shape[0], out_h = upsample_shape[1],
             out_w = upsample_shape[2], scale_h, scale_w, scale_d,
             precision](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildNearest(input, {in_d, in_h, in_w}, {out_d, out_h, out_w},
                              {scale_h, scale_w, scale_d},
                              UpsampleMode::kNearestExact, precision,
                              /*is_backward=*/false);
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

}  // namespace torch_tpu
