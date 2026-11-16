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

#include "csrc/ops/upsample/upsample_bicubic2d_aten_kernels.h"

#include <algorithm>
#include <array>
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
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/einsum/einsum_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/precision_context.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

constexpr int64_t kBicubicHeightDim = 2;
constexpr int64_t kBicubicWidthDim = 3;

enum class AlignCorners {
  kNo,
  kYes,
};

enum class BicubicDirection {
  kForward,
  kBackward,
};

// Evaluates Keys' cubic convolution polynomial for |d| < 1 with A = -0.75
// (`at::native::cubic_convolution1` in `ATen/native/UpSample.h`):
//   W_near(d) = ((A + 2) * d - (A + 3)) * d * d + 1
//             = ((1.25 * d - 2.25) * d) * d + 1.0
mlir::MlirOp EvalCubicNear(mlir::MlirBuilder& builder, mlir::MlirOp d,
                           const mlir::Type calc_type, const int64_t out_size) {
  mlir::MlirOp c1_25 = MakeConstant(builder, 1.25, calc_type, out_size);
  mlir::MlirOp c2_25 = MakeConstant(builder, 2.25, calc_type, out_size);
  mlir::MlirOp c1_0 = MakeConstant(builder, 1.0, calc_type, out_size);

  mlir::MlirOp t1 = mlir::stablehlo::Mul(d, c1_25);
  mlir::MlirOp t2 = mlir::stablehlo::Subtract(t1, c2_25);
  mlir::MlirOp t3 = mlir::stablehlo::Mul(d, t2);
  mlir::MlirOp t4 = mlir::stablehlo::Mul(d, t3);
  return mlir::stablehlo::Add(t4, c1_0);
}

// Evaluates Keys' cubic convolution polynomial for 1 <= |d| < 2 with A = -0.75
// (`at::native::cubic_convolution2` in `ATen/native/UpSample.h`):
//   W_far(d) = ((A * d - 5 * A) * d + 8 * A) * d - 4 * A
//            = ((-0.75 * d + 3.75) * d - 6.0) * d + 3.0
mlir::MlirOp EvalCubicFar(mlir::MlirBuilder& builder, mlir::MlirOp d,
                          const mlir::Type calc_type, const int64_t out_size) {
  mlir::MlirOp c_neg0_75 = MakeConstant(builder, -0.75, calc_type, out_size);
  mlir::MlirOp c3_75 = MakeConstant(builder, 3.75, calc_type, out_size);
  mlir::MlirOp c6_0 = MakeConstant(builder, 6.0, calc_type, out_size);
  mlir::MlirOp c3_0 = MakeConstant(builder, 3.0, calc_type, out_size);

  mlir::MlirOp t1 = mlir::stablehlo::Mul(d, c_neg0_75);
  mlir::MlirOp t2 = mlir::stablehlo::Add(t1, c3_75);
  mlir::MlirOp t3 = mlir::stablehlo::Mul(d, t2);
  mlir::MlirOp t4 = mlir::stablehlo::Subtract(t3, c6_0);
  mlir::MlirOp t5 = mlir::stablehlo::Mul(d, t4);
  return mlir::stablehlo::Add(t5, c3_0);
}

// Inspired by `jax.image.scale` (`jax/_src/image/scale.py`) and
// `BuildInterpolationMatrix` in `upsample_aten_kernels.cc`, builds the
// [out_size, in_size] 1D Keys bicubic interpolation weight matrix
//   A[out_idx, in_idx] = sum_{k=0..3} w_k[out_idx] * (in_idx == idx_k[out_idx])
// directly in StableHLO from `Iota` and scalar splat constants.
mlir::MlirOp BuildBicubicInterpolationMatrix(mlir::MlirBuilder& builder,
                                             const int64_t in_size,
                                             const int64_t out_size,
                                             const AlignCorners align_corners,
                                             const std::optional<double> scale,
                                             const mlir::Type calc_type) {
  const bool align = align_corners == AlignCorners::kYes;
  const double ratio =
      calc_type.isF64()
          ? at::native::area_pixel_compute_scale<double>(in_size, out_size,
                                                         align, scale)
          : static_cast<double>(at::native::area_pixel_compute_scale<float>(
                in_size, out_size, align, scale));

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
  mlir::MlirOp one_vec = MakeConstant(builder, 1.0, calc_type, out_size);
  mlir::MlirOp src_idx;
  if (align) {
    src_idx = mlir::stablehlo::Mul(dst_idx, ratio_vec);
  } else {
    // Follows `at::native::area_pixel_compute_source_index(..., cubic=true)`:
    // negative `src_idx` values are not clamped before `floor` so the 4-tap
    // cubic neighborhood [-1, 0, 1, 2] preserves its fractional distance.
    mlir::MlirOp half_vec = MakeConstant(builder, 0.5, calc_type, out_size);
    mlir::MlirOp centered = mlir::stablehlo::Add(dst_idx, half_vec);
    mlir::MlirOp scaled = mlir::stablehlo::Mul(centered, ratio_vec);
    src_idx = mlir::stablehlo::Subtract(scaled, half_vec);
  }

  mlir::MlirOp idx_floor = mlir::stablehlo::Floor(src_idx);
  mlir::MlirOp lambda = mlir::stablehlo::Subtract(src_idx, idx_floor);
  mlir::MlirOp one_minus_lambda = mlir::stablehlo::Subtract(one_vec, lambda);

  std::array<mlir::MlirOp, 4> weights = {
      EvalCubicFar(builder, mlir::stablehlo::Add(lambda, one_vec), calc_type,
                   out_size),
      EvalCubicNear(builder, lambda, calc_type, out_size),
      EvalCubicNear(builder, one_minus_lambda, calc_type, out_size),
      EvalCubicFar(builder, mlir::stablehlo::Add(one_minus_lambda, one_vec),
                   calc_type, out_size),
  };

  mlir::MlirOp base_idx =
      mlir::stablehlo::ConvertElementType(idx_floor, i32_type);
  mlir::MlirOp zero_i32 = MakeConstant(builder, 0, i32_type, out_size);
  mlir::MlirOp one_i32 = MakeConstant(builder, 1, i32_type, out_size);
  mlir::MlirOp two_i32 = MakeConstant(builder, 2, i32_type, out_size);
  mlir::MlirOp max_in_i32 = MakeConstant(
      builder, std::max<int64_t>(0, in_size - 1), i32_type, out_size);

  std::array<mlir::MlirOp, 4> raw_indices = {
      mlir::stablehlo::Subtract(base_idx, one_i32),
      base_idx,
      mlir::stablehlo::Add(base_idx, one_i32),
      mlir::stablehlo::Add(base_idx, two_i32),
  };

  mlir::MlirOp col_indices =
      mlir::stablehlo::Iota(builder, index_mat_type, /*iotaDimension=*/1);
  mlir::MlirOp zeros =
      MakeConstant(builder, 0.0, calc_type, {out_size, in_size});

  mlir::MlirOp matrix;
  for (int tap = 0; tap < 4; ++tap) {
    mlir::MlirOp clamped_idx =
        mlir::stablehlo::Clamp(zero_i32, raw_indices[tap], max_in_i32);
    mlir::MlirOp index_mat =
        mlir::stablehlo::BroadcastInDim(index_mat_type, clamped_idx, {0});
    mlir::MlirOp weight_mat =
        mlir::stablehlo::BroadcastInDim(weight_mat_type, weights[tap], {0});
    mlir::MlirOp is_tap = mlir::stablehlo::Compare(
        col_indices, index_mat, mlir::stablehlo::ComparisonDirection::EQ);
    mlir::MlirOp term = mlir::stablehlo::Select(is_tap, weight_mat, zeros);
    matrix = (tap == 0) ? term : mlir::stablehlo::Add(matrix, term);
  }
  return matrix;
}

// Interpolates `operand` along `spatial_dim` (H=2 or W=3) by contracting it
// with the [out_size, in_size] 1D bicubic weight matrix (or its transpose when
// `direction == BicubicDirection::kBackward`).
absl::StatusOr<mlir::MlirOp> InterpolateSpatialDim(
    mlir::MlirOp operand, const int64_t in_size, const int64_t out_size,
    const AlignCorners align_corners, const std::optional<double> scale,
    const int64_t spatial_dim, const mlir::stablehlo::Precision precision,
    const BicubicDirection direction) {
  ABSL_DCHECK(spatial_dim == kBicubicHeightDim ||
              spatial_dim == kBicubicWidthDim);
  ABSL_DCHECK_EQ(GetTensorTypeOrDie(operand).getRank(), 4);
  if (in_size == out_size && (align_corners == AlignCorners::kYes ||
                              !scale.has_value() || *scale == 1.0)) {
    return operand;
  }
  mlir::MlirBuilder& builder = operand.getBuilder();
  const mlir::Type calc_type = GetTensorTypeOrDie(operand).getElementType();
  mlir::MlirOp matrix = BuildBicubicInterpolationMatrix(
      builder, in_size, out_size, align_corners, scale, calc_type);
  const c10::string_view equation =
      (direction == BicubicDirection::kBackward)
          ? (spatial_dim == kBicubicHeightDim ? "ncow,oh->nchw"
                                              : "ncho,ow->nchw")
          : (spatial_dim == kBicubicHeightDim ? "nchw,oh->ncow"
                                              : "nchw,ow->ncho");
  return BuildEinsumShlo({operand, matrix}, equation, precision);
}

// Builds the 2D bicubic interpolation (or its adjoint when
// `direction == BicubicDirection::kBackward`) of 4D NCHW `operand`.
// `in_h`/`in_w` and `out_h`/`out_w` are always the forward input and output
// spatial sizes.
absl::StatusOr<mlir::MlirOp> BuildBicubic2d(
    mlir::MlirOp operand, const int64_t in_h, const int64_t in_w,
    const int64_t out_h, const int64_t out_w, const AlignCorners align_corners,
    const std::optional<double> scale_h, const std::optional<double> scale_w,
    const mlir::stablehlo::Precision precision,
    const BicubicDirection direction) {
  if (in_h == out_h && in_w == out_w &&
      (direction == BicubicDirection::kBackward ||
       align_corners == AlignCorners::kYes ||
       ((!scale_h.has_value() || *scale_h == 1.0) &&
        (!scale_w.has_value() || *scale_w == 1.0)))) {
    return operand;
  }

  mlir::MlirBuilder& builder = operand.getBuilder();
  const mlir::Type elem_type = GetTensorTypeOrDie(operand).getElementType();
  // Matches ATen's `at::opmath_type<scalar_t>` (`ATen/OpMathType.h`) and
  // `UpSampleBicubic2d.cpp`, which promote narrow floating-point types (f16,
  // bf16, f8) to f32 (and keep f64 as f64) for intermediate interpolation and
  // backward gradient accumulation.
  const mlir::Type calc_type = elem_type.isF64()
                                   ? builder.getOpBuilder().getF64Type()
                                   : builder.getOpBuilder().getF32Type();
  mlir::MlirOp result =
      elem_type == calc_type
          ? operand
          : mlir::stablehlo::ConvertElementType(operand, calc_type);

  TT_ASSIGN_OR_RETURN(
      result, InterpolateSpatialDim(result, in_h, out_h, align_corners, scale_h,
                                    kBicubicHeightDim, precision, direction));
  TT_ASSIGN_OR_RETURN(
      result, InterpolateSpatialDim(result, in_w, out_w, align_corners, scale_w,
                                    kBicubicWidthDim, precision, direction));

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

// Lowers 2D bicubic interpolation as separable contractions along H and W:
// Y = A_h * X * A_w^T.
at::Tensor& AtenUpsampleBicubic2dOut(const at::Tensor& self,
                                     const at::IntArrayRef output_size,
                                     const bool align_corners,
                                     const std::optional<double> scale_h,
                                     const std::optional<double> scale_w,
                                     at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleBicubic2dOut, param_keys,
      (self, output_size, align_corners, scale_h, scale_w, out), {
        const Dimensions expected_output_shape = {
            self.size(0), self.size(1), output_size[0], output_size[1]};
        TT_THROW_IF_ERROR(
            ResizeTensorIfShapeDiffers(out, expected_output_shape));

        CheckUpsampleDtypes(self);

        TT_CHECK_THROW(self.scalar_type() == out.scalar_type(),
                       error::kInvalidArgument)
            << "expected out dtype " << ToString(self.scalar_type()) << ", got "
            << ToString(out.scalar_type());

        if (out.numel() == 0) {
          return out;
        }

        TT_ASSIGN_OR_THROW(const mlir::ElementType element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        const mlir::stablehlo::Precision precision =
            GetAndAddPrecisionWithDefaultTo(
                param_keys,
                mlir::stablehlo::Precision::
                    HIGHEST);  // EXPLICIT_PRECISION_OK=bicubic
        const AlignCorners align =
            align_corners ? AlignCorners::kYes : AlignCorners::kNo;

        const auto op_builder =
            [in_h = self.size(kBicubicHeightDim),
             in_w = self.size(kBicubicWidthDim),
             out_h = out.size(kBicubicHeightDim),
             out_w = out.size(kBicubicWidthDim), align, scale_h, scale_w,
             precision](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildBicubic2d(input, in_h, in_w, out_h, out_w, align, scale_h,
                                scale_w, precision, BicubicDirection::kForward);
        };

        TT_THROW_IF_ERROR(
            (DispatchOpOut<1>(std::move(op_builder), {self}, out,
                              {.out_dtype = element_type,
                               .out_dims = CopyIntVector(out.sizes()),
                               .op_param_cache_keys = std::move(param_keys)})));
        return out;
      });
}

// Lowers 2D bicubic interpolation backward as the adjoint separable
// contractions along H and W: dX = A_h^T * dY * A_w.
at::Tensor& AtenUpsampleBicubic2dBackwardGradInput(
    const at::Tensor& grad_output, const at::IntArrayRef output_size,
    const at::IntArrayRef input_size, const bool align_corners,
    const std::optional<double> scales_h, const std::optional<double> scales_w,
    at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kUpsampleBicubic2dBackwardGradInput, param_keys,
      (grad_output, IgnoreInCacheKey(output_size, "Unused"), input_size,
       align_corners, scales_h, scales_w, grad_input),
      {
        const Dimensions expected_grad_input_shape = CopyIntVector(input_size);
        TT_THROW_IF_ERROR(
            ResizeTensorIfShapeDiffers(grad_input, expected_grad_input_shape));

        CheckUpsampleDtypes(grad_output);

        TT_CHECK_THROW(grad_output.scalar_type() == grad_input.scalar_type(),
                       error::kInvalidArgument)
            << "expected grad_input dtype "
            << ToString(grad_output.scalar_type()) << ", got "
            << ToString(grad_input.scalar_type());

        if (grad_input.numel() == 0) {
          return grad_input;
        }

        TT_ASSIGN_OR_THROW(
            const mlir::ElementType element_type,
            ConvertTo<mlir::ElementType>(grad_input.scalar_type()));
        const mlir::stablehlo::Precision precision =
            GetAndAddPrecisionWithDefaultTo(
                param_keys,
                mlir::stablehlo::Precision::
                    HIGHEST);  // EXPLICIT_PRECISION_OK=bicubic
        const AlignCorners align =
            align_corners ? AlignCorners::kYes : AlignCorners::kNo;

        const auto op_builder =
            [in_h = input_size[kBicubicHeightDim],
             in_w = input_size[kBicubicWidthDim],
             out_h = grad_output.size(kBicubicHeightDim),
             out_w = grad_output.size(kBicubicWidthDim), align, scales_h,
             scales_w,
             precision](mlir::MlirOp grad_out) -> absl::StatusOr<mlir::MlirOp> {
          return BuildBicubic2d(grad_out, in_h, in_w, out_h, out_w, align,
                                scales_h, scales_w, precision,
                                BicubicDirection::kBackward);
        };

        TT_THROW_IF_ERROR(
            (DispatchOpOut<1>(std::move(op_builder), {grad_output}, grad_input,
                              {.out_dtype = element_type,
                               .out_dims = CopyIntVector(grad_input.sizes()),
                               .op_param_cache_keys = std::move(param_keys)})));
        return grad_input;
      });
}

}  // namespace torch_tpu
