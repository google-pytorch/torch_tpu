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

#include "csrc/ops/upsample/upsample_bicubic2d_aa_aten_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/native/UpSample.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/nullary_aten_kernels.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
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

template <typename T>
T CubicFilter(const T x) {
  const T abs_x = std::abs(x);
  if (abs_x < static_cast<T>(1.0)) {
    return at::native::cubic_convolution1<T>(abs_x, static_cast<T>(-0.5));
  } else if (abs_x < static_cast<T>(2.0)) {
    return at::native::cubic_convolution2<T>(abs_x, static_cast<T>(-0.5));
  }
  return static_cast<T>(0.0);
}

template <typename T>
std::vector<T> BuildWeightMatrix(const int64_t in_size, const int64_t out_size,
                                 const AlignCorners align_corners,
                                 const std::optional<double> scale_opt) {
  std::vector<T> w_mat(in_size * out_size, static_cast<T>(0));
  const T scale = at::native::area_pixel_compute_scale<T>(
      in_size, out_size, align_corners == AlignCorners::kYes, scale_opt);

  // Anti-aliasing filter formulation: expand support area proportional to the
  // downsampling scale. The distance to query pixel centers is multiplied by
  // invscale to retrieve the correct normalized filter weights.
  const T support = (scale >= static_cast<T>(1.0))
                        ? (static_cast<T>(2.0) * scale)
                        : static_cast<T>(2.0);
  const T invscale = (scale >= static_cast<T>(1.0))
                         ? (static_cast<T>(1.0) / scale)
                         : static_cast<T>(1.0);

  std::vector<T> w;
  w.reserve(
      static_cast<size_t>(support * static_cast<T>(2.0) + static_cast<T>(2.0)));

  for (int64_t i = 0; i < out_size; ++i) {
    const T center = scale * (static_cast<T>(i) + static_cast<T>(0.5));

    const int64_t xmin = std::max<int64_t>(
        static_cast<int64_t>(center - support + static_cast<T>(0.5)), 0);
    const int64_t xmax = std::min<int64_t>(
        static_cast<int64_t>(center + support + static_cast<T>(0.5)), in_size);
    const int64_t xsize = std::max<int64_t>(xmax - xmin, 0);

    T sum_w = static_cast<T>(0);
    w.resize(xsize);
    for (int64_t j = 0; j < xsize; ++j) {
      const T val =
          (static_cast<T>(j + xmin) - center + static_cast<T>(0.5)) * invscale;
      w[j] = CubicFilter<T>(val);
      sum_w += w[j];
    }

    if (sum_w != static_cast<T>(0)) {
      for (int64_t j = 0; j < xsize; ++j) {
        w_mat[(xmin + j) * out_size + i] = w[j] / sum_w;
      }
    } else {
      for (int64_t j = 0; j < xsize; ++j) {
        w_mat[(xmin + j) * out_size + i] = w[j];
      }
    }
  }
  return w_mat;
}

template <typename T>
absl::StatusOr<mlir::MlirOp> ComputeAndSetWeights(
    mlir::MlirBuilder& builder, const int64_t in_size, const int64_t out_size,
    const AlignCorners align_corners, const std::optional<double> scale_opt,
    const mlir::Type calc_type) {
  const std::vector<T> w_mat =
      BuildWeightMatrix<T>(in_size, out_size, align_corners, scale_opt);
  return MakeConstant(builder, w_mat, calc_type, Dimensions{in_size, out_size});
}

absl::StatusOr<mlir::MlirOp> BuildUpsampleBicubic2dAaShlo(
    mlir::MlirOp input, const Dimensions& output_shape,
    const AlignCorners align_corners, const std::optional<double> scales_h,
    const std::optional<double> scales_w, const mlir::Type calc_type) {
  mlir::MlirBuilder& builder = input.getBuilder();
  mlir::MLIRContext* const ctx = &input.getContext();
  const Dimensions input_shape =
      CopyIntVector(GetTensorTypeOrDie(input).getShape());

  const int64_t in_h = input_shape[2];
  const int64_t in_w = input_shape[3];
  const int64_t out_h = output_shape[2];
  const int64_t out_w = output_shape[3];

  mlir::MlirOp weight_h;
  mlir::MlirOp weight_w;
  if (calc_type.isF64()) {
    TT_ASSIGN_OR_RETURN(weight_h, ComputeAndSetWeights<double>(
                                      builder, in_h, out_h, align_corners,
                                      scales_h, calc_type));
    TT_ASSIGN_OR_RETURN(weight_w, ComputeAndSetWeights<double>(
                                      builder, in_w, out_w, align_corners,
                                      scales_w, calc_type));
  } else {
    TT_ASSIGN_OR_RETURN(weight_h, ComputeAndSetWeights<float>(
                                      builder, in_h, out_h, align_corners,
                                      scales_h, calc_type));
    TT_ASSIGN_OR_RETURN(weight_w, ComputeAndSetWeights<float>(
                                      builder, in_w, out_w, align_corners,
                                      scales_w, calc_type));
  }

  // Separable 2D convolution via two 1D operations:
  // 1. input [N, C, H_in, W_in] x W_weights [W_in, W_out]
  //    Contracting over W_in (dim 3 on input, dim 0 on weights)
  //    Output -> interm [N, C, H_in, W_out]
  const auto dot_dimension_numbers_w =
      mlir::stablehlo::DotDimensionNumbersAttr::get(
          ctx, /*lhs_batching_dims=*/{}, /*rhs_batching_dims=*/{},
          /*lhs_contracting_dims=*/{3}, /*rhs_contracting_dims=*/{0});
  const mlir::MlirOp interm =
      mlir::stablehlo::DotGeneral(input, weight_w, dot_dimension_numbers_w);

  // 2. interm [N, C, H_in, W_out] x H_weights [H_in, H_out]
  //    Contracting over H_in (dim 2 on interm, dim 0 on weights)
  //    Output -> out_val [N, C, W_out, H_out]
  const auto dot_dimension_numbers_h =
      mlir::stablehlo::DotDimensionNumbersAttr::get(
          ctx, /*lhs_batching_dims=*/{}, /*rhs_batching_dims=*/{},
          /*lhs_contracting_dims=*/{2}, /*rhs_contracting_dims=*/{0});
  // Non-const: mlir::stablehlo::Transpose requires a non-const MlirOp&.
  mlir::MlirOp out_val =
      mlir::stablehlo::DotGeneral(interm, weight_h, dot_dimension_numbers_h);

  // Transpose [N, C, W_out, H_out] to final [N, C, H_out, W_out] shape
  return mlir::stablehlo::Transpose(out_val, {0, 1, 3, 2});
}

}  // namespace

at::Tensor& AtenUpsampleBicubic2dAaOut(const at::Tensor& self,
                                       at::IntArrayRef output_size,
                                       const bool align_corners,
                                       const std::optional<double> scales_h,
                                       const std::optional<double> scales_w,
                                       at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleBicubic2dAaOut, param_keys,
      (self, output_size, align_corners, scales_h, scales_w, out), {
        TT_CHECK_THROW(self.dim() == 4, error::kInvalidArgument)
            << "expected 4D tensor, got " << ToString(self.dim()) << "D";
        TT_CHECK_THROW(output_size.size() == 2, error::kInvalidArgument)
            << "expected output_size to have 2 elements, got "
            << ToString(output_size.size());
        TT_CHECK_THROW(output_size[0] > 0 && output_size[1] > 0,
                       error::kInvalidArgument)
            << "expected positive output_size, got [" << output_size[0] << ", "
            << output_size[1] << "]";
        TT_CHECK_THROW(
            self.numel() != 0 ||
                (self.size(1) > 0 && self.size(2) > 0 && self.size(3) > 0),
            error::kInvalidArgument)
            << "expected non-empty 4D data tensor, got a tensor with sizes "
            << ToString(self.sizes());
        TT_CHECK_THROW(self.is_floating_point(),
                       error::kPythonNotImplementedError)
            << "expected floating-point dtype, got "
            << ToString(self.scalar_type());
        TT_CHECK_THROW(self.scalar_type() == out.scalar_type(),
                       error::kInvalidArgument)
            << "expected out dtype " << ToString(self.scalar_type()) << ", got "
            << ToString(out.scalar_type());

        const Dimensions expected_output_shape = {
            self.size(0), self.size(1), output_size[0], output_size[1]};

        TT_THROW_IF_ERROR(
            ResizeTensorIfShapeDiffers(out, expected_output_shape));

        if (self.numel() == 0 || out.numel() == 0) {
          if (out.numel() > 0) {
            out.copy_(AtenEfficientZeroTensor(expected_output_shape,
                                              out.scalar_type(), std::nullopt,
                                              out.device(), std::nullopt));
          }
          return out;
        }

        TT_ASSIGN_OR_THROW(const auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        mlir::ElementType computation_dtype = mlir::ElementType::F32;
        switch (self.scalar_type()) {
          case at::kDouble:
            computation_dtype = mlir::ElementType::F64;
            break;
          default:
            computation_dtype = mlir::ElementType::F32;
            break;
        }

        const auto op_builder =
            [out_shape = expected_output_shape,
             align_corners =
                 align_corners ? AlignCorners::kYes : AlignCorners::kNo,
             scales_h, scales_w,
             real_dtype = c10::toRealValueType(self.scalar_type())](
                mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          const mlir::Type calc_type =
              (real_dtype == at::kDouble)
                  ? input.getBuilder().getOpBuilder().getF64Type()
                  : input.getBuilder().getOpBuilder().getF32Type();
          return BuildUpsampleBicubic2dAaShlo(input, out_shape, align_corners,
                                              scales_h, scales_w, calc_type);
        };

        TT_THROW_IF_ERROR(
            DispatchOpOut<1>(std::move(op_builder), self, out,
                             {.out_dtype = element_type,
                              .out_dims = CopyIntVector(out.sizes()),
                              .computation_dtype = computation_dtype,
                              .op_param_cache_keys = std::move(param_keys)}));
        return out;
      });
}

}  // namespace torch_tpu
