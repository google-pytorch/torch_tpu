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
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "c10/util/ArrayRef.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

void CheckUpsampleDtypes(const at::Tensor& tensor) {
  const at::ScalarType st = tensor.scalar_type();
  TT_CHECK_THROW(st == at::kFloat || st == at::kDouble || st == at::kHalf ||
                     st == at::kBFloat16,
                 error::kPythonNotImplementedError)
      << "expected the input dtype to be floating point, got " << ToString(st);
}

template <typename T>
T CalculateResampleScale(const int64_t in_size, const int64_t out_size,
                         const bool align_corners,
                         const std::optional<double>& opt_scale) {
  if (align_corners) {
    if (out_size > 1) {
      return static_cast<T>(in_size - 1) / static_cast<T>(out_size - 1);
    }
    return static_cast<T>(0);
  }
  if (opt_scale.has_value() && *opt_scale > 0.0) {
    return static_cast<T>(1.0 / *opt_scale);
  }
  return static_cast<T>(in_size) / static_cast<T>(out_size);
}

template <typename T>
void ComputeAndNormalizeRowWeights(T* row_ptr, const int64_t in_size,
                                   const T center, const T support,
                                   const T invscale) {
  const int64_t xmin = std::max(
      static_cast<int64_t>(center - support + static_cast<T>(0.5)), int64_t{0});
  const int64_t xsize =
      std::min(static_cast<int64_t>(center + support + static_cast<T>(0.5)),
               in_size) -
      xmin;

  T total_w = static_cast<T>(0);
  for (int64_t j = 0; j < xsize; ++j) {
    const int64_t in_idx = xmin + j;
    const T arg =
        (static_cast<T>(in_idx) + static_cast<T>(0.5) - center) * invscale;
    const T abs_arg = std::abs(arg);
    const T w = (abs_arg < static_cast<T>(1.0))
                    ? (static_cast<T>(1.0) - abs_arg)
                    : static_cast<T>(0.0);
    row_ptr[in_idx] = w;
    total_w += w;
  }

  if (total_w > static_cast<T>(0)) {
    for (int64_t j = 0; j < xsize; ++j) {
      const int64_t in_idx = xmin + j;
      row_ptr[in_idx] /= total_w;
    }
  }
}

template <typename T>
std::vector<T> Compute1dAntiAliasBilinearWeights(
    const int64_t in_size, const int64_t out_size, const bool align_corners,
    const std::optional<double>& opt_scale) {
  const T scale =
      CalculateResampleScale<T>(in_size, out_size, align_corners, opt_scale);
  const T support =
      (scale >= static_cast<T>(1.0)) ? scale : static_cast<T>(1.0);
  const T invscale = (scale >= static_cast<T>(1.0))
                         ? (static_cast<T>(1.0) / scale)
                         : static_cast<T>(1.0);

  std::vector<T> weights(out_size * in_size, static_cast<T>(0));

  for (int64_t i = 0; i < out_size; ++i) {
    const T center = scale * (static_cast<T>(i) + static_cast<T>(0.5));
    ComputeAndNormalizeRowWeights(weights.data() + i * in_size, in_size, center,
                                  support, invscale);
  }
  return weights;
}

mlir::MlirOp Create1dWeightsConstant(mlir::MlirBuilder& builder,
                                     const int64_t in_size,
                                     const int64_t out_size,
                                     const bool align_corners,
                                     const std::optional<double>& scale,
                                     const mlir::Type calc_type) {
  const auto weight_type =
      mlir::RankedTensorType::get({out_size, in_size}, calc_type);
  if (calc_type.isF64()) {
    const std::vector<double> weights =
        Compute1dAntiAliasBilinearWeights<double>(in_size, out_size,
                                                  align_corners, scale);
    return mlir::stablehlo::Constant(builder,
                                     mlir::makeConstant(weights, weight_type));
  }
  const std::vector<float> weights = Compute1dAntiAliasBilinearWeights<float>(
      in_size, out_size, align_corners, scale);
  return mlir::stablehlo::Constant(builder,
                                   mlir::makeConstant(weights, weight_type));
}

absl::StatusOr<mlir::MlirOp> ResampleWidthShlo(
    mlir::MlirOp input, const int64_t w_in, const int64_t w_out,
    const bool align_corners, const std::optional<double>& scale_w) {
  if (w_in == w_out && CalculateResampleScale<double>(
                           w_in, w_out, align_corners, scale_w) == 1.0) {
    return input;
  }
  const mlir::Type calc_type = GetTensorTypeOrDie(input).getElementType();
  auto& builder = input.getBuilder();
  auto& ctx = builder.getContext();
  const mlir::MlirOp w_weights = Create1dWeightsConstant(
      builder, w_in, w_out, align_corners, scale_w, calc_type);
  const auto dot_dims_w = mlir::stablehlo::DotDimensionNumbersAttr::get(
      &ctx, /*lhs_batching_dimensions=*/{}, /*rhs_batching_dimensions=*/{},
      /*lhs_contracting_dimensions=*/{3}, /*rhs_contracting_dimensions=*/{1});
  return mlir::stablehlo::DotGeneral(input, w_weights, dot_dims_w);
}

absl::StatusOr<mlir::MlirOp> ResampleHeightShlo(
    mlir::MlirOp input, const int64_t h_in, const int64_t h_out,
    const bool align_corners, const std::optional<double>& scale_h) {
  if (h_in == h_out && CalculateResampleScale<double>(
                           h_in, h_out, align_corners, scale_h) == 1.0) {
    return input;
  }
  const mlir::Type calc_type = GetTensorTypeOrDie(input).getElementType();
  auto& builder = input.getBuilder();
  auto& ctx = builder.getContext();
  const mlir::MlirOp h_weights = Create1dWeightsConstant(
      builder, h_in, h_out, align_corners, scale_h, calc_type);
  const auto dot_dims_h = mlir::stablehlo::DotDimensionNumbersAttr::get(
      &ctx, /*lhs_batching_dimensions=*/{}, /*rhs_batching_dimensions=*/{},
      /*lhs_contracting_dimensions=*/{2}, /*rhs_contracting_dimensions=*/{1});
  mlir::MlirOp h_dot =
      mlir::stablehlo::DotGeneral(input, h_weights, dot_dims_h);
  const Dimensions perm = {0, 1, 3, 2};
  return mlir::stablehlo::Transpose(h_dot, AsArrayRef<int64_t>(perm));
}

absl::StatusOr<mlir::MlirOp> BuildUpsampleBilinear2dAaShlo(
    mlir::MlirOp input, const Dimensions& output_shape,
    const bool align_corners, const std::optional<double>& scale_h,
    const std::optional<double>& scale_w) {
  const Dimensions input_shape =
      CopyIntVector(GetTensorTypeOrDie(input).getShape());
  const int64_t h_in = input_shape[2];
  const int64_t w_in = input_shape[3];
  const int64_t h_out = output_shape[2];
  const int64_t w_out = output_shape[3];

  TT_ASSIGN_OR_RETURN(
      const mlir::MlirOp resampled_w,
      ResampleWidthShlo(input, w_in, w_out, align_corners, scale_w));
  TT_ASSIGN_OR_RETURN(
      const mlir::MlirOp resampled_h,
      ResampleHeightShlo(resampled_w, h_in, h_out, align_corners, scale_h));

  return resampled_h;
}

}  // namespace

at::Tensor& AtenUpsampleBilinear2dAaOut(const at::Tensor& self,
                                        c10::IntArrayRef output_size,
                                        const bool align_corners,
                                        const std::optional<double> scales_h,
                                        const std::optional<double> scales_w,
                                        at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleBilinear2dAaOut, param_keys,
      (self, output_size, align_corners, scales_h, scales_w, out), {
        const Dimensions expected_output_shape = {
            self.size(0), self.size(1), output_size[0], output_size[1]};
        TT_THROW_IF_ERROR(
            ResizeTensorIfShapeDiffers(out, expected_output_shape));

        CheckUpsampleDtypes(self);

        TT_CHECK_THROW(self.scalar_type() == out.scalar_type(),
                       error::kInvalidArgument)
            << "expected out dtype " << ToString(self.scalar_type()) << ", got "
            << ToString(out.scalar_type());

        if (self.numel() == 0 || out.numel() == 0) {
          return out;
        }

        TT_ASSIGN_OR_THROW(const auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        const mlir::ElementType computation_dtype =
            (self.scalar_type() == at::kDouble) ? mlir::ElementType::F64
                                                : mlir::ElementType::F32;

        auto op_builder = [out_shape = CopyIntVector(out.sizes()),
                           align_corners, scales_h,
                           scales_w](mlir::MlirOp input) {
          return BuildUpsampleBilinear2dAaShlo(input, out_shape, align_corners,
                                               scales_h, scales_w);
        };

        TT_THROW_IF_ERROR(
            (DispatchOpOut<1>(std::move(op_builder), self, out,
                              {.out_dtype = element_type,
                               .out_dims = CopyIntVector(out.sizes()),
                               .computation_dtype = computation_dtype,
                               .op_param_cache_keys = std::move(param_keys)})));
        return out;
      });
}

}  // namespace torch_tpu
