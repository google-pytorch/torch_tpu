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

#include "csrc/ops/upsample/upsample_linear1d_aten_kernels.h"

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

static constexpr int64_t kLinearWidthDim = 2;

enum class AlignCorners {
  kNo,
  kYes,
};

enum class UpsampleDirection {
  kForward,
  kBackward,
};

// Inspired by `jax.image.scale` (`jax/_src/image/scale.py`),
// builds the [out_size, in_size] 1D linear interpolation weight matrix
//   A[out_idx, in_idx] =
//       lambda0[out_idx] * (in_idx == index0[out_idx]) +
//       lambda1[out_idx] * (in_idx == index1[out_idx])
// directly in StableHLO from `Iota` and scalar splat constants.
mlir::MlirOp BuildLinear1dInterpolationMatrix(mlir::MlirBuilder& builder,
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
  mlir::MlirOp zero_vec = MakeConstant(builder, 0.0, calc_type, out_size);
  mlir::MlirOp one_vec = MakeConstant(builder, 1.0, calc_type, out_size);
  mlir::MlirOp src_idx;
  if (align) {
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

// Builds the 1D linear interpolation (or its adjoint when `direction` is
// `UpsampleDirection::kBackward`) of 3D NCW `operand`. `in_w` and `out_w` are
// always the forward input and output spatial sizes.
absl::StatusOr<mlir::MlirOp> BuildLinear1d(
    mlir::MlirOp operand, const int64_t in_w, const int64_t out_w,
    const AlignCorners align_corners, const std::optional<double> scale,
    const mlir::stablehlo::Precision precision,
    const UpsampleDirection direction) {
  ABSL_DCHECK_EQ(GetTensorTypeOrDie(operand).getRank(), 3);
  if (in_w == out_w) {
    return operand;
  }

  mlir::MlirBuilder& builder = operand.getBuilder();
  const mlir::Type elem_type = GetTensorTypeOrDie(operand).getElementType();
  // Matches ATen's `at::opmath_type<scalar_t>` (`ATen/OpMathType.h`) and
  // `UpSampleKernel.cpp`, which promote narrow floating-point types (f16,
  // bf16, f8) to f32 (and keep f64 as f64) for intermediate interpolation and
  // backward gradient accumulation.
  const mlir::Type calc_type = elem_type.isF64()
                                   ? builder.getOpBuilder().getF64Type()
                                   : builder.getOpBuilder().getF32Type();
  mlir::MlirOp operand_calc =
      elem_type == calc_type
          ? operand
          : mlir::stablehlo::ConvertElementType(operand, calc_type);

  mlir::MlirOp matrix = BuildLinear1dInterpolationMatrix(
      builder, in_w, out_w, align_corners, scale, calc_type);
  const c10::string_view equation =
      direction == UpsampleDirection::kBackward ? "nco,ow->ncw" : "ncw,ow->nco";
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp result,
      BuildEinsumShlo({operand_calc, matrix}, equation, precision));

  return elem_type == calc_type
             ? result
             : mlir::stablehlo::ConvertElementType(result, elem_type);
}

void CheckUpsampleLinear1dDtypes(const at::Tensor& tensor) {
  TT_CHECK_THROW(at::isFloatingType(tensor.scalar_type()),
                 error::kPythonNotImplementedError)
      << "not implemented for " << ToString(tensor.scalar_type());
}

void ValidateUpsampleLinear1dInputs(const at::Tensor& self,
                                    at::IntArrayRef output_size,
                                    const at::Tensor& out) {
  TT_CHECK_THROW(output_size.size() == 1, error::kInvalidArgument)
      << "expected output_size to have 1 element, got "
      << ToString(output_size.size());

  TT_CHECK_THROW(self.dim() == 3, error::kInvalidArgument)
      << "expected 3D input tensor, got " << ToString(self.dim()) << "D tensor";

  const int64_t input_width = self.size(2);
  const int64_t output_width = output_size[0];

  TT_CHECK_THROW(input_width > 0 && output_width > 0, error::kInvalidArgument)
      << "expected input and output sizes to be greater than 0, got input (W: "
      << ToString(input_width) << ") and output (W: " << ToString(output_width)
      << ")";

  CheckUpsampleLinear1dDtypes(self);

  TT_CHECK_THROW(self.scalar_type() == out.scalar_type(),
                 error::kInvalidArgument)
      << "expected out dtype " << ToString(self.scalar_type()) << ", got "
      << ToString(out.scalar_type());
}

void ValidateUpsampleLinear1dBackwardInputs(const at::Tensor& grad_output,
                                            at::IntArrayRef output_size,
                                            at::IntArrayRef input_size,
                                            const at::Tensor& grad_input) {
  TT_CHECK_THROW(output_size.size() == 1, error::kInvalidArgument)
      << "expected output_size to have 1 element, got "
      << ToString(output_size.size());

  TT_CHECK_THROW(input_size.size() == 3, error::kInvalidArgument)
      << "expected input_size to have 3 elements, got "
      << ToString(input_size.size());

  const int64_t nbatch = input_size[0];
  const int64_t channels = input_size[1];
  const int64_t input_width = input_size[2];
  const int64_t output_width = output_size[0];

  TT_CHECK_THROW(input_width > 0 && output_width > 0, error::kInvalidArgument)
      << "expected input and output sizes to be greater than 0, got input (W: "
      << ToString(input_width) << ") and output (W: " << ToString(output_width)
      << ")";

  TT_CHECK_THROW(grad_output.dim() == 3, error::kInvalidArgument)
      << "expected 3D grad_output tensor, got " << ToString(grad_output.dim())
      << "D tensor";

  const Dimensions expected_grad_output_shape = {nbatch, channels,
                                                 output_width};
  TT_CHECK_THROW(
      grad_output.sizes() == AsArrayRef<int64_t>(expected_grad_output_shape),
      error::kInvalidArgument)
      << "expected grad_output to have size "
      << ToString(expected_grad_output_shape) << ", got "
      << ToString(grad_output.sizes());

  CheckUpsampleLinear1dDtypes(grad_output);

  TT_CHECK_THROW(grad_output.scalar_type() == grad_input.scalar_type(),
                 error::kInvalidArgument)
      << "expected grad_input dtype " << ToString(grad_output.scalar_type())
      << ", got " << ToString(grad_input.scalar_type());
}

}  // namespace

// Lowers 1D linear interpolation as a matrix contraction along W:
// Y = X * A_w^T.
at::Tensor& AtenUpsampleLinear1dOut(const at::Tensor& self,
                                    at::IntArrayRef output_size,
                                    bool align_corners,
                                    std::optional<double> scales,
                                    at::Tensor& out) {
  TT_KERNEL(
      OpName::kUpsampleLinear1dOut, param_keys,
      (self, output_size, align_corners, scales, out), {
        ValidateUpsampleLinear1dInputs(self, output_size, out);

        const Dimensions expected_output_shape = {self.size(0), self.size(1),
                                                  output_size[0]};
        TT_THROW_IF_ERROR(
            ResizeTensorIfShapeDiffers(out, expected_output_shape));

        if (self.numel() == 0) {
          return out;
        }

        TT_ASSIGN_OR_THROW(const auto element_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        const mlir::stablehlo::Precision precision =
            GetAndAddPrecisionWithDefaultTo(
                param_keys,
                mlir::stablehlo::Precision::
                    HIGHEST);  // EXPLICIT_PRECISION_OK=linear

        const AlignCorners align =
            align_corners ? AlignCorners::kYes : AlignCorners::kNo;
        const auto op_builder =
            [in_w = self.size(kLinearWidthDim),
             out_w = out.size(kLinearWidthDim), align, scales,
             precision](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildLinear1d(input, in_w, out_w, align, scales, precision,
                               UpsampleDirection::kForward);
        };

        TT_THROW_IF_ERROR(
            (DispatchOpOut<1>(std::move(op_builder), {self}, out,
                              {.out_dtype = element_type,
                               .out_dims = CopyIntVector(out.sizes()),
                               .op_param_cache_keys = std::move(param_keys)})));
        return out;
      });
}

// Lowers 1D linear interpolation backward as the adjoint matrix contraction
// along W: dX = dY * A_w.
at::Tensor& AtenUpsampleLinear1dBackwardGradInput(const at::Tensor& grad_output,
                                                  at::IntArrayRef output_size,
                                                  at::IntArrayRef input_size,
                                                  bool align_corners,
                                                  std::optional<double> scales,
                                                  at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kUpsampleLinear1dBackwardGradInput, param_keys,
      (grad_output, IgnoreInCacheKey(output_size, "Doesn't affect SHLO"),
       input_size, align_corners, scales, grad_input),
      {
        ValidateUpsampleLinear1dBackwardInputs(grad_output, output_size,
                                               input_size, grad_input);

        const Dimensions expected_grad_input_shape = CopyIntVector(input_size);
        TT_THROW_IF_ERROR(
            ResizeTensorIfShapeDiffers(grad_input, expected_grad_input_shape));

        if (grad_input.numel() == 0) {
          return grad_input;
        }

        TT_ASSIGN_OR_THROW(
            const auto element_type,
            ConvertTo<mlir::ElementType>(grad_input.scalar_type()));
        const mlir::stablehlo::Precision precision =
            GetAndAddPrecisionWithDefaultTo(
                param_keys,
                mlir::stablehlo::Precision::
                    HIGHEST);  // EXPLICIT_PRECISION_OK=linear

        const AlignCorners align =
            align_corners ? AlignCorners::kYes : AlignCorners::kNo;
        const auto op_builder =
            [in_w = grad_input.size(kLinearWidthDim),
             out_w = grad_output.size(kLinearWidthDim), align, scales,
             precision](mlir::MlirOp grad_out) -> absl::StatusOr<mlir::MlirOp> {
          return BuildLinear1d(grad_out, in_w, out_w, align, scales, precision,
                               UpsampleDirection::kBackward);
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
