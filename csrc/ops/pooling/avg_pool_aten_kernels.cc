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

#include "csrc/ops/pooling/avg_pool_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/avg_pool1d_native.h"
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/GradMode.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fake_tensor_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/einsum/einsum_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/pooling/pooling.h"
#include "csrc/ops/precision_context.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

// Number of spatial dimensions (D, H, W) for 3D average pooling.
// For 3D average pooling, reduced-precision floating-point inputs (e.g.,
// bfloat16) are promoted to f32 during window reduction and division to match
// CUDA's float accumulator precision (`acc_type<scalar_t, true>`).
constexpr int64_t k3dSpatialDimCount = 3;

enum class AvgPoolAxisDivisorMode {
  kUnit,
  kWindowCount,
  kConstantOverride,
};

mlir::stablehlo::Precision GetAvgPoolPrecision(OpParamCacheKeys& param_keys) {
  return GetAndAddPrecisionWithDefaultTo(
      param_keys,
      mlir::stablehlo::Precision::HIGHEST);  // EXPLICIT_PRECISION_OK=avg_pool
}

mlir::MlirOp BuildRuntimePoolingOutputSize(mlir::MlirBuilder& builder,
                                           mlir::MlirOp runtime_in_size,
                                           const int64_t kernel,
                                           const int64_t stride,
                                           const int64_t pad,
                                           const bool ceil_mode) {
  const mlir::Type i32_type = builder.getOpBuilder().getI32Type();
  auto cst = [&](const int64_t val) {  // REF_CAPTURE_OK=called locally
    return MakeScalarConstant(builder, static_cast<int32_t>(val), i32_type);
  };
  mlir::MlirOp one = cst(1);
  mlir::MlirOp stride_op = cst(stride);
  mlir::MlirOp offset = cst(2 * pad - kernel);
  mlir::MlirOp numerator = mlir::stablehlo::Add(runtime_in_size, offset);
  if (!ceil_mode) {
    mlir::MlirOp div_res = mlir::stablehlo::Div(numerator, stride_op);
    mlir::MlirOp out = mlir::stablehlo::Add(div_res, one);
    return mlir::stablehlo::Max(out, one);
  }
  mlir::MlirOp stride_minus_one = cst(stride - 1);
  mlir::MlirOp ceil_num = mlir::stablehlo::Add(numerator, stride_minus_one);
  mlir::MlirOp div_res = mlir::stablehlo::Div(ceil_num, stride_op);
  mlir::MlirOp out = mlir::stablehlo::Add(div_res, one);
  mlir::MlirOp out_minus_one = mlir::stablehlo::Subtract(out, one);
  mlir::MlirOp last_start = mlir::stablehlo::Mul(out_minus_one, stride_op);
  mlir::MlirOp pad_op = cst(pad);
  mlir::MlirOp in_plus_pad = mlir::stablehlo::Add(runtime_in_size, pad_op);
  mlir::MlirOp overflow = mlir::stablehlo::Compare(
      last_start, in_plus_pad, mlir::stablehlo::ComparisonDirection::GE);
  out = mlir::stablehlo::Select(overflow, out_minus_one, out);
  return mlir::stablehlo::Max(out, one);
}

mlir::MlirOp BuildAvgPool1dWindowWeights(
    mlir::MlirBuilder& builder, const int64_t in_size, const int64_t out_size,
    const int64_t kernel, const int64_t stride, const int64_t pad,
    const bool ceil_mode, const bool count_include_pad,
    const AvgPoolAxisDivisorMode divisor_mode,
    const int64_t divisor_override_val, const bool has_dynamic_in_size,
    mlir::MlirOp in_size_col, mlir::MlirOp unclipped_starts,
    mlir::MlirOp unclipped_ends, mlir::MlirOp starts, mlir::MlirOp ends,
    const mlir::Type compute_dtype) {
  const Dimensions col_shape = {out_size, 1};
  if (divisor_mode == AvgPoolAxisDivisorMode::kUnit) {
    return MakeConstant(builder, 1.0, compute_dtype, col_shape);
  }
  if (divisor_mode == AvgPoolAxisDivisorMode::kConstantOverride) {
    if (divisor_override_val != 0) {
      return MakeConstant(builder,
                          1.0 / static_cast<double>(divisor_override_val),
                          compute_dtype, col_shape);
    }
    mlir::MlirOp one_calc =
        MakeConstant(builder, 1.0, compute_dtype, col_shape);
    mlir::MlirOp zero_calc =
        MakeConstant(builder, 0.0, compute_dtype, col_shape);
    return mlir::stablehlo::Div(one_calc, zero_calc);
  }
  const mlir::Type i32_type = builder.getOpBuilder().getI32Type();
  mlir::MlirOp lengths = mlir::stablehlo::Subtract(ends, starts);
  if (count_include_pad) {
    const int64_t last_window_end = (out_size - 1) * stride + kernel;
    if (!ceil_mode ||
        (!has_dynamic_in_size && last_window_end <= in_size + 2 * pad)) {
      return MakeConstant(builder, 1.0 / static_cast<double>(kernel),
                          compute_dtype, col_shape);
    }
    mlir::MlirOp pad_col = MakeConstant(builder, pad, i32_type, col_shape);
    mlir::MlirOp padded_end_bound = mlir::stablehlo::Add(in_size_col, pad_col);
    mlir::MlirOp padded_ends =
        mlir::stablehlo::Min(unclipped_ends, padded_end_bound);
    lengths = mlir::stablehlo::Subtract(padded_ends, unclipped_starts);
  }
  if (has_dynamic_in_size) {
    mlir::MlirOp one_i32 = MakeConstant(builder, 1, i32_type, col_shape);
    lengths = mlir::stablehlo::Max(lengths, one_i32);
  }
  mlir::MlirOp lengths_calc =
      mlir::stablehlo::ConvertElementType(lengths, compute_dtype);
  mlir::MlirOp one_calc = MakeConstant(builder, 1.0, compute_dtype, col_shape);
  return mlir::stablehlo::Div(one_calc, lengths_calc);
}

// Builds the 1D average pooling weight matrix W of shape [out_size, in_size]
// in `compute_dtype`, where for each output window o in [0, out_size):
//   unclipped_start = o * stride - pad
//   unclipped_end   = unclipped_start + kernel
//   s_o = max(0, unclipped_start)
//   e_o = min(in_size, unclipped_end)
//   W[o, i] = w_o  if s_o <= i < e_o, else 0.0
mlir::MlirOp BuildAvgPool1dWeightMatrix(
    mlir::MlirBuilder& builder, const int64_t in_size, const int64_t out_size,
    const int64_t kernel, const int64_t stride, const int64_t pad,
    const bool ceil_mode, const bool count_include_pad,
    const AvgPoolAxisDivisorMode divisor_mode,
    const int64_t divisor_override_val,
    std::optional<mlir::MlirOp> runtime_in_size,
    const mlir::Type compute_dtype) {
  const mlir::Type i32_type = builder.getOpBuilder().getI32Type();
  const Dimensions col_shape = {out_size, 1};
  const mlir::RankedTensorType col_i32_type =
      mlir::RankedTensorType::get(col_shape, i32_type);

  mlir::MlirOp row_iota = mlir::stablehlo::Iota(builder, col_i32_type, 0);
  mlir::MlirOp stride_const =
      MakeConstant(builder, stride, i32_type, col_shape);
  mlir::MlirOp pad_const = MakeConstant(builder, pad, i32_type, col_shape);
  mlir::MlirOp kernel_const =
      MakeConstant(builder, kernel, i32_type, col_shape);
  mlir::MlirOp zero_i32 = MakeConstant(builder, 0, i32_type, col_shape);
  mlir::MlirOp in_size_col =
      runtime_in_size.has_value()
          ? mlir::stablehlo::BroadcastInDim(col_i32_type, *runtime_in_size,
                                            /*broadcast_dimensions=*/{})
          : MakeConstant(builder, in_size, i32_type, col_shape);

  mlir::MlirOp row_scaled = mlir::stablehlo::Mul(row_iota, stride_const);
  mlir::MlirOp unclipped_starts =
      mlir::stablehlo::Subtract(row_scaled, pad_const);
  mlir::MlirOp unclipped_ends =
      mlir::stablehlo::Add(unclipped_starts, kernel_const);
  mlir::MlirOp starts = mlir::stablehlo::Max(unclipped_starts, zero_i32);
  mlir::MlirOp ends = mlir::stablehlo::Min(unclipped_ends, in_size_col);

  mlir::MlirOp window_weights = BuildAvgPool1dWindowWeights(
      builder, in_size, out_size, kernel, stride, pad, ceil_mode,
      count_include_pad, divisor_mode, divisor_override_val,
      runtime_in_size.has_value(), in_size_col, unclipped_starts,
      unclipped_ends, starts, ends, compute_dtype);

  return BuildPooling1dWeightMatrix(builder, in_size, out_size, starts, ends,
                                    window_weights, compute_dtype);
}

absl::StatusOr<mlir::MlirOp> ZeroAndUnboundPoolingAxis(
    mlir::MlirBuilder& builder, mlir::MlirOp current, const int64_t dim,
    const int64_t contract_bound) {
  if (!GetTensorTypeOrDie(current).isDynamicDim(dim)) {
    return current;
  }
  mlir::MlirOp runtime_current_size =
      mlir::stablehlo::GetDimensionSize(current, dim);
  const mlir::Type i32_type = builder.getOpBuilder().getI32Type();
  mlir::MlirOp bound_op = MakeScalarConstant(
      builder, static_cast<int32_t>(contract_bound), i32_type);
  current = mlir::stablehlo::SetDimensionSize(current, bound_op, dim);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp index, Iota(current, dim, i32_type));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp size,
                      BroadcastIfNeeded(runtime_current_size, index));
  mlir::MlirOp in_bounds = mlir::stablehlo::Compare(
      index, size, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp zero = MakeConstantLike(current, 0);
  return mlir::stablehlo::Select(in_bounds, current, zero);
}

// Applies 1D average pooling (forward or backward) along `dim` via an MXU
// einsum contraction with the 1D pooling weight matrix.
absl::StatusOr<mlir::MlirOp> ApplyAvgPool1D(
    mlir::MlirBuilder& builder, mlir::MlirOp current,
    const int64_t spatial_dim_count, const int64_t dim, const int64_t in_size,
    const int64_t out_size, const int64_t kernel, const int64_t stride,
    const int64_t pad, const bool ceil_mode, const bool count_include_pad,
    const AvgPoolAxisDivisorMode divisor_mode,
    const int64_t divisor_override_val, const PoolingDirection direction,
    std::optional<mlir::MlirOp> runtime_in_size,
    const mlir::stablehlo::Precision precision) {
  const bool is_unit_divisor =
      divisor_mode != AvgPoolAxisDivisorMode::kConstantOverride ||
      divisor_override_val == 1;
  if (in_size == out_size && kernel == 1 && stride == 1 && pad == 0 &&
      is_unit_divisor) {
    return current;
  }
  const int64_t contract_bound =
      direction == PoolingDirection::kForward ? in_size : out_size;
  TT_ASSIGN_OR_RETURN(current, ZeroAndUnboundPoolingAxis(builder, current, dim,
                                                         contract_bound));
  const mlir::Type compute_dtype = GetTensorTypeOrDie(current).getElementType();
  mlir::MlirOp weight_mat = BuildAvgPool1dWeightMatrix(
      builder, in_size, out_size, kernel, stride, pad, ceil_mode,
      count_include_pad, divisor_mode, divisor_override_val, runtime_in_size,
      compute_dtype);
  mlir::MlirOp contract_ops[] = {current, weight_mat};
  TT_ASSIGN_OR_RETURN(mlir::MlirOp result,
                      BuildEinsumShlo(contract_ops,
                                      GetPoolingEinsumEquation(
                                          spatial_dim_count, dim, direction),
                                      precision));
  if (!runtime_in_size.has_value()) {
    return result;
  }
  const int64_t result_bound =
      direction == PoolingDirection::kForward ? out_size : in_size;
  if (result_bound <= 1) {
    return result;
  }
  mlir::MlirOp runtime_result_size =
      direction == PoolingDirection::kForward
          ? BuildRuntimePoolingOutputSize(builder, *runtime_in_size, kernel,
                                          stride, pad, ceil_mode)
          : *runtime_in_size;
  return mlir::stablehlo::SetDimensionSize(result, runtime_result_size, dim);
}

absl::StatusOr<mlir::MlirOp> BuildAvgPoolEinsumShlo(
    mlir::MlirBuilder& builder, mlir::MlirOp current, mlir::MlirOp batch_input,
    mlir::ArrayRef<int64_t> input_static_shape,
    mlir::ArrayRef<int64_t> out_spatial_sizes, const int64_t spatial_dim_count,
    mlir::ArrayRef<int64_t> kernel_size_attr,
    mlir::ArrayRef<int64_t> stride_attr, mlir::ArrayRef<int64_t> padding_attr,
    const bool ceil_mode, const bool count_include_pad,
    const std::optional<int64_t> divisor_override,
    const PoolingDirection direction,
    const mlir::stablehlo::Precision precision) {
  const mlir::RankedTensorType batch_input_type =
      GetTensorTypeOrDie(batch_input);
  for (int64_t step = 0; step < spatial_dim_count; ++step) {
    const int64_t d = direction == PoolingDirection::kForward
                          ? step
                          : (spatial_dim_count - 1 - step);
    const int64_t dim = 2 + d;
    const int64_t in_size = input_static_shape[dim];
    const int64_t out_size = out_spatial_sizes[d];
    std::optional<mlir::MlirOp> runtime_in_size;
    if (batch_input_type.isDynamicDim(dim)) {
      runtime_in_size = mlir::stablehlo::GetDimensionSize(batch_input, dim);
    }
    AvgPoolAxisDivisorMode divisor_mode = AvgPoolAxisDivisorMode::kWindowCount;
    int64_t divisor_override_val = 1;
    if (divisor_override.has_value()) {
      if (d == spatial_dim_count - 1) {
        divisor_mode = AvgPoolAxisDivisorMode::kConstantOverride;
        divisor_override_val = *divisor_override;
      } else {
        divisor_mode = AvgPoolAxisDivisorMode::kUnit;
      }
    }
    TT_ASSIGN_OR_RETURN(
        current,
        ApplyAvgPool1D(builder, current, spatial_dim_count, dim, in_size,
                       out_size, kernel_size_attr[d], stride_attr[d],
                       padding_attr[d], ceil_mode, count_include_pad,
                       divisor_mode, divisor_override_val, direction,
                       runtime_in_size, precision));
  }
  return current;
}

absl::StatusOr<mlir::MlirOp> BuildAvgPoolShlo(
    mlir::MlirOp input, const int64_t spatial_dim_count, Dimensions kernel_size,
    Dimensions stride, Dimensions padding, const bool ceil_mode,
    const bool count_include_pad, const std::optional<int64_t> divisor_override,
    const mlir::stablehlo::Precision precision) {
  const mlir::ElementType orig_dtype = GetElementTypeOrDie(input);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(orig_dtype));
  TT_ASSIGN_OR_RETURN(input, CastIfNeeded(input, compute_dtype));
  auto& builder = input.getBuilder();

  // 1. Create a batch input by normalizing the input tensor.
  TT_ASSIGN_OR_RETURN(auto batch_input_info,
                      CreateBatchInput(input, spatial_dim_count));
  mlir::MlirOp batch_input = batch_input_info.batch_input;
  const int64_t original_dim_size = batch_input_info.original_dim_size;
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(batch_input);
  const Dimensions input_static_shape = GetStaticShape(batch_input);

  // 2. Expand attributes to match the spatial_dim_count.
  TT_ASSIGN_OR_RETURN(Dimensions kernel_size_attr,
                      ExpandAttribute(kernel_size, spatial_dim_count));
  Dimensions stride_attr;
  if (stride.empty()) {
    stride_attr = kernel_size_attr;
  } else {
    TT_ASSIGN_OR_RETURN(stride_attr,
                        ExpandAttribute(stride, spatial_dim_count));
  }
  Dimensions padding_attr;
  if (padding.empty()) {
    padding_attr = Dimensions(spatial_dim_count, 0);
  } else {
    TT_ASSIGN_OR_RETURN(padding_attr,
                        ExpandAttribute(padding, spatial_dim_count));
  }
  const Dimensions dilation_attr(spatial_dim_count, 1);

  TT_ASSIGN_OR_RETURN(
      const Dimensions output_size,
      GetPoolingOutputSize(input_static_shape, kernel_size_attr, stride_attr,
                           padding_attr, dilation_attr, ceil_mode,
                           spatial_dim_count));
  if (absl::c_linear_search(input_static_shape, 0)) {
    mlir::MlirOp zero_out =
        MakeConstant(builder, 0, input_type.getElementType(), output_size);
    mlir::MlirOp result =
        RemoveTrivialBatch(zero_out, original_dim_size, spatial_dim_count);
    return CastIfNeeded(result, orig_dtype);
  }

  const mlir::ArrayRef<int64_t> out_spatial_sizes =
      AsArrayRef<int64_t>(output_size).drop_front(2);
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp pooled,
      BuildAvgPoolEinsumShlo(builder, batch_input, batch_input,
                             AsArrayRef<int64_t>(input_static_shape),
                             out_spatial_sizes, spatial_dim_count,
                             kernel_size_attr, stride_attr, padding_attr,
                             ceil_mode, count_include_pad, divisor_override,
                             PoolingDirection::kForward, precision));
  mlir::MlirOp result =
      RemoveTrivialBatch(pooled, original_dim_size, spatial_dim_count);
  return CastIfNeeded(result, orig_dtype);
}

absl::StatusOr<mlir::MlirOp> BuildAvgPoolBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp input,
    const int64_t spatial_dim_count, const Dimensions& kernel_size,
    const Dimensions& stride, const Dimensions& padding, const bool ceil_mode,
    const bool count_include_pad, const std::optional<int64_t> divisor_override,
    const mlir::stablehlo::Precision precision) {
  const mlir::ElementType orig_dtype = GetElementTypeOrDie(grad_output);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(orig_dtype));
  TT_ASSIGN_OR_RETURN(grad_output, CastIfNeeded(grad_output, compute_dtype));
  auto& builder = grad_output.getBuilder();

  // 1. Create batch inputs for both input and grad_output.
  TT_ASSIGN_OR_RETURN(auto batch_input_info,
                      CreateBatchInput(input, spatial_dim_count));
  TT_ASSIGN_OR_RETURN(auto batch_grad_output_info,
                      CreateBatchInput(grad_output, spatial_dim_count));
  mlir::MlirOp batch_input = batch_input_info.batch_input;
  mlir::MlirOp batch_grad_output = batch_grad_output_info.batch_input;

  const mlir::RankedTensorType grad_output_type =
      GetTensorTypeOrDie(batch_grad_output);
  const Dimensions input_static_shape = GetStaticShape(batch_input);
  const Dimensions grad_output_static_shape = GetStaticShape(batch_grad_output);
  if (absl::c_linear_search(input_static_shape, 0) ||
      absl::c_linear_search(grad_output_static_shape, 0)) {
    mlir::MlirOp zero_grad = MakeConstant(
        builder, 0, grad_output_type.getElementType(), input_static_shape);
    mlir::MlirOp result = RemoveTrivialBatch(
        zero_grad, batch_input_info.original_dim_size, spatial_dim_count);
    return CastIfNeeded(result, orig_dtype);
  }

  // 2. Expand attributes to match the spatial_dim_count.
  TT_ASSIGN_OR_RETURN(Dimensions kernel_size_attr,
                      ExpandAttribute(kernel_size, spatial_dim_count));
  Dimensions stride_attr = kernel_size_attr;
  if (!stride.empty()) {
    TT_ASSIGN_OR_RETURN(stride_attr,
                        ExpandAttribute(stride, spatial_dim_count));
  }
  Dimensions padding_attr(spatial_dim_count, 0);
  if (!padding.empty()) {
    TT_ASSIGN_OR_RETURN(padding_attr,
                        ExpandAttribute(padding, spatial_dim_count));
  }

  const mlir::ArrayRef<int64_t> out_spatial_sizes =
      AsArrayRef<int64_t>(grad_output_static_shape).drop_front(2);
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp grad_in,
      BuildAvgPoolEinsumShlo(builder, batch_grad_output, batch_input,
                             AsArrayRef<int64_t>(input_static_shape),
                             out_spatial_sizes, spatial_dim_count,
                             kernel_size_attr, stride_attr, padding_attr,
                             ceil_mode, count_include_pad, divisor_override,
                             PoolingDirection::kBackward, precision));
  mlir::MlirOp result = RemoveTrivialBatch(
      grad_in, batch_input_info.original_dim_size, spatial_dim_count);
  return CastIfNeeded(result, orig_dtype);
}

auto GetAvgPoolOpBuilder(at::IntArrayRef kernel_size, at::IntArrayRef stride,
                         at::IntArrayRef padding, const bool ceil_mode,
                         const bool count_include_pad,
                         const std::optional<int64_t> divisor_override,
                         const int64_t spatial_dim_count,
                         const mlir::stablehlo::Precision precision) {
  return
      [kernel_size_vec = CopyIntVector(kernel_size),
       stride_vec = CopyIntVector(stride), padding_vec = CopyIntVector(padding),
       ceil_mode, count_include_pad, divisor_override, spatial_dim_count,
       precision](mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
        return BuildAvgPoolShlo(input_op, spatial_dim_count, kernel_size_vec,
                                stride_vec, padding_vec, ceil_mode,
                                count_include_pad, divisor_override, precision);
      };
}

absl::Status ValidateAvgPool1dDtype(const at::Tensor& self) {
  TT_RET_CHECK(self.numel() == 0 || self.scalar_type() != at::ScalarType::Long,
               error::kPythonNotImplementedError)
      << "not implemented for " << ToString(self.scalar_type());
  TT_RET_CHECK(self.scalar_type() != at::ScalarType::Short &&
                   self.scalar_type() != at::ScalarType::Int &&
                   self.scalar_type() != at::ScalarType::Char &&
                   self.scalar_type() != at::ScalarType::Byte &&
                   self.scalar_type() != at::ScalarType::ComplexFloat &&
                   self.scalar_type() != at::ScalarType::Bool,
               error::kInvalidArgument)
      << "expected input dtype to be none of (bool, uint8, int8, "
         "int16, int32, complex64), got "
      << ToString(self.scalar_type());
  return absl::OkStatus();
}

}  // namespace

// Helper function to build and dispatch N-dimensional average pooling
// operations.
absl::StatusOr<DeviceBufferRef> BuildAvgPoolNd(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override, mlir::ElementType out_dtype,
    at::IntArrayRef out_sizes, int64_t spatial_dim_count,
    OpParamCacheKeys param_keys, std::optional<OpName> override_op_name) {
  const mlir::stablehlo::Precision precision = GetAvgPoolPrecision(param_keys);
  auto op_builder = GetAvgPoolOpBuilder(kernel_size, stride, padding, ceil_mode,
                                        count_include_pad, divisor_override,
                                        spatial_dim_count, precision);

  return DispatchOp<1>(std::move(op_builder), self,
                       {.op_name = override_op_name,
                        .out_dtype = out_dtype,
                        .out_dims = CopyIntVector(out_sizes),
                        .op_param_cache_keys = std::move(param_keys)});
}

absl::StatusOr<at::Tensor> BuildAvgPoolOutNd(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override, at::Tensor& out,
    int64_t spatial_dim_count, OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(
      const Dimensions output_size,
      GetPoolingOutputSize(self.sizes(), kernel_size, stride, padding,
                           at::IntArrayRef({1}), ceil_mode, spatial_dim_count));
  TT_RETURN_IF_ERROR(ValidateOutDtype(out, self.scalar_type(),
                                      /*allow_out_dtype_cast=*/false));
  TT_ASSIGN_OR_RETURN(auto out_type,
                      ConvertTo<mlir::ElementType>(out.scalar_type()));
  const mlir::stablehlo::Precision precision = GetAvgPoolPrecision(param_keys);
  auto op_builder = GetAvgPoolOpBuilder(kernel_size, stride, padding, ceil_mode,
                                        count_include_pad, divisor_override,
                                        spatial_dim_count, precision);
  TT_RETURN_IF_ERROR(
      DispatchOpOut<1>(std::move(op_builder), self, out,
                       {.out_dtype = out_type,
                        .out_dims = output_size,
                        .op_param_cache_keys = std::move(param_keys)}));
  return out;
}

// Helper function to build and dispatch N-dimensional average pooling backward
// operations.
absl::StatusOr<at::Tensor> BuildAvgPoolBackwardGradInputNd(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override, at::Tensor& grad_input,
    int64_t spatial_dim_count, OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateOutDtype(grad_input, self.scalar_type(),
                                      /*allow_out_dtype_cast=*/false));
  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(grad_input.scalar_type()));
  const mlir::stablehlo::Precision precision = GetAvgPoolPrecision(param_keys);

  auto op_builder = [kernel_size_vec = CopyIntVector(kernel_size),
                     stride_vec = CopyIntVector(stride),
                     padding_vec = CopyIntVector(padding), ceil_mode,
                     count_include_pad, divisor_override, spatial_dim_count,
                     precision](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    return BuildAvgPoolBackwardShlo(
        inputs[0], inputs[1], spatial_dim_count, kernel_size_vec, stride_vec,
        padding_vec, ceil_mode, count_include_pad, divisor_override, precision);
  };

  DispatchOpOptions<1> options = {
      .out_dtype = output_dtype,
      .out_dims = self.sizes(),
      .op_param_cache_keys = std::move(param_keys),
  };
  TT_RETURN_IF_ERROR(DispatchOpOut<2>(std::move(op_builder),
                                      {grad_output, self}, grad_input,
                                      std::move(options)));
  return grad_input;
}

at::Tensor AtenAvgPool1d(const at::Tensor& self, at::IntArrayRef kernel_size,
                         at::IntArrayRef stride, at::IntArrayRef padding,
                         bool ceil_mode, bool count_include_pad) {
  TT_KERNEL(
      OpName::kAvgPool1d, param_keys,
      (self, kernel_size, stride, padding, ceil_mode, count_include_pad), {
        TT_THROW_IF_ERROR(ValidateAvgPool1dDtype(self));

        TT_ASSIGN_OR_THROW(
            const Dimensions output_size,
            GetPoolingOutputSize(self.sizes(), kernel_size, stride, padding,
                                 at::IntArrayRef({1}), ceil_mode,
                                 /*spatial_dim_count=*/1));
        TT_ASSIGN_OR_THROW(const auto out_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        TT_ASSIGN_OR_THROW(
            auto result_buf,
            BuildAvgPoolNd(self, kernel_size, stride, padding, ceil_mode,
                           count_include_pad,
                           /*divisor_override=*/std::nullopt, out_type,
                           output_size, /*spatial_dim_count=*/1,
                           std::move(param_keys)));
        return MakeTensor(std::move(result_buf));
      });
}

at::Tensor& AtenAvgPool1dOut(const at::Tensor& self,
                             at::IntArrayRef kernel_size,
                             at::IntArrayRef stride, at::IntArrayRef padding,
                             bool ceil_mode, bool count_include_pad,
                             at::Tensor& out) {
  TT_KERNEL(
      OpName::kAvgPool1dOut, param_keys,
      (self, kernel_size, stride, padding, ceil_mode, count_include_pad, out), {
        TT_THROW_IF_ERROR(ValidateAvgPool1dDtype(self));

        TT_ASSIGN_OR_THROW(
            auto result,
            BuildAvgPoolOutNd(self, kernel_size, stride, padding, ceil_mode,
                              count_include_pad,
                              /*divisor_override=*/std::nullopt, out,
                              /*spatial_dim_count=*/1, std::move(param_keys)));
        return out;
      });
}

at::Tensor AtenAvgPool1dAutograd(const at::Tensor& self,
                                 at::IntArrayRef kernel_size,
                                 at::IntArrayRef stride,
                                 at::IntArrayRef padding, bool ceil_mode,
                                 bool count_include_pad) {
  // Differentiation (reverse or forward mode), FakeTensors and functorch
  // transforms use the upstream decomposition (unsqueeze, avg_pool2d,
  // squeeze), whose ops have derivative formulas. Other calls use the direct
  // 1D kernel.
  if (IsFakeOrFunctorchTensor(self) || self._fw_grad(/*level=*/0).defined() ||
      (c10::GradMode::is_enabled() && self.requires_grad())) {
    return at::native::avg_pool1d(self, kernel_size, stride, padding, ceil_mode,
                                  count_include_pad);
  }
  at::AutoDispatchBelowAutograd guard;
  return AtenAvgPool1d(self, kernel_size, stride, padding, ceil_mode,
                       count_include_pad);
}

at::Tensor& AtenAvgPool2dOut(const at::Tensor& self,
                             at::IntArrayRef kernel_size,
                             at::IntArrayRef stride, at::IntArrayRef padding,
                             bool ceil_mode, bool count_include_pad,
                             std::optional<int64_t> divisor_override,
                             at::Tensor& out) {
  TT_KERNEL(
      OpName::kAvgPool2dOut, param_keys,
      (self, kernel_size, stride, padding, ceil_mode, count_include_pad,
       divisor_override, out),
      {
        TT_CHECK_THROW(
            self.numel() == 0 || self.scalar_type() != at::ScalarType::Long,
            error::kPythonNotImplementedError)
            << "not implemented for " << ToString(self.scalar_type());
        TT_CHECK_THROW(self.scalar_type() != at::ScalarType::Bool &&
                           self.scalar_type() != at::ScalarType::Short &&
                           self.scalar_type() != at::ScalarType::Int &&
                           self.scalar_type() != at::ScalarType::Char &&
                           self.scalar_type() != at::ScalarType::Byte &&
                           self.scalar_type() != at::ScalarType::ComplexFloat,
                       error::kInvalidArgument)
            << "expected input dtype to be none of (bool, uint8, int8, int16, "
               "int32, complex64), got "
            << ToString(self.scalar_type());

        TT_ASSIGN_OR_THROW(
            auto result,
            BuildAvgPoolOutNd(self, kernel_size, stride, padding, ceil_mode,
                              count_include_pad, divisor_override, out,
                              /*spatial_dim_count=*/2, std::move(param_keys)));
        return out;
      });
}

at::Tensor& AtenAvgPool3dOut(const at::Tensor& self,
                             at::IntArrayRef kernel_size,
                             at::IntArrayRef stride, at::IntArrayRef padding,
                             bool ceil_mode, bool count_include_pad,
                             std::optional<int64_t> divisor_override,
                             at::Tensor& out) {
  TT_KERNEL(
      OpName::kAvgPool3dOut, param_keys,
      (self, kernel_size, stride, padding, ceil_mode, count_include_pad,
       divisor_override, out),
      {
        TT_CHECK_THROW(self.scalar_type() != at::ScalarType::Long,
                       error::kPythonNotImplementedError)
            << "not implemented for " << ToString(self.scalar_type());
        TT_CHECK_THROW(self.scalar_type() != at::ScalarType::Bool &&
                           self.scalar_type() != at::ScalarType::Byte &&
                           self.scalar_type() != at::ScalarType::Char &&
                           self.scalar_type() != at::ScalarType::Short &&
                           self.scalar_type() != at::ScalarType::Int &&
                           self.scalar_type() != at::ScalarType::ComplexFloat &&
                           self.scalar_type() != at::ScalarType::Half,
                       error::kInvalidArgument)
            << "expected input dtype to be none of (bool, uint8, int8, "
               "int16, int32, complex64), got "
            << ToString(self.scalar_type());

        TT_ASSIGN_OR_THROW(
            auto result,
            BuildAvgPoolOutNd(self, kernel_size, stride, padding, ceil_mode,
                              count_include_pad, divisor_override, out,
                              k3dSpatialDimCount, std::move(param_keys)));
        return out;
      });
}

at::Tensor& AtenAvgPool2dBackwardGradInput(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override, at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kAvgPool2dBackwardGradInput, param_keys,
      (grad_output, self, kernel_size, stride, padding, ceil_mode,
       count_include_pad, divisor_override, grad_input),
      {
        TT_ASSIGN_OR_THROW(
            auto result,
            BuildAvgPoolBackwardGradInputNd(
                grad_output, self, kernel_size, stride, padding, ceil_mode,
                count_include_pad, divisor_override, grad_input,
                /*spatial_dim_count=*/2, std::move(param_keys)));
        return grad_input;
      });
}

at::Tensor& AtenAvgPool3dBackwardGradInput(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override, at::Tensor& grad_input) {
  TT_KERNEL(OpName::kAvgPool3dBackwardGradInput, param_keys,
            (grad_output, self, kernel_size, stride, padding, ceil_mode,
             count_include_pad, divisor_override, grad_input),
            {
              TT_ASSIGN_OR_THROW(
                  auto result,
                  BuildAvgPoolBackwardGradInputNd(
                      grad_output, self, kernel_size, stride, padding,
                      ceil_mode, count_include_pad, divisor_override,
                      grad_input, k3dSpatialDimCount, std::move(param_keys)));
              return grad_input;
            });
}
}  // namespace torch_tpu
