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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/avg_pool1d_native.h"
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
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/pooling/pooling.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
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

// Scales the output of the pooling reduction (sum) by the divisor.
//
// For floating point element types with a constant divisor, we multiply by the
// reciprocal (1.0 / divisor) using MulOp, avoiding hardware vector division
// which is significantly slower on TPU. If divisor is 1, this is a no-op.
// For dynamic divisors (e.g., asymmetric padding with ceil_mode, or when
// count_include_pad is false), we compute the per-window count tensor and
// divide.
absl::StatusOr<mlir::MlirOp> ScaleAvgPoolOutput(
    mlir::MlirOp dividend, const mlir::RankedTensorType& input_type,
    const ReduceWindowAttributes& reduce_window_attributes,
    bool count_include_pad, std::optional<int64_t> divisor_override) {
  auto& builder = dividend.getBuilder();
  const mlir::RankedTensorType dividend_type = GetTensorTypeOrDie(dividend);
  const mlir::Type element_type = dividend_type.getElementType();

  // Helper to scale dividend by a constant integer divisor.
  auto scale_by_constant_divisor = [&]  // REF_CAPTURE_OK=called synchronously
      (int64_t divisor_val) -> absl::StatusOr<mlir::MlirOp> {
    if (divisor_val == 1) {
      return dividend;
    }
    if (llvm::isa<mlir::FloatType>(element_type)) {
      const double reciprocal = 1.0 / static_cast<double>(divisor_val);
      auto scale_const = MakeConstantLike(dividend, reciprocal);
      return mlir::stablehlo::Mul(dividend, scale_const);
    }
    auto divisor_const = MakeConstantLike(dividend, divisor_val);
    return mlir::stablehlo::Div(dividend, divisor_const);
  };

  // Case A: Divisor is specified
  if (divisor_override.has_value()) {
    return scale_by_constant_divisor(divisor_override.value());
  }

  // Case B: Count includes padding, so divisor is the product of window
  // dimensions, EXCEPT when ceil_mode adds implicit padding.
  if (count_include_pad) {
    auto padding_attr = reduce_window_attributes.padding.getValues<int64_t>();
    llvm::SmallVector<int64_t> padding_vals(padding_attr.begin(),
                                            padding_attr.end());
    bool has_extra_padding = false;

    // Check for asymmetric padding (extra padding on the right added by
    // ceil_mode)
    for (size_t i = 0; i < padding_vals.size(); i += 2) {
      if (padding_vals[i] != padding_vals[i + 1]) {
        has_extra_padding = true;
        break;
      }
    }

    if (!has_extra_padding) {
      // Fast path: Symmetric padding, window size is constant.
      const int64_t window_size = std::accumulate(
          reduce_window_attributes.window_dimensions.asArrayRef().begin(),
          reduce_window_attributes.window_dimensions.asArrayRef().end(),
          int64_t{1}, std::multiplies<int64_t>());
      return scale_by_constant_divisor(window_size);
    }

    // Dynamically compute divisor: explicitly padded regions count as 1s,
    // but the extra out-of-bounds padding created by ceil_mode counts as 0s.
    mlir::MlirOp ones =
        MakeConstant(builder, 1, element_type, input_type.getShape());
    mlir::MlirOp one_scalar = MakeScalarConstant(builder, 1, element_type);
    mlir::MlirOp zero_scalar = MakeScalarConstant(builder, 0, element_type);

    const int64_t rank = input_type.getRank();
    Dimensions edge_padding_low(rank, 0);
    Dimensions edge_padding_high(rank, 0);
    Dimensions interior_padding(rank, 0);

    llvm::SmallVector<int64_t> reduce_padding;
    reduce_padding.reserve(rank * 2);

    for (int i = 0; i < rank; ++i) {
      const int64_t pad_low = padding_vals[2 * i];
      const int64_t pad_high = padding_vals[2 * i + 1];
      TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Expanded padding values are not
                     // reachable by the user.
          pad_high >= pad_low, error::kInvalidArgument)
          << "expected pad_high >= pad_low for implicit ceil_mode padding, "
             "got "
          << pad_high << " vs " << pad_low;
      const int64_t extra_pad = pad_high - pad_low;

      edge_padding_low[i] = pad_low;
      edge_padding_high[i] =
          pad_low;  // Pad both sides symmetrically with explicit padding

      reduce_padding.push_back(0);
      reduce_padding.push_back(
          extra_pad);  // ReduceWindow handles the extra implicit padding
    }

    mlir::MlirOp padded_ones =
        mlir::stablehlo::Pad(ones, one_scalar, edge_padding_low,
                             edge_padding_high, interior_padding);

    auto add_reduce_builder = [element_type](mlir::RegionBuilder& rb) {
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
          element_type, rb.getRegion(), rb.getOpBuilder());
    };

    auto reduce_padding_attr = mlir::DenseIntElementsAttr::get(
        mlir::makeTensorType(builder.getContext(), {rank, 2},
                             builder.getOpBuilder().getI64Type()),
        reduce_padding);

    mlir::MlirOp divisor = mlir::stablehlo::ReduceWindow(
        builder,
        /*inputs=*/{padded_ones},
        /*init_values=*/{zero_scalar},
        /*body=*/add_reduce_builder, reduce_window_attributes.window_dimensions,
        reduce_window_attributes.window_strides,
        reduce_window_attributes.base_dilations,
        reduce_window_attributes.window_dilations, reduce_padding_attr)[0];
    return mlir::stablehlo::Div(dividend, divisor);
  }

  // Case C: Count does not include padding, so divisor is the number of the
  // elements in the output
  mlir::MlirOp ones =
      MakeConstant(builder, 1, element_type, input_type.getShape());
  mlir::MlirOp init_value = MakeScalarConstant(builder, 0, element_type);

  auto add_reduce_builder = [element_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        element_type, rb.getRegion(), rb.getOpBuilder());
  };

  mlir::MlirOp divisor = mlir::stablehlo::ReduceWindow(
      builder,
      /*inputs=*/{ones},
      /*init_values=*/{init_value},
      /*body=*/add_reduce_builder, reduce_window_attributes.window_dimensions,
      reduce_window_attributes.window_strides,
      reduce_window_attributes.base_dilations,
      reduce_window_attributes.window_dilations,
      reduce_window_attributes.padding)[0];
  return mlir::stablehlo::Div(dividend, divisor);
}

absl::StatusOr<mlir::MlirOp> BuildAvgPoolShlo(
    mlir::MlirOp input, int64_t spatial_dim_count, Dimensions kernel_size,
    Dimensions stride, Dimensions padding, bool ceil_mode,
    bool count_include_pad, std::optional<int64_t> divisor_override) {
  const mlir::ElementType orig_dtype = GetElementTypeOrDie(input);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(orig_dtype));
  TT_ASSIGN_OR_RETURN(input, CastIfNeeded(input, compute_dtype));
  auto& builder = input.getBuilder();

  // 1. Create a batch input by normalizing the input tensor
  TT_ASSIGN_OR_RETURN(auto batch_input_info,
                      CreateBatchInput(input, spatial_dim_count));
  mlir::MlirOp batch_input = batch_input_info.batch_input;
  const int64_t original_dim_size = batch_input_info.original_dim_size;
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(batch_input);
  const mlir::Type element_type = input_type.getElementType();
  const int64_t num_dims = input_type.getRank();

  // 2. Expand attributes to match the spatial_dim_count.
  // The parameters kernel_size, stride, padding, dilation can either be a
  // single int or a tuple of ints.
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
  Dimensions dilation_attr(spatial_dim_count, 1);

  // 3. Compute the padding pairs given the padding and ceil mode.
  auto ceil_padding_pairs =
      CeilModePadding(input_type, kernel_size_attr, stride_attr, padding_attr,
                      dilation_attr, ceil_mode);

  // 4. Compute the final reduce window attributes.
  ReduceWindowAttributes reduce_window_attributes = GetReduceWindowAttributes(
      builder, kernel_size_attr, stride_attr, dilation_attr, ceil_padding_pairs,
      spatial_dim_count, num_dims);

  // 5. Compute the avg_pool using a single ReduceWindow.
  mlir::MlirOp init_value = MakeScalarConstant(builder, 0, element_type);

  auto add_reduce_builder = [element_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        element_type, rb.getRegion(), rb.getOpBuilder());
  };

  mlir::MlirOp sum_output = mlir::stablehlo::ReduceWindow(
      builder,
      /*inputs=*/{batch_input},
      /*init_values=*/{init_value},
      /*body=*/add_reduce_builder, reduce_window_attributes.window_dimensions,
      reduce_window_attributes.window_strides,
      reduce_window_attributes.base_dilations,
      reduce_window_attributes.window_dilations,
      reduce_window_attributes.padding)[0];

  TT_ASSIGN_OR_RETURN(
      auto final_output,
      ScaleAvgPoolOutput(sum_output, input_type, reduce_window_attributes,
                         count_include_pad, divisor_override));

  auto result =
      RemoveTrivialBatch(final_output, original_dim_size, spatial_dim_count);
  return CastIfNeeded(result, orig_dtype);
}

// Mathematically, average pooling is a linear transformation that can be
// decomposed into two steps:
// * A Convolution with a kernel consisting entirely of 1s.
// * Element-wise division by the counts.
//
// Forward:  Y = Conv(X, 1s) / Count
// Backward: dX = ConvBackprop(dY / Count, 1s)
//
// The input gradients are computed by a convolution of the output gradients
// and the filter, with some appropriate padding.
// The filter shape is [ H, W, ..., inC, outC ]
//
// For an explanation of backpropagation for convolution, see the comments
// in third_party/tensorflow/core/kernels/conv_grad_ops.h
//
// The following key steps are performed to compute the input gradients:
//
// 1. Normalize: normalized_grad_output = grad_output / divisor
//
// 2. Simulate conv backpropagation:
//    We use a `ReduceWindow` op to simulate the convolution of the
//    normalized gradients with a kernel of 1s.
//
//    a. Dilation (Handling Stride):
//       If the Forward pass had stride > 1, the output gradients are sparse
//       relative to the input. Insert holes (zeros) between gradient elements.
//       Mapping: Forward strides -> Backward base dilation.
//
//    b. Full Padding (Full Convolution):
//       We pad the gradient tensor by (Kernel - 1) on both sides.
//       This allows the sliding window to fully cross the boundaries.
//
//    c. Accumulation:
//       ReduceWindow with AddOp accumulates the contributions from overlapping
//       windows, which is mathematically equivalent to the convolution sum.
//
// 3. Slice (Crop to original size):
//    The result of the full convolution corresponds to the gradient of the
//    padded input. We must slice this result to remove the regions
//    corresponding to the forward padding, returning a tensor matching
//    the original input shape.
mlir::MlirOp SliceBackwardResult(
    mlir::MlirOp full_conv_result, mlir::MlirOp init_value,
    const mlir::RankedTensorType& input_type,
    const std::vector<PaddingParams>& ceil_padding_pairs, int64_t num_dims,
    int64_t spatial_dim_count) {
  Dimensions start_indices(num_dims, 0);
  Dimensions limit_indices(num_dims, 0);
  Dimensions slice_strides(num_dims, 1);

  const auto conv_shape = GetTensorTypeOrDie(full_conv_result).getShape();
  Dimensions result_pad_high(num_dims, 0);
  bool needs_padding = false;

  for (int i = 0; i < num_dims; ++i) {
    if (i < num_dims - spatial_dim_count) {  // Batch/Channel dims
      start_indices[i] = 0;
      limit_indices[i] = input_type.getShape()[i];
    } else {  // Spatial dims
      const int spatial_idx = i - (num_dims - spatial_dim_count);
      const int64_t fwd_pad_low = ceil_padding_pairs[spatial_idx].left;
      start_indices[i] = fwd_pad_low;
      limit_indices[i] = fwd_pad_low + input_type.getShape()[i];

      // If the limit is out of bounds, we need to pad the result.
      if (limit_indices[i] > conv_shape[i]) {
        result_pad_high[i] = limit_indices[i] - conv_shape[i];
        needs_padding = true;
      }
    }
  }

  mlir::MlirOp slicable_result = full_conv_result;
  if (needs_padding) {
    // Pad the high dimension to accommodate the slice limit
    slicable_result =
        mlir::stablehlo::Pad(full_conv_result, init_value,
                             /*low=*/Dimensions(num_dims, 0),
                             /*high=*/result_pad_high,
                             /*interior=*/Dimensions(num_dims, 0));
  }

  return mlir::stablehlo::Slice(slicable_result, start_indices, limit_indices,
                                slice_strides);
}

absl::StatusOr<mlir::MlirOp> BuildAvgPoolBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp input, int64_t spatial_dim_count,
    const Dimensions& kernel_size, const Dimensions& stride,
    const Dimensions& padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  const mlir::ElementType orig_dtype = GetElementTypeOrDie(grad_output);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(orig_dtype));
  TT_ASSIGN_OR_RETURN(grad_output, CastIfNeeded(grad_output, compute_dtype));
  TT_ASSIGN_OR_RETURN(input, CastIfNeeded(input, compute_dtype));
  auto& builder = grad_output.getBuilder();

  // 1. Create batch inputs for both input and grad_output
  TT_ASSIGN_OR_RETURN(auto batch_input_info,
                      CreateBatchInput(input, spatial_dim_count));
  TT_ASSIGN_OR_RETURN(auto batch_grad_output_info,
                      CreateBatchInput(grad_output, spatial_dim_count));
  mlir::MlirOp batch_input = batch_input_info.batch_input;
  mlir::MlirOp batch_grad_output = batch_grad_output_info.batch_input;

  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(batch_input);
  const mlir::Type element_type = input_type.getElementType();
  const int64_t num_dims = input_type.getRank();

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
  Dimensions dilation_attr(spatial_dim_count, 1);

  // 3. Compute the padding pairs given the padding and ceil mode.
  auto ceil_padding_pairs =
      CeilModePadding(input_type, kernel_size_attr, stride_attr, padding_attr,
                      dilation_attr, ceil_mode);

  // 4. Compute the final reduce window attributes.
  ReduceWindowAttributes reduce_window_attributes = GetReduceWindowAttributes(
      builder, kernel_size_attr, stride_attr, dilation_attr, ceil_padding_pairs,
      spatial_dim_count, num_dims);

  // 5. Normalize grad_output by the divisor.
  TT_ASSIGN_OR_RETURN(auto normalized_grad_output,
                      ScaleAvgPoolOutput(batch_grad_output, input_type,
                                         reduce_window_attributes,
                                         count_include_pad, divisor_override));

  // 6. Perform transposed convolution with interior/edge padding
  auto bw_window_dimensions = reduce_window_attributes.window_dimensions;
  Dimensions bw_strides(num_dims, 1);
  auto bw_base_dilations = reduce_window_attributes.window_strides;
  Dimensions bw_window_dilations(num_dims, 1);

  // Create a flat list of padding values for StableHLO's ReduceWindowOp
  llvm::SmallVector<int64_t> bw_padding_flat;
  bw_padding_flat.reserve(num_dims * 2);
  for (int i = 0; i < num_dims; ++i) {
    if (i < num_dims - spatial_dim_count) {  // Batch/Channel Padding is 0
      bw_padding_flat.push_back(0);
      bw_padding_flat.push_back(0);
    } else {
      int64_t k = bw_window_dimensions.asArrayRef()[i];
      bw_padding_flat.push_back(k - 1);
      bw_padding_flat.push_back(k - 1);
    }
  }

  mlir::MlirOp init_value = MakeScalarConstant(builder, 0, element_type);

  auto add_reduce_builder = [element_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        element_type, rb.getRegion(), rb.getOpBuilder());
  };

  auto full_conv_result = mlir::stablehlo::ReduceWindow(
      builder,
      /*inputs=*/{normalized_grad_output},
      /*init_values=*/{init_value},
      /*body=*/add_reduce_builder,
      /*window_dimensions=*/
      mlir::DenseI64ArrayAttr::get(&builder.getContext(), bw_window_dimensions),
      /*window_strides=*/
      mlir::DenseI64ArrayAttr::get(&builder.getContext(), bw_strides),
      /*base_dilations=*/
      mlir::DenseI64ArrayAttr::get(&builder.getContext(), bw_base_dilations),
      /*window_dilations=*/
      mlir::DenseI64ArrayAttr::get(&builder.getContext(), bw_window_dilations),
      /*padding=*/
      mlir::DenseIntElementsAttr::get(
          mlir::makeTensorType(builder.getContext(), {num_dims, 2},
                               builder.getOpBuilder().getI64Type()),
          bw_padding_flat))[0];

  // 7. Slice out the region corresponding to the original input shape
  auto sliced_grad =
      SliceBackwardResult(full_conv_result, init_value, input_type,
                          ceil_padding_pairs, num_dims, spatial_dim_count);

  auto result = RemoveTrivialBatch(
      sliced_grad, batch_input_info.original_dim_size, spatial_dim_count);
  return CastIfNeeded(result, orig_dtype);
}

auto GetAvgPoolOpBuilder(at::IntArrayRef kernel_size, at::IntArrayRef stride,
                         at::IntArrayRef padding, bool ceil_mode,
                         bool count_include_pad,
                         std::optional<int64_t> divisor_override,
                         int64_t spatial_dim_count) {
  return
      [kernel_size_vec = CopyIntVector(kernel_size),
       stride_vec = CopyIntVector(stride), padding_vec = CopyIntVector(padding),
       ceil_mode, count_include_pad, divisor_override, spatial_dim_count](
          mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
        return BuildAvgPoolShlo(input_op, spatial_dim_count, kernel_size_vec,
                                stride_vec, padding_vec, ceil_mode,
                                count_include_pad, divisor_override);
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
  auto op_builder = GetAvgPoolOpBuilder(kernel_size, stride, padding, ceil_mode,
                                        count_include_pad, divisor_override,
                                        spatial_dim_count);

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
  auto op_builder = GetAvgPoolOpBuilder(kernel_size, stride, padding, ceil_mode,
                                        count_include_pad, divisor_override,
                                        spatial_dim_count);
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

  auto op_builder = [kernel_size_vec = CopyIntVector(kernel_size),
                     stride_vec = CopyIntVector(stride),
                     padding_vec = CopyIntVector(padding), ceil_mode,
                     count_include_pad, divisor_override,
                     spatial_dim_count](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    return BuildAvgPoolBackwardShlo(
        inputs[0], inputs[1], spatial_dim_count, kernel_size_vec, stride_vec,
        padding_vec, ceil_mode, count_include_pad, divisor_override);
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
        return MakeTensor(result_buf);
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
