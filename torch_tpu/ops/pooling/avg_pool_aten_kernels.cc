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

#include "torch_tpu/ops/pooling/avg_pool_aten_kernels.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/pooling/pooling.h"

namespace torch_tpu {
namespace {

// Common computation shared between AvgPool and AvgPoolGrad.
//
// We use division instead of multiplication by reciprocal (1.0 / divisor)
// here for better numerical correctness, even though it might be slower.
// For integer types, calculating 1.0 / divisor can result in a
// fraction (e.g., -0.5), which truncates to 0 when converted to an integer
// for use with MakeConstant or multiplication. This would lead to incorrect
// results (e.g., an all-zero tensor).
//
// The divisor for average pooling is computed based on pooling parameters:
// 1. If divisor_override is provided, it is used.
// 2. If count_include_pad is true, divisor is the window size.
// 3. Otherwise, divisor is the number of valid elements in the window,
//    ignoring padded elements.
absl::StatusOr<mlir::MlirOp> ComputeAvgPoolDivisor(
    mlir::MlirBuilder& builder, const mlir::RankedTensorType& input_type,
    const mlir::RankedTensorType& sum_type,
    const ReduceWindowAttributes& reduce_window_attributes,
    bool count_include_pad, std::optional<int64_t> divisor_override) {
  mlir::Type element_type = input_type.getElementType();
  auto sum_shape = sum_type.getShape();

  // Case A: Divisor is specified
  if (divisor_override.has_value()) {
    return MakeConstant(builder, divisor_override.value(), element_type,
                        sum_shape);
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
      int64_t window_size = std::accumulate(
          reduce_window_attributes.window_dimensions.asArrayRef().begin(),
          reduce_window_attributes.window_dimensions.asArrayRef().end(),
          int64_t{1}, std::multiplies<int64_t>());
      return MakeConstant(builder, window_size, element_type, sum_shape);
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

    return mlir::stablehlo::ReduceWindow(
        builder,
        /*inputs=*/{padded_ones},
        /*init_values=*/{zero_scalar},
        /*body=*/add_reduce_builder, reduce_window_attributes.window_dimensions,
        reduce_window_attributes.window_strides,
        reduce_window_attributes.base_dilations,
        reduce_window_attributes.window_dilations, reduce_padding_attr)[0];
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

  return mlir::stablehlo::ReduceWindow(
      builder,
      /*inputs=*/{ones},
      /*init_values=*/{init_value},
      /*body=*/add_reduce_builder, reduce_window_attributes.window_dimensions,
      reduce_window_attributes.window_strides,
      reduce_window_attributes.base_dilations,
      reduce_window_attributes.window_dilations,
      reduce_window_attributes.padding)[0];
}

absl::StatusOr<mlir::MlirOp> BuildAvgPoolShlo(
    mlir::MlirOp input, int64_t spatial_dim_count, Dimensions kernel_size,
    Dimensions stride, Dimensions padding, bool ceil_mode,
    bool count_include_pad, std::optional<int64_t> divisor_override) {
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
  auto sum_type = GetTensorTypeOrDie(sum_output);

  TT_ASSIGN_OR_RETURN(
      auto divisor, ComputeAvgPoolDivisor(builder, input_type, sum_type,
                                          reduce_window_attributes,
                                          count_include_pad, divisor_override));
  auto final_output = mlir::stablehlo::Div(sum_output, divisor);

  return RemoveTrivialBatch(final_output, original_dim_size, spatial_dim_count);
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
absl::StatusOr<mlir::MlirOp> BuildAvgPoolBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp input, int64_t spatial_dim_count,
    const Dimensions& kernel_size, const Dimensions& stride,
    const Dimensions& padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  auto& builder = grad_output.getBuilder();

  // 1. Create batch inputs for both input and grad_output
  TT_ASSIGN_OR_RETURN(auto batch_input_info,
                      CreateBatchInput(input, spatial_dim_count));
  TT_ASSIGN_OR_RETURN(auto batch_grad_output_info,
                      CreateBatchInput(grad_output, spatial_dim_count));
  mlir::MlirOp batch_input = batch_input_info.batch_input;
  mlir::MlirOp batch_grad_output = batch_grad_output_info.batch_input;

  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(batch_input);
  const mlir::RankedTensorType grad_output_type =
      GetTensorTypeOrDie(batch_grad_output);
  const mlir::Type element_type = input_type.getElementType();
  const int64_t num_dims = input_type.getRank();

  // 2. Expand attributes to match the spatial_dim_count.
  TT_ASSIGN_OR_RETURN(Dimensions kernel_size_attr,
                      ExpandAttribute(kernel_size, spatial_dim_count));
  const Dimensions stride_attr =
      stride.empty() ? kernel_size_attr
                     : ExpandAttribute(stride, spatial_dim_count).value();
  Dimensions padding_attr =
      padding.empty() ? Dimensions(spatial_dim_count, 0)
                      : ExpandAttribute(padding, spatial_dim_count).value();
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
  TT_ASSIGN_OR_RETURN(
      auto divisor, ComputeAvgPoolDivisor(builder, input_type, grad_output_type,
                                          reduce_window_attributes,
                                          count_include_pad, divisor_override));
  auto normalized_grad_output =
      mlir::stablehlo::Div(batch_grad_output, divisor);

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
  // The 'full_conv_result' represents the reconstructed gradients.
  // Due to stride > 1 or ceil_mode, its size might be smaller than the required
  // range [pad_low, pad_low + input_size). We must check dimensions and PAD the
  // result if necessary before slicing.
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
      int spatial_idx = i - (num_dims - spatial_dim_count);
      int64_t fwd_pad_low = ceil_padding_pairs[spatial_idx].first;
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

  auto sliced_grad = mlir::stablehlo::Slice(slicable_result, start_indices,
                                            limit_indices, slice_strides);

  return RemoveTrivialBatch(sliced_grad, batch_input_info.original_dim_size,
                            spatial_dim_count);
}
}  // namespace

// Helper function to build and dispatch N-dimensional average pooling
// operations.
absl::StatusOr<at::Tensor> BuildAvgPoolOutNd(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override, at::Tensor& out,
    int64_t spatial_dim_count, OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(auto out_type,
                      ConvertTo<mlir::ElementType>(out.scalar_type()));

  auto op_builder = [kernel_size_vec = CopyIntVector(kernel_size),
                     stride_vec = CopyIntVector(stride),
                     padding_vec = CopyIntVector(padding), ceil_mode,
                     count_include_pad, divisor_override, spatial_dim_count](
                        mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
    return BuildAvgPoolShlo(input_op, spatial_dim_count, kernel_size_vec,
                            stride_vec, padding_vec, ceil_mode,
                            count_include_pad, divisor_override);
  };

  TT_ASSIGN_OR_RETURN(
      auto result_buf,
      DispatchOp<1>(std::move(op_builder), self,
                    {.out_dtype = out_type,
                     .out_dims = CopyIntVector(out.sizes()),
                     .op_param_cache_keys = std::move(param_keys)}));

  TT_RETURN_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
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

  TT_ASSIGN_OR_RETURN(
      auto result,
      (DispatchOp<2>(std::move(op_builder), {grad_output, self},
                     {.out_dtype = output_dtype,
                      .out_dims = CopyIntVector(grad_input.sizes()),
                      .op_param_cache_keys = std::move(param_keys)})));

  TT_RETURN_IF_ERROR(AssignBufferToAtTensor(std::move(result), grad_input));
  return grad_input;
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
        TT_CHECK_THROW(self.scalar_type() != at::ScalarType::Short &&
                           self.scalar_type() != at::ScalarType::Int &&
                           self.scalar_type() != at::ScalarType::Char &&
                           self.scalar_type() != at::ScalarType::Byte &&
                           self.scalar_type() != at::ScalarType::ComplexFloat,
                       error::kInvalidArgument)
            << "not yet implemented for uint8, int8, int16, int32,"
            << " and complex64 dtypes, got "
            << torch_tpu::ToString(self.scalar_type());

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
        TT_CHECK_THROW(self.scalar_type() != at::ScalarType::Bool &&
                           self.scalar_type() != at::ScalarType::BFloat16 &&
                           self.scalar_type() != at::ScalarType::Half &&
                           self.scalar_type() != at::ScalarType::Byte &&
                           self.scalar_type() != at::ScalarType::Char &&
                           self.scalar_type() != at::ScalarType::Short &&
                           self.scalar_type() != at::ScalarType::Int &&
                           self.scalar_type() != at::ScalarType::ComplexFloat,
                       error::kInvalidArgument)
            << "not yet implemented for bool, bfloat16, float16, uint8, int8,"
            << " int16, int32, and complex64 dtypes, got "
            << torch_tpu::ToString(self.scalar_type());

        TT_ASSIGN_OR_THROW(
            auto result,
            BuildAvgPoolOutNd(self, kernel_size, stride, padding, ceil_mode,
                              count_include_pad, divisor_override, out,
                              /*spatial_dim_count=*/3, std::move(param_keys)));
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
  TT_KERNEL(
      OpName::kAvgPool3dBackwardGradInput, param_keys,
      (grad_output, self, kernel_size, stride, padding, ceil_mode,
       count_include_pad, divisor_override, grad_input),
      {
        TT_ASSIGN_OR_THROW(
            auto result,
            BuildAvgPoolBackwardGradInputNd(
                grad_output, self, kernel_size, stride, padding, ceil_mode,
                count_include_pad, divisor_override, grad_input,
                /*spatial_dim_count=*/3, std::move(param_keys)));
        return grad_input;
      });
}
}  // namespace torch_tpu
