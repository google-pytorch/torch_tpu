// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "csrc/ops/col2im/col2im_aten_kernels.h"

#include <algorithm>
#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

// This function implements the col2im operation.
//
// Algorithm:
// col2im is the gradient of im2col. While im2col extracts sliding windows from
// an image into a column matrix, col2im sums up these columns back into the
// original image locations. This is mathematically equivalent to a transposed
// convolution (deconvolution).
//
// In StableHLO, we implement this using a standard ConvolutionOp by
// constructing a special "identity weight" tensor.
//
// 1. Input: (N, C * kernel_prod, L)
//    - N: Batch size
//    - C: Input channels
//    - kernel_prod: Kernel product (kH * kW)
//    - L: Length of the flattened column (col_h * col_w)
//
// 2. Weight Construction: (C, kernel_prod, kH, kW)
//    - We create a weight tensor that effectively scatters each of the
//    'kernel_prod'
//      elements in the input channel dimension to their correct spatial
//      location in the 'kH x kW' kernel window.
//    - This is done by creating a identity mapping of shape (kernel_prod,
//    kernel_prod) and
//      broadcasting it.
//
// 3. Convolution:
//    - We reshape input to (N, C * kernel_prod, col_h, col_w).
//    - We convolve with the identity weights.
//    - lhs_dilation = stride (this spaces out the input elements).
//    - rhs_dilation = dilation (this spaces out the kernel elements).
//    - feature_group_count = C (each channel is independent).
//    - The result effectively sums up overlapping windows, achieving the
//      col2im reduction.
absl::StatusOr<mlir::MlirOp> BuildCol2ImShlo(
    mlir::MlirOp input, SmallInt64Vector output_size, SmallInt64Vector col_size,
    SmallInt64Vector kernel_size, SmallInt64Vector dilation,
    SmallInt64Vector padding, SmallInt64Vector stride) {
  auto& builder = input.getBuilder();
  auto& op_builder = builder.getOpBuilder();

  // Input: (N, C * kH * kW, L)
  const auto input_type = GetTensorTypeOrDie(input);
  const auto input_shape = input_type.getShape();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Error is caught before this function is
                 // called.
      input_shape.size() == 3, error::kInvalidArgument)
      << "input must be 3D, got " << input_shape.size() << "D";

  // N
  const int64_t batch = input_shape[0];
  // C * kH * kW
  const int64_t channels_col = input_shape[1];

  const int64_t kH = kernel_size[0];
  const int64_t kW = kernel_size[1];
  const int64_t kernel_prod = kH * kW;

  // C
  const int64_t channels = channels_col / kernel_prod;
  const int64_t output_h = output_size[0];
  const int64_t col_h = col_size[0];

  const int64_t output_w = output_size[1];
  const int64_t col_w = col_size[1];

  // shape: (kernel_prod)
  const auto iota = mlir::stablehlo::Iota(
      builder,
      mlir::RankedTensorType::get({kernel_prod}, op_builder.getI32Type()), 0);

  // Create (kernel_prod, kernel_prod) identity matrix.
  const auto matrix_type = mlir::RankedTensorType::get(
      {kernel_prod, kernel_prod}, op_builder.getI32Type());
  // shape: (kernel_prod, 1)
  auto iota_row_reshaped = mlir::stablehlo::Reshape(iota, {kernel_prod, 1});
  // shape: (kernel_prod, kernel_prod)
  auto iota_row =
      mlir::stablehlo::BroadcastInDim(matrix_type, iota_row_reshaped, {0, 1});

  // shape: (1, kernel_prod)
  auto iota_col_reshaped = mlir::stablehlo::Reshape(iota, {1, kernel_prod});
  // shape: (kernel_prod, kernel_prod)
  auto iota_col =
      mlir::stablehlo::BroadcastInDim(matrix_type, iota_col_reshaped, {0, 1});

  // shape: (kernel_prod, kernel_prod)
  const auto identity_bool = mlir::stablehlo::Compare(
      iota_row, iota_col, mlir::stablehlo::ComparisonDirection::EQ);
  const auto element_type = input_type.getElementType();
  // shape: (kernel_prod, kernel_prod)
  const auto identity =
      mlir::stablehlo::ConvertElementType(identity_bool, element_type);

  // Reshape Identity to (kH, kW, kernel_prod)
  auto weight_squeezed =
      mlir::stablehlo::Reshape(identity, {kH, kW, kernel_prod});

  const int64_t pad_h_low = (kH - 1) * dilation[0];
  const int64_t pad_w_low = (kW - 1) * dilation[1];
  const int64_t conv_out_h =
      std::max((col_h - 1) * stride[0] + 1 + pad_h_low, padding[0] + output_h);
  const int64_t conv_out_w =
      std::max((col_w - 1) * stride[1] + 1 + pad_w_low, padding[1] + output_w);
  const int64_t pad_h_high = conv_out_h - (col_h - 1) * stride[0] - 1;
  const int64_t pad_w_high = conv_out_w - (col_w - 1) * stride[1] - 1;

  // shape: (N, C, conv_out_h, conv_out_w)
  const auto conv_output_type = mlir::RankedTensorType::get(
      {batch, channels, conv_out_h, conv_out_w}, element_type);

  // shape: (N, C*kernel_prod, col_h, col_w)
  auto reshaped_input_op =
      mlir::stablehlo::Reshape(input, {batch, channels_col, col_h, col_w});

  // shape: (C, kp, kH, kW)
  const auto weight_type = mlir::RankedTensorType::get(
      {channels, kernel_prod, kH, kW}, element_type);
  auto weights =
      mlir::stablehlo::BroadcastInDim(weight_type, weight_squeezed, {2, 3, 1});

  // Padding: (spatial_dims, 2)
  const auto padding_type =
      mlir::RankedTensorType::get({2, 2}, op_builder.getI64Type());
  const auto padding_attr = mlir::DenseIntElementsAttr::get(
      padding_type, {pad_h_low, pad_h_high, pad_w_low, pad_w_high});

  // Convolution output shape: (N, C, conv_out_h, conv_out_w)
  auto conv = mlir::stablehlo::Convolution(
      conv_output_type, reshaped_input_op, weights,
      /*dimension_numbers=*/
      mlir::stablehlo::ConvDimensionNumbersAttr::get(
          &builder.getContext(),
          /*inputBatchDimension=*/0,
          /*inputFeatureDimension=*/1,
          /*inputSpatialDimensions=*/{2, 3},
          /*kernelInputFeatureDimension=*/1,
          /*kernelOutputFeatureDimension=*/0,
          /*kernelSpatialDimensions=*/{2, 3},
          /*outputBatchDimension=*/0,
          /*outputFeatureDimension=*/1,
          /*outputSpatialDimensions=*/{2, 3}),
      /*feature_group_count=*/channels,
      /*batch_group_count=*/1,
      /*window_strides=*/op_builder.getDenseI64ArrayAttr({1, 1}),
      /*padding=*/padding_attr,
      /*lhs_dilation=*/
      op_builder.getDenseI64ArrayAttr({stride[0], stride[1]}),
      /*rhs_dilation=*/
      op_builder.getDenseI64ArrayAttr({dilation[0], dilation[1]}),
      /*window_reversal=*/op_builder.getDenseBoolArrayAttr({true, true}),
      /*precision_config=*/{});

  // Slice to final output shape: (N, C, output_h, output_w)
  return mlir::stablehlo::Slice(
      conv, {0, 0, padding[0], padding[1]},
      {batch, channels, padding[0] + output_h, padding[1] + output_w},
      {1, 1, 1, 1});
}

// Returns the number of blocks to slide over the input tensor.
//
// All of the input arrays should have exactly 2 elements. This condition is
// directly (by `GetCol2ImOutputInfo()` function) or indirectly (by
// `AtenCol2Im()` function) checked calling the `GetCol2ImOutputInfo()`
// function.
absl::StatusOr<SmallInt64Vector> ComputeOutputSize(at::IntArrayRef input_size,
                                                   at::IntArrayRef padding,
                                                   at::IntArrayRef dilation,
                                                   at::IntArrayRef kernel_size,
                                                   at::IntArrayRef stride) {
  bool are_strides_positive = std::all_of(
      stride.begin(), stride.end(), [](const int64_t s) { return s > 0; });

  TT_RET_CHECK(are_strides_positive, error::kInvalidArgument)
      << "expected all stride elements to be positive, got "
      << ToString(stride);

  auto compute_size_of_dimension = [input_size, padding, dilation, kernel_size,
                                    stride](const int64_t dim) {
    return (input_size[dim] + 2 * padding[dim] -
            dilation[dim] * (kernel_size[dim] - 1) - 1) /
               stride[dim] +
           1;
  };

  return SmallInt64Vector(
      {compute_size_of_dimension(0), compute_size_of_dimension(1)});
}

// Holds the output dimensions and column size for col2im.
struct Col2ImOutputInfo {
  Dimensions output_dims;
  SmallInt64Vector col_size;
};

// Returns the output dimensions and column size for a col2im operation, and
// validates consistency between input shape and output parameters.
absl::StatusOr<Col2ImOutputInfo> GetCol2ImOutputInfo(
    const at::Tensor& input, at::IntArrayRef output_size,
    at::IntArrayRef kernel_size, at::IntArrayRef dilation,
    at::IntArrayRef padding, at::IntArrayRef stride) {
  TT_ASSIGN_OR_RETURN(const auto elem_dtype,
                      ConvertTo<mlir::ElementType>(input.scalar_type()));
  TT_RET_CHECK(!IsInteger(elem_dtype, /*includeBool=*/false),
               error::kInvalidArgument)
      << "expected non-integer dtype, got " << ToString(input.scalar_type());
  TT_RET_CHECK(output_size.size() == 2, error::kInvalidArgument)
      << "expected output_size to have 2 dimensions, got "
      << output_size.size();
  TT_RET_CHECK(kernel_size.size() == 2, error::kInvalidArgument)
      << "expected kernel_size to have 2 dimensions, got "
      << kernel_size.size();
  TT_RET_CHECK(dilation.size() == 2, error::kInvalidArgument)
      << "expected dilation to have 2 dimensions, got " << dilation.size();
  TT_RET_CHECK(padding.size() == 2, error::kInvalidArgument)
      << "expected padding to have 2 dimensions, got " << padding.size();
  TT_RET_CHECK(stride.size() == 2, error::kInvalidArgument)
      << "expected stride to have 2 dimensions, got " << stride.size();
  TT_RET_CHECK(input.dim() == 3, error::kInvalidArgument)
      << "expected input to have 3 dimensions (batch, channels, length), got "
      << input.dim();

  // Output: (N, C, output_h, output_w)
  // Input: (N, C * kH * kW, L)
  const int64_t n = input.size(0);
  const int64_t c_col = input.size(1);
  const int64_t k_h = kernel_size[0];
  const int64_t k_w = kernel_size[1];
  const int64_t kernel_prod = k_h * k_w;

  TT_RET_CHECK(kernel_prod > 0, error::kInvalidArgument)
      << "expected kernel size to be positive, got " << kernel_prod;
  TT_RET_CHECK(c_col % kernel_prod == 0, error::kInvalidArgument)
      << "expected input channels to be divisible by kernel product ("
      << kernel_prod << "), got " << c_col;

  TT_ASSIGN_OR_RETURN(
      auto col_size,
      ComputeOutputSize(output_size, padding, dilation, kernel_size, stride));

  const int64_t col_h = col_size[0];
  const int64_t col_w = col_size[1];

  // Verify input length matches calculated column size
  const int64_t length_col = input.size(2);

  TT_RET_CHECK(length_col == col_h * col_w, error::kInvalidArgument)
      << "expected input length to be divisible by col size (" << col_h << " * "
      << col_w << " = " << col_h * col_w << "), got " << length_col;

  // C = input(1) / (kernel_size(0) * kernel_size(1))
  const int64_t c = c_col / kernel_prod;
  return Col2ImOutputInfo{
      .output_dims = Dimensions{n, c, output_size[0], output_size[1]},
      .col_size = std::move(col_size),
  };
}

absl::StatusOr<DeviceBufferRef> AtenCol2Im(const at::Tensor& input,
                                           at::IntArrayRef output_size,
                                           at::IntArrayRef kernel_size,
                                           at::IntArrayRef dilation,
                                           at::IntArrayRef padding,
                                           at::IntArrayRef stride,
                                           OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(auto output_info,
                      GetCol2ImOutputInfo(input, output_size, kernel_size,
                                          dilation, padding, stride));

  auto op_builder = [output_size = CopyIntVector(output_size),
                     col_size = std::move(output_info.col_size),
                     kernel_size = CopyIntVector(kernel_size),
                     dilation = CopyIntVector(dilation),
                     padding = CopyIntVector(padding),
                     stride = CopyIntVector(stride)](mlir::MlirOp input) {
    return BuildCol2ImShlo(input, output_size, col_size, kernel_size, dilation,
                           padding, stride);
  };

  TT_ASSIGN_OR_RETURN(const auto elem_type,
                      ConvertTo<mlir::ElementType>(input.scalar_type()));

  return DispatchOp<1>(std::move(op_builder), {input},
                       {.out_dtype = elem_type,
                        .out_dims = std::move(output_info.output_dims),
                        .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor AtenCol2Im(const at::Tensor& input, at::IntArrayRef output_size,
                      at::IntArrayRef kernel_size, at::IntArrayRef dilation,
                      at::IntArrayRef padding, at::IntArrayRef stride) {
  TT_KERNEL(OpName::kCol2Im, param_keys,
            (input, output_size, kernel_size, dilation, padding, stride), {
              TT_ASSIGN_OR_THROW(
                  DeviceBufferRef result,
                  AtenCol2Im(input, output_size, kernel_size, dilation, padding,
                             stride, std::move(param_keys)));
              return MakeTensor(std::move(result));
            });
}

at::Tensor& AtenCol2ImOut(const at::Tensor& input, at::IntArrayRef output_size,
                          at::IntArrayRef kernel_size, at::IntArrayRef dilation,
                          at::IntArrayRef padding, at::IntArrayRef stride,
                          at::Tensor& out) {
  TT_KERNEL(OpName::kCol2ImOut, param_keys,
            (input, output_size, kernel_size, dilation, padding, stride, out), {
              TT_ASSIGN_OR_THROW(
                  DeviceBufferRef result,
                  AtenCol2Im(input, output_size, kernel_size, dilation, padding,
                             stride, std::move(param_keys)));
              TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
              return out;
            });
}

}  // namespace torch_tpu
