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
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "csrc/common/aten_utils.h"
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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

// Computes the number of sliding-window positions along one spatial dimension
// using floor division, matching PyTorch's at::native::sliding_window_count.
// Requires size > 0, pad >= 0, dilation > 0, kernel > 0, and stride > 0.
absl::StatusOr<int64_t> SlidingWindowCount(const int64_t size,
                                           const int64_t pad,
                                           const int64_t dilation,
                                           const int64_t kernel,
                                           const int64_t stride) {
  TT_ASSIGN_OR_RETURN(const int64_t two_pad, SafeMultiply(int64_t{2}, pad));
  TT_ASSIGN_OR_RETURN(const int64_t padded_size, SafeAdd(size, two_pad));
  TT_ASSIGN_OR_RETURN(const int64_t dilated_span,
                      SafeMultiply(dilation, kernel - 1));
  TT_ASSIGN_OR_RETURN(const int64_t effective_kernel,
                      SafeAdd(dilated_span, int64_t{1}));
  TT_ASSIGN_OR_RETURN(const int64_t numerator,
                      SafeAdd(padded_size, -effective_kernel));
  int64_t quotient = numerator / stride;
  if (numerator % stride < 0) {
    --quotient;
  }
  return SafeAdd(quotient, int64_t{1});
}

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
//      'kernel_prod' elements in the input channel dimension to their correct
//      spatial location in the 'kH x kW' kernel window.
//    - This is done by creating an identity mapping of shape
//      (kernel_prod, kernel_prod) and broadcasting it.
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
    mlir::MlirOp input, const SmallInt64Vector& output_size,
    const SmallInt64Vector& col_size, const SmallInt64Vector& kernel_size,
    const SmallInt64Vector& dilation, const SmallInt64Vector& padding,
    const SmallInt64Vector& stride) {
  mlir::MlirBuilder& builder = input.getBuilder();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();

  // Input: (N, C * kH * kW, L).
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const llvm::ArrayRef<int64_t> input_shape = input_type.getShape();
  ABSL_CHECK_EQ(input_shape.size(), 3)  // CRASH_OK=Validated before lowering.
      << "Input must be 3D. This is a TorchTPU bug.";

  const int64_t batch = input_shape[0];
  const int64_t channels_col = input_shape[1];
  const int64_t k_h = kernel_size[0];
  const int64_t k_w = kernel_size[1];
  const int64_t kernel_prod = k_h * k_w;
  const int64_t channels = channels_col / kernel_prod;
  const int64_t output_h = output_size[0];
  const int64_t output_w = output_size[1];
  const int64_t col_h = col_size[0];
  const int64_t col_w = col_size[1];
  const mlir::Type element_type = input_type.getElementType();

  if (batch == 0) {
    return MakeZeroSizedTensor(builder, element_type,
                               {0, channels, output_h, output_w});
  }

  // shape: (kernel_prod)
  mlir::MlirOp iota = mlir::stablehlo::Iota(
      builder,
      mlir::RankedTensorType::get({kernel_prod}, op_builder.getI32Type()), 0);

  // Create (kernel_prod, kernel_prod) identity matrix.
  const mlir::RankedTensorType matrix_type = mlir::RankedTensorType::get(
      {kernel_prod, kernel_prod}, op_builder.getI32Type());
  // shape: (kernel_prod, 1)
  mlir::MlirOp iota_row_reshaped =
      mlir::stablehlo::Reshape(iota, {kernel_prod, 1});
  // shape: (kernel_prod, kernel_prod)
  mlir::MlirOp iota_row =
      mlir::stablehlo::BroadcastInDim(matrix_type, iota_row_reshaped, {0, 1});

  // shape: (1, kernel_prod)
  mlir::MlirOp iota_col_reshaped =
      mlir::stablehlo::Reshape(iota, {1, kernel_prod});
  // shape: (kernel_prod, kernel_prod)
  mlir::MlirOp iota_col =
      mlir::stablehlo::BroadcastInDim(matrix_type, iota_col_reshaped, {0, 1});

  // shape: (kernel_prod, kernel_prod)
  mlir::MlirOp identity_bool = mlir::stablehlo::Compare(
      iota_row, iota_col, mlir::stablehlo::ComparisonDirection::EQ);
  // shape: (kernel_prod, kernel_prod)
  mlir::MlirOp identity =
      mlir::stablehlo::ConvertElementType(identity_bool, element_type);

  // Reshape identity to (kH, kW, kernel_prod).
  mlir::MlirOp weight_squeezed =
      mlir::stablehlo::Reshape(identity, {k_h, k_w, kernel_prod});

  const int64_t pad_h_low = (k_h - 1) * dilation[0];
  const int64_t pad_w_low = (k_w - 1) * dilation[1];
  const int64_t conv_out_h =
      std::max((col_h - 1) * stride[0] + 1 + pad_h_low, padding[0] + output_h);
  const int64_t conv_out_w =
      std::max((col_w - 1) * stride[1] + 1 + pad_w_low, padding[1] + output_w);
  const int64_t pad_h_high = conv_out_h - (col_h - 1) * stride[0] - 1;
  const int64_t pad_w_high = conv_out_w - (col_w - 1) * stride[1] - 1;

  // shape: (N, C, conv_out_h, conv_out_w)
  const mlir::RankedTensorType conv_output_type = mlir::RankedTensorType::get(
      {batch, channels, conv_out_h, conv_out_w}, element_type);

  // shape: (N, C * kernel_prod, col_h, col_w)
  mlir::MlirOp reshaped_input_op =
      mlir::stablehlo::Reshape(input, {batch, channels_col, col_h, col_w});

  // shape: (C, kp, kH, kW)
  const mlir::RankedTensorType weight_type = mlir::RankedTensorType::get(
      {channels, kernel_prod, k_h, k_w}, element_type);
  mlir::MlirOp weights =
      mlir::stablehlo::BroadcastInDim(weight_type, weight_squeezed, {2, 3, 1});

  // Padding: (spatial_dims, 2)
  const mlir::RankedTensorType padding_type =
      mlir::RankedTensorType::get({2, 2}, op_builder.getI64Type());
  const mlir::DenseIntElementsAttr padding_attr =
      mlir::DenseIntElementsAttr::get(
          padding_type, {pad_h_low, pad_h_high, pad_w_low, pad_w_high});

  // Convolution output shape: (N, C, conv_out_h, conv_out_w)
  mlir::MlirOp conv = mlir::stablehlo::Convolution(
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

  // Slice to final output shape: (N, C, output_h, output_w).
  return mlir::stablehlo::Slice(
      conv, {0, 0, padding[0], padding[1]},
      {batch, channels, padding[0] + output_h, padding[1] + output_w},
      {1, 1, 1, 1});
}

// Holds the validated output dimensions, dtype, and column size for col2im.
struct Col2ImOutputInfo {
  mlir::ElementType out_dtype;
  Dimensions output_dims;
  SmallInt64Vector col_size;
  bool is_2d;
};

// Returns the output dimensions and column size for a col2im operation, and
// validates consistency between input shape and output parameters.
absl::StatusOr<Col2ImOutputInfo> ValidateAndGetCol2ImOutputInfo(
    const at::Tensor& input, const at::IntArrayRef output_size,
    const at::IntArrayRef kernel_size, const at::IntArrayRef dilation,
    const at::IntArrayRef padding, const at::IntArrayRef stride) {
  TT_RETURN_IF_ERROR(GetBaseBuffer(input).status());
  TT_ASSIGN_OR_RETURN(const mlir::ElementType elem_dtype,
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

  const int64_t output_h = output_size[0];
  const int64_t output_w = output_size[1];
  const int64_t k_h = kernel_size[0];
  const int64_t k_w = kernel_size[1];
  const int64_t d_h = dilation[0];
  const int64_t d_w = dilation[1];
  const int64_t p_h = padding[0];
  const int64_t p_w = padding[1];
  const int64_t s_h = stride[0];
  const int64_t s_w = stride[1];

  TT_RET_CHECK(k_h > 0 && k_w > 0, error::kInvalidArgument)
      << "expected all kernel_size elements to be positive, got "
      << ToString(kernel_size);
  TT_RET_CHECK(s_h > 0 && s_w > 0, error::kInvalidArgument)
      << "expected all stride elements to be positive, got "
      << ToString(stride);
  TT_RET_CHECK(d_h > 0 && d_w > 0, error::kInvalidArgument)
      << "expected all dilation elements to be positive, got "
      << ToString(dilation);
  TT_RET_CHECK(p_h >= 0 && p_w >= 0, error::kInvalidArgument)
      << "expected all padding elements to be >= 0, got " << ToString(padding);
  TT_RET_CHECK(output_h > 0 && output_w > 0, error::kInvalidArgument)
      << "expected all output_size elements to be positive, got "
      << ToString(output_size);

  const int64_t ndim = input.dim();
  TT_RET_CHECK(ndim == 2 || ndim == 3, error::kInvalidArgument)
      << "expected input to have 2 or 3 dimensions, got " << ndim;

  const bool is_2d = ndim == 2;
  const int64_t batch_offset = is_2d ? -1 : 0;
  const int64_t c_col = input.size(batch_offset + 1);
  const int64_t length_col = input.size(batch_offset + 2);

  TT_RET_CHECK(c_col > 0 && length_col > 0, error::kInvalidArgument)
      << "expected non-batch input dimensions to be > 0, got shape "
      << ToString(input.sizes());

  TT_ASSIGN_OR_RETURN(const int64_t kernel_prod, SafeMultiply(k_h, k_w));
  TT_RET_CHECK(c_col % kernel_prod == 0, error::kInvalidArgument)
      << "expected input channels to be divisible by kernel product ("
      << kernel_prod << "), got " << c_col;

  TT_ASSIGN_OR_RETURN(const int64_t col_h,
                      SlidingWindowCount(output_h, p_h, d_h, k_h, s_h));
  TT_ASSIGN_OR_RETURN(const int64_t col_w,
                      SlidingWindowCount(output_w, p_w, d_w, k_w, s_w));

  TT_RET_CHECK(col_h > 0 && col_w > 0, error::kInvalidArgument)
      << "expected calculated col size to be positive, got [" << col_h << ", "
      << col_w << "]";

  TT_ASSIGN_OR_RETURN(const int64_t expected_length,
                      SafeMultiply(col_h, col_w));
  TT_RET_CHECK(length_col == expected_length, error::kInvalidArgument)
      << "expected input length to match col size (" << col_h << " * " << col_w
      << " = " << expected_length << "), got " << length_col;

  const int64_t c = c_col / kernel_prod;
  Dimensions output_dims;
  if (is_2d) {
    output_dims = {c, output_h, output_w};
  } else {
    output_dims = {input.size(0), c, output_h, output_w};
  }

  return Col2ImOutputInfo{
      .out_dtype = elem_dtype,
      .output_dims = std::move(output_dims),
      .col_size = SmallInt64Vector{col_h, col_w},
      .is_2d = is_2d,
  };
}

MlirUnaryOpBuilder MakeCol2ImOpBuilder(
    SmallInt64Vector output_size, SmallInt64Vector col_size,
    SmallInt64Vector kernel_size, SmallInt64Vector dilation,
    SmallInt64Vector padding, SmallInt64Vector stride, const bool is_2d) {
  return [output_size = std::move(output_size), col_size = std::move(col_size),
          kernel_size = std::move(kernel_size), dilation = std::move(dilation),
          padding = std::move(padding), stride = std::move(stride),
          is_2d](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
    if (is_2d) {
      TT_ASSIGN_OR_RETURN(input, Unsqueeze(input, 0));
    }
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp result,
        BuildCol2ImShlo(input, output_size, col_size, kernel_size, dilation,
                        padding, stride));
    if (is_2d) {
      return Squeeze(result, {0});
    }
    return result;
  };
}

}  // namespace

at::Tensor AtenCol2Im(const at::Tensor& input,
                      const at::IntArrayRef output_size,
                      const at::IntArrayRef kernel_size,
                      const at::IntArrayRef dilation,
                      const at::IntArrayRef padding,
                      const at::IntArrayRef stride) {
  TT_KERNEL(
      OpName::kCol2Im, param_keys,
      (input, output_size, kernel_size, dilation, padding, stride), {
        TT_ASSIGN_OR_THROW(
            Col2ImOutputInfo info,
            ValidateAndGetCol2ImOutputInfo(input, output_size, kernel_size,
                                           dilation, padding, stride));
        MlirUnaryOpBuilder op_builder = MakeCol2ImOpBuilder(
            CopyIntVector(output_size), std::move(info.col_size),
            CopyIntVector(kernel_size), CopyIntVector(dilation),
            CopyIntVector(padding), CopyIntVector(stride), info.is_2d);
        TT_ASSIGN_OR_THROW(
            DeviceBufferRef result,
            DispatchOp<1>(std::move(op_builder), input,
                          {.out_dtype = info.out_dtype,
                           .out_dims = info.output_dims,
                           .op_param_cache_keys = std::move(param_keys)}));
        return MakeTensor(std::move(result));
      });
}

at::Tensor& AtenCol2ImOut(const at::Tensor& input,
                          const at::IntArrayRef output_size,
                          const at::IntArrayRef kernel_size,
                          const at::IntArrayRef dilation,
                          const at::IntArrayRef padding,
                          const at::IntArrayRef stride, at::Tensor& out) {
  TT_KERNEL(OpName::kCol2ImOut, param_keys,
            (input, output_size, kernel_size, dilation, padding, stride, out), {
              TT_THROW_IF_ERROR(GetBaseBuffer(input).status());
              TT_THROW_IF_ERROR(GetBaseBuffer(out).status());
              TT_THROW_IF_ERROR(ValidateOutDtype(
                  out, input.scalar_type(), /*allow_out_dtype_cast=*/false));
              TT_ASSIGN_OR_THROW(Col2ImOutputInfo info,
                                 ValidateAndGetCol2ImOutputInfo(
                                     input, output_size, kernel_size, dilation,
                                     padding, stride));
              MlirUnaryOpBuilder op_builder = MakeCol2ImOpBuilder(
                  CopyIntVector(output_size), std::move(info.col_size),
                  CopyIntVector(kernel_size), CopyIntVector(dilation),
                  CopyIntVector(padding), CopyIntVector(stride), info.is_2d);
              TT_THROW_IF_ERROR(DispatchOpOut<1>(
                  std::move(op_builder), input, out,
                  {.out_dtype = info.out_dtype,
                   .out_dims = info.output_dims,
                   .op_param_cache_keys = std::move(param_keys)}));
              return out;
            });
}

}  // namespace torch_tpu
