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

#include "torch_tpu/ops/col2im/col2im.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

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

  const int64_t pad_h = (kH - 1) * dilation[0];
  const int64_t pad_w = (kW - 1) * dilation[1];
  const int64_t conv_out_h = col_h + 2 * pad_h - (kH - 1) * dilation[0];
  const int64_t conv_out_w = col_w + 2 * pad_w - (kW - 1) * dilation[1];

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
      padding_type, {pad_h, pad_h, pad_w, pad_w});

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

}  // namespace torch_tpu
