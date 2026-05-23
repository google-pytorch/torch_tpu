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

#include "torch_tpu/ops/convolution/convolution.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/convolution/convolution_checks.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {
struct GroupedWeightParams {
  mlir::MlirOp input;
  mlir::MlirOp grad_output;
  int64_t input_batch_dim;
  int64_t input_feature_dim;
  int64_t kernel_in_feat_dim;
  int64_t kernel_out_feat_dim;
};

struct SpatialPadding {
  int64_t lo;
  int64_t hi;
};

// Calculates the padding required for the adjoint (backward/transposed)
// convolution to match a target output dimension.
//
// Adjoint convolution padding logic:
// pad_total = output_dim + k_eff - 1 - ((input_dim - 1) * stride + 1)
// pad_lo = k_eff - 1 - padding
SpatialPadding CalculateAdjointPadding(int64_t input_dim, int64_t output_dim,
                                       int64_t kernel, int64_t stride,
                                       int64_t padding, int64_t dilation) {
  int64_t k_eff = (kernel - 1) * dilation + 1;
  int64_t pad_total = output_dim + k_eff - 1 - ((input_dim - 1) * stride + 1);
  int64_t pad_lo = k_eff - 1 - padding;
  return {pad_lo, pad_total - pad_lo};
}

// Calculates padding for backward weight calculation, accounting for
// windows that don't perfectly align with the input end (overhang).
//
// In a strided convolution (stride > 1), the windows might not perfectly
// align with the end of the padded input.
// 'overhang' represents the trailing pixels in the padded input that were
// ignored during the forward pass.
// If we use symmetric padding in backward, these 'dead pixels' contribute
// to the weight gradient calculation, causing the inferred gradient shape
// to be larger than the original weight shape (e.g., 4x4 vs 3x3).
// We add asymmetric padding to cancel out the overhang in the backward pass.
SpatialPadding CalculateBackwardWeightPadding(int64_t input_dim,
                                              int64_t grad_out_dim,
                                              int64_t kernel, int64_t stride,
                                              int64_t padding,
                                              int64_t dilation) {
  int64_t k_eff = (kernel - 1) * dilation + 1;
  int64_t last_window_end = (grad_out_dim - 1) * stride + k_eff;
  int64_t overhang = (input_dim + 2 * padding) - last_window_end;
  return {padding, padding - overhang};
}

absl::StatusOr<mlir::MlirOp> BuildTransposedConvolution(
    mlir::MlirOp input, mlir::MlirOp weight, std::optional<mlir::MlirOp> bias,
    absl::Span<const int64_t> stride, absl::Span<const int64_t> padding,
    absl::Span<const int64_t> dilation,
    absl::Span<const int64_t> output_padding, int64_t groups,
    absl::Span<const int64_t> output_dims, mlir::ElementType output_dtype,
    mlir::stablehlo::Precision precision) {
  mlir::MlirBuilder& builder = input.getBuilder();
  mlir::MLIRContext& ctx = builder.getContext();
  const int num_spatial_dims = stride.size();

  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const mlir::RankedTensorType weight_type = GetTensorTypeOrDie(weight);

  // Dimension numbers for transposed convolution. We transform the weight
  // to match forward convolution format (out, in/G, spatial...) so we can
  // use the same dimension numbers and grouped convolution logic.
  Dimensions input_spatial_dims(num_spatial_dims);
  absl::c_iota(input_spatial_dims, 2);
  Dimensions kernel_spatial_dims(num_spatial_dims);
  absl::c_iota(kernel_spatial_dims, 2);
  Dimensions output_spatial_dims(num_spatial_dims);
  absl::c_iota(output_spatial_dims, 2);

  const auto dimension_numbers = stablehlo::ConvDimensionNumbersAttr::get(
      &ctx,
      /*inputBatchDimension*/ 0,
      /*inputFeatureDimension*/ 1, input_spatial_dims,
      /*kernelInputFeatureDimension*/ 1,
      /*kernelOutputFeatureDimension*/ 0, kernel_spatial_dims,
      /*outputBatchDimension*/ 0,
      /*outputFeatureDimension*/ 1, output_spatial_dims);

  // PyTorch weight: (C_in, C_out/G, kH, kW)
  // Desired weight: (C_out, C_in/G, kH, kW)
  mlir::MlirOp transformed_weight = weight;
  if (groups > 1) {
    int64_t c_in = weight_type.getDimSize(0);
    int64_t c_out_group = weight_type.getDimSize(1);
    int64_t c_in_group = c_in / groups;

    // 1. Reshape to (G, C_in/G, C_out/G, ...)
    Dimensions reshaped_dims = {groups, c_in_group, c_out_group};
    for (size_t i = 2; i < weight_type.getRank(); ++i) {
      reshaped_dims.push_back(weight_type.getDimSize(i));
    }
    transformed_weight =
        stablehlo::Reshape(mlir::RankedTensorType::get(
                               reshaped_dims, weight_type.getElementType()),
                           transformed_weight);

    // 2. Transpose to (G, C_out/G, C_in/G, ...) -> Actually, to get
    // (C_out, C_in/G), we want (G * C_out/G, C_in/G).
    // So we transpose to (G, C_out/G, C_in/G, ...) then reshape.
    Dimensions transpose_perm = {0, 2, 1};
    for (size_t i = 3; i < reshaped_dims.size(); ++i) {
      transpose_perm.push_back(i);
    }
    transformed_weight =
        stablehlo::Transpose(transformed_weight, transpose_perm);

    // 3. Reshape to (C_out, C_in/G, ...)
    Dimensions final_dims = {groups * c_out_group, c_in_group};
    for (size_t i = 2; i < weight_type.getRank(); ++i) {
      final_dims.push_back(weight_type.getDimSize(i));
    }
    transformed_weight = stablehlo::Reshape(
        mlir::RankedTensorType::get(final_dims, weight_type.getElementType()),
        transformed_weight);
  } else {
    // (C_in, C_out) -> (C_out, C_in)
    Dimensions transpose_perm = {1, 0};
    for (size_t i = 2; i < weight_type.getRank(); ++i) {
      transpose_perm.push_back(i);
    }
    transformed_weight =
        stablehlo::Transpose(transformed_weight, transpose_perm);
  }

  const auto window_strides =
      mlir::DenseI64ArrayAttr::get(&ctx, Dimensions(num_spatial_dims, 1));
  const auto lhs_dilation = mlir::DenseI64ArrayAttr::get(&ctx, stride);
  const auto rhs_dilation = mlir::DenseI64ArrayAttr::get(&ctx, dilation);

  // PyTorch conv_transpose2d uses window_reversal=true for the adjoint logic.
  absl::InlinedVector<bool, kNumInlinedDimensions> window_reversal_vec(
      num_spatial_dims, true);
  const auto window_reversal =
      mlir::DenseBoolArrayAttr::get(&ctx, window_reversal_vec);

  Dimensions symmetric_padding_dims;
  symmetric_padding_dims.reserve(num_spatial_dims * 2);
  for (int i = 0; i < num_spatial_dims; ++i) {
    int64_t k = weight_type.getDimSize(2 + i);
    int64_t d = dilation[i];
    int64_t s = stride[i];
    int64_t p = padding[i];
    int64_t in_dim = input_type.getDimSize(2 + i);
    int64_t out_dim = output_dims[2 + i];

    auto [pad_lo, pad_hi] =
        CalculateAdjointPadding(in_dim, out_dim, k, s, p, d);
    symmetric_padding_dims.push_back(pad_lo);
    symmetric_padding_dims.push_back(pad_hi);
  }

  const mlir::RankedTensorType padding_type = mlir::RankedTensorType::get(
      {num_spatial_dims, 2}, builder.getOpBuilder().getI64Type());
  const auto dims_padding = mlir::DenseIntElementsAttr::get(
      padding_type, mlir::ArrayRef<int64_t>(symmetric_padding_dims));

  const stablehlo::Precision precisions[2] = {precision, precision};
  const auto precision_config =
      stablehlo::PrecisionConfigAttr::get(&ctx, precisions);

  const mlir::RankedTensorType result_type = mlir::RankedTensorType::get(
      output_dims, mlir::getElementType(ctx, output_dtype));

  mlir::MlirOp conv_input = stablehlo::ConvertElementType(input, output_dtype);
  transformed_weight =
      stablehlo::ConvertElementType(transformed_weight, output_dtype);

  mlir::MlirOp conv_op = stablehlo::Convolution(
      result_type, conv_input, transformed_weight, dimension_numbers, groups,
      /*batch_group_count=*/1, window_strides, dims_padding, lhs_dilation,
      rhs_dilation, window_reversal, precision_config);

  if (!bias.has_value()) {
    return conv_op;
  }

  bias = stablehlo::ConvertElementType(*bias, output_dtype);
  mlir::MlirOp shaped_bias = stablehlo::BroadcastInDim(
      result_type, bias.value(), /*broadcast_dimensions=*/{1});
  return stablehlo::Add(conv_op, shaped_bias);
}

absl::StatusOr<mlir::MlirOp> BuildConvolution(
    mlir::MlirOp input, mlir::MlirOp weight, std::optional<mlir::MlirOp> bias,
    absl::Span<const int64_t> stride, absl::Span<const int64_t> padding,
    absl::Span<const int64_t> dilation, int64_t groups,
    absl::Span<const int64_t> output_dims, mlir::ElementType output_dtype,
    mlir::stablehlo::Precision precision) {
  mlir::MlirBuilder& builder = input.getBuilder();
  mlir::MLIRContext& ctx = builder.getContext();
  const int num_spatial_dims = stride.size();

  // Strides will have one entry per spatial dimension (1D, 2D, 3D, etc.)
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Callers guarantees that padding.size()
                 // will always have the right size.
      padding.size() == num_spatial_dims * 2, error::kInvalidArgument)
      << "expected the padding size to be " << num_spatial_dims * 2
      << " (2 elements for each of the " << num_spatial_dims
      << " spatial dimensions), got " << padding.size() << " integers "
      << ToString(padding);

  TT_RETURN_IF_ERROR(CheckConvolutionSpatialDimensionsMatch(
      num_spatial_dims, dilation, "dilation"));

  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  // Input should be shaped as (B, C_in, *spatial_dims)
  //   B = batch dimension
  //   C_in = in_channels
  TT_RETURN_IF_ERROR(CheckConvolutionInput(input_type.getShape()));

  const int64_t in_channels = input_type.getDimSize(1);
  const mlir::RankedTensorType weight_type = GetTensorTypeOrDie(weight);

  // Weight should be shaped as (C_out, C_in / groups, *spatial_dims)
  //   C_out = out_channels
  TT_RETURN_IF_ERROR(CheckConvolutionWeight(
      weight_type.getShape(), num_spatial_dims, in_channels, groups,
      /* transposed= */ false));

  // Spatial dims are all but the first two dimensions.
  Dimensions input_spatial_dims(num_spatial_dims);
  absl::c_iota(input_spatial_dims, 2);
  Dimensions kernel_spatial_dims(num_spatial_dims);
  absl::c_iota(kernel_spatial_dims, 2);
  Dimensions output_spatial_dims(num_spatial_dims);
  absl::c_iota(output_spatial_dims, 2);

  // Operation axes are (B, C_in, *spatial_dims) x (C_out, C_in / groups,
  //   *spatial_dims) -> (B, C_out, *spatial_dims)
  const auto dimension_numbers = stablehlo::ConvDimensionNumbersAttr::get(
      &ctx,
      /*inputBatchDimension*/ 0,
      /*inputFeatureDimension*/ 1, input_spatial_dims,
      /*kernelInputFeatureDimension*/ 1,
      /*kernelOutputFeatureDimension*/ 0, kernel_spatial_dims,
      /*outputBatchDimension*/ 0,
      /*outputFeatureDimension*/ 1, output_spatial_dims);

  const uint64_t feature_group_count = groups;
  // Torch does not group over batch, only over features.
  const uint64_t batch_group_count = 1;
  const auto window_strides = mlir::DenseI64ArrayAttr::get(&ctx, stride);

  const mlir::RankedTensorType padding_type = mlir::RankedTensorType::get(
      {num_spatial_dims, 2}, builder.getOpBuilder().getI64Type());
  const auto dims_padding = mlir::DenseIntElementsAttr::get(
      padding_type, mlir::ArrayRef<int64_t>(padding));

  // Torch only dilates the filter, not the input.
  const mlir::DenseI64ArrayAttr lhs_dilation = {};
  const auto rhs_dilation = mlir::DenseI64ArrayAttr::get(&ctx, dilation);
  // Torch does not support window reversal.
  const mlir::DenseBoolArrayAttr window_reversal = {};

  const stablehlo::Precision precisions[2] = {precision, precision};
  const auto precision_config =
      stablehlo::PrecisionConfigAttr::get(&ctx, precisions);

  const mlir::RankedTensorType result_type = mlir::RankedTensorType::get(
      output_dims, mlir::getElementType(ctx, output_dtype));

  input = stablehlo::ConvertElementType(input, output_dtype);
  weight = stablehlo::ConvertElementType(weight, output_dtype);

  auto conv_op = stablehlo::Convolution(
      result_type, input, weight, dimension_numbers, feature_group_count,
      batch_group_count, window_strides, dims_padding, lhs_dilation,
      rhs_dilation, window_reversal, precision_config);

  if (!bias.has_value()) {
    return conv_op;
  }

  const int64_t out_channels = weight_type.getDimSize(0);
  const mlir::RankedTensorType bias_type = GetTensorTypeOrDie(*bias);

  // If it exists, bias should be shaped as (C_out,)
  TT_RETURN_IF_ERROR(CheckConvolutionBias(bias_type.getShape(), out_channels));

  bias = stablehlo::ConvertElementType(*bias, output_dtype);
  // Broadcast bias from shape (C_out,) to (B, C_out, *result_spatial_dims)...
  mlir::MlirOp shaped_bias = stablehlo::BroadcastInDim(
      result_type, bias.value(), /*broadcast_dimensions=*/{1});
  return stablehlo::Add(conv_op, shaped_bias);
}

// Helper function to prepare grouped convolution inputs for convolution
// backward when groups > 1.
GroupedWeightParams PrepareGroupedWeightInputs(
    mlir::MlirOp input, mlir::MlirOp grad_output, int64_t groups,
    int64_t input_batch_dimension, int64_t input_feature_dimension,
    int64_t kernel_input_feature_dimension,
    int64_t kernel_output_feature_dimension) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const mlir::RankedTensorType grad_type = GetTensorTypeOrDie(grad_output);
  auto input_shape = input_type.getShape();
  auto grad_shape = grad_type.getShape();

  int64_t c_in = input_type.getDimSize(1);
  int64_t c_in_g = c_in / groups;

  // 1. Prepare Input (LHS)

  Dimensions input_reshape_dims;
  input_reshape_dims.push_back(input_shape[0]);  // N
  input_reshape_dims.push_back(groups);
  input_reshape_dims.push_back(c_in_g);
  for (size_t i = 2; i < input_shape.size(); ++i) {
    input_reshape_dims.push_back(input_shape[i]);
  }

  input =
      stablehlo::Reshape(mlir::RankedTensorType::get(
                             input_reshape_dims, input_type.getElementType()),
                         input);

  Dimensions input_transpose_perm;
  input_transpose_perm.push_back(2);  // C_in/G
  input_transpose_perm.push_back(1);  // G
  input_transpose_perm.push_back(0);  // N
  for (size_t i = 3; i < input_reshape_dims.size(); ++i) {
    input_transpose_perm.push_back(i);
  }

  input = stablehlo::Transpose(input, input_transpose_perm);

  Dimensions input_flat_dims;
  input_flat_dims.push_back(c_in_g);
  input_flat_dims.push_back(groups * input_shape[0]);  // G*N
  for (size_t i = 2; i < input_shape.size(); ++i) {
    input_flat_dims.push_back(input_shape[i]);
  }
  input = stablehlo::Reshape(
      mlir::RankedTensorType::get(input_flat_dims, input_type.getElementType()),
      input);

  // 2. Prepare GradOutput (RHS)
  int64_t c_out = grad_type.getDimSize(1);
  int64_t c_out_g = c_out / groups;

  Dimensions grad_reshape_dims;
  grad_reshape_dims.push_back(grad_shape[0]);  // N
  grad_reshape_dims.push_back(groups);
  grad_reshape_dims.push_back(c_out_g);
  for (size_t i = 2; i < grad_shape.size(); ++i) {
    grad_reshape_dims.push_back(grad_shape[i]);
  }
  grad_output =
      stablehlo::Reshape(mlir::RankedTensorType::get(
                             grad_reshape_dims, grad_type.getElementType()),
                         grad_output);

  Dimensions grad_transpose_perm;
  grad_transpose_perm.push_back(1);  // G
  grad_transpose_perm.push_back(2);  // C_out/G
  grad_transpose_perm.push_back(0);  // N
  for (size_t i = 3; i < grad_reshape_dims.size(); ++i) {
    grad_transpose_perm.push_back(i);
  }

  grad_output = stablehlo::Transpose(grad_output, grad_transpose_perm);

  Dimensions grad_flat_dims;
  grad_flat_dims.push_back(groups * c_out_g);  // G*C_out/G
  grad_flat_dims.push_back(grad_shape[0]);     // N
  for (size_t i = 2; i < grad_shape.size(); ++i) {
    grad_flat_dims.push_back(grad_shape[i]);
  }

  grad_output = stablehlo::Reshape(
      mlir::RankedTensorType::get(grad_flat_dims, grad_type.getElementType()),
      grad_output);

  std::swap(input_batch_dimension, input_feature_dimension);
  std::swap(kernel_input_feature_dimension, kernel_output_feature_dimension);

  return {input,
          grad_output,
          input_batch_dimension,
          input_feature_dimension,
          kernel_input_feature_dimension,
          kernel_output_feature_dimension};
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildConvolution(
    mlir::MlirOp input, mlir::MlirOp weight, std::optional<mlir::MlirOp> bias,
    absl::Span<const int64_t> stride, absl::Span<const int64_t> padding,
    absl::Span<const int64_t> dilation, bool transposed,
    absl::Span<const int64_t> output_padding, int64_t groups,
    absl::Span<const int64_t> output_dims, mlir::ElementType output_dtype,
    mlir::stablehlo::Precision precision) {
  if (transposed) {
    return BuildTransposedConvolution(input, weight, bias, stride, padding,
                                      dilation, output_padding, groups,
                                      output_dims, output_dtype, precision);
  }

  // Torch pads symmetrically (above and below) on each spatial dimension,
  // so padding is 1D of shape (spatial_dims,).
  // StableHLO allows asymmetric padding, expecting a 2D array of shape
  // (spatial_dims, 2).
  Dimensions symmetric_padding_dims;
  symmetric_padding_dims.reserve(padding.size() * 2);
  for (int64_t p : padding) {
    symmetric_padding_dims.push_back(p);
    symmetric_padding_dims.push_back(p);
  }
  return BuildConvolution(input, weight, bias, stride, symmetric_padding_dims,
                          dilation, groups, output_dims, output_dtype,
                          precision);
}

// From the chain The gradient of loss function L w.r.t. input x equals dL/y *
// dy/dx, where y = conv(x, w) is the output of the convolution. For
// colvolution, dL/dx = transposed_convolution(dL/dy, w).
absl::StatusOr<mlir::MlirOp> BuildConvolutionBackwardInput(
    mlir::MlirOp grad_output, mlir::MlirOp weight,
    absl::Span<const int64_t> input_dims, absl::Span<const int64_t> stride,
    absl::Span<const int64_t> padding, absl::Span<const int64_t> dilation,
    int64_t groups, bool transposed, absl::Span<const int64_t> output_padding,
    mlir::ElementType output_dtype, mlir::stablehlo::Precision precision) {
  if (transposed) {
    // When computing the gradient w.r.t. input for a transposed convolution,
    // we perform a standard convolution.
    // The padding required for this backward convolution must be calculated
    // to invert the effect of forward padding and output_padding.
    // Forward (transposed):
    //   H_out = (H_in - 1)*S - 2*P + D*(K-1) + OP + 1
    // Backward (standard):
    //   H_in = (H_out + P_grad - D*(K-1) - 1)/S + 1
    //   where P_grad = Pad_lo + Pad_hi
    //
    // From derived formula: P_grad = 2*P - OP.
    // We split this as: Pad_lo = P, Pad_hi = P - OP.
    // Note that Pad_hi can be negative if OP > P, effectively cropping the
    // output.
    Dimensions asymmetric_padding;
    asymmetric_padding.reserve(stride.size() * 2);
    for (size_t i = 0; i < stride.size(); ++i) {
      int64_t p = padding[i];
      int64_t op = output_padding.empty() ? 0 : output_padding[i];
      asymmetric_padding.push_back(p);
      asymmetric_padding.push_back(p - op);
    }
    return BuildConvolution(grad_output, weight, std::nullopt, stride,
                            asymmetric_padding, dilation, groups, input_dims,
                            output_dtype, precision);
  }

  mlir::MlirBuilder& builder = grad_output.getBuilder();
  mlir::MLIRContext& ctx = builder.getContext();
  const int num_spatial_dims = stride.size();

  mlir::RankedTensorType grad_out_type = GetTensorTypeOrDie(grad_output);
  auto grad_out_shape = grad_out_type.getShape();
  mlir::RankedTensorType weight_type = GetTensorTypeOrDie(weight);
  auto weight_shape = weight_type.getShape();

  if (groups > 1) {
    // Current weight shape: (C_out, C_in/groups, spatial...)
    // C_out = groups * (C_out/groups).

    // We want to transform it to be compatible with grouped convolution where
    // Input is GradOutput (C_out split into groups), and Output is GradInput
    // (C_in split into groups).

    // GradOutput: (N, C_out, ...) -> (N, G, C_out/G, ...)
    // GradInput: (N, C_in, ...) -> (N, G, C_in/G, ...)

    // Kernel should map C_out/G to C_in/G for each group G.
    // Original Weight: (C_out, C_in/G, ...)
    // Reshaped Weight: (G, C_out/G, C_in/G, ...)
    // Transpose to: (G, C_in/G, C_out/G, ...) for Conv Backprop Input logic?
    //
    // Actually, to use feature_group_count = G:
    // LHS (GradOutput): (N, C_out, ...) -> (N, G * C_out/G, ...)
    // RHS (Kernel): needs to map C_out/G -> C_in/G.
    // Kernel Input Feature Dim Size: C_out/G.
    // Kernel Output Feature Dim Size: C_in (G * C_in/G).
    //
    // So we need Kernel shape to have InputFeatureDim = C_out/G and
    // OutputFeatureDim = C_in.
    //
    // Original Weight: (C_out, C_in/G, ...).
    // Reshape: (G, C_out/G, C_in/G, ...).
    // Transpose to: (C_out/G, G, C_in/G, ...).
    // Reshape to: (C_out/G, G * C_in/G, ...) = (C_out/G, C_in, ...).
    //
    // Then set dnums:
    // Kernel Input Feature = 0.
    // Kernel Output Feature = 1.

    int64_t c_out = weight_shape[0];
    int64_t c_in_group = weight_shape[1];
    int64_t c_out_group = c_out / groups;

    Dimensions reshaped_dims;
    reshaped_dims.push_back(groups);
    reshaped_dims.push_back(c_out_group);
    reshaped_dims.push_back(c_in_group);
    for (size_t i = 2; i < weight_shape.size(); ++i) {
      reshaped_dims.push_back(weight_shape[i]);
    }

    auto reshaped_type = mlir::RankedTensorType::get(
        reshaped_dims, weight_type.getElementType());
    weight = stablehlo::Reshape(reshaped_type, weight);

    // Transpose to (C_out/group, groups, C_in/group, spatial...)
    Dimensions transpose_perm;
    transpose_perm.push_back(1);
    transpose_perm.push_back(0);
    transpose_perm.push_back(2);
    for (size_t i = 3; i < reshaped_dims.size(); ++i) {
      transpose_perm.push_back(i);
    }

    // Transpose doesn't take result type
    weight = stablehlo::Transpose(weight, transpose_perm);

    // Reshape to (C_out/group, C_in, spatial...)
    // C_in = groups * c_in_group
    Dimensions final_dims;
    final_dims.push_back(c_out_group);
    final_dims.push_back(groups * c_in_group);
    for (size_t i = 2; i < weight_shape.size(); ++i) {
      final_dims.push_back(weight_shape[i]);
    }

    auto final_type =
        mlir::RankedTensorType::get(final_dims, weight_type.getElementType());
    weight = stablehlo::Reshape(final_type, weight);
    weight_type = GetTensorTypeOrDie(weight);
    weight_shape = weight_type.getShape();
  }

  Dimensions input_spatial_dims(num_spatial_dims);
  absl::c_iota(input_spatial_dims, 2);
  Dimensions kernel_spatial_dims(num_spatial_dims);
  absl::c_iota(kernel_spatial_dims, 2);
  Dimensions output_spatial_dims(num_spatial_dims);
  absl::c_iota(output_spatial_dims, 2);

  // Input: grad_output (B, C_out, ...)
  // Weight: (C_out, C_in/G, ...)
  // Output: grad_input (B, C_in, ...)
  const auto dimension_numbers = stablehlo::ConvDimensionNumbersAttr::get(
      &ctx,
      /*inputBatchDimension*/ 0,
      /*inputFeatureDimension*/ 1, input_spatial_dims,
      /*kernelInputFeatureDimension*/ 0,
      /*kernelOutputFeatureDimension*/ 1, kernel_spatial_dims,
      /*outputBatchDimension*/ 0,
      /*outputFeatureDimension*/ 1, output_spatial_dims);

  // lhs_dilation = stride
  const auto window_strides =
      mlir::DenseI64ArrayAttr::get(&ctx, Dimensions(num_spatial_dims, 1));
  const auto lhs_dilation = mlir::DenseI64ArrayAttr::get(&ctx, stride);
  const auto rhs_dilation = mlir::DenseI64ArrayAttr::get(&ctx, dilation);

  // window_reversal = true
  absl::InlinedVector<bool, kNumInlinedDimensions> window_reversal_vec(
      num_spatial_dims, true);
  const auto window_reversal =
      mlir::DenseBoolArrayAttr::get(&ctx, window_reversal_vec);

  // Padding
  Dimensions symmetric_padding_dims;
  symmetric_padding_dims.reserve(num_spatial_dims * 2);
  for (int i = 0; i < num_spatial_dims; ++i) {
    int64_t k = weight_shape[2 + i];
    int64_t d = dilation[i];
    int64_t s = stride[i];
    int64_t p = padding[i];
    int64_t in_dim = input_dims[2 + i];
    int64_t out_dim = grad_out_shape[2 + i];

    auto [pad_lo, pad_hi] =
        CalculateAdjointPadding(out_dim, in_dim, k, s, p, d);

    symmetric_padding_dims.push_back(pad_lo);
    symmetric_padding_dims.push_back(pad_hi);
  }

  // Logic: pad_lo = (k-1)*d - p
  //        pad_hi = (k-1)*d - p + op
  // input_dims is passed. We can derive output_padding.
  // Actually, we can just calculate padding needed to match input_dims.

  const mlir::RankedTensorType padding_type = mlir::RankedTensorType::get(
      {num_spatial_dims, 2}, builder.getOpBuilder().getI64Type());
  const auto dims_padding = mlir::DenseIntElementsAttr::get(
      padding_type, mlir::ArrayRef<int64_t>(symmetric_padding_dims));

  const stablehlo::Precision precisions[2] = {precision, precision};
  const auto precision_config =
      stablehlo::PrecisionConfigAttr::get(&ctx, precisions);

  Dimensions output_dims_vec(input_dims.begin(), input_dims.end());
  const mlir::RankedTensorType result_type = mlir::RankedTensorType::get(
      output_dims_vec, mlir::getElementType(ctx, output_dtype));

  grad_output = stablehlo::ConvertElementType(grad_output, output_dtype);
  weight = stablehlo::ConvertElementType(weight, output_dtype);

  return stablehlo::Convolution(
      result_type, grad_output, weight, dimension_numbers, groups,
      /*batch_group_count=*/1, window_strides, dims_padding, lhs_dilation,
      rhs_dilation, window_reversal, precision_config);
}

absl::StatusOr<mlir::MlirOp> BuildConvolutionBackwardWeight(
    mlir::MlirOp input, mlir::MlirOp grad_output,
    absl::Span<const int64_t> weight_dims, absl::Span<const int64_t> stride,
    absl::Span<const int64_t> padding, absl::Span<const int64_t> dilation,
    int64_t groups, bool transposed, absl::Span<const int64_t> output_padding,
    mlir::ElementType output_dtype, mlir::stablehlo::Precision precision) {
  if (transposed) {
    std::swap(input, grad_output);
  }
  mlir::MlirBuilder& builder = input.getBuilder();
  mlir::MLIRContext& ctx = builder.getContext();
  const int num_spatial_dims = stride.size();

  int input_batch_dimension = 1;
  int input_feature_dimension = 0;
  int kernel_input_feature_dimension = 0;
  int kernel_output_feature_dimension = 1;

  if (groups > 1) {
    // Optimized grouped convolution backward weight using feature_group_count.
    //
    // Input (LHS): (N, C_in, spatial...)
    // GradOutput (RHS): (N, C_out, spatial...)
    //
    // We reshape dimensions to align with StableHLO's convolution with feature
    // groups.
    //
    // LHS: (N, G, C_in/G, ...) -> (C_in/G, G, N, ...) -> (C_in/G, G*N, ...)
    //   Batch: C_in/G
    //   Feature: G*N
    //
    // RHS: (N, G, C_out/G, ...) -> (G, C_out/G, N, ...) -> (G*C_out/G, N, ...)
    //   OutFeature: G*C_out/G
    //   InFeature: N
    //
    // feature_group_count = G.
    // LHS Feature (G*N) is split into G groups of N.
    // RHS OutFeature (G*C_out/G) is split into G groups of C_out/G.
    //
    // Result: (C_in/G, G*C_out/G, ...) -> (C_out, C_in/G, ...) via transpose.
    auto params = PrepareGroupedWeightInputs(
        input, grad_output, groups, input_batch_dimension,
        input_feature_dimension, kernel_input_feature_dimension,
        kernel_output_feature_dimension);
    input = params.input;
    grad_output = params.grad_output;
    input_batch_dimension = params.input_batch_dim;
    input_feature_dimension = params.input_feature_dim;
    kernel_input_feature_dimension = params.kernel_in_feat_dim;
    kernel_output_feature_dimension = params.kernel_out_feat_dim;
  }

  // 3. Convolution

  Dimensions input_spatial_dims(num_spatial_dims);
  absl::c_iota(input_spatial_dims, 2);
  Dimensions kernel_spatial_dims(num_spatial_dims);
  absl::c_iota(kernel_spatial_dims, 2);
  Dimensions output_spatial_dims(num_spatial_dims);
  absl::c_iota(output_spatial_dims, 2);

  const auto dimension_numbers = stablehlo::ConvDimensionNumbersAttr::get(
      &ctx, input_batch_dimension, input_feature_dimension, input_spatial_dims,
      kernel_input_feature_dimension, kernel_output_feature_dimension,
      kernel_spatial_dims,
      /*outputBatchDimension*/ 0,
      /*outputFeatureDimension*/ 1, output_spatial_dims);

  const auto window_strides = mlir::DenseI64ArrayAttr::get(&ctx, dilation);
  const auto lhs_dilation =
      mlir::DenseI64ArrayAttr::get(&ctx, Dimensions(num_spatial_dims, 1));
  const auto rhs_dilation = mlir::DenseI64ArrayAttr::get(&ctx, stride);
  absl::InlinedVector<bool, kNumInlinedDimensions> window_reversal_vec(
      num_spatial_dims, false);
  const auto window_reversal =
      mlir::DenseBoolArrayAttr::get(&ctx, window_reversal_vec);

  mlir::RankedTensorType grad_type = GetTensorTypeOrDie(grad_output);
  auto grad_shape = grad_type.getShape();

  mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  auto input_shape = input_type.getShape();

  Dimensions symmetric_padding_dims;
  symmetric_padding_dims.reserve(num_spatial_dims * 2);

  for (int i = 0; i < num_spatial_dims; ++i) {
    int64_t p = padding[i];
    int64_t s = stride[i];
    int64_t d = dilation[i];
    int64_t k = weight_dims[2 + i];
    int64_t in_dim = input_shape[2 + i];
    int64_t out_dim = grad_shape[2 + i];

    auto [pad_lo, pad_hi] =
        CalculateBackwardWeightPadding(in_dim, out_dim, k, s, p, d);

    symmetric_padding_dims.push_back(pad_lo);
    symmetric_padding_dims.push_back(pad_hi);
  }

  const mlir::RankedTensorType padding_type = mlir::RankedTensorType::get(
      {num_spatial_dims, 2}, builder.getOpBuilder().getI64Type());
  const auto dims_padding = mlir::DenseIntElementsAttr::get(
      padding_type, mlir::ArrayRef<int64_t>(symmetric_padding_dims));

  const stablehlo::Precision precisions[2] = {precision, precision};
  const auto precision_config =
      stablehlo::PrecisionConfigAttr::get(&ctx, precisions);

  Dimensions intermediate_dims;
  intermediate_dims.push_back(weight_dims[1]);  // C_in
  intermediate_dims.push_back(weight_dims[0]);  // C_out
  for (size_t i = 0; i < num_spatial_dims; ++i) {
    intermediate_dims.push_back(weight_dims[2 + i]);
  }

  const mlir::RankedTensorType intermediate_type = mlir::RankedTensorType::get(
      intermediate_dims, mlir::getElementType(ctx, output_dtype));
  input = stablehlo::ConvertElementType(input, output_dtype);
  grad_output = stablehlo::ConvertElementType(grad_output, output_dtype);

  auto conv = stablehlo::Convolution(
      intermediate_type, input, grad_output, dimension_numbers,
      /*feature_group_count=*/groups,
      /*batch_group_count=*/1, window_strides, dims_padding, lhs_dilation,
      rhs_dilation, window_reversal, precision_config);

  Dimensions transpose_perm;
  transpose_perm.push_back(1);
  transpose_perm.push_back(0);
  for (size_t i = 0; i < num_spatial_dims; ++i) {
    transpose_perm.push_back(2 + i);
  }

  return stablehlo::Transpose(conv, transpose_perm);
}

absl::StatusOr<mlir::MlirOp> BuildConvolutionBackwardBias(
    mlir::MlirOp grad_output, absl::Span<const int64_t> output_padding,
    mlir::ElementType output_dtype) {
  mlir::MlirBuilder& builder = grad_output.getBuilder();
  mlir::RankedTensorType grad_type = GetTensorTypeOrDie(grad_output);
  int64_t rank = grad_type.getRank();
  SmallInt64Vector reduction_dims;
  reduction_dims.push_back(0);  // Batch
  for (int64_t i = 2; i < rank; ++i) {
    reduction_dims.push_back(i);  // Spatial
  }

  grad_output = stablehlo::ConvertElementType(grad_output, output_dtype);

  mlir::MlirOp init_val = MakeScalarConstant(builder, 0, output_dtype);

  absl::Status reduce_status;
  auto body_builder = [&](mlir::RegionBuilder& rb) {
    reduce_status = BuildReduceBody(
        rb, mlir::getElementType(builder.getContext(), output_dtype),
        c10d::ReduceOp::SUM);
  };

  auto results = stablehlo::Reduce(builder, {grad_output}, {init_val},
                                   body_builder, reduction_dims);
  TT_RETURN_IF_ERROR(reduce_status);
  return results[0];
}

}  // namespace torch_tpu
