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

#include "csrc/ops/convolution/convolution_checks.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"

namespace torch_tpu {

namespace {

absl::Status ValidateNonZeroDimensions(absl::Span<const int64_t> dims,
                                       const std::string_view message,
                                       const size_t start_index = 0) {
  for (size_t i = start_index; i < dims.size(); ++i) {
    TT_RET_CHECK(dims[i] != 0, error::kInvalidArgument)
        << message << ToString(dims);
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ValidateConvolutionInput(absl::Span<const int64_t> input) {
  TT_RET_CHECK(input.size() > 2, error::kInvalidArgument)
      << "expected the input to have >= 3 dimensions of shape [batch, in "
         "channels, ... spatial dimensions ...], got shape "
      << ToString(input);
  return ValidateNonZeroDimensions(
      input,
      "expected non-zero input channels and spatial dimensions, got shape ",
      /*start_index=*/1);
}

absl::Status ValidateConvolutionSpatialDimensionsMatch(
    int num_spatial_dims, absl::Span<const int64_t> thing,
    std::string_view arg_name) {
  TT_RET_CHECK(thing.size() == num_spatial_dims, error::kInvalidArgument)
      << "expected " << arg_name << " to be either an integer or a "
      << num_spatial_dims
      << "-element list that matches the convolution dimensions, got "
      << ToString(thing);

  return absl::OkStatus();
}

absl::Status ValidateConvolutionSpatialParameters(
    absl::Span<const int64_t> stride, absl::Span<const int64_t> padding,
    absl::Span<const int64_t> dilation,
    absl::Span<const int64_t> output_padding, const bool transposed) {
  for (size_t i = 0; i < stride.size(); ++i) {
    TT_RET_CHECK(stride[i] > 0, error::kInvalidArgument)
        << "expected stride to be greater than 0, got " << ToString(stride);
    TT_RET_CHECK(padding[i] >= 0, error::kInvalidArgument)
        << "expected padding to be >= 0, got " << ToString(padding);
    TT_RET_CHECK(dilation[i] > 0, error::kInvalidArgument)
        << "expected dilation to be greater than 0, got " << ToString(dilation);
    TT_RET_CHECK(output_padding[i] >= 0, error::kInvalidArgument)
        << "expected output_padding to be >= 0, got "
        << ToString(output_padding);
    TT_RET_CHECK(!transposed || output_padding[i] < stride[i] ||
                     output_padding[i] < dilation[i],
                 error::kInvalidArgument)
        << "expected output_padding to be smaller than either stride or "
           "dilation, got output_padding="
        << ToString(output_padding) << ", stride=" << ToString(stride)
        << ", dilation=" << ToString(dilation);
  }
  return absl::OkStatus();
}

absl::Status ValidateConvolutionWeight(absl::Span<const int64_t> weight,
                                       const int64_t num_spatial_dims,
                                       const int64_t in_channels,
                                       const int64_t groups,
                                       const bool transposed) {
  TT_RET_CHECK(groups > 0, error::kInvalidArgument)
      << "expected groups to be greater than 0, got " << groups;
  TT_RET_CHECK(in_channels % groups == 0, error::kInvalidArgument)
      << "expected in_channels to be divisible by groups, got in_channels="
      << in_channels << " and groups=" << groups;
  if (transposed) {
    TT_RET_CHECK(weight.size() == num_spatial_dims + 2, error::kInvalidArgument)
        << "expected the weight tensor to have " << num_spatial_dims + 2
        << " dimensions of shape [in channels, out channels per group, ... "
        << num_spatial_dims << " spatial dimensions ...], got shape "
        << ToString(weight);

    TT_RET_CHECK(weight[0] == in_channels, error::kInvalidArgument)
        << "expected the first dimension of the weight tensor of shape "
        << ToString(weight) << " to be " << in_channels
        << " (number of in channels), got " << weight[0];
    return ValidateNonZeroDimensions(
        weight, "expected non-zero weight dimensions, got shape ");
  }

  TT_RET_CHECK(weight.size() == num_spatial_dims + 2, error::kInvalidArgument)
      << "expected the weight tensor to have " << num_spatial_dims + 2
      << " dimensions of shape [out channels, in channels per group, ... "
      << num_spatial_dims << " spatial dimensions ...], got shape "
      << ToString(weight);

  TT_RET_CHECK(weight[1] * groups == in_channels, error::kInvalidArgument)
      << "expected the second dimension of the weight tensor of shape "
      << ToString(weight) << " to be " << in_channels / groups << " ("
      << in_channels << " in channels divided by " << groups << " groups), got "
      << weight[1];
  TT_RET_CHECK(weight[0] % groups == 0, error::kInvalidArgument)
      << "expected out_channels to be divisible by groups, got out_channels="
      << weight[0] << " and groups=" << groups;
  return ValidateNonZeroDimensions(
      weight, "expected non-zero weight dimensions, got shape ");
}

absl::Status ValidateConvolutionBias(absl::Span<const int64_t> bias,
                                     const int64_t out_channels) {
  TT_RET_CHECK(bias.size() == 1 && bias[0] == out_channels,
               error::kInvalidArgument)
      << "expected the bias tensor to have 1 dimension of shape ["
      << out_channels << " (out channels)], got shape " << ToString(bias);

  return absl::OkStatus();
}

}  // namespace torch_tpu
