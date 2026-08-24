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

#include "torch_tpu/csrc/ops/convolution/convolution_checks.h"

#include <cstdint>
#include <string_view>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/to_string.h"

namespace torch_tpu {

absl::Status CheckConvolutionInput(absl::Span<const int64_t> input) {
  TT_RET_CHECK(input.size() > 2, error::kInvalidArgument)
      << "expected the input to have >= 3 dimensions of shape [batch, in "
         "channels, ... spatial dimensions ...], got shape "
      << ToString(input);

  return absl::OkStatus();
}

absl::Status CheckConvolutionSpatialDimensionsMatch(
    int num_spatial_dims, absl::Span<const int64_t> thing,
    std::string_view arg_name) {
  TT_RET_CHECK(thing.size() == num_spatial_dims, error::kInvalidArgument)
      << "expected " << arg_name << " to be either an integer or a "
      << num_spatial_dims
      << "-element list that matches the convolution dimensions, got "
      << ToString(thing);

  return absl::OkStatus();
}

absl::Status CheckConvolutionWeight(absl::Span<const int64_t> weight,
                                    const int64_t num_spatial_dims,
                                    const int64_t in_channels,
                                    const int64_t groups,
                                    const bool transposed) {
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
  } else {
    TT_RET_CHECK(weight.size() == num_spatial_dims + 2, error::kInvalidArgument)
        << "expected the weight tensor to have " << num_spatial_dims + 2
        << " dimensions of shape [out channels, in channels per group, ... "
        << num_spatial_dims << " spatial dimensions ...], got shape "
        << ToString(weight);

    TT_RET_CHECK(weight[1] * groups == in_channels, error::kInvalidArgument)
        << "expected the second dimension of the weight tensor of shape "
        << ToString(weight) << " to be " << in_channels / groups << " ("
        << in_channels << " in channels divided by " << groups
        << " groups), got " << weight[1];
  }

  return absl::OkStatus();
}

absl::Status CheckConvolutionBias(absl::Span<const int64_t> bias,
                                  const int64_t out_channels) {
  TT_RET_CHECK(bias.size() == 1 && bias[0] == out_channels,
               error::kInvalidArgument)
      << "expected the bias tensor to have 1 dimension of shape ["
      << out_channels << " (out channels)], got shape " << ToString(bias);

  return absl::OkStatus();
}

}  // namespace torch_tpu
