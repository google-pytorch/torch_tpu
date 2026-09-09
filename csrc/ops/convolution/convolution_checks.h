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

#ifndef TORCH_TPU_CSRC_OPS_CONVOLUTION_CONVOLUTION_CHECKS_H_
#define TORCH_TPU_CSRC_OPS_CONVOLUTION_CONVOLUTION_CHECKS_H_

#include <cstdint>
#include <string_view>

#include "absl/status/status.h"
#include "absl/types/span.h"

namespace torch_tpu {

absl::Status ValidateConvolutionInput(absl::Span<const int64_t> input);

absl::Status ValidateConvolutionSpatialDimensionsMatch(
    int num_spatial_dims, absl::Span<const int64_t> thing,
    std::string_view arg_name);

absl::Status ValidateConvolutionWeight(absl::Span<const int64_t> weight,
                                       int64_t num_spatial_dims,
                                       int64_t in_channels, int64_t groups,
                                       bool transposed);

absl::Status ValidateConvolutionBias(absl::Span<const int64_t> bias,
                                     int64_t out_channels);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_CONVOLUTION_CONVOLUTION_CHECKS_H_
