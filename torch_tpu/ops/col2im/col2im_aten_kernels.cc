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

#include "torch_tpu/ops/col2im/col2im_aten_kernels.h"

#include <algorithm>
#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/col2im/col2im.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {

// Returns the number of blocks to slide over the input tensor.
//
// All of the input arrays should have exactly 2 elements. This condition is
// directly (by `GetOutputDimensions()` function) or indirectly (by
// `AtenCol2Im()` function) checked calling the `GetOutputDimensions()`
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

// Returns the output dimensions for a col2im operation, and validates
// consistency between input shape and output parameters.
absl::StatusOr<Dimensions> GetOutputDimensions(const at::Tensor& input,
                                               at::IntArrayRef output_size,
                                               at::IntArrayRef kernel_size,
                                               at::IntArrayRef dilation,
                                               at::IntArrayRef padding,
                                               at::IntArrayRef stride) {
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
      const auto col_size,
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
  return Dimensions{n, c, output_size[0], output_size[1]};
}

absl::StatusOr<DeviceBufferRef> AtenCol2Im(const at::Tensor& input,
                                           at::IntArrayRef output_size,
                                           at::IntArrayRef kernel_size,
                                           at::IntArrayRef dilation,
                                           at::IntArrayRef padding,
                                           at::IntArrayRef stride,
                                           OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(Dimensions output_dims,
                      GetOutputDimensions(input, output_size, kernel_size,
                                          dilation, padding, stride));

  TT_ASSIGN_OR_RETURN(
      const auto col_size,
      ComputeOutputSize(output_size, padding, dilation, kernel_size, stride));

  auto op_builder =
      [output_size = CopyIntVector(output_size), col_size = std::move(col_size),
       kernel_size = CopyIntVector(kernel_size),
       dilation = CopyIntVector(dilation), padding = CopyIntVector(padding),
       stride = CopyIntVector(stride)](mlir::MlirOp input) {
        return BuildCol2ImShlo(input, output_size, col_size, kernel_size,
                               dilation, padding, stride);
      };

  TT_ASSIGN_OR_RETURN(const auto elem_type,
                      ConvertTo<mlir::ElementType>(input.scalar_type()));

  return DispatchOp<1>(std::move(op_builder), {input},
                       {.out_dtype = elem_type,
                        .out_dims = output_dims,
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
