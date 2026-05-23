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

#include "torch_tpu/ops/im2col/im2col_aten_kernels.h"

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
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/im2col/im2col.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {

absl::StatusOr<DeviceBufferRef> AtenIm2Col(
    const OpName op_name, const at::Tensor& self, at::IntArrayRef kernel_size,
    at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride,
    OpParamCacheKeys param_keys) {
  TT_RET_CHECK(kernel_size.size() == 2, error::kInvalidArgument)
      << "expected kernel_size to have 2 dimensions, got "
      << kernel_size.size();
  TT_RET_CHECK(dilation.size() == 2, error::kInvalidArgument)
      << "expected dilation to have 2 dimensions, got " << dilation.size();
  TT_RET_CHECK(padding.size() == 2, error::kInvalidArgument)
      << "expected padding to have 2 dimensions, got " << padding.size();
  TT_RET_CHECK(stride.size() == 2, error::kInvalidArgument)
      << "expected stride to have 2 dimensions, got " << stride.size();

  const bool is_3d = self.dim() == 3;
  at::Tensor input = is_3d ? self.unsqueeze(0) : self;

  TT_RET_CHECK(input.dim() == 4, error::kInvalidArgument)
      << "expected input to have 3 or 4 dimensions, got " << self.dim();

  const int64_t batch = input.size(0);
  const int64_t channels = input.size(1);
  const int64_t h = input.size(2);
  const int64_t w = input.size(3);

  const int64_t kH = kernel_size[0];
  const int64_t kW = kernel_size[1];
  const int64_t dH = dilation[0];
  const int64_t dW = dilation[1];
  const int64_t pH = padding[0];
  const int64_t pW = padding[1];
  const int64_t sH = stride[0];
  const int64_t sW = stride[1];

  const int64_t out_h = (h + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
  const int64_t out_w = (w + 2 * pW - dW * (kW - 1) - 1) / sW + 1;

  TT_RET_CHECK(out_h > 0 && out_w > 0, error::kInvalidArgument)
      << "expected output shape to be positive, got " << out_h << "x" << out_w;

  Dimensions output_dims;
  if (is_3d) {
    output_dims = {channels * kH * kW, out_h * out_w};
  } else {
    output_dims = {batch, channels * kH * kW, out_h * out_w};
  }

  auto op_builder =
      [kernel_size = CopyIntVector(kernel_size),
       dilation = CopyIntVector(dilation), padding = CopyIntVector(padding),
       stride = CopyIntVector(stride),
       is_3d](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
    TT_ASSIGN_OR_RETURN(
        auto result,
        BuildIm2ColShlo(input, kernel_size, dilation, padding, stride));
    if (is_3d) {
      // BuildIm2ColShlo returns (1, C*kH*kW, L) for 3D input that was
      // unsqueezed. Squeeze it back to (C*kH*kW, L).
      return Squeeze(result, {0});
    }
    return absl::StatusOr<mlir::MlirOp>(result);
  };

  TT_ASSIGN_OR_RETURN(const auto elem_type,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));

  return DispatchOp<1>(std::move(op_builder), input,
                       {.op_name = op_name,
                        .out_dtype = elem_type,
                        .out_dims = output_dims,
                        .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor AtenIm2Col(const at::Tensor& self, at::IntArrayRef kernel_size,
                      at::IntArrayRef dilation, at::IntArrayRef padding,
                      at::IntArrayRef stride) {
  TT_KERNEL(OpName::kIm2Col, param_keys,
            (self, kernel_size, dilation, padding, stride), {
              TT_ASSIGN_OR_THROW(
                  DeviceBufferRef result,
                  AtenIm2Col(OpName::kIm2Col, self, kernel_size, dilation,
                             padding, stride, std::move(param_keys)));
              return MakeTensor(std::move(result));
            });
}

at::Tensor& AtenIm2ColOut(const at::Tensor& self, at::IntArrayRef kernel_size,
                          at::IntArrayRef dilation, at::IntArrayRef padding,
                          at::IntArrayRef stride, at::Tensor& out) {
  TT_KERNEL(OpName::kIm2ColOut, param_keys,
            (self, kernel_size, dilation, padding, stride, out), {
              TT_ASSIGN_OR_THROW(
                  DeviceBufferRef result,
                  AtenIm2Col(OpName::kIm2ColOut, self, kernel_size, dilation,
                             padding, stride, std::move(param_keys)));
              TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
              return out;
            });
}

}  // namespace torch_tpu
