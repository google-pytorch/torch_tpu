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

#include "csrc/ops/im2col/im2col_aten_kernels.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Value.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildIm2ColShlo(mlir::MlirOp input,
                                             SmallInt64Vector kernel_size,
                                             SmallInt64Vector dilation,
                                             SmallInt64Vector padding,
                                             SmallInt64Vector stride) {
  auto& builder = input.getBuilder();

  // Input: (N, C, H, W)
  const auto input_type = GetTensorTypeOrDie(input);
  const auto input_shape = input_type.getShape();
  TT_RET_CHECK(input_shape.size() == 4, error::kInvalidArgument)
      << "input must be 4D, got " << input_shape.size() << "D";

  const int64_t batch = input_shape[0];
  const int64_t channels = input_shape[1];
  const int64_t h = input_shape[2];
  const int64_t w = input_shape[3];

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

  // 1. Pad input

  mlir::MlirOp padding_value =
      MakeScalarConstant(builder, 0.0, input_type.getElementType());

  auto padded_input = mlir::stablehlo::Pad(
      input, padding_value, llvm::ArrayRef<int64_t>({0, 0, pH, pW}),
      llvm::ArrayRef<int64_t>({0, 0, pH, pW}),
      llvm::ArrayRef<int64_t>({0, 0, 0, 0}));

  // 2. Extract slices
  std::vector<mlir::Value> slices;
  slices.reserve(kH * kW);

  for (int64_t i = 0; i < kH; ++i) {
    for (int64_t j = 0; j < kW; ++j) {
      // Start indices for this window element: [0, 0, i * dH, j * dW]
      SmallInt64Vector start_indices = {0, 0, i * dH, j * dW};
      // End indices: [batch, channels, i*dH + (out_h-1)*sH + 1,
      //               j*dW + (out_w-1)*sW + 1]
      SmallInt64Vector limit_indices = {batch, channels,
                                        i * dH + (out_h - 1) * sH + 1,
                                        j * dW + (out_w - 1) * sW + 1};
      SmallInt64Vector strides = {1, 1, sH, sW};

      auto slice = mlir::stablehlo::Slice(padded_input, start_indices,
                                          limit_indices, strides);
      slices.push_back(slice.getValue());
    }
  }

  // 3. Concat slices along a new dimension
  // Each slice is (N, C, out_h, out_w). Reshape each to (N, C, 1, out_h,
  // out_w).
  std::vector<mlir::MlirOp> reshaped_slices;
  reshaped_slices.reserve(kH * kW);
  for (auto& slice_val : slices) {
    auto slice_op = mlir::MlirOp(builder, slice_val);
    auto reshaped_slice = mlir::stablehlo::Reshape(
        slice_op, llvm::ArrayRef<int64_t>({batch, channels, 1, out_h, out_w}));
    reshaped_slices.push_back(reshaped_slice);
  }

  auto concat = mlir::stablehlo::Concatenate(builder, reshaped_slices, 2);

  // 4. Transpose and Reshape
  // Current: (N, C, kH*kW, out_h, out_w)
  // Goal: (N, C * kH * kW, out_h * out_w)
  // PyTorch im2col order is (N, C, kH, kW, out_H, out_W) ->
  // (N, C*kH*kW, out_H*out_W).
  // My slices loop was:
  // for i in kH:
  //   for j in kW:
  // So the concat dimension (index 2) has kH * kW elements in order
  // (0,0), (0,1), ..., (kH-1, kW-1). This matches (N, C, kH, kW, out_h, out_w)
  // after a reshape of the concat dim.
  // Reshape to (N, C, kH, kW, out_h, out_w)
  auto reshaped_concat = mlir::stablehlo::Reshape(
      concat, llvm::ArrayRef<int64_t>({batch, channels, kH, kW, out_h, out_w}));

  // In PyTorch, channels are grouped first: (N, C * kH * kW, L).
  // The layout is that for each batch, we have C groups, and each group has
  // kH*kW elements. My reshaped_concat is (N, C, kH, kW, out_h, out_w).
  // Reshaping to (N, C * kH * kW, out_h * out_w) should be correct.
  return mlir::stablehlo::Reshape(
      reshaped_concat,
      llvm::ArrayRef<int64_t>({batch, channels * kH * kW, out_h * out_w}));
}

absl::StatusOr<DeviceBufferRef> AtenIm2Col(const at::Tensor& self,
                                           at::IntArrayRef kernel_size,
                                           at::IntArrayRef dilation,
                                           at::IntArrayRef padding,
                                           at::IntArrayRef stride,
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
                       {.out_dtype = elem_type,
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
                  AtenIm2Col(self, kernel_size, dilation, padding, stride,
                             std::move(param_keys)));
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
                  AtenIm2Col(self, kernel_size, dilation, padding, stride,
                             std::move(param_keys)));
              TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result), out));
              return out;
            });
}

}  // namespace torch_tpu
