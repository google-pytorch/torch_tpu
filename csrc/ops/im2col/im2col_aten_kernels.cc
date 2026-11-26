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
#include "mlir/IR/BuiltinTypes.h"
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

absl::StatusOr<mlir::MlirOp> BuildIm2ColShlo(
    mlir::MlirOp input, const SmallInt64Vector& out_size,
    const SmallInt64Vector& kernel_size, const SmallInt64Vector& dilation,
    const SmallInt64Vector& padding, const SmallInt64Vector& stride) {
  mlir::MlirBuilder& builder = input.getBuilder();

  // Input: (N, C, H, W).
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const llvm::ArrayRef<int64_t> input_shape = input_type.getShape();
  ABSL_CHECK_EQ(input_shape.size(), 4)  // CRASH_OK=Validated before lowering.
      << "Input must be 4D. This is a TorchTPU bug.";

  const int64_t batch = input_shape[0];
  const int64_t channels = input_shape[1];
  const int64_t out_h = out_size[0];
  const int64_t out_w = out_size[1];

  const int64_t k_h = kernel_size[0];
  const int64_t k_w = kernel_size[1];
  const int64_t d_h = dilation[0];
  const int64_t d_w = dilation[1];
  const int64_t p_h = padding[0];
  const int64_t p_w = padding[1];
  const int64_t s_h = stride[0];
  const int64_t s_w = stride[1];

  if (batch == 0) {
    return MakeZeroSizedTensor(builder, input_type.getElementType(),
                               {0, channels * k_h * k_w, out_h * out_w});
  }

  // 1. Pad input.
  mlir::MlirOp padding_value =
      MakeScalarConstant(builder, 0.0, input_type.getElementType());
  mlir::MlirOp padded_input = mlir::stablehlo::Pad(
      input, padding_value, llvm::ArrayRef<int64_t>({0, 0, p_h, p_w}),
      llvm::ArrayRef<int64_t>({0, 0, p_h, p_w}),
      llvm::ArrayRef<int64_t>({0, 0, 0, 0}));

  // 2. Extract sliding window slices and reshape each to
  // (N, C, 1, out_h, out_w).
  std::vector<mlir::MlirOp> reshaped_slices;
  reshaped_slices.reserve(k_h * k_w);
  for (int64_t i = 0; i < k_h; ++i) {
    for (int64_t j = 0; j < k_w; ++j) {
      const SmallInt64Vector start_indices = {0, 0, i * d_h, j * d_w};
      const SmallInt64Vector limit_indices = {batch, channels,
                                              i * d_h + (out_h - 1) * s_h + 1,
                                              j * d_w + (out_w - 1) * s_w + 1};
      const SmallInt64Vector strides = {1, 1, s_h, s_w};

      mlir::MlirOp slice = mlir::stablehlo::Slice(padded_input, start_indices,
                                                  limit_indices, strides);
      mlir::MlirOp reshaped_slice = mlir::stablehlo::Reshape(
          slice, llvm::ArrayRef<int64_t>({batch, channels, 1, out_h, out_w}));
      reshaped_slices.push_back(reshaped_slice);
    }
  }

  // 3. Concatenate along the kernel dimension to (N, C, kH * kW, out_h, out_w)
  // and reshape to PyTorch's (N, C * kH * kW, out_h * out_w) layout.
  mlir::MlirOp concat =
      mlir::stablehlo::Concatenate(builder, reshaped_slices, 2);
  return mlir::stablehlo::Reshape(
      concat,
      llvm::ArrayRef<int64_t>({batch, channels * k_h * k_w, out_h * out_w}));
}

struct Im2ColOutputInfo {
  mlir::ElementType out_dtype;
  Dimensions output_dims;
  SmallInt64Vector out_size;
  bool is_3d;
};

absl::StatusOr<Im2ColOutputInfo> ValidateAndGetIm2ColOutputInfo(
    const at::Tensor& self, const at::IntArrayRef kernel_size,
    const at::IntArrayRef dilation, const at::IntArrayRef padding,
    const at::IntArrayRef stride) {
  TT_RETURN_IF_ERROR(GetBaseBuffer(self).status());
  TT_ASSIGN_OR_RETURN(const mlir::ElementType elem_type,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));

  TT_RET_CHECK(kernel_size.size() == 2, error::kInvalidArgument)
      << "expected kernel_size to have 2 dimensions, got "
      << kernel_size.size();
  TT_RET_CHECK(dilation.size() == 2, error::kInvalidArgument)
      << "expected dilation to have 2 dimensions, got " << dilation.size();
  TT_RET_CHECK(padding.size() == 2, error::kInvalidArgument)
      << "expected padding to have 2 dimensions, got " << padding.size();
  TT_RET_CHECK(stride.size() == 2, error::kInvalidArgument)
      << "expected stride to have 2 dimensions, got " << stride.size();

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
  TT_RET_CHECK(d_h > 0 && d_w > 0, error::kInvalidArgument)
      << "expected all dilation elements to be positive, got "
      << ToString(dilation);
  TT_RET_CHECK(p_h >= 0 && p_w >= 0, error::kInvalidArgument)
      << "expected all padding elements to be >= 0, got " << ToString(padding);
  TT_RET_CHECK(s_h > 0 && s_w > 0, error::kInvalidArgument)
      << "expected all stride elements to be positive, got "
      << ToString(stride);

  const int64_t ndim = self.dim();
  TT_RET_CHECK(ndim == 3 || ndim == 4, error::kInvalidArgument)
      << "expected input to have 3 or 4 dimensions, got " << ndim;

  const bool is_3d = ndim == 3;
  const int64_t batch_offset = is_3d ? -1 : 0;
  const int64_t batch = is_3d ? 1 : self.size(0);
  const int64_t channels = self.size(batch_offset + 1);
  const int64_t h = self.size(batch_offset + 2);
  const int64_t w = self.size(batch_offset + 3);

  TT_RET_CHECK(channels > 0 && h > 0 && w > 0, error::kInvalidArgument)
      << "expected non-batch input dimensions to be > 0, got shape "
      << ToString(self.sizes());

  TT_ASSIGN_OR_RETURN(const int64_t out_h,
                      SlidingWindowCount(h, p_h, d_h, k_h, s_h));
  TT_ASSIGN_OR_RETURN(const int64_t out_w,
                      SlidingWindowCount(w, p_w, d_w, k_w, s_w));

  TT_RET_CHECK(out_h > 0 && out_w > 0, error::kInvalidArgument)
      << "expected output shape to be positive, got " << out_h << "x" << out_w;

  TT_ASSIGN_OR_RETURN(const int64_t kernel_prod, SafeMultiply(k_h, k_w));
  TT_ASSIGN_OR_RETURN(const int64_t out_channels,
                      SafeMultiply(channels, kernel_prod));
  TT_ASSIGN_OR_RETURN(const int64_t out_length, SafeMultiply(out_h, out_w));

  Dimensions output_dims;
  if (is_3d) {
    output_dims = {out_channels, out_length};
  } else {
    output_dims = {batch, out_channels, out_length};
  }

  return Im2ColOutputInfo{
      .out_dtype = elem_type,
      .output_dims = std::move(output_dims),
      .out_size = SmallInt64Vector{out_h, out_w},
      .is_3d = is_3d,
  };
}

MlirUnaryOpBuilder MakeIm2ColOpBuilder(SmallInt64Vector out_size,
                                       SmallInt64Vector kernel_size,
                                       SmallInt64Vector dilation,
                                       SmallInt64Vector padding,
                                       SmallInt64Vector stride,
                                       const bool is_3d) {
  return [out_size = std::move(out_size), kernel_size = std::move(kernel_size),
          dilation = std::move(dilation), padding = std::move(padding),
          stride = std::move(stride),
          is_3d](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
    if (is_3d) {
      TT_ASSIGN_OR_RETURN(input, Unsqueeze(input, 0));
    }
    TT_ASSIGN_OR_RETURN(mlir::MlirOp result,
                        BuildIm2ColShlo(input, out_size, kernel_size, dilation,
                                        padding, stride));
    if (is_3d) {
      return Squeeze(result, {0});
    }
    return result;
  };
}

}  // namespace

at::Tensor AtenIm2Col(const at::Tensor& self, const at::IntArrayRef kernel_size,
                      const at::IntArrayRef dilation,
                      const at::IntArrayRef padding,
                      const at::IntArrayRef stride) {
  TT_KERNEL(
      OpName::kIm2Col, param_keys,
      (self, kernel_size, dilation, padding, stride), {
        TT_ASSIGN_OR_THROW(Im2ColOutputInfo info,
                           ValidateAndGetIm2ColOutputInfo(
                               self, kernel_size, dilation, padding, stride));
        MlirUnaryOpBuilder op_builder = MakeIm2ColOpBuilder(
            std::move(info.out_size), CopyIntVector(kernel_size),
            CopyIntVector(dilation), CopyIntVector(padding),
            CopyIntVector(stride), info.is_3d);
        TT_ASSIGN_OR_THROW(
            DeviceBufferRef result,
            DispatchOp<1>(std::move(op_builder), self,
                          {.out_dtype = info.out_dtype,
                           .out_dims = info.output_dims,
                           .op_param_cache_keys = std::move(param_keys)}));
        return MakeTensor(std::move(result));
      });
}

at::Tensor& AtenIm2ColOut(const at::Tensor& self,
                          const at::IntArrayRef kernel_size,
                          const at::IntArrayRef dilation,
                          const at::IntArrayRef padding,
                          const at::IntArrayRef stride, at::Tensor& out) {
  TT_KERNEL(
      OpName::kIm2ColOut, param_keys,
      (self, kernel_size, dilation, padding, stride, out), {
        TT_THROW_IF_ERROR(GetBaseBuffer(self).status());
        TT_THROW_IF_ERROR(GetBaseBuffer(out).status());
        TT_THROW_IF_ERROR(ValidateOutDtype(out, self.scalar_type(),
                                           /*allow_out_dtype_cast=*/false));
        TT_ASSIGN_OR_THROW(Im2ColOutputInfo info,
                           ValidateAndGetIm2ColOutputInfo(
                               self, kernel_size, dilation, padding, stride));
        MlirUnaryOpBuilder op_builder = MakeIm2ColOpBuilder(
            std::move(info.out_size), CopyIntVector(kernel_size),
            CopyIntVector(dilation), CopyIntVector(padding),
            CopyIntVector(stride), info.is_3d);
        TT_THROW_IF_ERROR(
            DispatchOpOut<1>(std::move(op_builder), self, out,
                             {.out_dtype = info.out_dtype,
                              .out_dims = info.output_dims,
                              .op_param_cache_keys = std::move(param_keys)}));
        return out;
      });
}

}  // namespace torch_tpu
