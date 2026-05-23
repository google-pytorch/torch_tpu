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

#include "torch_tpu/ops/im2col/im2col.h"

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Value.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

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

}  // namespace torch_tpu
