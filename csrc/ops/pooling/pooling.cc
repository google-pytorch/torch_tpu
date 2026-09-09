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

#include "csrc/ops/pooling/pooling.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

// Adds a batch dimension of size 1 if the input tensor doesn't have it.
// Returns the batch input and the original dimension size.
absl::StatusOr<BatchInput> CreateBatchInput(mlir::MlirOp input,
                                            const int64_t spatial_dim_count) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  auto num_dims = input_type.getRank();

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Rank is already checked by
                 // GetPoolingOutputSize.
      num_dims == spatial_dim_count + 1 || num_dims == spatial_dim_count + 2,
      error::kInvalidArgument)
      << "expected non-empty " << spatial_dim_count + 1 << "D or "
      << spatial_dim_count + 2 << "D (batch mode) tensor for input, got "
      << num_dims << "D tensor";

  // If the input doesn't have a batch dimension, add a size 1 batch dimension
  if (num_dims == spatial_dim_count + 1) {
    Dimensions new_shape = {1};
    new_shape.insert(new_shape.end(), input_type.getShape().begin(),
                     input_type.getShape().end());
    auto batch_input = mlir::stablehlo::Reshape(input, new_shape);
    return BatchInput{batch_input, num_dims};
  }
  return BatchInput{input, num_dims};
}

// Expands the attribute to match the dim size.
absl::StatusOr<Dimensions> ExpandAttribute(const Dimensions& values,
                                           const int64_t dim) {
  if (values.size() == 1 && dim > 1) {
    return Dimensions(dim, values[0]);
  }

  Dimensions expanded_values = Dimensions(values.begin(), values.end());
  ABSL_CHECK_EQ(values.size(), dim)  // CRASH_OK
      << "the attribute was not expanded correctly to match the dim size"
      << ", got expanded values size " << expanded_values.size()
      << " and expected dim size " << dim;
  return expanded_values;
}

// Removes the batch dimension if it doesn't have it at the beginning.
mlir::MlirOp RemoveTrivialBatch(mlir::MlirOp batch_op,
                                int64_t original_dim_size,
                                int64_t spatial_dim_count) {
  if (original_dim_size == spatial_dim_count + 2) {
    return batch_op;
  }

  const mlir::RankedTensorType current_type = GetTensorTypeOrDie(batch_op);
  llvm::SmallVector<int64_t> squeezed_shape;
  squeezed_shape.append(current_type.getShape().begin() + 1,
                        current_type.getShape().end());
  return mlir::stablehlo::Reshape(batch_op, squeezed_shape);
}

// Calculates the adjusted (left_padding, right_padding) pairs given the padding
// (which is symmetric) and ceil mode.
// Additional high padding could be required when ceil mode is set.
// This ensures that the last pooling is fully inside the image.
std::vector<PaddingParams> CeilModePadding(
    const mlir::RankedTensorType& input_shape,  // (N, C, H, W)/(N, C, D, H, W)
    mlir::ArrayRef<int64_t> kernel_size,        // [K_h, K_w]
    mlir::ArrayRef<int64_t> stride,             // [S_h, S_w]
    mlir::ArrayRef<int64_t> padding,            // [P_h, P_w]
    mlir::ArrayRef<int64_t> dilation,           // [D_h, D_w]
    bool ceil_mode) {
  std::vector<PaddingParams> ceil_mode_padding;

  for (size_t i = 0; i < padding.size(); ++i) {
    const int64_t P = padding[i];
    const int64_t K = kernel_size[i];
    const int64_t S = stride[i];
    const int64_t D = dilation[i];

    // Spatial dimensions start from index 2
    const int64_t input_size = input_shape.getDimSize(2 + i);
    const int64_t left_padding = P;
    int64_t right_padding = P;

    if (!ceil_mode) {
      ceil_mode_padding.push_back({left_padding, right_padding});
      continue;
    }

    // Effective kernel_size of the current spatial dimension
    const int64_t K_eff = (K - 1) * D + 1;
    // Remainder for the output size formula with symmetric padding
    const int64_t output_size_num =
        input_size + left_padding + right_padding - K_eff;
    int64_t output_size_rem = output_size_num % S;
    // Ensure output_size remainder is in [0, S-1] range
    if (output_size_rem < 0) {
      output_size_rem += S;
    }

    // Extra padding might be needed for ceil_mode
    if (output_size_rem != 0) {
      const int64_t extra_padding = S - output_size_rem;
      const int64_t target_output_size =
          (output_size_num + extra_padding) / S + 1;
      const int64_t last_window_start_index = (target_output_size - 1) * S;
      if (last_window_start_index < left_padding + input_size) {
        // Add extra_padding to the right side if the constraint is met
        right_padding += extra_padding;
      }
    }
    ceil_mode_padding.push_back({left_padding, right_padding});
  }

  return ceil_mode_padding;
}

// Creates the padding config to be used to pad the input tensor and iota tensor
// consistently, which is also required by StableHLO's ReduceWindowOp.
ReduceWindowAttributes GetReduceWindowAttributes(
    mlir::MlirBuilder& builder, Dimensions kernel_size_attr,
    Dimensions stride_attr, Dimensions dilation_attr,
    std::vector<PaddingParams> padding_pairs, const int64_t spatial_dim_count,
    const int64_t total_num_dims) {
  Dimensions kernel_size(2, 1);
  kernel_size.insert(kernel_size.end(), kernel_size_attr.begin(),
                     kernel_size_attr.end());

  Dimensions stride;
  stride.resize(2, 1);
  stride.insert(stride.end(), stride_attr.begin(), stride_attr.end());

  Dimensions dilation(2, 1);
  dilation.insert(dilation.end(), dilation_attr.begin(), dilation_attr.end());

  Dimensions base_dilations(total_num_dims, 1);

  // Create a flat list of padding values for StableHLO's ReduceWindowOp
  llvm::SmallVector<int64_t> flat_padding_list;
  flat_padding_list.reserve(total_num_dims * 2);

  for (int i = 0; i < total_num_dims - spatial_dim_count; ++i) {
    flat_padding_list.push_back(0);
    flat_padding_list.push_back(0);
  }
  for (const auto& dim_padding : padding_pairs) {
    flat_padding_list.push_back(dim_padding.left);   // low
    flat_padding_list.push_back(dim_padding.right);  // high
  }

  auto reduce_padding_attr = mlir::DenseIntElementsAttr::get(
      mlir::makeTensorType(builder.getContext(), {total_num_dims, 2},
                           builder.getOpBuilder().getI64Type()),
      flat_padding_list);

  return ReduceWindowAttributes{
      .window_dimensions =
          mlir::DenseI64ArrayAttr::get(&builder.getContext(), kernel_size),
      .window_strides =
          mlir::DenseI64ArrayAttr::get(&builder.getContext(), stride),
      .base_dilations =
          mlir::DenseI64ArrayAttr::get(&builder.getContext(), base_dilations),
      .window_dilations =
          mlir::DenseI64ArrayAttr::get(&builder.getContext(), dilation),
      .padding = reduce_padding_attr};
}

namespace {

int64_t GetSingleDim(const int64_t in, const int64_t i,
                     at::IntArrayRef kernel_size, at::IntArrayRef stride,
                     at::IntArrayRef padding, at::IntArrayRef dilation,
                     const bool ceil_mode) {
  const int64_t k = kernel_size.size() == 1 ? kernel_size[0] : kernel_size[i];
  const int64_t s =
      stride.empty() ? k : (stride.size() == 1 ? stride[0] : stride[i]);
  const int64_t p = padding.size() == 1 ? padding[0] : padding[i];
  const int64_t d = dilation.size() == 1 ? dilation[0] : dilation[i];
  const int64_t numerator = in + 2 * p - d * (k - 1) - 1;

  if (!ceil_mode) {
    return numerator / s + 1;
  }

  const int64_t out = (numerator + s - 1) / s + 1;
  if ((out - 1) * s >= in + p) {
    return out - 1;
  }
  return out;
}

}  // namespace

absl::StatusOr<Dimensions> GetPoolingOutputSize(
    at::IntArrayRef input_size, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    const bool ceil_mode, const int64_t spatial_dim_count) {
  const int64_t dims_num = input_size.size();
  TT_RET_CHECK(
      dims_num == spatial_dim_count + 1 || dims_num == spatial_dim_count + 2,
      error::kInvalidArgument)
      << "expected non-empty " << spatial_dim_count + 1 << "D or "
      << spatial_dim_count + 2 << "D (batch mode) tensor for input, got "
      << dims_num << "D tensor";

  // If (N, C, ...), spatial starts at 2. If (C, ...), starts at 1.
  const int64_t spatial_start_index =
      (dims_num == spatial_dim_count + 2) ? 2 : 1;

  Dimensions output_size(dims_num);
  std::copy(input_size.begin(), input_size.begin() + spatial_start_index,
            output_size.begin());

  // Calculates Spatial Dims (D, H, W).
  for (int i = 0; i < spatial_dim_count; ++i) {
    const int64_t in = input_size[spatial_start_index + i];
    const int64_t out =
        GetSingleDim(in, i, kernel_size, stride, padding, dilation, ceil_mode);
    output_size[spatial_start_index + i] = std::max<int64_t>(out, 1);
  }
  return output_size;
}

}  // namespace torch_tpu
