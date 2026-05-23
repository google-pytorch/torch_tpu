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

#ifndef TORCH_TPU_OPS_POOLING_POOLING_H_
#define TORCH_TPU_OPS_POOLING_POOLING_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/dimension_types.h"

namespace torch_tpu {

struct BatchInput {
  mlir::MlirOp batch_input;
  int64_t original_dim_size;
};

struct ReduceWindowAttributes {
  mlir::DenseI64ArrayAttr window_dimensions;
  mlir::DenseI64ArrayAttr window_strides;
  mlir::DenseI64ArrayAttr base_dilations;
  mlir::DenseI64ArrayAttr window_dilations;
  mlir::DenseIntElementsAttr padding;
};

// Adds a batch dimension of size 1 if the input tensor doesn't have it.
absl::StatusOr<BatchInput> CreateBatchInput(mlir::MlirOp input,
                                            int64_t spatial_dim_count);

// Expands the attribute to match the dim size.
absl::StatusOr<Dimensions> ExpandAttribute(const Dimensions& values,
                                           int64_t dim);

// Removes the batch dimension if it doesn't have it at the beginning.
mlir::MlirOp RemoveTrivialBatch(mlir::MlirOp batch_op,
                                int64_t original_dim_size,
                                int64_t spatial_dim_count);

// Calculates the adjusted (left_padding, right_padding) pairs given the padding
// (which is symmetric) and ceil mode.
std::vector<std::pair<int64_t, int64_t>> CeilModePadding(
    const mlir::RankedTensorType& input_shape,  // (N, C, H, W)
    mlir::ArrayRef<int64_t> kernel_size,        // [K_h, K_w]
    mlir::ArrayRef<int64_t> stride,             // [S_h, S_w]
    mlir::ArrayRef<int64_t> padding,            // [P_h, P_w]
    mlir::ArrayRef<int64_t> dilation,           // [D_h, D_w]
    bool ceil_mode);

// Creates the padding config to be used to pad the input tensor and iota tensor
// consistently, which is also required by StableHLO's ReduceWindowOp.
ReduceWindowAttributes GetReduceWindowAttributes(
    mlir::MlirBuilder& builder, Dimensions kernel_size_attr,
    Dimensions stride_attr, Dimensions dilation_attr,
    std::vector<std::pair<int64_t, int64_t>> padding_pairs,
    int64_t spatial_dim_count, int64_t total_num_dims);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_POOLING_POOLING_H_
