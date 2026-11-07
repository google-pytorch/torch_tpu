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

#ifndef TORCH_TPU_CSRC_OPS_POOLING_POOLING_H_
#define TORCH_TPU_CSRC_OPS_POOLING_POOLING_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/string_view.h"
#include "csrc/common/dimension_types.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

enum class PoolingDirection : bool {
  kForward = false,
  kBackward = true,
};

// Returns the 1D einsum contraction equation for pooling along spatial `dim`
// (where `dim` is in [2, spatial_dim_count + 2)) of a batch-normalized
// [N, C, ...] tensor with a 2D [out_size, in_size] weight matrix.
c10::string_view GetPoolingEinsumEquation(int64_t spatial_dim_count,
                                          int64_t dim,
                                          PoolingDirection direction);

// Builds a 2D pooling weight matrix W of shape [out_size, in_size] in
// `compute_dtype` from column vectors `starts` ([out_size, 1], i32),
// `ends` ([out_size, 1], i32), and `window_weights` ([out_size, 1],
// `compute_dtype`), where W[o, i] = window_weights[o, 0] if
// starts[o, 0] <= i < ends[o, 0], else 0.0.
mlir::MlirOp BuildPooling1dWeightMatrix(mlir::MlirBuilder& builder,
                                        int64_t in_size, int64_t out_size,
                                        mlir::MlirOp starts, mlir::MlirOp ends,
                                        mlir::MlirOp window_weights,
                                        mlir::Type compute_dtype);

// Returns the initial reduction value attribute for max pooling: -inf for
// floating-point types and the minimum finite value for integer types.
mlir::Attribute GetMaxPoolInitValueAttr(mlir::Type element_type,
                                        mlir::OpBuilder& builder);

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

// Defines the asymmetric padding (start and end) applied to a single spatial
// dimension during a pooling operation, typically adjusting for ceil_mode.
struct PaddingParams {
  // The amount of padding added to the start (left, top, front) of the
  // dimension.
  int64_t left = 0;
  // The amount of padding added to the end (right, bottom, back) of the
  // dimension.
  int64_t right = 0;
};

// Calculates the adjusted (left_padding, right_padding) pairs given the padding
// (which is symmetric) and ceil mode.
std::vector<PaddingParams> CeilModePadding(
    const mlir::RankedTensorType& input_shape,  // (N, C, H, W)
    mlir::ArrayRef<int64_t> kernel_size,        // [K_h, K_w]
    mlir::ArrayRef<int64_t> stride,             // [S_h, S_w]
    mlir::ArrayRef<int64_t> padding,            // [P_h, P_w]
    mlir::ArrayRef<int64_t> dilation,           // [D_h, D_w]
    bool ceil_mode);

// Calculates the output dimensions for N-dimensional pooling operations (max
// pool, avg pool, etc.).
//
// Computes spatial dimensions (D, H, W) based on kernel_size, stride, padding,
// dilation, and ceil_mode. Handles input shapes formatted as (N, C, ...) or (C,
// ...). Returns an error status if input_size rank is insufficient for
// spatial_dim_count.
absl::StatusOr<Dimensions> GetPoolingOutputSize(
    at::IntArrayRef input_size, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool ceil_mode, int64_t spatial_dim_count);

// Creates the padding config to be used to pad the input tensor and iota tensor
// consistently, which is also required by StableHLO's ReduceWindowOp.
ReduceWindowAttributes GetReduceWindowAttributes(
    mlir::MlirBuilder& builder, Dimensions kernel_size_attr,
    Dimensions stride_attr, Dimensions dilation_attr,
    std::vector<PaddingParams> padding_pairs, int64_t spatial_dim_count,
    int64_t total_num_dims);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_POOLING_POOLING_H_
