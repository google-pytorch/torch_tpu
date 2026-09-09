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

#ifndef TORCH_TPU_CSRC_OPS_EXPERIMENTAL_SPARSE_DENSE_MATMUL_SPARSE_DENSE_MATMUL_GRADIENT_STACK_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_EXPERIMENTAL_SPARSE_DENSE_MATMUL_SPARSE_DENSE_MATMUL_GRADIENT_STACK_ATEN_KERNELS_H_

#include <cstdint>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"

namespace torch_tpu {

/**
 * @brief Validates the inputs to the sparse-dense matmul gradient stack op.
 *
 * @param unstacked_gradients List of 2D unstacked gradient tensors.
 * @param stacked_batch_size Expected total stacked batch size.
 * @param stacked_feature_dim Expected padded stacked embedding dimension.
 *
 * @return OK if the inputs are valid, otherwise an error status.
 */
absl::Status ValidateSparseDenseMatmulGradientStackInputs(
    at::TensorList unstacked_gradients, int64_t stacked_batch_size,
    int64_t stacked_feature_dim);

/**
 * @brief Stacks and interleaves unstacked gradients across SparseCore cores.
 *
 * @param unstacked_gradients List of 2D unstacked gradient tensors.
 * @param stacked_batch_size Expected total stacked batch size.
 * @param stacked_feature_dim Expected padded stacked embedding dimension.
 *
 * @return A dense 2D float tensor of stacked gradients with shape
 * [stacked_batch_size, stacked_feature_dim].
 */
at::Tensor AtenSparseDenseMatmulGradientStack(
    at::TensorList unstacked_gradients, int64_t stacked_batch_size,
    int64_t stacked_feature_dim);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_EXPERIMENTAL_SPARSE_DENSE_MATMUL_SPARSE_DENSE_MATMUL_GRADIENT_STACK_ATEN_KERNELS_H_
