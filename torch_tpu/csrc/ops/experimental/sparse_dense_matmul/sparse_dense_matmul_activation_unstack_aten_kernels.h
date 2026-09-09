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

#ifndef TORCH_TPU_CSRC_OPS_EXPERIMENTAL_SPARSE_DENSE_MATMUL_SPARSE_DENSE_MATMUL_ACTIVATION_UNSTACK_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_EXPERIMENTAL_SPARSE_DENSE_MATMUL_SPARSE_DENSE_MATMUL_ACTIVATION_UNSTACK_ATEN_KERNELS_H_

#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"

namespace torch_tpu {

/**
 * @brief Validates the inputs to the sparse-dense matmul activation unstack op.
 *
 * @param stacked_activations 2D float tensor of activations with shape
 * [total_batch_size, padded_embedding_dim].
 * @param per_feature_batch_sizes Array of batch sizes for each feature.
 * @param per_feature_dims Array of embedding dimensions for each feature.
 *
 * @return OK if the inputs are valid, otherwise an error status.
 */
absl::Status ValidateSparseDenseMatmulActivationUnstackInputs(
    const at::Tensor& stacked_activations,
    at::IntArrayRef per_feature_batch_sizes, at::IntArrayRef per_feature_dims);

/**
 * @brief Unstacks and uninterleaves stacked activations across SparseCore
 * cores.
 *
 * @param stacked_activations 2D float tensor of activations with shape
 * [total_batch_size, padded_embedding_dim].
 * @param per_feature_batch_sizes Array of batch sizes for each feature.
 * @param per_feature_dims Array of embedding dimensions for each feature.
 *
 * @return A vector of unstacked 2D float tensors, each with shape
 * [per_feature_batch_sizes[i], per_feature_dims[i]].
 */
std::vector<at::Tensor> AtenSparseDenseMatmulActivationUnstack(
    const at::Tensor& stacked_activations,
    at::IntArrayRef per_feature_batch_sizes, at::IntArrayRef per_feature_dims);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_EXPERIMENTAL_SPARSE_DENSE_MATMUL_SPARSE_DENSE_MATMUL_ACTIVATION_UNSTACK_ATEN_KERNELS_H_
