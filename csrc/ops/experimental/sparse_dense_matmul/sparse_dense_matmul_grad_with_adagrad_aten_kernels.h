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

#ifndef TORCH_TPU_OPS_EXPERIMENTAL_SPARSE_DENSE_MATMUL_SPARSE_DENSE_MATMUL_GRAD_WITH_ADAGRAD_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_EXPERIMENTAL_SPARSE_DENSE_MATMUL_SPARSE_DENSE_MATMUL_GRAD_WITH_ADAGRAD_ATEN_KERNELS_H_

#include <string_view>
#include <tuple>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

/**
 * @brief Computes gradients and updates the embedding table using the Adagrad
 * optimizer.
 *
 * This is a fused backward pass operator that propagates gradients from
 * activations back to the embedding table and applies the Adagrad update rule.
 * It ensures mathematical exactness by correctly aggregating gradients for
 * duplicate embedding IDs within the same batch before updating the parameters.
 *
 * @details
 * The Adagrad algorithm adapts the learning rate for each parameter by keeping
 * a running sum of historical squared gradients (\p accumulator). This fused
 * kernel avoids the overhead of materializing large sparse gradient tensors.
 *
 * The sparse inputs (\p row_pointers, \p embedding_ids, \p sample_ids, \p
 * gains) describe the forward pass connectivity in "CSR-wrapped COO" format.
 *
 *
 * @param row_pointers Integer 1D tensor containing indices that represent row
 * pointers in CSR format.
 * @param embedding_ids Integer 1D tensor of the local embedding IDs within each
 * SparseCore to be looked up (local column indices of the embedding matrix).
 * @param sample_ids Integer 1D tensor of sample/batch indices associated with
 * each lookup (row indices of the sparse matrix).
 * @param gains Float 1D tensor of weights/coefficients applied to each
 * looked-up embedding vector.
 * @param embedding_table The dense 2D tensor containing the actual embedding
 * vectors of shape [V, D].
 * @param accumulator The dense 2D tensor of shape [V, D] storing the sum of
 * historical squared gradients.
 * @param activations_grad Gradients of the loss with respect to the output
 * activations, shape [B, D] where B is the device_batch_size.
 * @param learning_rate Learning rate scalar or tensor (applied to all updates).
 * @param epsilon Small constant used for numerical stability in division.
 * @param device_batch_size The logical batch size processed by this device.
 * @param max_ids_per_partition Hardware constraint: Maximum total IDs allowed
 * per processing partition.
 * @param max_unique_ids_per_partition Hardware constraint: Maximum unique IDs
 * allowed per processing partition, influencing deduplication.
 *
 * @return A tuple containing:
 *         - The updated \p embedding_table tensor.
 *         - The updated \p accumulator tensor.
 */

std::tuple<at::Tensor, at::Tensor> AtenSparseDenseMatmulGradWithAdagrad(
    const at::Tensor& row_pointers, const at::Tensor& embedding_ids,
    const at::Tensor& sample_ids, const at::Tensor& gains,
    const at::Tensor& embedding_table, const at::Tensor& accumulator,
    const at::Tensor& activations_grad, const at::Tensor& learning_rate,
    double epsilon, int64_t device_batch_size, int64_t max_ids_per_partition,
    int64_t max_unique_ids_per_partition, std::string_view computation_name);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_EXPERIMENTAL_SPARSE_DENSE_MATMUL_SPARSE_DENSE_MATMUL_GRAD_WITH_ADAGRAD_ATEN_KERNELS_H_
