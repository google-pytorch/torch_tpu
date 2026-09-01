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

#ifndef TORCH_TPU_OPS_EXPERIMENTAL_SPARSE_GATHER_SPARSE_GATHER_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_EXPERIMENTAL_SPARSE_GATHER_SPARSE_GATHER_ATEN_KERNELS_H_

#include <cstdint>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"

namespace torch_tpu {

/**
 * @brief Validates the inputs to the sparse gather op.
 *
 * @param row_pointers Integer 1D tensor containing row end pointers in CSR
 * format.
 * @param indices Integer 1D tensor of column/embedding indices in CSR format
 * (padded with INT_MAX to multiples of PAD_WIDTH = 8).
 * @param operand The dense 2D tensor containing the embedding table of shape
 * [V, D].
 * @param max_non_zeroes_per_row Hardware constraint: Maximum non-zero elements
 * allowed per row in the CSR format.
 *
 * @return OK if the inputs are valid, otherwise an error status.
 */
absl::Status ValidateSparseGatherInputs(const at::Tensor& row_pointers,
                                        const at::Tensor& indices,
                                        const at::Tensor& operand,
                                        int64_t max_non_zeroes_per_row);

/**
 * @brief Performs sparse gather from an embedding table on TPU SparseCore.
 *
 * @param row_pointers Integer 1D tensor containing row end pointers in CSR
 * format.
 * @param indices Integer 1D tensor of column/embedding indices in CSR format
 * (padded with INT_MAX to multiples of PAD_WIDTH = 8).
 * @param operand The dense 2D tensor containing the embedding table of shape
 * [V, D].
 * @param max_non_zeroes_per_row Hardware constraint: Maximum non-zero elements
 * allowed per row in the CSR format.
 *
 * @return A dense tensor of gathered embeddings with shape [N, D],
 * where N = row_pointers.size(0) * max_non_zeroes_per_row == indices.size(0).
 */
at::Tensor AtenSparseGather(const at::Tensor& row_pointers,
                            const at::Tensor& indices,
                            const at::Tensor& operand,
                            int64_t max_non_zeroes_per_row);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_EXPERIMENTAL_SPARSE_GATHER_SPARSE_GATHER_ATEN_KERNELS_H_
