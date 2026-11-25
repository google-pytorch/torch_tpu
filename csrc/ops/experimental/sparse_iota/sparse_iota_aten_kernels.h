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

#ifndef TORCH_TPU_CSRC_OPS_EXPERIMENTAL_SPARSE_IOTA_SPARSE_IOTA_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_EXPERIMENTAL_SPARSE_IOTA_SPARSE_IOTA_ATEN_KERNELS_H_

#include <cstdint>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"

namespace torch_tpu {

/**
 * @brief Validates the inputs to the sparse iota op.
 *
 * @param row_pointers Integer 1D tensor containing row end pointers in CSR
 * format.
 * @param max_non_zeroes Total maximum number of non-zero elements (output
 * length).
 * @param max_non_zeroes_per_row Hardware constraint: Maximum non-zero elements
 * allowed per row.
 *
 * @return OK if the inputs are valid, otherwise an error status.
 */
absl::Status ValidateSparseIotaInputs(const at::Tensor& row_pointers,
                                      int64_t max_non_zeroes,
                                      int64_t max_non_zeroes_per_row);

/**
 * @brief Applies SparseIota to a CSR matrix on TPU SparseCore.
 *
 * @param row_pointers Integer 1D tensor containing row end pointers in CSR
 * format.
 * @param max_non_zeroes Total maximum number of non-zero elements in output
 * buffer.
 * @param max_non_zeroes_per_row Maximum non-zero elements allowed per row.
 *
 * @return A 1D tensor of shape [max_non_zeroes] containing the sparse iota.
 */
[[nodiscard]] at::Tensor AtenSparseIota(const at::Tensor& row_pointers,
                                        int64_t max_non_zeroes,
                                        int64_t max_non_zeroes_per_row);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_EXPERIMENTAL_SPARSE_IOTA_SPARSE_IOTA_ATEN_KERNELS_H_
