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

#ifndef TORCH_TPU_CSRC_OPS_EXPERIMENTAL_SPARSE_MAP_ROW_SPARSE_MAP_ROW_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_EXPERIMENTAL_SPARSE_MAP_ROW_SPARSE_MAP_ROW_ATEN_KERNELS_H_

#include <cstdint>
#include <string_view>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "c10/util/ArrayRef.h"  // IWYU pragma: keep for at::IntArrayRef

namespace torch_tpu {

// Validates input tensors, parameters, and device capabilities for
// `sparse_map_row`.
//
// Checks that:
// - `csr_tensors` and `csr_lengths` are non-empty.
// - The sum of `csr_lengths` equals the total number of `csr_tensors`.
// - `result_row_pointer_op_index` is in bounds [0, len(csr_lengths) - 1].
// - `max_non_zeroes_per_row` is strictly positive.
// - `body_mlir` module string and `dummy_outputs` are non-empty.
// - The active TPU device supports SparseCore (unless tracing for Meta).
absl::Status ValidateSparseMapRowInputs(
    const std::vector<at::Tensor>& csr_tensors, at::IntArrayRef csr_lengths,
    const std::vector<at::Tensor>& dense_args, std::string_view body_mlir,
    const std::vector<at::Tensor>& dummy_outputs,
    int64_t max_non_zeroes_per_row, int64_t result_row_pointer_op_index);

// ATen implementation of `torch.ops.tpu._sparse_map_row`.
//
// Maps a row-wise user computation across one or more sparse matrices in CSR
// format on TPU SparseCore. Lowers to a `stablehlo.custom_call` targeting
// "SparseMapRow" with the body computation embedded as a called computation.
std::vector<at::Tensor> AtenSparseMapRow(
    const std::vector<at::Tensor>& csr_tensors, at::IntArrayRef csr_lengths,
    const std::vector<at::Tensor>& dense_args, std::string_view body_mlir,
    const std::vector<at::Tensor>& dummy_outputs,
    int64_t max_non_zeroes_per_row, int64_t pad_value, int64_t max_pad_per_row,
    double wait_threshold, int64_t result_row_pointer_op_index);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_EXPERIMENTAL_SPARSE_MAP_ROW_SPARSE_MAP_ROW_ATEN_KERNELS_H_
