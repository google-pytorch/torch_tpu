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

#ifndef TORCH_TPU_CSRC_OPS_SPARSE_COO_SPARSE_COO_H_
#define TORCH_TPU_CSRC_OPS_SPARSE_COO_SPARSE_COO_H_

#include <array>
#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

// Lowers sparse-to-dense conversion to StableHLO using ScatterOp with AddOp.
absl::StatusOr<mlir::MlirOp> BuildSparseToDenseShlo(
    mlir::MlirOp indices_op, mlir::MlirOp values_op,
    absl::Span<const int64_t> out_sizes, mlir::ElementType out_dtype);

// Lowers dense + alpha * sparse -> dense to StableHLO using ScatterOp with
// AddOp.
absl::StatusOr<mlir::MlirOp> BuildAddDenseSparseShlo(
    mlir::MlirOp dense_self_op, mlir::MlirOp sparse_indices_op,
    mlir::MlirOp sparse_values_op, std::optional<mlir::MlirOp> alpha_op,
    mlir::ElementType out_dtype);

// Lowers sparse + alpha * sparse -> uncoalesced sparse (indices, values) to
// StableHLO using ConcatenateOp and MultiplyOp.
absl::StatusOr<std::array<mlir::MlirOp, 2>> BuildAddSparseSparseShlo(
    mlir::MlirOp indices1_op, mlir::MlirOp values1_op, mlir::MlirOp indices2_op,
    mlir::MlirOp values2_op, std::optional<mlir::MlirOp> alpha_op,
    mlir::ElementType out_dtype);

// Multiplies sparse values by a 0-D scalar tensor in StableHLO.
absl::StatusOr<mlir::MlirOp> BuildMulSparseScalarShlo(
    mlir::MlirOp values_op, mlir::MlirOp scalar_op,
    mlir::ElementType out_dtype);

// Negates sparse values in StableHLO.
absl::StatusOr<mlir::MlirOp> BuildNegSparseShlo(mlir::MlirOp values_op,
                                                mlir::ElementType out_dtype);

// Coalesces a sparse COO tensor in StableHLO by lexicographically sorting
// coordinate rows + permutation iota, identifying unique coordinate groups,
// scatter-adding duplicate values into compacted positions, compacting unique
// indices to the front, and returning {compacted_indices, compacted_values,
// unique_count}.
absl::StatusOr<std::array<mlir::MlirOp, 3>> BuildCoalesceAndCountShlo(
    mlir::MlirOp indices_op, mlir::MlirOp values_op);

// Slices compacted coalesced indices [k, nnz] and values [nnz, ...] to their
// true unique count output_size in StableHLO.
absl::StatusOr<std::array<mlir::MlirOp, 2>> BuildSliceCoalescedShlo(
    mlir::MlirOp compacted_indices_op, mlir::MlirOp compacted_values_op,
    int64_t output_size);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_SPARSE_COO_SPARSE_COO_H_
