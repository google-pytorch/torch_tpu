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

#ifndef TORCH_TPU_CSRC_DISTRIBUTED_ALLTOALL_H_
#define TORCH_TPU_CSRC_DISTRIBUTED_ALLTOALL_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/distributed/types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

// 1D ragged all-to-all factoring size: one `(8, 128)` TPU tile (1024 elements).
inline constexpr int64_t k1DRaggedAllToAllTileElements = 1024;

absl::StatusOr<mlir::MlirOp> BuildDistributedAllToAllBaseShlo(
    mlir::MlirOp input, const DeviceGroupList& device_groups);

// Returns the greatest common divisor of all rank-local dimension-0 sizes and
// offsets that take part in a 1D uneven all-to-all on this rank: the input
// and output buffer sizes, the local input offsets, the send sizes, the local
// output placement offsets, and the receive sizes. Zero-valued entries are
// neutral.
int64_t Compute1DRaggedAllToAllLocalGcd(
    int64_t input_size, int64_t output_size,
    absl::Span<const int32_t> input_offsets,
    absl::Span<const int32_t> send_sizes,
    absl::Span<const int32_t> local_output_offsets,
    absl::Span<const int32_t> recv_sizes);

// Selects the 1D factoring size (`k1DRaggedAllToAllTileElements = 1024` or 1)
// for a group-wide `group_gcd` so that a flat `[N]` operand can be viewed as
// `[N / 1024, 8, 128]` before `stablehlo.ragged_all_to_all`.
int64_t Select1DRaggedAllToAllFactor(int64_t group_gcd, int element_bitwidth);

// Builds StableHLO ops for `all_to_all_single` with uneven split sizes.
//
// Emits `input_offsets`, `send_sizes`, `output_offsets`, and `recv_sizes` as
// compile-time `stablehlo.constant` vectors in global HBM. Keeping metadata
// constant keeps the `ragged_all_to_all` temporary buffer as the sole
// `HLO temp` allocation at byte offset 0 on every rank when buffer capacities
// differ across ranks. When `flat_factor > 1` on 1D operands, reshapes `[N]`
// to `[N / flat_factor, 8, 128]` and scales metadata by `flat_factor` so the
// XLA:TPU backend avoids rank-1 lane padding.
absl::StatusOr<mlir::MlirOp> BuildDistributedAllToAllBaseUnevenSplitsShlo(
    mlir::MlirOp input, mlir::MlirOp output,
    absl::Span<const int32_t> input_offsets,
    absl::Span<const int32_t> send_sizes,
    absl::Span<const int32_t> output_offsets,
    absl::Span<const int32_t> recv_sizes, int64_t flat_factor,
    const DeviceGroupList& device_groups);

absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> BuildDistributedAllToAllShlo(
    absl::Span<mlir::MlirOp> inputs, const DeviceGroupList& device_groups);

// Concatenates non-empty input tensors along dimension 0, or emits a zero-sized
// tensor of shape `total_input_shape` if `inputs` is empty.
absl::StatusOr<mlir::MlirOp> ConcatenateTensorsAlongDim0(
    absl::Span<mlir::MlirOp> inputs, mlir::ElementType output_dtype,
    absl::Span<const int64_t> total_input_shape, mlir::MlirBuilder& builder);

// Slices a concatenated tensor along dimension 0 into individual tensors
// matching `split_sizes`.
mlir::SmallVector<mlir::MlirOp> SliceTensorsAlongDim0(
    mlir::MlirOp tensor, absl::Span<const int32_t> split_sizes);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_DISTRIBUTED_ALLTOALL_H_
