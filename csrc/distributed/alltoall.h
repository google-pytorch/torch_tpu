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

absl::StatusOr<mlir::MlirOp> BuildDistributedAllToAllBaseShlo(
    mlir::MlirOp input, const DeviceGroupList& device_groups);

// Builds and emits a StableHLO ragged all-to-all custom call for
// `all_to_all_single` with uneven / non-uniform split sizes across ranks.
//
// Uneven all-to-all communicates non-uniform slices of `input` across devices
// in `device_groups`, where each rank sends dynamic slice sizes specified by
// `send_sizes` and receives dynamic slice sizes specified by `recv_sizes`.
//
// Args:
//   input: The local input tensor containing slices to send to peer ranks.
//   output: The output tensor providing shape, rank, and element type for the
//     destination buffer where incoming slices will be written.
//   input_offsets: Starting offsets (in elements along dimension 0) of each
//     outgoing slice in `input`.
//   send_sizes: Number of elements along dimension 0 to send to each peer rank.
//   output_offsets: Starting offsets (in elements along dimension 0) in the
//     destination rank's output buffer where this rank's sent slice should be
//     placed.
//   recv_sizes: Number of elements along dimension 0 to receive from each peer
//     rank.
//   device_groups: List of process group device IDs participating in the
//     collective communication.
//
// Returns:
//   An `mlir::MlirOp` representing the result of the ragged_all_to_all custom
//   call operation, or an error status on failure.
absl::StatusOr<mlir::MlirOp> BuildDistributedAllToAllBaseUnevenSplitsShlo(
    mlir::MlirOp input, mlir::MlirOp output,
    absl::Span<const int32_t> input_offsets,
    absl::Span<const int32_t> send_sizes,
    absl::Span<const int32_t> output_offsets,
    absl::Span<const int32_t> recv_sizes, const DeviceGroupList& device_groups);

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
