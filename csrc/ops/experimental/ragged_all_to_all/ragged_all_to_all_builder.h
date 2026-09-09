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

#ifndef TORCH_TPU_CSRC_OPS_EXPERIMENTAL_RAGGED_ALL_TO_ALL_RAGGED_ALL_TO_ALL_BUILDER_H_
#define TORCH_TPU_CSRC_OPS_EXPERIMENTAL_RAGGED_ALL_TO_ALL_RAGGED_ALL_TO_ALL_BUILDER_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

// Builds and emits a StableHLO custom call for the TPU `ragged_all_to_all`
// collective operation.
//
// `ragged_all_to_all` performs an all-to-all scatter-gather across devices
// where each rank can send and receive dynamic/uneven slice sizes.
//
// Args:
//   operand: The local input tensor containing slices to be sent to peer ranks.
//   output: The output tensor providing shape, rank, and element type for the
//     destination buffer where incoming slices will be written.
//   input_offsets: 1D integer tensor specifying the starting offset (in
//     elements along dimension 0) of each outgoing slice in `operand`.
//   send_sizes: 1D integer tensor specifying the number of elements along
//     dimension 0 to send to each peer rank.
//   output_offsets: 1D integer tensor specifying the starting offset (in
//     elements along dimension 0) in `output` where slices received from each
//     peer rank should be placed.
//   recv_sizes: 1D integer tensor specifying the number of elements along
//     dimension 0 to receive from each peer rank.
//   replica_groups_attr: DenseIntElementsAttr defining the process group mesh /
//     replica subgroups participating in the collective communication.
//   channel_id: Communication channel ID for the collective custom call.
//
// Returns:
//   An `mlir::MlirOp` representing the result of the ragged_all_to_all custom
//   call operation, or an error status on failure.
absl::StatusOr<mlir::MlirOp> BuildRaggedAllToAllShlo(
    mlir::MlirOp operand, mlir::MlirOp output, mlir::MlirOp input_offsets,
    mlir::MlirOp send_sizes, mlir::MlirOp output_offsets,
    mlir::MlirOp recv_sizes, mlir::DenseIntElementsAttr replica_groups_attr,
    // Currently, this is always set to 1.
    int64_t channel_id = 1);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_EXPERIMENTAL_RAGGED_ALL_TO_ALL_RAGGED_ALL_TO_ALL_BUILDER_H_
