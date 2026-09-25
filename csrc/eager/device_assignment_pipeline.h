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

#ifndef TORCH_TPU_CSRC_EAGER_DEVICE_ASSIGNMENT_PIPELINE_H_
#define TORCH_TPU_CSRC_EAGER_DEVICE_ASSIGNMENT_PIPELINE_H_

#include <cstdint>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/ops/op_names.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

namespace torch_tpu {

struct OpDeviceAssignment {
  OpName op_name;
  absl::Span<const int64_t> device_assignment;
};

// Returns true if `d_sub` is a contiguous tile of `d_full` aligned to a
// multiple of `d_sub.size()`.
bool IsCompatibleDeviceAssignmentTile(absl::Span<const int64_t> d_sub,
                                      absl::Span<const int64_t> d_full);

// Resolves a single target device assignment across `ops` using a 4-stage
// pipeline:
//   1. Merge compatible Pallas custom kernel device assignments and reject
//      conflicting Pallas meshes.
//   2. Promote to larger compatible StableHLO collective device assignments.
//   3. Validate that all remaining smaller/orthogonal StableHLO collective
//      device assignments are valid subgroups of the resolved mesh.
//   4. Tail-pad the resolved device assignment to the global PjRt device count
//      if smaller than world size.
absl::StatusOr<std::vector<int64_t>>  // INT_VEC_OK
ResolveDeviceAssignment(absl::Span<const OpDeviceAssignment> ops);

// Resolves a single target device assignment across all deferred operations in
// `execution_order` by delegating to `ResolveDeviceAssignment`.
absl::StatusOr<std::vector<int64_t>>  // INT_VEC_OK
ResolveDeviceAssignmentForTraversal(
    absl::Span<const SharedDeviceBufferList> execution_order);

// Rewrites rank-local collective attributes (`replica_groups`,
// `source_target_pairs`, and CustomCall `backend_config` replica groups) from
// `op_da` into partition indices of `resolved_da`, tiles non-wrapping cosets
// across the full resolved mesh, and wraps `PartitionIdOp` results with
// `rem(partition_id, sub_size)` when `resolved_da.size() > op_da.size()`.
void RewriteAndTileOperationForSubgroup(mlir::Operation* absl_nonnull root_op,
                                        absl::Span<const int64_t> op_da,
                                        absl::Span<const int64_t> resolved_da);

// Applies subgroup tiling and partition-index replica rewriting to all MLIR
// operations emitted for `deferred_op` (starting at `first_new_op` in the
// current block, plus any newly added functions in `new_func_ops`).
void ApplySubgroupTilingOrReplicaRewriting(
    mlir::Operation* absl_nullable first_new_op,
    absl::Span<const mlir::func::FuncOp> new_func_ops,
    const DeferredOp& deferred_op, absl::Span<const int64_t> resolved_da);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_DEVICE_ASSIGNMENT_PIPELINE_H_
