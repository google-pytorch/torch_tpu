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
#include "absl/status/status.h"
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
  // The devices of every group of the layout of the op's process group (see
  // DeviceAssignment::global); empty if unknown. Used when the program's ops
  // run on different device sets.
  absl::Span<const int64_t> global_device_assignment = {};
};

struct PallasDeviceAssignment {
  absl::Span<const int64_t> device_assignment;
};

// Resolves a single target device assignment across `ops` and `pallas_ops`.
// If `pallas_ops` is non-empty, its device assignment serves as the
// authoritative anchor. If the ops run on different device sets, resolves to
// the smallest set closed under the groups of their layouts (their global
// device assignments) when those agree on that set. Otherwise,
// differing device assignments emit an explicit diagnostic warning and return
// an error (or throw if Pallas DAs conflict).
absl::StatusOr<std::vector<int64_t>>  // INT_VEC_OK
ResolveDeviceAssignment(
    absl::Span<const OpDeviceAssignment> ops,
    absl::Span<const PallasDeviceAssignment> pallas_ops = {});

// Resolves a single target device assignment across all deferred operations in
// `execution_order` by delegating to `ResolveDeviceAssignment`.
absl::StatusOr<std::vector<int64_t>>  // INT_VEC_OK
ResolveDeviceAssignmentForTraversal(
    absl::Span<const SharedDeviceBufferList> execution_order,
    absl::Span<const int64_t> pallas_device_assignment = {});

// Rewrites rank-local collective attributes (`replica_groups`,
// `source_target_pairs`, and CustomCall `backend_config` replica groups) from
// `op_da` into partition indices of `resolved_da` using 1-to-1 index
// permutation. If `op_da` is not a permutation of `resolved_da`, instead tiles
// the attributes over the groups of the op's layout, `global_da` (the devices
// of every group, group by group), inside `resolved_da`, which together must
// be a permutation of `resolved_da`.
absl::Status RewriteAndTileOperationForSubgroup(
    mlir::Operation* absl_nonnull root_op, absl::Span<const int64_t> op_da,
    absl::Span<const int64_t> resolved_da,
    absl::Span<const int64_t> global_da = {});

// Applies partition-index replica rewriting to all MLIR operations emitted for
// `deferred_op` (starting at `first_new_op` in the current block, plus any
// newly added functions in `new_func_ops`). Returns an error if any rewrite
// fails.
absl::Status ApplySubgroupTilingOrReplicaRewriting(
    mlir::Operation* absl_nullable first_new_op,
    absl::Span<const mlir::func::FuncOp> new_func_ops,
    const DeferredOp& deferred_op, absl::Span<const int64_t> resolved_da);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_DEVICE_ASSIGNMENT_PIPELINE_H_
