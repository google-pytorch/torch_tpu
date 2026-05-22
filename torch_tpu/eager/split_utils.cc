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

#include "torch_tpu/eager/split_utils.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/materialize_common.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/eager/traversal.h"

namespace torch_tpu {

namespace {

absl::StatusOr<std::vector<ExecutionTask>> ApplySplitPointsUnsorted(
    std::vector<SharedDeviceBufferList> queue,
    const absl::flat_hash_set<const DeviceBufferList*>& split_points,
    const absl::flat_hash_set<const DeviceBufferList*>& required_outputs,
    MaterializationReason reason,
    mlir::MLIRContext* absl_nullable mlir_context) {
  std::vector<ExecutionTask> tasks;
  for (const auto& node : queue) {
    if (!split_points.contains(node.get())) continue;
    MaterializationReason split_reason =
        (required_outputs.contains(node.get())
             ? reason
             : MaterializationReason::kInternalSplitPoint);
    TT_ASSIGN_OR_RETURN(auto new_traversal, Traversal::Create({node}));
    TT_ASSIGN_OR_RETURN(
        auto execution_task,
        ExecutionTask::FromTraversalWithLogging(std::move(new_traversal),
                                                split_reason, mlir_context));
    tasks.push_back(std::move(execution_task));
  }
  return tasks;
}

absl::StatusOr<std::vector<ExecutionTask>> ApplySplitPointsSorted(
    std::vector<SharedDeviceBufferList> queue,
    const absl::flat_hash_set<const DeviceBufferList*>& split_points,
    const absl::flat_hash_set<const DeviceBufferList*>& required_outputs,
    MaterializationReason reason,
    mlir::MLIRContext* absl_nullable mlir_context) {
  std::vector<SharedDeviceBufferList> execution_order;
  std::vector<SharedDeviceBufferList> nodes_to_materialize;
  std::vector<ExecutionTask> tasks;
  for (auto& node : queue) {
    if (!split_points.contains(node.get())) {
      execution_order.push_back(std::move(node));
    } else {
      MaterializationReason split_reason =
          (required_outputs.contains(node.get())
               ? reason
               : MaterializationReason::kInternalSplitPoint);
      execution_order.push_back(node);  // intentional copy
      nodes_to_materialize.push_back(std::move(node));
      TT_ASSIGN_OR_RETURN(auto new_traversal,
                          Traversal::CreateFromExecutionOrder(
                              execution_order, nodes_to_materialize));
      TT_ASSIGN_OR_RETURN(
          auto execution_task,
          ExecutionTask::FromTraversalWithLogging(std::move(new_traversal),
                                                  split_reason, mlir_context));
      tasks.push_back(std::move(execution_task));
      execution_order.clear();
      nodes_to_materialize.clear();
    }
  }
  return tasks;
}

}  // namespace

absl::StatusOr<std::vector<ExecutionTask>> ApplySplitPoints(
    absl_nonnull std::unique_ptr<Traversal> traversal,
    const absl::flat_hash_set<const DeviceBufferList*>& required_outputs,
    const absl::flat_hash_set<const DeviceBufferList*>& split_points,
    bool use_sorted, MaterializationReason reason,
    mlir::MLIRContext* absl_nullable mlir_context) {
  std::vector<SharedDeviceBufferList> queue;
  {
    Traversal::Parts traversal_parts = traversal->IntoParts();
    traversal.reset();
    queue = std::move(traversal_parts.execution_order);
  }
  if (use_sorted) {
    return ApplySplitPointsSorted(std::move(queue), split_points,
                                  required_outputs, reason, mlir_context);
  }
  return ApplySplitPointsUnsorted(std::move(queue), split_points,
                                  required_outputs, reason, mlir_context);
}

}  // namespace torch_tpu
