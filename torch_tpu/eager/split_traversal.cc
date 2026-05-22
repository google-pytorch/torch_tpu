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

#include "torch_tpu/eager/split_traversal.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/materialize_common.h"
#include "torch_tpu/eager/safe_materialization_rule.h"
#include "torch_tpu/eager/split_utils.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/eager/traversal.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/traceme.h"

namespace torch_tpu {

absl::StatusOr<std::vector<ExecutionTask>> SplitTraversal(
    absl_nonnull std::unique_ptr<Traversal> traversal,
    const absl::flat_hash_set<const DeviceBufferList*>& required_outputs,
    MaterializationReason reason,
    mlir::MLIRContext* absl_nullable mlir_context) {
  ABSL_VLOG(1) << ">>> SplitTraversal " << traversal->execution_order().size();
  tsl::profiler::TraceMe t("SplitTraversal");

  // Reorder the nodes in the traversal to prefer materializing the
  // earliest-dispatched output DeferredOps first. This results in a
  // topological sort which is expected to be better, assuming the user's code
  // is organized for eager execution.
  traversal->SortByCreationOrder();

  // Here we collect a set of nodes that should be split into separate
  // traversals.
  absl::flat_hash_set<const DeviceBufferList*> split_points =
      EnforceOrderedMaterialization(*traversal, required_outputs);

  ABSL_VLOG(1) << "Found " << required_outputs.size()
               << " required outputs and "
               << split_points.size() - required_outputs.size()
               << " additional split points.";

  if (split_points.size() <= required_outputs.size()) {
    // Keep the full traversal when there are no internal split points.
    TT_ASSIGN_OR_RETURN(auto execution_task,
                        ExecutionTask::FromTraversalWithLogging(
                            std::move(traversal), reason, mlir_context));
    std::vector<ExecutionTask> tasks;
    tasks.push_back(std::move(execution_task));
    return tasks;
  }

  return ApplySplitPoints(std::move(traversal), required_outputs, split_points,
                          /*use_sorted=*/true, reason, mlir_context);
}

}  // namespace torch_tpu
