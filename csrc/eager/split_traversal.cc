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

#include "csrc/eager/split_traversal.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/types/span.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/traversal.h"
#include "tsl/profiler/lib/traceme.h"

namespace torch_tpu {

namespace {

// Ensures that if a tensor is used after a materialization point, it is also
// materialized before the materialization point.
//
// For example, a function like:
// ```
//   x = foo()
//   y = bar()
//   z = baz(x)
//   print(y.item())
// ```
// will have:
//   execution_order: [foo, bar, baz]
//   required_outputs: {bar}
//   materialization_points: {bar}
// This function will add a materialization point for `x = foo()`, and will
// remove any materialization points after the last required output.
void SplitAllMaterializationPoints(
    absl::Span<const SharedDeviceBufferList> execution_order,
    const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        required_outputs,
    absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        materialization_points) {
  // The index of the node we're currently planning to materialize.
  // Will be nullopt for all nodes after the highest-index required output.
  std::optional<uint64_t> materializing_index = std::nullopt;

  for (auto node_it = execution_order.rbegin();
       node_it != execution_order.rend(); ++node_it) {
    const auto& node = *node_it;

    if (required_outputs.contains(node.get())) {
      // Found a required output, ensure that materializing_index has a value
      // and stop dropping materialization points.
      materializing_index = node->creation_index();
    } else if (!materializing_index.has_value()) {
      // Remove any materialization points after the last required output.
      materialization_points.erase(node.get());
    } else if (materialization_points.contains(node.get())) {
      // Retain any materialization points before the last required output.
      materializing_index = node->creation_index();
    } else if (node->last_child_index() > materializing_index.value()) {
      // This node has an edge from node->creation_index() to
      // node->last_child_index(), which crosses the current
      // materializing_index.
      // We need to materialize this node so that the materialization order of
      // {node->creation_index(), materializing_index, node->last_child_index()}
      // is maintained.
      materialization_points.insert(node.get());
      materializing_index = node->creation_index();
    }
  }
}

// Creates a new set of traversals, one for each split point in the original
// traversal.
std::vector<absl_nonnull std::unique_ptr<Traversal>> ApplySplitPoints(
    absl_nonnull std::unique_ptr<Traversal> traversal,
    const absl::flat_hash_set<const DeviceBufferList*>& split_points) {
  std::vector<SharedDeviceBufferList> queue;
  {
    auto parts = traversal->IntoParts();
    queue = std::move(parts.execution_order);
  }
  std::vector<SharedDeviceBufferList> execution_order;
  std::vector<SharedDeviceBufferList> nodes_to_materialize;
  std::vector<absl_nonnull std::unique_ptr<Traversal>> traversals;
  for (auto& node : queue) {
    if (!split_points.contains(node.get())) {
      execution_order.push_back(std::move(node));
    } else {
      execution_order.push_back(node);  // intentional copy
      nodes_to_materialize.push_back(std::move(node));
      auto new_traversal_or = Traversal::CreateFromExecutionOrder(
          execution_order, nodes_to_materialize);
      ABSL_CHECK_OK(new_traversal_or);  // CRASH_OK=subsequence of original
      traversals.push_back(std::move(*new_traversal_or));
      execution_order.clear();
      nodes_to_materialize.clear();
    }
  }
  return traversals;
}

}  // namespace

std::vector<absl_nonnull std::unique_ptr<Traversal>> SplitTraversal(
    absl_nonnull std::unique_ptr<Traversal> traversal,
    absl::flat_hash_set<const DeviceBufferList*>& required_outputs) {
  ABSL_VLOG(1) << ">>> SplitTraversal " << traversal->execution_order().size();
  tsl::profiler::TraceMe t("SplitTraversal");

  required_outputs.clear();
  for (auto& output : traversal->outputs()) {
    required_outputs.insert(output.device_buffer_list().get());
  }
  absl::flat_hash_set<const DeviceBufferList*> split_points = required_outputs;
  SplitAllMaterializationPoints(traversal->execution_order(), required_outputs,
                                split_points);

  ABSL_VLOG(1) << "Found " << required_outputs.size()
               << " required outputs and "
               << split_points.size() - required_outputs.size()
               << " additional split points.";

  if (split_points.size() <= required_outputs.size()) {
    // Keep the full traversal when there are no internal split points.
    std::vector<absl_nonnull std::unique_ptr<Traversal>> traversals;
    traversals.push_back(std::move(traversal));
    return traversals;
  }

  return ApplySplitPoints(std::move(traversal), split_points);
}

}  // namespace torch_tpu
