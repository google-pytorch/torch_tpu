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
#include "absl/flags/flag.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/dynamic_op_split_heuristic.h"
#include "torch_tpu/eager/fanout_heuristic.h"
#include "torch_tpu/eager/forced_split_heuristic.h"
#include "torch_tpu/eager/reexecution_heuristic.h"
#include "torch_tpu/eager/safe_materialization_rule.h"
#include "torch_tpu/eager/stale_heuristic.h"
#include "torch_tpu/eager/traversal.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/traceme.h"

ABSL_FLAG(bool, torch_tpu_internal_fanout_heuristic, true,
          "Use a materialization heuristic that looks at node fanout.");
ABSL_FLAG(bool, torch_tpu_internal_reexecution_heuristic, true,
          "Use a materialization heuristic that materializes on reexecution.");
ABSL_FLAG(bool, torch_tpu_internal_stale_heuristic, true,
          "Use a materialization heuristic that materializes around the stale "
          "regions of a graph.");

namespace torch_tpu {

namespace {

// Config struct indicating which materialization heuristic flags are enabled.
struct CustomHeuristics {
  void Initialize() {
    reexecution = absl::GetFlag(FLAGS_torch_tpu_internal_reexecution_heuristic);
    fanout = absl::GetFlag(FLAGS_torch_tpu_internal_fanout_heuristic);
    stale = absl::GetFlag(FLAGS_torch_tpu_internal_stale_heuristic);

    initialized = true;
  }

  // Whether the heuristics have been initialized.
  bool initialized = false;

  bool reexecution = false;
  bool fanout = false;
  bool stale = false;
};

// Applies all enabled materialization heuristics on a given `traversal` and
// return the set of nodes in `traversal` that at least one heuristic decides
// to materialize.
[[nodiscard]] absl::flat_hash_set<const DeviceBufferList* absl_nonnull>
ApplyAllMaterializationHeuristicsOn(
    const Traversal& traversal,
    const absl::flat_hash_set<const DeviceBufferList*>& required_outputs,
    MaterializationMode mode) {
  absl::flat_hash_set<const DeviceBufferList*> nodes_to_materialize =
      required_outputs;
  if (mode == MaterializationMode::kSplitSafe) {
    auto safe_rule = SafeMaterializationRule(required_outputs);
    safe_rule(traversal, nodes_to_materialize);
  } else if (mode == MaterializationMode::kSplitCustom) {
    tsl::profiler::TraceMe t("LocalHeuristics");
    static CustomHeuristics enabled_heuristics;
    if (!enabled_heuristics.initialized) {
      enabled_heuristics.Initialize();
    }

    for (const auto& node : traversal.execution_order()) {
      if (enabled_heuristics.reexecution) {
        ReexecutionHeuristic(*node, nodes_to_materialize);
      }
      ForcedSplitHeuristic(*node, nodes_to_materialize);
      DynamicOpSplitHeuristic(*node, nodes_to_materialize);
      if (enabled_heuristics.fanout) {
        FanoutHeuristic(*node, nodes_to_materialize);
      }
      if (enabled_heuristics.stale) {
        StaleHeuristic(*node, nodes_to_materialize);
      }
    }
  }
  return nodes_to_materialize;
}

}  // namespace

absl::StatusOr<std::vector<Traversal>> SplitTraversal(
    Traversal traversal,
    const absl::flat_hash_set<const DeviceBufferList*>& required_outputs,
    MaterializationMode mode) {
  ABSL_VLOG(1) << ">>> SplitTraversal " << traversal.execution_order().size();
  tsl::profiler::TraceMe t("SplitTraversal");
  std::vector<Traversal> traversals;

  if (mode == MaterializationMode::kFuseAll) {
    traversals.push_back(std::move(traversal));
    return traversals;
  }

  // Reorder the nodes in the traversal to prefer materializing the
  // earliest-dispatched output DeferredOps first. This results in a
  // topological sort which is expected to be better, assuming the user's code
  // is organized for eager execution.
  traversal.SortByCreationOrder();

  // Here we collect a set of materialization nodes internal to the input
  // `traversal`.
  absl::flat_hash_set<const DeviceBufferList*> materialization_nodes_set =
      ApplyAllMaterializationHeuristicsOn(traversal, required_outputs, mode);

  if (materialization_nodes_set.size() <= required_outputs.size()) {
    // Shortcut in the simple case where there is no internal materialization
    // node.
    traversals.push_back(std::move(traversal));
    return traversals;
  }

  ABSL_VLOG(1) << "Found " << required_outputs.size()
               << " required outputs and "
               << materialization_nodes_set.size() - required_outputs.size()
               << " internal materialization nodes.";

  // In the following code, we split the original traversal into multiple
  // traversals, based on the identified materialization points.

  // Split traversal into one node per materialization point, maintaining the
  // creation index order established above.
  std::vector<DeviceBufferRef> traversal_outputs;
  for (const auto& node : traversal.execution_order()) {
    if (!materialization_nodes_set.contains(node.get())) continue;
    traversal_outputs.clear();
    for (auto i = 0; i < node->size(); ++i) {
      TT_ASSIGN_OR_RETURN(auto buffer_ref, DeviceBufferRef::Create(node, i));
      traversal_outputs.push_back(std::move(buffer_ref));
    }
    TT_ASSIGN_OR_RETURN(auto new_traversal,
                        Traversal::Create(std::move(traversal_outputs),
                                          materialization_nodes_set));
    traversals.push_back(std::move(new_traversal));
  }
  return traversals;
}

}  // namespace torch_tpu
