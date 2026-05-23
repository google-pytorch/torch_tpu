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

#include "torch_tpu/eager/safe_materialization_rule.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/traversal.h"
#include "tsl/profiler/lib/traceme.h"

namespace torch_tpu {

namespace {

// Returns the set of nodes that should be split into separate traversals,
// as indicated by OpSplitMode on the DeferredOp.
absl::flat_hash_set<const DeviceBufferList* absl_nonnull> GetSplitPoints(
    absl::Span<const SharedDeviceBufferList> execution_order) {
  absl::flat_hash_set<const DeviceBufferList* absl_nonnull> split_points;
  for (const auto& node : execution_order) {
    const auto deferred_op = node->deferred_op();
    if (!deferred_op) continue;

    if (deferred_op->split_mode() == OpSplitMode::kSplitAfter ||
        deferred_op->split_mode() == OpSplitMode::kSplitBoth) {
      split_points.insert(node.get());
    }

    if (deferred_op->split_mode() == OpSplitMode::kSplitBefore ||
        deferred_op->split_mode() == OpSplitMode::kSplitBoth) {
      for (const auto& input : deferred_op->inputs()) {
        split_points.insert(input.device_buffer_list().get());
      }
    }
  }
  return split_points;
}

// Returns the set of nodes that must be materialized.
absl::flat_hash_set<const DeviceBufferList* absl_nonnull>
GetMaterializationPoints(
    absl::Span<const SharedDeviceBufferList> execution_order,
    const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        required_outputs,
    const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        split_points) {
  absl::flat_hash_set<const DeviceBufferList* absl_nonnull>
      materialization_points;

  // We insert a "sync point" after every materialization. Each sync point will
  // ensure that all earlier nodes are either materialized or dropped.
  uint64_t synced_to = std::numeric_limits<uint64_t>::min();

  // After the last required output, we stop adding sync points, but we still
  // check for dependencies across the already-added sync points.
  size_t required_outputs_found = 0;

  absl::flat_hash_set<const DeviceBufferList* absl_nonnull>
      pending_materialization_points;

  for (const auto& node : execution_order) {
    const auto deferred_op = node->deferred_op();
    if (!deferred_op) continue;

    // If there is a dependency edge across a sync point, then any
    // nodes before the split point must be materialized.
    for (const auto& input : deferred_op->inputs()) {
      if (input.state() == DeviceBufferRefState::kDeferred &&
          input.device_buffer_list()->creation_index() < synced_to) {
        materialization_points.insert(input.device_buffer_list().get());
      }
    }

    bool sync = false;
    if (required_outputs.contains(node.get())) {
      // Materialize all required outputs.
      pending_materialization_points.insert(node.get());
      if (++required_outputs_found >= required_outputs.size()) {
        // Sync after the last required output.
        sync = true;
      }
    }

    // After the last required output, we still check for dependencies across
    // the already-added sync points (above), but we don't add any more
    // materialization or sync points.
    if (required_outputs_found < required_outputs.size()) {
      if (split_points.contains(node.get())) {
        // Materialize and sync after each split point.
        pending_materialization_points.insert(node.get());
        sync = true;
      } else if (!node->is_stale()) {
        // Materialize any live nodes, but don't necessarily sync.
        pending_materialization_points.insert(node.get());
      } else {
        // Materialize any nodes with dynamic dimensions, sync not required.
        for (int i = 0; i < node->size(); ++i) {
          if (!node->dynamic_dimensions(i).empty()) {
            pending_materialization_points.insert(node.get());
            break;
          }
        }
      }
    }

    if (sync) {
      // Materialize all pending nodes, and update the recorded sync point.
      for (const auto* pending_node : pending_materialization_points) {
        materialization_points.insert(pending_node);
      }
      pending_materialization_points.clear();
      synced_to = node->creation_index();
    }
  }
  return materialization_points;
}

// Insert a (virtual) sync point after every materialization point.
// This forces every output graph to have a single output node.
void SplitAllMaterializationPoints(
    absl::Span<const SharedDeviceBufferList> execution_order,
    absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        materialization_points) {
  absl::flat_hash_set<const DeviceBufferList*> live_edges;
  for (auto node_it = execution_order.rbegin();
       node_it != execution_order.rend(); ++node_it) {
    const auto& node = *node_it;

    if (materialization_points.contains(node.get())) {
      // Insert a sync point after this materialization point to make it a
      // split point.
      // This means that any edges that cross this point need to materialize the
      // earlier node.
      for (const auto* live_edge : live_edges) {
        materialization_points.insert(live_edge);
      }
      live_edges.clear();
    } else {
      // This edge doesn't need to be materialized.
      live_edges.erase(node.get());
    }

    // Insert any new live edges to deferred ops.
    if (const auto deferred_op = node->deferred_op()) {
      for (const auto& input : deferred_op->inputs()) {
        if (input.state() == DeviceBufferRefState::kDeferred) {
          live_edges.insert(input.device_buffer_list().get());
        }
      }
    }
  }
}

}  // namespace

absl::flat_hash_set<const DeviceBufferList* absl_nonnull>
EnforceOrderedMaterialization(
    const Traversal& traversal,
    const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        required_outputs) {
  tsl::profiler::TraceMe t("EnforceOrderedMaterialization");
  const absl::flat_hash_set<const DeviceBufferList* absl_nonnull> split_points =
      GetSplitPoints(traversal.execution_order());
  absl::flat_hash_set<const DeviceBufferList* absl_nonnull>
      materialization_points = GetMaterializationPoints(
          traversal.execution_order(), required_outputs, split_points);
  SplitAllMaterializationPoints(traversal.execution_order(),
                                materialization_points);
  return materialization_points;
}

}  // namespace torch_tpu
