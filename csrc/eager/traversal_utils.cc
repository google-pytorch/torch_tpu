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

#include "csrc/eager/traversal_utils.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/env_vars.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/split_traversal.h"
#include "csrc/eager/traversal.h"
#include "csrc/ops/op_names.h"
#include "tsl/profiler/lib/traceme.h"

namespace torch_tpu {
namespace {

// Lightweight matcher for target DeviceBufferList pointers during
// materialization. Avoids heap allocation and SwissTable hashing for the common
// single-node and small-span cases.
class NodeMatcher {
 public:
  explicit NodeMatcher(absl::Span<const SharedDeviceBufferList> nodes) {
    if (nodes.size() == 1) {
      single_node_ = nodes[0].get();
    } else if (nodes.size() <= 8) {
      for (const auto& node : nodes) {
        small_nodes_.push_back(node.get());
      }
    } else {
      large_nodes_.reserve(nodes.size());
      for (const auto& node : nodes) {
        large_nodes_.insert(node.get());
      }
    }
  }

  bool contains(const DeviceBufferList* absl_nonnull node) const {
    if (single_node_ != nullptr) {
      return node == single_node_;
    }
    if (!small_nodes_.empty()) {
      return std::find(small_nodes_.begin(), small_nodes_.end(), node) !=
             small_nodes_.end();
    }
    return !large_nodes_.empty() && large_nodes_.contains(node);
  }

 private:
  const DeviceBufferList* absl_nullable single_node_ = nullptr;
  absl::InlinedVector<const DeviceBufferList* absl_nonnull, 8> small_nodes_;
  absl::flat_hash_set<const DeviceBufferList* absl_nonnull> large_nodes_;
};

// Single-op fast path (common in Eager Default / DeferNever mode): when a
// single deferred op with already-materialized inputs is materialized, avoid
// constructing DefinedNodeMap hash tables and intermediate working vectors.
// Returns true if a traversal was created and added to the traversals list.
bool PrepareSingleNodeTraversal(
    SharedDeviceBufferList& device_buffer_list,
    absl::Span<const SharedDeviceBufferList> nodes_to_materialize,
    std::vector<absl_nonnull std::unique_ptr<Traversal>>& traversals) {
  const auto deferred_op = device_buffer_list->deferred_op();
  if (deferred_op == nullptr) return false;

  const bool has_deferred_input = std::any_of(
      deferred_op->inputs().begin(), deferred_op->inputs().end(),
      [](const DeviceBufferRef& input) { return input.is_deferred(); });
  const bool has_extra_constant =
      std::any_of(nodes_to_materialize.begin(), nodes_to_materialize.end(),
                  [&device_buffer_list](const SharedDeviceBufferList& node) {
                    return node.get() != device_buffer_list.get() &&
                           node->is_constant() && node->is_deferred();
                  });
  if (has_deferred_input || has_extra_constant) return false;

  std::vector<SharedDeviceBufferList> single_node = {
      std::move(device_buffer_list)};
  auto traversal_or =
      Traversal::CreateFromExecutionOrder(single_node, single_node);
  ABSL_CHECK_OK(traversal_or);  // CRASH_OK=internal logic error
  auto traversal = std::move(*traversal_or);
  traversals.push_back(std::move(traversal));
  return true;
}

// The usage of a node within an execution region.
enum class OpUsage {
  // This node is an output of the materialization.
  kOutput,
  // This node is an input to an output.
  kUsed,
  // This node is unused, and may need to be added as an output to ensure
  // it is executed.
  kUnused,
};

// Returns the OpUsage of the given device buffer list, switching between
// kOutput (if it's a required output) or kUnused (if it's not).
OpUsage GetOpUsage(const DeferredOp& deferred_op,
                   const SharedDeviceBufferList& device_buffer_list,
                   const NodeMatcher& nodes_to_materialize_set) {
  // Explicitly materialized nodes are always outputs, even if they normally
  // wouldn't be (e.g. constants)
  if (nodes_to_materialize_set.contains(device_buffer_list.get())) {
    ABSL_VLOG(3)
        << "[ProcessDeferredOpEvent] Adding live or explicitly materialized "
           "buffer "
        << device_buffer_list.get() << " as output";
    return OpUsage::kOutput;
  }

  if (device_buffer_list->has_dynamic_dimensions()) {
    bool has_dynamic_dimensions = false;
    for (int i = 0; i < device_buffer_list->size(); ++i) {
      if (!device_buffer_list->dynamic_dimensions(i).empty()) {
        has_dynamic_dimensions = true;
        break;
      }
    }
    if (has_dynamic_dimensions) {
      // Nodes with dynamic dimensions must be materialized to resolve them
      // to static shapes.
      ABSL_VLOG(3)
          << "[ProcessDeferredOpEvent] Adding dynamic-dimension buffer "
          << device_buffer_list.get() << " as output";
      return OpUsage::kOutput;
    }
  }

  // Non-output nodes need to be used by at least one output to ensure they get
  // executed.
  ABSL_VLOG(3) << "[ProcessDeferredOpEvent] Adding non-output buffer "
               << device_buffer_list.get() << " to execution order";
  return OpUsage::kUnused;
}

using DefinedNodeMap = absl::flat_hash_map<const DeviceBufferList*, OpUsage>;

// Helper function for PrepareMaterializationTraversals.
// Takes an execution_order, a map defining the usage of each node, and returns
// a traversal (unless the execution order and output nodes are empty).
void FinishTraversal(
    std::vector<SharedDeviceBufferList>& execution_order,
    DefinedNodeMap& defined_node_map,
    std::vector<SharedDeviceBufferList>& output_nodes,
    std::vector<absl_nonnull std::unique_ptr<Traversal>>& traversals) {
  if (execution_order.empty()) {
    ABSL_CHECK(defined_node_map.empty())  // CRASH_OK
        << "defined node map is not empty when execution order is empty. These "
           "should always have the same size";
    return;
  }

  // Build the output nodes vector, maintaining the execution order.
  output_nodes.clear();
  for (const auto& execution_node : execution_order) {
    auto it = defined_node_map.find(execution_node.get());
    ABSL_CHECK(it != defined_node_map.end())  // CRASH_OK
        << "node in execution order was not found in defined node map";
    switch (it->second) {
      case OpUsage::kOutput:
        output_nodes.push_back(execution_node);
        break;
      case OpUsage::kUnused:
        // Unused nodes are added as outputs to ensure they get executed.
        ABSL_VLOG(3) << "[FinishTraversal] Promoting unused buffer "
                     << execution_node.get() << " as output";
        output_nodes.push_back(execution_node);
        break;
      case OpUsage::kUsed:
        break;
    }
  }

  // Execution order was asserted to be non-empty above.
  // The last node in the execution order cannot be kUsed as there are no later
  // nodes to use it. Thus, it must either be kOutput or kUnused, which then
  // gets promoted to kOutput, and there is at least one output.
  ABSL_CHECK(!output_nodes.empty());  // CRASH_OK=internal logic error

  auto traversal_or =
      Traversal::CreateFromExecutionOrder(execution_order, output_nodes);
  ABSL_CHECK_OK(traversal_or);  // CRASH_OK=internal logic error
  execution_order.clear();
  defined_node_map.clear();
  output_nodes.clear();
  traversals.push_back(std::move(*traversal_or));
}

// Helper function for PrepareMaterializationTraversals.
// Pushes the device buffer list to the execution order (unless it is empty),
// and updates the defined node map.
void ProcessDeferredOpEvent(
    SharedDeviceBufferList&& device_buffer_list, const DeferredOp& deferred_op,
    const NodeMatcher& nodes_to_materialize_set,
    std::vector<SharedDeviceBufferList>& execution_order,
    DefinedNodeMap& defined_node_map) {
  // Mark all inputs as used, and insert any newly-used empty tensors.
  for (const auto& input : deferred_op.inputs()) {
    if (!input.is_deferred()) continue;

    const DeviceBufferList* input_device_buffer_list =
        input.device_buffer_list().get();

    if (auto input_def_it = defined_node_map.find(input_device_buffer_list);
        input_def_it != defined_node_map.end()) {
      // Update the usage of the input tensor from unused -> used, but leave
      // outputs as outputs.
      switch (input_def_it->second) {
        case OpUsage::kUnused:
          input_def_it->second = OpUsage::kUsed;
          break;
        case OpUsage::kUsed:
        case OpUsage::kOutput:
          break;
      }
    } else if (input.is_constant() &&
               defined_node_map
                   .try_emplace(input_device_buffer_list, OpUsage::kUsed)
                   .second) {
      // The first time a constant tensor is read by a later op, we insert
      // it into the execution order, but not as an output as we don't want to
      // materialize it unless the user explicitly asks for it.
      ABSL_VLOG(3) << "[ProcessDeferredOpEvent] Constant buffer "
                   << input_device_buffer_list << " is read by "
                   << device_buffer_list.get() << "("
                   << ToString(deferred_op.op_name())
                   << ").\nInserting into execution order.";
      execution_order.push_back(input.device_buffer_list());
    }
  }

  defined_node_map[device_buffer_list.get()] =
      GetOpUsage(deferred_op, device_buffer_list, nodes_to_materialize_set);

  execution_order.push_back(std::move(device_buffer_list));
}

// We never materialize a constant (or empty) tensor unless it is explicitly
// requested by the user. If that does happen, then we append these constant
// tensors to the last Traversal to make sure they have defined buffers.
// This is necessary for downstream uses that require physical data buffers,
// such as torch.compile invocations.
void AppendMustMaterializeConstants(
    absl::Span<const SharedDeviceBufferList> nodes_to_materialize,
    DefinedNodeMap& defined_node_map,
    std::vector<SharedDeviceBufferList>& execution_order) {
  for (const auto& node : nodes_to_materialize) {
    if (node->is_constant() && node->is_deferred()) {
      bool inserted =
          defined_node_map.insert_or_assign(node.get(), OpUsage::kOutput)
              .second;
      if (inserted) {
        ABSL_VLOG(3) << "[PrepareTraversals] constant buffer " << node.get()
                     << " is an explicit output. Appending to final traversal.";
        execution_order.push_back(node);
      }
    }
  }
}

std::vector<absl_nonnull std::unique_ptr<Traversal>> PrepareInitialTraversals(
    std::vector<SharedDeviceBufferList> must_execute,
    absl::Span<const SharedDeviceBufferList> nodes_to_materialize) {
  NodeMatcher nodes_to_materialize_set(nodes_to_materialize);
  std::vector<absl_nonnull std::unique_ptr<Traversal>> traversals;

  if (must_execute.empty() && nodes_to_materialize.empty()) {
    return traversals;
  }

  if (must_execute.size() == 1 &&
      PrepareSingleNodeTraversal(must_execute[0], nodes_to_materialize,
                                 traversals)) {
    return traversals;
  }

  // Partition the deferred ops queue into separate traversals with these rules:
  //   - If an op is SplitBefore, it must be first in its execution order.
  //   - If an op is SplitAfter, it must be last in its execution order, and
  //     must be an output.
  //   - If a node has live c10::DataPtrs, it must be an output.
  //   - If a node is in nodes_to_materialize, it must be an output.
  //   - Every non-constant op in execution_order must be executed by at least
  //     one output.
  //   - Constant and empty ops are added immediately before the first read, or
  //     at the end if they are explicitly materialized but not read.
  // Reuse working memory for efficiency.
  std::vector<SharedDeviceBufferList> execution_order;
  std::vector<SharedDeviceBufferList> output_nodes;
  DefinedNodeMap defined_node_map;

  for (auto& device_buffer_list : must_execute) {
    // Filter to only non-expired, deferred nodes.
    const auto deferred_op = device_buffer_list->deferred_op();
    if (!deferred_op) continue;

    const auto split_mode = deferred_op->split_mode();

    if (IsSplitBefore(split_mode)) {
      FinishTraversal(execution_order, defined_node_map, output_nodes,
                      traversals);
    }

    ProcessDeferredOpEvent(std::move(device_buffer_list), *deferred_op,
                           nodes_to_materialize_set, execution_order,
                           defined_node_map);

    if (IsSplitAfter(split_mode)) {
      FinishTraversal(execution_order, defined_node_map, output_nodes,
                      traversals);
    }
  }

  AppendMustMaterializeConstants(nodes_to_materialize, defined_node_map,
                                 execution_order);
  FinishTraversal(execution_order, defined_node_map, output_nodes, traversals);
  ABSL_VLOG(1) << "[PrepareTraversals] Created " << traversals.size()
               << " traversals to materialize " << nodes_to_materialize.size()
               << " nodes.";
  return traversals;
}

void ApplyGraphSplitMode(
    std::vector<absl_nonnull std::unique_ptr<Traversal>>& traversals,
    const GraphSplitMode graph_split_mode) {
  if (graph_split_mode == GraphSplitMode::kFullGraph) {
    return;
  }
  tsl::profiler::TraceMe t("SplitTraversal");
  std::vector<absl_nonnull std::unique_ptr<Traversal>> post_split_traversals;
  absl::flat_hash_set<const DeviceBufferList*> scratch_set;
  for (auto& pre_split_traversal : traversals) {
    auto split_traversals =
        SplitTraversal(std::move(pre_split_traversal), scratch_set);
    post_split_traversals.insert(
        post_split_traversals.end(),
        std::make_move_iterator(split_traversals.begin()),
        std::make_move_iterator(split_traversals.end()));
  }
  std::swap(traversals, post_split_traversals);
}

void ApplyCorePinning(
    std::vector<absl_nonnull std::unique_ptr<Traversal>>& traversals,
    const CorePinningMode core_pinning_mode,
    const SkinnySparseCoreOffload skinny_sparse_core_offload_mode) {
  if (core_pinning_mode != CorePinningMode::kUnpinned) {
    for (auto& traversal : traversals) {
      traversal->SetCorePinningMode(core_pinning_mode);
    }
    return;
  }
  if (skinny_sparse_core_offload_mode == SkinnySparseCoreOffload::kDisabled) {
    return;
  }

  // We have CorePinningMode::kUnpinned && SkinnySparseCoreOffload::kEnabled.
  // Find all skinny collective traversals and pin them to the SparseCore.
  for (auto& traversal : traversals) {
    if (traversal->execution_order().size() == 1) {
      auto deferred_op = traversal->execution_order()[0]->deferred_op();
      if (deferred_op != nullptr && IsDistributedOp(deferred_op->op_name())) {
        traversal->SetCorePinningMode(CorePinningMode::kSparseCore);
      }
    }
  }
}

}  // namespace

std::vector<absl_nonnull std::unique_ptr<Traversal>> PrepareTraversals(
    std::vector<SharedDeviceBufferList> must_execute,
    absl::Span<const SharedDeviceBufferList> nodes_to_materialize,
    const PrepareTraversalsOptions& options) {
  auto traversals =
      PrepareInitialTraversals(std::move(must_execute), nodes_to_materialize);
  ApplyGraphSplitMode(traversals, options.graph_split_mode);
  ApplyCorePinning(traversals, options.core_pinning_mode,
                   options.skinny_sparse_core_offload_mode);
  return traversals;
}

SkinnySparseCoreOffload GetSkinnySparseCoreOffloadMode() {
  if (torch_tpu::GetBooleanEnvOnce<kTorchTpuInternalSkinnyScOffloadEnvVar>()
          .value_or(false)) {
    return SkinnySparseCoreOffload::kEnabled;
  }
  return SkinnySparseCoreOffload::kDisabled;
}

}  // namespace torch_tpu
