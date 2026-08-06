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

#include "torch_tpu/eager/events_queue.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "c10/core/Stream.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/eager/current_stream.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/traversal.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "xla/future.h"

namespace torch_tpu {

namespace {

// Creates a new deferred DeviceBufferList which represents a data dependency
// but no actual computation.
//
// This is used to enforce device-side execution timing across event snapshots.
//
// The created op can take any number of inputs (which may be zero), ensuring
// that any executable containing them will not begin executing until the prior
// execution has completed. The returned value will be zero-sized; awaiting its
// materialization will enforce timing without actual execution or memory use.
//
// Note however that any input will still be kept alive by this op, which may
// delay memory freeing if the awaited input could otherwise have been freed.
// NOLINTNEXTLINE:add usage in future CL
SharedDeviceBufferList CreateNoOpDependency(
    absl::Span<const DeviceBufferRef> wait_for = {}) {
  std::vector<Shape> output_shapes = {Shape({0}, mlir::ElementType::UI8)};

  auto op_name = OpName::kTorchTpuInternalDataDependency;
  ScopedPythonContextCapturer capturer(op_name);
  auto op_builder = [](mlir::MlirBuilder& builder,
                       absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    // Intentionally do not check the number of inputs and discard them.
    auto ranked_tensor_type =
        mlir::makeTensorType(builder.getContext(), {0}, mlir::ElementType::UI8);
    auto dense_elements_attr =
        mlir::DenseElementsAttr::getFromRawBuffer(ranked_tensor_type, {});
    return DynamicMlirOpResults{
        mlir::stablehlo::Constant(builder, dense_elements_attr)};
  };
  auto refs_or = DeviceBufferList::CreateDeferred(
      op_name, std::move(op_builder), /*inputs=*/{}, OpParamCacheKeys::Empty(),
      std::move(output_shapes));
  ABSL_CHECK_OK(refs_or);  // CRASH_OK
  return refs_or->at(0).device_buffer_list();
}

// A singleton class that records events related to the creation and destruction
// of c10::DataPtrs referencing DeviceBufferRefs.s
class EventsQueue {
 public:
  // Returns the singleton instance of the EventsQueue.
  static EventsQueue& GetInstance() {
    static absl::NoDestructor<EventsQueue> instance;
    return *instance;
  }

  // Records on the events queue that a new c10::DataPtr referencing the given
  // DeviceBufferRef has been created.
  void RecordNewDataPtrCreated(const DeviceBufferRef& device_buffer_ref) {
    // Placeholders and empty tensors are never inserted into the map.
    if (device_buffer_ref.is_placeholder() || device_buffer_ref.is_empty()) {
      return;
    }
    absl::MutexLock lock(data_ptr_mu_);
    if (!device_buffer_ref.is_materialized()) {
      // Insert or increment the count for the DeviceBufferList.
      live_nodes_[device_buffer_ref.device_buffer_list()]++;
    } else {
      // Once the ref has a ready PjRtBuffer, we can stop tracking it.
      live_nodes_.erase(device_buffer_ref.device_buffer_list());
    }
  }

  // Records on the events queue that a c10::DataPtr referencing the given
  // DeviceBufferRef has been destroyed.
  void RecordDataPtrDestroyed(const DeviceBufferRef& device_buffer_ref) {
    // Placeholders and empty tensors are never inserted into the map.
    if (device_buffer_ref.is_placeholder() || device_buffer_ref.is_empty()) {
      return;
    }
    absl::MutexLock lock(data_ptr_mu_);
    // If the ref is already removed from the map, we do nothing. This can
    // happen if the ref became ready after insertion, or if the queue was
    // cleared.
    auto it = live_nodes_.find(device_buffer_ref.device_buffer_list());
    if (it == live_nodes_.end()) {
      return;
    }
    // Once the ref is materialized, or the count drops to zero, we can remove
    // it from the map.
    if (device_buffer_ref.is_materialized() || --it->second <= 0) {
      live_nodes_.erase(it);
    }
  }

  // Returns a vector of all the DeviceBufferLists that are currently referenced
  // by at least one c10::DataPtr, and are not in a final "ready" state.
  std::vector<SharedDeviceBufferList> GetAllLiveUnsyncedDataPtrs() {
    absl::MutexLock lock(data_ptr_mu_);
    std::vector<SharedDeviceBufferList> result;
    result.reserve(live_nodes_.size());
    // Can't clear the map while also iterating over it.
    std::vector<const DeviceBufferList*> to_remove;
    for (const auto& [node, _] : live_nodes_) {
      if (node->is_materialized()) {
        to_remove.push_back(node.get());
      } else {
        result.push_back(node);
      }
    }
    for (const auto* device_buffer_list : to_remove) {
      live_nodes_.erase(device_buffer_list);
    }
    return result;
  }

  // Clears all tracked DeviceBufferLists from the events queue.
  void Clear() {
    {
      absl::MutexLock lock(data_ptr_mu_);
      live_nodes_.clear();
    }
    // Explicitly do NOT clear the deferred ops queue.
    // Deferred ops may have side effects; clearing them could result in
    // deadlocks or errors later on.
    // If deferred ops are side-effect free and all Tensors using them are
    // dropped, they'll be cleaned up as dead code by the weak_ptr mechanism.
  }

  // If a deferred op has no side effects, then as soon as all of its DataPtrs
  // go out of scope, it is dead code and can be removed. But if it does have
  // side effects, it must be kept alive until it is executed, even if it is
  // dead code.
  struct DeferredOpEvent {
    static std::optional<DeferredOpEvent> FromDeferredOp(
        const SharedDeviceBufferList& device_buffer_list) {
      const auto deferred_op = device_buffer_list->deferred_op();
      if (!deferred_op || deferred_op->depends_on_placeholder()) {
        // TODO: better identify compiled mode tracing.
        return std::nullopt;
      }
      DeferredOpEvent event;
      if (IsSideEffectingOp(deferred_op->op_name())) {
        ABSL_VLOG(3) << "[DeferredOpEvent] Created side-effecting event for "
                     << device_buffer_list.get() << "("
                     << ToString(deferred_op->op_name()) << ")";
        event.side_effects = device_buffer_list;
      } else {
        ABSL_VLOG(3)
            << "[DeferredOpEvent] Created non-side-effecting event for "
            << device_buffer_list.get() << "("
            << ToString(deferred_op->op_name()) << ")";
        event.no_side_effects = device_buffer_list;
      }
      return event;
    }

    absl_nullable std::shared_ptr<DeviceBufferList> lock() const {
      return side_effects ? side_effects : no_side_effects.lock();
    }

    // At most one of these will be a valid shared pointer.
    // If neither is valid, that indicates the DeviceBufferList has been freed.
    std::weak_ptr<DeviceBufferList> no_side_effects;
    absl_nullable std::shared_ptr<DeviceBufferList> side_effects;
  };

  void RecordDeferredOpCreated(
      const SharedDeviceBufferList& device_buffer_list) {
    if (auto event = DeferredOpEvent::FromDeferredOp(device_buffer_list);
        event.has_value()) {
      absl::MutexLock lock(deferred_ops_mu_);
      deferred_ops_.push_back(std::move(*event));
    }
  }

  // Partitions the deferred ops queue.
  //
  // The returned vector contains all the DeferredOpEvents that were in the
  // queue, up to and including the last node that is in nodes_to_materialize.
  //
  // Anything after the last node in nodes_to_materialize is left in the queue.
  // If the last node in the queue is in nodes_to_materialize, then the entire
  // queue is returned (as a vector) and the queue is cleared.
  //
  // Any nodes in nodes_to_materialize that are not in the deferred ops queue
  // are ignored. If no nodes are found, returns an empty vector and does not
  // modify the queue.
  std::vector<DeferredOpEvent> TakeUntil(
      const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
          nodes_to_materialize) {
    if (nodes_to_materialize.empty()) {
      return {};
    }

    absl::MutexLock lock(deferred_ops_mu_);
    std::deque<DeferredOpEvent> retained_ops;

    // Pop off the back of the queue until we see a node to materialize.
    while (!deferred_ops_.empty()) {
      auto op_list = deferred_ops_.back().lock();
      if (op_list && nodes_to_materialize.contains(op_list.get())) {
        // Found the last event in deferred_ops_ which needs to be materialized.
        break;
      } else {
        // Since nodes_to_materialize holds strong pointers, if we see an
        // expired weak pointer, we know it wasn't one of the nodes to
        // materialize.
        retained_ops.push_front(std::move(deferred_ops_.back()));
        deferred_ops_.pop_back();
      }
    }
    if (deferred_ops_.empty()) {
      ABSL_VLOG(1) << "[EventsQueue::TakeUntil] None of "
                   << nodes_to_materialize.size()
                   << " nodes to materialize were found in deferred ops queue.";
      std::swap(deferred_ops_, retained_ops);
      return {};
    }

    std::vector<DeferredOpEvent> result(
        std::make_move_iterator(deferred_ops_.begin()),
        std::make_move_iterator(deferred_ops_.end()));
    deferred_ops_.clear();
    if (retained_ops.empty()) {
      ABSL_VLOG(1) << "[EventsQueue::TakeUntil] The last node to materialize "
                      "was the last deferred op. Returning the entire "
                      "queue.\nReturning all "
                   << result.size() << " DeferredOpEvents.";
    } else {
      ABSL_VLOG(1) << "[EventsQueue::TakeUntil] Partitioned deferred ops "
                      "queue.\nRetaining the last "
                   << retained_ops.size()
                   << " DeferredOpEvents, and returning the first "
                   << result.size() << " DeferredOpEvents.";
      std::swap(deferred_ops_, retained_ops);
    }
    return result;
  }

 private:
  absl::Mutex data_ptr_mu_;
  // Hold a strong pointer to the DeviceBufferList as the key; as long as there
  // is a live DataPtr, the DeviceBufferList can't be dropped.
  absl::flat_hash_map<SharedDeviceBufferList, int64_t> live_nodes_
      ABSL_GUARDED_BY(data_ptr_mu_);

  absl::Mutex deferred_ops_mu_;
  std::deque<DeferredOpEvent> deferred_ops_ ABSL_GUARDED_BY(deferred_ops_mu_);
};

}  // namespace

void RecordNewDataPtrCreated(const DeviceBufferRef& device_buffer_ref) {
  EventsQueue::GetInstance().RecordNewDataPtrCreated(device_buffer_ref);
}

void RecordDataPtrDestroyed(const DeviceBufferRef& device_buffer_ref) {
  EventsQueue::GetInstance().RecordDataPtrDestroyed(device_buffer_ref);
}

void RecordDeferredOpCreated(const SharedDeviceBufferList& device_buffer_list) {
  EventsQueue::GetInstance().RecordDeferredOpCreated(device_buffer_list);
}

std::vector<SharedDeviceBufferList> GetAllLiveUnsyncedDataPtrs() {
  return EventsQueue::GetInstance().GetAllLiveUnsyncedDataPtrs();
}

void ClearEventsQueue() { EventsQueue::GetInstance().Clear(); }

namespace {

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

using DefinedNodeMap = absl::flat_hash_map<const DeviceBufferList*, OpUsage>;

// Helper function for PrepareMaterializationTraversals.
// Pushes the device buffer list to the execution order (unless it is empty),
// and updates the defined node map.
void ProcessDeferredOpEvent(
    SharedDeviceBufferList&& device_buffer_list, const DeferredOp& deferred_op,
    const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        nodes_to_materialize_set,
    std::vector<SharedDeviceBufferList>& execution_order,
    DefinedNodeMap& defined_node_map) {
  if (device_buffer_list->is_empty()) {
    // `torch.empty()` and similar ops represent uninitialized memory; correct
    // user programs should never read them, so we should typically never need
    // to execute them.
    //
    // However, partial in-place writes are implemented through
    // stablehlo.dynamic_update_slice, which merge in valid values with the
    // uninitialized ones. If we see one of these ops, we will need to
    // execute it, so we add it to the execution order immediately before this
    // first read. This is implemented later in this function (see the
    // input.is_empty() check) below.
    //
    // Additionally, users may try to explicitly materialize empty tensors. In
    // this case, we append them to the last traversal. This is implemented in
    // PrepareMaterializationTraversals before the final return.
    ABSL_VLOG(3) << "[ProcessDeferredOpEvent] Skipping empty buffer "
                 << device_buffer_list.get();
    return;
  }

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
    } else if (input.is_empty() &&
               defined_node_map
                   .try_emplace(input_device_buffer_list, OpUsage::kUsed)
                   .second) {
      // The first time an empty tensor is read by a later op, we insert
      // it into the execution order, but not as an output as we don't want to
      // materialize it unless the user explicitly asks for it.
      ABSL_VLOG(3) << "[ProcessDeferredOpEvent] Empty buffer "
                   << input_device_buffer_list << " is read by "
                   << device_buffer_list.get() << "("
                   << ToString(deferred_op.op_name())
                   << ").\nInserting into execution order.";
      execution_order.push_back(input.device_buffer_list());
    }
  }

  if (deferred_op.op_name() == OpName::kTorchTpuInternalConstant) {
    if (nodes_to_materialize_set.contains(device_buffer_list.get())) {
      ABSL_VLOG(3)
          << "[ProcessDeferredOpEvent] Adding explicitly materialized constant "
          << device_buffer_list.get() << " as output";
      defined_node_map[device_buffer_list.get()] = OpUsage::kOutput;
    } else {
      ABSL_VLOG(3)
          << "[ProcessDeferredOpEvent] Skipping materialization of constant "
          << device_buffer_list.get() << ", adding to execution order only";
      defined_node_map[device_buffer_list.get()] = OpUsage::kUsed;
    }
  } else if (!device_buffer_list->is_stale()) {
    // TODO(bawilson): use data pointer events to determine liveness instead
    // of the live_data_ptr atomic to remove the dispatch/materialize race.
    ABSL_VLOG(3) << "[ProcessDeferredOpEvent] Adding live buffer "
                 << device_buffer_list.get() << " as output";
    defined_node_map[device_buffer_list.get()] = OpUsage::kOutput;
  } else if (nodes_to_materialize_set.contains(device_buffer_list.get())) {
    ABSL_VLOG(3)
        << "[ProcessDeferredOpEvent] Adding explicitly materialized buffer "
        << device_buffer_list.get() << " as output";
    defined_node_map[device_buffer_list.get()] = OpUsage::kOutput;
  } else {
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
      defined_node_map[device_buffer_list.get()] = OpUsage::kOutput;
    } else {
      // Non-output nodes need to be used by at least one output to ensure
      // they get executed.
      ABSL_VLOG(3) << "[ProcessDeferredOpEvent] Adding non-output buffer "
                   << device_buffer_list.get() << " to execution order";
      defined_node_map[device_buffer_list.get()] = OpUsage::kUnused;
    }
  }

  execution_order.push_back(std::move(device_buffer_list));
}

// Helper function for PrepareMaterializationTraversals.
// Takes an execution_order, a map defining the usage of each node, and returns
// a traversal (unless the execution order and output nodes are empty).
absl::StatusOr<absl_nullable std::unique_ptr<Traversal>> FinishTraversal(
    std::vector<SharedDeviceBufferList>& execution_order,
    DefinedNodeMap& defined_node_map,
    std::vector<SharedDeviceBufferList>& output_nodes) {
  if (execution_order.empty()) {
    ABSL_CHECK(defined_node_map.empty())  // CRASH_OK
        << "defined node map is not empty when execution order is empty. These "
           "should always have the same size";
    return nullptr;
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

  TT_ASSIGN_OR_RETURN(auto traversal, Traversal::CreateFromExecutionOrder(
                                          execution_order, output_nodes));
  execution_order.clear();
  defined_node_map.clear();
  output_nodes.clear();
  return traversal;
}

}  // namespace

absl::StatusOr<std::vector<absl_nonnull std::unique_ptr<Traversal>>>
PrepareMaterializationTraversals(
    absl::Span<const SharedDeviceBufferList> nodes_to_materialize) {
  std::vector<absl_nonnull std::unique_ptr<Traversal>> traversals;
  if (nodes_to_materialize.empty()) {
    return traversals;
  }

  absl::flat_hash_set<const DeviceBufferList*> nodes_to_materialize_set;
  for (const auto& node : nodes_to_materialize) {
    nodes_to_materialize_set.insert(node.get());
  }

  // Do the partitioning of the deferred ops queue.
  // This is the only part than needs to hold the lock.
  std::vector<EventsQueue::DeferredOpEvent> deferred_op_events =
      EventsQueue::GetInstance().TakeUntil(nodes_to_materialize_set);

  // Partition the deferred ops queue into separate traversals with these rules:
  //   - If an op is SplitBefore, it must be first in its execution order.
  //   - If an op is SplitAfter, it must be last in its execution order, and
  //     must be an output.
  //   - If a node has live c10::DataPtrs, it must be an output.
  //   - If a node is in nodes_to_materialize, it must be an output.
  //   - Every non-empty op in execution_order must be executed by at least one
  //     output.
  //   - Empty() ops are added immediately before the first read, or at
  //     the end if they are explicitly materialized but not read.
  // Reuse working memory for efficiency.
  std::vector<SharedDeviceBufferList> execution_order;
  std::vector<SharedDeviceBufferList> output_nodes;
  DefinedNodeMap defined_node_map;

  for (const auto& event : deferred_op_events) {
    // Filter to only non-expired, deferred nodes.
    auto device_buffer_list = event.lock();
    if (!device_buffer_list) continue;
    const auto deferred_op = device_buffer_list->deferred_op();
    if (!deferred_op) continue;

    const auto split_mode = deferred_op->split_mode();

    if (IsSplitBefore(split_mode)) {
      TT_ASSIGN_OR_RETURN(
          auto maybe_traversal,
          FinishTraversal(execution_order, defined_node_map, output_nodes));
      if (maybe_traversal != nullptr) {
        ABSL_VLOG(2) << "[PrepareMaterializationTraversals] Split out "
                     << maybe_traversal->execution_order().size()
                     << " nodes before SplitBefore buffer "
                     << device_buffer_list.get() << "("
                     << ToString(deferred_op->op_name()) << ").";
        traversals.push_back(std::move(maybe_traversal));
      }
    }

    ProcessDeferredOpEvent(std::move(device_buffer_list), *deferred_op,
                           nodes_to_materialize_set, execution_order,
                           defined_node_map);

    if (IsSplitAfter(split_mode)) {
      TT_ASSIGN_OR_RETURN(
          auto maybe_traversal,
          FinishTraversal(execution_order, defined_node_map, output_nodes));
      if (maybe_traversal != nullptr) {
        ABSL_VLOG(2) << "[PrepareMaterializationTraversals] Split out "
                     << maybe_traversal->execution_order().size()
                     << " nodes after SplitAfter buffer "
                     << device_buffer_list.get() << "("
                     << ToString(deferred_op->op_name()) << ").";
        traversals.push_back(std::move(maybe_traversal));
      }
    }
  }

  // We never materialize an empty tensor unless it is explicitly
  // requested by the user. If that does happen, then we append the
  // empty tensors to the last Traversal to make sure they have defined buffers.
  // This is necessary for downstream uses that require physical data buffers,
  // such as torch.compile invocations.
  for (const auto& node : nodes_to_materialize) {
    if (node->is_empty() && node->is_deferred()) {
      bool inserted =
          defined_node_map.insert_or_assign(node.get(), OpUsage::kOutput)
              .second;
      if (inserted) {
        ABSL_VLOG(3) << "[PrepareMaterializationTraversals] empty buffer "
                     << node.get()
                     << " is an explicit output. Appending to final traversal.";
        execution_order.push_back(node);
      }
    }
  }

  TT_ASSIGN_OR_RETURN(
      auto maybe_traversal,
      FinishTraversal(execution_order, defined_node_map, output_nodes));
  if (maybe_traversal != nullptr) {
    ABSL_VLOG(2)
        << "[PrepareMaterializationTraversals] Final traversal has size "
        << maybe_traversal->execution_order().size() << ", ending with node "
        << maybe_traversal->execution_order().back().get();
    traversals.push_back(std::move(maybe_traversal));
  }
  ABSL_VLOG(1) << "[PrepareMaterializationTraversals] Created "
               << traversals.size() << " traversals to materialize "
               << nodes_to_materialize.size() << " nodes.";
  return traversals;
}

namespace {
struct StreamState {
  absl::Mutex mutex;
  absl::flat_hash_map<std::pair<  // STD_PAIR_OK=use default absl::Hash impl
                          c10::DeviceIndex, int64_t>,
                      std::vector<xla::Future<void>>>
      pending_futures ABSL_GUARDED_BY(mutex);
};

StreamState& GetStreamState() {
  static absl::NoDestructor<StreamState> state;
  return *state;
}

void PruneCompletedFutures(std::vector<xla::Future<void>>& futures) {
  futures.erase(std::remove_if(futures.begin(), futures.end(),
                               [](const xla::Future<void>& future) {
                                 return !future.IsValid() || future.IsReady();
                               }),
                futures.end());
}

void WaitForFutures(c10::DeviceIndex dev_idx, int64_t stream_id,
                    const std::vector<xla::Future<void>>& futures) {
  for (auto& future : futures) {
    if (future.IsValid()) {
      absl::Status s = future.Await();
      if (!s.ok()) {
        ABSL_LOG(ERROR) << "Stream synchronization failed for device "
                        << static_cast<int>(dev_idx) << " stream " << stream_id
                        << ": " << s;
      }
    }
  }
}

struct StreamFutures {
  c10::StreamId stream_id;
  std::vector<xla::Future<void>> futures;
};

void MarkStreamActive(c10::DeviceIndex device_index, int64_t stream_id,
                      std::vector<xla::Future<void>>&& new_futures) {
  if (new_futures.empty()) {
    return;
  }
  StreamState& state = GetStreamState();
  absl::MutexLock lock(state.mutex);
  auto& futures = state.pending_futures[{device_index, stream_id}];
  if (futures.size() + new_futures.size() > futures.capacity()) {
    PruneCompletedFutures(futures);
  }
  futures.insert(futures.end(), std::make_move_iterator(new_futures.begin()),
                 std::make_move_iterator(new_futures.end()));
}

void MarkStreamActive(std::vector<xla::Future<void>>&& new_futures) {
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  MarkStreamActive(device_index, stream_id, std::move(new_futures));
}

void MarkStreamActive(xla::Future<void> future) {
  std::vector<xla::Future<void>> futures;
  futures.reserve(1);
  futures.push_back(std::move(future));
  MarkStreamActive(std::move(futures));
}

}  // namespace

void RecordBackgroundMaterialization(
    absl::Span<const DeviceBufferRef> outputs) {
  std::vector<xla::Future<void>> new_futures;
  new_futures.reserve(outputs.size());
  for (const auto& output : outputs) {
    new_futures.push_back(output.GetReadyFuture());
  }
  MarkStreamActive(std::move(new_futures));
}

void RecordAsyncHostToDevice(const DeviceBufferRef& device_buffer_ref) {
  MarkStreamActive(device_buffer_ref.GetReadyFuture());
}

void RecordAsyncDeviceToHost(xla::Future<void> to_literal_future) {
  MarkStreamActive(std::move(to_literal_future));
}

// Blocks the calling thread until all previously enqueued operations on the
// specified device's stream have completed.
// If multiple threads call SynchronizeStream for the same device concurrently,
// all threads will block until the pending operations at the time of their
// call are finished.
void SynchronizeStream(c10::DeviceIndex device_index, int64_t stream_id) {
  ABSL_VLOG(1) << "SynchronizeStream: device=" << static_cast<int>(device_index)
               << ", stream=" << stream_id;
  StreamState& state = GetStreamState();
  std::vector<xla::Future<void>> futures;
  {
    absl::MutexLock lock(state.mutex);
    auto it = state.pending_futures.find({device_index, stream_id});
    if (it == state.pending_futures.end()) {
      // No pending futures for this device and stream.
      return;
    }
    futures = it->second;
  }
  for (auto& future : futures) {
    if (future.IsValid()) {
      absl::Status s = future.Await();
      if (!s.ok()) {
        ABSL_LOG(ERROR) << "Stream synchronization failed for device "
                        << static_cast<int>(device_index) << " stream "
                        << stream_id << ": " << s;
      }
    }
  }
  {
    absl::MutexLock lock(state.mutex);
    auto it = state.pending_futures.find({device_index, stream_id});
    if (it != state.pending_futures.end()) {
      PruneCompletedFutures(it->second);
    }
  }
}

void SynchronizeDevice(c10::DeviceIndex device_index) {
  ABSL_VLOG(1) << "SynchronizeDevice: device="
               << static_cast<int>(device_index);
  StreamState& state = GetStreamState();
  std::vector<StreamFutures> all_stream_futures;
  {
    absl::MutexLock lock(state.mutex);
    for (const auto& [key, stream_futures] : state.pending_futures) {
      if (key.first == device_index) {
        all_stream_futures.push_back({key.second, stream_futures});
      }
    }
  }

  for (const auto& sf : all_stream_futures) {
    WaitForFutures(device_index, sf.stream_id, sf.futures);
  }

  {
    absl::MutexLock lock(state.mutex);
    for (auto& [key, stream_futures] : state.pending_futures) {
      if (key.first == device_index) {
        PruneCompletedFutures(stream_futures);
      }
    }
  }
}

std::shared_ptr<EventSnapshot> EventSnapshot::Record(
    c10::DeviceIndex device_index, int64_t stream_id) {
  StreamState& state = GetStreamState();
  absl::MutexLock lock(state.mutex);
  auto it = state.pending_futures.find({device_index, stream_id});
  if (it == state.pending_futures.end()) {
    return std::shared_ptr<EventSnapshot>(
        new EventSnapshot(xla::Future<void>(absl::OkStatus())));
  }
  PruneCompletedFutures(it->second);
  if (it->second.empty()) {
    return std::shared_ptr<EventSnapshot>(
        new EventSnapshot(xla::Future<void>(absl::OkStatus())));
  }
  auto join_future = JoinFutures(it->second);
  it->second.clear();
  it->second.push_back(join_future);  // intentional copy

  // Can't use make_shared because the constructor is private.
  return std::shared_ptr<EventSnapshot>(
      new EventSnapshot(std::move(join_future)));
}

absl::Status EventSnapshot::Wait() const { return future_.Await(); }

absl::StatusOr<bool> EventSnapshot::Query() const { return future_.IsReady(); }

}  // namespace torch_tpu
