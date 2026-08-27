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

#include "torch_tpu/csrc/eager/events_queue.h"

#include <algorithm>
#include <array>
#include <atomic>
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
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/context_states.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/shape.h"
#include "torch_tpu/csrc/eager/current_stream.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/eager_mode.h"
#include "torch_tpu/csrc/eager/traversal.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/csrc/ops/python_context.h"
#include "xla/future.h"

namespace torch_tpu {

namespace {

constexpr int kMaxTorchDevices = 8;

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

// An ordered queue of "work" events on a stream. This includes:
//  - Deferred op creation (which represents an on-device execution)
//  - DataPtr creation and destruction (allocation/deallocation events)
class EventsQueue {
 public:
  // Records on the events queue that a new c10::DataPtr referencing the given
  // DeviceBufferRef has been created.
  void RecordNewDataPtrCreated(const DeviceBufferRef& device_buffer_ref) {
    // Placeholders and constant tensors are never inserted into the map.
    if (device_buffer_ref.is_placeholder() || device_buffer_ref.is_constant()) {
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
    // Placeholders and constant tensors are never inserted into the map.
    if (device_buffer_ref.is_placeholder() || device_buffer_ref.is_constant()) {
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
      if (device_buffer_list->is_constant()) {
        // Compiled mode constants and empty tensors are handled specially;
        // normally, memoizing a computation by materializing it is used to
        // prevent later reexecution.
        // However, materializing a constant prevents later constant folding;
        // re-execution preserves their constantness in later compilations.
        return std::nullopt;
      }

      const auto deferred_op = device_buffer_list->deferred_op();
      if (!deferred_op) {
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

  // Returns all the DeferredOpEvents in the queue and clears the queue.
  std::vector<DeferredOpEvent> TakeAll() {
    absl::MutexLock lock(deferred_ops_mu_);
    ABSL_VLOG(1) << "[EventsQueue::TakeAll] Returning all "
                 << deferred_ops_.size() << " DeferredOpEvents.";
    std::vector<DeferredOpEvent> result(
        std::make_move_iterator(deferred_ops_.begin()),
        std::make_move_iterator(deferred_ops_.end()));
    deferred_ops_.clear();
    return result;
  }

  // Partitions the deferred ops queue.
  //
  // The returned vector contains all the DeferredOpEvents that were in the
  // queue, up to and including the last-enqueued node that is in
  // nodes_to_materialize.
  //
  // Anything in the queue after the last-enqueued node in nodes_to_materialize
  // is left in the queue. If the last node in the queue is in
  // nodes_to_materialize, then the entire queue is returned (as a vector) and
  // the queue is cleared.
  //
  // Any nodes in nodes_to_materialize that are not in the deferred ops queue
  // are ignored. If no nodes are found, returns an empty vector and does not
  // modify the queue.
  std::vector<DeferredOpEvent> TakeUntilNodes(
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
        retained_ops.push_front(std::move(deferred_ops_.back()));
        deferred_ops_.pop_back();
      }
    }
    if (deferred_ops_.empty()) {
      ABSL_VLOG(1)
          << "[EventsQueue::TakeUntilNodes] No nodes to materialize were "
             "found in deferred ops queue.";
      std::swap(deferred_ops_, retained_ops);
      return {};
    }

    std::vector<DeferredOpEvent> result(
        std::make_move_iterator(deferred_ops_.begin()),
        std::make_move_iterator(deferred_ops_.end()));
    deferred_ops_.clear();
    if (retained_ops.empty()) {
      ABSL_VLOG(1)
          << "[EventsQueue::TakeUntilNodes] The last node to materialize "
             "was the last deferred op. Returning the entire "
             "queue.\nReturning all "
          << result.size() << " DeferredOpEvents.";
    } else {
      ABSL_VLOG(1) << "[EventsQueue::TakeUntilNodes] Partitioned deferred ops "
                      "queue.\nRetaining the last "
                   << retained_ops.size()
                   << " DeferredOpEvents, and returning the first "
                   << result.size() << " DeferredOpEvents.";
      std::swap(deferred_ops_, retained_ops);
    }
    return result;
  }

  // Returns the core pinning mode of this events queue.
  CorePinningMode core_pinning_mode() const { return core_pinning_mode_; }
  // Sets the device pinning mode of this events queue.
  void SetCorePinningMode(CorePinningMode core_pinning_mode) {
    core_pinning_mode_ = core_pinning_mode;
  }

 private:
  absl::Mutex data_ptr_mu_;
  // Hold a strong pointer to the DeviceBufferList as the key; as long as there
  // is a live DataPtr, the DeviceBufferList can't be dropped.
  absl::flat_hash_map<SharedDeviceBufferList, int64_t> live_nodes_
      ABSL_GUARDED_BY(data_ptr_mu_);

  absl::Mutex deferred_ops_mu_;
  std::deque<DeferredOpEvent> deferred_ops_ ABSL_GUARDED_BY(deferred_ops_mu_);

  std::atomic<CorePinningMode> core_pinning_mode_ = CorePinningMode::kUnpinned;
};

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

// Returns the OpUsage of the given device buffer list, switching between
// kOutput (if it's a required output) or kUnused (if it's not).
OpUsage GetOpUsage(
    const DeferredOp& deferred_op,
    const SharedDeviceBufferList& device_buffer_list,
    const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        nodes_to_materialize_set) {
  // Explicitly materialized nodes are always outputs, even if they normally
  // wouldn't be (e.g. constants)
  if (nodes_to_materialize_set.contains(device_buffer_list.get())) {
    ABSL_VLOG(3)
        << "[ProcessDeferredOpEvent] Adding explicitly materialized buffer "
        << device_buffer_list.get() << " as output";
    return OpUsage::kOutput;
  }

  if (!device_buffer_list->is_stale()) {
    // TODO(bawilson): use data pointer events to determine liveness instead
    // of the live_data_ptr atomic to remove the dispatch/materialize race.
    ABSL_VLOG(3) << "[ProcessDeferredOpEvent] Adding live buffer "
                 << device_buffer_list.get() << " as output";
    return OpUsage::kOutput;
  }

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
    ABSL_VLOG(3) << "[ProcessDeferredOpEvent] Adding dynamic-dimension buffer "
                 << device_buffer_list.get() << " as output";
    return OpUsage::kOutput;
  }

  // Non-output nodes need to be used by at least one output to ensure they get
  // executed.
  ABSL_VLOG(3) << "[ProcessDeferredOpEvent] Adding non-output buffer "
               << device_buffer_list.get() << " to execution order";
  return OpUsage::kUnused;
}

// Helper function for PrepareMaterializationTraversals.
// Pushes the device buffer list to the execution order (unless it is empty),
// and updates the defined node map.
void ProcessDeferredOpEvent(
    SharedDeviceBufferList&& device_buffer_list, const DeferredOp& deferred_op,
    const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        nodes_to_materialize_set,
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

// Helper function for PrepareMaterializationTraversals.
// Takes an execution_order, a map defining the usage of each node, and returns
// a traversal (unless the execution order and output nodes are empty).
absl::StatusOr<absl_nullable std::unique_ptr<Traversal>> FinishTraversal(
    std::vector<SharedDeviceBufferList>& execution_order,
    DefinedNodeMap& defined_node_map,
    std::vector<SharedDeviceBufferList>& output_nodes,
    CorePinningMode core_pinning_mode) {
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
  absl_nullable std::unique_ptr<Traversal> traversal = nullptr;
  if (!output_nodes.empty()) {
    TT_ASSIGN_OR_RETURN(traversal, Traversal::CreateFromExecutionOrder(
                                       execution_order, output_nodes));
  }
  execution_order.clear();
  defined_node_map.clear();
  output_nodes.clear();
  if (core_pinning_mode != CorePinningMode::kUnpinned) {
    traversal->SetCorePinningMode(core_pinning_mode);
  }

  return traversal;
}

absl::StatusOr<std::vector<absl_nonnull std::unique_ptr<Traversal>>>
PrepareTraversals(
    absl::Span<const EventsQueue::DeferredOpEvent> deferred_op_events,
    absl::Span<const SharedDeviceBufferList> nodes_to_materialize,
    const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        nodes_to_materialize_set,
    const CorePinningMode core_pinning_mode) {
  std::vector<absl_nonnull std::unique_ptr<Traversal>> traversals;

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

  for (const auto& event : deferred_op_events) {
    // Filter to only non-expired, deferred nodes.
    auto device_buffer_list = event.lock();
    if (!device_buffer_list) continue;
    const auto deferred_op = device_buffer_list->deferred_op();
    if (!deferred_op) continue;

    const auto split_mode = deferred_op->split_mode();

    if (IsSplitBefore(split_mode)) {
      TT_ASSIGN_OR_RETURN(auto maybe_traversal,
                          FinishTraversal(execution_order, defined_node_map,
                                          output_nodes, core_pinning_mode));
      if (maybe_traversal != nullptr) {
        ABSL_VLOG(2) << "[PrepareTraversals] Split out "
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
      TT_ASSIGN_OR_RETURN(auto maybe_traversal,
                          FinishTraversal(execution_order, defined_node_map,
                                          output_nodes, core_pinning_mode));
      if (maybe_traversal != nullptr) {
        ABSL_VLOG(2) << "[PrepareTraversals] Split out "
                     << maybe_traversal->execution_order().size()
                     << " nodes after SplitAfter buffer "
                     << device_buffer_list.get() << "("
                     << ToString(deferred_op->op_name()) << ").";
        traversals.push_back(std::move(maybe_traversal));
      }
    }
  }

  // We never materialize a constant (or empty) tensor unless it is explicitly
  // requested by the user. If that does happen, then we append these constant
  // tensors to the last Traversal to make sure they have defined buffers.
  // This is necessary for downstream uses that require physical data buffers,
  // such as torch.compile invocations.
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

  TT_ASSIGN_OR_RETURN(auto maybe_traversal,
                      FinishTraversal(execution_order, defined_node_map,
                                      output_nodes, core_pinning_mode));
  if (maybe_traversal != nullptr) {
    ABSL_VLOG(2) << "[PrepareTraversals] Final traversal has size "
                 << maybe_traversal->execution_order().size()
                 << ", ending with node "
                 << maybe_traversal->execution_order().back().get();
    traversals.push_back(std::move(maybe_traversal));
  }
  ABSL_VLOG(1) << "[PrepareTraversals] Created " << traversals.size()
               << " traversals to materialize " << nodes_to_materialize.size()
               << " nodes.";
  return traversals;
}

void PruneCompletedFutures(std::vector<xla::Future<void>>& futures) {
  futures.erase(std::remove_if(futures.begin(), futures.end(),
                               [](const xla::Future<void>& future) {
                                 return !future.IsValid() || future.IsReady();
                               }),
                futures.end());
}

struct StreamState {
  void MarkActive(std::vector<xla::Future<void>>&& new_futures) {
    if (new_futures.empty()) {
      return;
    }
    absl::MutexLock lock(futures_mu);
    if (futures.size() + new_futures.size() > futures.capacity()) {
      PruneCompletedFutures(futures);
    }
    futures.insert(futures.end(), std::make_move_iterator(new_futures.begin()),
                   std::make_move_iterator(new_futures.end()));
  }

  xla::Future<void> JoinFutures() {
    absl::MutexLock lock(futures_mu);
    if (!futures.empty()) {
      PruneCompletedFutures(futures);
    }
    if (futures.empty()) {
      return xla::Future<void>(absl::OkStatus());
    }
    auto join_future = xla::JoinFutures(futures);
    futures.clear();
    futures.push_back(join_future);  // intentional copy
    return join_future;
  }

  void Clear() {
    events_queue.Clear();
    absl::MutexLock lock(futures_mu);
    futures.clear();
  }

  absl::Mutex futures_mu;
  std::vector<xla::Future<void>> futures ABSL_GUARDED_BY(futures_mu);

  EventsQueue events_queue;
};

struct DeviceState {
  StreamState* absl_nonnull GetOrCreateStreamState(int64_t stream_id);

  std::vector<StreamState* absl_nonnull> GetAllStreamStates() {
    absl::MutexLock lock(mutex);
    std::vector<StreamState* absl_nonnull> including_default;
    including_default.reserve(stream_states.size() + 1);
    including_default.push_back(&default_stream_state);
    for (const auto& stream_state : stream_states) {
      including_default.push_back(stream_state.get());
    }
    return including_default;
  }

  void Clear() {
    // Keep the default stream but clear it so it is empty.
    default_stream_state.Clear();

    // Drop all other stream states.
    absl::MutexLock lock(mutex);
    stream_states.clear();
  }

  StreamState default_stream_state;
  absl::Mutex mutex;
  // TODO: currently this vector only grows and never deallocates a StreamState
  // once created. This could cause host-side OOMs if the user creates new
  // streams in a loop.
  // Possible alternatives:
  //   - Allocate a fixed number of stream states and reuse them; this is the
  //     approach used by XPU. This may have unexpected behavior for users with
  //     many streams, as multiple Python streams will actually be the same C++
  //     stream.
  //   - Use a destructor on the TorchTPU specific torch.tpu.Stream object with
  //     a free-list to reuse streams. This will *not* work for the generic
  //     c10::Stream object as this has no destructor we can hook into, so
  //     torch.Stream(device="tpu") objects cannot be reused.
  std::vector<std::unique_ptr<StreamState>> stream_states
      ABSL_GUARDED_BY(mutex);
};

StreamState* absl_nonnull DeviceState::GetOrCreateStreamState(
    int64_t stream_id) {
  ABSL_CHECK(stream_id >= 0);  // CRASH_OK
  if (stream_id == 0) {
    return &default_stream_state;
  }
  auto stream_index = stream_id - 1;
  absl::MutexLock lock(mutex);
  while (stream_states.size() <= stream_index) {
    stream_states.push_back(std::make_unique<StreamState>());
  }
  return stream_states[stream_index].get();
}

struct StreamStates {
  DeviceState& GetDeviceState(c10::DeviceIndex device_index) {
    ABSL_CHECK(device_index >= 0 &&  // CRASH_OK
               device_index < kMaxTorchDevices);
    return device_states[device_index];
  }

  void Clear() {
    for (auto& device_state : device_states) {
      device_state.Clear();
    }
  }

  std::array<DeviceState, kMaxTorchDevices> device_states;
};

StreamStates& GetStreamStates() {
  static absl::NoDestructor<StreamStates> states;
  return *states;
}

StreamState* absl_nonnull GetOrCreateStreamState(c10::DeviceIndex device_index,
                                                 int64_t stream_id) {
  return GetStreamStates()
      .GetDeviceState(device_index)
      .GetOrCreateStreamState(stream_id);
}

std::vector<StreamState* absl_nonnull> GetDeviceStreamStates(
    c10::DeviceIndex device_index) {
  return GetStreamStates().GetDeviceState(device_index).GetAllStreamStates();
}

void MarkStreamActive(c10::DeviceIndex device_index, int64_t stream_id,
                      std::vector<xla::Future<void>>&& new_futures) {
  GetOrCreateStreamState(device_index, stream_id)
      ->MarkActive(std::move(new_futures));
}

void MarkStreamActive(c10::DeviceIndex device_index, int64_t stream_id,
                      xla::Future<void> future) {
  std::vector<xla::Future<void>> futures;
  futures.reserve(1);
  futures.push_back(std::move(future));
  MarkStreamActive(device_index, stream_id, std::move(futures));
}

StreamState* absl_nonnull GetStreamFor(
    const SharedDeviceBufferList& device_buffer_list) {
  return GetOrCreateStreamState(device_buffer_list->device_index(),
                                device_buffer_list->stream_id());
}

StreamState* absl_nonnull GetStreamFor(
    const DeviceBufferRef& device_buffer_ref) {
  return GetOrCreateStreamState(device_buffer_ref.device_index(),
                                device_buffer_ref.stream_id());
}

}  // namespace

void RecordNewDataPtrCreated(const DeviceBufferRef& device_buffer_ref) {
  if (GetEagerMode() == EagerMode::kInternalCompileFxGraph) {
    // In FX trace mode, nothing happens on device, so we don't add anything to
    // any stream.
    return;
  }
  GetStreamFor(device_buffer_ref)
      ->events_queue.RecordNewDataPtrCreated(device_buffer_ref);
}

void RecordDataPtrDestroyed(const DeviceBufferRef& device_buffer_ref) {
  if (GetEagerMode() == EagerMode::kInternalCompileFxGraph) {
    // In FX trace mode, nothing happens on device, so we don't add anything to
    // any stream.
    return;
  }
  GetStreamFor(device_buffer_ref)
      ->events_queue.RecordDataPtrDestroyed(device_buffer_ref);
}

void RecordDeferredOpCreated(const SharedDeviceBufferList& device_buffer_list) {
  if (GetEagerMode() == EagerMode::kInternalCompileFxGraph) {
    // In FX trace mode, nothing happens on device, so we don't add anything to
    // any stream.
    return;
  }
  GetStreamFor(device_buffer_list)
      ->events_queue.RecordDeferredOpCreated(device_buffer_list);
}

void RecordBackgroundMaterialization(
    absl::Span<const DeviceBufferRef> outputs) {
  if (outputs.empty()) return;

  const auto first_device_index = outputs[0].device_index();
  const auto first_stream_id = outputs[0].stream_id();
  if (outputs.size() == 1 ||
      std::all_of(
          outputs.begin(), outputs.end(),
          [first_device_index, first_stream_id](const DeviceBufferRef& output) {
            return output.device_index() == first_device_index &&
                   output.stream_id() == first_stream_id;
          })) {
    // All outputs are on the same stream, only need to mark this one stream
    // active.
    std::vector<xla::Future<void>> new_futures;
    new_futures.reserve(outputs.size());
    for (const auto& output : outputs) {
      new_futures.push_back(output.GetReadyFuture());
    }
    MarkStreamActive(first_device_index, first_stream_id,
                     std::move(new_futures));
    return;
  }

  // Create a list of futures for each stream.
  struct NewStreamFutures {
    c10::DeviceIndex device_index;
    int64_t stream_id;
    std::vector<xla::Future<void>> futures;
  };
  std::vector<NewStreamFutures> new_stream_futures;
  for (const auto& output : outputs) {
    auto it = std::find_if(new_stream_futures.begin(), new_stream_futures.end(),
                           [&output](const NewStreamFutures& new_futures) {
                             return new_futures.device_index ==
                                        output.device_index() &&
                                    new_futures.stream_id == output.stream_id();
                           });
    if (it == new_stream_futures.end()) {
      new_stream_futures.push_back({output.device_index(),
                                    output.stream_id(),
                                    {output.GetReadyFuture()}});
    } else {
      it->futures.push_back(output.GetReadyFuture());
    }
  }
  for (auto& stream : new_stream_futures) {
    MarkStreamActive(stream.device_index, stream.stream_id,
                     std::move(stream.futures));
  }
}

absl::StatusOr<std::vector<absl_nonnull std::unique_ptr<Traversal>>>
PrepareMaterializationTraversals(
    absl::Span<const SharedDeviceBufferList> nodes_to_materialize) {
  if (nodes_to_materialize.empty()) {
    return std::vector<absl_nonnull std::unique_ptr<Traversal>>();
  }

  // Partition nodes_to_materialize into streams.
  // Using a vector instead of a map because the number of distinct streams is
  // expected to be small.
  struct StreamNodes {
    c10::DeviceIndex device_index;
    c10::StreamId stream_id;
    std::vector<SharedDeviceBufferList> stream_nodes;
    absl::flat_hash_set<const DeviceBufferList*> stream_node_set;
  };
  std::vector<StreamNodes> nodes_by_stream;
  for (const auto& node : nodes_to_materialize) {
    auto it = std::find_if(nodes_by_stream.begin(), nodes_by_stream.end(),
                           [&node](const StreamNodes& stream_nodes) {
                             return stream_nodes.device_index ==
                                        node->device_index() &&
                                    stream_nodes.stream_id == node->stream_id();
                           });
    if (it == nodes_by_stream.end()) {
      nodes_by_stream.push_back(
          StreamNodes{.device_index = node->device_index(),
                      .stream_id = node->stream_id(),
                      .stream_nodes = {node},  // intentional copy
                      .stream_node_set = {node.get()}});
    } else {
      if (it->stream_node_set.insert(node.get()).second) {
        it->stream_nodes.push_back(node);
      }
    }
  }

  // Get the traversals for each stream represented in nodes_to_materialize,
  // and concatenate them into a single list.
  std::vector<absl_nonnull std::unique_ptr<Traversal>> traversals;

  for (const auto& stream_nodes : nodes_by_stream) {
    // Do the partitioning of the deferred ops queue for this stream.
    // This is the only part that needs to hold the lock.
    ABSL_CHECK(!stream_nodes.stream_nodes.empty());  // CRASH_OK
    auto stream_state = GetStreamFor(stream_nodes.stream_nodes.front());
    auto deferred_op_events =
        stream_state->events_queue.TakeUntilNodes(stream_nodes.stream_node_set);
    const auto pinning_mode = stream_state->events_queue.core_pinning_mode();
    TT_ASSIGN_OR_RETURN(
        auto stream_traversals,
        PrepareTraversals(deferred_op_events, stream_nodes.stream_nodes,
                          stream_nodes.stream_node_set, pinning_mode));
    if (nodes_by_stream.size() == 1) {
      // Only one stream, return directly.
      ABSL_VLOG(1) << "[PrepareMaterializationTraversals] Created "
                   << stream_traversals.size() << " traversals for "
                   << nodes_to_materialize.size() << " nodes, all on device "
                   << static_cast<int>(stream_nodes.device_index)
                   << " and stream "
                   << static_cast<int>(stream_nodes.stream_id);
      return stream_traversals;
    } else {
      ABSL_VLOG(2) << "[PrepareMaterializationTraversals] Created "
                   << stream_traversals.size() << " traversals for device "
                   << static_cast<int>(stream_nodes.device_index) << ", stream "
                   << static_cast<int>(stream_nodes.stream_id);
    }

    traversals.insert(traversals.end(),
                      std::make_move_iterator(stream_traversals.begin()),
                      std::make_move_iterator(stream_traversals.end()));
  }
  ABSL_VLOG(1) << "[PrepareMaterializationTraversals] Created "
               << traversals.size() << " traversals for "
               << nodes_to_materialize.size() << " nodes over "
               << nodes_by_stream.size() << " streams.";
  return traversals;
}

absl::StatusOr<std::vector<absl_nonnull std::unique_ptr<Traversal>>>
PrepareStreamTraversals(c10::DeviceIndex device_index,
                        c10::StreamId stream_id) {
  StreamState* stream_state = GetOrCreateStreamState(device_index, stream_id);
  // Do the partitioning of the deferred ops queue.
  // This is the only part than needs to hold the lock.
  std::vector<EventsQueue::DeferredOpEvent> deferred_op_events =
      stream_state->events_queue.TakeAll();
  return PrepareTraversals(deferred_op_events, /*nodes_to_materialize=*/{},
                           /*nodes_to_materialize_set=*/{},
                           stream_state->events_queue.core_pinning_mode());
}

absl::StatusOr<std::vector<absl_nonnull std::unique_ptr<Traversal>>>
PrepareDeviceTraversals(c10::DeviceIndex device_index) {
  // Get all the stream states for the device.
  auto streams = GetDeviceStreamStates(device_index);

  std::vector<absl_nonnull std::unique_ptr<Traversal>> device_traversals;

  // Partition each stream's deferred ops queue into separate traversals, and
  // concatenate them into a single list.
  for (auto* stream : streams) {
    auto deferred_op_events = stream->events_queue.TakeAll();
    TT_ASSIGN_OR_RETURN(
        auto stream_traversals,
        PrepareTraversals(deferred_op_events, /*nodes_to_materialize=*/{},
                          /*nodes_to_materialize_set=*/{},
                          stream->events_queue.core_pinning_mode()));
    device_traversals.insert(device_traversals.end(),
                             std::make_move_iterator(stream_traversals.begin()),
                             std::make_move_iterator(stream_traversals.end()));
  }
  return device_traversals;
}

void RecordAsyncHostToDevice(const DeviceBufferRef& device_buffer_ref) {
  ABSL_CHECK_NE(  // CRASH_OK=no device transfers during FX trace
      GetEagerMode(), EagerMode::kInternalCompileFxGraph);
  MarkStreamActive(device_buffer_ref.device_index(),
                   device_buffer_ref.stream_id(),
                   device_buffer_ref.GetReadyFuture());
}

void RecordAsyncDeviceToHost(xla::Future<void> to_literal_future) {
  ABSL_CHECK_NE(  // CRASH_OK=no device transfers during FX trace
      GetEagerMode(), EagerMode::kInternalCompileFxGraph);
  // TODO: should this be moved into the function signature?
  // Is there ever a case where we would want to record an async d2h on a
  // different stream than the current stream?
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  MarkStreamActive(device_index, stream_id, std::move(to_literal_future));
}

std::vector<std::shared_ptr<EventSnapshot>> RecordDeviceSnapshots(
    c10::DeviceIndex device_index) {
  ABSL_VLOG(1) << "RecordDeviceSnapshots: device="
               << static_cast<int>(device_index);
  auto device_streams = GetDeviceStreamStates(device_index);
  std::vector<std::shared_ptr<EventSnapshot>> snapshots;
  snapshots.reserve(device_streams.size());
  for (auto& stream_state : device_streams) {
    auto stream_future = stream_state->JoinFutures();
    // Can't use make_shared because the constructor is private.
    snapshots.push_back(std::shared_ptr<EventSnapshot>(
        new EventSnapshot(std::move(stream_future))));
  }
  return snapshots;
}

std::shared_ptr<EventSnapshot> EventSnapshot::Record(
    c10::DeviceIndex device_index, int64_t stream_id) {
  auto join_future =
      GetOrCreateStreamState(device_index, stream_id)->JoinFutures();
  // Can't use make_shared because the constructor is private.
  return std::shared_ptr<EventSnapshot>(
      new EventSnapshot(std::move(join_future)));
}

absl::Status EventSnapshot::Wait() const { return future_.Await(); }

absl::StatusOr<bool> EventSnapshot::Query() const { return future_.IsReady(); }

void ClearAllStreams() {
  GetStreamStates().Clear();
  ResetStreamIdCounters();
}

void SetCorePinningMode(c10::DeviceIndex device_index, c10::StreamId stream_id,
                        CorePinningMode pinned) {
  GetOrCreateStreamState(device_index, stream_id)
      ->events_queue.SetCorePinningMode(pinned);
}

}  // namespace torch_tpu
