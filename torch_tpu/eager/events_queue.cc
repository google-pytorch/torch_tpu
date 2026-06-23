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
#include <array>
#include <cstddef>
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

constexpr int kMaxDevices = 8;

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

// Removes all invalid or known-ready futures from the given vector of futures.
void PruneCompletedFutures(std::vector<xla::Future<void>>& futures) {
  futures.erase(std::remove_if(futures.begin(), futures.end(),
                               [](const xla::Future<void>& future) {
                                 return !future.IsValid() ||
                                        future.IsKnownReady();
                               }),
                futures.end());
}

// A class representing a queue of work that has been pushed to a logical stream
// by torch. "Work" includes both data transfers and dispatched computations.
//
// Whenever "work" is initiated, the StreamState is updated to reflect the
// pending transfer or computation. Later, the StreamState can be caught up to a
// specific point through one of two mechanisms:
//   - Background materialization: this is a TorchTPU internal concept, updating
//     deferred-state tensors to pending-state. This is not visible to PyTorch;
//     the deferred state doesn't exist in CUDA and is not part of PyTorch's
//     standard interface. This may only partially catch up the stream and may
//     happen asynchronously relative to the user's main thread. Transfers are
//     not awaited during a background materialization, unless there is a data
//     dependency on them.
//   - Synchronization events: this is the analogue to a CUDA event, creating an
//     observable "snapshot" which forces all prior snapshots, all prior
//     transfers, and all prior dispatched ops to complete.
class StreamState {
 public:
  // Records on the events queue that a new c10::DataPtr referencing the given
  // DeviceBufferRef has been created.
  void RecordNewDataPtrCreated(const DeviceBufferRef& device_buffer_ref) {
    // Placeholders and empty tensors are never inserted into the map.
    if (device_buffer_ref.is_placeholder() || device_buffer_ref.is_empty()) {
      return;
    }
    absl::MutexLock lock(mu_);
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
    absl::MutexLock lock(mu_);
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

  // Clears the events queue as much as possible without risking deadlocks.
  void Clear() {
    {
      absl::MutexLock lock(mu_);
      live_nodes_.clear();
      async_device_to_host_futures_.clear();
      async_host_to_device_refs_.clear();
      background_materializations_.clear();

      // Explicitly do NOT fully clear the deferred ops queue.
      // Deferred ops may have side effects; clearing them could result in
      // deadlocks or errors later on.
      // If deferred ops are side-effect free and all Tensors using them are
      // dropped, they'll be weak pointers; we can filter these out using a
      // rotation over the deque to avoid reallocation.
      const size_t n = deferred_ops_.size();
      for (auto i = 0; i < n; ++i) {
        DeferredOpEvent front = std::move(deferred_ops_.front());
        deferred_ops_.pop_front();
        if (!front.is_dead()) {
          deferred_ops_.push_back(std::move(front));
        }
      }
    }
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

    // Returns true if the DeferredOpEvent is dead code.
    bool is_dead() const {
      return (!side_effects && no_side_effects.expired());
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
      absl::MutexLock lock(mu_);
      deferred_ops_.push_back(std::move(*event));
    }
  }

  void RecordAsyncHostToDevice(const DeviceBufferRef& device_buffer_ref) {
    absl::MutexLock lock(mu_);
    async_host_to_device_refs_.push_back(device_buffer_ref);
  }

  void RecordAsyncDeviceToHost(xla::Future<void>&& to_literal_future) {
    absl::MutexLock lock(mu_);
    async_device_to_host_futures_.push_back(std::move(to_literal_future));
  }

  // Partitions the deferred ops queue for a background materialization.
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
  // modify the stream state.
  std::vector<DeferredOpEvent> TakeBackgroundEvents(
      const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
          nodes_to_materialize) {
    if (nodes_to_materialize.empty()) {
      return {};
    }

    absl::MutexLock lock(mu_);
    std::deque<DeferredOpEvent> retained_ops;

    // Pop off the back of the queue until we see a node to materialize.
    while (!deferred_ops_.empty()) {
      auto op_list = deferred_ops_.back().lock();
      if (op_list && nodes_to_materialize.contains(op_list.get())) {
        // Found the last event in deferred_ops_ which needs to be materialized.
        background_materializations_.push_back(op_list);
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
      ABSL_VLOG(1) << "[StreamState::TakeBackgroundEvents] None of "
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
      ABSL_VLOG(1)
          << "[StreamState::TakeBackgroundEvents] The last node to materialize "
             "was the last deferred op. Returning the entire "
             "queue.\nReturning all "
          << result.size() << " DeferredOpEvents.";
    } else {
      ABSL_VLOG(1)
          << "[StreamState::TakeBackgroundEvents] Partitioned deferred ops "
             "queue.\nRetaining the last "
          << retained_ops.size()
          << " DeferredOpEvents, and returning the first " << result.size()
          << " DeferredOpEvents.";
      std::swap(deferred_ops_, retained_ops);
    }
    return result;
  }

  // Data that is necessary to construct the Traversal execution plan for a
  // new EventSnapshot. This clears the StreamState, only potentially retaining
  // a copy of the async_device_to_host_future in the retained future.
  struct EventSnapshotData {
    absl::flat_hash_map<SharedDeviceBufferList, int64_t> live_nodes;
    std::vector<DeferredOpEvent> ops_to_execute;
    xla::Future<void> new_future = xla::Future<void>(absl::OkStatus());
    absl_nullable std::shared_ptr<DeviceBufferList> new_dummy_list = nullptr;
  };

  EventSnapshotData TakeSnapshot() {
    absl::MutexLock lock(mu_);

    // Merge all device-to-host futures into a single future.
    // If this is not a trivial ready future, retain a copy of it as the only
    // value in async_device_to_host_futures_ so future snapshots can also
    // wait on it.
    std::vector<xla::Future<void>> d2h_futures =
        std::move(async_device_to_host_futures_);
    PruneCompletedFutures(d2h_futures);
    xla::Future<void> new_future;
    if (d2h_futures.empty()) {
      new_future = xla::Future<void>(absl::OkStatus());
    } else if (d2h_futures.size() == 1) {
      new_future = std::move(d2h_futures[0]);
      async_device_to_host_futures_.push_back(new_future);  // intentional copy
    } else {
      new_future = xla::JoinFutures(d2h_futures);
      async_device_to_host_futures_.push_back(new_future);  // intentional copy
    }

    // Make sure the snapshot will wait for all async host-to-device transfers
    // to complete.
    std::vector<DeviceBufferRef> wait_for =
        std::move(async_host_to_device_refs_);

    // Check if there are any deferred ops that need to be materialized.
    const bool nothing_to_materialize =
        (deferred_ops_.empty() ||
         std::all_of(
             deferred_ops_.begin(), deferred_ops_.end(),
             [](const DeferredOpEvent& event) { return event.is_dead(); }));

    // Create the device-side-awaitable dummy list for this new event, including
    // both any new work to execute and any background materializations to wait
    // on.
    std::vector<DeferredOpEvent> ops_to_execute;
    absl_nullable std::shared_ptr<DeviceBufferList> new_dummy_list;
    if (nothing_to_materialize) {
      // There is no additional real execution to start to reach this snapshot;
      // all ops (may be none) are already being handled by background
      // materializations. We just need to wait for them to complete.
      for (const auto& background_event : background_materializations_) {
        if (auto list = background_event.lock()) {
          for (auto i = 0; i < list->size(); ++i) {
            auto ref_or = DeviceBufferRef::Create(list, i);
            if (!ref_or.ok()) {
              continue;
            }
            wait_for.push_back(std::move(*ref_or));
          }
        }
      }
    } else {
      // If there is any new work to initiate, then we don't need to explicitly
      // wait on background materializations; each stream is executed in
      // single-file so the new ops_to_execute will intrinsically wait on any
      // prior background materializations on this stream.
      ops_to_execute.insert(ops_to_execute.end(),
                            std::make_move_iterator(deferred_ops_.begin()),
                            std::make_move_iterator(deferred_ops_.end()));
    }
    deferred_ops_.clear();
    background_materializations_.clear();

    // If there is any device-side event to await (background materializations,
    // new ops_to_execute, or async host-to-device transfers) then the last
    // thing to execute will be the dummy list.
    if (!wait_for.empty() || !ops_to_execute.empty()) {
      new_dummy_list = CreateNoOpDependency(wait_for);
      ops_to_execute.push_back(DeferredOpEvent{.side_effects = new_dummy_list});
      wait_for.clear();
    }

    if (ops_to_execute.empty() && new_future.IsKnownReady()) {
      // If there are no device-side events (no background materializations, no
      // new ops_to_execute, and no async host-to-device transfers) then
      // ops_to_execute is empty.
      // If there are no host-to-device transfers to await, then new_future will
      // be known ready.
      // If both are true, then the stream is already caught up and we can
      // return without creating a new event snapshot.
      live_nodes_.clear();
      return EventSnapshotData{};
    }

    return EventSnapshotData{
        .live_nodes = std::move(live_nodes_),
        .ops_to_execute = std::move(ops_to_execute),
        .new_future = std::move(new_future),
        .new_dummy_list = std::move(new_dummy_list),
    };
  }

 private:
  absl::Mutex mu_;

  // Hold a strong pointer to the DeviceBufferList as the key; as long as there
  // is a live DataPtr, the DeviceBufferList can't be dropped.
  absl::flat_hash_map<SharedDeviceBufferList, int64_t> live_nodes_
      ABSL_GUARDED_BY(mu_);

  // Any DeferredOps which have been created but not yet materialized.
  // Non-side-effecting ops are allowed to degrade into dead code if all
  // DataPtrs using them expire; side-effecting ops are retained permanently to
  // ensure execution.
  std::deque<DeferredOpEvent> deferred_ops_ ABSL_GUARDED_BY(mu_);

  // Because completion of a device-to-host transfer has no device-side
  // effects, anything that needs to await it necessarily must have a host-side
  // wait at some point. We track this using futures.
  std::vector<xla::Future<void>> async_device_to_host_futures_
      ABSL_GUARDED_BY(mu_);

  // Any DeviceBufferRefs that are being populated by async host-to-device
  // transfers since the last EventSnapshot was recorded.
  // These always need to be explicitly synced by the user for safety, as they
  // can't free the CPU buffer until the accelerator has finished reading it.
  std::vector<DeviceBufferRef> async_host_to_device_refs_ ABSL_GUARDED_BY(mu_);

  // Any DeferredOps that are being materialized asynchronously in the
  // background.
  // Since they're already being materialized we don't need to hold onto strong
  // pointers; if they haven't finished executing then the materialization or
  // execution thread will have a strong pointer to them, so the only way they
  // can expire is if they're done materializing and nothing else is using
  // them, at which point we no longer care.
  std::vector<std::weak_ptr<DeviceBufferList>> background_materializations_
      ABSL_GUARDED_BY(mu_);
};

// A singleton that routes calls to the appropriate stream state.
class StreamStates {
 public:
  static StreamStates& GetInstance() {
    static absl::NoDestructor<StreamStates> instance;
    return *instance;
  }

  // Returns the StreamState for the given device and stream ID.
  // If the stream ID is 0, returns the default stream state for the device.
  std::shared_ptr<StreamState> GetOrCreateStreamState(
      c10::DeviceIndex device_index, c10::StreamId stream_id) {
    ABSL_CHECK_GE(device_index, 0);            // CRASH_OK
    ABSL_CHECK_LT(device_index, kMaxDevices);  // CRASH_OK
    auto& device_entry = devices_[device_index];

    ABSL_CHECK_GE(stream_id, 0);  // CRASH_OK
    if (stream_id == 0) {
      // Torch expects that a stream ID of 0 is a special value that indicates
      // the default stream: see commenting in c10/core/Stream.h.
      // Since this always exists, we can just return it without locking.
      return device_entry.default_stream;
    }
    absl::MutexLock lock(device_entry.mu);
    if (stream_id - 1 >= device_entry.streams.size()) {
      device_entry.streams.resize(stream_id - 1);
    }
    return device_entry.streams[stream_id - 1];
  }

  // Returns all stream states for the given device, including the default
  // stream. The default stream is always the first element (index 0) of the\
  // returned vector.
  std::vector<std::shared_ptr<StreamState>> GetDeviceStreamStates(
      c10::DeviceIndex device_index) {
    ABSL_CHECK_GE(device_index, 0);            // CRASH_OK
    ABSL_CHECK_LT(device_index, kMaxDevices);  // CRASH_OK
    auto& device_entry = devices_[device_index];
    std::vector<std::shared_ptr<StreamState>> stream_states;
    absl::MutexLock lock(device_entry.mu);
    stream_states.reserve(device_entry.streams.size() + 1);
    stream_states.push_back(device_entry.default_stream);
    for (const auto& stream_state : device_entry.streams) {
      stream_states.push_back(stream_state);
    }
    return stream_states;
  }

  void ClearAll() {
    for (auto& device_entry : devices_) {
      device_entry.default_stream->Clear();
      absl::MutexLock lock(device_entry.mu);
      for (auto& stream_state : device_entry.streams) {
        stream_state->Clear();
      }
    }
  }

 private:
  struct DeviceEntry {
    // Every device always has a default stream, so we don't need to guard
    // access to it with a mutex.
    std::shared_ptr<StreamState> default_stream;

    // Every time we allocate a new stream it might cause a dynamic vector
    // resize.
    absl::Mutex mu;
    // TODO: this vector is never cleared and will grow without bound if
    // users create new streams repeatedly using `torch.Stream("tpu")`.
    // Options:
    //   - Only support creating streams through `torch.tpu.Stream`, not the
    //     generic `torch.Stream`
    //   - Allocate a fixed number of streams per c10::Device and error/crash if
    //     more than this are created
    //   - Use a round-robin assignment over a fixed number of streams,
    //     similar to what XPU does (see torch/c10/xpu/XPUStream.h)
    //   - Find some way to safely clear streams when they are done
    std::vector<std::shared_ptr<StreamState>> streams ABSL_GUARDED_BY(mu);
  };

  std::array<DeviceEntry, kMaxDevices> devices_;
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

// Helper function for PrepareMaterializationTraversals.
// Pushes the device buffer list to the execution order (unless it is empty),
// and updates the defined node map.
void ProcessDeferredOpEvent(
    SharedDeviceBufferList&& device_buffer_list, const DeferredOp& deferred_op,
    const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        nodes_to_materialize_set,
    std::vector<SharedDeviceBufferList>& execution_order,
    DefinedNodeMap& defined_node_map,
    absl::flat_hash_set<const DeviceBufferList*>& materialized_empty_ops,
    absl::flat_hash_map<SharedDeviceBufferList, int64_t>* absl_nullable
        maybe_live_nodes) {
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

    if (auto input_def_it =
            defined_node_map.find(input.device_buffer_list().get());
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
               materialized_empty_ops.insert(input.device_buffer_list().get())
                   .second) {
      // The first time an empty tensor is read by a later op, we insert
      // it into the execution order, as an output if it is live.
      if (input.device_buffer_list()->is_stale()) {
        ABSL_VLOG(3) << "[ProcessDeferredOpEvent] Stale empty buffer "
                     << input.device_buffer_list().get() << " is read by "
                     << device_buffer_list.get() << "("
                     << ToString(deferred_op.op_name())
                     << ").\nInserting into execution order.";
        defined_node_map[input.device_buffer_list().get()] = OpUsage::kUsed;
      } else {
        ABSL_VLOG(3) << "[ProcessDeferredOpEvent] Live empty buffer "
                     << input.device_buffer_list().get() << " is read by "
                     << device_buffer_list.get() << "("
                     << ToString(deferred_op.op_name())
                     << ").\nInserting as output.";
        defined_node_map[input.device_buffer_list().get()] = OpUsage::kOutput;
      }
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
  } else if ((maybe_live_nodes == nullptr
                  ? !device_buffer_list->is_stale()
                  : maybe_live_nodes->contains(device_buffer_list))) {
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

absl::StatusOr<std::vector<absl_nonnull std::unique_ptr<Traversal>>>
CreateExecutionPlan(
    std::vector<StreamState::DeferredOpEvent> deferred_op_events,
    absl::Span<const SharedDeviceBufferList> nodes_to_materialize,
    const absl::flat_hash_set<const DeviceBufferList*>&
        nodes_to_materialize_set,
    absl::flat_hash_map<SharedDeviceBufferList, int64_t>* absl_nullable
        maybe_live_nodes = nullptr) {
  std::vector<absl_nonnull std::unique_ptr<Traversal>> traversals;

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
  absl::flat_hash_set<const DeviceBufferList*> materialized_empty_ops;

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
        ABSL_VLOG(2) << "[CreateExecutionPlan] Split out "
                     << maybe_traversal->execution_order().size()
                     << " nodes before SplitBefore buffer "
                     << device_buffer_list.get() << "("
                     << ToString(deferred_op->op_name()) << ").";
        traversals.push_back(std::move(maybe_traversal));
      }
    }

    ProcessDeferredOpEvent(std::move(device_buffer_list), *deferred_op,
                           nodes_to_materialize_set, execution_order,
                           defined_node_map, materialized_empty_ops,
                           maybe_live_nodes);

    if (IsSplitAfter(split_mode)) {
      TT_ASSIGN_OR_RETURN(
          auto maybe_traversal,
          FinishTraversal(execution_order, defined_node_map, output_nodes));
      if (maybe_traversal != nullptr) {
        ABSL_VLOG(2) << "[CreateExecutionPlan] Split out "
                     << maybe_traversal->execution_order().size()
                     << " nodes after SplitAfter buffer "
                     << device_buffer_list.get() << "("
                     << ToString(deferred_op->op_name()) << ").";
        traversals.push_back(std::move(maybe_traversal));
      }
    }
  }

  // If any node to materialize is an explicitly materialized empty tensor and
  // we didn't see it in the execution order (possibly because we skipped it
  // on an earlier call to CreateExecutionPlan), add it to the
  // last traversal.
  for (const auto& node : nodes_to_materialize) {
    if (node->is_empty() && materialized_empty_ops.insert(node.get()).second &&
        node->is_deferred()) {
      execution_order.push_back(node);
      defined_node_map[node.get()] = OpUsage::kOutput;
    }
  }

  TT_ASSIGN_OR_RETURN(
      auto maybe_traversal,
      FinishTraversal(execution_order, defined_node_map, output_nodes));
  if (maybe_traversal != nullptr) {
    ABSL_VLOG(2) << "[CreateExecutionPlan] Final traversal has size "
                 << maybe_traversal->execution_order().size()
                 << ", ending with node "
                 << maybe_traversal->execution_order().back().get();
    traversals.push_back(std::move(maybe_traversal));
  }
  ABSL_VLOG(1) << "[CreateExecutionPlan] Created " << traversals.size()
               << " traversals to materialize " << nodes_to_materialize.size()
               << " nodes.";
  return traversals;
}

}  // namespace

void RecordNewDataPtrCreated(const DeviceBufferRef& device_buffer_ref) {
  const c10::DeviceIndex device_index = device_buffer_ref.device_index();
  const c10::StreamId stream_id = device_buffer_ref.stream_id();
  StreamStates::GetInstance()
      .GetOrCreateStreamState(device_index, stream_id)
      ->RecordNewDataPtrCreated(device_buffer_ref);
}

void RecordDataPtrDestroyed(const DeviceBufferRef& device_buffer_ref) {
  const c10::DeviceIndex device_index = device_buffer_ref.device_index();
  const c10::StreamId stream_id = device_buffer_ref.stream_id();
  StreamStates::GetInstance()
      .GetOrCreateStreamState(device_index, stream_id)
      ->RecordDataPtrDestroyed(device_buffer_ref);
}

void RecordDeferredOpCreated(const SharedDeviceBufferList& device_buffer_list) {
  const c10::DeviceIndex device_index = device_buffer_list->device_index();
  const c10::StreamId stream_id = device_buffer_list->stream_id();
  StreamStates::GetInstance()
      .GetOrCreateStreamState(device_index, stream_id)
      ->RecordDeferredOpCreated(device_buffer_list);
}

void ClearAllStreams() { StreamStates::GetInstance().ClearAll(); }

void RecordAsyncHostToDevice(const DeviceBufferRef& device_buffer_ref) {
  const c10::DeviceIndex device_index = device_buffer_ref.device_index();
  const c10::StreamId stream_id = device_buffer_ref.stream_id();
  auto stream_state = StreamStates::GetInstance().GetOrCreateStreamState(
      device_index, stream_id);
  stream_state->RecordAsyncHostToDevice(device_buffer_ref);
}

void RecordAsyncDeviceToHost(xla::Future<void> to_literal_future) {
  const c10::DeviceIndex device_index = GetCurrentDeviceIndex();
  const c10::StreamId stream_id = GetCurrentStreamId(device_index);
  auto stream_state = StreamStates::GetInstance().GetOrCreateStreamState(
      device_index, stream_id);
  stream_state->RecordAsyncDeviceToHost(std::move(to_literal_future));
}

absl::StatusOr<std::vector<absl_nonnull std::unique_ptr<Traversal>>>
PrepareMaterializationTraversals(
    absl::Span<const SharedDeviceBufferList> nodes_to_materialize) {
  std::vector<absl_nonnull std::unique_ptr<Traversal>> traversals;
  if (nodes_to_materialize.empty()) {
    return traversals;
  }

  const c10::DeviceIndex device_index = nodes_to_materialize[0]->device_index();
  const c10::StreamId stream_id = nodes_to_materialize[0]->stream_id();

  absl::flat_hash_set<const DeviceBufferList*> nodes_to_materialize_set;
  for (const auto& node : nodes_to_materialize) {
    TT_RET_CHECK(node->device_index() == device_index, error::kInvalidArgument)
        << "all nodes to materialize must be on the same device and stream";
    TT_RET_CHECK(node->stream_id() == stream_id, error::kInvalidArgument)
        << "all nodes to materialize must be on the same device and stream";
    nodes_to_materialize_set.insert(node.get());
  }

  // Do the partitioning of the deferred ops queue.
  std::vector<StreamState::DeferredOpEvent> deferred_op_events;
  {
    auto stream_state = StreamStates::GetInstance().GetOrCreateStreamState(
        device_index, stream_id);
    deferred_op_events =
        stream_state->TakeBackgroundEvents(nodes_to_materialize_set);
  }
  return CreateExecutionPlan(std::move(deferred_op_events),
                             nodes_to_materialize, nodes_to_materialize_set);
}

absl::StatusOr<EventSnapshot::RecordedEventSnapshot> EventSnapshot::Record(
    c10::DeviceIndex device_index, int64_t stream_id) {
  StreamState::EventSnapshotData snapshot_data;
  {
    auto stream_state = StreamStates::GetInstance().GetOrCreateStreamState(
        device_index, stream_id);
    snapshot_data = stream_state->TakeSnapshot();
  }
  std::vector<absl_nonnull std::unique_ptr<Traversal>> traversals;
  if (!snapshot_data.ops_to_execute.empty()) {
    TT_ASSIGN_OR_RETURN(
        traversals, CreateExecutionPlan(std::move(snapshot_data.ops_to_execute),
                                        /*nodes_to_materialize=*/{},
                                        /*nodes_to_materialize_set=*/{},
                                        &snapshot_data.live_nodes));
  }
  absl_nonnull std::shared_ptr<EventSnapshot> event_snapshot =
      std::make_shared<EventSnapshot>(device_index, stream_id,
                                      std::move(snapshot_data.new_future),
                                      std::move(snapshot_data.new_dummy_list));
  return EventSnapshot::RecordedEventSnapshot{
      .traversals = std::move(traversals),
      .event_snapshot = std::move(event_snapshot),
  };
}

absl::Status EventSnapshot::Wait() const {
  TT_RETURN_IF_ERROR(async_device_to_host_future_.Await());
  if (dummy_list_ != nullptr) {
    TT_RETURN_IF_ERROR(dummy_list_->Synchronize());
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<EventSnapshot::RecordedEventSnapshot>>
RecordDeviceSnapshot(c10::DeviceIndex device_index) {
  TT_RET_CHECK(0 <= device_index && device_index < kMaxDevices,
               error::kInvalidArgument)
      << "device index out of bounds: " << device_index;
  auto device_streams =
      StreamStates::GetInstance().GetDeviceStreamStates(device_index);
  std::vector<EventSnapshot::RecordedEventSnapshot> recorded_snapshots;
  for (c10::StreamId stream_id = 0; stream_id < device_streams.size();
       ++stream_id) {
    const auto& stream_state = device_streams[stream_id];
    auto snapshot_data = stream_state->TakeSnapshot();
    if (!snapshot_data.ops_to_execute.empty()) {
      TT_ASSIGN_OR_RETURN(
          auto traversals,
          CreateExecutionPlan(std::move(snapshot_data.ops_to_execute),
                              /*nodes_to_materialize=*/{},
                              /*nodes_to_materialize_set=*/{},
                              &snapshot_data.live_nodes));
      absl_nonnull std::shared_ptr<EventSnapshot> event_snapshot =
          std::make_shared<EventSnapshot>(
              device_index, stream_id, std::move(snapshot_data.new_future),
              std::move(snapshot_data.new_dummy_list));
      recorded_snapshots.push_back(EventSnapshot::RecordedEventSnapshot{
          .traversals = std::move(traversals),
          .event_snapshot = std::move(event_snapshot),
      });
    }
  }
  return recorded_snapshots;
}

absl::StatusOr<bool> EventSnapshot::Query() const {
  TT_RET_CHECK(async_device_to_host_future_.IsValid(), error::kInternal)
      << "event snapshot contains an invalid future";
  if (!async_device_to_host_future_.IsReady()) {
    // Still waiting on an async host-to-device transfer.
    return false;
  }
  if (dummy_list_ == nullptr) {
    // No device-side events to await and all host-side events have completed.
    return true;
  }
  if (!dummy_list_->is_materialized()) {
    // Host-side events have not yet completed.
    return false;
  }
  TT_ASSIGN_OR_RETURN(auto pjrt_buffer, dummy_list_->AwaitBuffer(0));
  return pjrt_buffer->GetReadyFuture().IsReady();
}

absl::StatusOr<std::vector<EventSnapshot::RecordedEventSnapshot>>
RecordGlobalSnapshot() {
  std::vector<EventSnapshot::RecordedEventSnapshot> recorded_snapshots;
  for (c10::DeviceIndex device_index = 0; device_index < kMaxDevices;
       ++device_index) {
    TT_ASSIGN_OR_RETURN(auto device_snapshots,
                        RecordDeviceSnapshot(device_index));
    recorded_snapshots.insert(recorded_snapshots.end(),
                              std::make_move_iterator(device_snapshots.begin()),
                              std::make_move_iterator(device_snapshots.end()));
  }
  return recorded_snapshots;
}

}  // namespace torch_tpu
