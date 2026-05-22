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

#include "torch_tpu/experimental/eager/materialize_new.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "torch_tpu/common/compilation_spec.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/eager_mode.h"
#include "torch_tpu/eager/materialize_common.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/eager/traversal.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/traceme.h"

ABSL_FLAG(bool, torch_tpu_internal_enable_new_materialization, false,
          "Enable new materialization algorithm (experimental).");

namespace torch_tpu {

namespace {

class MaterializationWorker {
 public:
  static MaterializationWorker& GetInstance() {
    static absl::NoDestructor<MaterializationWorker> worker;
    return *worker;
  }

  // Callback to be invoked upon dispatching a new deferred op.
  absl::Status OnNewOpDispatch(
      const SharedDeviceBufferList& device_buffer_list);

  // Materialize all ops dispatched so far.
  absl::Status Materialize(
      absl::Span<const SharedDeviceBufferList> nodes_to_materialize,
      MaterializationReason reason);

  absl::Status BlockOnPendingMaterializations();

  void Reset();

  void Exit();

 private:
  friend class absl::NoDestructor<MaterializationWorker>;
  static constexpr int kNumQueues = 2;

  struct QueueState {
    std::vector<SharedDeviceBufferList> queue;
    std::vector<SharedDeviceBufferList> nodes_to_materialize;
    MaterializationReason reason;
    bool is_full;
    QueueState() { Reset(); }
    void Reset() {
      is_full = false;
      queue.clear();
      nodes_to_materialize.clear();
      reason = MaterializationReason::kUnknown;
    }
  };

  mutable absl::Mutex mutex_;
  int dispatch_queue_id_ ABSL_GUARDED_BY(mutex_);
  int execution_queue_id_ ABSL_GUARDED_BY(mutex_);
  QueueState queues_[kNumQueues] ABSL_GUARDED_BY(mutex_);
  bool must_exit_ ABSL_GUARDED_BY(mutex_);

  absl::Status last_status_ ABSL_GUARDED_BY(mutex_) = absl::OkStatus();
  std::thread execution_thread_;

  MaterializationWorker() {
    Reset();
    execution_thread_ = std::thread([this]() { ThreadLoop(); });
  }

  ~MaterializationWorker() { Exit(); }

  absl::Status GetLastStatus() const {
    absl::MutexLock lock(mutex_);
    return last_status_;
  }

  void IncrementQueueId(int& id) { id = (id + 1) % kNumQueues; }

  bool IsReadyToDispatch() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return !queues_[dispatch_queue_id_].is_full;
  }

  bool IsExecutionThreadCaughtUp() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return (execution_queue_id_ == dispatch_queue_id_) &&
           !queues_[dispatch_queue_id_].is_full;
  }

  bool IsReadyToExitOrExecute() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return must_exit_ || queues_[execution_queue_id_].is_full;
  }

  void ThreadLoop();

  std::vector<absl::Span<const SharedDeviceBufferList>> SplitQueueIntoRegions(
      const std::vector<SharedDeviceBufferList>& queue) const;

  absl::Status MaterializeQueue(
      const std::vector<SharedDeviceBufferList>& queue,
      const std::vector<SharedDeviceBufferList>& nodes_to_materialize,
      MaterializationReason reason);
};

absl::Status MaterializationWorker::OnNewOpDispatch(
    const SharedDeviceBufferList& device_buffer_list) {
  tsl::profiler::TraceMe t("Worker_OnNewOpDispatch");
  TT_RETURN_IF_ERROR(GetLastStatus());
  absl::MutexLock lock(mutex_);
  // Wait if the current dispatch queue is being processed.
  mutex_.Await(
      absl::Condition(this, &MaterializationWorker::IsReadyToDispatch));
  queues_[dispatch_queue_id_].queue.push_back(device_buffer_list);
  return absl::OkStatus();
}

absl::Status MaterializationWorker::Materialize(
    absl::Span<const SharedDeviceBufferList> nodes_to_materialize,
    MaterializationReason reason) {
  tsl::profiler::TraceMe t("Worker_Materialize");
  TT_RETURN_IF_ERROR(GetLastStatus());
  {
    absl::MutexLock lock(mutex_);
    // Wait if the current dispatch queue is being processed.
    mutex_.Await(
        absl::Condition(this, &MaterializationWorker::IsReadyToDispatch));

    // Deterministically prune stale nodes from the current dispatch queue.
    // Because this runs synchronously with the Python thread (which called
    // Materialize), the GC state is fully deterministic here. This allows us to
    // safely dead-code-eliminate completely unused tensors without causing XLA
    // cache thrashing in the background worker.
    auto& current_queue = queues_[dispatch_queue_id_].queue;
    current_queue.erase(
        std::remove_if(current_queue.begin(), current_queue.end(),
                       [](const SharedDeviceBufferList& node) {
                         return node->is_stale();
                       }),
        current_queue.end());

    if (current_queue.empty() && nodes_to_materialize.empty()) {
      return absl::OkStatus();
    }
    ABSL_CHECK(  // CRASH_OK
        queues_[dispatch_queue_id_].nodes_to_materialize.empty());
    queues_[dispatch_queue_id_].nodes_to_materialize.insert(
        queues_[dispatch_queue_id_].nodes_to_materialize.end(),
        nodes_to_materialize.begin(), nodes_to_materialize.end());
    queues_[dispatch_queue_id_].reason = reason;
    queues_[dispatch_queue_id_].is_full = true;
    IncrementQueueId(dispatch_queue_id_);
  }
  return absl::OkStatus();
}

absl::Status MaterializationWorker::BlockOnPendingMaterializations() {
  tsl::profiler::TraceMe t("Worker_BlockOnPendingMaterializations");
  ABSL_VLOG(1) << ">>> BlockOnPendingMaterializations";
  TT_RETURN_IF_ERROR(GetLastStatus());

  {
    absl::MutexLock lock(mutex_);
    // Wait for the execution thread to be caught up.
    mutex_.Await(absl::Condition(
        this, &MaterializationWorker::IsExecutionThreadCaughtUp));
  }

  ABSL_VLOG(1) << ">>> BlockOnPendingMaterializations DONE";
  return GetLastStatus();
}

void MaterializationWorker::Reset() {
  absl::MutexLock lock(mutex_);
  mutex_.Await(
      absl::Condition(this, &MaterializationWorker::IsExecutionThreadCaughtUp));
  dispatch_queue_id_ = 0;
  execution_queue_id_ = 0;
  for (auto& queue : queues_) {
    queue.Reset();
  }
  must_exit_ = false;
  last_status_ = absl::OkStatus();
}

void MaterializationWorker::Exit() {
  ABSL_VLOG(1) << ">>> MaterializationWorker::Exit";
  {
    absl::MutexLock lock(mutex_);
    must_exit_ = true;
  }
  if (execution_thread_.joinable()) {
    execution_thread_.join();
  }
}

void MaterializationWorker::ThreadLoop() {
  ABSL_VLOG(1) << ">>> MaterializationWorker::ThreadLoop";
  while (true) {
    int current_execution_id;
    std::vector<SharedDeviceBufferList> current_queue;
    std::vector<SharedDeviceBufferList> nodes_to_materialize;
    MaterializationReason current_reason;

    {
      absl::MutexLock lock(
          mutex_, absl::Condition(
                      this, &MaterializationWorker::IsReadyToExitOrExecute));
      if (must_exit_) {
        break;
      }

      // Claim the current queue for processing.  We can safely move it out of
      // the shared array because the dispatch thread is blocked from accessing
      // this index by queue_full_[execution_queue_id_] == true.
      ABSL_CHECK(queues_[execution_queue_id_].is_full);  // CRASH_OK
      current_execution_id = execution_queue_id_;
      std::swap(current_queue, queues_[execution_queue_id_].queue);
      std::swap(nodes_to_materialize,
                queues_[execution_queue_id_].nodes_to_materialize);
      current_reason = queues_[execution_queue_id_].reason;
    }

    // Materialize the claimed queue without holding the mutex.
    absl::Status status =
        MaterializeQueue(current_queue, nodes_to_materialize, current_reason);

    {
      absl::MutexLock lock(mutex_);
      if (!status.ok()) {
        ABSL_LOG(ERROR) << "Failed to materialize queue: " << status;
        last_status_ = status;
      }

      // In order to avoid capacity reallocation in
      // queue_[execution_queue_id_]), put the swapped vector back and clear it,
      // then release the slod and move to the next one.
      ABSL_CHECK(current_execution_id == execution_queue_id_);  // CRASH_OK
      std::swap(current_queue, queues_[current_execution_id].queue);
      std::swap(nodes_to_materialize,
                queues_[execution_queue_id_].nodes_to_materialize);
      queues_[current_execution_id].queue.clear();
      queues_[execution_queue_id_].nodes_to_materialize.clear();
      queues_[current_execution_id].is_full = false;
      IncrementQueueId(execution_queue_id_);
    }
  }
}

bool IsSplitPoint(const SharedDeviceBufferList& current_op,
                  const SharedDeviceBufferList* next_op) {
  const auto* deferred_op = current_op->deferred_op();
  if (!deferred_op) {
    return false;
  }

  // TODO: Enabling this lowers the initial compilation time, but doesn't affect
  // the long-term step time.
#if 0
  if (deferred_op->has_been_executed()) {
    return true;
  }
#endif

  if (deferred_op->split_mode() == OpSplitMode::kSplitAfter ||
      deferred_op->split_mode() == OpSplitMode::kSplitBoth) {
    return true;
  }

  if (next_op) {
    if (const auto* next_deferred_op = (*next_op)->deferred_op();
        next_deferred_op) {
      OpSplitMode split_mode = next_deferred_op->split_mode();
      if (split_mode == OpSplitMode::kSplitBoth ||
          split_mode == OpSplitMode::kSplitBefore) {
        return true;
      }
    }
  }
  return false;
}

std::vector<absl::Span<const SharedDeviceBufferList>>
MaterializationWorker::SplitQueueIntoRegions(
    const std::vector<SharedDeviceBufferList>& queue) const {
  // Split the queue into multiple regions based on re-executed ops and
  // previously identified split points.
  std::vector<absl::Span<const SharedDeviceBufferList>> split_regions;

  auto begin = queue.begin();
  for (auto end = begin; end != queue.end(); ++end) {
    const SharedDeviceBufferList* next_op =
        (end + 1 != queue.end()) ? &(*(end + 1)) : nullptr;
    if (IsSplitPoint(*end, next_op)) {
      split_regions.emplace_back(&(*begin), std::distance(begin, end + 1));
      begin = end + 1;
    }
  }

  if (begin != queue.end()) {
    split_regions.emplace_back(&(*begin), std::distance(begin, queue.end()));
  }

  return split_regions;
}

absl::StatusOr<std::vector<SharedDeviceBufferList>> ExtractExecutionOrder(
    const std::vector<SharedDeviceBufferList>& queue,
    const std::vector<SharedDeviceBufferList>& nodes_to_materialize) {
  // Unfortunately the nodes that have been dispatched in `queue` are not all
  // the nodes that need to be scheduled. That's because view ops are not
  // dispatched, rather they show as deferred op inputs of other dispatched or
  // view ops, including as the nodes in `nodes_to_materialized`. In order to
  // extract a complete and properly sorted (by creation index) list of nodes to
  // dispatch we use the DFS search implemented in Traversal::Create() by
  // passing all nodes we know about as traversal outputs. Then we can return
  // the traversal's execution order and discard the other information.

  absl::flat_hash_set<SharedDeviceBufferList> unique_nodes;
  for (auto& node : nodes_to_materialize) {
    unique_nodes.insert(node);
  }
  for (auto& node : queue) {
    unique_nodes.insert(node);
  }

  // Sort nodes deterministically, so as to lead to identical traversal
  // creations across different workers.
  std::vector<SharedDeviceBufferList> sorted_nodes(unique_nodes.begin(),
                                                   unique_nodes.end());
  std::sort(sorted_nodes.begin(), sorted_nodes.end(), [](auto& n1, auto& n2) {
    return n1->creation_index() < n2->creation_index();
  });

  std::vector<DeviceBufferRef> traversal_outputs;
  for (auto& node : sorted_nodes) {
    for (size_t i = 0; i < node->size(); ++i) {
      TT_ASSIGN_OR_RETURN(auto output, DeviceBufferRef::Create(node, i));
      traversal_outputs.push_back(std::move(output));
    }
  }

  TT_ASSIGN_OR_RETURN(auto traversal, Traversal::Create(traversal_outputs));
  auto parts = traversal->IntoParts();
  return parts.execution_order;
}

// Compute the values used in each region and return a vector of sets, where
// each set i contains the values used in region i.
std::vector<absl::flat_hash_set<const DeviceBufferList*>> GetPerRegionUses(
    absl::Span<const absl::Span<const SharedDeviceBufferList>> regions) {
  auto num_regions = regions.size();
  std::vector<absl::flat_hash_set<const DeviceBufferList*>> uses(num_regions);
  for (auto i = 0; i < num_regions; ++i) {
    const auto& region = regions[i];
    auto& uses_ = uses[i];
    absl::flat_hash_set<const DeviceBufferList*> visited;
    for (const auto& n : region) {
      const auto* deferred_op = n->deferred_op();
      if (deferred_op) {
        for (const auto& input : deferred_op->inputs()) {
          // A "use" is ANY input to an operation in this region.
          auto* node = input.device_buffer_list().get();
          if (visited.insert(node).second) {
            uses_.insert(node);
          }
        }
      }
    }
  }

  return uses;
}

std::string ToString(const std::vector<SharedDeviceBufferList>& v) {
  std::ostringstream os;
  os << "[size: " << v.size() << "\n";
  for (auto& n : v) {
    os << n->DebugString() << "\n";
  }
  os << "]";
  return os.str();
}

absl::Status MaterializationWorker::MaterializeQueue(
    const std::vector<SharedDeviceBufferList>& queue,
    const std::vector<SharedDeviceBufferList>& nodes_to_materialize,
    MaterializationReason reason) {
  tsl::profiler::TraceMe t("Worker_MaterializeQueue");

  ABSL_VLOG(1) << ">>> MaterializationWorker::MaterializeQueue "
               << queue.size();

  ABSL_VLOG(1) << "Queue: " << ToString(queue);
  ABSL_VLOG(1) << "Nodes to Materialize: " << ToString(nodes_to_materialize);

  TT_ASSIGN_OR_RETURN(std::vector<SharedDeviceBufferList> execution_order,
                      ExtractExecutionOrder(queue, nodes_to_materialize));

  ABSL_VLOG(1) << "Execution Order: " << ToString(execution_order);

  auto regions = SplitQueueIntoRegions(execution_order);
  const auto num_regions = regions.size();

  // Launch compilations for the identified regions.
  auto compilation_mode = (GetEagerMode() == EagerMode::kDeferAndFuse)
                              ? CompilationMode::kFastRuntime
                              : CompilationMode::kFastCompile;

  std::vector<ExecutionTask> execution_tasks;
  execution_tasks.reserve(num_regions);

  auto per_region_uses = GetPerRegionUses(regions);

  absl::flat_hash_set<const DeviceBufferList*> explicit_targets;
  for (const auto& n : nodes_to_materialize) {
    explicit_targets.insert(n.get());
  }

  for (auto i = 0; i < num_regions; ++i) {
    const auto& region = regions[i];

    // For each region, we compute the desired outputs. Those are nodes that
    // have already executed in a previous region, or nodes that are used as
    // input in a subsequent region, or nodes that are defined in the current
    // region, but not used in any subsequent region and, hence, could be used
    // in the future.
    std::vector<SharedDeviceBufferList> region_outputs;
    for (const auto& n : region) {
      bool must_materialize_node = false;
      const auto* deferred_op = n->deferred_op();

      if (deferred_op && deferred_op->has_been_executed()) {
        // The node was already executed.
        must_materialize_node = true;
      } else if (explicit_targets.contains(n.get())) {
        // The node was explicitly requested by the caller.
        must_materialize_node = true;
      } else if (per_region_uses[i].find(n.get()) == per_region_uses[i].end()) {
        // The node is defined in the current region, but not used there.
        must_materialize_node = true;
      } else {
        for (auto j = i + 1; j < num_regions; ++j) {
          if (per_region_uses[j].find(n.get()) != per_region_uses[j].end()) {
            // The node is defined in the current region and used in a next
            // region.
            must_materialize_node = true;
            break;
          }
        }
      }
      if (must_materialize_node) {
        region_outputs.push_back(n);
      }
    }

    // Sort `region_outputs` deterministically so as to ensure identical
    // traversal creations across different workers.
    std::sort(region_outputs.begin(), region_outputs.end(),
              [](const auto& a, const auto& b) {
                return a->creation_index() < b->creation_index();
              });
    // Remove any duplicates.
    region_outputs.erase(
        std::unique(region_outputs.begin(), region_outputs.end()),
        region_outputs.end());

    // We create a traversal for the given region because that's the only way to
    // compile the DeferredOps in the region. Note that this traversal
    // constructor doesn't perform a DFS.
    TT_ASSIGN_OR_RETURN(auto traversal, Traversal::CreateFromExecutionOrder(
                                            region, region_outputs));
    // Don't launch kernels if this is part of TorchTPU tracing of an FX graph,
    // which is indicated by placeholder inputs.
    bool has_placeholder = false;
    for (const auto& arg : traversal->arguments()) {
      if (arg.state() == DeviceBufferRefState::kPlaceholder) {
        has_placeholder = true;
        break;
      }
    }
    if (has_placeholder) {
      ABSL_VLOG(1)
          << "[MaterializationWorker] Skipping region "
          << absl::StrCat(traversal->GetCacheKey(compilation_mode))
          << " because it depends on a placeholder (likely traced by Dynamo).";
      continue;
    }
    ABSL_VLOG(1) << "==== TRAVERSAL ===\n" << traversal->DebugString();
    // TODO(cbasile): support bounded dynamism, requires an MLIR context
    TT_ASSIGN_OR_RETURN(
        auto execution_task,
        ExecutionTask::FromTraversalWithLogging(std::move(traversal), reason));
    execution_tasks.push_back(std::move(execution_task));
  }

  // Launch compiled kernels.
  for (auto& execution_task : execution_tasks) {
    TT_RETURN_IF_ERROR(execution_task.Run());
  }

  return absl::OkStatus();
}

}  // namespace

absl::Status OnNewOpDispatch(const SharedDeviceBufferList& device_buffer_list) {
  ABSL_VLOG(1) << ">>> OnNewOpDispatch " << device_buffer_list->DebugString();
  return MaterializationWorker::GetInstance().OnNewOpDispatch(
      device_buffer_list);
}

absl::Status MaterializeImplNew(
    absl::Span<const SharedDeviceBufferList> nodes_to_materialize,
    MaterializationReason reason) {
  ABSL_VLOG(1) << ">>> MaterializeImplNew " << nodes_to_materialize.size();
  return MaterializationWorker::GetInstance().Materialize(nodes_to_materialize,
                                                          reason);
}

absl::Status BlockOnPendingMaterializations() {
  ABSL_VLOG(1) << ">>> BlockOnPendingMaterializations";
  return MaterializationWorker::GetInstance().BlockOnPendingMaterializations();
}

void ResetNewMaterializationState() {
  ABSL_VLOG(1) << ">>> ResetNewMaterializationState";
  return MaterializationWorker::GetInstance().Reset();
}

void ShutDownNewMaterializationState() {
  ABSL_VLOG(1) << ">>> ShutDownNewMaterializationState";
  MaterializationWorker::GetInstance().Exit();
}

}  // namespace torch_tpu
