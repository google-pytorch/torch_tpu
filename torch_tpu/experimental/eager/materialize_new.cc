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
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
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
#include "mlir/IR/MLIRContext.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/eager_mode.h"
#include "torch_tpu/eager/materialize_common.h"
#include "torch_tpu/eager/split_traversal.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/eager/traversal.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/xla_data.pb.h"

ABSL_FLAG(bool, torch_tpu_internal_enable_new_materialization, false,
          "Enable new materialization algorithm (experimental).");

namespace torch_tpu {

namespace {

absl::StatusOr<std::unique_ptr<Traversal>> ExtractTraversal(
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

  return Traversal::Create(traversal_outputs);
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

absl::Status MaterializeQueue(
    const std::vector<SharedDeviceBufferList>& queue,
    const std::vector<SharedDeviceBufferList>& nodes_to_materialize,
    MaterializationReason reason, mlir::MLIRContext& mlir_context) {
  tsl::profiler::TraceMe t("MaterializeQueue");

  // Under concurrent multi-threaded eager dispatch, a deferred operation could
  // have already been compiled and materialized on device by an earlier
  // background task (since queue dispatch is non-blocking on the Python
  // thread). We dynamically filter out already materialized nodes (where
  // deferred_op() is null) to prevent redundant execution and avoid
  // DeviceBufferList state conflicts.
  std::vector<SharedDeviceBufferList> filtered_queue;
  filtered_queue.reserve(queue.size());
  for (const auto& node : queue) {
    if (node->deferred_op() != nullptr) {
      filtered_queue.push_back(node);
    }
  }

  std::vector<SharedDeviceBufferList> filtered_nodes_to_materialize;
  filtered_nodes_to_materialize.reserve(nodes_to_materialize.size());
  for (const auto& node : nodes_to_materialize) {
    if (node->deferred_op() != nullptr) {
      filtered_nodes_to_materialize.push_back(node);
    }
  }

  ABSL_VLOG(1)
      << ">>> MaterializationWorker::MaterializeQueue filtered queue size "
      << filtered_queue.size() << " (original: " << queue.size() << ")";

  ABSL_VLOG(1) << "Queue: " << ToString(filtered_queue);
  ABSL_VLOG(1) << "Nodes to Materialize: "
               << ToString(filtered_nodes_to_materialize);

  if (filtered_queue.empty() && filtered_nodes_to_materialize.empty()) {
    ABSL_VLOG(1) << "[MaterializationWorker] All nodes are already "
                    "materialized, skipping queue.";
    return absl::OkStatus();
  }

  ABSL_VLOG(1) << ">>> MaterializationWorker::MaterializeQueue "
               << filtered_queue.size();

  ABSL_VLOG(1) << "Queue: " << ToString(filtered_queue);
  ABSL_VLOG(1) << "Nodes to Materialize: "
               << ToString(filtered_nodes_to_materialize);

  std::vector<absl_nonnull std::unique_ptr<Traversal>> split_traversals;
  {
    ABSL_VLOG(1) << "[MaterializationWorker] Creating traversal";
    TT_ASSIGN_OR_RETURN(
        auto traversal,
        ExtractTraversal(filtered_queue, filtered_nodes_to_materialize));
    absl::flat_hash_set<const DeviceBufferList*> required_outputs;
    for (const auto& node : filtered_nodes_to_materialize) {
      required_outputs.insert(node.get());
    }
    TT_ASSIGN_OR_RETURN(split_traversals,
                        SplitTraversal(std::move(traversal), required_outputs));
  }

  ABSL_VLOG(1) << "[MaterializationWorker] Split traversal into "
               << split_traversals.size() << " traversals";

  // Launch compilations for the identified regions.
  auto compilation_mode = GetCompilationMode(GetEagerMode());

  std::vector<ExecutionTask> execution_tasks;
  execution_tasks.reserve(split_traversals.size());

  for (auto& traversal : split_traversals) {
    // Don't launch kernels if this is part of TorchTPU tracing of an FX graph,
    // which is indicated by placeholder inputs.
    bool has_placeholder = false;
    for (const auto& arg : traversal->arguments()) {
      if (arg.is_placeholder()) {
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
    // Supported bounded dynamism using the passed mlir_context.
    TT_ASSIGN_OR_RETURN(auto execution_task,
                        ExecutionTask::FromTraversalWithLogging(
                            std::move(traversal), mlir_context, reason));
    execution_tasks.push_back(std::move(execution_task));
  }

  // Launch compiled kernels.
  for (auto& execution_task : execution_tasks) {
    TT_RETURN_IF_ERROR(execution_task.Run());
  }

  return absl::OkStatus();
}

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

  absl::Status GetAndResetLastStatus() {
    absl::MutexLock lock(mutex_);
    absl::Status status = std::move(last_status_);
    last_status_ = absl::OkStatus();
    return status;
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
};

absl::Status MaterializationWorker::OnNewOpDispatch(
    const SharedDeviceBufferList& device_buffer_list) {
  tsl::profiler::TraceMe t("Worker_OnNewOpDispatch");
  TT_RETURN_IF_ERROR(GetAndResetLastStatus());
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
  TT_RETURN_IF_ERROR(GetAndResetLastStatus());
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
  TT_RETURN_IF_ERROR(GetAndResetLastStatus());

  {
    absl::MutexLock lock(mutex_);
    // Wait for the execution thread to be caught up.
    mutex_.Await(absl::Condition(
        this, &MaterializationWorker::IsExecutionThreadCaughtUp));
  }

  ABSL_VLOG(1) << ">>> BlockOnPendingMaterializations DONE";
  return GetAndResetLastStatus();
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
  absl_nonnull std::unique_ptr<mlir::MLIRContext> mlir_context =
      MakeMlirContext();
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
    absl::Status status = MaterializeQueue(current_queue, nodes_to_materialize,
                                           current_reason, *mlir_context);

    {
      absl::MutexLock lock(mutex_);
      if (!status.ok()) {
        ABSL_LOG(ERROR) << "Failed to materialize queue: " << status;
        // This typically indicates a compilation failure, rather than an
        // execution failure.
        // Mark all nodes in the job as materialization failures so that
        // AwaitBuffer() will return the compilation error instead of hanging.
        for (const auto& node : nodes_to_materialize) {
          node->SetAsError(status);
        }
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
