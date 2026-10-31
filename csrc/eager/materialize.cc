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

#include "csrc/eager/materialize.h"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <iterator>
#include <memory>
#include <optional>
#include <queue>
#include <string_view>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "c10/core/Stream.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/context_manager.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/reroute_warnings.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_builder.h"
#include "csrc/eager/current_stream.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/execution_task.h"
#include "csrc/eager/split_traversal.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/traversal_utils.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {
namespace {

// A task to materialize a stream.
// This will check the stream state and enqueue all necessary executions to
// catch the stream up to the most recent snapshot (created by
// EventSnapshot::Record or RecordDeviceSnapshots)
struct MaterializationTask {
  c10::DeviceIndex device_index;
  c10::StreamId stream_id;
  xla::Promise<void> started_promise;
};

// Checks that all must_materialize nodes in the task have started
// materializing. If any are not materializing, then those nodes are marked
// as an error and the first error is returned to signal an overall task
// failure.
absl::Status VerifyMustMaterialize(
    absl::Span<const SharedDeviceBufferList> must_materialize) {
  absl::Status first_missing_node_error = absl::OkStatus();
  for (const auto& node : must_materialize) {
    if (!node->is_materializing()) {
      auto error = TT_ERROR(error::kInternal)
                   << "materialization failed for node " << node;
      node->SetAsError(error);
      if (first_missing_node_error.ok()) {
        first_missing_node_error = error;
      }
    }
  }
  return first_missing_node_error;
}

enum class DeviceOutputTracking {
  kDeferToSnapshot,
  kRecordReadyFuture,
};

// Returns a ready future for `ref`, treating a buffer already deleted or
// donated to a subsequent execution as complete on device.
xla::Future<void> GetSnapshotDeviceReadyFuture(const DeviceBufferRef& ref) {
  xla::Future<void> ready = ref.GetReadyFuture();
  if (ready.IsKnownReady() && !ready.Await().ok() &&
      ref.GetMaterializationFuture().IsKnownReady()) {
    if (const auto buf = ref.AwaitBuffer(); buf.ok() && (*buf)->IsDeleted()) {
      return xla::Future<void>(absl::OkStatus());
    }
  }
  return ready;
}

// Appends futures for an ExecutionTask's outputs to `futures` and updates
// `last_device_output` when `output_tracking == kDeferToSnapshot`.
//
// All device-resident outputs of a single PjRtLoadedExecutable::Execute call
// share the same underlying TPU completion event, while pinned-host outputs
// additionally wait on their individual D2H transfer events. Because TensorCore
// executions on `execute_thread_` are enqueued onto a single in-order PJRT
// device queue, awaiting the latest TensorCore output at EventSnapshot time is
// sufficient for device completion, while `GetMaterializationFuture()` still
// surfaces compilation and synchronous enqueue errors without registering a
// `PJRT_Event_OnReady` callback on every op. Standalone SparseCore programs
// (`CorePinningMode::kSparseCore`, which run on a separate hardware queue when
// `!TcControl()`) and `CompiledExecution` tasks record `GetReadyFuture()`
// directly without overwriting `last_device_output`.
void AppendExecutionTaskReadyFutures(
    const ExecutionTask& execution_task, DeviceOutputTracking output_tracking,
    std::vector<xla::Future<void>>& futures,
    std::optional<DeviceBufferRef>& last_device_output) {
  const DeviceBufferRef* representative_device_output = nullptr;
  for (const DeviceBufferRef& output : execution_task.outputs()) {
    if (output.is_pinned_host()) {
      futures.push_back(output.GetReadyFuture());
    } else {
      representative_device_output = &output;
    }
  }
  if (representative_device_output != nullptr) {
    if (output_tracking == DeviceOutputTracking::kDeferToSnapshot) {
      futures.push_back(
          representative_device_output->GetMaterializationFuture());
      last_device_output = *representative_device_output;
    } else {
      futures.push_back(representative_device_output->GetReadyFuture());
    }
  }
}

// Attempts to compile one or more eager execution tasks using the given
// MLIR context.
// All successfully compiled ExecutionTasks are added to
// `flattened_execution_tasks` and futures for their execution are added to
// `materialization_futures`.
// If any error occurs, the error is stored in `materialization_error` if it is
// currently not set (OkStatus()).
void ProcessEagerExecution(
    EagerExecution eager_execution, mlir::MLIRContext& mlir_context,
    std::vector<ExecutionTask>& flattened_execution_tasks,
    std::vector<xla::Future<void>>& materialization_futures,
    std::optional<DeviceBufferRef>& last_device_output,
    absl::Status& materialization_error) {
  PrepareTraversalsOptions options{
      .graph_split_mode = eager_execution.settings.graph_split_mode,
      .core_pinning_mode = eager_execution.core_pinning_mode,
      .skinny_sparse_core_offload_mode = GetSkinnySparseCoreOffloadMode(),
  };
  auto traversals =
      PrepareTraversals(std::move(eager_execution.must_execute),
                        eager_execution.must_materialize, options);

  auto tasks_result =
      MakeExecutionTasks(std::move(traversals), mlir_context,
                         *eager_execution.settings.compilation_context,
                         eager_execution.settings.compilation_mode,
                         eager_execution.settings.reason);

  if (tasks_result.first_error.ok()) {
    tasks_result.first_error =
        VerifyMustMaterialize(eager_execution.must_materialize);
  }
  if (!tasks_result.first_error.ok() && materialization_error.ok()) {
    materialization_error = tasks_result.first_error;
  }
  for (auto& execution_task : tasks_result.execution_tasks) {
    const DeviceOutputTracking output_tracking =
        execution_task.core_pinning_mode() == CorePinningMode::kSparseCore
            ? DeviceOutputTracking::kRecordReadyFuture
            : DeviceOutputTracking::kDeferToSnapshot;
    AppendExecutionTaskReadyFutures(execution_task, output_tracking,
                                    materialization_futures,
                                    last_device_output);
    flattened_execution_tasks.push_back(std::move(execution_task));
  }
}

void ProcessCompiledExecution(
    CompiledExecution compiled_execution,
    std::vector<ExecutionTask>& flattened_execution_tasks,
    std::vector<xla::Future<void>>& materialization_futures,
    std::optional<DeviceBufferRef>& last_device_output) {
  AppendExecutionTaskReadyFutures(compiled_execution,
                                  DeviceOutputTracking::kRecordReadyFuture,
                                  materialization_futures, last_device_output);
  flattened_execution_tasks.push_back(std::move(compiled_execution));
}

void ProcessSnapshotRequest(
    SnapshotRequest snapshot_request,
    std::vector<xla::Future<void>>& materialization_futures,
    std::optional<DeviceBufferRef>& last_device_output,
    absl::Status& materialization_error) {
  const std::optional<DeviceBufferRef>& latest_output =
      last_device_output.has_value() ? last_device_output
                                     : snapshot_request.last_device_output;
  if (materialization_error.ok()) {
    if (latest_output.has_value()) {
      snapshot_request.futures.push_back(
          GetSnapshotDeviceReadyFuture(*latest_output));
    }
    snapshot_request.futures.insert(
        snapshot_request.futures.end(),
        std::make_move_iterator(materialization_futures.begin()),
        std::make_move_iterator(materialization_futures.end()));
    auto snapshot = std::make_shared<EventSnapshot>(
        xla::JoinFutures(snapshot_request.futures),
        std::move(snapshot_request.ok_callback_promise));
    snapshot_request.snapshot_promise.Set(std::move(snapshot));
  } else {
    snapshot_request.snapshot_promise.Set(std::make_shared<EventSnapshot>(
        materialization_error,
        std::move(snapshot_request.ok_callback_promise)));
    materialization_error = absl::OkStatus();
  }
  last_device_output.reset();
  snapshot_request.futures.clear();
  materialization_futures.clear();
}

void ProcessReadyWorkItem(
    ReadyWork ready_work_item, mlir::MLIRContext& mlir_context,
    std::vector<ExecutionTask>& flattened_execution_tasks,
    std::vector<xla::Future<void>>& materialization_futures,
    std::optional<DeviceBufferRef>& last_device_output,
    absl::Status& materialization_error) {
  if (auto* eager_execution = std::get_if<EagerExecution>(&ready_work_item)) {
    ProcessEagerExecution(std::move(*eager_execution), mlir_context,
                          flattened_execution_tasks, materialization_futures,
                          last_device_output, materialization_error);
  } else if (auto* compiled_execution =
                 std::get_if<CompiledExecution>(&ready_work_item)) {
    ProcessCompiledExecution(std::move(*compiled_execution),
                             flattened_execution_tasks, materialization_futures,
                             last_device_output);
  } else {
    auto& snapshot_request = std::get<SnapshotRequest>(ready_work_item);
    ProcessSnapshotRequest(std::move(snapshot_request), materialization_futures,
                           last_device_output, materialization_error);
  }
}

// Processes a MaterializationTask, converting it into a sequence of
// ExecutionTasks.
std::vector<ExecutionTask> ProcessMaterializationTask(
    MaterializationTask& task, mlir::MLIRContext& mlir_context) {
  ABSL_VLOG(1) << "[MaterializationWorker] Processing MaterializationTask with "
               << "stream " << task.stream_id << " on device "
               << task.device_index;
  std::vector<ExecutionTask> flattened_execution_tasks;
  std::vector<xla::Future<void>> materialization_futures;
  std::optional<DeviceBufferRef> last_device_output;
  absl::Status materialization_error = absl::OkStatus();
  auto ready_work = GetReadyWork(task.device_index, task.stream_id);
  for (auto& ready_work_item : ready_work) {
    ProcessReadyWorkItem(std::move(ready_work_item), mlir_context,
                         flattened_execution_tasks, materialization_futures,
                         last_device_output, materialization_error);
  }

  ABSL_VLOG(2) << "[MaterializationWorker] Created "
               << flattened_execution_tasks.size()
               << " ExecutionTasks for stream " << task.stream_id
               << " on device " << task.device_index << " to satisfy "
               << ready_work.size() << " ready work blocks";

  if (!materialization_error.ok()) {
    materialization_futures.push_back(xla::Future<void>(materialization_error));
  }
  if (!materialization_futures.empty() || last_device_output.has_value()) {
    WriteBackMaterializationFutures(task.device_index, task.stream_id,
                                    std::move(materialization_futures),
                                    std::move(last_device_output));
  }

  task.started_promise.Set(materialization_error);
  return flattened_execution_tasks;
}

absl::Status MarkPinnedHostOutputs(
    absl::Span<const DeviceBufferRef> outputs,
    const SharedLoadedExecutableWithMetadata& executable) {
  TT_ASSIGN_OR_RETURN(
      const auto output_memory_kinds,
      executable->GetLoadedExecutable()->GetOutputMemoryKinds());

  if (output_memory_kinds.empty()) {
    return absl::OkStatus();
  }

  const auto& kinds = output_memory_kinds.front();
  ABSL_DCHECK_EQ(outputs.size(), kinds.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (kinds[i] == "pinned_host") {
      outputs[i].set_is_pinned_host();
    }
  }

  return absl::OkStatus();
}

// Signals that a shutdown has been initiated.
enum class ShutdownSentinel {};

using MaterializationOrShutdown =
    std::variant<ShutdownSentinel, MaterializationTask>;
using ExecutionOrShutdown = std::variant<ShutdownSentinel, ExecutionTask>;

class MaterializationWorker {
 public:
  // This class is move-only.
  MaterializationWorker(MaterializationWorker&& other) = default;
  MaterializationWorker& operator=(MaterializationWorker&& other) = default;
  MaterializationWorker(const MaterializationWorker&) = delete;
  MaterializationWorker& operator=(const MaterializationWorker&) = delete;

  MaterializationWorker() { StartThreads(); }

  ~MaterializationWorker() { Shutdown(); }

  // Shuts down the worker threads. Reordering the shutdown sequence ensures
  // that materialize_thread_ finishes enqueuing tasks before execute_thread_
  // shuts down.
  void Shutdown() {
    bool expected_shutdown = false;
    // The shutdown_ flag prevents duplicate shutdown runs.
    if (shutdown_.compare_exchange_strong(expected_shutdown, true)) {
      {
        absl::MutexLock lock(materialize_mu_);
        materialize_tasks_.push(ShutdownSentinel{});
      }
      // Release the lock so that the thread can dequeue until the sentinel.
      if (materialize_thread_.joinable()) {
        materialize_thread_.join();
      }
      {
        absl::MutexLock lock(execute_mu_);
        execute_tasks_.push(ShutdownSentinel{});
      }
      if (execute_thread_.joinable()) {
        execute_thread_.join();
      }
      FailRemainingExecutionTasks();
    }
  }

  xla::Future<void> EnqueueStream(const c10::DeviceIndex device_index,
                                  const c10::StreamId stream_id) {
    ABSL_VLOG(1) << "[MaterializationWorker] Enqueuing stream " << stream_id
                 << " on device " << device_index << " for materialization";
    absl::MutexLock lock(materialize_mu_);
    // Check shutdown state after acquiring the lock to avoid a time-of-check/
    // time-of-use error relative to Shutdown().
    if (shutdown_.load()) {
      return xla::Future<void>(
          TT_ERROR(error::kCancelled)  // ERROR_COV_INFEASIBLE=unreachable
                                       // from Python
          << "MaterializationWorker is shutting down");
    }

    // If multiple materialization tasks are pushed to the same stream
    // back-to-back, the materialization worker will process them all together
    // and set a single started future.
    if (!materialize_tasks_.empty()) {
      // Known to not be a shutdown sentinel because of the check above.
      auto& prev_task =
          std::get<MaterializationTask>(materialize_tasks_.back());
      if (prev_task.device_index == device_index &&
          prev_task.stream_id == stream_id) {
        return prev_task.started_promise.future();
      }
    }

    // Either the queue is empty or the previous task is on a different stream.
    // Create a new materialization task with a new start promise/future.
    auto [started_promise, started_future] = xla::MakePromise<void>();
    materialize_tasks_.push(MaterializationTask(device_index, stream_id,
                                                std::move(started_promise)));
    return started_future;
  }

  // Dequeues a materialization task or a shutdown signal.
  MaterializationOrShutdown DequeueMaterializationTask() {
    absl::MutexLock lock(materialize_mu_);
    materialize_mu_.Await(absl::Condition(
        +[](std::queue<MaterializationOrShutdown>* tasks) {
          return !tasks->empty();
        },
        &materialize_tasks_));
    auto popped_task = std::move(materialize_tasks_.front());
    materialize_tasks_.pop();
    return popped_task;
  }

  // Dequeues an execution task or a shutdown signal.
  ExecutionOrShutdown DequeueExecutionTask() {
    absl::MutexLock lock(execute_mu_);
    execute_mu_.Await(absl::Condition(
        +[](std::queue<ExecutionOrShutdown>* tasks) { return !tasks->empty(); },
        &execute_tasks_));
    auto popped_task = std::move(execute_tasks_.front());
    execute_tasks_.pop();
    return popped_task;
  }

  void StartThreads() {
    materialize_thread_ = std::thread([this]() { MaterializeLoop(); });
    execute_thread_ = std::thread([this]() { ExecuteLoop(); });
  }

 private:
  // Processes a single materialization task.
  // Returns true if the loop should continue, or false if the worker should
  // shutdown.
  bool MaterializeLoopStep(mlir::MLIRContext& mlir_context) {
    auto task_or_shutdown = DequeueMaterializationTask();
    if (std::holds_alternative<ShutdownSentinel>(task_or_shutdown)) {
      return false;
    }
    auto& task = std::get<MaterializationTask>(task_or_shutdown);
    ABSL_VLOG(1) << "[MaterializationWorker] Processing MaterializationTask";
    auto execution_tasks = ProcessMaterializationTask(task, mlir_context);

    // Enqueue as many execution tasks as were successfully created.
    ABSL_VLOG(1) << "[MaterializationWorker] Enqueuing "
                 << execution_tasks.size() << " ExecutionTasks";
    absl::MutexLock lock(execute_mu_);
    for (auto& execution_task : execution_tasks) {
      execute_tasks_.push(std::move(execution_task));
    }
    return true;
  }

  void MaterializeLoop() {
    // Prevent materialization worker threads from accessing
    // thread-local context states to enforce they always rely on
    // resolved configurations passed down from the dispatch thread.
    DisallowThisThreadToAccessContextState();
    // Let dispatch threads surface this thread's warnings to Python.
    RouteWarningsOnThisThreadToDispatchThread();

    // Create the MLIR context outside the loop once and reuse for
    // materialization tasks.
    absl_nonnull std::unique_ptr<mlir::MLIRContext> mlir_context =
        MakeMlirContext();
    while (MaterializeLoopStep(*mlir_context)) {
      // MaterializeLoopStep will return false when it dequeues a
      // ShutdownSentinel.
    }
  }

  void ExecuteLoop() {
    // Prevent execution worker threads from accessing thread-local context
    // states to enforce they always rely on resolved configurations passed
    // down from the dispatch thread.
    DisallowThisThreadToAccessContextState();
    // Let dispatch threads surface this thread's warnings to Python.
    RouteWarningsOnThisThreadToDispatchThread();

    while (true) {
      auto task_or_shutdown = DequeueExecutionTask();
      if (std::holds_alternative<ShutdownSentinel>(task_or_shutdown)) {
        break;
      }
      auto& task = std::get<ExecutionTask>(task_or_shutdown);
      ABSL_VLOG(1) << "[MaterializationWorker] Processing ExecutionTask";
      auto status = task.Run();
      if (!status.ok()) {
        ABSL_LOG(ERROR) << "[MaterializationWorker] ExecutionTask failed: "
                        << status;
      }
    }
  }

  void FailRemainingExecutionTasks() {
    absl::MutexLock lock(execute_mu_);
    if (execute_tasks_.empty()) return;

    // execute_tasks_ should be empty, but we clear it to be safe.
    ABSL_LOG(WARNING) << "[MaterializationWorker::Shutdown] "
                      << "execute_tasks_ is not empty after shutdown "
                         "sentinel. Discarding "
                      << execute_tasks_.size() << " tasks";
    auto shutdown_status =
        TT_ERROR(error::kCancelled)  // ERROR_COV_INFEASIBLE=unreachable
                                     // from Python
        << "MaterializationWorker is shutting down";
    while (!execute_tasks_.empty()) {
      auto& task_or_shutdown = execute_tasks_.front();
      if (auto* execution_task =
              std::get_if<ExecutionTask>(&task_or_shutdown)) {
        execution_task->SetOutputNodesAsError(shutdown_status);
      }
      execute_tasks_.pop();
    }
  }
  std::thread materialize_thread_;
  std::thread execute_thread_;

  // Set to true when the MaterializationWorker is undergoing shutdown,
  // preventing concurrent or duplicate shutdown sequences from executing.
  std::atomic<bool> shutdown_{false};

  absl::Mutex materialize_mu_;
  // Queue of materialization tasks. Storing std::variant allows enqueuing
  // ShutdownSentinel as a shutdown sentinel / poison pill.
  std::queue<MaterializationOrShutdown> materialize_tasks_
      ABSL_GUARDED_BY(materialize_mu_);

  absl::Mutex execute_mu_;
  // Queue of execution tasks. Storing std::variant allows enqueuing
  // ShutdownSentinel as a shutdown sentinel / poison pill.
  std::queue<ExecutionOrShutdown> execute_tasks_ ABSL_GUARDED_BY(execute_mu_);
};

MaterializationWorker& GetMaterializationWorker() {
  static absl::NoDestructor<MaterializationWorker> worker;
  return *worker;
}

inline const SharedDeviceBufferList& GetDeviceBufferList(
    const SharedDeviceBufferList& node) {
  return node;
}

inline const SharedDeviceBufferList& GetDeviceBufferList(
    const DeviceBufferRef& buffer_ref) {
  return buffer_ref.device_buffer_list();
}

// Makes a copy of all SharedDeviceBufferLists in `items` that are deferred and
// unique and on the specified stream.
// Returns an error if any item is a placeholder.
template <typename T>
absl::StatusOr<std::vector<SharedDeviceBufferList>> GetUniqueDeferredNodes(
    absl::Span<const T> items, c10::DeviceIndex device_index,
    std::optional<c10::StreamId> stream_id) {
  std::vector<SharedDeviceBufferList> nodes_to_materialize;  // return value

  // Depending on the number of items to scan, we either use an inlined vector
  // or a hash set to deduplicate nodes.
  if (items.size() <= 16) {
    absl::InlinedVector<const DeviceBufferList*, 16> seen;
    for (const T& item : items) {
      const SharedDeviceBufferList& node = GetDeviceBufferList(item);
      if (node->device_index() != device_index) {
        continue;
      }
      if (stream_id.has_value() && node->stream_id() != *stream_id) {
        continue;
      }
      if (ABSL_PREDICT_FALSE(node->is_placeholder())) {
        return TT_ERROR(error::kInternal)
               << "cannot Materialize() a placeholder tensor or a tensor that "
                  "depends on a placeholder. \nPlaceholder tensors should only "
                  "appear in compiled mode, which should never try to "
                  "materialize tensors";
      }
      if (node->is_deferred()) {
        const DeviceBufferList* ptr = node.get();
        bool duplicate = false;
        for (const DeviceBufferList* s : seen) {
          if (s == ptr) {
            duplicate = true;
            break;
          }
        }
        if (!duplicate) {
          seen.push_back(ptr);
          if (nodes_to_materialize.empty()) {
            nodes_to_materialize.reserve(items.size());
          }
          nodes_to_materialize.push_back(node);
        }
      }
    }
  } else {
    absl::flat_hash_set<const DeviceBufferList*> seen;
    for (const T& item : items) {
      const SharedDeviceBufferList& node = GetDeviceBufferList(item);
      if (node->device_index() != device_index) {
        continue;
      }
      if (stream_id.has_value() && node->stream_id() != *stream_id) {
        continue;
      }
      if (ABSL_PREDICT_FALSE(node->is_placeholder())) {
        return TT_ERROR(error::kInternal)
               << "cannot Materialize() a placeholder tensor or a tensor that "
                  "depends on a placeholder. \nPlaceholder tensors should only "
                  "appear in compiled mode, which should never try to "
                  "materialize tensors";
      }
      if (node->is_deferred()) {
        if (seen.empty()) {
          seen.reserve(items.size());
          nodes_to_materialize.reserve(items.size());
        }
        if (seen.insert(node.get()).second) {
          nodes_to_materialize.push_back(node);
        }
      }
    }
  }
  return nodes_to_materialize;
}

// Dispatches only as many stream materialization tasks as necessary to ensure
// all deferred nodes in `items` are enqueued for materialization.
// Note: unlike MaterializeStream (which always enqueues a stream sync, even if
// all must_materialize are already materialized), this function does not
// enqueue any stream syncs if all items are already materialized.
// Returns an error if any item is a placeholder.
template <typename T>
absl::Status MaterializeSpan(absl::Span<const T> items,
                             MaterializationReason reason,
                             GraphSplitMode graph_split_mode,
                             MaterializationBlockMode block_mode) {
  if (items.empty()) {
    return absl::OkStatus();
  }

  // In the majority of cases, we expect that there will very few distinct
  // streams, so we use a vector instead of a map.
  struct DeviceStreamId {
    c10::DeviceIndex device_index;
    c10::StreamId stream_id;
  };
  absl::InlinedVector<DeviceStreamId, 2> device_stream_ids;
  std::vector<xla::Future<void>> started_futures;
  for (const T& item : items) {
    const SharedDeviceBufferList& node = GetDeviceBufferList(item);
    // Check if this node needs to be materialized.
    if (ABSL_PREDICT_FALSE(node->is_placeholder())) {
      return TT_ERROR(error::kInternal)
             << "cannot Materialize() a placeholder tensor or a tensor that "
                "depends on a placeholder. \nPlaceholder tensors should only "
                "appear in compiled mode, which should never try to "
                "materialize tensors";
    }
    if (!node->is_deferred()) {
      continue;
    }

    // This node needs to be materialized. Check if we already enqueued its
    // stream for materialization.
    if (auto it =
            std::find_if(device_stream_ids.begin(), device_stream_ids.end(),
                         [&node](const DeviceStreamId& id) {
                           return id.device_index == node->device_index() &&
                                  id.stream_id == node->stream_id();
                         });
        it != device_stream_ids.end()) {
      continue;
    }

    // Found a new stream to enqueue.
    // Copy out only its unique, deferred nodes to a vector
    // and push the vector to the worker as a stream materialization task.
    TT_ASSIGN_OR_RETURN(
        std::vector<SharedDeviceBufferList> nodes_on_stream,
        GetUniqueDeferredNodes(items, node->device_index(), node->stream_id()));
    InsertGraphBreak(node->device_index(), node->stream_id(),
                     std::move(nodes_on_stream), reason, graph_split_mode);
    xla::Future<void> started_future = GetMaterializationWorker().EnqueueStream(
        node->device_index(), node->stream_id());
    if (block_mode == MaterializationBlockMode::kBlockUntilStarted) {
      started_futures.push_back(started_future);
    }
    // Record that we enqueued this stream.
    device_stream_ids.push_back(DeviceStreamId{
        .device_index = node->device_index(), .stream_id = node->stream_id()});
  }
  if (block_mode == MaterializationBlockMode::kBlockUntilStarted) {
    TT_RETURN_IF_ERROR(xla::JoinFutures(started_futures).Await());
  }
  return absl::OkStatus();
}

}  // namespace

void ShutDownMaterializationState() { GetMaterializationWorker().Shutdown(); }

absl::Status Materialize(absl::Span<const SharedDeviceBufferList> nodes,
                         MaterializationReason reason,
                         GraphSplitMode graph_split_mode,
                         MaterializationBlockMode block_mode) {
  return MaterializeSpan(nodes, reason, graph_split_mode, block_mode);
}

absl::Status Materialize(absl::Span<const DeviceBufferRef> buffer_refs,
                         MaterializationReason reason,
                         GraphSplitMode graph_split_mode) {
  return Materialize(buffer_refs, reason, graph_split_mode,
                     MaterializationBlockMode::kBlockUntilStarted);
}

absl::Status Materialize(absl::Span<const DeviceBufferRef> buffer_refs,
                         MaterializationReason reason,
                         GraphSplitMode graph_split_mode,
                         MaterializationBlockMode block_mode) {
  return MaterializeSpan(buffer_refs, reason, graph_split_mode, block_mode);
}

absl::StatusOr<std::vector<DeviceBufferRef>> EnqueueExecutable(
    SharedLoadedExecutableWithMetadata executable,
    std::vector<DeviceBufferRef> arguments,
    absl::Span<const Shape> output_shapes, std::string_view task_name) {
  // Get the stream from the calling thread, not from the execution worker
  // thread.
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  const auto eager_mode = GetEagerMode();
  // Create a set of output DeviceBufferRefs to hold the materialized results.
  std::vector<DeviceBufferRef> outputs;
  outputs.reserve(output_shapes.size());
  for (const auto& shape : output_shapes) {
    // Create pending buffer lists to hold the results.
    TT_ASSIGN_OR_RETURN(DeviceBufferRef output_ref,
                        DeviceBufferList::CreatePendingExecution(
                            shape, device_index, stream_id));
    outputs.push_back(std::move(output_ref));
  }

  TT_RETURN_IF_ERROR(MarkPinnedHostOutputs(outputs, executable));

  // FromExecutable will check to make sure all arguments are materializing
  // and error if they are not.
  // Intentional copy on outputs; we need to both include them in the task
  // and return them to the caller.
  TT_ASSIGN_OR_RETURN(
      ExecutionTask task,
      ExecutionTask::FromExecutable(std::move(executable), std::move(arguments),
                                    outputs, task_name));

  // Insert the task into the stream's event queue with a snapshot. This ensures
  // that anything which refers to the output of this task will be enqueued
  // after it.
  RecordCompiledExecutionTask(std::move(task), device_index, stream_id,
                              eager_mode);

  // Notify the materialization worker that there is work to do on this stream.
  GetMaterializationWorker().EnqueueStream(device_index, stream_id);

  return outputs;
}

xla::Future<std::shared_ptr<EventSnapshot>> MaterializeStream(
    c10::DeviceIndex device_index, c10::StreamId stream_id,
    MaterializationReason reason, GraphSplitMode mode,
    absl::Span<const SharedDeviceBufferList> must_materialize) {
#ifndef NDEBUG
  for (const auto& node : must_materialize) {
    ABSL_CHECK_EQ(node->device_index(),  // CRASH_OK=not visible to Python
                  device_index);
    ABSL_CHECK_EQ(node->stream_id(),  // CRASH_OK=not visible to Python
                  stream_id);
  }
#endif
  TT_ASSIGN_OR_RETURN(
      std::vector<SharedDeviceBufferList> must_materialize_vec,
      GetUniqueDeferredNodes(must_materialize, device_index, stream_id));
  auto future = EventSnapshot::Record(
      device_index, stream_id, std::move(must_materialize_vec), reason, mode);
  GetMaterializationWorker().EnqueueStream(device_index, stream_id);
  return future;
}

std::vector<xla::Future<std::shared_ptr<EventSnapshot>>> MaterializeDevice(
    c10::DeviceIndex device_index, MaterializationReason reason,
    GraphSplitMode mode,
    absl::Span<const SharedDeviceBufferList> must_materialize) {
#ifndef NDEBUG
  for (const auto& node : must_materialize) {
    ABSL_CHECK_EQ(node->device_index(),  // CRASH_OK=not visible to Python
                  device_index);
  }
#endif
  auto must_materialize_vec_or = GetUniqueDeferredNodes(
      must_materialize, device_index, /*stream_id=*/std::nullopt);
  if (!must_materialize_vec_or.ok()) {
    // Convert from a single error to a single-element vector of a ready, error
    // future.
    return {must_materialize_vec_or.status()};
  }

  auto device_snapshot_futures = RecordDeviceSnapshots(
      device_index, std::move(*must_materialize_vec_or), reason, mode);

  std::vector<xla::Future<std::shared_ptr<EventSnapshot>>> snapshot_futures;
  snapshot_futures.reserve(device_snapshot_futures.size());

  auto& materialization_worker = GetMaterializationWorker();
  for (auto& pending_stream : device_snapshot_futures) {
    materialization_worker.EnqueueStream(pending_stream.device_index,
                                         pending_stream.stream_id);
    snapshot_futures.push_back(std::move(pending_stream.snapshot_future));
  }

  return snapshot_futures;
}

xla::Future<std::shared_ptr<EventSnapshot>> MaterializeCurrentStream(
    MaterializationReason reason, GraphSplitMode mode,
    absl::Span<const SharedDeviceBufferList> must_materialize) {
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  return MaterializeStream(device_index, stream_id, reason, mode,
                           must_materialize);
}

void EagerlyExecuteStream(
    c10::DeviceIndex device_index, c10::StreamId stream_id,
    MaterializationReason reason, GraphSplitMode mode,
    absl::Span<const SharedDeviceBufferList> must_materialize) {
  InsertGraphBreak(device_index, stream_id,
                   std::vector<SharedDeviceBufferList>(must_materialize.begin(),
                                                       must_materialize.end()),
                   reason, mode);
  GetMaterializationWorker().EnqueueStream(device_index, stream_id);
}

void EagerlyExecuteCurrentStream(
    MaterializationReason reason, GraphSplitMode mode,
    absl::Span<const SharedDeviceBufferList> must_materialize) {
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  EagerlyExecuteStream(device_index, stream_id, reason, mode, must_materialize);
}

}  // namespace torch_tpu
