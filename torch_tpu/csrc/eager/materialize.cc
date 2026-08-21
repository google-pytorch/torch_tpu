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

#include "torch_tpu/csrc/eager/materialize.h"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iterator>
#include <memory>
#include <queue>
#include <string>
#include <string_view>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_log.h"
#include "absl/log/absl_vlog_is_on.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "c10/core/Stream.h"
#include "mlir/IR/MLIRContext.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/compilation.h"
#include "torch_tpu/csrc/common/compilation_spec.h"
#include "torch_tpu/csrc/common/context_manager.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/shape.h"
#include "torch_tpu/csrc/common/status_builder.h"
#include "torch_tpu/csrc/eager/current_stream.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/eager_mode.h"
#include "torch_tpu/csrc/eager/events_queue.h"
#include "torch_tpu/csrc/eager/materialize_common.h"
#include "torch_tpu/csrc/eager/split_traversal.h"
#include "torch_tpu/csrc/eager/structured_log_buffer.h"
#include "torch_tpu/csrc/eager/traversal.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/future.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

// A task to materialize a list of nodes.
// This will materialize all streams that have any of the given nodes, but only
// up to the last node in the list for each stream.
struct NodesMaterializationTask {
  std::vector<SharedDeviceBufferList> nodes_to_materialize;
  xla::Promise<void> completion_promise;
};

// A task to materialize a stream.
// This will check the current state of deferred work on the stream, and start
// all necessary executions to catch the stream up to a snapshot.
struct StreamMaterializationTask {
  c10::DeviceIndex device_index;
  c10::StreamId stream_id;
  xla::Promise<std::shared_ptr<EventSnapshot>> completion_promise;
};

// A task to materialize a device.
// This will check the current state of deferred work on each stream on the
// device, and start all necessary executions to catch each stream up to a
// snapshot.
struct DeviceMaterializationTask {
  c10::DeviceIndex device_index;
  xla::Promise<std::vector<std::shared_ptr<EventSnapshot>>> completion_promise;
};

using MaterializationKind =
    std::variant<NodesMaterializationTask, StreamMaterializationTask,
                 DeviceMaterializationTask>;

// Common properties for all materialization tasks.
struct MaterializationTaskCommon {
  MaterializationMode materialization_mode = MaterializationMode::kSplitGraph;
  MaterializationReason reason;
  CompilationSpec compilation_spec;
};

struct MaterializationTask {
  MaterializationKind kind;
  MaterializationTaskCommon common;
};

}  // namespace

// PRECONDITION: executables must be a sequence that is compatible with the
// arguments and composable, as they will be executed in order and the outputs
// of an executable are the arguments to the next. The final execuatable returns
// a complete set of outputs for each node in sequential blocks. That is, either
// all outputs of a node appear contiguously and in the same order as in
// results, or none of them do. Furthermore, the order of nodes must be
// consistent between the execution and results. For example, if node A is size
// 3, node B is size 1, and node C is size 2, then the output order must be [A0,
// A1, A2, B1, C0, C1]. This is established by all Materialize() functions in
// this file.

namespace {

void LogDeferredNodes(absl::Span<const SharedDeviceBufferList> nodes,
                      const std::string_view msg_prefix) {
  if (ABSL_VLOG_IS_ON(1)) {
    for (int64_t i = 0; i < nodes.size(); i++) {
      const auto& node = nodes[i];
      const auto deferred_op = node->deferred_op();
      ABSL_VLOG(1) << msg_prefix << i << ": " << node.get()
                   << (deferred_op
                           ? absl::StrCat(" op: ", deferred_op->op_name())
                           : "<Not Deferred>");
    }
  }
}

// When preparing a list of Traversals into a sequence of ExecutionTasks,
// some may fail. This struct holds the results, including the first error
// encountered, if any.
struct MaterializationStages {
  std::vector<ExecutionTask> execution_tasks;
  absl::Status first_error = absl::OkStatus();
};

// Converts a sequence of Traversals to a sequence of ExecutionTasks.
// If materialization_mode is kSplitGraph, then there may be more returned
// tasks than there were original traversals; otherwise, there will be exactly
// one ExecutionTask per traversal (provided all task creations succeed).
MaterializationStages ApplySplitMode(
    std::vector<absl_nonnull std::unique_ptr<Traversal>>&& traversals,
    const MaterializationTaskCommon& common, mlir::MLIRContext& mlir_context) {
  MaterializationStages result;

  if (common.materialization_mode == MaterializationMode::kSplitGraph) {
    tsl::profiler::TraceMe t("SplitTraversal");
    std::vector<absl_nonnull std::unique_ptr<Traversal>> post_split_traversals;
    absl::flat_hash_set<const DeviceBufferList*> required_outputs;
    for (auto& pre_split_traversal : traversals) {
      const auto core_pinning_mode = pre_split_traversal->core_pinning_mode();

      for (const auto& output : pre_split_traversal->outputs()) {
        required_outputs.insert(output.device_buffer_list().get());
      }
      auto split_traversals_or =
          SplitTraversal(std::move(pre_split_traversal), required_outputs);
      if (split_traversals_or.ok()) {
        // Retain the core pinning mode when splitting traversals.
        if (core_pinning_mode != CorePinningMode::kUnpinned) {
          for (auto& split_traversal : *split_traversals_or) {
            split_traversal->SetCorePinningMode(core_pinning_mode);
          }
        }

        post_split_traversals.insert(
            post_split_traversals.end(),
            std::make_move_iterator(split_traversals_or->begin()),
            std::make_move_iterator(split_traversals_or->end()));
      } else if (result.first_error.ok()) {
        result.first_error = split_traversals_or.status();
      }
      required_outputs.clear();
    }
    std::swap(traversals, post_split_traversals);
  }

  result.execution_tasks.reserve(traversals.size());
  for (auto& split_traversal : traversals) {
    auto execution_task_or = ExecutionTask::FromTraversalWithLogging(
        std::move(split_traversal), mlir_context,
        common.compilation_spec.Copy(), common.reason);
    if (execution_task_or.ok()) {
      result.execution_tasks.push_back(std::move(*execution_task_or));
    } else if (result.first_error.ok()) {
      result.first_error = execution_task_or.status();
    }
  }
  return result;
}

// Processes a MaterializationTask, converting it into a sequence of
// ExecutionTasks.
MaterializationStages ProcessMaterializationTask(
    MaterializationTask& task, mlir::MLIRContext& mlir_context) {
  std::vector<absl_nonnull std::unique_ptr<Traversal>> traversals;
  if (const auto* nodes_task =
          std::get_if<NodesMaterializationTask>(&task.kind)) {
    ABSL_VLOG(1)
        << "[MaterializationWorker] Processing MaterializationTask with "
        << nodes_task->nodes_to_materialize.size() << " nodes";
    LogDeferredNodes(nodes_task->nodes_to_materialize,
                     /* msg_prefix= */ "  Input node");

    std::vector<SharedDeviceBufferList> all_nodes =
        nodes_task->nodes_to_materialize;

    // Filter out non-deferred nodes that may have been materialized by an
    // earlier materialization task.
    std::erase_if(all_nodes, [](const SharedDeviceBufferList& node) {
      return !node->is_deferred();
    });
    if (all_nodes.empty()) {
      // Everything was already materialized, nothing more to do.
      return MaterializationStages{.execution_tasks = {},
                                   .first_error = absl::OkStatus()};
    }
    auto traversals_or = PrepareMaterializationTraversals(all_nodes);
    if (!traversals_or.ok()) {
      return MaterializationStages{.execution_tasks = {},
                                   .first_error = traversals_or.status()};
    }
    traversals = std::move(*traversals_or);
  } else if (const auto* stream_task =
                 std::get_if<StreamMaterializationTask>(&task.kind)) {
    ABSL_VLOG(1)
        << "[MaterializationWorker] Processing MaterializationTask with "
        << "stream " << stream_task->stream_id << " on device "
        << stream_task->device_index;
    auto traversals_or = PrepareStreamTraversals(stream_task->device_index,
                                                 stream_task->stream_id);
    if (!traversals_or.ok()) {
      return MaterializationStages{.execution_tasks = {},
                                   .first_error = traversals_or.status()};
    }
    traversals = std::move(*traversals_or);
  } else if (const auto* device_task =
                 std::get_if<DeviceMaterializationTask>(&task.kind)) {
    ABSL_VLOG(1)
        << "[MaterializationWorker] Processing MaterializationTask with "
        << "device " << device_task->device_index;
    auto traversals_or = PrepareDeviceTraversals(device_task->device_index);
    if (!traversals_or.ok()) {
      return MaterializationStages{.execution_tasks = {},
                                   .first_error = traversals_or.status()};
    }
    traversals = std::move(*traversals_or);
  } else {
    return MaterializationStages{
        .execution_tasks = {},
        .first_error = TT_ERROR(error::kInternal)
                       << "Unknown MaterializationTask kind"};
  }

  return ApplySplitMode(std::move(traversals), task.common, mlir_context);
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
      FailRemainingMaterializationTasks();
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

  xla::Future<void> EnqueueNodes(std::vector<SharedDeviceBufferList> nodes,
                                 MaterializationReason reason,
                                 MaterializationMode materialization_mode) {
    ABSL_VLOG(1) << "[MaterializationWorker] Enqueuing " << nodes.size()
                 << " nodes for materialization";
    auto [promise, future] = xla::MakePromise<void>();
    const CompilationMode compilation_mode = GetCompilationMode(GetEagerMode());

    absl::MutexLock lock(materialize_mu_);
    // Check shutdown state after acquiring the lock to avoid a time-of-check/
    // time-of-use error relative to Shutdown().
    if (shutdown_.load()) {
      // State is shutting down, fail the promise immediately.
      promise.Set(
          TT_ERROR(error::kCancelled)  // ERROR_COV_INFEASIBLE=unreachable
                                       // from Python
          << "MaterializationWorker is shutting down");
      return future;
    }
    materialize_tasks_.push(MaterializationTask{
        .kind =
            NodesMaterializationTask{
                .nodes_to_materialize = std::move(nodes),
                .completion_promise = std::move(promise),
            },
        .common =
            MaterializationTaskCommon{
                .materialization_mode = materialization_mode,
                .reason = reason,
                .compilation_spec = GetCompilationSpec(compilation_mode),
            },
    });
    return future;
  }

  xla::Future<std::shared_ptr<EventSnapshot>> EnqueueStream(
      const c10::DeviceIndex device_index, const c10::StreamId stream_id,
      MaterializationReason reason, MaterializationMode materialization_mode) {
    ABSL_VLOG(1) << "[MaterializationWorker] Enqueuing stream " << stream_id
                 << " on device " << device_index << " for materialization";
    auto [promise, future] = xla::MakePromise<std::shared_ptr<EventSnapshot>>();
    const CompilationMode compilation_mode = GetCompilationMode(GetEagerMode());

    absl::MutexLock lock(materialize_mu_);
    // Check shutdown state after acquiring the lock to avoid a time-of-check/
    // time-of-use error relative to Shutdown().
    if (shutdown_.load()) {
      // State is shutting down, fail the promise immediately.
      promise.Set(
          TT_ERROR(error::kCancelled)  // ERROR_COV_INFEASIBLE=unreachable
                                       // from Python
          << "MaterializationWorker is shutting down");
      return future;
    }
    materialize_tasks_.push(MaterializationTask{
        .kind =
            StreamMaterializationTask{
                .device_index = device_index,
                .stream_id = stream_id,
                .completion_promise = std::move(promise),
            },
        .common =
            MaterializationTaskCommon{
                .materialization_mode = materialization_mode,
                .reason = reason,
                .compilation_spec = GetCompilationSpec(compilation_mode),
            },
    });
    return future;
  }

  xla::Future<std::vector<std::shared_ptr<EventSnapshot>>> EnqueueDevice(
      const c10::DeviceIndex device_index, MaterializationReason reason,
      MaterializationMode materialization_mode) {
    ABSL_VLOG(1) << "[MaterializationWorker] Enqueuing device " << device_index
                 << " for materialization";
    auto [promise, future] =
        xla::MakePromise<std::vector<std::shared_ptr<EventSnapshot>>>();
    const CompilationMode compilation_mode = GetCompilationMode(GetEagerMode());

    absl::MutexLock lock(materialize_mu_);
    // Check shutdown state after acquiring the lock to avoid a time-of-check/
    // time-of-use error relative to Shutdown().
    if (shutdown_.load()) {
      // State is shutting down, fail the promise immediately.
      promise.Set(
          TT_ERROR(error::kCancelled)  // ERROR_COV_INFEASIBLE=unreachable
                                       // from Python
          << "MaterializationWorker is shutting down");
      return future;
    }
    materialize_tasks_.push(MaterializationTask{
        .kind =
            DeviceMaterializationTask{
                .device_index = device_index,
                .completion_promise = std::move(promise),
            },
        .common =
            MaterializationTaskCommon{
                .materialization_mode = materialization_mode,
                .reason = reason,
                .compilation_spec = GetCompilationSpec(compilation_mode),
            },
    });
    return future;
  }

  absl::StatusOr<std::vector<DeviceBufferRef>> EnqueueExecutable(
      SharedLoadedExecutableWithMetadata executable,
      std::vector<DeviceBufferRef> arguments,
      absl::Span<const Shape> output_shapes, std::string_view task_name,
      c10::DeviceIndex device_index, c10::StreamId stream_id) {
    // Create a set of output DeviceBufferRefs to hold the materialized results.
    std::vector<DeviceBufferRef> outputs;
    outputs.reserve(output_shapes.size());
    for (const auto& shape : output_shapes) {
      // Create pending buffer lists to hold the results.
      TT_ASSIGN_OR_RETURN(
          DeviceBufferRef output_ref,
          DeviceBufferList::CreatePending(shape, device_index, stream_id));
      outputs.push_back(std::move(output_ref));
    }
    RecordBackgroundMaterialization(outputs);

    // Intentional copy on outputs; we need to both include them in the task
    // and return them to the caller.
    TT_ASSIGN_OR_RETURN(
        ExecutionTask task,
        ExecutionTask::FromExecutable(
            std::move(executable), std::move(arguments), outputs, task_name));

    absl::MutexLock lock(execute_mu_);
    // Check shutdown state after acquiring the lock to avoid a time-of-check/
    // time-of-use error relative to Shutdown().
    if (shutdown_.load()) {
      // State is shutting down, fail the task and do not enqueue.
      absl::Status shutdown_status =
          TT_ERROR(error::kCancelled)  // ERROR_COV_INFEASIBLE=unreachable
                                       // from Python
          << "MaterializationWorker is shutting down";
      task.SetOutputNodesAsError(shutdown_status);
      return shutdown_status;
    }
    execute_tasks_.push(std::move(task));

    return outputs;
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
    auto materialization_stages =
        ProcessMaterializationTask(task, mlir_context);

    // Enqueue as many execution tasks as were successfully created.
    ABSL_VLOG(1) << "[MaterializationWorker] Enqueuing "
                 << materialization_stages.execution_tasks.size()
                 << " ExecutionTasks";
    {
      absl::MutexLock lock(execute_mu_);
      for (auto& execution_task : materialization_stages.execution_tasks) {
        execute_tasks_.push(std::move(execution_task));
      }
    }

    // Set the completion promise for the task.
    // If there were any errors, this typically indicates a compilation
    // failure, rather than an execution failure.
    if (auto* nodes_task = std::get_if<NodesMaterializationTask>(&task.kind)) {
      ABSL_VLOG(2) << "[MaterializationWorker] NodesMaterializationTask "
                      "processed, setting completion promise";
      if (!materialization_stages.first_error.ok()) {
        // If PrepareMaterializationTraversals succeeded, then all nodes should
        // either be in the pending-materialization state or already failed.
        // But if it failed, we need to fail them here to unblock waiters.
        for (const auto& node : nodes_task->nodes_to_materialize) {
          if (node->is_materializing()) continue;
          node->SetAsError(materialization_stages.first_error);
        }
      }
      nodes_task->completion_promise.Set(materialization_stages.first_error);
      return true;
    }

    if (auto* stream_task =
            std::get_if<StreamMaterializationTask>(&task.kind)) {
      if (materialization_stages.first_error.ok()) {
        ABSL_VLOG(2) << "[MaterializationWorker] StreamMaterializationTask "
                        "processed, setting completion promise";
        stream_task->completion_promise.Set(EventSnapshot::Record(
            stream_task->device_index, stream_task->stream_id));
      } else {
        stream_task->completion_promise.Set(materialization_stages.first_error);
      }
      return true;
    }

    auto& device_task = std::get<DeviceMaterializationTask>(task.kind);

    ABSL_VLOG(2) << "[MaterializationWorker] DeviceMaterializationTask "
                    "processed, setting completion promise";
    if (materialization_stages.first_error.ok()) {
      device_task.completion_promise.Set(
          RecordDeviceSnapshots(device_task.device_index));
    } else {
      device_task.completion_promise.Set(materialization_stages.first_error);
    }
    return true;
  }

  void MaterializeLoop() {
    // Prevent materialization worker threads from accessing
    // thread-local context states to enforce they always rely on
    // resolved configurations passed down from the dispatch thread.
    DisallowThisThreadToAccessContextState();

    // Create the MLIR context outside the loop once and reuse for
    // materialization tasks.
    absl_nonnull std::unique_ptr<mlir::MLIRContext> mlir_context =
        MakeMlirContext();
    while (MaterializeLoopStep(*mlir_context)) {
      // MaterializeLoopStep will return false when it dequeues a
      // ShutdownSentinel.
    }
  }

  void FailRemainingMaterializationTasks() {
    absl::MutexLock lock(materialize_mu_);
    if (materialize_tasks_.empty()) return;

    // materialize_tasks_ should be empty, but we clear it to be safe.
    ABSL_LOG(WARNING) << "[MaterializationWorker::Shutdown] "
                      << "materialize_tasks_ is not empty after shutdown "
                         "sentinel. Discarding "
                      << materialize_tasks_.size() << " tasks";
    auto shutdown_status =
        TT_ERROR(error::kCancelled)  // ERROR_COV_INFEASIBLE=unreachable
                                     // from Python
        << "MaterializationWorker is shutting down";
    while (!materialize_tasks_.empty()) {
      auto& task_or_shutdown = materialize_tasks_.front();
      if (auto* materialization_task =
              std::get_if<MaterializationTask>(&task_or_shutdown)) {
        if (auto* nodes_task = std::get_if<NodesMaterializationTask>(
                &materialization_task->kind)) {
          for (const auto& node : nodes_task->nodes_to_materialize) {
            node->SetAsError(shutdown_status);
          }
          nodes_task->completion_promise.Set(shutdown_status);
        } else if (auto* stream_task = std::get_if<StreamMaterializationTask>(
                       &materialization_task->kind)) {
          stream_task->completion_promise.Set(shutdown_status);
        } else if (auto* device_task = std::get_if<DeviceMaterializationTask>(
                       &materialization_task->kind)) {
          device_task->completion_promise.Set(shutdown_status);
        }
      }
      materialize_tasks_.pop();
    }
  }

  void ExecuteLoop() {
    // Prevent execution worker threads from accessing thread-local context
    // states to enforce they always rely on resolved configurations passed
    // down from the dispatch thread.
    DisallowThisThreadToAccessContextState();

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

// Common pathway for all Materialize() overloads.
absl::Status MaterializeImpl(
    std::vector<SharedDeviceBufferList>& nodes_to_materialize,
    MaterializationReason reason, MaterializationMode materialization_mode) {
  if (nodes_to_materialize.empty()) {
    return absl::OkStatus();
  }
  if (std::any_of(nodes_to_materialize.begin(), nodes_to_materialize.end(),
                  [](const SharedDeviceBufferList& node) {
                    return node->is_placeholder();
                  })) {
    return TT_ERROR(error::kInternal)
           << "cannot Materialize() a placeholder tensor or a tensor that "
              "depends on a placeholder. \nPlaceholder tensors should only "
              "appear in compiled mode, which should never try to materialize "
              "tensors";
  }

  tsl::profiler::TraceMe t("MaterializeImpl");

  // Deduplicate nodes_to_materialize.
  {
    absl::flat_hash_set<const DeviceBufferList*> unique_nodes;
    std::erase_if(nodes_to_materialize,
                  [&unique_nodes](const SharedDeviceBufferList& node) {
                    return !unique_nodes.insert(node.get()).second;
                  });
  }
  ABSL_VLOG(1) << "[MaterializeImpl] Materializing "
               << nodes_to_materialize.size() << " nodes";

  auto future = GetMaterializationWorker().EnqueueNodes(
      nodes_to_materialize, reason, materialization_mode);  // intentional copy
  TT_RETURN_IF_ERROR(future.Await()).SetPrepend()
      << "materialization failed with: ";

  // Check that all nodes to materialize have been put into the "pending
  // materialization" state.
  for (auto& node : nodes_to_materialize) {
    TT_RET_CHECK(node->is_materializing(), error::kInternal)
        << "materialization failed for node " << node;
  }

  return absl::OkStatus();
}

}  // namespace

void ShutDownMaterializationState() { GetMaterializationWorker().Shutdown(); }

absl::Status Materialize(absl::Span<const SharedDeviceBufferList> nodes,
                         MaterializationReason reason,
                         MaterializationMode materialization_mode) {
  if (nodes.empty()) {
    return absl::OkStatus();
  }
  // Optimistically assume that all nodes are deferred.
  // Copy to a vector to allow for deduplication inside MaterializeImpl.
  // Non-deferred nodes are skipped inside the MaterializationWorker
  std::vector<SharedDeviceBufferList> nodes_to_materialize(nodes.begin(),
                                                           nodes.end());

  return MaterializeImpl(nodes_to_materialize, reason, materialization_mode);
}

absl::Status Materialize(absl::Span<const DeviceBufferRef> buffer_refs,
                         MaterializationReason reason,
                         MaterializationMode materialization_mode) {
  if (buffer_refs.empty()) {
    return absl::OkStatus();
  }
  // Optimistically assume that all refs are deferred and unique.
  // Deduplication happens in MaterializeImpl.
  // Non-deferred nodes are skipped inside the MaterializationWorker.
  std::vector<SharedDeviceBufferList> nodes_to_materialize;
  nodes_to_materialize.reserve(buffer_refs.size());
  for (const DeviceBufferRef& buffer_ref : buffer_refs) {
    nodes_to_materialize.push_back(buffer_ref.device_buffer_list());
  }
  return MaterializeImpl(nodes_to_materialize, reason, materialization_mode);
}

absl::StatusOr<std::vector<DeviceBufferRef>> EnqueueExecutable(
    SharedLoadedExecutableWithMetadata executable,
    std::vector<DeviceBufferRef> arguments,
    absl::Span<const Shape> output_shapes, std::string_view task_name) {
  // Get the stream from the calling thread, not from the execution worker
  // thread.
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  return GetMaterializationWorker().EnqueueExecutable(
      std::move(executable), std::move(arguments), output_shapes, task_name,
      device_index, stream_id);
}

absl::StatusOr<std::shared_ptr<EventSnapshot>> MaterializeStream(
    c10::DeviceIndex device_index, c10::StreamId stream_id,
    MaterializationReason reason, MaterializationMode mode) {
  auto future = GetMaterializationWorker().EnqueueStream(
      device_index, stream_id, reason, mode);
  // We have to await the future here to ensure that more work is not pushed to
  // the stream while we are evaluating its current state.
  return future.Await();
}

absl::StatusOr<std::vector<std::shared_ptr<EventSnapshot>>> MaterializeDevice(
    c10::DeviceIndex device_index, MaterializationReason reason,
    MaterializationMode mode) {
  auto future =
      GetMaterializationWorker().EnqueueDevice(device_index, reason, mode);
  // We have to await the future here to ensure that more work is not pushed to
  // the device while we are evaluating the state of its current streams.
  return future.Await();
}

absl::StatusOr<std::shared_ptr<EventSnapshot>> MaterializeCurrentStream(
    MaterializationReason reason, MaterializationMode mode) {
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  return MaterializeStream(device_index, stream_id, reason, mode);
}

}  // namespace torch_tpu
