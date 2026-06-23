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

#include "torch_tpu/eager/materialize.h"

#include <unistd.h>

#include <algorithm>
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
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/compilation_spec.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/status_builder.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/eager_mode.h"
#include "torch_tpu/eager/events_queue.h"
#include "torch_tpu/eager/materialize_common.h"
#include "torch_tpu/eager/split_traversal.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/eager/traversal.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/future.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

// A task to materialize a specific list of nodes on a stream.
//
// This is a "background materialization" only. It updates the nodes from
// deferred state to pending, which is a TorchTPU internal state transition
// not visible to Python.
//
// This does *not* consider device transfers.
struct NodesMaterialization {
  // The device to materialize on. All nodes_to_materialize must be on this
  // device.
  c10::DeviceIndex device_index;
  // The stream to materialize. All nodes_to_materialize must be on this stream.
  c10::StreamId stream_id;
  // The nodes to materialize. This will also materialize all live tensors that
  // were dispatched earlier on the same stream. Tensors created after the last
  // node in nodes_to_materialize will not be materialized.
  std::vector<SharedDeviceBufferList> nodes_to_materialize;
  // When all nodes_to_materialize have been set to the pending state, this
  // promise will be fulfilled.
  xla::Promise<void> nodes_pending_promise;
};

// A task to materialize all deferred operations on a specific stream.
// This is used to synchronize the CPU and a specific stream, or to sync streams
// with each other.
struct StreamMaterialization {
  // The device containing the stream to materialize.
  c10::DeviceIndex device_index;
  // The stream to materialize.
  c10::StreamId stream_id;
  // When the state of the stream has been captured and materialization has
  // been enqueued (if necessary), this promise will be fulfilled with a
  // snapshot of the stream. When this snapshot is ready, the stream is
  // synchronized to that point, including deferred op materializations and
  // device transfers. If the stream is already caught up, the promise will be
  // fulfilled with nullptr.
  xla::Promise<absl_nullable std::shared_ptr<EventSnapshot>>
      snapshot_created_promise;
};

// A task to materialize all deferred operations on a specific device.
// This is used to synchronize the CPU and a specific device, or to sync devices
// with each other.
struct DeviceMaterialization {
  // The device to materialize.
  c10::DeviceIndex device_index;
  // When the state of the device has been captured and materialization has
  // been enqueued (if necessary), this promise will be fulfilled with a list of
  // snapshots, one per active stream on the device. When these snapshots are
  // ready, the device is synchronized to that point, including deferred op
  // materializations and device transfers. If the device is already caught up,
  // the promise will be fulfilled with an empty vector.
  xla::Promise<std::vector<absl_nonnull std::shared_ptr<EventSnapshot>>>
      snapshots_created_promise;
};

// A task to materialize all deferred operations on all devices.
struct GlobalMaterialization {
  // When the state of all devices has been captured and materialization has
  // been enqueued (if necessary), this promise will be fulfilled with a list of
  // snapshots, one per active stream on each device. When these snapshots are
  // ready, all devices are synchronized to that point, including deferred op
  // materializations and device transfers. If everything is already caught up,
  // the promise will be fulfilled with an empty vector.
  xla::Promise<std::vector<absl_nonnull std::shared_ptr<EventSnapshot>>>
      snapshots_created_promise;
};

// Common fields for all materialization tasks.
struct MaterializationCommon {
  MaterializationMode materialization_mode = MaterializationMode::kSplitGraph;
  MaterializationReason reason;
  CompilationSpec compilation_spec;
};

struct MaterializationTask {
  std::variant<NodesMaterialization, StreamMaterialization,
               DeviceMaterialization, GlobalMaterialization>
      task;
  MaterializationCommon common;
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

// Converts a sequence of Traversals to a sequence of ExecutionTasks.
// If materialization_mode is kSplitGraph, then there may be more returned
// tasks than there were original traversals; otherwise, there will be exactly
// one ExecutionTask per traversal.
absl::StatusOr<std::vector<ExecutionTask>> ApplySplitMode(
    std::vector<absl_nonnull std::unique_ptr<Traversal>>&& traversals,
    const MaterializationCommon& common, mlir::MLIRContext& mlir_context) {
  if (common.materialization_mode == MaterializationMode::kSplitGraph) {
    tsl::profiler::TraceMe t("SplitTraversal");
    std::vector<absl_nonnull std::unique_ptr<Traversal>> split_traversals;
    std::vector<absl_nonnull std::unique_ptr<Traversal>> post_split_traversals;
    absl::flat_hash_set<const DeviceBufferList*> required_outputs;
    for (auto& pre_split_traversal : traversals) {
      for (const auto& output : pre_split_traversal->outputs()) {
        required_outputs.insert(output.device_buffer_list().get());
      }
      TT_ASSIGN_OR_RETURN(
          split_traversals,
          SplitTraversal(std::move(pre_split_traversal), required_outputs));
      post_split_traversals.insert(
          post_split_traversals.end(),
          std::make_move_iterator(split_traversals.begin()),
          std::make_move_iterator(split_traversals.end()));
      required_outputs.clear();
      split_traversals.clear();
    }
    std::swap(traversals, post_split_traversals);
  }

  std::vector<ExecutionTask> execution_tasks;
  execution_tasks.reserve(traversals.size());
  for (auto& split_traversal : traversals) {
    auto execution_task_or = ExecutionTask::FromTraversalWithLogging(
        std::move(split_traversal), mlir_context,
        common.compilation_spec.Copy(), common.reason);
    if (!execution_task_or.ok()) {
      // Fail the execution tasks we already created to ensure anything
      // waiting on their outputs will not deadlock.
      for (auto& task : execution_tasks) {
        task.SetOutputNodesAsError(execution_task_or.status());
      }
      return execution_task_or.status();
    }
    execution_tasks.push_back(std::move(*execution_task_or));
  }
  return execution_tasks;
}

// Prepares execution tasks for a NodesMaterialization task.
// Sets the xla::Promise<void> in the NodesMaterialization task when done.
absl::StatusOr<std::vector<ExecutionTask>> ProcessNodesMaterialization(
    NodesMaterialization& task, const MaterializationCommon& common,
    mlir::MLIRContext& mlir_context) {
  ABSL_VLOG(1) << "[MaterializationWorker] Processing MaterializationTask with "
               << task.nodes_to_materialize.size() << " nodes";
  LogDeferredNodes(task.nodes_to_materialize,
                   /* msg_prefix= */ "  Input node");

  std::vector<SharedDeviceBufferList> all_nodes = task.nodes_to_materialize;

  // Filter out non-deferred nodes that may have been materialized by an
  // earlier materialization task.
  std::erase_if(all_nodes, [](const SharedDeviceBufferList& node) {
    return !node->is_deferred();
  });
  if (all_nodes.empty()) {
    // Everything was already materialized, nothing more to do.
    return std::vector<ExecutionTask>();
  }

  TT_ASSIGN_OR_RETURN(
      std::vector<absl_nonnull std::unique_ptr<Traversal>> traversals,
      PrepareMaterializationTraversals(all_nodes));
  TT_ASSIGN_OR_RETURN(
      std::vector<ExecutionTask> execution_tasks,
      ApplySplitMode(std::move(traversals), common, mlir_context));

  task.nodes_pending_promise.Set(absl::OkStatus());

  return execution_tasks;
}

// Prepares execution tasks for a StreamMaterialization task.
// Sets the xla::Promise for the single EventSnapshot when done.
absl::StatusOr<std::vector<ExecutionTask>> ProcessStreamMaterialization(
    StreamMaterialization& task, const MaterializationCommon& common,
    mlir::MLIRContext& mlir_context) {
  TT_ASSIGN_OR_RETURN(auto recorded_event_snapshot,
                      EventSnapshot::Record(task.device_index, task.stream_id));
  std::vector<ExecutionTask> execution_tasks;
  if (recorded_event_snapshot.event_snapshot == nullptr) {
    TT_RET_CHECK(recorded_event_snapshot.traversals.empty(), error::kInternal)
        << "must have a nonnull snapshot if there are traversals to "
           "materialize";
  } else if (!recorded_event_snapshot.traversals.empty()) {
    TT_ASSIGN_OR_RETURN(
        execution_tasks,
        ApplySplitMode(std::move(recorded_event_snapshot.traversals), common,
                       mlir_context));
  }
  task.snapshot_created_promise.Set(
      std::move(recorded_event_snapshot.event_snapshot));
  return execution_tasks;
}

// Prepares execution tasks for a DeviceMaterialization or GlobalMaterialization
// task. Sets the xla::Promise for the list of EventSnapshots when done.
absl::StatusOr<std::vector<ExecutionTask>> ProcessEventSnapshots(
    std::vector<EventSnapshot::RecordedEventSnapshot>&&
        recorded_event_snapshots,
    xla::Promise<std::vector<absl_nonnull std::shared_ptr<EventSnapshot>>>
        snapshots_created_promise,
    const MaterializationCommon& common, mlir::MLIRContext& mlir_context) {
  std::vector<ExecutionTask> execution_tasks;
  std::vector<absl_nonnull std::shared_ptr<EventSnapshot>> event_snapshots;
  for (auto& recorded_event_snapshot : recorded_event_snapshots) {
    if (recorded_event_snapshot.traversals.empty()) {
      event_snapshots.push_back(
          std::move(recorded_event_snapshot.event_snapshot));
      continue;
    }

    TT_ASSIGN_OR_RETURN(
        auto execution_tasks,
        ApplySplitMode(std::move(recorded_event_snapshot.traversals), common,
                       mlir_context));
    execution_tasks.insert(execution_tasks.end(),
                           std::make_move_iterator(execution_tasks.begin()),
                           std::make_move_iterator(execution_tasks.end()));
  }
  snapshots_created_promise.Set(std::move(event_snapshots));
  return execution_tasks;
}

absl::StatusOr<std::vector<ExecutionTask>> ProcessDeviceMaterialization(
    DeviceMaterialization& task, const MaterializationCommon& common,
    mlir::MLIRContext& mlir_context) {
  TT_ASSIGN_OR_RETURN(auto recorded_event_snapshots,
                      RecordDeviceSnapshot(task.device_index));
  TT_ASSIGN_OR_RETURN(
      auto execution_tasks,
      ProcessEventSnapshots(std::move(recorded_event_snapshots),
                            std::move(task.snapshots_created_promise), common,
                            mlir_context));
  return execution_tasks;
}

absl::StatusOr<std::vector<ExecutionTask>> ProcessGlobalMaterialization(
    GlobalMaterialization& task, const MaterializationCommon& common,
    mlir::MLIRContext& mlir_context) {
  TT_ASSIGN_OR_RETURN(auto recorded_event_snapshots, RecordGlobalSnapshot());
  TT_ASSIGN_OR_RETURN(
      auto execution_tasks,
      ProcessEventSnapshots(std::move(recorded_event_snapshots),
                            std::move(task.snapshots_created_promise), common,
                            mlir_context));
  return execution_tasks;
}

// Converts a MaterializationTask to a sequence of ExecutionTasks.
absl::StatusOr<std::vector<ExecutionTask>> ProcessMaterializationTask(
    MaterializationTask& task, mlir::MLIRContext& mlir_context) {
  if (auto* nodes_materialization =
          std::get_if<NodesMaterialization>(&task.task)) {
    return ProcessNodesMaterialization(*nodes_materialization, task.common,
                                       mlir_context);
  } else if (auto* stream_materialization =
                 std::get_if<StreamMaterialization>(&task.task)) {
    return ProcessStreamMaterialization(*stream_materialization, task.common,
                                        mlir_context);
  } else if (auto* device_materialization =
                 std::get_if<DeviceMaterialization>(&task.task)) {
    return ProcessDeviceMaterialization(*device_materialization, task.common,
                                        mlir_context);
  } else if (auto* global_materialization =
                 std::get_if<GlobalMaterialization>(&task.task)) {
    return ProcessGlobalMaterialization(*global_materialization, task.common,
                                        mlir_context);
  } else {
    return TT_ERROR(error::kInternal) << "Unknown materialization task type";
  }
}

class MaterializationWorker {
 public:
  // This class is move-only.
  MaterializationWorker(MaterializationWorker&& other) = default;
  MaterializationWorker& operator=(MaterializationWorker&& other) = default;
  MaterializationWorker(const MaterializationWorker&) = delete;
  MaterializationWorker& operator=(const MaterializationWorker&) = delete;

  MaterializationWorker() { StartThreads(); }

  xla::Future<void> EnqueueNodes(const c10::DeviceIndex device_index,
                                 const c10::StreamId stream_id,
                                 std::vector<SharedDeviceBufferList> nodes,
                                 MaterializationReason reason,
                                 MaterializationMode materialization_mode) {
    ABSL_VLOG(1) << "[MaterializationWorker] Enqueuing " << nodes.size()
                 << " nodes for materialization";
    auto [promise, future] = xla::MakePromise<void>();
    const CompilationMode compilation_mode = GetCompilationMode(GetEagerMode());

    absl::MutexLock lock(materialize_mu_);
    materialize_jobs_.push(MaterializationTask{
        .task =
            NodesMaterialization{
                .device_index = device_index,
                .stream_id = stream_id,
                .nodes_to_materialize = std::move(nodes),
                .nodes_pending_promise = std::move(promise),
            },
        .common = {.materialization_mode = materialization_mode,
                   .reason = reason,
                   .compilation_spec = GetCompilationSpec(compilation_mode)}});
    return future;
  }

  xla::Future<absl_nonnull std::shared_ptr<EventSnapshot>> EnqueueStream(
      c10::DeviceIndex device_index, c10::StreamId stream_id,
      MaterializationReason reason, MaterializationMode materialization_mode) {
    auto [promise, future] =
        xla::MakePromise<absl_nonnull std::shared_ptr<EventSnapshot>>();
    const CompilationMode compilation_mode = GetCompilationMode(GetEagerMode());
    absl::MutexLock lock(materialize_mu_);
    materialize_jobs_.push(MaterializationTask{
        .task =
            StreamMaterialization{
                .device_index = device_index,
                .stream_id = stream_id,
                .snapshot_created_promise = std::move(promise),
            },
        .common = {.materialization_mode = materialization_mode,
                   .reason = reason,
                   .compilation_spec = GetCompilationSpec(compilation_mode)}});
    return future;
  }

  xla::Future<std::vector<absl_nonnull std::shared_ptr<EventSnapshot>>>
  EnqueueDevice(c10::DeviceIndex device_index, MaterializationReason reason,
                MaterializationMode materialization_mode) {
    auto [promise, future] = xla::MakePromise<
        std::vector<absl_nonnull std::shared_ptr<EventSnapshot>>>();
    const CompilationMode compilation_mode = GetCompilationMode(GetEagerMode());
    absl::MutexLock lock(materialize_mu_);
    materialize_jobs_.push(MaterializationTask{
        .task =
            DeviceMaterialization{
                .device_index = device_index,
                .snapshots_created_promise = std::move(promise),
            },
        .common = {.materialization_mode = materialization_mode,
                   .reason = reason,
                   .compilation_spec = GetCompilationSpec(compilation_mode)}});
    return future;
  }

  xla::Future<std::vector<absl_nonnull std::shared_ptr<EventSnapshot>>>
  EnqueueGlobal(MaterializationReason reason,
                MaterializationMode materialization_mode) {
    auto [promise, future] = xla::MakePromise<
        std::vector<absl_nonnull std::shared_ptr<EventSnapshot>>>();
    const CompilationMode compilation_mode = GetCompilationMode(GetEagerMode());
    absl::MutexLock lock(materialize_mu_);
    materialize_jobs_.push(MaterializationTask{
        .task =
            GlobalMaterialization{
                .snapshots_created_promise = std::move(promise),
            },
        .common = {.materialization_mode = materialization_mode,
                   .reason = reason,
                   .compilation_spec = GetCompilationSpec(compilation_mode)}});
    return future;
  }

  absl::StatusOr<std::vector<DeviceBufferRef>> EnqueueExecutable(
      SharedLoadedExecutableWithMetadata executable,
      std::vector<DeviceBufferRef> arguments,
      absl::Span<const Shape> output_shapes, std::string_view task_name = "") {
    // Create a set of output DeviceBufferRefs to hold the materialized results.
    std::vector<DeviceBufferRef> outputs;
    outputs.reserve(output_shapes.size());
    for (const auto& shape : output_shapes) {
      // Create pending buffer lists to hold the results.
      TT_ASSIGN_OR_RETURN(
          DeviceBufferRef output_ref,
          DeviceBufferList::CreatePending(shape.dimensions(), shape.dtype()));
      outputs.push_back(std::move(output_ref));
    }

    // Intentional copy on outputs; we need to both include them in the task
    // and return them to the caller.
    TT_ASSIGN_OR_RETURN(
        ExecutionTask task,
        ExecutionTask::FromExecutable(
            std::move(executable), std::move(arguments), outputs,
            /*reason=*/MaterializationReason::kCompileModeExecution,
            task_name));

    absl::MutexLock lock(execute_mu_);
    execute_jobs_.push(std::move(task));

    return outputs;
  }

 private:
  MaterializationTask DequeueMaterializationJob() {
    absl::MutexLock lock(materialize_mu_);
    if (materialize_jobs_.empty()) {
      materialize_mu_.Await(absl::Condition(
          +[](std::queue<MaterializationTask>* jobs) { return !jobs->empty(); },
          &materialize_jobs_));
    }
    MaterializationTask popped_job = std::move(materialize_jobs_.front());
    materialize_jobs_.pop();
    return popped_job;
  }

  ExecutionTask DequeueExecutionJob() {
    absl::MutexLock lock(execute_mu_);
    if (execute_jobs_.empty()) {
      execute_mu_.Await(absl::Condition(
          +[](std::queue<ExecutionTask>* jobs) { return !jobs->empty(); },
          &execute_jobs_));
    }
    ExecutionTask popped_job = std::move(execute_jobs_.front());
    execute_jobs_.pop();
    return popped_job;
  }

  void StartThreads() {
    materialize_thread_ = std::thread([this]() {
      // Create the MLIR context outside the loop once and reuse for
      // materialization tasks.
      absl_nonnull std::unique_ptr<mlir::MLIRContext> mlir_context =
          MakeMlirContext();
      while (true) {
        MaterializationTask job = DequeueMaterializationJob();
        ABSL_VLOG(1)
            << "[MaterializationWorker] Processing MaterializationTask";
        absl::StatusOr<std::vector<ExecutionTask>> tasks =
            ProcessMaterializationTask(job, *mlir_context);

        if (tasks.ok()) {
          ABSL_VLOG(1) << "[MaterializationWorker] Enqueuing " << tasks->size()
                       << " ExecutionTasks";
          {
            absl::MutexLock lock(execute_mu_);
            for (auto& task : *tasks) {
              execute_jobs_.push(std::move(task));
            }
          }
        } else if (const auto* nodes_materialization =
                       std::get_if<NodesMaterialization>(&job.task)) {
          // This typically indicates a compilation failure, rather than
          // an execution failure.
          // Mark all nodes in the job as materialization failures so
          // that AwaitBuffer() will return the compilation error
          // instead of hanging.
          for (const auto& node : nodes_materialization->nodes_to_materialize) {
            node->SetAsError(tasks.status());
          }
        }
      }
    });

    execute_thread_ = std::thread([this]() {
      while (true) {
        ExecutionTask job = DequeueExecutionJob();
        ABSL_VLOG(1) << "[MaterializationWorker] Processing ExecutionTask";
        auto status = job.Run();
        if (!status.ok()) {
          ABSL_LOG(ERROR) << "[MaterializationWorker] ExecutionTask failed: "
                          << status;
        }
      }
    });
  }

  std::thread materialize_thread_;
  std::thread execute_thread_;

  absl::Mutex materialize_mu_;
  std::queue<MaterializationTask> materialize_jobs_
      ABSL_GUARDED_BY(materialize_mu_);

  absl::Mutex execute_mu_;
  std::queue<ExecutionTask> execute_jobs_ ABSL_GUARDED_BY(execute_mu_);
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
  const auto& first_node = nodes_to_materialize.front();
  const c10::DeviceIndex device_index = first_node->device_index();
  const c10::StreamId stream_id = first_node->stream_id();
  if (nodes_to_materialize.size() > 1 &&
      std::any_of(
          nodes_to_materialize.begin(), nodes_to_materialize.end(),
          [device_index, stream_id](const SharedDeviceBufferList& node) {
            return node->device_index() != device_index ||
                   node->stream_id() != stream_id;
          })) {
    return TT_ERROR(error::kInvalidArgument)
           << "multi-node materializations must be on a single device and "
              "stream";
  }

  if (std::any_of(nodes_to_materialize.begin(), nodes_to_materialize.end(),
                  [](const SharedDeviceBufferList& node) {
                    return node->is_placeholder() ||
                           node->depends_on_placeholder();
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
      device_index, stream_id, nodes_to_materialize, reason,
      materialization_mode);  // intentional copy
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
  return GetMaterializationWorker().EnqueueExecutable(
      std::move(executable), std::move(arguments), output_shapes, task_name);
}

xla::Future<absl_nonnull std::shared_ptr<EventSnapshot>> MaterializeStream(
    c10::DeviceIndex device_index, c10::StreamId stream_id,
    MaterializationReason reason, MaterializationMode mode) {
  return GetMaterializationWorker().EnqueueStream(device_index, stream_id,
                                                  reason, mode);
}

// Materializes all deferred operations on the given device.
// Returns a future that will resolve to a list of awaitable EventSnapshots
// (one per active stream on the device).
xla::Future<std::vector<absl_nonnull std::shared_ptr<EventSnapshot>>>
MaterializeDevice(c10::DeviceIndex device_index, MaterializationReason reason,
                  MaterializationMode mode) {
  return GetMaterializationWorker().EnqueueDevice(device_index, reason, mode);
}

xla::Future<std::vector<absl_nonnull std::shared_ptr<EventSnapshot>>>
MaterializeGlobal(MaterializationReason reason, MaterializationMode mode) {
  return GetMaterializationWorker().EnqueueGlobal(reason, mode);
}

}  // namespace torch_tpu
