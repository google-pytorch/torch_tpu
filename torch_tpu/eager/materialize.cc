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
#include "mlir/IR/MLIRContext.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/compilation_spec.h"
#include "torch_tpu/common/context_manager.h"
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

struct MaterializationTask {
  std::vector<SharedDeviceBufferList> nodes_to_materialize;
  xla::Promise<void> completion_promise;
  MaterializationMode materialization_mode = MaterializationMode::kSplitGraph;
  MaterializationReason reason;
  CompilationSpec compilation_spec;
};

using MaterializationJob = std::variant<ExecutionTask, MaterializationTask>;

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
    MaterializationMode materialization_mode,
    const CompilationSpec& compilation_spec, MaterializationReason reason,
    mlir::MLIRContext& mlir_context) {
  if (materialization_mode == MaterializationMode::kSplitGraph) {
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
        std::move(split_traversal), mlir_context, compilation_spec.Copy(),
        reason);
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

// Processes a MaterializationTask, converting it into a sequence of
// ExecutionTasks.
absl::StatusOr<std::vector<ExecutionTask>> ProcessMaterializationTask(
    MaterializationTask& task, mlir::MLIRContext& mlir_context) {
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

  return ApplySplitMode(std::move(traversals), task.materialization_mode,
                        task.compilation_spec, task.reason, mlir_context);
}

class MaterializationWorker {
 public:
  // This class is move-only.
  MaterializationWorker(MaterializationWorker&& other) = default;
  MaterializationWorker& operator=(MaterializationWorker&& other) = default;
  MaterializationWorker(const MaterializationWorker&) = delete;
  MaterializationWorker& operator=(const MaterializationWorker&) = delete;

  MaterializationWorker() { StartThreads(); }

  xla::Future<void> EnqueueNodes(std::vector<SharedDeviceBufferList> nodes,
                                 MaterializationReason reason,
                                 MaterializationMode materialization_mode) {
    ABSL_VLOG(1) << "[MaterializationWorker] Enqueuing " << nodes.size()
                 << " nodes for materialization";
    auto [promise, future] = xla::MakePromise<void>();
    const CompilationMode compilation_mode = GetCompilationMode(GetEagerMode());

    absl::MutexLock lock(materialize_mu_);
    materialize_jobs_.push(MaterializationTask{
        .nodes_to_materialize = std::move(nodes),
        .completion_promise = std::move(promise),
        .materialization_mode = materialization_mode,
        .reason = reason,
        .compilation_spec = GetCompilationSpec(compilation_mode),
    });
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
    materialize_thread_ =
        std::thread(
            [this]() {
              // Prevent materialization worker threads from accessing
              // thread-local context states to enforce they always rely on
              // resolved configurations passed down from the dispatch thread.
              DisallowThisThreadToAccessContextState();

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
                  ABSL_VLOG(1) << "[MaterializationWorker] Enqueuing "
                               << tasks->size() << " ExecutionTasks";
                  {
                    absl::MutexLock lock(execute_mu_);
                    for (auto& task : *tasks) {
                      execute_jobs_.push(std::move(task));
                    }
                  }
                } else {
                  // This typically indicates a compilation failure, rather than
                  // an execution failure.
                  // Mark all nodes in the job as materialization failures so
                  // that AwaitBuffer() will return the compilation error
                  // instead of hanging.
                  for (const auto& node : job.nodes_to_materialize) {
                    node->SetAsError(tasks.status());
                  }
                }

                job.completion_promise.Set(tasks.status());
              }
            });

    execute_thread_ = std::thread([this]() {
      while (true) {
        // Prevent materialization worker threads from accessing thread-local
        // context states to enforce they always rely on resolved configurations
        // passed down from the dispatch thread.
        DisallowThisThreadToAccessContextState();

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

}  // namespace torch_tpu
