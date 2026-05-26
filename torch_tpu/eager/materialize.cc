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

#include <cstdint>
#include <iterator>
#include <memory>
#include <queue>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/declare.h"
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
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/flags.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/status_builder.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/eager_mode.h"
#include "torch_tpu/eager/materialize_common.h"
#include "torch_tpu/eager/split_traversal.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/eager/traversal.h"
#include "torch_tpu/experimental/eager/materialize_new.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/future.h"
#include "xla/xla_data.pb.h"

ABSL_DECLARE_FLAG(bool, torch_tpu_internal_enable_new_materialization);

namespace torch_tpu {
namespace {

struct MaterializationTask {
  std::vector<SharedDeviceBufferList> nodes_to_materialize;
  xla::Promise<void> completion_promise;
  MaterializationMode materialization_mode = MaterializationMode::kSplitGraph;
  MaterializationReason reason;
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
    absl::MutexLock lock(materialize_mu_);
    materialize_jobs_.push(
        MaterializationTask{.nodes_to_materialize = std::move(nodes),
                            .completion_promise = std::move(promise),
                            .materialization_mode = materialization_mode,
                            .reason = reason});
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
      // Make a placeholders to hold the results.
      TT_ASSIGN_OR_RETURN(DeviceBufferRef output_ref,
                          DeviceBufferList::CreatePlaceholder(
                              shape.dimensions(), shape.dtype()));
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

  absl::StatusOr<std::vector<ExecutionTask>> ProcessMaterializationTask(
      MaterializationTask& task, mlir::MLIRContext& mlir_context) {
    ABSL_VLOG(1)
        << "[MaterializationWorker] Processing MaterializationTask with "
        << task.nodes_to_materialize.size() << " nodes";
    LogDeferredNodes(task.nodes_to_materialize,
                     /* msg_prefix= */ "  Input node");

    ABSL_VLOG(1) << "[MaterializationWorker] Getting leaf nodes";
    std::vector<SharedDeviceBufferList> all_nodes = task.nodes_to_materialize;

    // Filter out non-deferred nodes that may have been materialized by an
    // earlier materialization task. Return an error if any of the nodes are
    // placeholders.
    {
      absl::Status no_placeholder_status = absl::OkStatus();
      std::erase_if(all_nodes, [&no_placeholder_status](
                                   const SharedDeviceBufferList& node) {
        switch (node->state()) {
          case DeviceBufferRefState::kDeferred:
            return false;
          case DeviceBufferRefState::kPlaceholder:
            no_placeholder_status =
                TT_ERROR(error::kInternal)
                << "Materialize was called on a placeholder tensor. This "
                   "should "
                   "never happen.\nkPlaceholder tensors should only appear in "
                   "compiled mode, which should never try to materialize "
                   "tensors.";
            return true;
          case DeviceBufferRefState::kMaterialized:
            return true;
        }
      });
      if (!no_placeholder_status.ok()) {
        return no_placeholder_status;
      }
      if (all_nodes.empty()) {
        // Everything was already materialized, nothing more to do.
        return std::vector<ExecutionTask>();
      }
    }

    {
      tsl::profiler::TraceMe t("AddLeafNodes");
      AddLeafNodes(all_nodes);
    }

    ABSL_VLOG(1) << "[MaterializationWorker] Found " << all_nodes.size()
                 << " leaf nodes";
    LogDeferredNodes(all_nodes, /* msg_prefix= */ "  Output leaf node");

    ABSL_VLOG(1) << "[MaterializationWorker] Creating traversal";
    std::unique_ptr<Traversal> traversal;
    {
      tsl::profiler::TraceMe t("Traversal::Create");
      TT_ASSIGN_OR_RETURN(traversal, Traversal::Create(all_nodes));
    }

    ABSL_VLOG(3) << "[MaterializationWorker] Traversal created: "
                 << GetGraphviz(*traversal);

    std::vector<absl_nonnull std::unique_ptr<Traversal>> traversals;

    if (task.materialization_mode == MaterializationMode::kSplitGraph) {
      // Split the traversal while nodes are still in the deferred state.
      ABSL_VLOG(1) << "[MaterializationWorker] Splitting traversal";
      {
        tsl::profiler::TraceMe t("SplitTraversal");
        absl::flat_hash_set<const DeviceBufferList*> required_outputs;
        for (const auto& node : task.nodes_to_materialize) {
          required_outputs.insert(node.get());
        }
        TT_ASSIGN_OR_RETURN(
            traversals, SplitTraversal(std::move(traversal), required_outputs));
      }

      ABSL_VLOG(1) << "[MaterializationWorker] Split traversal into "
                   << traversals.size() << " traversals";
    } else {
      traversals.push_back(std::move(traversal));
    }

    std::vector<ExecutionTask> execution_tasks;
    for (auto& split_traversal : traversals) {
      TT_ASSIGN_OR_RETURN(
          ExecutionTask execution_task,
          ExecutionTask::FromTraversalWithLogging(std::move(split_traversal),
                                                  mlir_context, task.reason));
      execution_tasks.push_back(std::move(execution_task));
    }

    return execution_tasks;
  }

  void StartThreads() {
    materialize_thread_ =
        std::thread(
            [this]() {
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
  tsl::profiler::TraceMe t("MaterializeImpl");

  if (GetFlagOnce<bool,
                  &FLAGS_torch_tpu_internal_enable_new_materialization>()) {
    TT_RETURN_IF_ERROR(MaterializeImplNew(nodes_to_materialize, reason));
    if (GetEagerMode() == EagerMode::kDeferNeverAndLaunchBlocking) {
      TT_RETURN_IF_ERROR(BlockOnPendingMaterializations());
    }
    return absl::OkStatus();
  }

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

  // Check that all nodes to materialize have indeed been materialized.
  for (auto& node : nodes_to_materialize) {
    TT_RET_CHECK(node->state() == DeviceBufferRefState::kMaterialized,
                 error::kInternal)
        << "Materialization failed for node " << node;
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

void SetOutputNodesAsError(absl::Span<const DeviceBufferRef> outputs,
                           absl::Status status) {
  std::vector<SharedDeviceBufferList> nodes;
  nodes.reserve(outputs.size());
  for (const auto& output : outputs) {
    nodes.push_back(output.device_buffer_list());
  }
  SetOutputNodesAsError(absl::MakeSpan(nodes), status);
}

void SetOutputNodesAsError(absl::Span<const SharedDeviceBufferList> outputs,
                           absl::Status status) {
  ABSL_VLOG(1) << "[SetOutputNodesAsError] Starting";
  for (const auto& output : outputs) {
    output->SetAsError(status);
  }
  ABSL_VLOG(1) << "[SetOutputNodesAsError] Set error for nodes";
}

absl::StatusOr<std::vector<DeviceBufferRef>> EnqueueExecutable(
    SharedLoadedExecutableWithMetadata executable,
    std::vector<DeviceBufferRef> arguments,
    absl::Span<const Shape> output_shapes, std::string_view task_name) {
  return GetMaterializationWorker().EnqueueExecutable(
      std::move(executable), std::move(arguments), output_shapes, task_name);
}

void AddLeafNodes(std::vector<SharedDeviceBufferList>& nodes) {
  std::vector<SharedDeviceBufferList> leaf_nodes;

  // Each root subgraph only needs to be processed once.
  absl::flat_hash_set<Subgraph*> unique_roots;
  // If a task's output is also a leaf node, we don't need to include it
  // twice. We'll retain the original nodes in order, and append the leaf nodes
  // to the end.
  absl::flat_hash_set<const DeviceBufferList*> all_nodes_set;
  for (const auto& node : nodes) {
    all_nodes_set.insert(node.get());
  }

  for (const auto& node : nodes) {
    // Get the root of the deferred op and check that we haven't processed it
    // yet. Skip if there's no op, no subgraph, or a non-unique root.
    const auto deferred_op = node->deferred_op();
    if (!deferred_op) continue;

    std::shared_ptr<Subgraph> subgraph = deferred_op->subgraph();
    if (!subgraph) continue;

    std::shared_ptr<Subgraph> root = subgraph->Find();
    if (!unique_roots.insert(root.get()).second) continue;

    std::vector<SharedDeviceBufferList> subgraph_leaf_nodes =
        root->GetLeafNodes();
    for (auto& leaf_node : subgraph_leaf_nodes) {
      if (all_nodes_set.insert(leaf_node.get()).second) {
        leaf_nodes.push_back(std::move(leaf_node));
      }
    }
  }

  nodes.insert(nodes.end(), std::make_move_iterator(leaf_nodes.begin()),
               std::make_move_iterator(leaf_nodes.end()));
}

}  // namespace torch_tpu
