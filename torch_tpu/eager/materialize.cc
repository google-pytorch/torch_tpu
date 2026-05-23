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

#include "ATen/core/TensorBody.h"
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
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/flags.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/eager_mode.h"
#include "torch_tpu/eager/materialize_common.h"
#include "torch_tpu/eager/split_traversal.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/eager/traversal.h"
#include "torch_tpu/experimental/eager/materialize_new.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/future.h"
#include "xla/hlo/translate/register.h"
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
      ABSL_VLOG(1) << msg_prefix << i << ": " << node.get()
                   << (node->deferred_op()
                           ? absl::StrCat(" op: ",
                                          node->deferred_op()->op_name())
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
      TT_ASSIGN_OR_RETURN(
          DeviceBufferRef output_ref,
          DeviceBufferList::MakePlaceholder(shape.dimensions(), shape.dtype()));
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
    {
      tsl::profiler::TraceMe t("AddLeafNodes");
      AddLeafNodes(all_nodes);
    }

    ABSL_VLOG(1) << "[MaterializationWorker] Found " << all_nodes.size()
                 << " leaf nodes";
    LogDeferredNodes(all_nodes, /* msg_prefix= */ "  Output leaf node");

    if (all_nodes.empty()) {
      return std::vector<ExecutionTask>();
    }

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

// Translates a raw background execution Out-of-Memory (OOM) error to a
// high-level exception that is easy to understand by PyTorch users. Under
// experimental asynchronous materialization, compilation and execution are
// enqueued to the background thread. If an execution OOM happens in the
// background, it is caught early during BlockOnPendingMaterializations() inside
// GetMaterialized(), bypassing the standard copy-transfer wrapper's
// (TranslateXlaTensorOomError) execution blocks. We replicate the OOM mapping
// here to ensure user exception signature consistency.
absl::Status TranslateXlaOomError(const absl::Status& status,
                                  const at::Tensor& tensor) {
  if (!IsXlaOomError(status)) {
    return status;
  }
  auto dtype_or = ConvertTo<mlir::ElementType>(tensor.scalar_type());
  if (!dtype_or.ok()) {
    return status;
  }
  return TT_ERROR(error::kResourceExhausted)
         << "the TPU ran out of memory while awaiting the materialization of "
            "value "
         << ToString(*dtype_or) << "[" << absl::StrJoin(tensor.sizes(), ", ")
         << "]:\n"
         << status.message();
}

absl::Status MaybeBlockOnPendingMaterializationsNew(const at::Tensor& tensor) {
  if (GetFlagOnce<bool,
                  &FLAGS_torch_tpu_internal_enable_new_materialization>()) {
    // Under experimental asynchronous materialization, the main thread blocks
    // and awaits for the background worker compilation and execution thread to
    // catch up before returning the resolved device buffer reference.
    auto status = BlockOnPendingMaterializations();
    if (!status.ok()) {
      if (IsXlaOomError(status)) {
        // 1. If it is an execution OOM error, translate it to standard format
        //    while preserving the default copy caller context prefix
        //    ("to_copy").
        return TranslateXlaOomError(status, tensor);
      } else {
        // 2. If it is a compilation error, wrap the existing status natively
        //    via StatusBuilder to preserve the background thread's original
        //    operator name context payload (e.g., "gather", "scatter",
        //    "cumprod") captured during graph construction under
        //    ScopedPythonContextProvider. Do NOT reconstruct the status (e.g.
        //    using TT_ERROR), as that would overwrite the payload with the copy
        //    thread's "to_copy" wrapper.
        return ::torch_tpu::StatusBuilder(std::move(status)).SetPrepend()
               << "materialization failed with: ";
      }
    }
  }
  return absl::OkStatus();
}

// Common pathway for all Materialize() overloads.
absl::Status MaterializeImpl(
    absl::Span<const SharedDeviceBufferList> nodes_to_materialize,
    MaterializationReason reason, MaterializationMode materialization_mode) {
  tsl::profiler::TraceMe t([] { return "MaterializeImpl"; });

  if (GetFlagOnce<bool,
                  &FLAGS_torch_tpu_internal_enable_new_materialization>()) {
    TT_RETURN_IF_ERROR(MaterializeImplNew(nodes_to_materialize, reason));
    if (GetEagerMode() == EagerMode::kDeferNeverAndLaunchBlocking) {
      TT_RETURN_IF_ERROR(BlockOnPendingMaterializations());
    }
    return absl::OkStatus();
  }

  ABSL_VLOG(1) << "[MaterializeImpl] Materializing "
               << nodes_to_materialize.size() << " nodes";
  if (nodes_to_materialize.empty()) {
    return absl::OkStatus();
  }

  std::vector<SharedDeviceBufferList> nodes(nodes_to_materialize.begin(),
                                            nodes_to_materialize.end());

  auto future = GetMaterializationWorker().EnqueueNodes(
      std::move(nodes), reason, materialization_mode);
  TT_RETURN_IF_ERROR(future.Await()).SetPrepend()
      << "materialization failed with: ";

  // Check that all nodes to materialize have indeed been materialized.
  for (auto& node : nodes_to_materialize) {
    TT_RET_CHECK(node->state(0) == DeviceBufferRefState::kMaterialized,
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
  std::vector<SharedDeviceBufferList> nodes_to_materialize;
  // Optimistically assume all deferred and unique (most common case).
  nodes_to_materialize.reserve(nodes.size());
  absl::flat_hash_set<SharedDeviceBufferList> unique_nodes;
  for (const SharedDeviceBufferList& node : nodes) {
    if (node->deferred_op()) {
      if (unique_nodes.insert(node).second) {
        nodes_to_materialize.push_back(node);
      }
      continue;
    }
    // Node is not deferred; check to make sure it doesn't have any placeholder
    // buffers.
    for (int i = 0; i < node->size(); ++i) {
      TT_RET_CHECK(node->state(i) != DeviceBufferRefState::kPlaceholder,
                   error::kInternal)
          << "Materialize was called on a placeholder tensor. This should "
             "never happen.\nkPlaceholder tensors should only appear in "
             "compiled mode, which should never try to materialize tensors.";
    }
  }
  return MaterializeImpl(nodes_to_materialize, reason, materialization_mode);
}

absl::Status Materialize(absl::Span<const DeviceBufferRef> buffer_refs,
                         MaterializationReason reason,
                         MaterializationMode materialization_mode) {
  if (buffer_refs.empty()) {
    return absl::OkStatus();
  }
  absl::flat_hash_set<SharedDeviceBufferList> unique_deferred_nodes;
  std::vector<SharedDeviceBufferList> nodes_to_materialize;
  for (const DeviceBufferRef& buffer_ref : buffer_refs) {
    switch (buffer_ref.state()) {
      case DeviceBufferRefState::kMaterialized:
        // Already materialized, no-op for this node.
        continue;
      case DeviceBufferRefState::kDeferred: {
        if (unique_deferred_nodes.insert(buffer_ref.device_buffer_list())
                .second) {
          nodes_to_materialize.push_back(buffer_ref.device_buffer_list());
        }
        continue;
      }
      case DeviceBufferRefState::kPlaceholder:
        return TT_ERROR(error::kInternal)
               << "Materialize was called on a placeholder tensor. This should "
                  "never happen.\nkPlaceholder tensors should only appear in "
                  "compiled mode, which should never try to materialize "
                  "tensors.";
      default:
        return TT_ERROR(error::kInternal)
               << "DeviceBufferRef has unknown state";
    }
  }
  return MaterializeImpl(nodes_to_materialize, reason, materialization_mode);
}

absl::StatusOr<DeviceBufferRef> GetMaterialized(const at::Tensor& tensor,
                                                MaterializationReason reason) {
  tsl::profiler::TraceMe trace("GetMaterialized");
  // Make sure the base DeviceBufferRef is materialized
  const auto* tensor_impl = tensor.unsafeGetTensorImpl();
  TT_RET_CHECK(tensor_impl, error::kInvalidArgument) << "tensor is undefined";
  TT_ASSIGN_OR_RETURN(const DeviceBufferRef base_buffer_ref,
                      GetBaseBuffer(*tensor_impl));
  TT_RETURN_IF_ERROR(
      Materialize(base_buffer_ref, reason, MaterializationMode::kSplitGraph));

  // Get the view DeviceBufferRef (may be the same as the base)
  TT_ASSIGN_OR_RETURN(const DeviceBufferRef view_buffer_ref, GetBuffer(tensor));
  // Materialize the view (no-op if the tensor is a continuous base tensor)
  TT_RETURN_IF_ERROR(
      Materialize(view_buffer_ref, reason, MaterializationMode::kSplitGraph));
  // The new materialization algorithm needs us to block here until all pending
  // jobs are done.
  TT_RETURN_IF_ERROR(MaybeBlockOnPendingMaterializationsNew(tensor));

  return view_buffer_ref;
}

absl::StatusOr<std::vector<DeviceBufferRef>> GetMaterialized(
    absl::Span<const at::Tensor> tensors, MaterializationReason reason) {
  tsl::profiler::TraceMe trace("GetMaterialized (batch)");
  if (tensors.empty()) {
    return std::vector<DeviceBufferRef>();
  }
  // Materialize all of the base DeviceBufferRefs (in a single execution)
  std::vector<DeviceBufferRef> base_buffer_refs;
  base_buffer_refs.reserve(tensors.size());
  for (const at::Tensor& tensor : tensors) {
    const auto* tensor_impl = tensor.unsafeGetTensorImpl();
    TT_RET_CHECK(tensor_impl, error::kInvalidArgument) << "tensor is undefined";
    TT_ASSIGN_OR_RETURN(const DeviceBufferRef base_buffer_ref,
                        GetBaseBuffer(*tensor_impl));
    base_buffer_refs.push_back(base_buffer_ref);
  }
  TT_RETURN_IF_ERROR(
      Materialize(base_buffer_refs, reason, MaterializationMode::kSplitGraph));

  // Materialize all of the views (no-op if all tensors are contiguous bases)
  std::vector<DeviceBufferRef> view_buffer_refs;
  view_buffer_refs.reserve(tensors.size());
  for (const at::Tensor& tensor : tensors) {
    TT_ASSIGN_OR_RETURN(const DeviceBufferRef view_buffer_ref,
                        GetBuffer(tensor));
    view_buffer_refs.push_back(view_buffer_ref);
  }
  TT_RETURN_IF_ERROR(
      Materialize(view_buffer_refs, reason, MaterializationMode::kSplitGraph));
  // The new materialization algorithm needs us to block here until all pending
  // jobs are done.
  TT_RETURN_IF_ERROR(MaybeBlockOnPendingMaterializationsNew(tensors[0]));

  return view_buffer_refs;
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
    auto* absl_nonnull node = output.get();
    auto* materialized_buffers = node->materialized_buffers();
    if (materialized_buffers == nullptr) {
      continue;
    }
    materialized_buffers->SetAsError(status);
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
    const auto* deferred_op = node->deferred_op();
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
