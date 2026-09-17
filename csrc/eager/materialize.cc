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
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
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
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/context_manager.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_builder.h"
#include "csrc/eager/current_stream.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/materialization_heuristics.h"
#include "csrc/eager/materialize_common.h"
#include "csrc/eager/split_traversal.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/traversal.h"
#include "mlir/IR/MLIRContext.h"
#include "tsl/profiler/lib/traceme.h"

namespace torch_tpu {
namespace {

// A task to materialize a list of nodes.
// This will materialize all streams that have any of the given nodes, but only
// up to the last node in the list for each stream.
struct NodesMaterializationTask {};

// A task to materialize a stream.
// This will check the current state of deferred work on the stream, and start
// all necessary executions to catch the stream up to a snapshot.
struct StreamMaterializationTask {
  c10::DeviceIndex device_index;
  c10::StreamId stream_id;
};

// A task to materialize a device.
// This will check the current state of deferred work on each stream on the
// device, and start all necessary executions to catch each stream up to a
// snapshot.
struct DeviceMaterializationTask {
  c10::DeviceIndex device_index;
};

using MaterializationKind =
    std::variant<NodesMaterializationTask, StreamMaterializationTask,
                 DeviceMaterializationTask>;

// Common properties for all materialization tasks.
struct MaterializationTaskCommon {
  std::vector<SharedDeviceBufferList> must_materialize;
  MaterializationMode materialization_mode;
  MaterializationReason reason;
  CompilationMode compilation_mode;
  const CompilationContext* absl_nonnull compilation_context;

  MaterializationTaskCommon(
      std::vector<SharedDeviceBufferList> must_materialize,
      MaterializationReason reason, MaterializationMode materialization_mode)
      : must_materialize(std::move(must_materialize)),
        materialization_mode(materialization_mode),
        reason(reason),
        compilation_mode(GetCompilationMode(GetEagerMode())),
        compilation_context(&GetCompilationContextRef()) {}
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

// Returns the appropriate CompilationSpec for a split traversal.
// If the global compilation mode is already FastRuntime (O2), we retain it.
// Otherwise, if any deferred op in the traversal is an eligible convolution
// that benefits from O2, we selectively promote it to FastRuntime.
const CompilationSpec& GetCompilationSpecForTraversal(
    const MaterializationTaskCommon& common, const Traversal& traversal) {
  if (common.compilation_mode == CompilationMode::kFastRuntime) {
    return common.compilation_context->compilation_specs().at(
        CompilationMode::kFastRuntime);
  }

  const auto& execution_order = traversal.execution_order();
  const bool has_fast_runtime_conv =
      std::any_of(execution_order.begin(), execution_order.end(),
                  [](const SharedDeviceBufferList& node) {
                    const auto deferred_op = node->deferred_op();
                    return deferred_op != nullptr &&
                           IsFastRuntimeConvolutionCandidate(*deferred_op);
                  });

  if (has_fast_runtime_conv) {
    return common.compilation_context->compilation_specs().at(
        CompilationMode::kFastRuntime);
  }

  return common.compilation_context->compilation_specs().at(
      common.compilation_mode);
}

void SplitSingleTraversal(
    absl_nonnull std::unique_ptr<Traversal> pre_split_traversal,
    absl::flat_hash_set<const DeviceBufferList*>& required_outputs,
    std::vector<absl_nonnull std::unique_ptr<Traversal>>& post_split_traversals,
    absl::Status& first_error) {
  const auto core_pinning_mode = pre_split_traversal->core_pinning_mode();
  for (const auto& output : pre_split_traversal->outputs()) {
    required_outputs.insert(output.device_buffer_list().get());
  }
  auto split_traversals_or =
      SplitTraversal(std::move(pre_split_traversal), required_outputs);
  required_outputs.clear();
  if (!split_traversals_or.ok()) {
    if (first_error.ok()) {
      first_error = split_traversals_or.status();
    }
    return;
  }
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
}

bool DependsOnPlaceholder(
    const SharedDeviceBufferList& node,
    absl::flat_hash_map<const DeviceBufferList*, bool>& memo) {
  if (node == nullptr) return false;
  if (node->is_placeholder()) return true;
  auto it = memo.find(node.get());
  if (it != memo.end()) return it->second;

  const auto deferred_op = node->deferred_op();
  if (!deferred_op) {
    memo[node.get()] = false;
    return false;
  }

  memo[node.get()] = false;
  for (const auto& input : deferred_op->inputs()) {
    if (input.is_placeholder() ||
        DependsOnPlaceholder(input.device_buffer_list(), memo)) {
      memo[node.get()] = true;
      return true;
    }
  }
  return false;
}

bool OutputDependsOnPlaceholder(
    const DeviceBufferRef& ref,
    absl::flat_hash_map<const DeviceBufferList*, bool>& memo) {
  if (ref.is_placeholder()) return true;
  return DependsOnPlaceholder(ref.device_buffer_list(), memo);
}

void PruneTraversalPlaceholderOutputs(
    absl_nonnull std::unique_ptr<Traversal> traversal,
    absl::flat_hash_map<const DeviceBufferList*, bool>& memo,
    std::vector<absl_nonnull std::unique_ptr<Traversal>>& clean_traversals,
    absl::Status& first_error) {
  std::vector<DeviceBufferRef> clean_outputs;
  for (const auto& output : traversal->outputs()) {
    if (!OutputDependsOnPlaceholder(output, memo)) {
      clean_outputs.push_back(output);
    }
  }

  if (clean_outputs.empty()) {
    ABSL_VLOG(1) << "Skipping traversal containing only placeholder outputs";
    return;
  }

  if (clean_outputs.size() == traversal->outputs().size()) {
    clean_traversals.push_back(std::move(traversal));
    return;
  }

  ABSL_VLOG(1) << "Pruning placeholder outputs from traversal: kept "
               << clean_outputs.size() << " of " << traversal->outputs().size()
               << " outputs";
  absl::flat_hash_set<const DeviceBufferList* absl_nonnull> stopping_points;
  stopping_points.reserve(traversal->arguments().size());
  for (const DeviceBufferRef& arg : traversal->arguments()) {
    stopping_points.insert(arg.device_buffer_list().get());
  }
  const auto core_pinning_mode = traversal->core_pinning_mode();
  auto clean_traversal_or =
      Traversal::Create(std::move(clean_outputs), stopping_points);
  if (clean_traversal_or.ok()) {
    if (core_pinning_mode != CorePinningMode::kUnpinned) {
      (*clean_traversal_or)->SetCorePinningMode(core_pinning_mode);
    }
    clean_traversals.push_back(std::move(*clean_traversal_or));
  } else if (first_error.ok()) {
    first_error = clean_traversal_or.status();
  }
}

bool HasAnyPlaceholderArgumentOrOutput(const Traversal& traversal) {
  for (const DeviceBufferRef& arg : traversal.arguments()) {
    if (ABSL_PREDICT_FALSE(arg.device_buffer_list()->is_placeholder())) {
      return true;
    }
  }
  for (const DeviceBufferRef& out : traversal.outputs()) {
    if (ABSL_PREDICT_FALSE(out.device_buffer_list()->is_placeholder())) {
      return true;
    }
  }
  return false;
}

// Eager ops traced over placeholder tensors (e.g. aten::clone during
// direct_compile) leave abandoned DeferredOps on the events queue. Drop any
// output that transitively depends on a placeholder so the real ops sharing
// this traversal can still materialize.
//
// Performance Optimization:
//   Per-Traversal Fast Path (`HasAnyPlaceholderArgumentOrOutput`):
//   Because placeholder buffers are leaf nodes (they have no DeferredOp), any
//   placeholder in a traversal DAG must appear in `traversal.arguments()` or
//   `traversal.outputs()`. Most traversals do not involve placeholders; we only
//   construct the recursive DAG memoization map (`placeholder_memo`) and prune
//   outputs if at least one traversal directly references a placeholder
//   argument or output.
std::vector<absl_nonnull std::unique_ptr<Traversal>>
FilterPlaceholderDependentTraversals(
    std::vector<absl_nonnull std::unique_ptr<Traversal>>&& traversals,
    absl::Status& first_error) {
  const bool any_has_placeholder =
      std::any_of(traversals.begin(), traversals.end(),
                  [](const absl_nonnull std::unique_ptr<Traversal>& traversal) {
                    return HasAnyPlaceholderArgumentOrOutput(*traversal);
                  });
  if (!any_has_placeholder) {
    return std::move(traversals);
  }

  absl::flat_hash_map<const DeviceBufferList*, bool> placeholder_memo;
  std::vector<absl_nonnull std::unique_ptr<Traversal>> clean_traversals;
  clean_traversals.reserve(traversals.size());

  for (auto& traversal : traversals) {
    if (!HasAnyPlaceholderArgumentOrOutput(*traversal)) {
      clean_traversals.push_back(std::move(traversal));
      continue;
    }
    PruneTraversalPlaceholderOutputs(std::move(traversal), placeholder_memo,
                                     clean_traversals, first_error);
  }
  return clean_traversals;
}

// Converts a sequence of Traversals to a sequence of ExecutionTasks.
// If materialization_mode is kSplitGraph, then there may be more returned
// tasks than there were original traversals; otherwise, there will be exactly
// one ExecutionTask per traversal (provided all task creations succeed).
// Appends created tasks to `execution_tasks` so they can be enqueued in a
// single mutex acquisition, avoiding per-subtraversal futex wakeup/sleep
// thrashing across NUMA nodes.
absl::Status ApplySplitMode(
    std::vector<absl_nonnull std::unique_ptr<Traversal>>&& traversals,
    const MaterializationTaskCommon& common, mlir::MLIRContext& mlir_context,
    std::vector<ExecutionTask>& execution_tasks) {
  absl::Status first_error = absl::OkStatus();
  traversals =
      FilterPlaceholderDependentTraversals(std::move(traversals), first_error);

  if (common.materialization_mode == MaterializationMode::kSplitGraph) {
    // Fast path: 0-op or 1-op traversals cannot contain internal split points.
    const bool needs_split_check = std::any_of(
        traversals.begin(), traversals.end(),
        [](const absl_nonnull std::unique_ptr<Traversal>& traversal) {
          return traversal->execution_order().size() > 1;
        });
    if (needs_split_check) {
      tsl::profiler::TraceMe t("SplitTraversal");
      std::vector<absl_nonnull std::unique_ptr<Traversal>>
          post_split_traversals;
      post_split_traversals.reserve(traversals.size());
      absl::flat_hash_set<const DeviceBufferList*> required_outputs;
      for (auto& pre_split_traversal : traversals) {
        if (pre_split_traversal->execution_order().size() <= 1) {
          post_split_traversals.push_back(std::move(pre_split_traversal));
          continue;
        }
        SplitSingleTraversal(std::move(pre_split_traversal), required_outputs,
                             post_split_traversals, first_error);
      }
      std::swap(traversals, post_split_traversals);
    }
  }

  execution_tasks.reserve(execution_tasks.size() + traversals.size());
  for (auto& split_traversal : traversals) {
    const CompilationSpec& compilation_spec =
        GetCompilationSpecForTraversal(common, *split_traversal);
    auto execution_task_or = ExecutionTask::FromTraversalWithLogging(
        std::move(split_traversal), mlir_context, compilation_spec,
        common.reason);
    if (execution_task_or.ok()) {
      execution_tasks.push_back(std::move(*execution_task_or));
    } else if (first_error.ok()) {
      first_error = execution_task_or.status();
    }
  }
  return first_error;
}

// Processes a MaterializationTask, converting it into a sequence of
// ExecutionTasks appended to `execution_tasks`.
absl::Status ProcessMaterializationTask(
    MaterializationTask& task, mlir::MLIRContext& mlir_context,
    std::vector<ExecutionTask>& execution_tasks) {
  std::vector<absl_nonnull std::unique_ptr<Traversal>> traversals;
  if (std::holds_alternative<NodesMaterializationTask>(task.kind)) {
    ABSL_VLOG(1)
        << "[MaterializationWorker] Processing NodesMaterializationTask with "
        << task.common.must_materialize.size() << " nodes";
    LogDeferredNodes(task.common.must_materialize,
                     /* msg_prefix= */ "  Input node");

    bool has_non_deferred = false;
    for (const auto& node : task.common.must_materialize) {
      if (!node->is_deferred()) {
        has_non_deferred = true;
        break;
      }
    }
    absl::Span<const SharedDeviceBufferList> deferred_nodes =
        task.common.must_materialize;
    std::vector<SharedDeviceBufferList> filtered_nodes;
    if (ABSL_PREDICT_FALSE(has_non_deferred)) {
      filtered_nodes.reserve(task.common.must_materialize.size());
      for (const auto& node : task.common.must_materialize) {
        if (node->is_deferred()) {
          filtered_nodes.push_back(node);
        }
      }
      if (filtered_nodes.empty()) {
        // Everything was already materialized, nothing more to do.
        return absl::OkStatus();
      }
      deferred_nodes = filtered_nodes;
    }
    auto traversals_or = PrepareMaterializationTraversals(deferred_nodes);
    if (!traversals_or.ok()) {
      return traversals_or.status();
    }
    traversals = std::move(*traversals_or);
  } else if (const auto* stream_task =
                 std::get_if<StreamMaterializationTask>(&task.kind)) {
    ABSL_VLOG(1)
        << "[MaterializationWorker] Processing MaterializationTask with "
        << "stream " << stream_task->stream_id << " on device "
        << stream_task->device_index;
    auto traversals_or = PrepareStreamTraversals(stream_task->device_index,
                                                 stream_task->stream_id,
                                                 task.common.must_materialize);
    if (!traversals_or.ok()) {
      return traversals_or.status();
    }
    traversals = std::move(*traversals_or);
  } else if (const auto* device_task =
                 std::get_if<DeviceMaterializationTask>(&task.kind)) {
    ABSL_VLOG(1)
        << "[MaterializationWorker] Processing MaterializationTask with "
        << "device " << device_task->device_index;
    auto traversals_or = PrepareDeviceTraversals(device_task->device_index,
                                                 task.common.must_materialize);
    if (!traversals_or.ok()) {
      return traversals_or.status();
    }
    traversals = std::move(*traversals_or);
  } else {
    return TT_ERROR(error::kInternal) << "Unknown MaterializationTask kind";
  }

  return ApplySplitMode(std::move(traversals), task.common, mlir_context,
                        execution_tasks);
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

using ExecutionOrShutdown = std::variant<ShutdownSentinel, ExecutionTask>;

class MaterializationWorker {
 public:
  MaterializationWorker(const MaterializationWorker&) = delete;
  MaterializationWorker& operator=(const MaterializationWorker&) = delete;
  MaterializationWorker(MaterializationWorker&&) = delete;
  MaterializationWorker& operator=(MaterializationWorker&&) = delete;

  MaterializationWorker() : mlir_context_(MakeMlirContext()) { StartThreads(); }

  ~MaterializationWorker() { Shutdown(); }

  // Shuts down the worker thread. Waiting on materialize_mu_ ensures any
  // in-flight materialization finishes enqueuing execution tasks before
  // execute_thread_ shuts down.
  void Shutdown() {
    bool expected_shutdown = false;
    // The shutdown_ flag prevents duplicate shutdown runs.
    if (shutdown_.compare_exchange_strong(expected_shutdown, true)) {
      {
        absl::MutexLock lock(materialize_mu_);
      }
      {
        absl::MutexLock lock(execute_mu_);
        execute_tasks_.push(ShutdownSentinel{});
        // Release the lock so that the thread can dequeue until the sentinel.
      }
      if (execute_thread_.joinable()) {
        execute_thread_.join();
      }
      FailRemainingExecutionTasks();
    }
  }

  absl::Status MaterializeNodes(std::vector<SharedDeviceBufferList> nodes,
                                MaterializationReason reason,
                                MaterializationMode materialization_mode) {
    ABSL_VLOG(1) << "[MaterializationWorker] Materializing " << nodes.size()
                 << " nodes";
    MaterializationTaskCommon common(std::move(nodes), reason,
                                     materialization_mode);

    absl::MutexLock lock(materialize_mu_);
    // Check shutdown state after acquiring the lock to avoid a time-of-check/
    // time-of-use error relative to Shutdown().
    if (shutdown_.load()) {
      // State is shutting down, fail the task immediately.
      return TT_ERROR(error::kCancelled)  // ERROR_COV_INFEASIBLE=unreachable
                                          // from Python
             << "MaterializationWorker is shutting down";
    }

    MaterializationTask task{
        .kind = NodesMaterializationTask{},
        .common = std::move(common),
    };

    // Temporarily disallow thread-local context state access while running
    // materialization logic inline on the dispatch thread, enforcing that
    // materialization always uses resolved configurations from
    // MaterializationTaskCommon.
    ScopedDisallowContextStateAccess disallow_context_state_access;
    std::vector<ExecutionTask> execution_tasks;
    absl::Status first_error =
        ProcessMaterializationTask(task, *mlir_context_, execution_tasks);
    if (!execution_tasks.empty()) {
      // Enqueue as many execution tasks as were successfully created in a
      // single lock acquisition.
      absl::Status enqueue_status =
          EnqueueExecutionTasks(std::move(execution_tasks));
      if (!enqueue_status.ok() && first_error.ok()) {
        first_error = enqueue_status;
      }
    }

    return FinishMaterializationTask(task, first_error);
  }

  absl::StatusOr<std::shared_ptr<EventSnapshot>> MaterializeStream(
      const c10::DeviceIndex device_index, const c10::StreamId stream_id,
      MaterializationReason reason, MaterializationMode materialization_mode,
      std::vector<SharedDeviceBufferList> must_materialize) {
    ABSL_VLOG(1) << "[MaterializationWorker] Materializing stream " << stream_id
                 << " on device " << device_index;
    MaterializationTaskCommon common(std::move(must_materialize), reason,
                                     materialization_mode);

    absl::MutexLock lock(materialize_mu_);
    // Check shutdown state after acquiring the lock to avoid a time-of-check/
    // time-of-use error relative to Shutdown().
    if (shutdown_.load()) {
      return TT_ERROR(error::kCancelled)  // ERROR_COV_INFEASIBLE=unreachable
                                          // from Python
             << "MaterializationWorker is shutting down";
    }

    MaterializationTask task{
        .kind =
            StreamMaterializationTask{
                .device_index = device_index,
                .stream_id = stream_id,
            },
        .common = std::move(common),
    };

    ScopedDisallowContextStateAccess disallow_context_state_access;
    std::vector<ExecutionTask> execution_tasks;
    absl::Status first_error =
        ProcessMaterializationTask(task, *mlir_context_, execution_tasks);
    if (!execution_tasks.empty()) {
      absl::Status enqueue_status =
          EnqueueExecutionTasks(std::move(execution_tasks));
      if (!enqueue_status.ok() && first_error.ok()) {
        first_error = enqueue_status;
      }
    }

    TT_RETURN_IF_ERROR(FinishMaterializationTask(task, first_error));
    return EventSnapshot::Record(device_index, stream_id);
  }

  absl::StatusOr<std::vector<std::shared_ptr<EventSnapshot>>> MaterializeDevice(
      const c10::DeviceIndex device_index, MaterializationReason reason,
      MaterializationMode materialization_mode,
      std::vector<SharedDeviceBufferList> must_materialize) {
    ABSL_VLOG(1) << "[MaterializationWorker] Materializing device "
                 << device_index;
    MaterializationTaskCommon common(std::move(must_materialize), reason,
                                     materialization_mode);

    absl::MutexLock lock(materialize_mu_);
    // Check shutdown state after acquiring the lock to avoid a time-of-check/
    // time-of-use error relative to Shutdown().
    if (shutdown_.load()) {
      return TT_ERROR(error::kCancelled)  // ERROR_COV_INFEASIBLE=unreachable
                                          // from Python
             << "MaterializationWorker is shutting down";
    }

    MaterializationTask task{
        .kind =
            DeviceMaterializationTask{
                .device_index = device_index,
            },
        .common = std::move(common),
    };

    ScopedDisallowContextStateAccess disallow_context_state_access;
    std::vector<ExecutionTask> execution_tasks;
    absl::Status first_error =
        ProcessMaterializationTask(task, *mlir_context_, execution_tasks);
    if (!execution_tasks.empty()) {
      absl::Status enqueue_status =
          EnqueueExecutionTasks(std::move(execution_tasks));
      if (!enqueue_status.ok() && first_error.ok()) {
        first_error = enqueue_status;
      }
    }

    TT_RETURN_IF_ERROR(FinishMaterializationTask(task, first_error));
    return RecordDeviceSnapshots(device_index);
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

    TT_RETURN_IF_ERROR(MarkPinnedHostOutputs(outputs, executable));
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

  // Dequeues all currently available execution tasks or a shutdown signal in
  // a single lock acquisition using an O(1) std::queue swap. Draining the
  // entire queue in one critical section eliminates cross-NUMA mutex/futex
  // ping-ponging between the dispatch thread and execution thread.
  std::queue<ExecutionOrShutdown> DequeueExecutionTasks() {
    absl::MutexLock lock(execute_mu_);
    execute_mu_.Await(absl::Condition(
        +[](std::queue<ExecutionOrShutdown>* tasks) { return !tasks->empty(); },
        &execute_tasks_));
    std::queue<ExecutionOrShutdown> popped_tasks;
    std::swap(popped_tasks, execute_tasks_);
    return popped_tasks;
  }

  void StartThreads() {
    execute_thread_ = std::thread([this]() { ExecuteLoop(); });
  }

 private:
  // Verifies that all must_materialize nodes have transitioned to materializing
  // or sets error status on any unmaterialized nodes upon failure.
  static absl::Status FinishMaterializationTask(MaterializationTask& task,
                                                absl::Status first_error) {
    if (!first_error.ok()) {
      for (const auto& node : task.common.must_materialize) {
        if (!node->is_materializing()) {
          node->SetAsError(first_error);
        }
      }
      return first_error;
    }
    absl::Status first_missing_node_error = absl::OkStatus();
    for (const auto& node : task.common.must_materialize) {
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

  absl::Status EnqueueExecutionTasks(
      std::vector<ExecutionTask>&& execution_tasks) {
    if (execution_tasks.empty()) {
      return absl::OkStatus();
    }
    ABSL_VLOG(1) << "[MaterializationWorker] Enqueuing "
                 << execution_tasks.size() << " ExecutionTasks";
    absl::MutexLock lock(execute_mu_);
    if (shutdown_.load()) {
      absl::Status shutdown_status =
          TT_ERROR(error::kCancelled)  // ERROR_COV_INFEASIBLE=unreachable
                                       // from Python
          << "MaterializationWorker is shutting down";
      for (auto& execution_task : execution_tasks) {
        execution_task.SetOutputNodesAsError(shutdown_status);
      }
      return shutdown_status;
    }
    for (auto& execution_task : execution_tasks) {
      execute_tasks_.push(std::move(execution_task));
    }
    return absl::OkStatus();
  }

  void ExecuteLoop() {
    // Prevent execution worker threads from accessing thread-local context
    // states to enforce they always rely on resolved configurations passed
    // down from the dispatch thread.
    DisallowThisThreadToAccessContextState();

    while (true) {
      std::queue<ExecutionOrShutdown> tasks = DequeueExecutionTasks();
      bool should_shutdown = false;
      // Pop and process each task one by one so that `task_or_shutdown` goes
      // out of scope and is destroyed immediately after `task.Run()` finishes.
      // This ensures strong DeviceBufferRef handles to intermediate tensors are
      // dropped immediately rather than held alive across the entire batch.
      while (!tasks.empty()) {
        ExecutionOrShutdown task_or_shutdown = std::move(tasks.front());
        tasks.pop();
        if (std::holds_alternative<ShutdownSentinel>(task_or_shutdown)) {
          should_shutdown = true;
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
      if (should_shutdown) {
        break;
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

  std::thread execute_thread_;

  // Set to true when the MaterializationWorker is undergoing shutdown,
  // preventing concurrent or duplicate shutdown sequences from executing.
  std::atomic<bool> shutdown_{false};

  absl::Mutex materialize_mu_;
  absl_nonnull std::unique_ptr<mlir::MLIRContext> mlir_context_
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
// Precondition: `nodes_to_materialize` is non-empty, contains no placeholders,
// and contains only unique deferred nodes.
absl::Status MaterializeImpl(
    std::vector<SharedDeviceBufferList>&& nodes_to_materialize,
    MaterializationReason reason, MaterializationMode materialization_mode) {
  tsl::profiler::TraceMe t("MaterializeImpl");
  ABSL_VLOG(1) << "[MaterializeImpl] Materializing "
               << nodes_to_materialize.size() << " nodes";

  TT_RETURN_IF_ERROR(
      GetMaterializationWorker().MaterializeNodes(
          std::move(nodes_to_materialize), reason, materialization_mode))
          .SetPrepend()
      << "materialization failed with: ";

  return absl::OkStatus();
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
// unique.
// Returns an error if any item is a placeholder.
template <typename T>
absl::StatusOr<std::vector<SharedDeviceBufferList>> GetUniqueDeferredNodes(
    absl::Span<const T> items) {
  std::vector<SharedDeviceBufferList> nodes_to_materialize;
  if (items.size() <= 16) {
    absl::InlinedVector<const DeviceBufferList*, 16> seen;
    for (const T& item : items) {
      const SharedDeviceBufferList& node = GetDeviceBufferList(item);
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

// Common implementation for Materialize(Span<SharedDeviceBufferList>) and
// Materialize(Span<DeviceBufferRef>).
//
// Performance Invariants & Rationale:
//   1. Zero-Allocation Fast Path & Single-Element Specialization:
//      `Materialize` is frequently invoked on step boundaries or explicit sync
//      points (e.g., `torch_tpu.sync.synchronize(tensors, wait=False)`) where
//      some or all tensors may already be materializing or materialized (such
//      as in DeferNever / Eager Default mode). For `items.size() == 1` (which
//      covers ~99% of single-tensor synchronize calls), we check
//      `is_deferred()` directly without any loop or deduplication container.
//      For multi-element spans, we perform a single pass and return
//      `absl::OkStatus()` immediately with zero heap allocations if no deferred
//      nodes are found.
//   2. Lazy Single-Allocation Reservation (`reserve(items.size())`):
//      When a deferred node is encountered, reserving `items.size()` lazily on
//      the first deferred item guarantees at most one heap allocation for
//      `nodes_to_materialize` while preserving zero allocations when all nodes
//      are already materialized.
//   3. Zero-Heap-Allocation Caller-Side Deduplication:
//      Multiple tensors in `items` frequently share the same underlying
//      `DeviceBufferList` (e.g., multi-output ops or tensor views/slices passed
//      to `synchronize`). Deduplicating `DeviceBufferList*` pointers on the
//      caller thread avoids redundant atomic `std::shared_ptr` reference-count
//      increments/decrements and avoids enqueuing duplicate pointers. For
//      typical small spans (`num_deferred <= 16`), we use a stack-allocated
//      `absl::InlinedVector<const DeviceBufferList*, 16>` (zero heap
//      allocation, L1-cache-resident linear scan); for larger spans, we use
//      `absl::flat_hash_set`.
//   4. Zero-Copy Move into Worker Queue:
//      `nodes_to_materialize` is moved (`std::move`) directly into
//      `MaterializeImpl` -> `EnqueueNodes`, avoiding a second vector heap
//      allocation and `2 * N` atomic `shared_ptr` refcount operations.
template <typename T>
absl::Status MaterializeSpan(absl::Span<const T> items,
                             MaterializationReason reason,
                             MaterializationMode materialization_mode) {
  if (items.empty()) {
    return absl::OkStatus();
  }
  if (items.size() == 1) {
    const SharedDeviceBufferList& node = GetDeviceBufferList(items[0]);
    if (ABSL_PREDICT_FALSE(node->is_placeholder())) {
      return TT_ERROR(error::kInternal)
             << "cannot Materialize() a placeholder tensor or a tensor that "
                "depends on a placeholder. \nPlaceholder tensors should only "
                "appear in compiled mode, which should never try to "
                "materialize tensors";
    }
    if (!node->is_deferred()) {
      return absl::OkStatus();
    }
    std::vector<SharedDeviceBufferList> nodes_to_materialize;
    nodes_to_materialize.reserve(1);
    nodes_to_materialize.push_back(node);
    return MaterializeImpl(std::move(nodes_to_materialize), reason,
                           materialization_mode);
  }
  TT_ASSIGN_OR_RETURN(std::vector<SharedDeviceBufferList> nodes_to_materialize,
                      GetUniqueDeferredNodes(items));
  if (nodes_to_materialize.empty()) {
    return absl::OkStatus();
  }
  return MaterializeImpl(std::move(nodes_to_materialize), reason,
                         materialization_mode);
}

}  // namespace

void ShutDownMaterializationState() { GetMaterializationWorker().Shutdown(); }

absl::Status Materialize(absl::Span<const SharedDeviceBufferList> nodes,
                         MaterializationReason reason,
                         MaterializationMode materialization_mode) {
  return MaterializeSpan(nodes, reason, materialization_mode);
}

absl::Status Materialize(absl::Span<const DeviceBufferRef> buffer_refs,
                         MaterializationReason reason,
                         MaterializationMode materialization_mode) {
  return MaterializeSpan(buffer_refs, reason, materialization_mode);
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
    MaterializationReason reason, MaterializationMode mode,
    absl::Span<const SharedDeviceBufferList> must_materialize) {
#ifndef NDEBUG
  for (const auto& node : must_materialize) {
    ABSL_CHECK_EQ(node->device_index(),  // CRASH_OK=not visible to Python
                  device_index);
    ABSL_CHECK_EQ(node->stream_id(),  // CRASH_OK=not visible to Python
                  stream_id);
  }
#endif
  TT_ASSIGN_OR_RETURN(std::vector<SharedDeviceBufferList> must_materialize_vec,
                      GetUniqueDeferredNodes(must_materialize));
  // Synchronously materialize the stream to ensure that more work is not pushed
  // to the stream while we are evaluating its current state.
  return GetMaterializationWorker().MaterializeStream(
      device_index, stream_id, reason, mode, std::move(must_materialize_vec));
}

absl::StatusOr<std::vector<std::shared_ptr<EventSnapshot>>> MaterializeDevice(
    c10::DeviceIndex device_index, MaterializationReason reason,
    MaterializationMode mode,
    absl::Span<const SharedDeviceBufferList> must_materialize) {
#ifndef NDEBUG
  for (const auto& node : must_materialize) {
    ABSL_CHECK_EQ(node->device_index(),  // CRASH_OK=not visible to Python
                  device_index);
  }
#endif
  TT_ASSIGN_OR_RETURN(std::vector<SharedDeviceBufferList> must_materialize_vec,
                      GetUniqueDeferredNodes(must_materialize));
  // Synchronously materialize the device to ensure that more work is not pushed
  // to the device while we are evaluating the state of its current streams.
  return GetMaterializationWorker().MaterializeDevice(
      device_index, reason, mode, std::move(must_materialize_vec));
}

absl::StatusOr<std::shared_ptr<EventSnapshot>> MaterializeCurrentStream(
    MaterializationReason reason, MaterializationMode mode,
    absl::Span<const SharedDeviceBufferList> must_materialize) {
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  return MaterializeStream(device_index, stream_id, reason, mode,
                           must_materialize);
}

}  // namespace torch_tpu
