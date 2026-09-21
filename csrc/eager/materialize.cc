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
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "c10/core/Stream.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/context_manager.h"
#include "csrc/common/env_vars.h"
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
#include "csrc/ops/op_names.h"
#include "mlir/IR/MLIRContext.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/future.h"

namespace torch_tpu {
namespace {

// A task to materialize a stream.
// This will check the stream state and enqueue all necessary executions to
// catch the stream up to the most recent snapshot (created by
// EventSnapshot::Record or RecordDeviceSnapshots)
struct MaterializationTask {
  c10::DeviceIndex device_index;
  c10::StreamId stream_id;
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

// When preparing a list of Traversals into a sequence of ExecutionTasks,
// some may fail. This struct holds the results, including the first error
// encountered, if any.
struct MaterializationStages {
  std::vector<ExecutionTask> execution_tasks;
  absl::Status first_error = absl::OkStatus();
};

// Returns the appropriate CompilationSpec for a split traversal.
// If the global compilation mode is already FastRuntime (O2), we retain it.
// Otherwise, if any deferred op in the traversal is an eligible convolution
// that benefits from O2, we selectively promote it to FastRuntime.
const CompilationSpec& GetCompilationSpecForTraversal(
    const MaterializationSettings& materialization_settings,
    const Traversal& traversal) {
  const auto& compilation_specs =
      materialization_settings.compilation_context->compilation_specs();
  if (materialization_settings.compilation_mode !=
      CompilationMode::kFastCompile) {
    // Either we're already using O2, or the O2 spec is not applicable.
    return compilation_specs.at(materialization_settings.compilation_mode);
  }

  const auto& execution_order = traversal.execution_order();

  if (std::any_of(execution_order.begin(), execution_order.end(),
                  [](const SharedDeviceBufferList& node) {
                    const auto deferred_op = node->deferred_op();
                    return deferred_op != nullptr &&
                           IsFastRuntimeConvolutionCandidate(*deferred_op);
                  })) {
    return compilation_specs.at(CompilationMode::kFastRuntime);
  }

  return compilation_specs.at(materialization_settings.compilation_mode);
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
MaterializationStages ApplySplitMode(
    std::vector<absl_nonnull std::unique_ptr<Traversal>>&& traversals,
    const MaterializationSettings& settings, mlir::MLIRContext& mlir_context) {
  MaterializationStages result;
  traversals = FilterPlaceholderDependentTraversals(std::move(traversals),
                                                    result.first_error);

  if (settings.materialization_mode == MaterializationMode::kSplitGraph) {
    tsl::profiler::TraceMe t("SplitTraversal");
    std::vector<absl_nonnull std::unique_ptr<Traversal>> post_split_traversals;
    absl::flat_hash_set<const DeviceBufferList*> required_outputs;
    for (auto& pre_split_traversal : traversals) {
      SplitSingleTraversal(std::move(pre_split_traversal), required_outputs,
                           post_split_traversals, result.first_error);
    }
    std::swap(traversals, post_split_traversals);
  }

  if (torch_tpu::GetBooleanEnvOnce<kTorchTpuInternalSkinnyScOffloadEnvVar>()
          .value_or(false)) {
    for (auto& traversal : traversals) {
      if (traversal->core_pinning_mode() == CorePinningMode::kUnpinned &&
          traversal->execution_order().size() == 1) {
        auto deferred_op = traversal->execution_order()[0]->deferred_op();
        if (deferred_op != nullptr && IsDistributedOp(deferred_op->op_name())) {
          traversal->SetCorePinningMode(CorePinningMode::kSparseCore);
        }
      }
    }
  }

  result.execution_tasks.reserve(traversals.size());
  for (auto& split_traversal : traversals) {
    const CompilationSpec& compilation_spec =
        GetCompilationSpecForTraversal(settings, *split_traversal);
    auto execution_task_or = ExecutionTask::FromTraversalWithLogging(
        std::move(split_traversal), mlir_context, compilation_spec,
        settings.reason);
    if (execution_task_or.ok()) {
      result.execution_tasks.push_back(std::move(*execution_task_or));
    } else if (result.first_error.ok()) {
      result.first_error = execution_task_or.status();
    }
  }
  return result;
}

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

// Processes a MaterializationTask, converting it into a sequence of
// ExecutionTasks. Uses the provided vector to hold the results, which is
// cleared before adding the new tasks.
void ProcessMaterializationTask(MaterializationTask& task,
                                mlir::MLIRContext& mlir_context,
                                std::vector<ExecutionTask>& execution_tasks) {
  execution_tasks.clear();
  ABSL_VLOG(1) << "[MaterializationWorker] Processing MaterializationTask with "
               << "stream " << task.stream_id << " on device "
               << task.device_index;
  auto snapshots_to_materialize =
      PrepareSnapshotMaterializations(task.device_index, task.stream_id);
  for (auto& snapshot_to_materialize : snapshots_to_materialize) {
    MaterializationStages stages =
        ApplySplitMode(std::move(snapshot_to_materialize.traversals),
                       snapshot_to_materialize.settings, mlir_context);

    if (snapshot_to_materialize.execution_task.has_value()) {
      stages.execution_tasks.push_back(
          std::move(*snapshot_to_materialize.execution_task));
    }

    if (stages.first_error.ok()) {
      stages.first_error =
          VerifyMustMaterialize(snapshot_to_materialize.must_materialize);
    }

    if (!stages.first_error.ok()) {
      snapshot_to_materialize.snapshot_promise.Set(stages.first_error);
    } else {
      for (const auto& execution_task : stages.execution_tasks) {
        for (const auto& output : execution_task.outputs()) {
          snapshot_to_materialize.futures.push_back(output.GetReadyFuture());
        }
      }
      snapshot_to_materialize.snapshot_promise.Set(
          std::make_shared<EventSnapshot>(
              xla::JoinFutures(snapshot_to_materialize.futures)));
    }

    execution_tasks.insert(
        execution_tasks.end(),
        std::make_move_iterator(stages.execution_tasks.begin()),
        std::make_move_iterator(stages.execution_tasks.end()));
  }

  ABSL_VLOG(2) << "[MaterializationWorker] Created " << execution_tasks.size()
               << " ExecutionTasks for stream " << task.stream_id
               << " on device " << task.device_index << " to satisfy "
               << snapshots_to_materialize.size() << " snapshots";
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
        materialize_tasks_.push_back(ShutdownSentinel{});
      }
      // Release the lock so that the thread can dequeue until the sentinel.
      if (materialize_thread_.joinable()) {
        materialize_thread_.join();
      }
      {
        absl::MutexLock lock(execute_mu_);
        execute_tasks_.push_back(ShutdownSentinel{});
      }
      if (execute_thread_.joinable()) {
        execute_thread_.join();
      }
      FailRemainingExecutionTasks();
    }
  }

  void EnqueueStream(const c10::DeviceIndex device_index,
                     const c10::StreamId stream_id) {
    ABSL_VLOG(1) << "[MaterializationWorker] Enqueuing stream " << stream_id
                 << " on device " << device_index << " for materialization";
    absl::MutexLock lock(materialize_mu_);
    // Check shutdown state after acquiring the lock to avoid a time-of-check/
    // time-of-use error relative to Shutdown().
    if (shutdown_.load()) return;
    materialize_tasks_.push_back(MaterializationTask(device_index, stream_id));
  }

  // Dequeues all available materialization tasks (and possibly a shutdown
  // sentinel), and returns them in the provided vector. This uses a swap to
  // avoid repeated allocations.
  // This will block until at least one task is available.
  void DequeueMaterializationTasks(
      std::vector<MaterializationOrShutdown>& tasks_out) {
    absl::MutexLock lock(materialize_mu_);
    if (!materialize_tasks_.empty()) {
      return std::swap(materialize_tasks_, tasks_out);
    }
    materialize_mu_.Await(absl::Condition(
        +[](std::vector<MaterializationOrShutdown>* tasks) {
          return !tasks->empty();
        },
        &materialize_tasks_));
    return std::swap(materialize_tasks_, tasks_out);
  }

  // Dequeues all available execution tasks(and possibly a shutdown
  // sentinel), and returns them in the provided vector. This uses a swap to
  // avoid repeated allocations.
  // This will block until at least one task is available.
  void DequeueExecutionTasks(std::vector<ExecutionOrShutdown>& tasks_out) {
    absl::MutexLock lock(execute_mu_);
    if (!execute_tasks_.empty()) {
      return std::swap(execute_tasks_, tasks_out);
    }
    execute_mu_.Await(absl::Condition(
        +[](std::vector<ExecutionOrShutdown>* tasks) {
          return !tasks->empty();
        },
        &execute_tasks_));
    return std::swap(execute_tasks_, tasks_out);
  }

  void StartThreads() {
    materialize_thread_ = std::thread([this]() { MaterializeLoop(); });
    execute_thread_ = std::thread([this]() { ExecuteLoop(); });
  }

 private:
  // Processes a single materialization task.
  // Returns true if the loop should continue, or false if the worker should
  // shutdown.
  bool MaterializeLoopStep(
      mlir::MLIRContext& mlir_context,
      std::vector<MaterializationOrShutdown>& task_or_shutdown_list,
      std::vector<ExecutionTask>& execution_tasks) {
    DequeueMaterializationTasks(task_or_shutdown_list);

    for (auto& task_or_shutdown : task_or_shutdown_list) {
      if (std::holds_alternative<ShutdownSentinel>(task_or_shutdown)) {
        task_or_shutdown_list.clear();
        return false;
      }
      auto& task = std::get<MaterializationTask>(task_or_shutdown);
      ABSL_VLOG(1) << "[MaterializationWorker] Processing MaterializationTask";
      ProcessMaterializationTask(task, mlir_context, execution_tasks);

      if (!execution_tasks.empty()) {
        // Enqueue as many execution tasks as were successfully created.
        ABSL_VLOG(1) << "[MaterializationWorker] Enqueuing "
                     << execution_tasks.size() << " ExecutionTasks";
        absl::MutexLock lock(execute_mu_);
        for (auto& execution_task : execution_tasks) {
          execute_tasks_.push_back(std::move(execution_task));
        }
        execution_tasks.clear();
      }
    }
    task_or_shutdown_list.clear();

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
    std::vector<ExecutionTask> execution_tasks;
    std::vector<MaterializationOrShutdown> task_or_shutdown_list;
    while (MaterializeLoopStep(*mlir_context, task_or_shutdown_list,
                               execution_tasks)) {
      // MaterializeLoopStep will return false when it dequeues a
      // ShutdownSentinel.
    }
  }

  void ExecuteLoop() {
    // Prevent execution worker threads from accessing thread-local context
    // states to enforce they always rely on resolved configurations passed
    // down from the dispatch thread.
    DisallowThisThreadToAccessContextState();
    bool shutdown = false;
    std::vector<ExecutionOrShutdown> task_or_shutdown_list;
    while (!shutdown) {
      DequeueExecutionTasks(task_or_shutdown_list);
      for (auto& task_or_shutdown : task_or_shutdown_list) {
        if (std::holds_alternative<ShutdownSentinel>(task_or_shutdown)) {
          shutdown = true;
          break;
        }
        ABSL_VLOG(1) << "[MaterializationWorker] Processing ExecutionTask";
        auto& task = std::get<ExecutionTask>(task_or_shutdown);
        auto status = task.Run();
        if (!status.ok()) {
          ABSL_LOG(ERROR) << "[MaterializationWorker] ExecutionTask failed: "
                          << status;
        }
        // Free the task once it completes (or fails) so we release the shared
        // pointers to the argument and output buffers to free device memory.
        task_or_shutdown = ShutdownSentinel{};
      }
      task_or_shutdown_list.clear();
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
    for (auto& task_or_shutdown : execute_tasks_) {
      if (auto* execution_task =
              std::get_if<ExecutionTask>(&task_or_shutdown)) {
        execution_task->SetOutputNodesAsError(shutdown_status);
      }
    }
    execute_tasks_.clear();
  }
  std::thread materialize_thread_;
  std::thread execute_thread_;

  // Set to true when the MaterializationWorker is undergoing shutdown,
  // preventing concurrent or duplicate shutdown sequences from executing.
  std::atomic<bool> shutdown_{false};

  absl::Mutex materialize_mu_;
  // Queue of materialization tasks. Storing std::variant allows enqueuing
  // ShutdownSentinel as a shutdown sentinel / poison pill.
  std::vector<MaterializationOrShutdown> materialize_tasks_
      ABSL_GUARDED_BY(materialize_mu_);

  absl::Mutex execute_mu_;
  // Queue of execution tasks. Storing std::variant allows enqueuing
  // ShutdownSentinel as a shutdown sentinel / poison pill.
  std::vector<ExecutionOrShutdown> execute_tasks_ ABSL_GUARDED_BY(execute_mu_);
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
                             MaterializationMode materialization_mode,
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
    auto future = EventSnapshot::Record(node->device_index(), node->stream_id(),
                                        std::move(nodes_on_stream), reason,
                                        materialization_mode);
    GetMaterializationWorker().EnqueueStream(node->device_index(),
                                             node->stream_id());

    if (block_mode == MaterializationBlockMode::kBlockUntilStarted) {
      // Make sure that the stream task was enqueued successfully.
      TT_RETURN_IF_ERROR(future.Await());
    }

    // Record that we enqueued this stream.
    device_stream_ids.push_back(DeviceStreamId{
        .device_index = node->device_index(), .stream_id = node->stream_id()});
  }
  return absl::OkStatus();
}

}  // namespace

void ShutDownMaterializationState() { GetMaterializationWorker().Shutdown(); }

absl::Status Materialize(absl::Span<const SharedDeviceBufferList> nodes,
                         MaterializationReason reason,
                         MaterializationMode materialization_mode,
                         MaterializationBlockMode block_mode) {
  return MaterializeSpan(nodes, reason, materialization_mode, block_mode);
}

absl::Status Materialize(absl::Span<const DeviceBufferRef> buffer_refs,
                         MaterializationReason reason,
                         MaterializationMode materialization_mode) {
  return Materialize(buffer_refs, reason, materialization_mode,
                     MaterializationBlockMode::kBlockUntilStarted);
}

absl::Status Materialize(absl::Span<const DeviceBufferRef> buffer_refs,
                         MaterializationReason reason,
                         MaterializationMode materialization_mode,
                         MaterializationBlockMode block_mode) {
  return MaterializeSpan(buffer_refs, reason, materialization_mode, block_mode);
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
    TT_ASSIGN_OR_RETURN(
        DeviceBufferRef output_ref,
        DeviceBufferList::CreatePending(shape, device_index, stream_id));
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
    MaterializationMode mode,
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
    MaterializationReason reason, MaterializationMode mode,
    absl::Span<const SharedDeviceBufferList> must_materialize) {
  const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
  return MaterializeStream(device_index, stream_id, reason, mode,
                           must_materialize);
}

}  // namespace torch_tpu
