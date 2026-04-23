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

#include "torch_tpu/eager/materialize_new.h"

#include <algorithm>
#include <iterator>
#include <memory>
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
#include "torch_tpu/common/compilation.h"
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
  absl::Status Materialize();

  absl::Status BlockOnPendingMaterializations();

 private:
  friend class absl::NoDestructor<MaterializationWorker>;
  static constexpr int kNumQueues = 2;

  mutable absl::Mutex mutex_;
  int dispatch_queue_id_ ABSL_GUARDED_BY(mutex_) = 0;
  int execution_queue_id_ ABSL_GUARDED_BY(mutex_) = 0;
  std::vector<SharedDeviceBufferList> queue_[kNumQueues] ABSL_GUARDED_BY(
      mutex_);
  bool queue_full_[kNumQueues] ABSL_GUARDED_BY(mutex_);
  bool must_exit_ ABSL_GUARDED_BY(mutex_) = false;
  absl::Status last_status_ ABSL_GUARDED_BY(mutex_) = absl::OkStatus();
  std::thread execution_thread_;

  MaterializationWorker() {
    std::fill(queue_full_, queue_full_ + kNumQueues, false);
    execution_thread_ = std::thread([this]() { ThreadLoop(); });
  }

  ~MaterializationWorker() { Exit(); }

  absl::Status GetLastStatus() const {
    absl::MutexLock lock(mutex_);
    return last_status_;
  }

  void IncrementQueueId(int& id) { id = (id + 1) % kNumQueues; }

  void Exit();

  bool IsReadyToDispatch() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return !queue_full_[dispatch_queue_id_];
  }

  bool IsExecutionThreadCaughtUp() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return (execution_queue_id_ == dispatch_queue_id_) &&
           !queue_full_[dispatch_queue_id_];
  }

  bool IsReadyToExitOrExecute() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return must_exit_ || queue_full_[execution_queue_id_];
  }

  void ThreadLoop();

  std::vector<absl::Span<const SharedDeviceBufferList>> SplitQueueIntoRegions(
      const std::vector<SharedDeviceBufferList>& queue) const;

  absl::Status MaterializeQueue(
      const std::vector<SharedDeviceBufferList>& queue);
};

absl::Status MaterializationWorker::OnNewOpDispatch(
    const SharedDeviceBufferList& device_buffer_list) {
  tsl::profiler::TraceMe t("Worker_OnNewOpDispatch");
  TT_RETURN_IF_ERROR(GetLastStatus());
  absl::MutexLock lock(mutex_);
  // Wait if the current dispatch queue is being processed.
  mutex_.Await(
      absl::Condition(this, &MaterializationWorker::IsReadyToDispatch));
  queue_[dispatch_queue_id_].push_back(device_buffer_list);
  return absl::OkStatus();
}

absl::Status MaterializationWorker::Materialize() {
  tsl::profiler::TraceMe t("Worker_Materialize");
  TT_RETURN_IF_ERROR(GetLastStatus());
  {
    absl::MutexLock lock(mutex_);
    // Wait if the current dispatch queue is being processed.
    mutex_.Await(
        absl::Condition(this, &MaterializationWorker::IsReadyToDispatch));
    if (queue_[dispatch_queue_id_].empty()) {
      return absl::OkStatus();
    }
    queue_full_[dispatch_queue_id_] = true;
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
    std::vector<SharedDeviceBufferList> current_queue;
    int current_execution_id;

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
      ABSL_CHECK(queue_full_[execution_queue_id_]);  // CRASH_OK
      std::swap(current_queue, queue_[execution_queue_id_]);
      current_execution_id = execution_queue_id_;
    }

    // Materialize the claimed queue without holding the mutex.
    absl::Status status = MaterializeQueue(current_queue);

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
      std::swap(current_queue, queue_[current_execution_id]);
      queue_[current_execution_id].clear();
      queue_full_[current_execution_id] = false;
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

  if (deferred_op->has_been_executed()) {
    return true;
  }
  if (deferred_op->split_mode() == OpSplitMode::kSplitAfter) {
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

absl::Status MaterializationWorker::MaterializeQueue(
    const std::vector<SharedDeviceBufferList>& queue) {
  tsl::profiler::TraceMe t("Worker_MaterializeQueue");

  ABSL_VLOG(1) << ">>> MaterializationWorker::MaterializeQueue "
               << queue.size();

  auto regions = SplitQueueIntoRegions(queue);

  // Launch compilations for the identified regions.
  auto compilation_mode = (GetEagerMode() == EagerMode::kDeferAndFuse)
                              ? CompilationMode::kFastRuntime
                              : CompilationMode::kFastCompile;

  struct ExecutionTask {
    std::string name;
    std::vector<DeviceBufferRef> arguments;
    std::vector<DeviceBufferRef> outputs;
    CompiledKernel compiled_kernel;
    std::vector<SharedDeviceBufferList> execution_order;
  };

  std::vector<ExecutionTask> execution_tasks;
  execution_tasks.reserve(regions.size());

  for (const auto& region : regions) {
    TT_ASSIGN_OR_RETURN(auto traversal,
                        Traversal::CreateFromLinearRegion(region));
    ABSL_VLOG(1) << "==== TRAVERSAL ===\n" << traversal.DebugString();

    ABSL_VLOG(1) << "[MaterializationWorker] Launching compilation for "
                    "region: cache_key="
                 << traversal.cache_key();
    TT_ASSIGN_OR_RETURN(auto compiled_kernel,
                        traversal.Compile(compilation_mode));

    // Mark all outputs of the split as scheduled/materialized.
    {
      absl::flat_hash_set<const DeviceBufferList*> marked_materialized;
      for (const auto& output : traversal.outputs()) {
        if (marked_materialized.insert(output.device_buffer_list().get())
                .second) {
          ABSL_VLOG(1)
              << "[MaterializationWorker] Marking output as materialized: "
              << output.device_buffer_list();
          TT_RETURN_IF_ERROR(output.device_buffer_list()->SetAsMaterialized());
        }
      }
    }

    // Mark all deferred ops in the task as having been executed (or scheduled).
    for (const auto& node : traversal.execution_order()) {
      auto* deferred_op = node->deferred_op();
      // We need to check for (deferred_op != nullptr) since nodes producing
      // region outputs have been materialized and no longer have a DeferredOp
      // attached.
      if (deferred_op) {
        deferred_op->mark_executed();
      }
    }

    std::string name = absl::StrCat(traversal.cache_key());
    Traversal::Parts traversal_parts = traversal.IntoParts();
    execution_tasks.push_back(ExecutionTask{
        .name = std::move(name),
        .arguments = std::move(traversal_parts.arguments),
        .outputs = std::move(traversal_parts.outputs),
        .compiled_kernel = std::move(compiled_kernel),
        .execution_order = std::move(traversal_parts.execution_order)});
  }

  // Launch compiled kernels.
  for (const auto& execution_task : execution_tasks) {
    ABSL_VLOG(1) << "[MaterializationWorker] Waiting for compilation of region "
                 << execution_task.name;

    std::vector<SharedLoadedExecutableWithMetadata> executables;
    if (execution_task.compiled_kernel.dynamic_kernel_adapter) {
      TT_ASSIGN_OR_RETURN(auto preamble,
                          execution_task.compiled_kernel.dynamic_kernel_adapter
                              ->preamble.get());
      executables.push_back(std::move(preamble));
    }
    TT_ASSIGN_OR_RETURN(
        auto fixed_shape_kernel,
        execution_task.compiled_kernel.fixed_shape_kernel.get());
    executables.push_back(std::move(fixed_shape_kernel));

    ABSL_VLOG(1) << "[MaterializationWorker] Compilation of region "
                 << execution_task.name << " is done";

    ABSL_VLOG(1) << "[MaterializationWorker] Executing region "
                 << execution_task.name
                 << " arguments: " << execution_task.arguments.size()
                 << " outputs: " << execution_task.outputs.size();

    TT_RETURN_IF_ERROR(ExecuteMaterializationJob(
        execution_task.arguments, execution_task.outputs, executables,
        execution_task.name));

    ABSL_VLOG(1)
        << "[MaterializationWorker] Marking region ops as executed: task="
        << execution_task.name;
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
  (void)nodes_to_materialize;
  (void)reason;
  return MaterializationWorker::GetInstance().Materialize();
}

absl::Status BlockOnPendingMaterializations() {
  ABSL_VLOG(1) << ">>> BlockOnPendingMaterializations";
  return MaterializationWorker::GetInstance().BlockOnPendingMaterializations();
}

}  // namespace torch_tpu
