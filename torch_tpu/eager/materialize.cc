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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <future>
#include <iterator>
#include <memory>
#include <optional>
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
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/absl_vlog_is_on.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/MLIRContext.h"
#include "ATen/core/TensorBody.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/dynamism_utils.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/eager_mode.h"
#include "torch_tpu/eager/prevent_graph_splits.h"
#include "torch_tpu/eager/split_traversal.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/eager/traversal.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "torch_tpu/pjrt/pjrt_utils.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"
#include "xla/future.h"
#include "xla/hlo/translate/register.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/traceme.h"

ABSL_FLAG(std::optional<int64_t>, torch_tpu_internal_throttle_limit_gib,
          std::nullopt,
          "If set, throttle active materializations to try to stay below this "
          "memory limit (in GiB).");
ABSL_FLAG(std::optional<float>, torch_tpu_internal_throttle_limit_rate,
          std::nullopt,
          "If set, throttle active materializations to try to stay below this "
          "memory limit (as a fraction of total device memory). Must be in "
          "the range [0.0, 1.0].");

namespace torch_tpu {
namespace {

absl_nonnull std::unique_ptr<mlir::MLIRContext> MakeMlirContext() {
  auto context = std::make_unique<mlir::MLIRContext>();
  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();
  return context;
}

absl::Status SetOutputNodesAsMaterialized(std::vector<DeviceBufferRef>& outputs,
                                          PjRtBufferPointers results) {
  tsl::profiler::TraceMe trace("SetOutputNodesAsMaterialized");
  ABSL_VLOG(1) << "[SetOutputNodesAsMaterialized] Starting";
  PjRtBufferPointers buffers_to_assign;
  buffers_to_assign.reserve(outputs.size());
  auto i = 0;
  while (i < outputs.size()) {
    ABSL_VLOG(1) << "[AssignMaterializedBuffers] Assigning materialized "
                    "buffers for node "
                 << i;
    auto* absl_nonnull node = outputs[i].device_buffer_list().get();
    buffers_to_assign.clear();
    buffers_to_assign.reserve(node->size());
    if (i + node->size() > results.size()) {
      ABSL_VLOG(1)
          << "[AssignMaterializedBuffers] Not enough results to materialize "
             "node "
          << node;
      return TT_ERROR(error::kFailedPrecondition)
             << "Not enough results to materialize node " << node;
    }
    for (auto j = 0; j < node->size(); ++j) {
      ABSL_VLOG(1)
          << "[AssignMaterializedBuffers] Assigning materialized buffer for "
             "node "
          << node << " index " << j;
      buffers_to_assign.push_back(std::move(results[i + j]));
    }
    TT_RETURN_IF_ERROR(node->SetAsMaterialized(std::move(buffers_to_assign)))
        << "Failed to set node " << node << " as materialized";
    i += node->size();
  }
  ABSL_VLOG(1) << "[AssignMaterializedBuffers] Assigned materialized buffers "
                  "for nodes";
  return absl::OkStatus();
}

struct ExecutionTask {
  std::vector<DeviceBufferRef> inputs;
  std::vector<DeviceBufferRef> outputs;
  CompiledKernel compiled_kernel;
  std::string task_name;
};

struct MaterializationTask {
  std::vector<SharedDeviceBufferList> nodes_to_materialize;
  xla::Promise<void> completion_promise;
  MaterializationMode materialization_mode = MaterializationMode::kSplitGraph;
};

using MaterializationJob = std::variant<ExecutionTask, MaterializationTask>;

ExecutionTask CreateExecutionTask(Traversal traversal,
                                  CompiledKernel compiled_kernel) {
  std::string task_name;
  if (ABSL_VLOG_IS_ON(1)) {
    task_name = absl::StrCat(traversal.cache_key());
  }
  Traversal::Parts parts = traversal.IntoParts();
  return ExecutionTask{.inputs = std::move(parts.inputs),
                       .outputs = std::move(parts.outputs),
                       .compiled_kernel = std::move(compiled_kernel),
                       .task_name = std::move(task_name)};
}

absl::StatusOr<std::vector<xla::PjRtBuffer* absl_nullable>> GetRootArgs(
    absl::Span<const DeviceBufferRef> inputs) {
  std::vector<xla::PjRtBuffer*> root_args;
  for (const auto&& [index, input] : llvm::enumerate(inputs)) {
    switch (input.state()) {
      case DeviceBufferRefState::kMaterialized: {
        ABSL_VLOG(1) << "kMaterialized DeviceBufferRef index: " << index;
        TT_ASSIGN_OR_RETURN(xla::PjRtBuffer * pjrt_buffer,
                            input.GetOrMaterializeBuffer());
        root_args.push_back(pjrt_buffer);
        break;
      }
      case DeviceBufferRefState::kZeroSize:
        // Zero-sized constants are not arguments.
        break;
      case DeviceBufferRefState::kPlaceholder:
        return TT_ERROR(error::kInternal)
               << "Materialize was called on a placeholder tensor. This "
                  "should never happen.\nkPlaceholder tensors should only "
                  "appear in compiled mode, which should never try to "
                  "materialize tensors."
               << input.DebugString();

      case DeviceBufferRefState::kDeferred:
        return TT_ERROR(error::kInternal)
               << "Traversal input is unexpectedly deferred";
      default:
        return TT_ERROR(error::kInternal)
               << "Traversal input has unknown state";
    }
  }
  return root_args;
}

// PRECONDITION: executables must be a sequence that is compatible with the
// inputs and composable, as they will be executed in order and the outputs of
// an executable are the inputs to the next. The final execuatable returns a
// complete set of outputs for each node in sequential blocks. That is, either
// all outputs of a node appear contiguously and in the same order as in
// results, or none of them do. Furthermore, the order of nodes must be
// consistent between the execution and results. For example, if node A is size
// 3, node B is size 1, and node C is size 2, then the output order must be [A0,
// A1, A2, B1, C0, C1]. This is established by all Materialize() functions in
// this file.
absl::Status ExecuteMaterializationJob(
    absl::Span<const DeviceBufferRef> inputs,
    absl::Span<const DeviceBufferRef> outputs,
    std::vector<SharedLoadedExecutable> executables,
    std::string_view task_name = "anonymous") {
  tsl::profiler::TraceMe trace("ExecuteMaterializationJob");
  ABSL_VLOG(1) << "[ExecuteMaterializationJob]: task_name=" << task_name
               << " input arg count: " << inputs.size()
               << " output arg count: " << outputs.size();
  ABSL_CHECK(!executables.empty()) << "No executables to execute";  // CRASH_OK

  TT_ASSIGN_OR_RETURN(std::vector<xla::PjRtBuffer*> root_args,
                      GetRootArgs(inputs));

  ABSL_VLOG(1)
      << "[ExecuteMaterializationJob]: Arguments materialization completed "
         "for task name="
      << task_name << ", root_args size: " << root_args.size();

  std::vector<PjRtBufferPointers> intermediate_results;
  for (auto& executable : executables) {
    TT_ASSIGN_OR_RETURN(
        PjRtBufferPointers results, Execute(executable, std::move(root_args)),
        _.SetPrepend() << "failed to enqueue execution for task_name="
                       << task_name << ": ");
    root_args.clear();
    root_args.reserve(results.size());
    for (const auto& result : results) {
      root_args.push_back(result.get());
    }
    intermediate_results.push_back(std::move(results));
  }
  PjRtBufferPointers final_results = std::move(intermediate_results.back());

  ABSL_VLOG(1)
      << "[ExecuteMaterializationJob]: Enqueued execution for task_name="
      << task_name << ", results size: " << final_results.size();

  ABSL_VLOG(1) << "[ExecuteMaterializationJob]: Materialization enqueue has "
                  "completed for task_name="
               << task_name;

  std::vector<DeviceBufferRef> output_refs;
  for (const auto& output : outputs) {
    output_refs.push_back(output);
  }
  return SetOutputNodesAsMaterialized(output_refs, std::move(final_results));
}

// An InFlightExecution is an execution that has started but may not yet be
// finished. This is used to measure memory usage to determine when to throttle.
struct InFlightExecution {
  // The first output node of the execution. All output nodes complete at the
  // same time, so we only need to keep track of one.
  std::weak_ptr<DeviceBufferList> node;
  // The total execution cost, in bytes, including input size, output size,
  // program size, and temporary stack memory, less any input/output aliasing.
  int64_t size_bytes;
};

// Determines the memory throttling limit based on the flags.
// If both absolute and relative limits are set, then the smaller one is used.
// If the device memory limit cannot be determined, then the relative limit is
// ignored.
// Returns nullopt if no throttling should be applied.
std::optional<int64_t> GetThrottleLimitBytes() {
  std::optional<int64_t> throttle_limit_absolute =
      absl::GetFlag(FLAGS_torch_tpu_internal_throttle_limit_gib);
  if (throttle_limit_absolute.has_value() && *throttle_limit_absolute > 0) {
    // Convert from GiB to bytes.
    throttle_limit_absolute =
        throttle_limit_absolute.value() * 1024 * 1024 * 1024;
  }

  std::optional<float> throttle_limit_rate =
      absl::GetFlag(FLAGS_torch_tpu_internal_throttle_limit_rate);
  if (!throttle_limit_rate.has_value()) {
    // No relative limit is set, just use the absolute limit (possibly nullopt)
    return throttle_limit_absolute;
  }
  ABSL_CHECK(throttle_limit_rate.value() >= 0.0 &&  // CRASH_OK
             throttle_limit_rate.value() <= 1.0)
      << "Invalid value for --torch_tpu_internal_throttle_limit_fraction: "
      << throttle_limit_rate.value() << ". Must be in range [0.0, 1.0].";

  // Try to get the maximum byte size from the PjRtClient.
  // This only works if there is a single addressable device, which is the
  // expectation for current torch.distributed configurations on TorchTPU.
  int64_t device_bytes_limit = -1;
  if (auto* client = GetPjRtClient(); client != nullptr) {
    if (auto devices = client->addressable_devices(); devices.size() == 1) {
      const auto& device = devices[0];
      if (auto stats = device->GetAllocatorStats(); stats.ok()) {
        device_bytes_limit = stats->bytes_limit.value_or(-1);
      }
    }
  }
  if (device_bytes_limit < 0) {
    // Could not get the device bytes limit, so just use the absolute limit
    // (possibly nullopt).
    return throttle_limit_absolute;
  }
  int64_t throttle_limit_relative =
      static_cast<int64_t>(device_bytes_limit * throttle_limit_rate.value());
  if (throttle_limit_absolute.has_value()) {
    // If both absolute and relative limits are set, then we use the smaller
    // one.
    return std::min(throttle_limit_absolute.value(), throttle_limit_relative);
  } else {
    // Only a relative limit is set, so we use that.
    return throttle_limit_relative;
  }
}

// Blocks the calling thread until enqueuing the given executable onto the queue
// would not exceed the throttle limit, or until the queue is empty.
// Returns the size of the executable.
int64_t ApplyMemoryThrottling(const int64_t throttle_limit_bytes,
                              int64_t& in_flight_bytes,
                              std::queue<InFlightExecution>& in_flight_nodes,
                              const xla::PjRtLoadedExecutable& executable) {
  tsl::profiler::TraceMe t("ApplyMemoryThrottling");
  // Compute the execution cost of the executable.
  const auto memory_stats_status = executable.GetCompiledMemoryStats();
  int64_t execution_size_bytes;
  if (memory_stats_status.ok()) {
    const xla::CompiledMemoryStats& memory_stats = *memory_stats_status;
    // This is technically the lower bound of the execution cost, but it's a
    // reasonable measurement for throttling purposes.
    // See comment in PjRtExecutable::CompiledMemoryStats for details.
    execution_size_bytes = memory_stats.generated_code_size_in_bytes +
                           memory_stats.argument_size_in_bytes +
                           memory_stats.output_size_in_bytes -
                           memory_stats.alias_size_in_bytes +
                           memory_stats.temp_size_in_bytes;
  } else {
    // If we can't get the memory stats, then we conservatively assume
    // the executable is so big that we need to flush the queue both
    // before and after execution.
    execution_size_bytes = throttle_limit_bytes;
  }

  // Block on all executions until either the queue is empty or adding the new
  // executable would not exceed the throttle limit.
  while (!in_flight_nodes.empty() &&
         in_flight_bytes + execution_size_bytes >= throttle_limit_bytes) {
    // Pop the first node and update the in-flight bytes.
    InFlightExecution in_flight_node = in_flight_nodes.front();
    in_flight_nodes.pop();
    in_flight_bytes -= in_flight_node.size_bytes;
    // If the node is expired, then that means it was computed, used,
    // and freed, so it's done.
    if (auto maybe_list = in_flight_node.node.lock()) {
      const auto* materialized_buffers = maybe_list->materialized_buffers();
      // If the Await() fails, then the execution failed, which means it's no
      // longer consuming resources.
      if (materialized_buffers != nullptr &&
          materialized_buffers->Await().ok() &&
          materialized_buffers->size() > 0) {
        // As soon as the first PjRtBuffer is ready, then the whole execution is
        // complete.
        // If the execution fails, then the result will be an error, but it's
        // done using resources so we can proceed.
        if (auto* buffer = (*materialized_buffers)[0]) {
          buffer->GetReadyFuture().Await().IgnoreError();
        }
      }
    }
  }
  return execution_size_bytes;
}

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
                                 MaterializationMode materialization_mode) {
    ABSL_VLOG(1) << "[MaterializationWorker] Enqueuing " << nodes.size()
                 << " nodes for materialization";
    auto [promise, future] = xla::MakePromise<void>();
    TT_MUTEX_LOCK(lock, materialize_mu_);
    materialize_jobs_.push(
        MaterializationTask{.nodes_to_materialize = std::move(nodes),
                            .completion_promise = std::move(promise),
                            .materialization_mode = materialization_mode});
    return future;
  }

  absl::StatusOr<std::vector<DeviceBufferRef>> EnqueueExecutable(
      SharedLoadedExecutable executable, std::vector<DeviceBufferRef> inputs,
      absl::Span<const Shape> output_shapes, std::string_view task_name = "") {
    // Create a set of output DeviceBufferRefs to hold the materialized results.
    std::vector<DeviceBufferRef> outputs;
    outputs.reserve(output_shapes.size());
    for (const auto& shape : output_shapes) {
      // Make a placeholder and then immediately put it in the
      // pending-materialization state.
      TT_ASSIGN_OR_RETURN(
          DeviceBufferRef output_ref,
          DeviceBufferList::MakePlaceholder(shape.Dimensions(), shape.Dtype()));
      TT_RETURN_IF_ERROR(output_ref.device_buffer_list()->SetAsMaterialized());
      outputs.push_back(std::move(output_ref));
    }

    // Create a promise/future that is already done using the executable.
    LoadedExecutablePromise promise;
    CompiledKernel compiled_kernel{.fixed_shape_kernel = promise.get_future()};
    promise.set_value(std::move(executable));

    ABSL_VLOG(1) << "[MaterializationWorker] Enqueuing executable: task_name="
                 << (task_name.empty() ? "anonymous" : task_name)
                 << " input arg count: " << inputs.size()
                 << "  output arg count: " << outputs.size();

    ExecutionTask task = {.inputs = std::move(inputs),
                          .outputs = outputs,  // intentional copy
                          .compiled_kernel = std::move(compiled_kernel),
                          .task_name = std::string(task_name)};
    TT_MUTEX_LOCK(lock, execute_mu_);
    execute_jobs_.push(std::move(task));

    return outputs;
  }

 private:
  MaterializationTask DequeueMaterializationJob() {
    TT_MUTEX_LOCK(lock, materialize_mu_);
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
    TT_MUTEX_LOCK(lock, execute_mu_);
    if (execute_jobs_.empty()) {
      execute_mu_.Await(absl::Condition(
          +[](std::queue<ExecutionTask>* jobs) { return !jobs->empty(); },
          &execute_jobs_));
    }
    ExecutionTask popped_job = std::move(execute_jobs_.front());
    execute_jobs_.pop();
    return popped_job;
  }

  void ProcessExecutionTask(
      ExecutionTask task, const std::optional<int64_t>& throttle_limit_bytes,
      int64_t& in_flight_bytes,
      std::queue<InFlightExecution>& in_flight_executions) {
    tsl::profiler::TraceMe trace("ProcessExecutionTask");
    std::string_view task_name = "anonymous";
    if (!task.task_name.empty()) {
      task_name = task.task_name;
    }

    ABSL_VLOG(1) << "[MaterializationWorker] Waiting for compilation for "
                    "task_name="
                 << task_name;

    if (ABSL_VLOG_IS_ON(1)) {
      // Wait for the compilation to complete with a timeout.
      while (task.compiled_kernel.fixed_shape_kernel.wait_for(
                 std::chrono::seconds(5)) == std::future_status::timeout) {
        ABSL_VLOG(1) << "Still waiting for compilation of task_name="
                     << task_name;
      }
      if (task.compiled_kernel.dynamic_kernel_adapter.has_value()) {
        while (task.compiled_kernel.dynamic_kernel_adapter->preamble.wait_for(
                   std::chrono::seconds(5)) == std::future_status::timeout) {
          ABSL_VLOG(1) << "Still waiting for preamble compilation of task_name="
                       << task_name;
        }
      }
    }
    ABSL_VLOG(1)
        << "[MaterializationWorker] Compilation complete for task_name="
        << task_name;

    std::vector<SharedLoadedExecutable> cached_executables;

    if (task.compiled_kernel.dynamic_kernel_adapter.has_value()) {
      absl::StatusOr<SharedLoadedExecutable> preamble =
          task.compiled_kernel.dynamic_kernel_adapter->preamble.get();
      if (!preamble.ok()) {
        ABSL_VLOG(1) << "[MaterializationWorker] Failed to compile "
                        "task_name="
                     << task_name << ": " << preamble.status();
        SetOutputNodesAsError(task.outputs, preamble.status());
        return;
      }
      cached_executables.push_back(std::move(*preamble));
    }

    absl::StatusOr<SharedLoadedExecutable> fixed_shape_kernel =
        task.compiled_kernel.fixed_shape_kernel.get();
    if (!fixed_shape_kernel.ok()) {
      ABSL_VLOG(1) << "[MaterializationWorker] Failed to compile "
                      "task_name="
                   << task_name << ": " << fixed_shape_kernel.status();
      SetOutputNodesAsError(task.outputs, fixed_shape_kernel.status());
      return;
    }
    cached_executables.push_back(std::move(*fixed_shape_kernel));

    ABSL_VLOG(1) << "[MaterializationWorker] Cached executables size: "
                 << cached_executables.size();

    int64_t execution_size_bytes = 0;
    for (const auto& cached_executable : cached_executables) {
      if (throttle_limit_bytes.has_value()) {
        ABSL_VLOG(1) << "[MaterializationWorker] Applying memory throttling "
                        "for task_name="
                     << task_name;
        execution_size_bytes +=
            ApplyMemoryThrottling(*throttle_limit_bytes, in_flight_bytes,
                                  in_flight_executions, *cached_executable);
      }
    }

    ABSL_VLOG(1) << "[MaterializationWorker] Executing job for task_name="
                 << task_name;
    absl::Status status = ExecuteMaterializationJob(
        task.inputs, task.outputs, std::move(cached_executables), task_name);
    if (!status.ok()) {
      ABSL_VLOG(1)
          << "[MaterializationWorker] ExecuteMaterializationJob failed "
             "for task_name="
          << task_name << " with status: " << status;
      SetOutputNodesAsError(task.outputs, status);
      return;
    }

    // Add the now in-flight execution to the throttling queue.
    if (throttle_limit_bytes.has_value()) {
      ABSL_VLOG(1)
          << "[MaterializationWorker] Adding in-flight execution of size "
          << execution_size_bytes / (1024 * 1024 * 1024)
          << " GiB for task_name=" << task_name;
      InFlightExecution in_flight_execution = {
          .node = task.outputs[0].device_buffer_list(),
          .size_bytes = execution_size_bytes,
      };
      in_flight_bytes += execution_size_bytes;
      in_flight_executions.push(std::move(in_flight_execution));
    }

    ABSL_VLOG(1) << "[MaterializationWorker] ExecuteMaterializationJob "
                    "succeeded for task_name="
                 << task_name;
  }

  absl::StatusOr<std::vector<ExecutionTask>> ProcessMaterializationTask(
      MaterializationTask& task, mlir::MLIRContext& mlir_context) {
    ABSL_VLOG(1)
        << "[MaterializationWorker] Processing MaterializationTask with "
        << task.nodes_to_materialize.size() << " nodes";
    LogDeferredNodes(task.nodes_to_materialize,
                     /* msg_prefix= */ "  Input node");

    ABSL_VLOG(1) << "[MaterializationWorker] Getting leaf nodes";
    std::vector<SharedDeviceBufferList> all_nodes =
        std::move(task.nodes_to_materialize);
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

    std::vector<DeviceBufferRef> refs;
    for (const auto& node : all_nodes) {
      for (int64_t i = 0; i < node->size(); ++i) {
        TT_ASSIGN_OR_RETURN(DeviceBufferRef ref,
                            DeviceBufferRef::Create(node, i));
        refs.push_back(std::move(ref));
      }
    }

    ABSL_VLOG(1) << "[MaterializationWorker] Creating traversal";
    absl::StatusOr<Traversal> traversal;
    {
      tsl::profiler::TraceMe t("Traversal::Create");
      TT_ASSIGN_OR_RETURN(traversal, Traversal::Create(std::move(refs)));
    }

    ABSL_VLOG(3) << "[MaterializationWorker] Traversal created: "
                 << GetGraphviz(*traversal);

    std::vector<Traversal> traversals;

    // Split the graph only if both are true:
    //   - `materialization_mode` is `kSplitGraph`
    //   - `PreventGraphSplit()` returns false (i.e. do not prevent the graph
    //      to be split)
    if (task.materialization_mode == MaterializationMode::kSplitGraph) {
      TT_ASSIGN_OR_RETURN(bool prevent_graph_split,
                          PreventGraphSplit(*traversal));

      if (!prevent_graph_split) {
        // Split the traversal while nodes are still in the deferred state.
        ABSL_VLOG(1) << "[MaterializationWorker] Splitting traversal";
        {
          tsl::profiler::TraceMe t("SplitTraversal");
          TT_ASSIGN_OR_RETURN(traversals,
                              SplitTraversal(std::move(*traversal)));
        }

        ABSL_VLOG(1) << "[MaterializationWorker] Split traversal into "
                     << traversals.size() << " traversals";
      } else {
        traversals.push_back(std::move(*traversal));
      }
    }

    TT_RETURN_IF_ERROR(
        PropagateBoundedDynamism(absl::MakeSpan(traversals), mlir_context));

    std::vector<ExecutionTask> execution_tasks;
    for (auto& split_traversal : traversals) {
      ABSL_VLOG(1) << "[MaterializationWorker] Compiling traversal";
      auto compilation_mode = (GetEagerMode() == EagerMode::kOptimized)
                                  ? CompilationMode::kFastRuntime
                                  : CompilationMode::kFastCompile;
      absl::StatusOr<CompiledKernel> compiled_kernel;
      {
        tsl::profiler::TraceMe t("CompileTraversal");
        TT_ASSIGN_OR_RETURN(compiled_kernel,
                            split_traversal.Compile(compilation_mode));
      }

      // Mark all outputs of the split as scheduled/materialized.
      absl::flat_hash_set<const DeviceBufferList*> marked_materialized;
      for (const auto& output : split_traversal.outputs()) {
        if (!marked_materialized.insert(output.device_buffer_list().get())
                 .second) {
          continue;
        }

        ABSL_VLOG(1)
            << "[MaterializationWorker] Marking output as materialized: "
            << output.device_buffer_list();

        TT_RETURN_IF_ERROR(output.device_buffer_list()->SetAsMaterialized());
      }

      ABSL_VLOG(1) << "[MaterializationWorker] Enqueuing traversal: cache_key="
                   << split_traversal.cache_key()
                   << " traversal input arg count: "
                   << split_traversal.inputs().size()
                   << " traversal output arg count: "
                   << split_traversal.outputs().size();

      // Mark all deferred ops in the split as having been executed (scheduled).
      for (const auto& node : split_traversal.execution_order()) {
        auto* deferred_op = node->deferred_op();
        if (deferred_op) {
          deferred_op->mark_executed();
        }
      }

      execution_tasks.push_back(CreateExecutionTask(
          std::move(split_traversal), std::move(*compiled_kernel)));
    }

    return execution_tasks;
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
            TT_MUTEX_LOCK(lock, execute_mu_);
            for (auto& task : *tasks) {
              execute_jobs_.push(std::move(task));
            }
          }
        }

        job.completion_promise.Set(tasks.status());
      }
    });

    execute_thread_ = std::thread([this]() {
      // Throttling parameters
      const std::optional<int64_t> throttle_limit_bytes =
          GetThrottleLimitBytes();
      int64_t in_flight_bytes = 0;
      std::queue<InFlightExecution> in_flight_executions;

      while (true) {
        ExecutionTask job = DequeueExecutionJob();
        ABSL_VLOG(1) << "[MaterializationWorker] Processing ExecutionTask";
        ProcessExecutionTask(std::move(job), throttle_limit_bytes,
                             in_flight_bytes, in_flight_executions);
      }
    });
  }

  // Propagates bounded dynamism annotations from one traversal to others
  // when one traversal's output is bounded dynamic and is another traversal's
  // input.
  absl::Status PropagateBoundedDynamism(absl::Span<Traversal> traversals,
                                        mlir::MLIRContext& mlir_context);

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

absl::Status MaterializationWorker::PropagateBoundedDynamism(
    absl::Span<Traversal> traversals, mlir::MLIRContext& mlir_context) {
  for (auto& traversal : traversals) {
    if (!traversal.IsBoundedDynamic()) {
      continue;
    }
    ABSL_VLOG(1) << "[PropagateBoundedDynamism] Traversal: "
                 << traversal.DebugString();
    TT_ASSIGN_OR_RETURN(std::vector<DeviceRefDimensions> output_dimensions,
                        GetTraversalOutputDimensions(mlir_context, traversal));
    for (const auto& output_dimension : output_dimensions) {
      const DeviceBufferRef& ref = output_dimension.ref;
      const auto& dims = output_dimension.dims;
      for (const mlir::stablehlo::DimensionInfo& dim : dims) {
        if (dim.boundOp.has_value()) {
          TT_RETURN_IF_ERROR(ref.MarkDynamic(dim.boundOpDim, 2, dim.size));
          ABSL_VLOG(1) << "[PropagateBoundedDynamism] Marked dynamic: "
                       << ref.DebugString() << " dimension: " << dim.boundOpDim
                       << " upper bound: " << dim.size;
          // Multiple dynamic dimension for a tensor is currently not supported.
          break;
        }
      }
    }
  }
  return absl::OkStatus();
}

// Common pathway for all Materialize() overloads.
absl::Status MaterializeImpl(
    absl::Span<const SharedDeviceBufferList> nodes_to_materialize,
    MaterializationMode materialization_mode) {
  ABSL_VLOG(1) << "[MaterializeImpl] Materializing "
               << nodes_to_materialize.size() << " nodes";
  if (nodes_to_materialize.empty()) {
    return absl::OkStatus();
  }

  tsl::profiler::TraceMe t([] { return "MaterializeImpl"; });

  std::vector<SharedDeviceBufferList> nodes(nodes_to_materialize.begin(),
                                            nodes_to_materialize.end());

  auto future = GetMaterializationWorker().EnqueueNodes(std::move(nodes),
                                                        materialization_mode);
  TT_RETURN_IF_ERROR(future.Await()).SetPrepend()
      << "materialization failed with: ";

  // Check that all nodes to materialize have indeed been materialized.
  for (auto& node : nodes_to_materialize) {
    TT_RET_CHECK(node->IsMaterializedOrZeroState(), error::kInternal)
        << "Materialization failed for node " << node;
  }

  return absl::OkStatus();
}

}  // namespace

absl::Status Materialize(absl::Span<const SharedDeviceBufferList> nodes,
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
  return MaterializeImpl(nodes_to_materialize, materialization_mode);
}

absl::Status Materialize(absl::Span<const DeviceBufferRef> buffer_refs,
                         MaterializationMode materialization_mode) {
  if (buffer_refs.empty()) {
    return absl::OkStatus();
  }
  absl::flat_hash_set<SharedDeviceBufferList> unique_deferred_nodes;
  std::vector<SharedDeviceBufferList> nodes_to_materialize;
  for (const DeviceBufferRef& buffer_ref : buffer_refs) {
    switch (buffer_ref.state()) {
      case DeviceBufferRefState::kMaterialized:
      case DeviceBufferRefState::kZeroSize:
        // Only materialize the DeviceBufferLists we actually need.
        // If a multi-output deferred op produces zero-sized buffer_refs, we
        // don't need to materialize the op, unless we also need a non-zero
        // sized output for a different buffer_ref from that same node.
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
  return MaterializeImpl(nodes_to_materialize, materialization_mode);
}

absl::StatusOr<DeviceBufferRef> GetMaterialized(const at::Tensor& tensor) {
  tsl::profiler::TraceMe trace("GetMaterialized");
  // Make sure the base DeviceBufferRef is materialized
  const auto* tensor_impl = tensor.unsafeGetTensorImpl();
  TT_RET_CHECK(tensor_impl, error::kInvalidArgument) << "tensor is undefined";
  TT_ASSIGN_OR_RETURN(const DeviceBufferRef base_buffer_ref,
                      GetBaseBufferFromAtTensor(*tensor_impl));
  TT_RETURN_IF_ERROR(Materialize(base_buffer_ref));

  // Get the view DeviceBufferRef (may be the same as the base)
  TT_ASSIGN_OR_RETURN(const DeviceBufferRef view_buffer_ref,
                      GetBufferFromAtTensor(tensor));
  // Materialize the view (no-op if the tensor is a continuous base tensor)
  TT_RETURN_IF_ERROR(Materialize(view_buffer_ref));
  return view_buffer_ref;
}

absl::StatusOr<std::vector<DeviceBufferRef>> GetMaterialized(
    absl::Span<const at::Tensor> tensors) {
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
                        GetBaseBufferFromAtTensor(*tensor_impl));
    base_buffer_refs.push_back(base_buffer_ref);
  }
  TT_RETURN_IF_ERROR(Materialize(base_buffer_refs));

  // Materialize all of the views (no-op if all tensors are contiguous bases)
  std::vector<DeviceBufferRef> view_buffer_refs;
  view_buffer_refs.reserve(tensors.size());
  for (const at::Tensor& tensor : tensors) {
    TT_ASSIGN_OR_RETURN(const DeviceBufferRef view_buffer_ref,
                        GetBufferFromAtTensor(tensor));
    view_buffer_refs.push_back(view_buffer_ref);
  }
  TT_RETURN_IF_ERROR(Materialize(view_buffer_refs));

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
    SharedLoadedExecutable executable, std::vector<DeviceBufferRef> inputs,
    absl::Span<const Shape> output_shapes, std::string_view task_name) {
  return GetMaterializationWorker().EnqueueExecutable(
      std::move(executable), std::move(inputs), output_shapes, task_name);
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
