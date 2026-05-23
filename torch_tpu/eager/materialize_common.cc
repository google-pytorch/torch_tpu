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

#include "torch_tpu/eager/materialize_common.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/declare.h"
#include "absl/log/absl_log.h"
#include "absl/log/absl_vlog_is_on.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/MLIRContext.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/compilation_spec.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/dynamism_utils.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/eager_mode.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/eager/traversal.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/pjrt/pjrt_utils.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/traceme.h"

ABSL_DECLARE_FLAG(bool, torch_tpu_internal_enable_new_materialization);

namespace torch_tpu {

namespace {

#ifndef NDEBUG
// Formatting helper for error messages when VerifyPerNodeOutputs fails for any
// reason.
absl::Status InvalidNodeOutputsError(
    absl::Span<const DeviceBufferRef> outputs) {
  std::ostringstream outputs_error_stream;
  if (!outputs.empty()) {
    for (const auto& output : outputs) {
      outputs_error_stream << "\ndevice buffer list: "
                           << output.device_buffer_list().get() << ", buffer #"
                           << output.index();
    }
  } else {
    outputs_error_stream << "<no outputs>";
  }
  return TT_ERROR(error::kInternal)
         << "execution task must have sequential, non-duplicate node "
            "outputs.\nActual outputs:"
         << outputs_error_stream.str();
}

// Debug mode-only verification that output nodes are correctly ordered.
// If a traversal is expected to materialize a unique set of tensors
// {A, B, C, ...} then the outputs should be ordered as A0, A1, ... A(n-1), B0,
// B1, ... B(m-1), C0, C1, ... C(k-1), ... etc assuming that A.size() = n,
// B.size() = m, C.size() = k, etc.
absl::Status VerifyPerNodeOutputs(absl::Span<const DeviceBufferRef> outputs) {
  if (outputs.empty()) {
    return InvalidNodeOutputsError(outputs);
  }
  if (outputs.front().index() != 0) {
    return InvalidNodeOutputsError(outputs);
  }
  absl::flat_hash_set<const DeviceBufferList*> seen_nodes;
  seen_nodes.insert(outputs.front().device_buffer_list().get());
  auto node_expected_size = outputs.front().device_buffer_list()->size();

  for (int i = 1; i < outputs.size(); ++i) {
    if (outputs[i].device_buffer_list() ==
        outputs[i - 1].device_buffer_list()) {
      if (outputs[i].index() != outputs[i - 1].index() + 1) {
        return InvalidNodeOutputsError(outputs);
      }
    } else {
      if (outputs[i - 1].index() + 1 != node_expected_size) {
        return InvalidNodeOutputsError(outputs);
      }
      if (outputs[i].index() != 0) {
        return InvalidNodeOutputsError(outputs);
      }
      if (!seen_nodes.insert(outputs[i].device_buffer_list().get()).second) {
        return InvalidNodeOutputsError(outputs);
      }
      node_expected_size = outputs[i].device_buffer_list()->size();
    }
  }
  if (outputs.back().index() + 1 != node_expected_size) {
    return InvalidNodeOutputsError(outputs);
  }

  return absl::OkStatus();
}
#endif  // NDEBUG

// Propagates bounded dynamism annotations from one traversal to others
// when one traversal's output is bounded dynamic and is another traversal's
// input.
absl::Status PropagateBoundedDynamism(Traversal& traversal,
                                      mlir::MLIRContext& mlir_context) {
  if (!traversal.IsBoundedDynamic()) {
    return absl::OkStatus();
  }
  ABSL_VLOG(1) << "[PropagateBoundedDynamism] Traversal: "
               << traversal.DebugString();
  TT_ASSIGN_OR_RETURN(std::vector<DeviceRefDimensions> output_dimensions,
                      GetTraversalOutputDimensions(mlir_context, traversal));
  for (const auto& output_dimension : output_dimensions) {
    const DeviceBufferRef& ref = output_dimension.ref;
    const auto& dims = output_dimension.dims;
    for (int d = 0; d < dims.size(); ++d) {
      if (dims[d].boundOp.has_value() || dims[d].size > ref.dimensions()[d]) {
        TT_RETURN_IF_ERROR(ref.MarkDynamic(d, 2, dims[d].size));
        ABSL_VLOG(1) << "[PropagateBoundedDynamism] Marked dynamic: "
                     << ref.DebugString() << " dimension: " << d
                     << " upper bound: " << dims[d].size;
      }
    }
  }
  return absl::OkStatus();
}

// Returns nullptr if tracing is disabled. The returned event is only
// partially populated; FinalizePushTraceEvent must be called after Compile.
std::unique_ptr<StructuredLogEvent> MaybeStartTraceEvent(
    MaterializationReason reason, const Traversal& split_traversal) {
  if (!StructuredLogBuffer::GetInstance().enabled()) return nullptr;
  auto event = std::make_unique<StructuredLogEvent>();
  event->reason = reason;
  event->timestamp = absl::Now();
  std::string_view first_op;
  for (const SharedDeviceBufferList& node : split_traversal.execution_order()) {
    if (const auto op = node->deferred_op()) {
      first_op = torch_tpu::ToString(op->op_name());
      break;
    }
  }
  event->name = first_op.empty() ? "torchtpu_eager"
                                 : absl::StrCat("torchtpu_eager/", first_op);
  return event;
}

void FinalizePushTraceEvent(std::unique_ptr<StructuredLogEvent> event,
                            std::string captured_mlir, bool compile_ok,
                            std::string aten_graph_payload) {
  event->duration = absl::Now() - event->timestamp;
  event->cache_hit = captured_mlir.empty();
  if (!event->cache_hit || !compile_ok) {
    event->aten_graph_payload = std::move(aten_graph_payload);
    event->mlir_payload = std::move(captured_mlir);
  }
  event->compile_failed = !compile_ok;
  event->chromium_payload = absl::StrCat(
      R"({"name":")", absl::CEscape(event->name), R"(","ph":"X","ts":)",
      absl::ToUnixMicros(event->timestamp), R"(,"dur":)",
      std::max<int64_t>(0, absl::ToInt64Microseconds(event->duration)),
      R"(,"cat":"torchtpu_eager","pid":)", getpid(),
      R"(,"tid":0,"args":{"cache_hit":)", event->cache_hit ? "true" : "false",
      R"(,"compile_failed":)", event->compile_failed ? "true" : "false",
      R"(,"reason":")", absl::CEscape(ToString(event->reason)), R"("}})");
  StructuredLogBuffer::GetInstance().Push(std::move(event));
}

}  // namespace

absl::StatusOr<ExecutionTask> ExecutionTask::FromTraversal(
    absl_nonnull std::unique_ptr<Traversal> traversal,
    mlir::MLIRContext& mlir_context, MaterializationReason reason,
    std::string* absl_nullable out_mlir_text) {
  // Propagate bounded dynamism annotations if needed.
  if (traversal->IsBoundedDynamic()) {
    TT_RETURN_IF_ERROR(PropagateBoundedDynamism(*traversal, mlir_context));
  }

#ifndef NDEBUG
  // Check that the traversal has a valid set of outputs before we start
  // compiling.
  TT_RETURN_IF_ERROR(VerifyPerNodeOutputs(traversal->outputs()));
#endif  // NDEBUG

  // Start compiling the traversal.
  ABSL_VLOG(1) << "[ExecutionTask] Compiling traversal";
  auto compilation_mode = GetCompilationMode(GetEagerMode());
  absl::StatusOr<CompiledKernel> compiled_kernel;
  {
    tsl::profiler::TraceMe t("CompileTraversal");
    TT_ASSIGN_OR_RETURN(compiled_kernel,
                        traversal->Compile(compilation_mode, out_mlir_text));
  }

  // Mark all outputs of the split as scheduled/materialized.
  absl::flat_hash_set<const DeviceBufferList*> marked_materialized;
  for (const auto& output : traversal->outputs()) {
    if (!marked_materialized.insert(output.device_buffer_list().get()).second) {
      continue;
    }

    ABSL_VLOG(1) << "[ExecutionTask] Marking output as materialized: "
                 << output.device_buffer_list();

    output.device_buffer_list()->SetMaterializationPending();
  }

  ABSL_VLOG(1) << "[ExecutionTask] Marking all nodes as having been executed";

  // Mark all deferred ops in the split as having been executed (scheduled).
  for (const auto& node : traversal->execution_order()) {
    if (auto deferred_op = node->deferred_op()) {
      deferred_op->mark_executed();
    }
  }

  std::string task_name;
  if (ABSL_VLOG_IS_ON(1)) {
    task_name = absl::StrCat(traversal->GetCacheKey(compilation_mode));
  }
  Traversal::Parts traversal_parts = traversal->IntoParts();
  return ExecutionTask(
      std::move(task_name), std::move(traversal_parts.arguments),
      std::move(traversal_parts.outputs), std::move(*compiled_kernel), reason);
}

absl::StatusOr<ExecutionTask> ExecutionTask::FromExecutable(
    SharedLoadedExecutableWithMetadata executable,
    std::vector<DeviceBufferRef> arguments,
    std::vector<DeviceBufferRef> outputs, MaterializationReason reason,
    std::string_view task_name) {
#ifndef NDEBUG
  // Check that the outputs buffers are valid.
  TT_RETURN_IF_ERROR(VerifyPerNodeOutputs(outputs));
#endif  // NDEBUG

  for (const auto& output : outputs) {
    output.device_buffer_list()->SetMaterializationPending();
  }

  // Create a promise/future that is already done using the executable.
  LoadedExecutablePromise promise;
  CompiledKernel compiled_kernel{.fixed_shape_kernel = promise.get_future()};
  promise.set_value(std::move(executable));

  return ExecutionTask(std::string(task_name), std::move(arguments),
                       std::move(outputs), std::move(compiled_kernel), reason);
}

absl::StatusOr<ExecutionTask> ExecutionTask::FromTraversalWithLogging(
    absl_nonnull std::unique_ptr<Traversal> traversal,
    mlir::MLIRContext& mlir_context, MaterializationReason reason) {
  auto event = MaybeStartTraceEvent(reason, *traversal);
  if (!event) {
    return ExecutionTask::FromTraversal(std::move(traversal), mlir_context,
                                        reason, /*out_mlir_text=*/nullptr);
  }

  // Get the aten graph payload before moving the traversal.
  std::string aten_graph_payload = traversal->ReadableString(reason);

  // Try to build the execution task and capture the MLIR if possible.
  std::string captured_mlir;
  auto execution_task_or = ExecutionTask::FromTraversal(
      std::move(traversal), mlir_context, reason, &captured_mlir);

  // Whether or not the execution task was created successfully, finish the
  // trace event and return.
  FinalizePushTraceEvent(std::move(event), std::move(captured_mlir),
                         execution_task_or.ok(), std::move(aten_graph_payload));
  return execution_task_or;
}

void ExecutionTask::SetOutputNodesAsError(absl::Status status) {
  for (const auto& output : outputs_) {
    output.device_buffer_list()->SetAsError(status);
  }
}

namespace {

void LogCompilationWait(const CompiledKernel& compiled_kernel,
                        std::string_view task_name) {
  // Wait for the compilation to complete with a timeout.
  while (compiled_kernel.fixed_shape_kernel.wait_for(std::chrono::seconds(5)) ==
         std::future_status::timeout) {
    ABSL_VLOG(1) << "Still waiting for compilation of task_name=" << task_name;
  }
  if (compiled_kernel.dynamic_kernel_adapter.has_value()) {
    if (compiled_kernel.dynamic_kernel_adapter->preamble.valid()) {
      while (compiled_kernel.dynamic_kernel_adapter->preamble.wait_for(
                 std::chrono::seconds(5)) == std::future_status::timeout) {
        ABSL_VLOG(1) << "Still waiting for preamble compilation of task_name="
                     << task_name;
      }
    }
    if (compiled_kernel.dynamic_kernel_adapter->postamble.valid()) {
      while (compiled_kernel.dynamic_kernel_adapter->postamble.wait_for(
                 std::chrono::seconds(5)) == std::future_status::timeout) {
        ABSL_VLOG(1) << "Still waiting for postamble compilation of task_name="
                     << task_name;
      }
    }
  }
  ABSL_VLOG(1) << "[ExecutionTask::Run] Compilation complete for task_name="
               << task_name;
}

absl::StatusOr<PjRtBufferPointers> RunSequentialExecutables(
    absl::Span<const SharedLoadedExecutableWithMetadata> executables,
    std::vector<xla::PjRtBuffer*> argument_buffers,
    std::string_view task_name) {
  std::vector<PjRtBufferPointers> intermediate_results;
  for (auto& executable : executables) {
    TT_ASSIGN_OR_RETURN(PjRtBufferPointers results,
                        Execute(executable, std::move(argument_buffers)),
                        _.SetPrepend()
                            << "failed to enqueue execution for task_name="
                            << task_name << ": ");
    argument_buffers.clear();
    argument_buffers.reserve(results.size());
    for (const auto& result : results) {
      argument_buffers.push_back(result.get());
    }
    intermediate_results.push_back(std::move(results));
  }
  return std::move(intermediate_results.back());
}

}  // namespace

absl::StatusOr<std::vector<SharedLoadedExecutableWithMetadata>>
ExecutionTask::GetCachedExecutables(std::string_view task_name) {
  if (ABSL_VLOG_IS_ON(1)) {
    LogCompilationWait(compiled_kernel_, task_name);
  }

  std::vector<SharedLoadedExecutableWithMetadata> cached_executables;

  // Get the preamble if it exists.
  if (compiled_kernel_.dynamic_kernel_adapter.has_value()) {
    if (compiled_kernel_.dynamic_kernel_adapter->preamble.valid()) {
      absl::StatusOr<SharedLoadedExecutableWithMetadata> preamble =
          compiled_kernel_.dynamic_kernel_adapter->preamble.get();
      if (!preamble.ok()) {
        ABSL_VLOG(1) << "[ExecutionTask::Run] Failed to compile "
                        "task_name="
                     << task_name << ": " << preamble.status();
        return preamble.status();
      }
      cached_executables.push_back(std::move(*preamble));
    }
  }

  // Get the fixed shape kernel. This should always exist.
  absl::StatusOr<SharedLoadedExecutableWithMetadata> fixed_shape_kernel =
      compiled_kernel_.fixed_shape_kernel.get();
  if (!fixed_shape_kernel.ok()) {
    ABSL_VLOG(1) << "[ExecutionTask::Run] Failed to compile "
                    "task_name="
                 << task_name << ": " << fixed_shape_kernel.status();
    return fixed_shape_kernel.status();
  }
  cached_executables.push_back(std::move(*fixed_shape_kernel));

  // Get the postamble if it exists.
  if (compiled_kernel_.dynamic_kernel_adapter.has_value()) {
    if (compiled_kernel_.dynamic_kernel_adapter->postamble.valid()) {
      absl::StatusOr<SharedLoadedExecutableWithMetadata> postamble =
          compiled_kernel_.dynamic_kernel_adapter->postamble.get();
      if (!postamble.ok()) {
        ABSL_VLOG(1) << "[ExecutionTask::Run] Failed to compile "
                        "task_name="
                     << task_name << ": " << postamble.status();
        return postamble.status();
      }
      cached_executables.push_back(std::move(*postamble));
    }
  }
  return cached_executables;
}

absl::StatusOr<std::vector<xla::PjRtBuffer* absl_nullable>>
ExecutionTask::GetArgumentBuffers() {
  // Check that this task does not have any placeholder arguments.
  for (const auto& input : arguments_) {
    if (input.state() == DeviceBufferRefState::kPlaceholder) {
      return TT_ERROR(error::kInternal)
             << "Materialize was called on a placeholder tensor. This "
                "should never happen.\nkPlaceholder tensors should only "
                "appear in compiled mode, which should never try to "
                "materialize tensors."
             << input.DebugString();
    }
  }
  std::vector<xla::PjRtBuffer*> root_args;
  for (const auto&& [index, argument] : llvm::enumerate(arguments_)) {
    switch (argument.state()) {
      case DeviceBufferRefState::kMaterialized: {
        ABSL_VLOG(1) << "kMaterialized DeviceBufferRef index: " << index;
        TT_ASSIGN_OR_RETURN(xla::PjRtBuffer * pjrt_buffer,
                            argument.AwaitBuffer());
        root_args.push_back(pjrt_buffer);
        break;
      }
      case DeviceBufferRefState::kPlaceholder:
        return TT_ERROR(error::kInternal)
               << "Materialize was called on a placeholder tensor. This "
                  "should never happen.\nkPlaceholder tensors should only "
                  "appear in compiled mode, which should never try to "
                  "materialize tensors."
               << argument.DebugString();

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

absl::Status ExecutionTask::SetOutputNodesAsMaterialized(
    PjRtBufferPointers results) {
  tsl::profiler::TraceMe trace("SetOutputNodesAsMaterialized");
  ABSL_VLOG(1) << "[SetOutputNodesAsMaterialized] Starting";
  PjRtBufferPointers buffers_to_assign;
  buffers_to_assign.reserve(outputs_.size());
  auto i = 0;
  while (i < outputs_.size()) {
    ABSL_VLOG(1) << "[AssignMaterializedBuffers] Assigning materialized "
                    "buffers for node "
                 << i;
    auto* absl_nonnull node = outputs_[i].device_buffer_list().get();
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

absl::Status ExecutionTask::RunInternal() {
  tsl::profiler::TraceMe trace("ExecutionTask::Run");
  std::string_view task_name = "anonymous";
  if (!name_.empty()) {
    task_name = name_;
  }

  // Wait for compilation to complete.
  ABSL_VLOG(1) << "[ExecutionTask::Run] Waiting for compilation for "
                  "task_name="
               << task_name;
  TT_ASSIGN_OR_RETURN(
      std::vector<SharedLoadedExecutableWithMetadata> cached_executables,
      GetCachedExecutables(task_name));
  ABSL_VLOG(1) << "[ExecutionTask::Run] Cached executables size: "
               << cached_executables.size();

  // Wait for arguments to be materialized.
  TT_ASSIGN_OR_RETURN(std::vector<xla::PjRtBuffer*> argument_buffers,
                      GetArgumentBuffers());
  ABSL_VLOG(1) << "[ExecutionTask::Run] Retrieved " << argument_buffers.size()
               << " argument buffers for task_name=" << task_name;

  // Run all cached executables in sequence to get the PjRtBuffer pointers.
  ABSL_VLOG(1) << "[ExecutionTask::Run] Executing job for task_name="
               << task_name;
  TT_ASSIGN_OR_RETURN(
      PjRtBufferPointers final_results,
      RunSequentialExecutables(cached_executables, std::move(argument_buffers),
                               task_name));

  // Assign the final results to the output nodes.
  TT_RETURN_IF_ERROR(SetOutputNodesAsMaterialized(std::move(final_results)));

  ABSL_VLOG(1) << "[ExecutionTask::Run] ExecuteMaterializationJob "
                  "succeeded for task_name="
               << task_name;
  return absl::OkStatus();
}

absl::Status ExecutionTask::Run() {
  absl::Status status = RunInternal();
  if (!status.ok()) {
    SetOutputNodesAsError(status);
  }
  return status;
}

CompilationMode GetCompilationMode(EagerMode eager_mode) {
  switch (eager_mode) {
    case EagerMode::kInternalDeferAll:
    case EagerMode::kDeferAndFuse:
      return CompilationMode::kFastRuntime;
    case EagerMode::kDeferNever:
    case EagerMode::kDeferNeverAndLaunchBlocking:
      return CompilationMode::kFastCompile;
  };
}

}  // namespace torch_tpu
