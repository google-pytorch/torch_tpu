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

#include "csrc/eager/execution_task.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_log.h"
#include "absl/log/absl_vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dynamism_utils.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/layout_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/materialization_heuristics.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/traversal.h"
#include "csrc/ops/op_names.h"
#include "csrc/pjrt/pjrt_execute.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/MLIRContext.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/pjrt/pjrt_client.h"

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

absl::StatusOr<std::vector<CustomLayout>>
ExtractArgumentLayoutsIfDifferentFromDefault(const Traversal& traversal) {
  std::vector<CustomLayout> argument_layouts;
  const auto& arguments = traversal.arguments();
  for (size_t i = 0; i < arguments.size(); ++i) {
    TT_ASSIGN_OR_RETURN(
        std::optional<CustomLayout> custom_layout,
        ExtractCustomLayoutIfDifferentFromDefault(arguments[i].shape()));
    if (!custom_layout.has_value()) {
      continue;
    }
    if (argument_layouts.empty()) {
      argument_layouts.resize(arguments.size());
    }
    argument_layouts[i] = *std::move(custom_layout);
  }
  return argument_layouts;
}

struct CompiledTraversal {
  CompilationCacheKey compilation_cache_key;
  CompiledKernel compiled_kernel;
};

absl::StatusOr<CompiledTraversal> VerifyAndCompileTraversal(
    Traversal& traversal, mlir::MLIRContext& mlir_context,
    const CompilationSpec& compilation_spec,
    std::string* absl_nullable out_mlir_text) {
  // Propagate bounded dynamism annotations if needed.
  if (traversal.IsBoundedDynamic()) {
    TT_RETURN_IF_ERROR(PropagateBoundedDynamism(traversal, mlir_context));
  }

  for (const auto& argument : traversal.arguments()) {
    if (argument.is_placeholder()) {
      return TT_ERROR(error::kInternal)
             << "materialize was called on a placeholder tensor. This "
                "should never happen.\nkPlaceholder tensors should only "
                "appear in compiled mode, which should never try to "
                "materialize tensors.\n"
             << argument.DebugString();
    } else if (argument.is_executing()) {
      // Argument has had a PjRtBuffer already created for it, so we can
      // check immediately to see if it is in a failed state.
      auto buffer_or = argument.AwaitBuffer();
      if (!buffer_or.ok()) {
        // Don't prefix or suffix this here; if there is a chain of tasks and
        // one of them fails, anything appended here gets duplicated over and
        // over. Just let the original error flow down the chained tasks.
        return buffer_or.status();
      }
    } else if (argument.is_materializing()) {
      // Input has been pushed onto the execution thread, but has not yet been
      // assigned a PjRtBuffer. We will recheck it once this task reaches the
      // front of the execution queue (see GetArgumentBuffers).
      continue;
    } else if (const auto deferred_op = argument.deferred_op()) {
      // Surface the cache key only in the ABSL_LOG, not in the TT_ERROR.
      const CompilationCacheKey compilation_cache_key =
          traversal.GetCacheKey(compilation_spec.compile_options_key,
                                /*argument_layouts=*/{},
                                /*donated_inputs=*/{});
      ABSL_LOG(ERROR)
          << "traversal (cache key: " << compilation_cache_key
          << ") has deferred argument " << ToString(deferred_op->op_name())
          << ToString(argument.dimensions())
          << ".\nAll arguments must be set to pending materialization "
             "before creating chained ExecutionTasks";
      return TT_ERROR(error::kFailedPrecondition)
             << "detected an out of order execution; an execution task "
                "depends on an argument that is not scheduled for "
                "materialization.\nIf you are using multiple streams, please "
                "use torch.Events to coordinate cross-stream dependencies";
    } else {
      return TT_ERROR(error::kInternal)
             << "traversal argument has an unknown state (not deferred, "
                "placeholder, or materializing):\n"
             << argument.DebugString();
    }
  }

#ifndef NDEBUG
  // Check that the traversal has a valid set of outputs before we start
  // compiling.
  TT_RETURN_IF_ERROR(VerifyPerNodeOutputs(traversal.outputs()));
#endif  // NDEBUG

  TT_ASSIGN_OR_RETURN(const std::vector<CustomLayout> argument_layouts,
                      ExtractArgumentLayoutsIfDifferentFromDefault(traversal));

  // Start compiling the traversal.
  ABSL_VLOG(1) << "[ExecutionTask] Compiling traversal";
  CompiledKernel compiled_kernel;
  std::optional<CompilationCacheKey> compilation_cache_key;
  {
    tsl::profiler::TraceMe t("CompileTraversal");
    TT_ASSIGN_OR_RETURN(
        compiled_kernel,
        traversal.Compile(compilation_spec, out_mlir_text,
                          /*use_stablehlo_bounds=*/false, argument_layouts,
                          /*donated_inputs=*/{}, /*module_name=*/std::nullopt,
                          &compilation_cache_key));
  }
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Compile guarantees out_cache_key.
      compilation_cache_key.has_value(), error::kInternal)
      << "traversal compilation succeeded without populating cache key";
  return CompiledTraversal{
      .compilation_cache_key = std::move(*compilation_cache_key),
      .compiled_kernel = std::move(compiled_kernel)};
}

std::vector<Shape> GetShapesFromDeviceBuffers(
    absl::Span<const DeviceBufferRef> buffers) {
  std::vector<Shape> shapes;
  shapes.reserve(buffers.size());
  for (const auto& buffer : buffers) {
    // Only copy the immutable dimensions and element_type needed by Execute().
    shapes.push_back(
        Shape(CopyIntVector(buffer.dimensions()), buffer.element_type()));
  }
  return shapes;
}

}  // namespace

absl::StatusOr<ExecutionTask> ExecutionTask::FromTraversal(
    absl_nonnull std::unique_ptr<Traversal> traversal,
    mlir::MLIRContext& mlir_context, const CompilationSpec& compilation_spec,
    std::string* absl_nullable out_mlir_text) {
  auto compiled_traversal_or = VerifyAndCompileTraversal(
      *traversal, mlir_context, compilation_spec, out_mlir_text);
  if (!compiled_traversal_or.ok()) {
    for (const auto& output : traversal->outputs()) {
      output.device_buffer_list()->SetAsError(compiled_traversal_or.status());
    }
    return compiled_traversal_or.status();
  }

  // Mark all outputs of the split as scheduled/materialized.
  ABSL_VLOG(1) << "[ExecutionTask] Setting " << traversal->outputs().size()
               << " output buffers as pending materialization";
  for (const auto& output : traversal->outputs()) {
    TT_RETURN_IF_ERROR(
        output.device_buffer_list()->SetAsPendingMaterialization());
  }

  std::string task_name;
  if (ABSL_VLOG_IS_ON(1)) {
    task_name = absl::StrCat(compiled_traversal_or->compilation_cache_key);
  }
  const CorePinningMode core_pinning_mode = traversal->core_pinning_mode();
  Traversal::Parts traversal_parts = traversal->IntoParts();
  // Fixed-shape eager traversals without pinned-host buffers extract and key
  // compilation on the exact custom layouts of `arguments`, so runtime layout
  // validation is redundant. When any argument or output is on pinned host
  // memory (where ApplyPlacementsToCompileSpec clears custom device layouts on
  // the compile options) or a dynamic kernel adapter is used, retain runtime
  // layout validation.
  const auto is_pinned_host = [](const DeviceBufferRef& ref) {
    return ref.is_pinned_host();
  };
  const ArgumentLayoutValidation layout_validation =
      (compiled_traversal_or->compiled_kernel.dynamic_kernel_adapter
           .has_value() ||
       llvm::any_of(traversal_parts.arguments, is_pinned_host) ||
       llvm::any_of(traversal_parts.outputs, is_pinned_host))
          ? ArgumentLayoutValidation::kValidate
          : ArgumentLayoutValidation::kSkip;
  return ExecutionTask(std::move(task_name),
                       std::move(traversal_parts.arguments),
                       std::move(traversal_parts.outputs),
                       std::move(compiled_traversal_or->compiled_kernel),
                       layout_validation, core_pinning_mode);
}

absl::StatusOr<ExecutionTask> ExecutionTask::FromExecutable(
    SharedLoadedExecutableWithMetadata executable,
    std::vector<DeviceBufferRef> arguments,
    std::vector<DeviceBufferRef> outputs, std::string_view task_name) {
  for (const auto& argument : arguments) {
    if (!argument.is_materializing()) {
      if (argument.is_placeholder()) {
        return TT_ERROR(error::kInternal)
               << "ExecutionTask::FromExecutable was called on a placeholder "
                  "tensor. This should never happen.\nkPlaceholder tensors "
                  "should only appear at compile time, not at execution time.\n"
               << argument.DebugString();
      } else if (const auto deferred_op = argument.deferred_op()) {
        return TT_ERROR(error::kInternal)
               << "argument to task " << task_name << " has deferred input op"
               << ToString(deferred_op->op_name());
      } else {
        return TT_ERROR(error::kInternal)
               << "execution task argument has an unknown state (not deferred, "
                  "placeholder, or materializing):\n"
               << argument.DebugString();
      }
    }
  }
#ifndef NDEBUG
  // Check that the outputs buffers are valid.
  TT_RETURN_IF_ERROR(VerifyPerNodeOutputs(outputs));
#endif  // NDEBUG

  // Create a promise/future that is already done using the executable.
  LoadedExecutablePromise promise;
  CompiledKernel compiled_kernel{.fixed_shape_kernel = promise.get_future()};
  promise.set_value(std::move(executable));

  return ExecutionTask(std::string(task_name), std::move(arguments),
                       std::move(outputs), std::move(compiled_kernel));
}

absl::StatusOr<ExecutionTask> ExecutionTask::FromTraversalWithLogging(
    absl_nonnull std::unique_ptr<Traversal> traversal,
    mlir::MLIRContext& mlir_context, const CompilationSpec& compilation_spec,
    MaterializationReason reason) {
  auto event = MaybeStartTraceEvent(reason, *traversal);
  if (!event) {
    return ExecutionTask::FromTraversal(std::move(traversal), mlir_context,
                                        compilation_spec,
                                        /*out_mlir_text=*/nullptr);
  }

  // Get the aten graph payload before moving the traversal.
  std::string aten_graph_payload = traversal->ReadableString(reason);

  // Try to build the execution task and capture the MLIR if possible.
  std::string captured_mlir;
  auto execution_task_or = ExecutionTask::FromTraversal(
      std::move(traversal), mlir_context, compilation_spec, &captured_mlir);

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
    absl::Span<const Shape> logical_output_shapes, std::string_view task_name,
    ArgumentLayoutValidation layout_validation) {
  std::vector<PjRtBufferPointers> intermediate_results;
  for (auto& executable : executables) {
    TT_ASSIGN_OR_RETURN(PjRtBufferPointers results,
                        Execute(executable, std::move(argument_buffers),
                                logical_output_shapes, layout_validation),
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

// Returns the appropriate CompilationSpec for a split traversal.
// If the global compilation mode is already FastRuntime (O2), we retain it.
// Otherwise, if any deferred op in the traversal is an eligible convolution
// that benefits from O2, we selectively promote it to FastRuntime.
const CompilationSpec& GetCompilationSpecForTraversal(
    const Traversal& traversal, const CompilationContext& compilation_context,
    const CompilationMode compilation_mode) {
  const auto& compilation_specs = compilation_context.compilation_specs();
  if (compilation_mode != CompilationMode::kFastCompile) {
    // Either we're already using O2, or the O2 spec is not applicable.
    return compilation_specs.at(compilation_mode);
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

  return compilation_specs.at(compilation_mode);
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
  std::vector<xla::PjRtBuffer*> root_args;
  root_args.reserve(arguments_.size());
  for (const auto&& [index, argument] : llvm::enumerate(arguments_)) {
    if (argument.is_executing() || argument.is_transferring()) {
      // If is_executing() is true, then this will not block as the PjRtBuffer
      // already exists. If is_transferring() is true, then this will block
      // until the transfer has started and provides a PjRtBuffer.
      ABSL_VLOG(1) << "Get PjRtBuffer for DeviceBufferRef index: " << index;
      TT_ASSIGN_OR_RETURN(xla::PjRtBuffer * pjrt_buffer,
                          argument.AwaitBuffer());
      root_args.push_back(pjrt_buffer);
    } else if (argument.is_materializing()) {
      return TT_ERROR(error::kFailedPrecondition)
             << "detected an out of order execution; an earlier execution task "
                "depends on an argument which will be materialized by a later "
                "execution task.\nIf you are using multiple streams, please "
                "use torch.Events to coordinate cross-stream dependencies";
    } else if (argument.is_placeholder()) {
      // This should already have been checked in MaterializeImpl, but we can
      // return the same error here if that check was somehow bypassed.
      return TT_ERROR(error::kInternal)
             << "cannot Materialize() a placeholder tensor or a tensor that "
                "depends on a placeholder. \nPlaceholder tensors should only "
                "appear in compiled mode, which should never try to "
                "materialize tensors."
             << argument.DebugString();
    } else {
      return TT_ERROR(error::kFailedPrecondition)
             << "detected an out of order execution; an execution task "
                "depends on an argument is not scheduled for "
                "materialization.\nIf you are using multiple streams, please "
                "use torch.Events to coordinate cross-stream dependencies";
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
  const DeviceBufferRef& output = outputs_.front();

  // Wait for compilation to complete.
  ABSL_VLOG(1) << "[ExecutionTask::Run] Waiting for compilation for "
                  "task_name="
               << task_name;
  TT_ASSIGN_OR_RETURN(
      std::vector<SharedLoadedExecutableWithMetadata> cached_executables,
      GetCachedExecutables(task_name),
      TranslateXlaTensorOomError(_, ToString(output.element_type()),
                                 output.dimensions()));
  ABSL_VLOG(1) << "[ExecutionTask::Run] Cached executables size: "
               << cached_executables.size();

  // Wait for arguments to be materialized.
  TT_ASSIGN_OR_RETURN(std::vector<xla::PjRtBuffer*> argument_buffers,
                      GetArgumentBuffers());
  ABSL_VLOG(1) << "[ExecutionTask::Run] Retrieved " << argument_buffers.size()
               << " argument buffers for task_name=" << task_name;

  std::vector<Shape> output_shapes;
  if (llvm::any_of(
          cached_executables,
          [](const SharedLoadedExecutableWithMetadata& exe) {
            return exe->unbounded_dynamic_execution_metadata().has_value();
          })) {
    output_shapes = GetShapesFromDeviceBuffers(outputs_);
  }

  // Run all cached executables in sequence to get the PjRtBuffer pointers.
  ABSL_VLOG(1) << "[ExecutionTask::Run] Executing job for task_name="
               << task_name;
  TT_ASSIGN_OR_RETURN(
      PjRtBufferPointers final_results,
      RunSequentialExecutables(cached_executables, std::move(argument_buffers),
                               output_shapes, task_name, layout_validation_),
      TranslateXlaTensorOomError(_, ToString(output.element_type()),
                                 output.dimensions()));

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

namespace {

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

}  // namespace

MakeExecutionTasksResult MakeExecutionTasks(
    std::vector<absl_nonnull std::unique_ptr<Traversal>>&& traversals,
    mlir::MLIRContext& mlir_context,
    const CompilationContext& compilation_context,
    CompilationMode compilation_mode, MaterializationReason reason) {
  MakeExecutionTasksResult result;
  traversals = FilterPlaceholderDependentTraversals(std::move(traversals),
                                                    result.first_error);

  result.execution_tasks.reserve(traversals.size());
  for (auto& split_traversal : traversals) {
    const CompilationSpec& compilation_spec = GetCompilationSpecForTraversal(
        *split_traversal, compilation_context, compilation_mode);
    auto execution_task_or = ExecutionTask::FromTraversalWithLogging(
        std::move(split_traversal), mlir_context, compilation_spec, reason);
    if (execution_task_or.ok()) {
      result.execution_tasks.push_back(std::move(*execution_task_or));
    } else if (result.first_error.ok()) {
      result.first_error = execution_task_or.status();
    }
  }
  return result;
}

}  // namespace torch_tpu
