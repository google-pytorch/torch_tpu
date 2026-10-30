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

#include "csrc/internal/compile/compiled_mode.h"

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11): `std::shared_future::wait_for`.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>  // NOLINT(build/c++11): the executable future type is a
                   // `std::shared_future`.
#include <iterator>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/ops/clone.h"
#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorImpl.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/to_string.h"
#include "csrc/common/tpu_subslice_utils.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/materialize.h"
#include "csrc/eager/split_traversal.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/eager/traversal.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "csrc/ops/view_decomposition/decomposition.h"
#include "csrc/ops/view_decomposition/strided_layout.h"
#include "csrc/pjrt/pjrt_state.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/shape.h"

namespace torch_tpu {

namespace {

inline const at::Tensor& DerefTensor(const at::Tensor& t) { return t; }
inline const at::Tensor& DerefTensor(const at::Tensor* t) { return *t; }

// Checks whether a base DeviceBufferRef requires explicit materialization
// before being passed to a compiled PJRT executable.
//
// In steady-state compiled execution (and when inputs come from previous
// compiled steps or materialized parameters), `node->is_deferred()` is false.
// Tracking `has_deferred_or_placeholder` during argument preparation allows us
// to bypass `Materialize(argument_buffer_refs)` completely when all inputs are
// already materialized/pending on the device stream.
bool NeedsMaterialize(const DeviceBufferRef& buffer_ref) {
  const auto& node = buffer_ref.device_buffer_list();
  return node != nullptr && (node->is_deferred() || node->is_placeholder());
}

// To call a compiled PjRtExecutable, we need to get materialized base buffers
// for each argument tensor, in the same shape expected by the executable.
// This **must** match the logic in tpu_torch_compile's PyMakePlaceholderLike.
template <typename TensorSpan>
absl::StatusOr<std::vector<DeviceBufferRef>> PrepareCompiledModeArgumentsImpl(
    TensorSpan argument_tensors) {
  // Get each argument buffer ref's expected contiguous base shape.
  std::vector<DeviceBufferRef> argument_buffer_refs;
  argument_buffer_refs.reserve(argument_tensors.size());
  bool has_deferred_or_placeholder = false;
  for (const auto& item : argument_tensors) {
    const at::Tensor& argument_tensor = DerefTensor(item);
    // If argument_tensor is contiguous with offset 0, then we know that
    // base_sizes and base_stride below would be the same as
    // `argument_tensor`, and skipping GetContiguousBaseShape saves some
    // time.
    if (TensorHasTrivialLayout(argument_tensor)) {
      TT_ASSIGN_OR_RETURN(
          DeviceBufferRef buffer_ref, GetBuffer(argument_tensor),
          _.SetPrepend() << "failed to get buffer from argument tensor: "
                         << ToString(argument_tensor));
      if (NeedsMaterialize(buffer_ref)) {
        has_deferred_or_placeholder = true;
      }
      argument_buffer_refs.push_back(std::move(buffer_ref));
      continue;
    }

    // Using as_strided, restrict the base tensor to only the minimal contiguous
    // block of data needed for the view to be valid.
    TT_ASSIGN_OR_RETURN(
        Dimensions base_sizes,
        GetContiguousBaseShape(StridedLayout::FromTensor(argument_tensor)));
    Strides base_strides = GetStrides(MakeContiguousBaseLayout(base_sizes));

    at::Tensor base_tensor =
        argument_tensor.as_strided(base_sizes, base_strides,
                                   /*storage_offset=*/0);
    TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer_ref, GetBuffer(base_tensor),
                        _.SetPrepend()
                            << "failed to get buffer from argument tensor: "
                            << ToString(argument_tensor));
    if (NeedsMaterialize(buffer_ref)) {
      has_deferred_or_placeholder = true;
    }
    argument_buffer_refs.push_back(std::move(buffer_ref));
  }

  // Materialize the argument buffers if any are still deferred.
  // This has to block until all buffers start materialization, but not until
  // they complete, so that the validation in ExecutionTask creation passes.
  if (has_deferred_or_placeholder) {
    TT_RETURN_IF_ERROR(Materialize(
        argument_buffer_refs, MaterializationReason::kCompileModeExecution,
        GraphSplitMode::kSplitGraph,
        MaterializationBlockMode::kBlockUntilStarted));
  }

  // After materialization, each argument buffer will be a materialized,
  // contiguous tensor which the compiled executable can safely apply view
  // logic to.
  return argument_buffer_refs;
}

// Deduplicated Path: If we have argument_indices, populate only the unique
// base buffers expected by the executable.
//
// Performance Rationale:
//   - During FX graph compilation, `argument_indices` assigns a 0-based index
//     (`0, 1, 2, ...`) to each unique underlying storage buffer sequentially
//     upon its first appearance in the argument list. Aliased views or
//     duplicate arguments reuse the existing `base_idx <
//     argument_buffer_refs.size()`.
//   - Fast Path (`base_idx == argument_buffer_refs.size()`): We populate
//     `argument_buffer_refs` in a single pass with zero `std::optional`
//     allocations and skip duplicate `GetBaseBuffer` lookups whenever
//     `base_idx < argument_buffer_refs.size()`.
//   - Fallback (`base_idx > argument_buffer_refs.size()`): If indices are ever
//     non-sequential, we safely fall back to the indexed `std::optional` table.
template <typename TensorSpan>
absl::StatusOr<std::vector<DeviceBufferRef>>
PrepareCompiledModeArgumentsNonSequentialFallback(
    TensorSpan argument_tensors, const Indices& argument_indices,
    bool* has_deferred_or_placeholder) {
  int64_t num_base =
      *std::max_element(argument_indices.begin(), argument_indices.end()) + 1;
  std::vector<std::optional<DeviceBufferRef>> base_buffer_refs(num_base);
  for (size_t j = 0; j < argument_tensors.size(); ++j) {
    int64_t idx = argument_indices[j];
    if (!base_buffer_refs[idx].has_value()) {
      const at::Tensor& tensor = DerefTensor(argument_tensors[j]);
      TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer_ref, GetBaseBuffer(tensor),
                          _.SetPrepend()
                              << "failed to get buffer from argument tensor: "
                              << ToString(tensor));
      base_buffer_refs[idx] = std::move(buffer_ref);
    }
  }
  std::vector<DeviceBufferRef> argument_buffer_refs;
  argument_buffer_refs.reserve(num_base);
  for (int64_t j = 0; j < num_base; ++j) {
    ABSL_CHECK(base_buffer_refs[j].has_value())  // CRASH_OK
        << "base argument " << j << " was not populated";
    if (NeedsMaterialize(*base_buffer_refs[j])) {
      *has_deferred_or_placeholder = true;
    }
    argument_buffer_refs.push_back(std::move(*base_buffer_refs[j]));
  }
  return argument_buffer_refs;
}

template <typename TensorSpan>
absl::StatusOr<std::vector<DeviceBufferRef>> PrepareCompiledModeArgumentsImpl(
    TensorSpan argument_tensors, const Indices& argument_indices) {
  if (argument_indices.empty()) {
    return PrepareCompiledModeArgumentsImpl(argument_tensors);
  }

  TT_RET_CHECK(argument_tensors.size() == argument_indices.size(),
               error::kInvalidArgument)
      << "number of argument tensors (" << argument_tensors.size()
      << ") does not match compiled argument indices ("
      << argument_indices.size() << ")";

  std::vector<DeviceBufferRef> argument_buffer_refs;
  argument_buffer_refs.reserve(argument_tensors.size());
  bool has_deferred_or_placeholder = false;
  for (size_t i = 0; i < argument_tensors.size(); ++i) {
    const int64_t base_idx = argument_indices[i];
    if (base_idx == static_cast<int64_t>(argument_buffer_refs.size())) {
      const at::Tensor& tensor = DerefTensor(argument_tensors[i]);
      TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer_ref, GetBaseBuffer(tensor),
                          _.SetPrepend()
                              << "failed to get buffer from argument tensor: "
                              << ToString(tensor));
      if (NeedsMaterialize(buffer_ref)) {
        has_deferred_or_placeholder = true;
      }
      argument_buffer_refs.push_back(std::move(buffer_ref));
    } else if (base_idx > static_cast<int64_t>(argument_buffer_refs.size())) {
      // Fallback if argument_indices are not sequentially assigned on first
      // appearance.
      TT_ASSIGN_OR_RETURN(argument_buffer_refs,
                          PrepareCompiledModeArgumentsNonSequentialFallback(
                              argument_tensors, argument_indices,
                              &has_deferred_or_placeholder));
      break;
    }
  }

  // Materialize the argument buffers if any are still deferred.
  // This has to block until all buffers start materialization, but not until
  // they complete, so that the validation in ExecutionTask creation passes.
  if (has_deferred_or_placeholder) {
    TT_RETURN_IF_ERROR(Materialize(
        argument_buffer_refs, MaterializationReason::kCompileModeExecution,
        GraphSplitMode::kSplitGraph,
        MaterializationBlockMode::kBlockUntilStarted));
  }
  return argument_buffer_refs;
}

}  // namespace

absl::StatusOr<at::Tensor> MakePlaceholder(absl::Span<const int64_t> sizes,
                                           at::ScalarType dtype,
                                           bool requires_grad,
                                           bool is_pinned_host) {
  TT_ASSIGN_OR_RETURN(mlir::ElementType tensor_element_type,
                      ConvertTo<mlir::ElementType>(dtype));
  Dimensions physical_sizes(sizes.begin(), sizes.end());
  if (dtype == at::kFloat4_e2m1fn_x2) {
    TT_RET_CHECK(!physical_sizes.empty(), error::kInvalidArgument)
        << "expected float4_e2m1fn_x2 tensors to be at least 1-dimensional";
    physical_sizes.back() *= 2;
  }
  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef placeholder,
      DeviceBufferList::CreatePlaceholder(std::move(physical_sizes),
                                          tensor_element_type, is_pinned_host));
  auto new_tensor = MakeTensor(std::move(placeholder));
  new_tensor.set_requires_grad(requires_grad);
  return new_tensor;
}

absl::StatusOr<at::Tensor> MakePlaceholder(Shape shape, bool requires_grad,
                                           bool is_pinned_host) {
  TT_ASSIGN_OR_RETURN(DeviceBufferRef placeholder,
                      DeviceBufferList::CreatePlaceholder(
                          shape.dimensions(), shape.dtype(), is_pinned_host));
  for (const auto& dynamic_dim : shape.dynamic_dimensions()) {
    TT_RETURN_IF_ERROR(placeholder.MarkDynamic(dynamic_dim.dimension,
                                               dynamic_dim.lower_bound,
                                               dynamic_dim.upper_bound));
  }
  auto new_tensor = MakeTensor(std::move(placeholder));
  new_tensor.set_requires_grad(requires_grad);
  return new_tensor;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ExtractMlirFromGraph(
    mlir::MLIRContext& mlir_context, const std::vector<at::Tensor>& arg_tensors,
    const std::vector<at::Tensor>& result_tensors, bool use_stablehlo_bounds) {
  // Use artificial python context for compiled mode to that modules have better
  // names when dumped. Currently this will always point to `_export_to_fx`, and
  // can be improved in the future, but it is useful for distinguishing torch
  // compiled graphs from eager graphs.
  ScopedPythonContextCapturer capturer(OpName::kCompileMlir);
  ScopedPythonContextProvider provider(
      ScopedPythonContextCapturer::GetContext());

  TT_RET_CHECK(!result_tensors.empty(), error::kInvalidArgument)
      << "no result tensors provided";

  std::vector<DeviceBufferRef> argument_refs;
  argument_refs.reserve(arg_tensors.size());
  for (const at::Tensor& tensor : arg_tensors) {
    // Get the base buffer, not the view; view tensors will always have
    // deferred ops.
    TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer_ref, GetBaseBuffer(tensor),
                        _.SetPrepend()
                            << "failed to get buffer from argument tensor: "
                            << ToString(tensor));
    TT_RET_CHECK(!buffer_ref.is_deferred(), error::kInternal)
        << "argument tensor has deferred ops: " << ToString(tensor);
    ABSL_VLOG(3) << "[ExtractMlirFromGraph] arg_tensor: "
                 << buffer_ref.DebugString();
    argument_refs.push_back(std::move(buffer_ref));
  }

  std::vector<DeviceBufferRef> result_refs;
  result_refs.reserve(result_tensors.size());
  for (const at::Tensor& tensor : result_tensors) {
    TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer_ref, GetBuffer(tensor),
                        _.SetPrepend()
                            << "failed to get buffer from result tensor: "
                            << ToString(tensor));
    TT_RET_CHECK(!buffer_ref.is_materializing(), error::kInternal)
        << "result tensor is already materializing: " << ToString(tensor);
    ABSL_VLOG(3) << "[ExtractMlirFromGraph] result_tensor: "
                 << buffer_ref.DebugString();
    result_refs.push_back(std::move(buffer_ref));
  }

  TT_ASSIGN_OR_RETURN(auto traversal, Traversal::Create(std::move(result_refs)),
                      _.SetPrepend() << "failed to traverse graph: ");
  traversal->SortByCreationOrder();
  TT_RETURN_IF_ERROR(
      traversal->ValidateAndReorderArguments(std::move(argument_refs)))
          .SetPrepend()
      << "failed to validate and reorder inputs: ";
  TT_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
      traversal->BuildMlirModule(mlir_context, use_stablehlo_bounds));

  return mlir_module;
}

absl::StatusOr<AsyncCompileResult> TraverseAndCompile(
    const std::vector<at::Tensor>& result_tensors,
    const std::vector<at::Tensor>& argument_tensors,
    const TraverseAndCompileOptions& options) {
  tsl::profiler::TraceMe trace_await("TraverseAndCompile");

  ABSL_CHECK(  // CRASH_OK=implies a bug in compile backend if this happens
      !result_tensors.empty())
      << "no result tensors provided";

  std::vector<DeviceBufferRef> argument_refs;
  Indices argument_indices;
  argument_refs.reserve(argument_tensors.size());
  argument_indices.reserve(argument_tensors.size());
  for (const at::Tensor& tensor : argument_tensors) {
    // Get the base buffer, not the view; view tensors will always have
    // deferred ops.
    TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer_ref, GetBaseBuffer(tensor),
                        _.SetPrepend()
                            << "failed to get buffer from argument tensor: "
                            << ToString(tensor));
    ABSL_CHECK(  // CRASH_OK=implies a bug in compile backend if this happens
        !buffer_ref.is_deferred())
        << "argument tensor has deferred ops: " << ToString(tensor);

    // Using a vector scan instead of a hashset for deduplication
    // as `argument_tensors` is expected to have few elements (<100)
    std::vector<DeviceBufferRef>::const_iterator it =
        absl::c_find(argument_refs, buffer_ref);
    if (it == argument_refs.cend()) {
      argument_indices.push_back(argument_refs.size());
      argument_refs.push_back(std::move(buffer_ref));
    } else {
      argument_indices.push_back(std::distance(argument_refs.cbegin(), it));
    }
  }

  bool has_duplicates = argument_refs.size() < argument_tensors.size();

  std::vector<DeviceBufferRef> result_refs;
  result_refs.reserve(result_tensors.size());
  for (const at::Tensor& tensor : result_tensors) {
    TT_ASSIGN_OR_CRASH(  // CRASH_OK=implies a bug in compile backend if this
                         // happens
        auto buffer_ref, GetBuffer(tensor),
        _ << "failed to get device buffer from result tensor: "
          << ToString(tensor));
    ABSL_CHECK(  // CRASH_OK=implies a bug in compile backend if this happens
        !buffer_ref.is_materializing())
        << "result tensor is already materializing: " << ToString(tensor);
    result_refs.push_back(std::move(buffer_ref));
  }

  TT_ASSIGN_OR_CRASH(  // CRASH_OK=implies a bug in compile backend if this
                       // happens
      auto traversal, Traversal::Create(std::move(result_refs)),
      _ << "failed to create traversal");
  traversal->SortByCreationOrder();

  ABSL_CHECK_OK(  // CRASH_OK=implies a bug in compile backend if this happens
      traversal->ValidateAndReorderArguments(std::move(argument_refs)))
      << "failed to validate and reorder traversal inputs";

  // Re-map donated inputs to deduplicated unique base arguments.
  Indices unique_donated_inputs = options.donated_inputs;
  if (has_duplicates && !options.donated_inputs.empty()) {
    for (int64_t& donated_idx : unique_donated_inputs) {
      if (donated_idx >= 0 && donated_idx < argument_indices.size()) {
        donated_idx = argument_indices[donated_idx];
      }
    }
    std::sort(unique_donated_inputs.begin(), unique_donated_inputs.end());
    unique_donated_inputs.erase(
        std::unique(unique_donated_inputs.begin(), unique_donated_inputs.end()),
        unique_donated_inputs.end());
  }

  // Map argument layouts onto the deduplicated base buffers. A base buffer
  // takes its layout from an argument that covers the whole buffer unchanged:
  // same sizes, contiguous, and zero storage offset. Layouts of partial or
  // permuted views, such as slices and transposes, describe the view rather
  // than the base buffer, so they are ignored. If two such arguments request
  // different layouts, the base buffer gets no layout.
  std::vector<CustomLayout> base_argument_layouts;
  if (has_duplicates && !options.argument_layouts.empty()) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Checked by caller.
        options.argument_layouts.size() == argument_tensors.size(),
        error::kInvalidArgument)
        << "expected one layout per argument, got "
        << options.argument_layouts.size() << " layouts for "
        << argument_tensors.size() << " arguments";
    absl::Span<const DeviceBufferRef> bases = traversal->arguments();
    base_argument_layouts.assign(bases.size(), CustomLayout{});
    Indices conflicting_bases;
    for (int64_t i = 0; i < argument_tensors.size(); ++i) {
      const at::Tensor& tensor = argument_tensors[i];
      const CustomLayout& layout = options.argument_layouts[i];
      const int64_t base_idx = argument_indices[i];
      if (layout.minor_to_major.empty() || tensor.storage_offset() != 0 ||
          !tensor.is_contiguous() ||
          !absl::c_equal(tensor.sizes(), bases[base_idx].dimensions()) ||
          absl::c_linear_search(conflicting_bases, base_idx)) {
        continue;
      }
      CustomLayout& base_layout = base_argument_layouts[base_idx];
      if (!base_layout.minor_to_major.empty() && !(base_layout == layout)) {
        ABSL_LOG(WARNING) << "Ignoring conflicting layouts for arguments "
                             "sharing base buffer "
                          << base_idx;
        base_layout = CustomLayout{};
        conflicting_bases.push_back(base_idx);
        continue;
      }
      base_layout = layout;
    }
  } else {
    base_argument_layouts = options.argument_layouts;
  }

  const CompilationSpec compilation_spec =
      GetCompilationSpec(options.compilation_mode);
  TT_ASSIGN_OR_RETURN(
      auto compiled_kernel,
      traversal->Compile(compilation_spec, nullptr,
                         options.use_stablehlo_bounds, base_argument_layouts,
                         unique_donated_inputs, options.module_name),
      _.SetPrepend() << "failed to compile traversal: ");

  std::shared_ptr<ContextedModule> module = nullptr;
  if (options.build_mlir_module) {
    ABSL_VLOG(1) << "Building MLIR module as requested.";

    TT_ASSIGN_OR_CRASH(  // CRASH_OK=implies a bug in compile backend if this
                         // happens
        auto contexted_module,
        ContextedModule::Make(
            [&]  // REF_CAPTURE_OK=called synchronously
            (mlir::MLIRContext& mlir_context)
                -> absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> {
              return traversal->BuildMlirModule(
                  mlir_context, options.use_stablehlo_bounds,
                  unique_donated_inputs, options.module_name);
            }),
        _ << "failed to build MLIR module");
    // BuildMlirModule() already resolved (and cached) the device assignment.
    TT_ASSIGN_OR_RETURN(std::vector<int64_t> device_assignment,  // INT_VEC_OK
                        traversal->ResolveDeviceAssignment());
    contexted_module.set_device_assignment(std::move(device_assignment));
    module = std::make_shared<ContextedModule>(std::move(contexted_module));
  }

  return AsyncCompileResult(
      std::move(module), std::move(compiled_kernel.fixed_shape_kernel),
      has_duplicates ? std::move(argument_indices) : Indices{});
}

AsyncCompileResult::AsyncCompileResult(
    absl_nullable std::shared_ptr<ContextedModule> module,
    SharedLoadedExecutableWithMetadataFuture executable_future,
    Indices argument_indices)
    : module_(std::move(module)),
      executable_future_(std::move(executable_future)),
      argument_indices_(std::move(argument_indices)) {
  ABSL_DCHECK(executable_future_.valid());
}

bool AsyncCompileResult::IsReady() const {
  ABSL_DCHECK(executable_future_.valid());
  return executable_future_.wait_for(std::chrono::seconds(0)) ==
         std::future_status::ready;
}

absl::StatusOr<SharedLoadedExecutableWithMetadata> AsyncCompileResult::Get() {
  ABSL_DCHECK(once_ != nullptr);
  // What we're really lacking here is the ability to chain continuations to the
  // existing `executable_future_` which is simply a deficiency with the
  // `<future>` implementation in the current version of the c++ library used
  // as of this writing.
  //
  // The below acts as a poor-man's solution to this problem in that we want to
  // apply our deduping argument indices to the generated executable after
  // fulfillment but before returning to the user.
  absl::call_once(*once_, [this]() {
    ABSL_DCHECK(executable_future_.valid());
    absl::StatusOr<SharedLoadedExecutableWithMetadata> status_or =
        executable_future_.get();
    if (status_or.ok() && !argument_indices_.empty()) {
      resolved_executable_ = LoadedExecutableWithMetadata::WithArgumentIndices(
          *status_or, std::move(argument_indices_));
    } else {
      resolved_executable_ = std::move(status_or);
    }
  });
  return *resolved_executable_;
}

absl::StatusOr<SharedLoadedExecutableWithMetadata> CompileMlirExecutable(
    const std::string_view mlir_module_bytecode,
    UniqueCompileOptions compile_options) {
  TT_ASSIGN_OR_RETURN(
      ContextedModule module,
      ContextedModule::Make(
          [&]  // REF_CAPTURE_OK=called synchronously
          (mlir::MLIRContext& mlir_context)
              -> absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> {
            return mlir::stablehlo::deserializePortableArtifact(
                {mlir_module_bytecode.data(), mlir_module_bytecode.size()},
                &mlir_context);
          }));
  return CompileMlirExecutable(std::move(module).ToMaybeOwningMlirModule(),
                               std::move(compile_options));
}

absl::StatusOr<SharedLoadedExecutableWithMetadata> CompileMlirExecutable(
    xla::MaybeOwningMlirModule module, UniqueCompileOptions compile_options) {
  TT_ASSIGN_OR_RETURN(xla::PjRtClient* const client,
                      PjrtBackend::GetInstance().GetClient());

  LoadedExecutableBuilder executable_builder =
      [module = std::move(module)](
          xla::PjRtClient& client,
          UniqueCompileOptions compile_options) mutable {
        return CompileAndLoadMlirModule(client, std::move(module),
                                        std::move(*compile_options));
      };
  return Compile(*client, std::move(executable_builder),
                 std::move(compile_options));
}

namespace {

// Returns the output shapes for executing the given executable.
// Args:
//   executable: The loaded executable with metadata containing pjrt-inferred
//      output shapes.
//   runtime_output_shapes: If non-empty overrides the pjrt-inferred output
//   shapes. This is used for dynamic output buffers.
// Returns:
//   The output shapes for executing the given executable, or an error if the
//   `runtime_output_shapes` are invalid.
//   `runtime_output_shapes` are invalid if either of the following is true:
//      - Don't match the expected number of pjrt-inferred output tensors.
//      - Have a different number of dimensions than the corresponding
//      pjrt-inferred output shapes.
//      - Have a dimension exceeding the corresponding dimension in the
//      pjrt-inferred output shape.
absl::StatusOr<std::vector<Shape>> GetOutputShapes(
    const SharedLoadedExecutableWithMetadata& executable,
    absl::Span<const OutputShape> output_shapes) {
  const std::vector<Shape>& inferred_shapes = executable->output_shapes();
  if (output_shapes.empty()) {
    return inferred_shapes;
  }

  TT_RET_CHECK(output_shapes.size() == inferred_shapes.size(),
               error::kInvalidArgument)
      << "expected output shapes to be specified for all outputs or none, "
      << "got " << output_shapes.size() << " output shapes for "
      << inferred_shapes.size() << " output tensors";

  std::vector<Shape> result_shapes;
  result_shapes.reserve(inferred_shapes.size());

  for (size_t i = 0; i < inferred_shapes.size(); ++i) {
    Shape result_shape = inferred_shapes[i];
    const OutputShape& shape_spec = output_shapes[i];
    const auto& dimensions = shape_spec.dimensions;

    if (!dimensions.empty()) {
      TT_RET_CHECK(dimensions.size() == result_shape.dimensions().size(),
                   error::kInvalidArgument)
          << "expected output shape number of dimensions to match the "
             "statically inferred dimensions, got output shape dimensions "
          << dimensions.size() << " and inferred dimensions "
          << result_shape.dimensions().size() << " for output tensor " << i;

      for (size_t j = 0; j < dimensions.size(); ++j) {
        int64_t result_dim = result_shape.dimensions()[j];
        TT_RET_CHECK(dimensions[j] <= result_dim ||
                         result_dim == xla::Shape::kUnboundedSize,
                     error::kInvalidArgument)
            << "expected output shape dimension not to exceed the statically "
               "inferred bound, got output shape "
            << ToString(dimensions) << " and inferred shape "
            << ToString(result_shape.dimensions());
      }

      result_shape.dimensions().assign(dimensions.begin(), dimensions.end());
    }

    result_shape.set_on_device_shape_is_dynamic(shape_spec.is_dynamic);
    result_shapes.push_back(std::move(result_shape));
  }

  return result_shapes;
}

template <typename TensorSpan>
std::vector<at::Tensor> ExecuteCompiledModelImpl(
    const SharedLoadedExecutableWithMetadata& executable,
    TensorSpan argument_tensors, absl::Span<const OutputShape> output_shapes) {
  // Fast path: for static shapes (output_shapes is empty), the executable
  // already owns the precomputed output shapes. Borrowing an absl::Span avoids
  // allocating and deep-copying a std::vector<Shape> on every single step.
  absl::Span<const Shape> result_shapes;
  std::vector<Shape> dynamic_result_shapes;
  if (output_shapes.empty()) {
    result_shapes = executable->output_shapes();
  } else {
    TT_ASSIGN_OR_THROW(dynamic_result_shapes,
                       GetOutputShapes(executable, output_shapes));
    result_shapes = dynamic_result_shapes;
  }
  // Get the materialized buffers for the bases of the argument tensors.
  TT_ASSIGN_OR_THROW(std::vector<DeviceBufferRef> argument_buffer_refs,
                     PrepareCompiledModeArgumentsImpl(
                         argument_tensors, executable->argument_indices()),
                     _.SetPrepend()
                         << "failed to prepare compiled mode arguments: ");

  TT_ASSIGN_OR_THROW(
      std::vector<DeviceBufferRef> result_buffer_refs,
      EnqueueExecutable(executable, std::move(argument_buffer_refs),
                        result_shapes));

  auto num_outputs = result_buffer_refs.size();
  std::vector<at::Tensor> output_tensors;
  output_tensors.reserve(num_outputs);
  for (auto& result_buffer : result_buffer_refs) {
    output_tensors.push_back(MakeTensor(std::move(result_buffer)));
  }

  return output_tensors;
}

}  // namespace

// This is bound to Python function tpu_torch_compile.execute(), and thus
// allowed to throw exceptions.
std::vector<at::Tensor> ExecuteCompiledModel(
    const SharedLoadedExecutableWithMetadata& executable,
    absl::Span<const at::Tensor> argument_tensors,
    absl::Span<const OutputShape> output_shapes) {
  return ExecuteCompiledModelImpl(executable, argument_tensors, output_shapes);
}

std::vector<at::Tensor> ExecuteCompiledModel(
    const SharedLoadedExecutableWithMetadata& executable,
    absl::Span<const at::Tensor* const> argument_tensors,
    absl::Span<const OutputShape> output_shapes) {
  return ExecuteCompiledModelImpl(executable, argument_tensors, output_shapes);
}

absl::StatusOr<at::Tensor> MakeConstantTensor(const at::Tensor& cpu_tensor) {
  TT_RET_CHECK(cpu_tensor.is_cpu(), error::kInvalidArgument)
      << "the input to MakeConstantTensor must be a CPU tensor";
  // Ensure that the input tensor is contiguous with zero offset. This requires
  // a copy if the tensor has an offset or is not contiguous.
  at::Tensor contiguous_cpu_bytes_tensor = cpu_tensor;
  if (!contiguous_cpu_bytes_tensor.is_contiguous() ||
      contiguous_cpu_bytes_tensor.storage_offset() != 0) {
    // Use clone(Contiguous) instead of .contiguous() so that contiguous slices
    // are reset to zero offset.
    contiguous_cpu_bytes_tensor =
        at::clone(cpu_tensor, at::MemoryFormat::Contiguous);
  }

  // Memcpy the bytes into a vector. We have to do this so that dropping the
  // tensor does not invalidate the DeferredOp we will create.
  const void* const cpu_bytes_ptr = contiguous_cpu_bytes_tensor.data_ptr();
  // Use tensor nbytes, not storage nbytes; if cpu_tensor is a contiguous,
  // zero-offset slice of a larger tensor, we only want to copy the bytes that
  // are in the slice, not the entire underlying buffer.
  const size_t num_bytes = contiguous_cpu_bytes_tensor.nbytes();
  std::vector<char> cpu_bytes_copy(num_bytes);
  // In the case of zero-sized constants both the data_ptr() pointer from the
  // input buffer and the data() pointer on the vector will be null.
  // Calling std::memcpy with a nullptr for either src or dest arguments is
  // undefined behavior.
  if (num_bytes > 0) {
    std::memcpy(cpu_bytes_copy.data(), cpu_bytes_ptr, num_bytes);
  }

  // Create a DeferredOp that will interpret the bytes as a constant tensor
  // of the desired shape and dtype. This does *NOT* preserve the strides or
  // storage offset of the input tensor.
  Dimensions dimensions = CopyIntVector(cpu_tensor.sizes());
  TT_ASSIGN_OR_RETURN(mlir::ElementType element_type,
                      ConvertTo<mlir::ElementType>(cpu_tensor.scalar_type()));
  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef buffer_ref,
      CreateConstantDeviceBufferRef(std::move(cpu_bytes_copy),
                                    std::move(dimensions), element_type));

  return MakeTensor(std::move(buffer_ref));
}

absl::Status AssignConstantTensor(const at::Tensor& cpu_src_tensor,
                                  const at::Tensor& tpu_dst_tensor) {
  TT_RET_CHECK(cpu_src_tensor.is_cpu(), error::kInvalidArgument)
      << "cpu_src_tensor must be a CPU tensor";
  TT_RET_CHECK(cpu_src_tensor.sizes() == tpu_dst_tensor.sizes(),
               error::kInvalidArgument)
      << "cpu_src_tensor and tpu_dst_tensor must have the same shape";
  TT_RET_CHECK(cpu_src_tensor.scalar_type() == tpu_dst_tensor.scalar_type(),
               error::kInvalidArgument)
      << "cpu_src_tensor and tpu_dst_tensor must have the same scalar "
         "type";

  // Make a temporary constant tensor on the CPU.
  TT_ASSIGN_OR_RETURN(at::Tensor constant_tensor,
                      MakeConstantTensor(cpu_src_tensor));

  // Copy the constant tensor into the destination tensor. This will preserve
  // the layout metadata (strides, offset) of tpu_dst_tensor.
  tpu_dst_tensor.copy_(constant_tensor);
  return absl::OkStatus();
}

}  // namespace torch_tpu
