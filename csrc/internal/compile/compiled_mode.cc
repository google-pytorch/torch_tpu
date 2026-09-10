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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "absl/algorithm/container.h"
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
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/materialize.h"
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
#include "pybind11/pybind11.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/ScalarType.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

namespace {

// To call a compiled PjRtExecutable, we need to get materialized base buffers
// for each argument tensor, in the same shape expected by the executable.
// This **must** match the logic in tpu_torch_compile's PyMakePlaceholderLike.
absl::StatusOr<std::vector<DeviceBufferRef>> PrepareCompiledModeArguments(
    absl::Span<const at::Tensor> argument_tensors) {
  // Get each argument buffer ref's expected contiguous base shape.
  std::vector<DeviceBufferRef> argument_buffer_refs;
  argument_buffer_refs.reserve(argument_tensors.size());
  for (const at::Tensor& argument_tensor : argument_tensors) {
    // If argument_tensor is contiguous with offset 0, then we know that
    // base_sizes and base_stride below would be the same as
    // `argument_tensor`, and skipping GetContiguousBaseShape saves some
    // time.
    if (TensorHasTrivialLayout(argument_tensor)) {
      TT_ASSIGN_OR_RETURN(
          DeviceBufferRef buffer_ref, GetBuffer(argument_tensor),
          _.SetPrepend() << "failed to get buffer from argument tensor: "
                         << ToString(argument_tensor));
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
    argument_buffer_refs.push_back(std::move(buffer_ref));
  }

  // Materialize the argument buffers.
  TT_RETURN_IF_ERROR(Materialize(argument_buffer_refs,
                                 MaterializationReason::kCompileModeExecution));

  // After materialization, each argument buffer will be a materialized,
  // contiguous tensor which the compiled executable can safely apply view
  // logic to.
  return argument_buffer_refs;
}

// Deduplicated Path: If we have argument_indices, populate only the unique
// base buffers expected by the executable.
absl::StatusOr<std::vector<DeviceBufferRef>> PrepareCompiledModeArguments(
    absl::Span<const at::Tensor> argument_tensors,
    const Indices& argument_indices) {
  if (argument_indices.empty()) {
    return PrepareCompiledModeArguments(argument_tensors);
  }

  TT_RET_CHECK(argument_tensors.size() == argument_indices.size(),
               error::kInvalidArgument)
      << "number of argument tensors (" << argument_tensors.size()
      << ") does not match compiled argument indices ("
      << argument_indices.size() << ")";

  int64_t num_base =
      *std::max_element(argument_indices.begin(), argument_indices.end()) + 1;
  std::vector<std::optional<DeviceBufferRef>> base_buffer_refs(num_base);
  for (size_t i = 0; i < argument_tensors.size(); ++i) {
    int64_t base_idx = argument_indices[i];
    if (!base_buffer_refs[base_idx].has_value()) {
      TT_ASSIGN_OR_RETURN(
          DeviceBufferRef buffer_ref, GetBaseBuffer(argument_tensors[i]),
          _.SetPrepend() << "failed to get buffer from argument tensor: "
                         << ToString(argument_tensors[i]));
      base_buffer_refs[base_idx] = std::move(buffer_ref);
    }
  }

  std::vector<DeviceBufferRef> argument_buffer_refs;
  argument_buffer_refs.reserve(num_base);
  for (int64_t i = 0; i < num_base; ++i) {
    ABSL_CHECK(base_buffer_refs[i].has_value())  // CRASH_OK
        << "base argument " << i << " was not populated";
    argument_buffer_refs.push_back(std::move(*base_buffer_refs[i]));
  }

  TT_RETURN_IF_ERROR(Materialize(argument_buffer_refs,
                                 MaterializationReason::kCompileModeExecution));
  return argument_buffer_refs;
}

}  // namespace

absl::StatusOr<at::Tensor> MakePlaceholder(absl::Span<const int64_t> sizes,
                                           at::ScalarType dtype,
                                           bool requires_grad) {
  TT_ASSIGN_OR_RETURN(mlir::ElementType tensor_element_type,
                      ConvertTo<mlir::ElementType>(dtype));
  Dimensions physical_sizes(sizes.begin(), sizes.end());
  if (dtype == at::kFloat4_e2m1fn_x2) {
    TT_RET_CHECK(!physical_sizes.empty(), error::kInvalidArgument)
        << "expected float4_e2m1fn_x2 tensors to be at least 1-dimensional";
    physical_sizes.back() *= 2;
  }
  TT_ASSIGN_OR_RETURN(DeviceBufferRef placeholder,
                      DeviceBufferList::CreatePlaceholder(
                          std::move(physical_sizes), tensor_element_type));
  auto new_tensor = MakeTensor(std::move(placeholder));
  new_tensor.set_requires_grad(requires_grad);
  return new_tensor;
}

absl::StatusOr<at::Tensor> MakePlaceholder(Shape shape, bool requires_grad) {
  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef placeholder,
      DeviceBufferList::CreatePlaceholder(shape.dimensions(), shape.dtype()));
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
  TT_RETURN_IF_ERROR(
      traversal->ValidateAndReorderArguments(std::move(argument_refs)))
          .SetPrepend()
      << "failed to validate and reorder inputs: ";
  TT_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
      traversal->BuildMlirModule(mlir_context, use_stablehlo_bounds));

  return mlir_module;
}

absl::StatusOr<CompileResult> TraverseAndCompile(
    const std::vector<at::Tensor>& result_tensors,
    const std::vector<at::Tensor>& argument_tensors,
    const TraverseAndCompileOptions& options) {
  tsl::profiler::TraceMe trace_await("TraverseAndCompile");

  ScopedPythonContextCapturer capturer(OpName::kCompileMlir);
  ScopedPythonContextProvider provider(
      ScopedPythonContextCapturer::GetContext());

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
    // as argument_tensors is expected to have few elements (<100)
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

  // Release the Python GIL before XLA compilation to allow multi-threaded
  // compilation.
  pybind11::gil_scoped_release release;

  // 2. Compile Traversal and get exec
  auto compilation_spec = GetCompilationSpec(options.compilation_mode);
  TT_ASSIGN_OR_CRASH(  // CRASH_OK=implies a bug in compile backend if this
                       // happens
      auto compiled_kernel,
      traversal->Compile(std::move(compilation_spec), nullptr,
                         options.use_stablehlo_bounds, options.argument_layouts,
                         unique_donated_inputs, options.module_name),
      _ << "failed to compile traversal");

  TT_ASSIGN_OR_RETURN(auto executable, compiled_kernel.fixed_shape_kernel.get(),
                      _.SetPrepend() << "failed to get fixed shape kernel: ");
  if (has_duplicates) {
    executable = LoadedExecutableWithMetadata::WithArgumentIndices(
        executable, std::move(argument_indices));
  }

  std::shared_ptr<ContextedModule> module = nullptr;
  if (options.build_mlir_module) {
    ABSL_VLOG(1) << "Building MLIR module as requested.";

    TT_ASSIGN_OR_CRASH(  // CRASH_OK=implies a bug in compile backend if this
                         // happens
        auto contexted_module,
        ContextedModule::Make(
            [&](mlir::MLIRContext& mlir_context)
                -> absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> {
              return traversal->BuildMlirModule(
                  mlir_context, options.use_stablehlo_bounds,
                  unique_donated_inputs, options.module_name);
            }),
        _ << "failed to build MLIR module");
    module = std::make_shared<ContextedModule>(std::move(contexted_module));
  }

  return CompileResult{
      .module = std::move(module),
      .executable = std::move(executable),
  };
}

absl::StatusOr<SharedLoadedExecutableWithMetadata> CompileMlirExecutable(
    const std::string_view mlir_module_bytecode,
    UniqueCompileOptions compile_options) {
  TT_ASSIGN_OR_RETURN(
      ContextedModule module,
      ContextedModule::Make(
          [&](mlir::MLIRContext& mlir_context)
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
  xla::PjRtClient* const client = PjrtBackend::GetInstance().GetClient();
  TT_RET_CHECK(client, error::kFailedPrecondition)
      << "PjRtClient must be initialized";

  LoadedExecutableBuilder executable_builder =
      [module = std::move(module)](
          xla::PjRtClient& client,
          UniqueCompileOptions compile_options) mutable {
        return AdaptXlaError(client.CompileAndLoad(std::move(module),
                                                   std::move(*compile_options)),
                             /* context= */ "failed to compile MLIR module");
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
        TT_RET_CHECK(dimensions[j] <= result_shape.dimensions()[j],
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

}  // namespace

// This is bound to Python function tpu_torch_compile.execute(), and thus
// allowed to throw exceptions.
std::vector<at::Tensor> ExecuteCompiledModel(
    const SharedLoadedExecutableWithMetadata& executable,
    absl::Span<const at::Tensor> argument_tensors,
    absl::Span<const OutputShape> output_shapes) {
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
                     PrepareCompiledModeArguments(
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

absl::StatusOr<at::Tensor> MakeConstantTensor(const at::Tensor& cpu_tensor) {
  TT_RET_CHECK(cpu_tensor.is_cpu(), error::kInvalidArgument)
      << "the input to MakeConstantTensor must be a CPU tensor";
  // Ensure that the input tensor is a contiguous, 1D array of bytes.
  // If the input tensor is already a 1D array of bytes, this will be a no-op.
  at::Tensor contiguous_cpu_bytes_tensor =
      cpu_tensor.flatten().view(at::ScalarType::Byte);
  if (!contiguous_cpu_bytes_tensor.is_contiguous()) {
    contiguous_cpu_bytes_tensor = contiguous_cpu_bytes_tensor.contiguous();
  }

  // Memcpy the bytes into a vector. We have to do this so that dropping the
  // tensor does not invalidate the DeferredOp we will create.
  const void* const cpu_bytes_ptr = contiguous_cpu_bytes_tensor.data_ptr();
  const size_t num_bytes = contiguous_cpu_bytes_tensor.storage().nbytes();
  std::vector<char> cpu_bytes_copy(num_bytes);
  // In the case of zero-sized constants both the data_ptr() pointer from the
  // input buffer and the data() pointer on the vector will be null.
  // Calling std::memcpy with a nullptr for either src or dest arguments is
  // undefined behavior.
  if (num_bytes > 0) {
    std::memcpy(cpu_bytes_copy.data(), cpu_bytes_ptr, num_bytes);
  }

  // Create a DeferredOp that represents the 1D array of bytes as a constant.
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
