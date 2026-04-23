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

#include "torch_tpu/_internal/compile/compiled_mode.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "ATen/core/ATen_fwd.h"
#include "c10/core/TensorImpl.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/materialize.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/eager/traversal.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

absl::StatusOr<at::Tensor> MakePlaceholder(absl::Span<const int64_t> sizes,
                                           at::ScalarType dtype,
                                           bool requires_grad) {
  TT_ASSIGN_OR_RETURN(mlir::ElementType tensor_element_type,
                      ConvertTo<mlir::ElementType>(dtype));
  TT_ASSIGN_OR_RETURN(DeviceBufferRef placeholder,
                      DeviceBufferList::MakePlaceholder(CopyIntVector(sizes),
                                                        tensor_element_type));
  auto new_tensor = MakeTensor(std::move(placeholder));
  new_tensor.set_requires_grad(requires_grad);
  return new_tensor;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ExtractMlirFromGraph(
    mlir::MLIRContext& mlir_context, const std::vector<at::Tensor>& arg_tensors,
    const std::vector<at::Tensor>& result_tensors) {
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
    TT_ASSIGN_OR_RETURN(
        DeviceBufferRef buffer_ref, GetBufferFromAtTensor(tensor),
        _.SetPrepend() << "failed to get buffer from argument tensor: "
                       << ToString(tensor));
    TT_RET_CHECK(buffer_ref.state() != DeviceBufferRefState::kDeferred,
                 error::kInternal)
        << "argument tensor has deferred ops: " << ToString(tensor);
    ABSL_VLOG(3) << "[ExtractMlirFromGraph] arg_tensor: "
                 << buffer_ref.DebugString();
    argument_refs.push_back(std::move(buffer_ref));
  }

  std::vector<DeviceBufferRef> result_refs;
  result_refs.reserve(result_tensors.size());
  for (const at::Tensor& tensor : result_tensors) {
    TT_ASSIGN_OR_RETURN(
        DeviceBufferRef buffer_ref, GetBufferFromAtTensor(tensor),
        _.SetPrepend() << "failed to get buffer from result tensor: "
                       << ToString(tensor));
    TT_RET_CHECK(buffer_ref.state() != DeviceBufferRefState::kMaterialized,
                 error::kInternal)
        << "result tensor is already materialized: " << ToString(tensor);
    ABSL_VLOG(3) << "[ExtractMlirFromGraph] result_tensor: "
                 << buffer_ref.DebugString();
    result_refs.push_back(std::move(buffer_ref));
  }

  TT_ASSIGN_OR_RETURN(Traversal traversal,
                      Traversal::Create(std::move(result_refs)),
                      _.SetPrepend() << "failed to traverse graph: ");
  TT_RETURN_IF_ERROR(
      traversal.ValidateAndReorderArguments(std::move(argument_refs)))
          .SetPrepend()
      << "failed to validate and reorder inputs: ";
  TT_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                      traversal.BuildMlirModule(mlir_context));

  return mlir_module;
}

absl::StatusOr<SharedLoadedExecutableWithMetadata> CompileMlirExecutable(
    const std::string_view mlir_module_bytecode,
    const CompilationMode compilation_mode) {
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
                               compilation_mode);
}

absl::StatusOr<SharedLoadedExecutableWithMetadata> CompileMlirExecutable(
    xla::MaybeOwningMlirModule module, const CompilationMode compilation_mode) {
  TT_ASSIGN_OR_RETURN(UniqueCompileOptions compile_options,
                      MakeCompilerOptions(compilation_mode));
  xla::PjRtClient* const client = PjrtBackend::GetInstance().GetClient();
  TT_RET_CHECK(client, error::kFailedPrecondition)
      << "PjRtClient must be initialized";

  LoadedExecutableBuilder executable_builder =
      [module = std::move(module)](
          xla::PjRtClient& client,
          UniqueCompileOptions compile_options) mutable {
        return client.CompileAndLoad(std::move(module),
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
    absl::Span<const std::vector<int64_t>>  // INT_VEC_OK
        runtime_output_shapes) {
  const std::vector<Shape>& inferred_shapes = executable->output_shapes();

  if (runtime_output_shapes.empty()) {
    return inferred_shapes;
  }

  TT_RET_CHECK(runtime_output_shapes.size() == inferred_shapes.size(),
               error::kInvalidArgument)
      << "output shapes must be specified for all outputs or none, "
      << "got " << runtime_output_shapes.size() << " output shapes for "
      << inferred_shapes.size() << " output tensors";

  std::vector<Shape> result_shapes;
  result_shapes.reserve(inferred_shapes.size());

  for (size_t i = 0; i < inferred_shapes.size(); ++i) {
    Shape result_shape = inferred_shapes[i];
    const auto& shape = runtime_output_shapes[i];
    TT_RET_CHECK(shape.size() == result_shape.dimensions().size(),
                 error::kInvalidArgument)
        << "output shape number of dimensions must match the statically "
           "inferred dimensions, got output shape dimensions "
        << shape.size() << " and inferred dimensions "
        << result_shape.dimensions().size() << " for output tensor " << i;

    for (size_t j = 0; j < shape.size(); ++j) {
      TT_RET_CHECK(shape[j] <= result_shape.dimensions()[j],
                   error::kInvalidArgument)
          << "output shape dimension must not exceed the statically "
             "inferred bound, got output shape "
          << ToString(shape) << " and inferred shape "
          << ToString(result_shape.dimensions());
    }

    result_shape.dimensions().assign(shape.begin(), shape.end());
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
    absl::Span<const std::vector<int64_t>>  // INT_VEC_OK
        output_shapes) {
  TT_ASSIGN_OR_THROW(std::vector<Shape> result_shapes_vec,
                     GetOutputShapes(executable, output_shapes));
  // Get the materialized buffers for the argument tensors.
  TT_ASSIGN_OR_THROW(
      std::vector<DeviceBufferRef> argument_buffer_refs,
      GetMaterialized(argument_tensors,
                      MaterializationReason::kCompileModeExecution));

  TT_ASSIGN_OR_THROW(
      std::vector<DeviceBufferRef> result_buffer_refs,
      EnqueueExecutable(executable, std::move(argument_buffer_refs),
                        result_shapes_vec));

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
      DeviceBufferList::CreateConstant(std::move(cpu_bytes_copy),
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
