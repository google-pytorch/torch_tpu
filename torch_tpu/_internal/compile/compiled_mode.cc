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
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/log/log.h"
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
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/materialize.h"
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
#include "xla/shape.h"
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

absl::StatusOr<SharedLoadedExecutable> CompileMlirExecutable(
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

absl::StatusOr<SharedLoadedExecutable> CompileMlirExecutable(
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

absl::StatusOr<std::vector<Shape>> GetOutputShapes(
    const SharedLoadedExecutable& executable,
    absl::Span<const std::vector<int64_t>>  // INT_VEC_OK
        output_shapes) {
  // Get the flattened output shapes from the executable.
  // Executables can return tuples, we want individual tensor shapes.
  TT_ASSIGN_OR_RETURN(std::vector<xla::Shape> output_shapes_vec,
                      executable->GetOutputShapes());
  std::vector<const xla::Shape*> output_shapes_flat_vec;
  for (const auto& output_shape : output_shapes_vec) {
    xla::ShapeUtil::FlattenTupleShape(output_shape, output_shapes_flat_vec);
  }

  // Determine output shapes from the executable.
  std::vector<Shape> result_shapes_vec;
  result_shapes_vec.reserve(output_shapes_flat_vec.size());

  if (!output_shapes.empty()) {
    TT_RET_CHECK(output_shapes.size() == output_shapes_flat_vec.size(),
                 error::kInvalidArgument)
        << "output shapes must be specified for all outputs or none, "
        << "got " << output_shapes.size() << " output shapes for "
        << output_shapes_flat_vec.size() << " output tensors";
  }

  for (size_t i = 0; i < output_shapes_flat_vec.size(); ++i) {
    const xla::Shape* output_shape = output_shapes_flat_vec[i];
    TT_ASSIGN_OR_RETURN(Shape result_shape, MakeShape(*output_shape));
    if (!output_shapes.empty()) {
      const auto& shape = output_shapes[i];
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
    }
    result_shapes_vec.push_back(result_shape);
  }

  return result_shapes_vec;
}

}  // namespace

// This is bound to Python function tpu_torch_compile.execute(), and thus
// allowed to throw exceptions.
std::vector<at::Tensor> ExecuteCompiledModel(
    const SharedLoadedExecutable& executable,
    absl::Span<const at::Tensor> argument_tensors,
    absl::Span<const std::vector<int64_t>>  // INT_VEC_OK
        output_shapes) {
  TT_ASSIGN_OR_THROW(std::vector<Shape> result_shapes_vec,
                     GetOutputShapes(executable, output_shapes));
  // Get the materialized buffers for the argument tensors.
  TT_ASSIGN_OR_THROW(std::vector<DeviceBufferRef> argument_buffer_refs,
                     GetMaterialized(argument_tensors));

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

}  // namespace torch_tpu
