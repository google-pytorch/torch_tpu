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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "ATen/core/TensorBody.h"
#include "torch/extension.h"  // IWYU pragma: keep for aten::Tensor pybind type
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/_internal/compile/compiled_mode.h"
#include "torch_tpu/_internal/dynamism/dynamism_ops.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/context_manager.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_gen_impl.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/hlo/translate/register.h"
#include "xla/layout.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"

namespace torch_tpu {
namespace py = pybind11;

namespace {

at::Tensor PyMakePlaceholder(const std::vector<int64_t>& sizes,  // INT_VEC_OK
                             at::ScalarType dtype, bool requires_grad) {
  TT_ASSIGN_OR_THROW(at::Tensor prepared_tensor,
                     MakePlaceholder(sizes, dtype, requires_grad),
                     _.SetPrepend() << "failed to create placeholder tensor: ");
  return prepared_tensor;
}

at::Tensor PyMakePlaceholderLike(const at::Tensor& arg_tensor) {
  const std::vector<int64_t> sizes(arg_tensor.sizes().begin(),  // INT_VEC_OK
                                   arg_tensor.sizes().end());
  return PyMakePlaceholder(sizes, arg_tensor.scalar_type(),
                           arg_tensor.requires_grad());
}

py::object PyGetDeviceLayoutIfMaterialized(const at::Tensor& tensor) {
  TT_ASSIGN_OR_THROW(DeviceBufferRef buffer_ref, GetBufferFromAtTensor(tensor));
  if (!buffer_ref.IsMaterialized()) {
    return py::none();
  }
  TT_ASSIGN_OR_THROW(auto* pjrt_buffer, buffer_ref.GetOrMaterializeBuffer());
  const auto pjrt_layout = pjrt_buffer->layout();
  if (!pjrt_layout) {
    return py::none();
  }
  const xla::Layout& layout = pjrt_layout->xla_layout();

  std::vector<int64_t>  // INT_VEC_OK
      minor_to_major(layout.minor_to_major().begin(),
                     layout.minor_to_major().end());

  std::vector<std::vector<int64_t>> tiles;  // INT_VEC_OK
  tiles.reserve(layout.tiles().size());
  for (const auto& tile : layout.tiles()) {
    tiles.push_back(std::vector<int64_t>  // INT_VEC_OK
                    (tile.dimensions().begin(), tile.dimensions().end()));
  }

  return py::cast(
      std::make_tuple(minor_to_major, tiles, layout.element_size_in_bits()));
}

// Returns a ContextedModule corresponding to the graph terminating at
// result_tensors and taking argument_tensors as inputs.
std::shared_ptr<ContextedModule> PyBuildMlir(
    const std::vector<at::Tensor>& result_tensors,
    const std::vector<at::Tensor>& argument_tensors) {  // INT_VEC_OK
  TT_ASSIGN_OR_THROW(ContextedModule module,
                     ContextedModule::Make([&](mlir::MLIRContext& context) {
                       return ExtractMlirFromGraph(context, argument_tensors,
                                                   result_tensors);
                     }));

  mlir::BaseScopedDiagnosticHandler diag_handler(&module.context());
  if (failed(mlir::verify(module.get()))) {
    TT_THROW_IF_ERROR(
        TT_ERROR(error::kInternal)
        << "MLIR module is invalid:\n"
        << diag_handler.ConsumeStatus().message() << "\n"
        << DebugString(module.get(), DebugStringOptions::kEnableDebugInfo));
  }
  return std::make_shared<ContextedModule>(std::move(module));
}

// Parses MLIR text, verifies it, and returns a ContextedModule.
std::shared_ptr<ContextedModule> PyParseMlirText(const std::string& mlir_text) {
  TT_ASSIGN_OR_THROW(
      ContextedModule module,
      ContextedModule::Make(
          [&](mlir::MLIRContext& context)
              -> absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> {
            return mlir::parseSourceString<mlir::ModuleOp>(
                llvm::StringRef(mlir_text.data(), mlir_text.size()),
                mlir::ParserConfig{&context});
          }));
  if (!module.get()) {
    TT_THROW_IF_ERROR(TT_ERROR(error::kInvalidArgument)
                      << "Failed to parse MLIR module text.");
  }

  if (failed(mlir::verify(module.get()))) {
    TT_THROW_IF_ERROR(
        TT_ERROR(error::kInternal)
        << "MLIR module is invalid.\n"
        << DebugString(module.get(), DebugStringOptions::kEnableDebugInfo));
  }
  return std::make_shared<ContextedModule>(std::move(module));
}

// Serializes a ContextedModule to MLIR text.
std::string PySerializeMlirText(std::shared_ptr<ContextedModule> module,
                                bool enable_debug_info) {
  return DebugString(module->get(),
                     enable_debug_info ? DebugStringOptions::kEnableDebugInfo
                                       : DebugStringOptions::kDisableDebugInfo);
}

// Serializes a ContextedModule to bytecode.
py::bytes PySerializeMlirBytecode(std::shared_ptr<ContextedModule> module) {
  TT_ASSIGN_OR_THROW(std::string bytecode, SerializeBytecode(module->get()));
  return py::bytes(std::move(bytecode));
}

// Serializes a ContextedModule to a versioned portable artifact.
py::bytes PySerializePortableArtifact(std::shared_ptr<ContextedModule> module) {
  TT_ASSIGN_OR_THROW(std::string bytecode,
                     SerializePortableArtifact(module->get()));
  return py::bytes(std::move(bytecode));
}

// Compiles an MLIR module.
// Args:
//   module: The ContextedModule to compile.
//   fast_compile: If true, use the compiler profile optimized for eager
//         execution; otherwise, use the profile optimized for
//         torch.compile.
// Returns:
//   The compiled executable.
SharedLoadedExecutableWithMetadata PyCompileMlir(
    std::shared_ptr<ContextedModule> module, const bool fast_compile) {
  ScopedPythonContextCapturer capturer(OpName::kCompileMlir);
  // Provide the current python context to the compilation function so that
  // it can generate readable Mlir module names.
  ScopedPythonContextProvider provider(
      ScopedPythonContextCapturer::GetContext());
  TT_ASSIGN_OR_THROW(
      auto executable,
      CompileMlirExecutable(xla::MaybeOwningMlirModule(module->get()),
                            fast_compile ? CompilationMode::kFastCompile
                                         : CompilationMode::kFastRuntime));
  return executable;
}

CompileResult PyTraverseAndCompile(
    const std::vector<at::Tensor>& result_tensors,
    const std::vector<at::Tensor>& argument_tensors, bool fast_compile,
    bool build_mlir_module) {
  TT_ASSIGN_OR_THROW(
      CompileResult result,
      TraverseAndCompile(
          result_tensors, argument_tensors,
          TraverseAndCompileOptions{
              .compilation_mode = fast_compile ? CompilationMode::kFastCompile
                                               : CompilationMode::kFastRuntime,
              .build_mlir_module = build_mlir_module,
          }),
      _.SetPrepend() << "Failed to traverse and compile: ");
  return result;
}

// Executes a compiled PJRT executable with the given arguments.
//
// Args:
//   executable: The loaded PJRT executable to run.
//   argument_tensors: A list of PyTorch tensors to pass as inputs.
//   output_shapes: Optional list of shapes for the output tensors.
//     If not empty, the number of elements in output_shapes must match the
//     number of output tensors. These shapes override the static shapes
//     inferred from the executable. This is useful for bounded dynamic programs
//     where the actual output shape might differ from the static
//     upper bound.
std::vector<at::Tensor> PyExecuteCompiledModel(
    const SharedLoadedExecutableWithMetadata& executable,
    const std::vector<at::Tensor>& argument_tensors,
    const std::vector<std::vector<int64_t>>& output_shapes) {  // INT_VEC_OK
  return ExecuteCompiledModel(executable, argument_tensors, output_shapes);
}

void PyPushEnableTracebacks(std::optional<bool> enabled) {
  const EnableTracebacksContextState state =
      enabled.has_value()
          ? std::make_optional(enabled.value() ? TracebackMode::kEnabled
                                               : TracebackMode::kDisabled)
          : std::nullopt;
  PushContextState(state);
}

// Returns the internal TPU RNG state tensor from a generator.
at::Tensor PyGetDeviceStateTensor(at::Generator gen) {
  auto* device_gen = at::check_generator<DeviceGeneratorImpl>(gen);
  return device_gen
      ->DeviceStateTensor();  // DEVICE_STATE_TENSOR_OK=python callback
}

// Sets the internal TPU RNG state tensor on a generator.
void PySetDeviceStateTensor(at::Generator gen, at::Tensor rng_state) {
  auto* device_gen = at::check_generator<DeviceGeneratorImpl>(gen);
  TT_THROW_IF_ERROR(
      device_gen
          ->SetDeviceStateTensor(  // DEVICE_STATE_TENSOR_OK=python callback
              std::move(rng_state)));
}

}  // namespace

// Returns the MLIR module for a pad subgraph.
// Args:
//   tensor_info: A list of pairs, where each pair contains the shape of a
//     tensor and its scalar type.
//   bounds_list: A list of pairs, where each pair contains the dynamic
//     dimensions of a tensor and their upper bounds.
// Returns:
//   The MLIR module for the pad subgraph as bytes.
// Example:
//   tensor_info = [([1, 4], torch.int64)]
//   bounds_list = [([1], [8])]. // Dynamic dimension is 1, upper bound is 8.
//  MLIR module:
// module @pad_module {
//   func.func @main(%arg0: tensor<1x4xi64>) -> (tensor<1x8xi64>, tensor<i32>) {
//     %c = stablehlo.constant dense<0> : tensor<i64>
//     %0 = stablehlo.pad %arg0, %c, low = [0, 0], high = [0, 4],
//       interior = [0, 0] : (tensor<1x4xi64>, tensor<i64>) -> tensor<1x8xi64>
//     %1 = stablehlo.get_dimension_size %arg0, dim = 1 :
//        (tensor<1x4xi64>) -> tensor<i32>
//     return %0, %1 : tensor<1x8xi64>, tensor<i32>
//   }
// }

std::shared_ptr<ContextedModule> PyGetPadModuleMlir(
    const std::vector<
        std::pair<std::vector<int64_t>, at::ScalarType>>&  // INT_VEC_OK
        tensor_info,
    const std::vector<
        std::pair<std::vector<int64_t>, std::vector<int64_t>>>&  // INT_VEC_OK
        bounds_list) {
  auto context = std::make_unique<mlir::MLIRContext>();
  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();

  std::vector<Shape> shapes;
  shapes.reserve(tensor_info.size());
  for (int64_t i = 0; i < tensor_info.size(); ++i) {
    const auto& info = tensor_info[i];
    Shape shape;
    shape.dimensions().assign(info.first.begin(), info.first.end());
    TT_ASSIGN_OR_THROW(mlir::ElementType element_type,
                       internal::ToElementType(info.second));
    shape.set_dtype(element_type);

    if (i < bounds_list.size()) {
      const auto& dims = bounds_list[i].first;
      const auto& bounds = bounds_list[i].second;
      if (!bounds.empty() && !dims.empty()) {
        TT_CHECK_THROW(dims.size() == bounds.size(), error::kInvalidArgument)
            << "dimension indices and upper bounds must have the same size, "
            << "got " << dims.size() << " and " << bounds.size();
        for (size_t j = 0; j < dims.size(); ++j) {
          const int64_t dim_index = dims[j];
          const int64_t dim_upper_bound = bounds[j];
          TT_CHECK_THROW(
              dim_index >= 0 && dim_index < shape.dimensions().size(),
              error::kInvalidArgument)
              << "dimension index must be within bounds [0, "
              << shape.dimensions().size() - 1 << "], got " << dim_index
              << " for input tensor " << i << " with shape "
              << ToString(shape.dimensions());

          const int64_t dim_size = shape.dimensions()[dim_index];
          TT_CHECK_THROW(dim_upper_bound >= dim_size, error::kInvalidArgument)
              << "upper bound must be greater than or equal to the static "
                 "shape's dimension size, got upper bound "
              << dim_upper_bound << " for dimension " << dim_index
              << " for input tensor " << i << " with shape "
              << ToString(shape.dimensions());

          shape.dynamic_dimensions().push_back(
              {dim_index, dim_size, dim_upper_bound});
        }
      }
    }
    shapes.push_back(shape);
  }

  TT_ASSIGN_OR_THROW(mlir::OwningOpRef<mlir::ModuleOp> module,
                     GetPadModule(*context, shapes));

  return std::make_shared<torch_tpu::ContextedModule>(std::move(context),
                                                      std::move(module));
}

py::bytes PySerializeExecutable(
    const SharedLoadedExecutableWithMetadata& executable) {
  TT_ASSIGN_OR_THROW(const std::string serialized,
                     executable->GetExecutable()->SerializeExecutable(),
                     _.SetPrepend() << "Failed to serialize executable: ");
  return py::bytes(serialized);
}

SharedLoadedExecutableWithMetadata PyLoadSerializedExecutable(
    py::bytes& serialized_bytes) {
  const auto data = py::cast<std::string_view>(serialized_bytes);
  xla::PjRtClient* const client = PjrtBackend::GetInstance().GetClient();
  TT_CHECK_THROW(client != nullptr, error::kFailedPrecondition)
      << "PjRtClient must be initialized before loading a serialized "
         "executable.";
  TT_ASSIGN_OR_THROW(std::unique_ptr<xla::PjRtLoadedExecutable> pjrt_executable,
                     client->LoadSerializedExecutable(
                         data, /*options=*/std::nullopt, xla::LoadOptions()),
                     _.SetPrepend()
                         << "Failed to load serialized executable: ");
  TT_ASSIGN_OR_THROW(
      SharedLoadedExecutableWithMetadata executable,
      LoadedExecutableWithMetadata::MakeShared(std::move(pjrt_executable)),
      _.SetPrepend()
          << "Failed to create SharedLoadedExecutableWithMetadata: ");
  return executable;
}

at::Tensor PyMakeConstantTensor(const at::Tensor& cpu_tensor) {
  TT_ASSIGN_OR_THROW(at::Tensor tpu_tensor, MakeConstantTensor(cpu_tensor));
  return tpu_tensor;
}

void PyAssignConstantTensor(const at::Tensor& cpu_src_tensor,
                            const at::Tensor& tpu_dst_tensor) {
  TT_THROW_IF_ERROR(AssignConstantTensor(cpu_src_tensor, tpu_dst_tensor));
}

PYBIND11_MODULE(tpu_torch_compile, m) {
  py::class_<CompileResult>(m, "CompileResult")
      .def_readonly("module", &CompileResult::module)
      .def_readonly("executable", &CompileResult::executable);

  py::class_<torch_tpu::ContextedModule,  // NOLINT(bugprone-unused-raii)
             std::shared_ptr<torch_tpu::ContextedModule>>(m, "ContextedModule");

  py::class_<  // NOLINT(bugprone-unused-raii)
      torch_tpu::LoadedExecutableWithMetadata,
      std::shared_ptr<torch_tpu::LoadedExecutableWithMetadata>>(
      m, "LoadedExecutableWithMetadata");

  py::class_<xla::PjRtLoadedExecutable,  // NOLINT(bugprone-unused-raii)
             std::shared_ptr<xla::PjRtLoadedExecutable>>(
      m, "PjRtLoadedExecutable");
  m.def("execute", PyExecuteCompiledModel,
        // Type: PjRtLoadedExecutable
        py::arg("executable"), py::arg("argument_tensors"),
        py::arg("output_shapes") =
            std::vector<std::vector<int64_t>>(),  // INT_VEC_OK
        "Executes a compiled PJRT executable with the given arguments.\n\n"
        "Args:\n"
        "  executable: The loaded PJRT executable to run.\n"
        "  argument_tensors: A list of PyTorch tensors to pass as inputs.\n"
        "  output_shapes: Optional list of shapes for the output tensors.\n"
        "    If provided, the number of elements must match the number of "
        "output tensors. These shapes override the static shapes inferred\n"
        "    from the executable.\n"
        "    This is useful for bounded dynamic programs where the actual\n"
        "    output shape might differ from the static upper bound.");
  m.def("placeholder", PyMakePlaceholder, py::arg("sizes"), py::arg("dtype"),
        py::arg("requires_grad"));
  m.def("placeholder_like", PyMakePlaceholderLike, py::arg("arg_tensor"));
  m.def("get_device_layout_if_materialized", &PyGetDeviceLayoutIfMaterialized,
        py::arg("tensor"));
  m.def("traverse_and_compile", PyTraverseAndCompile, py::arg("result_tensors"),
        py::arg("argument_tensors"), py::arg("fast_compile") = false,
        py::arg("build_mlir_module") = false,
        "Traverses the graph from outputs to arguments and compiles it.");

  m.def("build_mlir", PyBuildMlir, py::arg("result_tensors"),
        py::arg("argument_tensors"));  // INT_VEC_OK
  // Returns: PjRtLoadedExecutable
  m.def("compile_mlir", PyCompileMlir, py::arg("module"),
        py::arg("fast_compile") = false);
  m.def("parse_mlir_text", PyParseMlirText, py::arg("mlir_text"),
        "Parses a StableHLO MLIR text string and returns a ContextedModule.");
  m.def("serialize_mlir_text", PySerializeMlirText, py::arg("module"),
        py::arg("enable_debug_info") = false,
        "Serializes a ContextedModule to MLIR text.");
  m.def("serialize_mlir_bytecode", PySerializeMlirBytecode, py::arg("module"),
        "Serializes a ContextedModule to bytecode.");
  m.def("serialize_mlir_portable_artifact", PySerializePortableArtifact,
        py::arg("module"),
        "Serializes a ContextedModule to a versioned portable artifact.");
  m.def("get_pad_module_mlir", PyGetPadModuleMlir, py::arg("tensor_info"),
        py::arg("bounds_list"),
        "Returns the MLIR module for a pad subgraph as bytecode.\n\n"
        "Args:\n"
        "  tensor_info: A list of (shape, dtype) pairs for each tensor.\n"
        "  bounds_list: A list of (dynamic_dimensions, upper_bounds) pairs.");
  m.def("pop_enable_tracebacks", &PopContextState<EnableTracebacksContextState>,
        "Pops the current state of the enable_tracebacks context manager.");
  m.def("push_enable_tracebacks", &PyPushEnableTracebacks, py::arg("enabled"),
        "Pushes a new state onto the enable_tracebacks context manager.");
  m.def("serialize_executable", PySerializeExecutable, py::arg("executable"),
        "Serializes a PjRtLoadedExecutable to bytes for caching.");
  m.def("load_serialized_executable", PyLoadSerializedExecutable,
        py::arg("serialized_bytes"),
        "Loads a PjRtLoadedExecutable from serialized bytes.");
  m.def("make_constant_tensor", PyMakeConstantTensor, py::arg("cpu_tensor"),
        "Creates a TPU tensor that represents the constant value of the CPU "
        "tensor.");
  m.def("assign_constant_tensor", PyAssignConstantTensor,
        py::arg("cpu_src_tensor"), py::arg("tpu_dst_tensor"),
        "Updates a TPU tensor to be a tensor with the constant value of the "
        "CPU tensor.");
  m.def("get_device_state_tensor", PyGetDeviceStateTensor, py::arg("generator"),
        "Returns the internal TPU RNG state tensor from a generator.");
  m.def("set_device_state_tensor", PySetDeviceStateTensor, py::arg("generator"),
        py::arg("rng_state"),
        "Sets the internal TPU RNG state tensor on a generator.");
}

}  // namespace torch_tpu
