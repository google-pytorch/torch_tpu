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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "torch/extension.h"  // IWYU pragma: keep for aten::Tensor pybind type
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/_internal/compile/compiled_mode.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "stablehlo/dialect/Version.h"
#include "xla/hlo/translate/register.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace torch_tpu {
namespace py = pybind11;

namespace {

// Enum class for supported MLIR printing configurations.
enum class MlirPrintConfig {
  kMlirPretty,
  kMlirDebugInfo,
  kMlirSerialized,
  kMlirVersionedSerialized,
};

// Converts between torch_tpu.export.MlirPrintConfig and the C++ enum class.
MlirPrintConfig PyToMlirPrintConfig(const std::string& print_config) {
  if (print_config == "MlirPretty") {
    return MlirPrintConfig::kMlirPretty;
  } else if (print_config == "MlirDebugInfo") {
    return MlirPrintConfig::kMlirDebugInfo;
  } else if (print_config == "MlirSerialized") {
    return MlirPrintConfig::kMlirSerialized;
  } else if (print_config == "MlirVersionedSerialized") {
    return MlirPrintConfig::kMlirVersionedSerialized;
  }
  TT_THROW_IF_ERROR(TT_ERROR(error::kInvalidArgument)
                    << "Unknown MLIR print config.");
  llvm_unreachable("throws");
}

// Serializes MLIR module to pass back to python.
// Supports `MlirPrintConfig` options.
std::string PrintMlirModule(mlir::ModuleOp module,
                            MlirPrintConfig print_config) {
  switch (print_config) {
    case MlirPrintConfig::kMlirPretty:
      return DebugString(module, DebugStringOptions::kDisableDebugInfo);
    case MlirPrintConfig::kMlirDebugInfo:
      return DebugString(module, DebugStringOptions::kEnableDebugInfo);
    case MlirPrintConfig::kMlirSerialized: {
      std::string mlir_str;
      llvm::raw_string_ostream os(mlir_str);
      if (mlir::failed(mlir::writeBytecodeToFile(module, os))) {
        TT_THROW_IF_ERROR(TT_ERROR(error::kInternal)
                          << "Failed to serialize MLIR module to bytecode.");
      }
      return mlir_str;
    }
    case MlirPrintConfig::kMlirVersionedSerialized: {
      TT_ASSIGN_OR_THROW(
          auto mlir_str,
          xla::SerializeUsingVersionedStablehlo(
              module, mlir::vhlo::Version::fromCompatibilityRequirement(
                          mlir::vhlo::Version::CompatibilityRequirement::WEEK_4)
                          .toString()));
      return mlir_str;
    }
  }
  llvm::report_fatal_error(
      "[PrintMlirModule] Unreachable, unknown MlirPrintConfig");
}

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

// Returns MLIR module corresponding to the graph terminating at result_tensors
// and taking argument_tensors as inputs.
//
// Supports `MlirPrintConfig` options.
py::bytes PyExtractMlirModule(
    const std::vector<at::Tensor>& result_tensors,
    const std::vector<at::Tensor>& argument_tensors,
    const std::string& print_config,
    const std::vector<int64_t>& donate_args) {  // INT_VEC_OK
  TT_ASSIGN_OR_THROW(ContextedModule module,
                     ContextedModule::Make([&](mlir::MLIRContext& context) {
                       return ExtractMlirFromGraph(context, argument_tensors,
                                                   result_tensors, donate_args);
                     }));

  mlir::BaseScopedDiagnosticHandler diag_handler(&module.context());
  if (failed(mlir::verify(module.get()))) {
    TT_THROW_IF_ERROR(
        TT_ERROR(error::kInternal)
        << "MLIR module is invalid:\n"
        << diag_handler.ConsumeStatus().message() << "\n"
        << DebugString(module.get(), DebugStringOptions::kEnableDebugInfo));
  }
  std::string mlir_str =
      PrintMlirModule(module.get(), PyToMlirPrintConfig(print_config));
  return py::bytes(mlir_str.data(), mlir_str.size());
}

// Parses a string containing MLIR, verifies it, and serializes it to bytecode.
// This function will throw a RuntimeError in Python if:
//  1. The input mlir_text is not syntactically valid MLIR.
//  2. The MLIR is semantically invalid (e.g., type mismatches) and fails
//     verification.
//  3. An internal error occurs during the final bytecode serialization.
py::bytes PySerializeMlirTextModule(const std::string& mlir_text) {
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

  std::string bytecode_str;
  llvm::raw_string_ostream os(bytecode_str);
  if (mlir::failed(mlir::writeBytecodeToFile(module.get(), os))) {
    TT_THROW_IF_ERROR(TT_ERROR(error::kInternal)
                      << "Failed to serialize MLIR module to bytecode.");
  }

  return py::bytes(bytecode_str);
}

// Compiles an MLIR module.
// Args:
//   mlir_module: The serialized MLIR module.
//   eager: If true, use the compiler profile optimized for eager execution;
//          otherwise, use the profile optimized for torch.compile.
// Returns:
//   The compiled executable.
SharedLoadedExecutable PyCompileMlir(py::bytes& mlir_module,
                                     const bool fast_compile) {
  const auto module_bytecode = py::cast<std::string_view>(mlir_module);
  ScopedPythonContextCapturer capturer(OpName::kCompileMlir);
  // Provide the current python context to the compilation function so that
  // it can generate readable Mlir module names.
  ScopedPythonContextProvider provider(
      ScopedPythonContextCapturer::GetContext());
  TT_ASSIGN_OR_THROW(
      auto executable,
      CompileMlirExecutable(module_bytecode,
                            fast_compile ? CompilationMode::kFastCompile
                                         : CompilationMode::kFastRuntime));
  return executable;
}

// Wraps ExecuteCompiledModel() and builds the result types from the PJRT
// executable.
//
// This method does not support bounded dynamic programs. A separate API will
// need to be introduced which allows explicitly setting the requested output
// shapes.
std::vector<at::Tensor> PyExecuteCompiledModel(
    const SharedLoadedExecutable& executable,
    const std::vector<at::Tensor>& argument_tensors) {
  // Get the flattened output shapes from the executable.
  // Executables can return tuples, we want individual tensor shapes.
  TT_ASSIGN_OR_THROW(std::vector<xla::Shape> output_shapes_vec,
                     executable->GetOutputShapes());
  std::vector<const xla::Shape*> output_shapes_flat_vec;
  for (const auto& output_shape : output_shapes_vec) {
    xla::ShapeUtil::FlattenTupleShape(output_shape, output_shapes_flat_vec);
  }

  // Determine output shapes from the executable.
  std::vector<Shape> result_shapes_vec;
  result_shapes_vec.reserve(output_shapes_flat_vec.size());
  for (const xla::Shape* output_shape : output_shapes_flat_vec) {
    TT_ASSIGN_OR_THROW(Shape result_shape, MakeShape(*output_shape));
    result_shapes_vec.push_back(result_shape);
  }
  return ExecuteCompiledModel(executable, argument_tensors, result_shapes_vec);
}

bool PyGetMlirTracebacksEnabled() {
  return GetTracebackMode() == TracebackMode::kEnabled;
}

void PySetMlirTracebacksEnabled(bool enabled) {
  SetTracebackMode(enabled ? TracebackMode::kEnabled
                           : TracebackMode::kDisabled);
}

}  // namespace

py::bytes PySerializeExecutable(const SharedLoadedExecutable& executable) {
  TT_ASSIGN_OR_THROW(const std::string serialized,
                     executable->GetExecutable()->SerializeExecutable(),
                     _.SetPrepend() << "Failed to serialize executable: ");
  return py::bytes(serialized);
}

SharedLoadedExecutable PyLoadSerializedExecutable(py::bytes& serialized_bytes) {
  const auto data = py::cast<std::string_view>(serialized_bytes);
  xla::PjRtClient* const client = GetPjRtClient();
  TT_CHECK_THROW(client != nullptr, error::kFailedPrecondition)
      << "PjRtClient must be initialized before loading a serialized "
         "executable.";
  TT_ASSIGN_OR_THROW(SharedLoadedExecutable executable,
                     client->LoadSerializedExecutable(
                         data, /*options=*/std::nullopt, xla::LoadOptions()),
                     _.SetPrepend()
                         << "Failed to load serialized executable: ");
  return executable;
}

PYBIND11_MODULE(tpu_torch_compile, m) {
  py::class_<xla::PjRtLoadedExecutable,  // NOLINT(bugprone-unused-raii)
             std::shared_ptr<xla::PjRtLoadedExecutable>>(
      m, "PjRtLoadedExecutable");
  m.def("execute", PyExecuteCompiledModel,
        // Type: PjRtLoadedExecutable
        py::arg("executable"), py::arg("argument_tensors"));
  m.def("placeholder", PyMakePlaceholder, py::arg("sizes"), py::arg("dtype"),
        py::arg("requires_grad"));
  m.def("placeholder_like", PyMakePlaceholderLike, py::arg("arg_tensor"));
  m.def("build_mlir", PyExtractMlirModule, py::arg("result_tensors"),
        py::arg("argument_tensors"), py::arg("print_config") = "MlirPretty",
        py::arg("donate_args") = std::vector<int64_t>());  // INT_VEC_OK
  // Returns: PjRtLoadedExecutable
  m.def("compile_mlir", PyCompileMlir, py::arg("mlir_module_bytecode"),
        py::arg("fast_compile") = false);
  m.def("serialize_mlir_text", PySerializeMlirTextModule, py::arg("mlir_text"),
        "Parses a StableHLO MLIR text string and returns its serialized "
        "bytecode.");
  m.def("get_mlir_tracebacks_enabled", &PyGetMlirTracebacksEnabled,
        "Return whether MLIR location tracebacks are currently enabled.");
  m.def(
      "set_mlir_tracebacks_enabled", &PySetMlirTracebacksEnabled,
      py::arg("enabled"),
      "Sets whether MLIR location tracebacks should be captured with each op.");
  m.def("serialize_executable", PySerializeExecutable, py::arg("executable"),
        "Serializes a PjRtLoadedExecutable to bytes for caching.");
  m.def("load_serialized_executable", PyLoadSerializedExecutable,
        py::arg("serialized_bytes"),
        "Loads a PjRtLoadedExecutable from serialized bytes.");
}

}  // namespace torch_tpu
