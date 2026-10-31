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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "ATen/core/TensorBody.h"
#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "c10/core/DeviceGuard.h"
#include "c10/util/Optional.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_cache.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/context_manager.h"
#include "csrc/common/context_states.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/pybind_error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/to_string.h"
#include "csrc/common/tpu_subslice_utils.h"
#include "csrc/common/utils.h"
#include "csrc/distributed/handshake.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_gen_impl.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/internal/compile/compiled_mode.h"
#include "csrc/internal/compile/dispatch_scan.h"
#include "csrc/internal/dynamism/dynamism_ops.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "csrc/ops/scan_builder.h"
#include "csrc/ops/view_decomposition/contiguous_to_view.h"
#include "csrc/ops/view_decomposition/decomposition.h"
#include "csrc/ops/view_decomposition/strided_layout.h"
#include "csrc/pjrt/pjrt_state.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/csrc/Generator.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/extension.h"  // IWYU pragma: keep for aten::Tensor pybind type
#include "torch/headeronly/core/ScalarType.h"
#include "tsl/platform/fingerprint.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_device_dimensions.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/service/device_assignment.h"

namespace torch_tpu {
namespace py = pybind11;

namespace {

struct TensorInfo {
  std::vector<int64_t> shape;  // INT_VEC_OK
  at::ScalarType dtype;
};

struct TensorBounds {
  std::vector<int64_t> dynamic_dims;  // INT_VEC_OK
  std::vector<int64_t> upper_bounds;  // INT_VEC_OK
};

using PyLayout =
    std::tuple<std::vector<int64_t>,               // minor_to_major, INT_VEC_OK
               std::vector<std::vector<int64_t>>,  // tiles, INT_VEC_OK
               int64_t>;                           // element_size_in_bits

bool IsValidLayout(const std::vector<int64_t>& layout,  // INT_VEC_OK
                   int64_t rank) {
  if (layout.size() != rank) {
    return false;
  }
  std::vector<bool> seen(rank, false);
  for (int64_t dim : layout) {
    if (dim < 0 || dim >= rank) {
      return false;
    }
    if (seen[dim]) {
      return false;
    }
    seen[dim] = true;
  }
  return true;
}

void ParseTupleLayout(const py::tuple& t, CustomLayout& cl,
                      std::vector<int64_t>& minor_to_major) {  // INT_VEC_OK
  if (!t.empty() && !t[0].is_none()) {
    minor_to_major = t[0].cast<std::vector<int64_t>>();  // INT_VEC_OK
  }
  if (t.size() >= 2 && !t[1].is_none()) {
    auto tiles = t[1].cast<std::vector<std::vector<int64_t>>>();  // INT_VEC_OK
    for (const auto& tile : tiles) {
      cl.tiles.push_back(CopyIntVector(tile));
    }
  }
  if (t.size() >= 3 && !t[2].is_none()) {
    cl.element_size_in_bits = t[2].cast<int64_t>();
  }
}

CustomLayout ParseCustomLayout(const py::object& obj, const at::Tensor& tensor,
                               size_t arg_index) {
  CustomLayout cl;
  std::vector<int64_t> minor_to_major;  // INT_VEC_OK

  // Layouts from Python can be specified as:
  // 1. A tuple: (minor_to_major, tiles, [element_size_in_bits])
  // 2. A legacy list: [minor_to_major]
  if (py::isinstance<py::tuple>(obj)) {
    ParseTupleLayout(obj.cast<py::tuple>(), cl, minor_to_major);
  } else if (py::isinstance<py::list>(obj)) {
    minor_to_major = obj.cast<std::vector<int64_t>>();  // INT_VEC_OK
  }

  if (!minor_to_major.empty()) {
    int64_t rank = tensor.dim();
    TT_CHECK_THROW(IsValidLayout(minor_to_major, rank), error::kInvalidArgument)
        << "expected valid layout for argument " << arg_index << ", got layout "
        << ToString(minor_to_major) << " for shape "
        << ToString(tensor.sizes());
    cl.minor_to_major = CopyIntVector(minor_to_major);
  }

  return cl;
}

bool PyIsPinnedHost(const at::Tensor& tensor) {
  if (!tensor.has_storage()) {
    return false;
  }
  const auto buffer_ref = GetBuffer(tensor);
  return buffer_ref.ok() && buffer_ref->is_pinned_host();
}

at::Tensor PyMakePlaceholder(const std::vector<int64_t>& sizes,  // INT_VEC_OK
                             at::ScalarType dtype, bool requires_grad,
                             bool is_pinned_host) {
  TT_ASSIGN_OR_THROW(
      const at::Tensor prepared_tensor,
      MakePlaceholder(sizes, dtype, requires_grad, is_pinned_host),
      _.SetPrepend() << "failed to create placeholder tensor: ");
  return prepared_tensor;
}

at::Tensor PyMakeDynamicPlaceholder(
    const std::vector<int64_t>& sizes,  // INT_VEC_OK
    at::ScalarType dtype, const TensorBounds& bounds, bool requires_grad,
    bool is_pinned_host) {
  TT_ASSIGN_OR_THROW(const mlir::ElementType element_type,
                     ConvertTo<mlir::ElementType>(dtype));
  Shape shape(CopyIntVector(sizes), element_type);
  shape.dynamic_dimensions().reserve(bounds.dynamic_dims.size());
  for (size_t i = 0; i < bounds.dynamic_dims.size(); ++i) {
    shape.dynamic_dimensions().push_back(
        BoundedDynamicDimension{.dimension = bounds.dynamic_dims[i],
                                .lower_bound = 2,  // Default lower bound
                                .upper_bound = bounds.upper_bounds[i]});
  }
  TT_ASSIGN_OR_THROW(
      const at::Tensor prepared_tensor,
      MakePlaceholder(std::move(shape), requires_grad, is_pinned_host),
      _.SetPrepend() << "failed to create dynamic placeholder tensor: ");
  return prepared_tensor;
}

at::Tensor PyMakePlaceholderLike(const at::Tensor& arg_tensor,
                                 bool is_pinned_host,
                                 const py::object& layout) {
  if (!is_pinned_host) {
    is_pinned_host = PyIsPinnedHost(arg_tensor);
  }

  // arg_tensor may be a view, but placeholder DeviceBufferRefs are always
  // interpreted as contiguous.
  // If we want to make an equivalent view, we need to create a contiguous base
  // tensor as a placeholder, and then take a view of it with the same striding
  // as the original view.

  // First, we get a base shape for the contiguous base tensor.
  // This is the smallest amount of data necessary to back the view.
  // If arg_tensor is already contiguous, then this will just be its shape.
  TT_ASSIGN_OR_THROW(
      const Dimensions minimal_base_sizes,
      GetContiguousBaseShape(StridedLayout::FromTensor(arg_tensor)));

  // Then, we create a contiguous placeholder tensor with this shape.
  TT_ASSIGN_OR_THROW(
      const at::Tensor base_tensor,
      MakePlaceholder(minimal_base_sizes, arg_tensor.scalar_type(),
                      /*requires_grad=*/false, is_pinned_host),
      _.SetPrepend() << "failed to create placeholder tensor: ");

  // If given, traced ops see the on-device layout of the argument.
  if (!layout.is_none()) {
    TT_ASSIGN_OR_THROW(const DeviceBufferRef buffer_ref,
                       GetBuffer(base_tensor));
    buffer_ref.set_layout(
        ParseCustomLayout(layout, base_tensor, /*arg_index=*/0));
  }

  // NOTE: This logic must match the logic from PrepareCompiledModeArguments in
  // compiled_mode.cc
  at::Tensor view_tensor;  // UNINITIALIZED_TENSOR_OK
  if (TensorHasTrivialLayout(arg_tensor)) {
    view_tensor = base_tensor;
  } else {
    // Finally, we create a view of the base tensor with the same striding as
    // the original view, and preserve the requires_grad property.
    view_tensor =
        base_tensor.as_strided(arg_tensor.sizes(), arg_tensor.strides(),
                               arg_tensor.sym_storage_offset().expect_int());
  }

  if (arg_tensor.requires_grad()) {
    view_tensor.requires_grad_(true);
  }

  return view_tensor;
}

PyLayout ToPyLayout(const xla::Layout& layout) {
  std::vector<int64_t>  // INT_VEC_OK
      minor_to_major(layout.minor_to_major().begin(),
                     layout.minor_to_major().end());
  std::vector<std::vector<int64_t>> tiles;  // INT_VEC_OK
  tiles.reserve(layout.tiles().size());
  for (const auto& tile : layout.tiles()) {
    tiles.push_back(std::vector<int64_t>  // INT_VEC_OK
                    (tile.dimensions().begin(), tile.dimensions().end()));
  }
  return std::make_tuple(minor_to_major, tiles, layout.element_size_in_bits());
}

std::vector<std::optional<PyLayout>> ToPyLayouts(
    const std::vector<std::shared_ptr<const xla::PjRtLayout>>& layouts) {
  std::vector<std::optional<PyLayout>> result;
  result.reserve(layouts.size());
  for (const auto& pjrt_layout : layouts) {
    if (pjrt_layout == nullptr) {
      result.push_back(std::nullopt);
      continue;
    }
    result.push_back(ToPyLayout(pjrt_layout->xla_layout()));
  }
  return result;
}

std::optional<PyLayout> PyGetDeviceLayoutIfMaterialized(
    const at::Tensor& tensor) {
  auto buffer_ref_or = GetBuffer(tensor);
  if (!buffer_ref_or.ok()) {
    TT_THROW_TT_ERROR_(buffer_ref_or.status(), TT_SOURCE_LOCATION);
  }
  DeviceBufferRef buffer_ref = std::move(buffer_ref_or).value();
  if (!buffer_ref.is_materializing()) {
    return std::nullopt;
  }

  auto pjrt_buffer_or = buffer_ref.AwaitBuffer();
  if (!pjrt_buffer_or.ok()) {
    TT_THROW_TT_ERROR_(pjrt_buffer_or.status(), TT_SOURCE_LOCATION);
  }
  auto* pjrt_buffer = std::move(pjrt_buffer_or).value();

  const auto pjrt_layout = pjrt_buffer->layout();
  if (!pjrt_layout) {
    return std::nullopt;
  }
  return ToPyLayout(pjrt_layout->xla_layout());
}

PyLayout PyGetDefaultLayout(at::ScalarType dtype,
                            const std::vector<int64_t>& shape) {  // INT_VEC_OK
  const absl::StatusOr<xla::PjRtClient* absl_nonnull> client_or =
      PjrtBackend::GetInstance().GetClient();
  if (!client_or.ok()) {
    ABSL_LOG_FIRST_N(WARNING, 1)
        << "PyGetDefaultLayout: PjRt backend unavailable: "
        << client_or.status();
    return PyLayout();
  }
  xla::PjRtClient& checked_client = **client_or;
  TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<xla::PrimitiveType>(dtype));

  TT_ASSIGN_OR_THROW(const auto& layout,
                     checked_client.GetDefaultLayout(element_type, shape));
  return ToPyLayout(layout);
}

bool PyIsDeviceShapeDynamic(const at::Tensor& tensor) {
  TT_ASSIGN_OR_THROW(DeviceBufferRef buffer_ref, GetBuffer(tensor));
  return buffer_ref.on_device_shape_is_dynamic();
}

std::vector<std::optional<PyLayout>> PyGetParameterLayouts(
    const xla::PjRtLoadedExecutable& executable) {
  TT_ASSIGN_OR_THROW(const auto& layouts, executable.GetParameterLayouts());
  return ToPyLayouts(layouts);
}

std::vector<std::optional<PyLayout>> PyGetOutputLayouts(
    const xla::PjRtLoadedExecutable& executable) {
  TT_ASSIGN_OR_THROW(const auto& layouts, executable.GetOutputLayouts());
  return ToPyLayouts(layouts);
}

std::vector<std::optional<PyLayout>> PyGetParameterLayoutsFromMetadata(
    const LoadedExecutableWithMetadata& executable_metadata) {
  const xla::PjRtLoadedExecutable* executable =
      executable_metadata.GetLoadedExecutable();
  return PyGetParameterLayouts(*executable);
}

std::vector<std::optional<PyLayout>> PyGetOutputLayoutsFromMetadata(
    const LoadedExecutableWithMetadata& executable_metadata) {
  const xla::PjRtLoadedExecutable* executable =
      executable_metadata.GetLoadedExecutable();
  return PyGetOutputLayouts(*executable);
}

// Returns a ContextedModule corresponding to the graph terminating at
// result_tensors and taking argument_tensors as inputs.
std::shared_ptr<ContextedModule> PyBuildMlir(
    const std::vector<at::Tensor>& result_tensors,
    const std::vector<at::Tensor>& argument_tensors,
    bool use_stablehlo_bounds) {  // INT_VEC_OK
  TT_ASSIGN_OR_THROW(
      ContextedModule module,
      ContextedModule::Make([&]  // REF_CAPTURE_OK=called synchronously
                            (mlir::MLIRContext& context) {
                              return ExtractMlirFromGraph(
                                  context, argument_tensors, result_tensors,
                                  use_stablehlo_bounds);
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
          [&]  // REF_CAPTURE_OK=called synchronously
          (mlir::MLIRContext& context)
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

// Check if the MLIR module contains SparseCore custom call operations.
// Check if the MLIR module contains SPMD sharding instructions.
bool HasSpmdShardingInstructions(mlir::ModuleOp module) {
  auto result = module.walk([](mlir::Operation* op) {
    for (const auto& attr : op->getAttrs()) {
      llvm::StringRef name = attr.getName().strref();
      if (name.contains("sharding")) {
        return mlir::WalkResult::interrupt();
      }
      if (auto string_attr =
              mlir::dyn_cast<mlir::StringAttr>(attr.getValue())) {
        llvm::StringRef val = string_attr.getValue();
        if (val.contains("mhlo.sharding") || val.contains("op_sharding")) {
          return mlir::WalkResult::interrupt();
        }
      }
    }
    return mlir::WalkResult::advance();
  });
  return result.wasInterrupted();
}

bool HasSparseCoreCustomCall(mlir::ModuleOp module) {
  auto result = module.walk([](mlir::Operation* op) {
    for (const auto& attr : op->getAttrs()) {
      if (auto string_attr =
              mlir::dyn_cast<mlir::StringAttr>(attr.getValue())) {
        llvm::StringRef val = string_attr.getValue();
        if (val.contains("SparseDenseMatmul") || val.contains("SparseGather") ||
            val.contains("SparseMapRow") || val.contains("CooToCsr")) {
          return mlir::WalkResult::interrupt();
        }
      }
    }
    return mlir::WalkResult::advance();
  });
  return result.wasInterrupted();
}

// Compiles an MLIR module.
// Args:
//   module: The ContextedModule to compile.
//   fast_compile: If true, use the compiler profile optimized for eager
//         execution; otherwise, use the profile optimized for
//         torch.compile.
//   argument_layouts: Optional layout of the input arguments
// Returns:
//   The compiled executable.
SharedLoadedExecutableWithMetadata PyCompileMlir(
    std::shared_ptr<ContextedModule> module, const bool fast_compile,
    const std::vector<std::vector<int64_t>>& argument_layouts) {  // INT_VEC_OK
  ScopedPythonContextCapturer capturer(OpName::kCompileMlir);
  // Provide the current python context to the compilation function so that
  // it can generate readable Mlir module names.
  ScopedPythonContextProvider provider(
      ScopedPythonContextCapturer::GetContext());

  const auto compilation_mode = fast_compile ? CompilationMode::kFastCompile
                                             : CompilationMode::kFastRuntime;
  auto compilation_spec = GetCompilationSpec(compilation_mode);

  // Force replicated compilation mode (num_replicas = num_devices,
  // num_partitions = 1) for multi-device SparseCore compilation. This prevents
  // SPMD partitioning compilation hangs and failures for table-parallel
  // asymmetric graphs (e.g. SparseCore lookups) without affecting other
  // SPMD/Pallas/Mosaic workloads.
  if (HasSparseCoreCustomCall(module->get()) &&
      !HasSpmdShardingInstructions(module->get())) {
    TT_ASSIGN_OR_THROW(const int num_devices,
                       PjrtBackend::GetInstance().GetGlobalDeviceCount());
    if (num_devices > 1) {
      xla::ExecutableBuildOptions& options =
          compilation_spec.xla_compile_options->executable_build_options;
      options.set_num_replicas(num_devices);
      options.set_num_partitions(1);
      xla::DeviceAssignment da(num_devices, 1);
      for (int idx = 0; idx < num_devices; ++idx) {
        da(idx, 0) = idx;
      }
      options.set_device_assignment(da);
      compilation_spec.compile_options_key =
          MakeCompileOptionsKey(*compilation_spec.xla_compile_options);
    }
  }

  if (!argument_layouts.empty()) {
    mlir::ModuleOp module_op = module->get();
    auto main_func = module_op.lookupSymbol<mlir::func::FuncOp>("main");
    TT_CHECK_THROW(main_func, error::kInvalidArgument)
        << "could not find 'main' function in MLIR module";

    auto arg_types = main_func.getArgumentTypes();
    TT_CHECK_THROW(arg_types.size() == argument_layouts.size(),
                   error::kInvalidArgument)
        << "number of argument layouts (" << argument_layouts.size()
        << ") does not match number of arguments in MLIR main function ("
        << arg_types.size() << ")";

    std::vector<xla::Shape> xla_arg_layouts;
    xla_arg_layouts.reserve(arg_types.size());

    for (size_t i = 0; i < arg_types.size(); ++i) {
      mlir::Type type = arg_types[i];
      xla::Shape shape = xla::TypeToShape(type);
      TT_CHECK_THROW(
          shape.element_type() != xla::PrimitiveType::PRIMITIVE_TYPE_INVALID,
          error::kInvalidArgument)
          << "failed to convert MLIR type to XLA shape";

      const std::vector<int64_t>& layout_indices =  // INT_VEC_OK
          argument_layouts[i];
      if (!layout_indices.empty()) {
        int64_t rank = shape.dimensions().size();
        TT_CHECK_THROW(IsValidLayout(layout_indices, rank),
                       error::kInvalidArgument)
            << "expected valid layout for argument " << i << ", got layout "
            << ToString(layout_indices) << " for shape "
            << ToString(shape.dimensions());
        shape.clear_layout();
        *shape.mutable_layout() = xla::LayoutUtil::MakeLayout(layout_indices);
      }
      xla_arg_layouts.push_back(shape);
    }
    compilation_spec.xla_compile_options->argument_layouts =
        std::move(xla_arg_layouts);
  }

  // PJRT compilation can take minutes. Release the GIL after all Python
  // context capture and MLIR inspection so independent modules can compile
  // concurrently from Python worker threads.
  py::gil_scoped_release release;
  TT_ASSIGN_OR_THROW(
      auto executable,
      CompileMlirExecutable(xla::MaybeOwningMlirModule(module->get()),
                            std::move(compilation_spec.xla_compile_options)));
  return executable;
}

std::vector<CustomLayout> ConvertArgumentLayouts(
    const std::vector<at::Tensor>& argument_tensors,
    std::optional<std::vector<py::object>> argument_layouts_opt) {
  if (!argument_layouts_opt.has_value() || argument_layouts_opt->empty()) {
    return {};
  }
  const std::vector<py::object>& argument_layouts = *argument_layouts_opt;
  TT_CHECK_THROW(argument_layouts.size() == argument_tensors.size(),
                 error::kInvalidArgument)
      << "expected number of argument_layouts to match the number of "
         "argument_tensors, got number of argument_layouts "
      << argument_layouts.size() << " and number of argument_tensors "
      << argument_tensors.size();

  std::vector<CustomLayout> converted_layouts;
  converted_layouts.reserve(argument_layouts.size());
  for (size_t i = 0; i < argument_layouts.size(); ++i) {
    converted_layouts.push_back(
        ParseCustomLayout(argument_layouts[i], argument_tensors[i], i));
  }
  return converted_layouts;
}

TraverseAndCompileOptions ConvertTraverseAndCompileOptions(
    const std::vector<at::Tensor>& argument_tensors, bool fast_compile,
    bool build_mlir_module, bool use_stablehlo_bounds,
    std::optional<std::vector<py::object>> argument_layouts_opt = std::nullopt,
    const std::vector<int64_t>& donated_inputs = {},  // INT_VEC_OK
    const std::optional<std::string>& module_name = std::nullopt) {
  return TraverseAndCompileOptions{
      .compilation_mode = fast_compile ? CompilationMode::kFastCompile
                                       : CompilationMode::kFastRuntime,
      .build_mlir_module = build_mlir_module,
      .use_stablehlo_bounds = use_stablehlo_bounds,
      .argument_layouts = ConvertArgumentLayouts(
          argument_tensors, std::move(argument_layouts_opt)),
      .donated_inputs = CopyIntVector(donated_inputs),
      .module_name = module_name,
  };
}

AsyncCompileResult PyAsyncTraverseAndCompile(
    const std::vector<at::Tensor>& result_tensors,
    const std::vector<at::Tensor>& argument_tensors, bool fast_compile = false,
    bool build_mlir_module = false, bool use_stablehlo_bounds = false,
    std::optional<std::vector<py::object>> argument_layouts_opt = std::nullopt,
    const std::vector<int64_t>& donated_inputs = {},  // INT_VEC_OK
    const std::optional<std::string>& module_name = std::nullopt) {
  // Capture the Python traceback while the GIL is held; symbolization calls
  // into CPython. Once captured, the context is a pure-C++ snapshot on a
  // thread-local stack, so the traversal and MLIR lowering below can read it
  // with the GIL released.
  ScopedPythonContextCapturer capturer(OpName::kCompileMlir);
  ScopedPythonContextProvider provider(
      ScopedPythonContextCapturer::GetContext());

  auto options = ConvertTraverseAndCompileOptions(
      argument_tensors, fast_compile, build_mlir_module, use_stablehlo_bounds,
      std::move(argument_layouts_opt), donated_inputs, module_name);

  absl::StatusOr<AsyncCompileResult> status_or;
  {
    // Release the GIL across traversal, MLIR lowering, and the cache query so
    // multiple Python threads can lower and enqueue compilations in parallel.
    py::gil_scoped_release release;
    status_or = TraverseAndCompile(result_tensors, argument_tensors, options);
  }
  TT_ASSIGN_OR_THROW(AsyncCompileResult result, std::move(status_or));
  return result;
}

CompileResult PyTraverseAndCompile(
    const std::vector<at::Tensor>& result_tensors,
    const std::vector<at::Tensor>& argument_tensors, bool fast_compile = false,
    bool build_mlir_module = false, bool use_stablehlo_bounds = false,
    std::optional<std::vector<py::object>> argument_layouts_opt = std::nullopt,
    const std::vector<int64_t>& donated_inputs = {},  // INT_VEC_OK
    const std::optional<std::string>& module_name = std::nullopt) {
  AsyncCompileResult async_result = PyAsyncTraverseAndCompile(
      result_tensors, argument_tensors, fast_compile, build_mlir_module,
      use_stablehlo_bounds, std::move(argument_layouts_opt), donated_inputs,
      module_name);

  absl::StatusOr<SharedLoadedExecutableWithMetadata> status_or;
  {
    py::gil_scoped_release release;
    status_or = async_result.Get();
  }
  TT_ASSIGN_OR_THROW(auto executable, std::move(status_or));
  return CompileResult{
      .module = async_result.module(),
      .executable = std::move(executable),
  };
}

void PyPushEnableTracebacks(std::optional<bool> enabled) {
  const EnableTracebacksContextState state =
      enabled.has_value()
          ? std::make_optional(enabled.value() ? TracebackMode::kEnabled
                                               : TracebackMode::kDisabled)
          : std::nullopt;
  PushContextState(state);
}

}  // namespace

// Returns the internal TPU RNG state tensor {seed, offset} from a generator.
at::Tensor PyGetDeviceStateTensor(const at::Generator& gen) {
  auto* device_gen = at::check_generator<DeviceGeneratorImplBase>(gen);
  return device_gen->DeviceStateTensor();
}

// Sets the internal TPU RNG state tensor {seed, offset} on a generator.
void PySetDeviceStateTensor(const at::Generator& gen, at::Tensor rng_state) {
  auto* device_gen = at::check_generator<DeviceGeneratorImplBase>(gen);
  TT_THROW_IF_ERROR(device_gen->SetDeviceStateTensor(std::move(rng_state)));
}

// Creates a new export-mode generator by cloning the state of an eager
// generator and wrapping its state tensor in a symbolic MLIR placeholder.
at::Generator PyMakeExportModeDeviceGenerator(const at::Generator& orig_gen) {
  // Ensure generator cloning, tensor state inspection, and placeholder
  // creation occur in the context of the generator's device when invoked
  // across threads or multi-device workflows.
  c10::DeviceGuard guard(orig_gen.device());
  auto gen = at::make_generator<ExportModeDeviceGeneratorImpl>(orig_gen);

  // Set placeholder tensor for export mode generator.
  const at::Tensor device_state_tensor = PyGetDeviceStateTensor(gen);
  at::Tensor placeholder_tensor = PyMakePlaceholderLike(
      device_state_tensor, /*is_pinned_host=*/false, /*layout=*/py::none());
  PySetDeviceStateTensor(gen, std::move(placeholder_tensor));

  return gen;
}

// Registers an export-mode generator override for GetDefaultDeviceGenerator().
void PySetExportModeDefaultDeviceGenerator(c10::optional<at::Generator> gen,
                                           const c10::DeviceIndex idx) {
  SetExportModeDefaultDeviceGenerator(std::move(gen), idx);
}

// Clears all registered export-mode default generator overrides.
void PyClearExportModeDefaultDeviceGenerators() {
  ClearExportModeDefaultDeviceGenerators();
}

namespace {
std::vector<Shape> MakePadDynamicShapes(
    const std::vector<TensorInfo>& tensor_info,
    const std::vector<TensorBounds>& bounds_list) {
  std::vector<Shape> dynamic_shapes;
  dynamic_shapes.reserve(tensor_info.size());
  for (int i = 0; i < tensor_info.size(); ++i) {
    const auto& dynamic_dims = bounds_list[i].dynamic_dims;
    const auto& upper_bounds = bounds_list[i].upper_bounds;
    Dimensions dims = CopyIntVector(tensor_info[i].shape);
    TT_ASSIGN_OR_THROW(mlir::ElementType element_type,
                       ConvertTo<mlir::ElementType>(tensor_info[i].dtype));
    Shape shape(dims, element_type);

    TT_CHECK_THROW(dynamic_dims.size() == upper_bounds.size(),
                   error::kInvalidArgument)
        << "expected dimension indices and upper bounds to have the same size, "
           "got "
        << dynamic_dims.size() << " and " << upper_bounds.size();

    for (int j = 0; j < dynamic_dims.size(); ++j) {
      const int64_t dim_index = dynamic_dims[j];
      const int64_t dim_upper_bound = upper_bounds[j];

      TT_CHECK_THROW(dim_index >= 0 && dim_index < dims.size(),
                     error::kInvalidArgument)
          << "expected dimension index to be within bounds [0, "
          << dims.size() - 1 << "], got " << dim_index << " for input tensor "
          << i << " with shape " << ToString(dims);

      const int64_t dim_size = dims[dim_index];
      TT_CHECK_THROW(dim_upper_bound >= dim_size, error::kInvalidArgument)
          << "expected upper bound to be >= the static shape's dimension size, "
             "got upper bound "
          << dim_upper_bound << " for dimension " << dim_index
          << " for input tensor " << i << " with shape " << ToString(dims);

      shape.dynamic_dimensions().push_back(BoundedDynamicDimension{
          .dimension = dim_index,
          .lower_bound = 2,  // Or shape_bounds[i].lower if available
          .upper_bound = dim_upper_bound});
    }
    dynamic_shapes.push_back(shape);
  }
  return dynamic_shapes;
}

std::vector<Shape> ToStaticShapes(const std::vector<Shape>& shapes) {
  std::vector<Shape> static_shapes;
  static_shapes.reserve(shapes.size());
  for (const auto& shape : shapes) {
    static_shapes.push_back({shape.dimensions(), shape.dtype()});
  }
  return static_shapes;
}

std::vector<Shape> ToPaddedShapes(const std::vector<Shape>& shapes) {
  std::vector<Shape> padded_shapes;
  padded_shapes.reserve(shapes.size());
  for (const auto& shape : shapes) {
    Dimensions padded_dimensions = shape.dimensions();
    for (const auto& dynamic_dim : shape.dynamic_dimensions()) {
      padded_dimensions[dynamic_dim.dimension] = dynamic_dim.upper_bound;
    }
    padded_shapes.push_back({padded_dimensions, shape.dtype()});
  }
  return padded_shapes;
}

struct SliceSubgraphInputs {
  // TODO: unda - Remove target_dims and padded_dims by reworking the
  // SliceModuleCacheKey and GetSliceModule function.
  std::vector<Dimensions> target_dims;
  std::vector<Dimensions> padded_dims;
  std::vector<mlir::ElementType> element_types;
  std::vector<Shape> input_shapes;
  std::vector<Shape> output_shapes;
};

SliceSubgraphInputs UnpackSliceInputs(
    const std::vector<std::vector<int64_t>>&  // INT_VEC_OK
        target_shapes,
    const std::vector<std::vector<int64_t>>&  // INT_VEC_OK
        padded_shapes,
    const std::vector<at::ScalarType>& input_scalar_types) {
  TT_CHECK_THROW(!target_shapes.empty(), error::kInvalidArgument)
      << "expected at least one target shape, got none";
  TT_CHECK_THROW(target_shapes.size() == padded_shapes.size(),
                 error::kInvalidArgument)
      << "expected target shapes and padded shapes to have the same size, got "
      << target_shapes.size() << " and " << padded_shapes.size();
  TT_CHECK_THROW(target_shapes.size() == input_scalar_types.size(),
                 error::kInvalidArgument)
      << "expected target shapes and input scalar types to have the same size, "
         "got "
      << target_shapes.size() << " and " << input_scalar_types.size();

  SliceSubgraphInputs inputs;
  const size_t num_tensors = target_shapes.size();
  inputs.target_dims.reserve(num_tensors);
  inputs.padded_dims.reserve(num_tensors);
  inputs.element_types.reserve(num_tensors);
  inputs.input_shapes.reserve(num_tensors);
  inputs.output_shapes.reserve(num_tensors);

  for (size_t i = 0; i < num_tensors; ++i) {
    const auto& target = target_shapes[i];
    const auto& padded = padded_shapes[i];
    TT_CHECK_THROW(target.size() == padded.size(), error::kInvalidArgument)
        << "expected target shape and padded shape to have the same number of "
           "dimensions, got "
        << target.size() << " and " << padded.size() << " for tensor index "
        << i;

    Dimensions target_dims = CopyIntVector(target);
    Dimensions padded_dims = CopyIntVector(padded);

    for (size_t j = 0; j < target.size(); ++j) {
      TT_CHECK_THROW(padded[j] >= target[j], error::kInvalidArgument)
          << "expected padded shape dimension sizes to be >= target shape "
             "dimension sizes, got padded shape "
          << ToString(padded_dims) << " and target shape "
          << ToString(target_dims) << " for tensor index " << i;
    }

    TT_ASSIGN_OR_THROW(mlir::ElementType element_type,
                       internal::ToElementType(input_scalar_types[i]));

    // Assemble both Shape vectors and primitive vectors in the same pass
    inputs.input_shapes.push_back(Shape(padded_dims, element_type));
    inputs.output_shapes.push_back(Shape(target_dims, element_type));

    inputs.target_dims.push_back(std::move(target_dims));
    inputs.padded_dims.push_back(std::move(padded_dims));
    inputs.element_types.push_back(element_type);
  }
  return inputs;
}

absl::StatusOr<CompileResult> CompileModuleWithCache(
    MlirComputationBuilder builder, const CompilationCacheKey& cache_key,
    const std::vector<Shape>& input_shapes,
    const std::vector<Shape>& output_shapes,
    UniqueCompileOptions compile_options, bool build_mlir_module,
    bool is_caching_disabled, bool compile_dynamic_adapter = false) {
  std::optional<ContextedModule> contexted_module;

  // Create the MLIR module before compilation, since GetOrCompile consumes the
  // builder.
  if (build_mlir_module || is_caching_disabled) {
    TT_ASSIGN_OR_RETURN(contexted_module, ContextedModule::Make(builder),
                        _ << "failed to build MLIR module");
  }

  if (is_caching_disabled) {
    TT_ASSIGN_OR_RETURN(SharedLoadedExecutableWithMetadata executable,
                        CompileMlirExecutable(
                            xla::MaybeOwningMlirModule(contexted_module->get()),
                            std::move(compile_options)));
    absl_nullable std::shared_ptr<ContextedModule> module = nullptr;
    if (build_mlir_module) {
      module = std::make_shared<ContextedModule>(std::move(*contexted_module));
    }
    return CompileResult{.module = std::move(module),
                         .executable = std::move(executable)};
  }

  TT_ASSIGN_OR_RETURN(
      CompiledKernel compiled_kernel,
      CompilationCache::GetInstance().GetOrCompile(
          cache_key, input_shapes, output_shapes, std::move(builder),
          std::move(compile_options), compile_dynamic_adapter));

  TT_ASSIGN_OR_RETURN(SharedLoadedExecutableWithMetadata executable,
                      compiled_kernel.fixed_shape_kernel.get(),
                      _.SetPrepend() << "failed to get fixed shape kernel: ");

  absl_nullable std::shared_ptr<ContextedModule> module = nullptr;
  if (build_mlir_module) {
    module = std::make_shared<ContextedModule>(std::move(*contexted_module));
  }
  return CompileResult{.module = std::move(module),
                       .executable = std::move(executable)};
}

}  // namespace
// Precompiles the pad subgraph for the given shapes. Doesn't wait for the
// compilation to complete.
//
// Args:
//   tensor_info: A list of pairs, where each pair contains the shape of a
//     tensor and its scalar type.
//   bounds_list: A list of pairs, where each pair contains the dynamic
//     dimensions of a tensor and their upper bounds.
//   fast_compile: If true, use the compiler profile optimized for eager
//         execution; otherwise, use the profile optimized for
//         torch.compile.
void PyPrecompilePadModule(const std::vector<TensorInfo>& tensor_info,
                           const std::vector<TensorBounds>& bounds_list,
                           const bool fast_compile) {
  std::vector<Shape> dynamic_shapes =
      MakePadDynamicShapes(tensor_info, bounds_list);

  MlirComputationBuilder pad_builder =
      [&]  // REF_CAPTURE_OK=called synchronously
      (mlir::MLIRContext& mlir_context) {
        return GetPadModule(mlir_context, dynamic_shapes,
                            /*pad_only_module=*/true);
      };

  const auto compilation_mode = fast_compile ? CompilationMode::kFastCompile
                                             : CompilationMode::kFastRuntime;
  CompilationSpec compilation_spec = GetCompilationSpec(compilation_mode);

  const CompilationCacheKey cache_key(
      /*graph_key=*/PadModuleCacheKey(dynamic_shapes, /*pad_only_module=*/true),
      compilation_spec.compile_options_key);

  std::vector<Shape> runtime_input_shapes = ToStaticShapes(dynamic_shapes);
  std::vector<Shape> padded_input_shapes = ToPaddedShapes(dynamic_shapes);
  TT_ASSIGN_OR_THROW(CompiledKernel compiled_pad,
                     CompilationCache::GetInstance().GetOrCompile(
                         cache_key, runtime_input_shapes, padded_input_shapes,
                         std::move(pad_builder),
                         std::move(compilation_spec.xla_compile_options)));
}

// Returns a CompileResult containing the executable and optionally the MLIR
// module for a pad subgraph.
// Args:
//   tensor_info: A list of pairs, where each pair contains the shape of a
//     tensor and its scalar type.
//   bounds_list: A list of pairs, where each pair contains the dynamic
//     dimensions of a tensor and their upper bounds.
//   fast_compile: If true, use the compiler profile optimized for eager
//         execution; otherwise, use the profile optimized for
//         torch.compile.
//   build_mlir_module: If true, populate the module field in the CompileResult
//     with the MLIR module for the pad subgraph.
//   is_caching_disabled: If true, do not use the cache.
// Returns:
//   A CompileResult containing the executable and optionally the MLIR module
//   for the pad subgraph.
// Example:
//   tensor_info = [([1, 4], torch.int64)]
//   bounds_list = [([1], [8])]. // Dynamic dimension is 1, upper bound is 8.
//  MLIR module:
// module @pad_module {
//   func.func @main(%arg0: tensor<1x4xi64>) -> tensor<1x8xi64> {
//     %c = stablehlo.constant dense<0> : tensor<i64>
//     %0 = stablehlo.pad %arg0, %c, low = [0, 0], high = [0, 4],
//       interior = [0, 0] : (tensor<1x4xi64>, tensor<i64>) -> tensor<1x8xi64>
//     return %0 : tensor<1x8xi64>
//   }
// }
CompileResult PyGetOrCompilePadModule(
    const std::vector<TensorInfo>& tensor_info,
    const std::vector<TensorBounds>& bounds_list, const bool fast_compile,
    const bool build_mlir_module, const bool is_caching_disabled) {
  std::vector<Shape> dynamic_shapes =
      MakePadDynamicShapes(tensor_info, bounds_list);

  MlirComputationBuilder pad_builder =
      [&]  // REF_CAPTURE_OK=called synchronously
      (mlir::MLIRContext& mlir_context) {
        return GetPadModule(mlir_context, dynamic_shapes,
                            /*pad_only_module=*/true);
      };

  const auto compilation_mode = fast_compile ? CompilationMode::kFastCompile
                                             : CompilationMode::kFastRuntime;
  CompilationSpec compilation_spec = GetCompilationSpec(compilation_mode);

  const CompilationCacheKey cache_key(
      /*graph_key=*/PadModuleCacheKey(dynamic_shapes, /*pad_only_module=*/true),
      /*compile_options_key=*/compilation_spec.compile_options_key);

  std::vector<Shape> runtime_input_shapes = ToStaticShapes(dynamic_shapes);
  std::vector<Shape> padded_input_shapes = ToPaddedShapes(dynamic_shapes);

  TT_ASSIGN_OR_THROW(
      CompileResult result,
      CompileModuleWithCache(std::move(pad_builder), cache_key,
                             runtime_input_shapes, padded_input_shapes,
                             std::move(compilation_spec.xla_compile_options),
                             build_mlir_module, is_caching_disabled));
  return result;
}

// Returns a CompileResult containing the executable and the MLIR
// module for a pad subgraph.
// Args:
//   tensor_info: A list of pairs, where each pair contains the shape of a
//     tensor and its scalar type.
//   bounds_list: A list of pairs, where each pair contains the dynamic
//     dimensions of a tensor and their upper bounds.
//   fast_compile: If true, use the compiler profile optimized for eager
//         execution; otherwise, use the profile optimized for
//         torch.compile.
// Returns:
//   A CompileResult containing the executable and optionally the MLIR module
//   for the pad subgraph.
// Example:
//   tensor_info = [([1, 4], torch.int64)]
//   bounds_list = [([1], [8])]. // Dynamic dimension is 1, upper bound is 8.
//  MLIR module:
// module @pad_module {
//   func.func @main(%arg0: tensor<1x4xi64>) -> tensor<1x?xi64,
//   #stablehlo.bounds<?, 8>> {
//     %c = stablehlo.constant dense<0> : tensor<i64>
//     %c1 = stablehlo.constant dense<1> : tensor<i64>
//     %0 = stablehlo.pad %arg0, %c, low = [0, 0], high = [0, 4],
//       interior = [0, 0] : (tensor<1x4xi64>, tensor<i64>) -> tensor<1x8xi64>
//     %1 = stablehlo.get_dimension_size %0, %c1 : (tensor<1x8xi64>) -> i64
//     %2 = stablehlo.set_dimension_size %0, %1, dim = 1 :
//       (tensor<1x8xi64>, tensor<i64>) ->
//        tensor<1x?xi64, #stablehlo.bounds<?,8>>
//     return %2
//   }
// }
CompileResult PyGetDynamicPadModule(
    const std::vector<TensorInfo>& tensor_info,
    const std::vector<TensorBounds>& bounds_list, const bool fast_compile) {
  std::vector<Shape> dynamic_shapes =
      MakePadDynamicShapes(tensor_info, bounds_list);

  MlirComputationBuilder pad_builder =
      [&]  // REF_CAPTURE_OK=called synchronously
      (mlir::MLIRContext& mlir_context) {
        return GetDynamicPadModule(mlir_context, dynamic_shapes);
      };

  const auto compilation_mode = fast_compile ? CompilationMode::kFastCompile
                                             : CompilationMode::kFastRuntime;
  CompilationSpec compilation_spec = GetCompilationSpec(compilation_mode);

  std::optional<ContextedModule> contexted_module;

  TT_ASSIGN_OR_THROW(contexted_module, ContextedModule::Make(pad_builder),
                     _ << "failed to build MLIR module");

  TT_ASSIGN_OR_THROW(
      SharedLoadedExecutableWithMetadata executable,
      CompileMlirExecutable(xla::MaybeOwningMlirModule(contexted_module->get()),
                            std::move(compilation_spec.xla_compile_options)));

  return CompileResult{
      .module = std::make_shared<ContextedModule>(std::move(*contexted_module)),
      .executable = std::move(executable)};
}

// Precompiles the slice subgraph for the given shapes. Doesn't wait for the
// compilation to complete.
//
// Args:
//   target_shapes: A list of target shapes.
//   padded_shapes: A list of padded shapes.
//   input_scalar_types: A list of scalar types for each tensor.
//   fast_compile: If true, use the compiler profile optimized for eager
//         execution; otherwise, use the profile optimized for
//         torch.compile.
void PyPrecompileSliceModule(
    const std::vector<std::vector<int64_t>>& target_shapes,  // INT_VEC_OK
    const std::vector<std::vector<int64_t>>& padded_shapes,  // INT_VEC_OK
    const std::vector<at::ScalarType>& input_scalar_types,
    const bool fast_compile) {
  SliceSubgraphInputs inputs =
      UnpackSliceInputs(target_shapes, padded_shapes, input_scalar_types);

  MlirComputationBuilder slice_builder =
      [&]  // REF_CAPTURE_OK=called synchronously
      (mlir::MLIRContext& mlir_context) {
        return GetSliceModule(mlir_context, inputs.target_dims,
                              inputs.padded_dims, inputs.element_types);
      };

  const auto compilation_mode = fast_compile ? CompilationMode::kFastCompile
                                             : CompilationMode::kFastRuntime;
  CompilationSpec compilation_spec = GetCompilationSpec(compilation_mode);

  const CompilationCacheKey cache_key(
      /*graph_key=*/SliceModuleCacheKey(inputs.target_dims, inputs.padded_dims,
                                        inputs.element_types),
      compilation_spec.compile_options_key);

  TT_ASSIGN_OR_THROW(CompiledKernel compiled_slice,
                     CompilationCache::GetInstance().GetOrCompile(
                         cache_key, inputs.input_shapes, inputs.output_shapes,
                         std::move(slice_builder),
                         std::move(compilation_spec.xla_compile_options)));
}

// Returns a CompileResult containing the executable and optionally the MLIR
// module for a slice subgraph.
//
// Args:
//   target_shapes: A list of target shapes.
//   padded_shapes: A list of padded shapes.
//   input_scalar_types: A list of scalar types for each tensor.
//   fast_compile: If true, use the compiler profile optimized for eager
//         execution; otherwise, use the profile optimized for
//         torch.compile.
//   build_mlir_module: If true, populate the module field in the CompileResult
//     with the MLIR module for the slice subgraph.
//   is_caching_disabled: If true, do not use the cache.
// Returns:
//   A CompileResult containing the executable and optionally the MLIR module
//   for the slice subgraph.
// Example:
//   target_shapes = [[1, 4]]
//   padded_shapes = [[1, 8]]
//   input_scalar_types = [torch.float32]
// Example MLIR module:
// module @slice_module {
//   func.func @main(%arg0: tensor<1x?xf32, #stablehlo.bounds<?, 8>>) ->
//   tensor<1x4xf32> {
//     %c = stablehlo.constant dense<8> : tensor<i32>
//     %0 = stablehlo.set_dimension_size %arg0, %c, dim = 1 :
//       (tensor<1x?xf32, #stablehlo.bounds<?, 8>>, tensor<i32>) ->
//        tensor<1x8xf32>
//     %1 = stablehlo.slice %0 [0:1, 0:4] : (tensor<1x8xf32>) -> tensor<1x4xf32>
//     return %1 : tensor<1x4xf32>
//   }
// }
CompileResult PyGetOrCompileSliceModule(
    const std::vector<std::vector<int64_t>>& target_shapes,  // INT_VEC_OK
    const std::vector<std::vector<int64_t>>& padded_shapes,  // INT_VEC_OK
    const std::vector<at::ScalarType>& input_scalar_types,
    const bool fast_compile, const bool build_mlir_module,
    const bool is_caching_disabled) {
  SliceSubgraphInputs inputs =
      UnpackSliceInputs(target_shapes, padded_shapes, input_scalar_types);

  MlirComputationBuilder slice_builder =
      [&]  // REF_CAPTURE_OK=called synchronously
      (mlir::MLIRContext& mlir_context) {
        return GetSliceModule(mlir_context, inputs.target_dims,
                              inputs.padded_dims, inputs.element_types);
      };

  // For slice, the inputs to the compiled module are the padded shapes, and
  // outputs are target shapes.
  const std::vector<Shape>& runtime_input_shapes = inputs.input_shapes;
  const std::vector<Shape>& cache_output_shapes = inputs.output_shapes;

  const auto compilation_mode = fast_compile ? CompilationMode::kFastCompile
                                             : CompilationMode::kFastRuntime;
  CompilationSpec compilation_spec = GetCompilationSpec(compilation_mode);

  const CompilationCacheKey cache_key(
      /*graph_key=*/SliceModuleCacheKey(inputs.target_dims, inputs.padded_dims,
                                        inputs.element_types),
      compilation_spec.compile_options_key);

  TT_ASSIGN_OR_THROW(
      CompileResult result,
      CompileModuleWithCache(std::move(slice_builder), cache_key,
                             runtime_input_shapes, cache_output_shapes,
                             std::move(compilation_spec.xla_compile_options),
                             build_mlir_module, is_caching_disabled));
  return result;
}

py::bytes PySerializeExecutable(
    const SharedLoadedExecutableWithMetadata& executable) {
  TT_ASSIGN_OR_THROW(const std::string serialized,
                     executable->GetExecutable()->SerializeExecutable(),
                     _.SetPrepend() << "failed to serialize executable: ");
  return py::bytes(serialized);
}

absl::StatusOr<SharedLoadedExecutableWithMetadata> LoadSerializedExecutableImpl(
    std::string_view data,
    const std::optional<std::vector<int32_t>>& subslice_origin) {
  TT_ASSIGN_OR_RETURN(xla::PjRtClient* const client,
                      PjrtBackend::GetInstance().GetClient());
  // `subslice_origin` comes from the pickle (see
  // LoadedExecutableWithMetadata::subslice_origin).
  std::optional<xla::PjRtDeviceDimensions> origin;
  if (subslice_origin.has_value()) {
    origin.emplace(absl::MakeConstSpan(*subslice_origin));
  }
  TT_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtLoadedExecutable> pjrt_executable,
      AdaptXlaError(LoadSerializedExecutableAtOrigin(*client, data, origin),
                    /* context= */ "failed to load serialized executable"));
  return LoadedExecutableWithMetadata::MakeShared(
      std::move(pjrt_executable), /*argument_indices=*/{},
      /*unbounded_dynamic_execution_metadata=*/std::nullopt, std::move(origin));
}

SharedLoadedExecutableWithMetadata PyLoadSerializedExecutable(
    py::bytes& serialized_bytes,
    const std::optional<std::vector<int32_t>>& subslice_origin) {
  const auto data = py::cast<std::string_view>(serialized_bytes);
  TT_ASSIGN_OR_THROW(SharedLoadedExecutableWithMetadata executable,
                     LoadSerializedExecutableImpl(data, subslice_origin),
                     _.SetPrepend() << "failed to load executable: ");
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

bool PyIsConstantTensor(const at::Tensor& tensor) {
  if (!tensor.defined() ||
      tensor.device().type() != GetPrivateUse1DeviceType()) {
    return false;
  }
  auto buffer_ref_or = GetBuffer(tensor);
  if (!buffer_ref_or.ok()) {
    return false;
  }
  auto deferred_op = buffer_ref_or->deferred_op();
  return deferred_op != nullptr &&
         deferred_op->op_name() == OpName::kTorchTpuInternalConstant;
}

// Returns whether `storage_offset` is zero and every element of a tensor with
// the given layout maps to the same memory offset as in the standard contiguous
// layout of `shape`, i.e. the strides differ at most along size-1 dimensions
// (or `shape` has no elements).
bool IsPhysicallyContiguousLayout(at::IntArrayRef shape, at::IntArrayRef stride,
                                  int64_t storage_offset) {
  if (storage_offset != 0 || shape.size() != stride.size()) {
    return false;
  }
  if (absl::c_linear_search(shape, 0)) {
    return true;
  }
  int64_t expected_stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    if (shape[i] > 1 && stride[i] != expected_stride) {
      return false;
    }
    expected_stride *= std::max<int64_t>(shape[i], 1);
  }
  return true;
}

at::Tensor ForceStrides(const at::Tensor& tensor,
                        const Dimensions& target_strides,
                        int64_t target_storage_offset) {
  TT_CHECK_THROW(tensor.dim() == target_strides.size(), error::kInvalidArgument)
      << "target strides must have the same number of dimensions "
      << "as the tensor, got " << tensor.ndimension() << " and "
      << target_strides.size();
  TT_CHECK_THROW(target_storage_offset >= 0, error::kInvalidArgument)
      << "target_storage_offset must be non-negative";

  const at::IntArrayRef target_strides_ref(target_strides.data(),
                                           target_strides.size());
  if (tensor.strides() == target_strides_ref &&
      tensor.storage_offset() == target_storage_offset) {
    return tensor;
  }

  // Fast path: if the tensor is already contiguous with zero storage offset and
  // the target layout only differs in strides along size-1 dimensions (e.g.
  // shape (1, 32, 512) with stride (512, 512, 1)), every element maps to the
  // exact same physical memory address. Update TensorImpl stride metadata on a
  // shallow alias without allocating a new device buffer or launching a TPU
  // copy kernel via ContiguousToView.
  if (target_storage_offset == 0 && tensor.storage_offset() == 0 &&
      tensor.is_contiguous() &&
      IsPhysicallyContiguousLayout(tensor.sizes(), target_strides_ref,
                                   target_storage_offset)) {
    at::Tensor view = tensor.alias();
    view.unsafeGetTensorImpl()->set_sizes_and_strides(tensor.sizes(),
                                                      target_strides_ref, 0);
    return view;
  }

  TT_ASSIGN_OR_THROW(DeviceBufferRef contiguous_buffer, GetBuffer(tensor));

  ScopedPythonContextCapturer capturer(OpName::kForceStrides);
  TT_ASSIGN_OR_THROW(at::Tensor view_tensor,
                     ContiguousToView(std::move(contiguous_buffer),
                                      target_strides, target_storage_offset));
  return view_tensor;
}

at::Tensor PyForceStrides(
    const at::Tensor& tensor,
    const std::vector<int64_t>& target_strides,  // INT_VEC_OK
    int64_t target_storage_offset) {
  return ForceStrides(tensor, CopyIntVector(target_strides),
                      target_storage_offset);
}

// A context manager for locking multiple generators' mutexes.
//
// This is necessary to prevent generator state conflicts in between getting the
// device state tensor, executing compiled executable, and setting the updated
// device state tensor.
//
// See Note [Acquire lock when using random generators].
class MultiGeneratorLocker {
 public:
  // Constructs a MultiGeneratorLocker.
  // Extracts mutexes from generators, sorts them by address to prevent
  // deadlocks, and removes duplicates to prevent self-deadlocking.
  explicit MultiGeneratorLocker(std::vector<at::Generator>& generators) {
    // Extract the underlying std::mutex pointers
    for (auto& gen : generators) {
      sorted_unique_mutexes_.push_back(&gen.mutex());
    }

    // Sort by memory address to prevent AB-BA deadlocks across threads.
    // If Thread 1 locks [A, B] and Thread 2 locks [B, A], sorting ensures
    // they both lock A first, then B.
    std::sort(sorted_unique_mutexes_.begin(), sorted_unique_mutexes_.end());

    // Remove duplicates to prevent self-deadlocking.
    // PyTorch uses std::mutex (which is non-recursive). Locking it twice
    // on the same thread is Undefined Behavior and will freeze the process.
    sorted_unique_mutexes_.erase(std::unique(sorted_unique_mutexes_.begin(),
                                             sorted_unique_mutexes_.end()),
                                 sorted_unique_mutexes_.end());
  }

  MultiGeneratorLocker(const MultiGeneratorLocker&) = delete;
  MultiGeneratorLocker& operator=(const MultiGeneratorLocker&) = delete;
  MultiGeneratorLocker(MultiGeneratorLocker&&) = delete;
  MultiGeneratorLocker& operator=(MultiGeneratorLocker&&) = delete;

  // Enters the context.
  // Releases the GIL while waiting for the locks to avoid deadlocking
  // the Python interpreter, then acquires locks for all mutexes. No double
  // entering is allowed to prevent self-deadlocking.
  void Enter() {
    // Release the GIL while waiting for the locks to avoid deadlocking
    // the Python interpreter.
    pybind11::gil_scoped_release release;
    ABSL_CHECK(  // CRASH_OK
        locks_.empty())
        << "double entering MultiGeneratorLocker is not allowed";
    locks_.reserve(sorted_unique_mutexes_.size());
    for (auto* m : sorted_unique_mutexes_) {
      locks_.emplace_back(*m);
    }
  }

  // Exits the context.
  // Releases all locks by clearing the locks_ vector.
  void Exit(pybind11::handle exc_type, pybind11::handle exc_val,
            pybind11::handle exc_tb) {
    locks_.clear();
  }

 private:
  // The generators' sorted unique mutexes to lock in between Enter and Exit.
  // - `sorted` to ensure consistent locking order and prevent deadlocks.
  // - `unique` to deduplicate generators and prevent self-deadlocking.
  std::vector<std::mutex* absl_nonnull> sorted_unique_mutexes_;  // NOLINT
  std::vector<std::unique_lock<std::mutex>> locks_;              //  NOLINT
};

// Reconstructs the original FX graph output structure from flattened MLIR
// outputs.
//
// In compiled mode, FX graphs return nested Python structures (e.g., tuples,
// lists, dicts, or single leaf tensors) where multiple output leaves may alias
// the same deduplicated tensor output from the compiled PJRT executable, pass
// through an input argument directly, or evaluate to None. This class caches
// the output schema at compile time so that runtime execution can reconstruct
// the FX output structure in C++ without PyTree/Python overhead.
using RawPyObjPtr = std::decay_t<decltype(py::handle().ptr())>;

// Returns `seq_ptr` directly when it is already an exact `tuple` or `list`, or
// materializes any other iterable into `fast_holder` via `PySequence_Fast` so
// callers can uniformly use `PySequence_Fast_*` macros.
RawPyObjPtr AsFastSequence(RawPyObjPtr seq_ptr, py::object* fast_holder) {
  if (PyTuple_CheckExact(seq_ptr) || PyList_CheckExact(seq_ptr)) {
    return seq_ptr;
  }
  RawPyObjPtr fast = PySequence_Fast(seq_ptr, "expected a sequence");
  if (!fast) {
    throw py::error_already_set();  // Propagate Python C-API error.
  }
  *fast_holder = py::reinterpret_steal<py::object>(fast);
  return fast;
}

class FxOutputReconstructor {
 public:
  // Specifies how a single leaf in the flattened FX output maps to runtime
  // values.
  enum class EntryType : uint8_t {
    kDedupedTensor,         // Indexes into `deduped_outputs`.
    kNonTensorPassthrough,  // Indexes into runtime `inputs` sequence.
    kNone,                  // Evaluates to Python `None`.
  };

  struct Entry {
    EntryType type;
    int index;
    bool needs_clone = false;
    bool can_move = false;
    bool needs_layout_check = true;
  };

  // Expected tensor layout metadata for each deduplicated output tensor.
  struct ExpectedTensorLayout {
    at::ScalarType dtype;
    Dimensions shape;
    Dimensions stride;
    int64_t storage_offset;
    bool is_dynamic = false;
    bool is_standard_contiguous = false;
    bool is_physically_contiguous = false;

    bool MatchesTensor(const at::Tensor& tensor) const {
      if (is_standard_contiguous && tensor.is_contiguous() &&
          tensor.storage_offset() == 0) {
        return tensor.scalar_type() == dtype &&
               tensor.sizes() == at::IntArrayRef(shape.data(), shape.size());
      }
      return tensor.scalar_type() == dtype &&
             tensor.storage_offset() == storage_offset &&
             tensor.sizes() == at::IntArrayRef(shape.data(), shape.size()) &&
             tensor.strides() == at::IntArrayRef(stride.data(), stride.size());
    }
  };

  static bool IsStandardContiguousLayout(const Dimensions& shape,
                                         const Dimensions& stride,
                                         int64_t storage_offset) {
    if (storage_offset != 0 || shape.size() != stride.size()) {
      return false;
    }
    int64_t expected_stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      if (stride[i] != expected_stride) {
        return false;
      }
      expected_stride *= std::max<int64_t>(shape[i], 1);
    }
    return true;
  }

  // Fast-path categories for PyTree unflattening.
  enum class SpecKind : uint8_t {
    kLeaf,       // Single leaf output (returns the leaf object directly).
    kFlatTuple,  // 1-level flat tuple of leaves (constructed via PyTuple_New).
    kFlatList,   // 1-level flat list of leaves (constructed via PyList_New).
    kGeneral,    // Arbitrary nested PyTree (delegates to spec.unflatten()).
  };

  static Dimensions ParseDimensions(py::handle seq) {
    Dimensions dims;
    for (py::handle val : seq) {
      dims.push_back(val.cast<int64_t>());
    }
    return dims;
  }

  static Entry ParseEntry(py::handle item) {
    if (item.is_none()) {
      return {EntryType::kNone, 0};
    }
    if (py::hasattr(item, "tensor_idx")) {
      return {EntryType::kDedupedTensor, item.attr("tensor_idx").cast<int>()};
    }
    if (py::hasattr(item, "input_idx")) {
      return {EntryType::kNonTensorPassthrough,
              item.attr("input_idx").cast<int>()};
    }
    throw py::value_error("unexpected item in output_map");  // For pybind11.
  }

  static SpecKind DetermineSpecKind(const py::object& spec_py) {
    if (spec_py.attr("is_leaf")().cast<bool>()) {
      return SpecKind::kLeaf;
    }
    const int num_nodes = spec_py.attr("num_nodes").cast<int>();
    const int num_leaves = spec_py.attr("num_leaves").cast<int>();
    const int num_children = spec_py.attr("num_children").cast<int>();
    // A PyTree is a 1-level flat container if it consists of exactly 1 root
    // container node plus `num_leaves` direct leaf children.
    if (num_nodes == num_leaves + 1 && num_children == num_leaves) {
      const py::object type_obj = spec_py.attr("type");
      if (type_obj.is(py::tuple().get_type())) {
        return SpecKind::kFlatTuple;
      }
      if (type_obj.is(py::list().get_type())) {
        return SpecKind::kFlatList;
      }
    }
    return SpecKind::kGeneral;
  }

  FxOutputReconstructor(py::object output_map, py::object expected_layouts,
                        py::object spec,
                        py::object dynamic_outputs = py::none())
      : spec_kind_(DetermineSpecKind(spec)),
        output_map_py_(std::move(output_map)),
        expected_layouts_py_(std::move(expected_layouts)),
        spec_py_(std::move(spec)),
        dynamic_outputs_py_(std::move(dynamic_outputs)) {
    for (py::handle item : output_map_py_) {
      entries_.push_back(ParseEntry(item));
    }

    absl::flat_hash_map<int, int> tensor_ref_counts;
    for (const Entry& entry : entries_) {
      if (entry.type == EntryType::kDedupedTensor) {
        tensor_ref_counts[entry.index]++;
      }
    }
    absl::flat_hash_set<int> seen_tensors;
    for (Entry& entry : entries_) {
      if (entry.type == EntryType::kDedupedTensor) {
        const bool is_first = seen_tensors.insert(entry.index).second;
        entry.needs_clone = !is_first;
        entry.can_move = is_first && (tensor_ref_counts[entry.index] == 1);
      }
    }

    InitExpectedLayouts();
  }

  void InitExpectedLayouts() {
    std::vector<bool> dynamic_flags;
    if (!dynamic_outputs_py_.is_none()) {
      dynamic_flags = dynamic_outputs_py_.cast<std::vector<bool>>();
    }

    size_t idx = 0;
    for (py::handle item : expected_layouts_py_) {
      ExpectedTensorLayout layout;
      layout.dtype = item.attr("dtype").cast<at::ScalarType>();
      layout.shape = ParseDimensions(item.attr("shape"));
      layout.stride = ParseDimensions(item.attr("stride"));
      layout.storage_offset = item.attr("storage_offset").cast<int64_t>();
      layout.is_dynamic = idx < dynamic_flags.size() && dynamic_flags[idx];
      layout.is_standard_contiguous = IsStandardContiguousLayout(
          layout.shape, layout.stride, layout.storage_offset);
      layout.is_physically_contiguous = IsPhysicallyContiguousLayout(
          at::IntArrayRef(layout.shape.data(), layout.shape.size()),
          at::IntArrayRef(layout.stride.data(), layout.stride.size()),
          layout.storage_offset);
      layouts_.push_back(std::move(layout));
      ++idx;
    }

    // Validate that each `kDedupedTensor` entry indexes a valid layout and
    // precompute whether it requires a runtime stride/layout verification.
    // Fresh PJRT buffers returned by ExecuteCompiledModel are allocated by the
    // runtime with standard contiguous row-major strides and zero storage
    // offset. When `outputs_are_fresh_pjrt` is true (the hot path in
    // CompiledExecutableRunner::RunFast), we skip `layout.MatchesTensor()`
    // completely unless the expected FX output explicitly specified a
    // non-standard strided/transposed layout
    // (`!layout.is_standard_contiguous`).
    for (Entry& entry : entries_) {
      if (entry.type != EntryType::kDedupedTensor) {
        continue;
      }
      if (entry.index < 0 ||
          static_cast<size_t>(entry.index) >= layouts_.size()) {
        throw py::index_error(  // For pybind11.
            absl::StrCat("expected output tensor index to be in [0, ",
                         layouts_.size(), "), got ", entry.index));
      }
      num_output_tensors_ =
          std::max(num_output_tensors_, static_cast<size_t>(entry.index) + 1);
      const ExpectedTensorLayout& layout = layouts_[entry.index];
      entry.needs_layout_check =
          !layout.is_dynamic && !layout.is_standard_contiguous;
    }
  }

  // Reconstructs the FX graph output structure from runtime `inputs_handle` and
  // `deduped_outputs`.
  //
  // Performance note: This method uses direct Python C-API calls
  // (`PyTuple_New`, `PyTuple_SET_ITEM`, `PyList_New`, `PyList_SET_ITEM`, and
  // `THPVariable_Wrap`) instead of pybind11 wrappers or Python-level PyTree
  // unflattening for leaf, flat tuple, and flat list outputs. This avoids
  // intermediate Python object allocations, dictionary lookups, and reference
  // counting overhead on every compiled step.
  py::object Reconstruct(py::handle inputs_handle,
                         absl::Span<at::Tensor> deduped_outputs,
                         bool outputs_are_fresh_pjrt = false) const {
    TT_CHECK_THROW(deduped_outputs.size() >= num_output_tensors_,
                   error::kPythonIndexError)
        << "expected the number of output tensors to be at least "
        << num_output_tensors_ << ", got " << deduped_outputs.size();
    const RawPyObjPtr inputs_ptr = inputs_handle.ptr();
    if (spec_kind_ == SpecKind::kLeaf && entries_.size() == 1) {
      return py::reinterpret_steal<py::object>(GetReconstructedLeafNewRef(
          0, inputs_ptr, deduped_outputs, outputs_are_fresh_pjrt));
    }

    // The containers are owned before being filled so that they are released
    // if `GetReconstructedLeafNewRef` throws; CPython deallocates the
    // not-yet-set (null) items safely.
    if (spec_kind_ == SpecKind::kFlatTuple) {
      py::object tuple =
          py::reinterpret_steal<py::object>(PyTuple_New(entries_.size()));
      if (!tuple) {
        throw py::error_already_set();  // Propagate Python C-API error.
      }
      for (size_t i = 0; i < entries_.size(); ++i) {
        PyTuple_SET_ITEM(
            tuple.ptr(), i,
            GetReconstructedLeafNewRef(i, inputs_ptr, deduped_outputs,
                                       outputs_are_fresh_pjrt));
      }
      return tuple;
    }

    py::object list =
        py::reinterpret_steal<py::object>(PyList_New(entries_.size()));
    if (!list) {
      throw py::error_already_set();  // Propagate Python C-API error.
    }
    for (size_t i = 0; i < entries_.size(); ++i) {
      PyList_SET_ITEM(list.ptr(), i,
                      GetReconstructedLeafNewRef(i, inputs_ptr, deduped_outputs,
                                                 outputs_are_fresh_pjrt));
    }
    if (spec_kind_ == SpecKind::kFlatList) {
      return list;
    }
    return spec_py_.attr("unflatten")(list);
  }

  py::object Call(py::handle inputs,
                  std::vector<at::Tensor> deduped_outputs) const {
    return Reconstruct(inputs, absl::MakeSpan(deduped_outputs),
                       /*outputs_are_fresh_pjrt=*/false);
  }

  // Enables pickle serialization (`__reduce__`) for compiled artifact caching.
  py::tuple Reduce() const {
    py::object cls =
        PyBindModule::Import("torch_tpu._internal.compile.tpu_torch_compile")
            .attr("FxOutputReconstructor");
    return py::make_tuple(
        cls, py::make_tuple(output_map_py_, expected_layouts_py_, spec_py_,
                            dynamic_outputs_py_));
  }

 private:
  static void CheckInputIndex(const Py_ssize_t size, const int index) {
    TT_CHECK_THROW(index >= 0 && index < size, error::kPythonIndexError)
        << "expected input index to be in [0, " << size << "), got " << index;
  }

  // Returns a new reference to `inputs_ptr[index]`, or throws a Python
  // `IndexError` if `index` is out of bounds.
  static RawPyObjPtr GetInputItemNewRef(RawPyObjPtr inputs_ptr,
                                        const int index) {
    py::object fast_holder;
    const RawPyObjPtr fast_inputs = AsFastSequence(inputs_ptr, &fast_holder);
    CheckInputIndex(PySequence_Fast_GET_SIZE(fast_inputs), index);
    RawPyObjPtr item = PySequence_Fast_GET_ITEM(fast_inputs, index);
    Py_INCREF(item);
    return item;
  }

  RawPyObjPtr GetReconstructedLeafNewRef(size_t i, RawPyObjPtr inputs_ptr,
                                         absl::Span<at::Tensor> deduped_outputs,
                                         bool outputs_are_fresh_pjrt) const {
    const Entry& entry = entries_[i];
    if (entry.type == EntryType::kNone) {
      Py_INCREF(Py_None);
      return Py_None;
    }
    if (entry.type == EntryType::kNonTensorPassthrough) {
      return GetInputItemNewRef(inputs_ptr, entry.index);
    }
    const int idx = entry.index;
    at::Tensor tensor = [&]() {  // REF_CAPTURE_OK=called locally
      if (entry.needs_clone) {
        return deduped_outputs[idx].clone();
      }
      if (entry.can_move) {
        return std::move(deduped_outputs[idx]);
      }
      return deduped_outputs[idx];
    }();
    const ExpectedTensorLayout& layout = layouts_[idx];
    if ((entry.needs_layout_check || !outputs_are_fresh_pjrt) &&
        !layout.is_dynamic && !layout.MatchesTensor(tensor)) {
      if (outputs_are_fresh_pjrt && entry.can_move &&
          layout.is_physically_contiguous) {
        tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
            at::IntArrayRef(layout.shape.data(), layout.shape.size()),
            at::IntArrayRef(layout.stride.data(), layout.stride.size()),
            layout.storage_offset);
      } else {
        tensor = ForceStrides(tensor, layout.stride, layout.storage_offset);
      }
    }
    return THPVariable_Wrap(std::move(tensor));
  }

  std::vector<Entry> entries_;
  std::vector<ExpectedTensorLayout> layouts_;
  // The number of deduplicated outputs that `entries_` read.
  size_t num_output_tensors_ = 0;
  SpecKind spec_kind_;
  py::object output_map_py_;
  py::object expected_layouts_py_;
  py::object spec_py_;
  py::object dynamic_outputs_py_;
};

// Dynamo's boxed calling convention passes compiled graph inputs as either
// `*args` (a tuple of arguments) or a single boxed `[arg0, arg1, ...]` list
// inside a 1-element tuple `(boxed_list,)`. UnboxSingleSequenceArg unwraps
// that outer 1-element tuple/list without allocating a new Python container,
// and ensures the returned pointer is an exact tuple or list
// (`AsFastSequence`).
RawPyObjPtr UnboxSingleSequenceArg(RawPyObjPtr args_ptr,
                                   py::object* fast_holder) {
  const RawPyObjPtr fast_args = AsFastSequence(args_ptr, fast_holder);
  if (PySequence_Fast_GET_SIZE(fast_args) == 1) {
    RawPyObjPtr first = PySequence_Fast_GET_ITEM(fast_args, 0);
    if (PyList_CheckExact(first) || PyTuple_CheckExact(first)) {
      return first;
    }
  }
  return fast_args;
}

// Fast-path argument unpacker that traverses the contiguous internal
// `PyObject*` array of the exact `PyTupleObject` or `PyListObject` returned by
// `UnboxSingleSequenceArg`.
//
// Why pointers (`const at::Tensor*`) instead of value copies (`at::Tensor`)?
// Copying an `at::Tensor` increments the atomic reference count of its
// `c10::TensorImpl`. For a model with 100+ parameters/inputs, copying tensors
// into a `std::vector<at::Tensor>` on every step incurs 200+ atomic
// increment/decrement operations on the hot path. Because the caller holds the
// Python GIL and the input container (`args_ptr` or `fast_holder`) remains
// alive throughout the call, taking raw pointers directly to
// `THPVariable_Unpack(item)` is 100% zero-copy and requires zero atomic
// reference count operations.
void UnpackInputTensorPointers(
    RawPyObjPtr fast_args_ptr,
    absl::InlinedVector<const at::Tensor*, 128>* argument_tensor_ptrs,
    absl::InlinedVector<at::Generator, 2>* generators = nullptr) {
  const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast_args_ptr);
  const RawPyObjPtr* const items = PySequence_Fast_ITEMS(fast_args_ptr);
  argument_tensor_ptrs->reserve(n);
  for (Py_ssize_t i = 0; i < n; ++i) {
    RawPyObjPtr item = items[i];
    if (THPVariable_Check(item)) {
      argument_tensor_ptrs->push_back(&THPVariable_Unpack(item));
    } else if (generators != nullptr && THPGenerator_Check(item)) {
      generators->push_back(THPGenerator_Unwrap(item));
    }
  }
}

std::vector<OutputShape> ParseOutputShapes(py::handle output_shapes_obj) {
  if (output_shapes_obj.is_none()) {
    return {};
  }
  RawPyObjPtr shapes_ptr = output_shapes_obj.ptr();
  const bool is_empty_seq =
      (PyTuple_CheckExact(shapes_ptr) && PyTuple_GET_SIZE(shapes_ptr) == 0) ||
      (PyList_CheckExact(shapes_ptr) && PyList_GET_SIZE(shapes_ptr) == 0);
  if (is_empty_seq) {
    return {};
  }
  return output_shapes_obj.cast<std::vector<OutputShape>>();
}

// Executes a compiled PJRT executable with the given arguments.
py::object PyExecuteCompiledModel(
    const SharedLoadedExecutableWithMetadata& executable, py::handle py_args,
    py::handle output_shapes_obj = py::none(),
    py::handle reconstruct_fn = py::none()) {
  py::object fast_args_holder;
  RawPyObjPtr args_ptr =
      UnboxSingleSequenceArg(py_args.ptr(), &fast_args_holder);

  absl::InlinedVector<const at::Tensor*, 128> argument_tensor_ptrs;
  UnpackInputTensorPointers(args_ptr, &argument_tensor_ptrs);

  std::vector<OutputShape> parsed_output_shapes =
      ParseOutputShapes(output_shapes_obj);

  std::vector<at::Tensor> outputs = ExecuteCompiledModel(
      executable, argument_tensor_ptrs, parsed_output_shapes);

  if (reconstruct_fn.is_none()) {
    return py::cast(outputs);
  }
  if (py::isinstance<FxOutputReconstructor>(reconstruct_fn)) {
    const auto& reconstructor =
        reconstruct_fn.cast<const FxOutputReconstructor&>();
    return reconstructor.Reconstruct(
        py::reinterpret_borrow<py::object>(args_ptr), absl::MakeSpan(outputs),
        /*outputs_are_fresh_pjrt=*/true);
  }
  return reconstruct_fn(py::reinterpret_borrow<py::object>(args_ptr),
                        py::cast(outputs));
}

// RAII lock guard for one or more PyTorch RNG generators.
//
// Concurrency and GIL invariants:
// 1. Deadlock prevention: When multiple generators are present, their mutex
//    pointers are sorted in ascending memory address order and deduplicated
//    before acquisition.
// 2. Optimistic non-blocking fast path: In the common case of a single
//    uncontended generator (`try_lock()`), the mutex is acquired immediately
//    WITHOUT releasing the Python GIL (`py::gil_scoped_release`). Dropping and
//    re-acquiring the GIL costs ~150-300ns of thread-state transitions;
//    avoiding it when uncontended keeps RNG-bearing graphs fast.
// 3. Deadlock-safe slow path: If `try_lock()` fails (or multiple generators
//    must be locked), the Python GIL is released BEFORE blocking on
//    `std::mutex::lock()`. This prevents ABBA deadlocks between the Python GIL
//    and generator mutexes if another C++ thread holds the generator lock while
//    waiting for the GIL.
class ScopedGeneratorLocks {
 public:
  explicit ScopedGeneratorLocks(absl::Span<at::Generator> generators) {
    if (generators.empty()) {
      return;
    }
    absl::InlinedVector<std::mutex*, 2> unique_mutexes;
    for (at::Generator& gen : generators) {
      unique_mutexes.push_back(&gen.mutex());
    }
    if (unique_mutexes.size() > 1) {
      std::sort(unique_mutexes.begin(), unique_mutexes.end());
      unique_mutexes.erase(
          std::unique(unique_mutexes.begin(), unique_mutexes.end()),
          unique_mutexes.end());
    }
    locks_.reserve(unique_mutexes.size());
    if (unique_mutexes.size() == 1 && unique_mutexes[0]->try_lock()) {
      locks_.emplace_back(*unique_mutexes[0], std::adopt_lock);
      return;
    }
    py::gil_scoped_release release;
    for (std::mutex* m : unique_mutexes) {
      locks_.emplace_back(*m);
    }
  }

 private:
  absl::InlinedVector<std::unique_lock<std::mutex>, 2> locks_;
};

// Pre-bound compiled executable runner that caches the unpacked C++ executable,
// output shapes, and FxOutputReconstructor pointer to bypass per-step pybind11
// type-registry hash table lookups and Python dispatch overhead.
//
// Why this exists:
// Calling a standard pybind11-bound function like `execute_compiled_model(exe,
// args, shapes, reconstruct)` from Python requires pybind11 to look up the C++
// type of `exe`, `shapes`, and `reconstruct` in its global
// `registered_types_py` hash map on every call, followed by multiple
// Python-to-C++ transition steps for RNG state extraction and output PyTree
// reconstruction.
//
// `CompiledExecutableRunner` is instantiated once when
// `TorchTpuCompiledExecutable` is created and caches raw C++
// pointers/structures (`executable_`, `cached_output_shapes_`, and
// `reconstructor_ptr_`). Its `RunFast` entry point executes the entire
// step—unboxing inputs, extracting zero-copy `const at::Tensor*` pointers,
// locking and updating C++ RNG generator state tensors, invoking PJRT
// execution, and reconstructing the output PyTree—in a single C++ call.
class CompiledExecutableRunner {
 public:
  CompiledExecutableRunner(SharedLoadedExecutableWithMetadata executable,
                           std::vector<OutputShape> cached_output_shapes,
                           py::object reconstruct_fn,
                           bool updates_default_generator_state = false)
      : executable_(std::move(executable)),
        cached_output_shapes_(std::move(cached_output_shapes)),
        reconstruct_fn_(std::move(reconstruct_fn)),
        updates_default_generator_state_(updates_default_generator_state) {
    // Pre-allocate the output shape metadata for the common 1-generator case so
    // `RunFast` does not allocate or copy `std::vector<OutputShape>` at
    // runtime.
    if (!cached_output_shapes_.empty()) {
      cached_output_shapes_with_1_gen_ = cached_output_shapes_;
      cached_output_shapes_with_1_gen_.push_back(OutputShape{
          std::vector<int64_t>{2}, /*is_dynamic=*/false});  // INT_VEC_OK
    }
    if (!reconstruct_fn_.is_none() &&
        py::isinstance<FxOutputReconstructor>(reconstruct_fn_)) {
      reconstructor_ptr_ =
          &reconstruct_fn_.cast<const FxOutputReconstructor&>();
    }
  }

  py::object RunFast(py::handle py_args) const {
    py::object fast_args_holder;
    RawPyObjPtr args_ptr =
        UnboxSingleSequenceArg(py_args.ptr(), &fast_args_holder);

    absl::InlinedVector<const at::Tensor*, 128> argument_tensor_ptrs;
    absl::InlinedVector<at::Generator, 2> generators;
    UnpackInputTensorPointers(args_ptr, &argument_tensor_ptrs, &generators);
    if (updates_default_generator_state_) {
      generators.push_back(GetDefaultDeviceGenerator(-1));
    }

    // Fast path for deterministic / RNG-free executables: dispatch directly to
    // PJRT and reconstruct outputs without acquiring generator locks.
    if (generators.empty()) {
      std::vector<at::Tensor> outputs = ExecuteCompiledModel(
          executable_, argument_tensor_ptrs, cached_output_shapes_);
      return FormatUserOutputs(args_ptr, outputs, outputs.size());
    }

    return RunWithGenerators(args_ptr, argument_tensor_ptrs,
                             absl::MakeSpan(generators));
  }

 private:
  py::object FormatUserOutputs(RawPyObjPtr args_ptr,
                               std::vector<at::Tensor>& outputs,
                               size_t user_outputs_count) const {
    absl::Span<at::Tensor> user_outputs =
        absl::MakeSpan(outputs).subspan(0, user_outputs_count);
    if (reconstructor_ptr_ != nullptr) {
      return reconstructor_ptr_->Reconstruct(
          py::reinterpret_borrow<py::object>(args_ptr), user_outputs,
          /*outputs_are_fresh_pjrt=*/true);
    }
    outputs.resize(user_outputs_count);
    if (reconstruct_fn_.is_none()) {
      return py::cast(std::move(outputs));
    }
    return reconstruct_fn_(py::reinterpret_borrow<py::object>(args_ptr),
                           py::cast(std::move(outputs)));
  }

  py::object RunWithGenerators(
      RawPyObjPtr args_ptr,
      absl::InlinedVector<const at::Tensor*, 128>& argument_tensor_ptrs,
      absl::Span<at::Generator> generators) const {
    // RNG-stateful path: lock all participating device generators, append their
    // current state tensors to the executable's input arguments, run PJRT
    // execution, and atomically update each generator's state tensor with the
    // trailing state outputs produced by the compiled graph.
    std::vector<at::Tensor> outputs;
    size_t user_outputs_count = 0;
    {
      ScopedGeneratorLocks gen_locks(generators);
      absl::InlinedVector<at::Tensor, 2> device_state_tensors;
      device_state_tensors.reserve(generators.size());
      for (const at::Generator& gen : generators) {
        device_state_tensors.push_back(PyGetDeviceStateTensor(gen));
        argument_tensor_ptrs.push_back(&device_state_tensors.back());
      }

      absl::Span<const OutputShape> exec_output_shapes = cached_output_shapes_;
      std::vector<OutputShape> dynamic_output_shapes_with_gens;
      if (!cached_output_shapes_.empty()) {
        if (generators.size() == 1 &&
            !cached_output_shapes_with_1_gen_.empty()) {
          exec_output_shapes = cached_output_shapes_with_1_gen_;
        } else {
          dynamic_output_shapes_with_gens = cached_output_shapes_;
          dynamic_output_shapes_with_gens.reserve(cached_output_shapes_.size() +
                                                  generators.size());
          for (size_t i = 0; i < generators.size(); ++i) {
            dynamic_output_shapes_with_gens.push_back(OutputShape{
                std::vector<int64_t>{2}, /*is_dynamic=*/false});  // INT_VEC_OK
          }
          exec_output_shapes = dynamic_output_shapes_with_gens;
        }
      }

      outputs = ExecuteCompiledModel(executable_, argument_tensor_ptrs,
                                     exec_output_shapes);

      const size_t num_gens = generators.size();
      user_outputs_count = outputs.size() - num_gens;
      for (size_t i = 0; i < num_gens; ++i) {
        PySetDeviceStateTensor(generators[i],
                               std::move(outputs[user_outputs_count + i]));
      }
    }

    return FormatUserOutputs(args_ptr, outputs, user_outputs_count);
  }
  SharedLoadedExecutableWithMetadata executable_;
  std::vector<OutputShape> cached_output_shapes_;
  std::vector<OutputShape> cached_output_shapes_with_1_gen_;
  py::object reconstruct_fn_;
  const FxOutputReconstructor* reconstructor_ptr_ = nullptr;
  bool updates_default_generator_state_ = false;
};

TT_PYBIND11_MODULE(tpu_torch_compile, m) {
  PyBindClass<FxOutputReconstructor>(
      m, "FxOutputReconstructor",
      "Reconstructs original FX graph output structure from flattened MLIR "
      "outputs.")
      .def(py::init<py::object, py::object, py::object, py::object>(),
           py::arg("output_map"), py::arg("expected_layouts"), py::arg("spec"),
           py::arg("dynamic_outputs") = py::none())
      .def(
          "__call__",
          [](const FxOutputReconstructor& self, py::handle inputs,
             std::vector<at::Tensor> deduped_outputs) {
            return self.Call(inputs, std::move(deduped_outputs));
          },
          py::arg("inputs"), py::arg("deduped_outputs"))
      .def("__reduce__",
           [](const FxOutputReconstructor& self) { return self.Reduce(); });
  m.attr("FxOutputReconstructor").attr("__module__") =
      "torch_tpu._internal.compile.tpu_torch_compile";

  PyBindClass<TensorInfo>(m, "TensorInfo")
      .def(py::init([](const py::tuple& t) {
        return TensorInfo{t[0].cast<std::vector<int64_t>>(),  // INT_VEC_OK
                          t[1].cast<at::ScalarType>()};
      }))
      .def_readonly("shape", &TensorInfo::shape)
      .def_readonly("dtype", &TensorInfo::dtype);

  PyBindClass<TensorBounds>(m, "TensorBounds")
      .def(py::init([](const py::tuple& t) {
        return TensorBounds{t[0].cast<std::vector<int64_t>>(),   // INT_VEC_OK
                            t[1].cast<std::vector<int64_t>>()};  // INT_VEC_OK
      }))
      .def_readonly("dynamic_dims", &TensorBounds::dynamic_dims)
      .def_readonly("upper_bounds", &TensorBounds::upper_bounds);

  PyBindClass<OutputShape>(m, "OutputShape")
      .def(py::init([](std::vector<int64_t> dimensions,  // INT_VEC_OK
                       bool is_dynamic) {
             return OutputShape{std::move(dimensions), is_dynamic};
           }),
           py::arg("dimensions") = std::vector<int64_t>(),  // INT_VEC_OK
           py::arg("is_dynamic") = false)
      .def_readwrite("dimensions", &OutputShape::dimensions)
      .def_readwrite("is_dynamic", &OutputShape::is_dynamic)
      .def("__repr__", [](const OutputShape& s) {
        return absl::StrCat("OutputShape(dimensions=", ToString(s.dimensions),
                            ", is_dynamic=", s.is_dynamic ? "True" : "False",
                            ")");
      });

  py::implicitly_convertible<py::tuple, TensorInfo>();
  py::implicitly_convertible<py::tuple, TensorBounds>();

  PyBindClass<CompileResult>(m, "CompileResult")
      .def_readonly("module", &CompileResult::module)
      .def_readonly("executable", &CompileResult::executable);

  PyBindClass<AsyncCompileResult>(
      m, "AsyncCompileResult",
      "Handle to an in-flight compilation started by "
      "`async_traverse_and_compile`.\n\n"
      "Thread-safe once constructed: multiple threads may call `is_ready()` "
      "and `get()` on the same instance concurrently.")
      .def_property_readonly(
          "module",
          [](const AsyncCompileResult& self) { return self.module(); })
      .def(
          "is_ready",
          [](const AsyncCompileResult& self) { return self.IsReady(); },
          "Returns `True` if compilation has finished, i.e. if a subsequent "
          "`get()` will not block. Never blocks.")
      .def(
          "get",
          [](AsyncCompileResult& self) -> SharedLoadedExecutableWithMetadata {
            absl::StatusOr<SharedLoadedExecutableWithMetadata> status_or;
            {
              // Release the GIL so that other Python threads, including ones
              // starting their own compilations, can run while this one waits.
              py::gil_scoped_release release;
              status_or = self.Get();
            }
            TT_ASSIGN_OR_THROW(auto executable, std::move(status_or));
            return executable;
          },
          "Blocks until compilation finishes and returns the executable, with "
          "the deduplicated argument indices already applied.\n\n"
          "Releases the GIL while waiting. May be called repeatedly, and "
          "concurrently from multiple threads; every call returns the same "
          "executable. Raises if compilation failed.");

  PyBindClass<ContextedModule,  // NOLINT(bugprone-unused-raii)
              std::shared_ptr<ContextedModule>>(m, "ContextedModule");

  PyBindClass<  // NOLINT(bugprone-unused-raii)
      LoadedExecutableWithMetadata,
      std::shared_ptr<LoadedExecutableWithMetadata>>(
      m, "LoadedExecutableWithMetadata")
      .def("get_parameter_layouts", &PyGetParameterLayoutsFromMetadata)
      .def("get_output_layouts", &PyGetOutputLayoutsFromMetadata)
      .def(
          "subslice_origin",
          [](const LoadedExecutableWithMetadata& executable)
              -> std::optional<std::vector<int32_t>> {
            const std::optional<xla::PjRtDeviceDimensions>& origin =
                executable.subslice_origin();
            if (!origin.has_value()) {
              return std::nullopt;
            }
            return std::vector<int32_t>(origin->begin(), origin->end());
          },
          "Chip coordinates (x, y, z) of the TPU subslice the executable was "
          "loaded at, or None if it was loaded for the full topology. Pass it "
          "to load_serialized_executable to load the executable again, e.g. "
          "after pickling.")
      .def("fingerprint_executable",
           [](const LoadedExecutableWithMetadata& executable) -> std::string {
             TT_ASSIGN_OR_THROW(
                 const std::string fingerprint,
                 executable.GetExecutable()->FingerprintExecutable(),
                 _.SetPrepend() << "failed to get executable fingerprint: ");
             return absl::BytesToHexString(fingerprint);
           });

  PyBindClass<xla::PjRtLoadedExecutable,  // NOLINT(bugprone-unused-raii)
              std::shared_ptr<xla::PjRtLoadedExecutable>>(
      m, "PjRtLoadedExecutable")
      .def("get_parameter_layouts", &PyGetParameterLayouts)
      .def("get_output_layouts", &PyGetOutputLayouts);

  PyBindClass<MultiGeneratorLocker>(
      m, "MultiGeneratorLocker",
      "A context manager for locking multiple generators' mutexes.\n\n"
      "This is necessary to prevent generator state conflicts in between\n"
      "getting the device state tensor, executing compiled executable, and\n"
      "setting the updated device state tensor.")
      .def(pybind11::init<std::vector<at::Generator>&>())
      .def("__enter__", [](MultiGeneratorLocker& self) { self.Enter(); })
      .def("__exit__", [](MultiGeneratorLocker& self, pybind11::handle exc_type,
                          pybind11::handle exc_val, pybind11::handle exc_tb) {
        self.Exit(exc_type, exc_val, exc_tb);
      });

  m.def("execute", PyExecuteCompiledModel,
        // Type: PjRtLoadedExecutable
        py::arg("executable"), py::arg("argument_tensors"),
        py::arg("output_shapes") = py::none(),
        py::arg("reconstruct_fn") = py::none(),
        "Executes a compiled PJRT executable with the given arguments.\n\n"
        "Args:\n"
        "  executable: The loaded PJRT executable to run.\n"
        "  argument_tensors: A list of PyTorch tensors to pass as inputs.\n"
        "  output_shapes: Optional list of OutputShape objects for the output "
        "tensors.\n"
        "    If provided, the number of elements must match the number of "
        "output tensors. These shapes override the static shapes inferred\n"
        "    from the executable.\n"
        "    This is useful for bounded dynamic programs where the actual\n"
        "    output shape might differ from the static upper bound.");
  m.def(
      "make_compiled_runner",
      [](SharedLoadedExecutableWithMetadata executable,
         py::handle output_shapes_obj, py::object reconstruct_fn,
         bool updates_default_generator_state) {
        auto runner = std::make_shared<CompiledExecutableRunner>(
            std::move(executable), ParseOutputShapes(output_shapes_obj),
            std::move(reconstruct_fn), updates_default_generator_state);
        return py::cpp_function(internal::WrapWithErrorHandling(
            "execute", [runner](py::handle py_args) -> py::object {
              return runner->RunFast(py_args);
            }));
      },
      py::arg("executable"), py::arg("output_shapes") = py::none(),
      py::arg("reconstruct_fn") = py::none(),
      py::arg("updates_default_generator_state") = false,
      "Creates a pre-bound C++ runner callable for zero-lookup fast dispatch.");
  m.def("placeholder", PyMakePlaceholder, py::arg("sizes"), py::arg("dtype"),
        py::arg("requires_grad") = false, py::arg("is_pinned_host") = false);
  m.def("placeholder_like", PyMakePlaceholderLike, py::arg("arg_tensor"),
        py::arg("is_pinned_host") = false, py::arg("layout") = py::none());
  m.def("dynamic_placeholder", PyMakeDynamicPlaceholder, py::arg("sizes"),
        py::arg("dtype"), py::arg("tensor_bounds"),
        py::arg("requires_grad") = false, py::arg("is_pinned_host") = false);
  m.def("is_pinned_host", &PyIsPinnedHost, py::arg("tensor"),
        "Returns whether the given tensor is host offloaded.");
  m.def("get_device_layout_if_materialized", &PyGetDeviceLayoutIfMaterialized,
        py::arg("tensor"));
  m.def("get_default_layout", &PyGetDefaultLayout, py::arg("dtype"),
        py::arg("shape"));
  m.def("is_device_shape_dynamic", &PyIsDeviceShapeDynamic, py::arg("tensor"));
  m.def("traverse_and_compile", PyTraverseAndCompile, py::arg("result_tensors"),
        py::arg("argument_tensors"), py::arg("fast_compile") = false,
        py::arg("build_mlir_module") = false,
        py::arg("use_stablehlo_bounds") = false,
        py::arg("argument_layouts") = py::none(),
        py::arg("donated_inputs") = std::vector<int64_t>{},  // INT_VEC_OK
        py::arg("module_name") = py::none(),
        "Traverses the graph from outputs to arguments and compiles it. \n\n"
        "Args:\n"
        "  result_tensors: The output tensors to compile.\n"
        "  argument_tensors: The input tensors to compile.\n"
        "  fast_compile: Whether to use the fast compile mode.\n"
        "  build_mlir_module: Whether to build the MLIR module.\n"
        "  use_stablehlo_bounds: Whether to use the StableHLO bounds for"
        "    dynamic inputs.\n"
        "  argument_layouts: Optional layout of the input arguments. If not"
        "    empty, the size must match the number of arguments.\n"
        "  donated_inputs: Optional list of argument indices to donate.\n"
        "  module_name: Optional name for the compiled MLIR module.\n"
        "Returns:\n"
        "  CompileResult: The compiled module and executable.");

  m.def("async_traverse_and_compile", PyAsyncTraverseAndCompile,
        py::arg("result_tensors"), py::arg("argument_tensors"),
        py::arg("fast_compile") = false, py::arg("build_mlir_module") = false,
        py::arg("use_stablehlo_bounds") = false,
        py::arg("argument_layouts") = py::none(),
        py::arg("donated_inputs") = std::vector<int64_t>{},  // INT_VEC_OK
        py::arg("module_name") = py::none(),
        "Traverses the graph from outputs to arguments and starts compiling "
        "it, without waiting for compilation to finish.\n"
        "Args:\n"
        "  result_tensors: The output tensors of the graph.\n"
        "  argument_tensors: The input tensors of the graph.\n"
        "  fast_compile: Whether to prioritize compilation latency over "
        "    runtime performance.\n"
        "  build_mlir_module: Whether to also build and return the MLIR "
        "    module.\n"
        "  use_stablehlo_bounds: Whether to use the StableHLO bounds for\n"
        "    dynamic inputs.\n"
        "  argument_layouts: Optional layout of the input arguments. If not"
        "    empty, the size must match the number of arguments.\n"
        "  donated_inputs: Optional list of argument indices to donate.\n"
        "  module_name: Optional name for the compiled MLIR module.\n"
        "Returns:\n"
        "  `AsyncCompileResult`: A handle whose `get()` blocks for the "
        "    executable.\n"
        "Concurrency:\n"
        "  Thread-compatible. Calls on graphs that do not share tensors may "
        "    run concurrently; the caller must not mutate the graph reachable "
        "    from `result_tensors` while this call is in progress.");

  m.def("build_mlir", PyBuildMlir, py::arg("result_tensors"),
        py::arg("argument_tensors"), py::arg("use_stablehlo_bounds") = false,
        "Builds an MLIR module from the graph.");
  // Returns: PjRtLoadedExecutable
  m.def("compile_mlir", PyCompileMlir, py::arg("module"),
        py::arg("fast_compile") = false,
        py::arg("argument_layouts") =
            std::vector<std::vector<int64_t>>{},  // INT_VEC_OK
        "Compiles an MLIR module to a PjRtLoadedExecutable.");
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
  m.def("get_or_compile_pad_module", PyGetOrCompilePadModule,
        py::arg("tensor_info"), py::arg("bounds_list"),
        py::arg("fast_compile") = false, py::arg("build_mlir_module") = false,
        py::arg("is_caching_disabled") = false,
        "Returns the compiled PJRT executable for a pad subgraph.\n\n"
        "Args:\n"
        "  tensor_info: A list of (shape, dtype) pairs for each tensor.\n"
        "  bounds_list: A list of (dynamic_dimensions, upper_bounds) pairs.\n"
        "  fast_compile: Whether to use the fast compile mode.\n"
        "  build_mlir_module: Whether to build the MLIR module.\n"
        "  is_caching_disabled: Whether to use the cache.");

  m.def("get_dynamic_pad_module", PyGetDynamicPadModule, py::arg("tensor_info"),
        py::arg("bounds_list"), py::arg("fast_compile") = false,
        "Returns the compiled PJRT executable for a dynamic pad subgraph.\n\n"
        "Args:\n"
        "  tensor_info: A list of (shape, dtype) pairs for each tensor.\n"
        "  bounds_list: A list of (dynamic_dimensions, upper_bounds) pairs.\n"
        "  fast_compile: Whether to use the fast compile mode.");

  m.def("precompile_pad_module", PyPrecompilePadModule, py::arg("tensor_info"),
        py::arg("bounds_list"), py::arg("fast_compile") = false,
        "Returns immediately after enqueuing compilation of a pad module.\n\n"
        "Args:\n"
        "  tensor_info: A list of (shape, dtype) pairs for each tensor.\n"
        "  bounds_list: A list of (dynamic_dimensions, upper_bounds) pairs.\n"
        "  fast_compile: Whether to use the fast compile mode.");
  m.def("get_or_compile_slice_module", PyGetOrCompileSliceModule,
        py::arg("target_shapes"), py::arg("padded_shapes"),
        py::arg("input_scalar_types"), py::arg("fast_compile") = false,
        py::arg("build_mlir_module") = false,
        py::arg("is_caching_disabled") = false,
        "Returns the compiled PJRT executable for a slice subgraph.\n\n"
        "Args:\n"
        "  target_shapes: A list of target shapes.\n"
        "  padded_shapes: A list of padded shapes.\n"
        "  input_scalar_types: A list of scalar types for each tensor.\n"
        "  fast_compile: Whether to use the fast compile mode.\n"
        "  build_mlir_module: Whether to build the MLIR module.\n"
        "  is_caching_disabled: Whether to use the cache.");
  m.def("precompile_slice_module", PyPrecompileSliceModule,
        py::arg("target_shapes"), py::arg("padded_shapes"),
        py::arg("input_scalar_types"), py::arg("fast_compile") = false,
        "Returns immediately after enqueuing compilation of a slice module.\n\n"
        "Args:\n"
        "  target_shapes: A list of target shapes.\n"
        "  padded_shapes: A list of padded shapes.\n"
        "  input_scalar_types: A list of scalar types for each tensor.\n"
        "  fast_compile: Whether to use the fast compile mode.");
  m.def("pop_enable_tracebacks", &PopContextState<EnableTracebacksContextState>,
        "Pops the current state of the enable_tracebacks context manager.");
  m.def("push_enable_tracebacks", &PyPushEnableTracebacks, py::arg("enabled"),
        "Pushes a new state onto the enable_tracebacks context manager.");
  m.def("append_name_scope", &ScopedPythonContextCapturer::AppendNameScope,
        py::arg("scope"),
        "Appends a single name scope to the active hierarchy on top of the "
        "PythonContext capturer stack (pushing the extended hierarchy so "
        "pop_name_scopes restores the previous one).");
  m.def("push_name_scopes", &ScopedPythonContextCapturer::PushNameScopes,
        py::arg("scopes"),
        "Pushes a full name scope hierarchy onto the PythonContext capturer "
        "stack, replacing the active hierarchy until popped.");
  m.def("pop_name_scopes", &ScopedPythonContextCapturer::PopNameScopes,
        "Pops the top name scope hierarchy from the PythonContext capturer "
        "stack, restoring the previous hierarchy.");
  m.def("serialize_executable", PySerializeExecutable, py::arg("executable"),
        "Serializes a PjRtLoadedExecutable to bytes for caching.");
  m.def("load_serialized_executable", PyLoadSerializedExecutable,
        py::arg("serialized_bytes"), py::arg("subslice_origin") = py::none(),
        "Loads a PjRtLoadedExecutable from serialized bytes. "
        "`subslice_origin` is the chip coordinates (x, y, z) of the TPU "
        "subslice the executable was compiled for (see "
        "LoadedExecutableWithMetadata.subslice_origin), or None to load it "
        "with no computation origin.");
  m.def("make_constant_tensor", PyMakeConstantTensor, py::arg("cpu_tensor"),
        "Creates a TPU tensor that represents the constant value of the CPU "
        "tensor.");
  m.def("is_constant_tensor", PyIsConstantTensor, py::arg("tensor"),
        "Returns whether the tensor is a constant TPU tensor created by "
        "make_constant_tensor.");
  m.def("assign_constant_tensor", PyAssignConstantTensor,
        py::arg("cpu_src_tensor"), py::arg("tpu_dst_tensor"),
        "Updates a TPU tensor to be a tensor with the constant value of the "
        "CPU tensor.");
  m.def("get_device_state_tensor", PyGetDeviceStateTensor, py::arg("generator"),
        "Returns the internal TPU RNG state tensor from a generator.");
  m.def("set_device_state_tensor", PySetDeviceStateTensor, py::arg("generator"),
        py::arg("rng_state"),
        "Sets the internal TPU RNG state tensor on a generator.");
  m.def("set_export_mode_default_device_generator",
        &PySetExportModeDefaultDeviceGenerator, py::arg("generator"),
        py::arg("idx") = -1,
        "Sets the export-mode default generator for override.");
  m.def("make_export_mode_device_generator", &PyMakeExportModeDeviceGenerator,
        py::arg("generator"),
        "Creates a new export-mode generator for the given generator.");
  m.def("clear_export_mode_default_device_generators",
        &PyClearExportModeDefaultDeviceGenerators,
        "Clears all export-mode default generator overrides.");
  m.def("force_strides", PyForceStrides, py::arg("tensor"),
        py::arg("target_strides"), py::arg("target_storage_offset"),
        "Returns a tensor with equivalent logical values to the input tensor, "
        "but with the given strides and storage offset, copying "
        "data as necessary.");

  py::enum_<HandshakeStage>(GetPyBindHandle(m), "HandshakeStage")
      .value("OFF", HandshakeStage::kOff)
      .value("COMPILE_STAGE", HandshakeStage::kCompileStage)
      .value("DISPATCH_STAGE", HandshakeStage::kDispatchStage)
      .export_values();

  m.def("get_handshake_stage_env_var_once", &GetHandshakeStageEnvVarOnce,
        "Returns configured handshake stage.");

  m.def(
      "fingerprint64",
      [](std::string_view input) { return tsl::Fingerprint64(input); },
      py::arg("input"),
      "Returns 64-bit unsigned integer fingerprint of input string.");

  py::enum_<ScanDirection>(GetPyBindHandle(m), "ScanDirection")
      .value("kForward", ScanDirection::kForward)
      .value("kReverse", ScanDirection::kReverse)
      .export_values();
  m.def("create_scan_op", &PyCreateScanOp, py::arg("inits"), py::arg("inputs"),
        py::arg("body_module"), py::arg("scan_direction"), py::arg("dummy_ys"),
        py::arg("num_scan_inputs"),
        "Creates a deferred scan operation and returns the output tensors.");
  m.def("create_associative_scan_op", &PyCreateAssociativeScanOp,
        py::arg("inputs"), py::arg("body_module"), py::arg("scan_direction"),
        py::arg("dim"), py::arg("num_scan_inputs"),
        "Creates a deferred associative scan operation and returns the output "
        "tensors.");
}

}  // namespace torch_tpu
