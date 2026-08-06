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
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "ATen/core/TensorBody.h"
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
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
#include "torch/extension.h"  // IWYU pragma: keep for aten::Tensor pybind type
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/_internal/compile/compiled_mode.h"
#include "torch_tpu/_internal/compile/dispatch_scan.h"
#include "torch_tpu/_internal/dynamism/dynamism_ops.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/compilation_cache.h"
#include "torch_tpu/common/compilation_spec.h"
#include "torch_tpu/common/context_manager.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_gen_impl.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "torch_tpu/ops/scan_builder.h"
#include "torch_tpu/ops/view_decomposition/contiguous_to_view.h"
#include "torch_tpu/ops/view_decomposition/decomposition.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/service/computation_placer.h"

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

at::Tensor PyMakePlaceholder(const std::vector<int64_t>& sizes,  // INT_VEC_OK
                             at::ScalarType dtype, bool requires_grad) {
  TT_ASSIGN_OR_THROW(at::Tensor prepared_tensor,
                     MakePlaceholder(sizes, dtype, requires_grad),
                     _.SetPrepend() << "failed to create placeholder tensor: ");
  return prepared_tensor;
}

at::Tensor PyMakeDynamicPlaceholder(
    const std::vector<int64_t>& sizes,  // INT_VEC_OK
    at::ScalarType dtype, const TensorBounds& bounds, bool requires_grad) {
  TT_ASSIGN_OR_THROW(mlir::ElementType element_type,
                     ConvertTo<mlir::ElementType>(dtype));
  Shape shape(CopyIntVector(sizes), element_type);
  shape.dynamic_dimensions().reserve(bounds.dynamic_dims.size());
  for (size_t i = 0; i < bounds.dynamic_dims.size(); ++i) {
    shape.dynamic_dimensions().push_back(
        BoundedDynamicDimension{.dimension = bounds.dynamic_dims[i],
                                .lower_bound = 2,  // Default lower bound
                                .upper_bound = bounds.upper_bounds[i]});
  }
  TT_ASSIGN_OR_THROW(at::Tensor prepared_tensor,
                     MakePlaceholder(std::move(shape), requires_grad),
                     _.SetPrepend()
                         << "failed to create dynamic placeholder tensor: ");
  return prepared_tensor;
}

at::Tensor PyMakePlaceholderLike(const at::Tensor& arg_tensor) {
  // arg_tensor may be a view, but placeholder DeviceBufferRefs are always
  // interpreted as contiguous.
  // If we want to make an equivalent view, we need to create a contiguous base
  // tensor as a placeholder, and then take a view of it with the same striding
  // as the original view.

  // First, we get a base shape for the contiguous base tensor.
  // This is the smallest amount of data necessary to back the view.
  // If arg_tensor is already contiguous, then this will just be its shape.
  TT_ASSIGN_OR_THROW(
      Dimensions minimal_base_sizes,
      GetContiguousBaseShape(StridedLayout::FromTensor(arg_tensor)));

  // Then, we create a contiguous placeholder tensor with this shape.
  TT_ASSIGN_OR_THROW(
      at::Tensor base_tensor,
      MakePlaceholder(minimal_base_sizes, arg_tensor.scalar_type(),
                      /*requires_grad=*/false),
      _.SetPrepend() << "failed to create placeholder tensor: ");

  // NOTE: This logic must match the logic from PrepareCompiledModeArguments in
  // compiled_mode.cc
  at::Tensor view_tensor;  // UNINITIALIZED_TENSOR_OK
  if (TensorHasTrivialLayout(arg_tensor)) {
    view_tensor = base_tensor;
  } else {
    // Finally, we create a view of the base tensor with the same striding as
    // the original view, and preserve the requires_grad property.
    view_tensor = base_tensor.as_strided(
        arg_tensor.sizes(), arg_tensor.strides(), arg_tensor.storage_offset());
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
  xla::PjRtClient* const client = PjrtBackend::GetInstance().GetClient();
  if (client == nullptr) {
    return PyLayout();
  }
  xla::PjRtClient& checked_client = *client;
  TT_ASSIGN_OR_THROW(auto element_type, ConvertTo<xla::PrimitiveType>(dtype));

  TT_ASSIGN_OR_THROW(const auto& layout,
                     checked_client.GetDefaultLayout(element_type, shape));
  return ToPyLayout(layout);
}

bool PyIsDeviceShapeDynamic(const at::Tensor& tensor) {
  TT_ASSIGN_OR_THROW(DeviceBufferRef buffer_ref, GetBuffer(tensor));
  if (!buffer_ref.dynamic_dimensions().empty()) {
    return true;
  }
  if (!buffer_ref.is_materializing()) {
    return false;
  }
  TT_ASSIGN_OR_THROW(auto* pjrt_buffer, buffer_ref.AwaitBuffer());
  return pjrt_buffer->on_device_shape().is_dynamic();
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
  TT_ASSIGN_OR_THROW(ContextedModule module,
                     ContextedModule::Make([&](mlir::MLIRContext& context) {
                       return ExtractMlirFromGraph(context, argument_tensors,
                                                   result_tensors,
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
          MakeCompileOptionsKey(GetEnvOnce<kXlaFlagsEnvVar>().value_or(""),
                                *compilation_spec.xla_compile_options);
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
            << "invalid layout for argument " << i << ", got layout "
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

  TT_ASSIGN_OR_THROW(
      auto executable,
      CompileMlirExecutable(xla::MaybeOwningMlirModule(module->get()),
                            std::move(compilation_spec.xla_compile_options)));
  return executable;
}

CompileResult PyTraverseAndCompile(
    const std::vector<at::Tensor>& result_tensors,
    const std::vector<at::Tensor>& argument_tensors, bool fast_compile,
    bool build_mlir_module, bool use_stablehlo_bounds,
    const std::vector<std::vector<int64_t>>& argument_layouts) {  // INT_VEC_OK
  if (!argument_layouts.empty()) {
    TT_CHECK_THROW(argument_layouts.size() == argument_tensors.size(),
                   error::kInvalidArgument)
        << "number of argument_layouts must match with the number of "
           "argument_tensors, got number of argument_layouts "
        << argument_layouts.size() << " and number of argument_tensors "
        << argument_tensors.size();

    for (size_t i = 0; i < argument_layouts.size(); ++i) {
      const auto& layout = argument_layouts[i];
      if (!layout.empty()) {
        int64_t rank = argument_tensors[i].dim();
        TT_CHECK_THROW(IsValidLayout(layout, rank), error::kInvalidArgument)
            << "invalid layout for argument " << i << ", got layout "
            << ToString(layout) << " for shape "
            << ToString(argument_tensors[i].sizes());
      }
    }
  }
  std::vector<Indices> converted_layouts;
  converted_layouts.reserve(argument_layouts.size());
  for (const auto& layout : argument_layouts) {
    converted_layouts.push_back(Indices(layout.begin(), layout.end()));
  }

  TT_ASSIGN_OR_THROW(
      CompileResult result,
      TraverseAndCompile(
          result_tensors, argument_tensors,
          TraverseAndCompileOptions{
              .compilation_mode = fast_compile ? CompilationMode::kFastCompile
                                               : CompilationMode::kFastRuntime,
              .build_mlir_module = build_mlir_module,
              .use_stablehlo_bounds = use_stablehlo_bounds,
              .argument_layouts = converted_layouts,
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

}  // namespace

// Returns the internal TPU RNG state tensor from a generator.
at::Tensor PyGetDeviceStateTensor(at::Generator gen) {
  auto* device_gen = at::check_generator<DeviceGeneratorImpl>(gen);
  return device_gen->DeviceStateTensor();
}

// Sets the internal TPU RNG state tensor on a generator.
void PySetDeviceStateTensor(at::Generator gen, at::Tensor rng_state) {
  auto* device_gen = at::check_generator<DeviceGeneratorImpl>(gen);
  TT_THROW_IF_ERROR(device_gen->SetDeviceStateTensor(std::move(rng_state)));
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
        << "dimension indices and upper bounds must have the same size, got "
        << dynamic_dims.size() << " and " << upper_bounds.size();

    for (int j = 0; j < dynamic_dims.size(); ++j) {
      const int64_t dim_index = dynamic_dims[j];
      const int64_t dim_upper_bound = upper_bounds[j];

      TT_CHECK_THROW(dim_index >= 0 && dim_index < dims.size(),
                     error::kInvalidArgument)
          << "dimension index must be within bounds [0, " << dims.size() - 1
          << "], got " << dim_index << " for input tensor " << i
          << " with shape " << ToString(dims);

      const int64_t dim_size = dims[dim_index];
      TT_CHECK_THROW(dim_upper_bound >= dim_size, error::kInvalidArgument)
          << "upper bound must be greater than or equal to the static shape's "
             "dimension size, got upper bound "
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
      << "target shapes and padded shapes must have the same size, got "
      << target_shapes.size() << " and " << padded_shapes.size();
  TT_CHECK_THROW(target_shapes.size() == input_scalar_types.size(),
                 error::kInvalidArgument)
      << "target shapes and input scalar types must have the same size, got "
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
        << "target shape and padded shape must have the same number of "
           "dimensions, got "
        << target.size() << " and " << padded.size() << " for tensor index "
        << i;

    Dimensions target_dims = CopyIntVector(target);
    Dimensions padded_dims = CopyIntVector(padded);

    for (size_t j = 0; j < target.size(); ++j) {
      TT_CHECK_THROW(padded[j] >= target[j], error::kInvalidArgument)
          << "padded shape dimension size must be greater than or equal to "
             "target shape dimension size, got padded shape "
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
  CompileResult result;
  std::optional<ContextedModule> contexted_module;

  // Create the MLIR module before compilation, since GetOrCompile consumes the
  // builder.
  if (build_mlir_module || is_caching_disabled) {
    TT_ASSIGN_OR_RETURN(contexted_module, ContextedModule::Make(builder),
                        _ << "failed to build MLIR module");
  }

  if (is_caching_disabled) {
    TT_ASSIGN_OR_RETURN(result.executable,
                        CompileMlirExecutable(
                            xla::MaybeOwningMlirModule(contexted_module->get()),
                            std::move(compile_options)));
  } else {
    TT_ASSIGN_OR_RETURN(
        CompiledKernel compiled_kernel,
        CompilationCache::GetInstance().GetOrCompile(
            cache_key, input_shapes, output_shapes, std::move(builder),
            std::move(compile_options), compile_dynamic_adapter));

    TT_ASSIGN_OR_RETURN(result.executable,
                        compiled_kernel.fixed_shape_kernel.get(),
                        _.SetPrepend() << "Failed to get fixed shape kernel: ");
  }

  if (build_mlir_module) {
    result.module =
        std::make_shared<ContextedModule>(std::move(*contexted_module));
  }
  return result;
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

  MlirComputationBuilder pad_builder = [&](mlir::MLIRContext& mlir_context) {
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

  MlirComputationBuilder pad_builder = [&](mlir::MLIRContext& mlir_context) {
    return GetPadModule(mlir_context, dynamic_shapes, /*pad_only_module=*/true);
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

  MlirComputationBuilder pad_builder = [&](mlir::MLIRContext& mlir_context) {
    return GetDynamicPadModule(mlir_context, dynamic_shapes);
  };

  const auto compilation_mode = fast_compile ? CompilationMode::kFastCompile
                                             : CompilationMode::kFastRuntime;
  CompilationSpec compilation_spec = GetCompilationSpec(compilation_mode);

  CompileResult result;
  std::optional<ContextedModule> contexted_module;

  TT_ASSIGN_OR_THROW(contexted_module, ContextedModule::Make(pad_builder),
                     _ << "failed to build MLIR module");

  TT_ASSIGN_OR_THROW(
      result.executable,
      CompileMlirExecutable(xla::MaybeOwningMlirModule(contexted_module->get()),
                            std::move(compilation_spec.xla_compile_options)));

  result.module =
      std::make_shared<ContextedModule>(std::move(*contexted_module));
  return result;
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

  MlirComputationBuilder slice_builder = [&](mlir::MLIRContext& mlir_context) {
    return GetSliceModule(mlir_context, inputs.target_dims, inputs.padded_dims,
                          inputs.element_types);
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

  MlirComputationBuilder slice_builder = [&](mlir::MLIRContext& mlir_context) {
    return GetSliceModule(mlir_context, inputs.target_dims, inputs.padded_dims,
                          inputs.element_types);
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
  TT_ASSIGN_OR_THROW(
      std::unique_ptr<xla::PjRtLoadedExecutable> pjrt_executable,
      AdaptXlaError(client->LoadSerializedExecutable(
                        data, /* options= */ std::nullopt, xla::LoadOptions()),
                    /* context= */ "failed to load serialized executable"));
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

at::Tensor PyForceStrides(
    const at::Tensor& tensor,
    const std::vector<int64_t>& target_strides,  // INT_VEC_OK
    int64_t target_storage_offset) {
  TT_CHECK_THROW(tensor.dim() == target_strides.size(), error::kInvalidArgument)
      << "target strides must have the same number of dimensions "
      << "as the tensor, got " << tensor.ndimension() << " and "
      << target_strides.size();
  TT_CHECK_THROW(target_storage_offset >= 0, error::kInvalidArgument)
      << "target_storage_offset must be non-negative";

  if (tensor.strides() == target_strides &&
      tensor.storage_offset() == target_storage_offset) {
    return tensor;
  }

  TT_ASSIGN_OR_THROW(DeviceBufferRef contiguous_buffer, GetBuffer(tensor));

  ScopedPythonContextCapturer capturer(OpName::kForceStrides);
  TT_ASSIGN_OR_THROW(
      at::Tensor view_tensor,
      ContiguousToView(std::move(contiguous_buffer),
                       CopyIntVector(target_strides), target_storage_offset));
  return view_tensor;
}

bool PyGetMaterializeCollectiveTensorsEnvValue() {
  return torch_tpu::GetMaterializeCollectiveTensorsEnvValue();
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

PYBIND11_MODULE(tpu_torch_compile, m) {
  py::class_<TensorInfo>(m, "TensorInfo")
      .def(py::init([](const py::tuple& t) {
        return TensorInfo{t[0].cast<std::vector<int64_t>>(),  // INT_VEC_OK
                          t[1].cast<at::ScalarType>()};
      }))
      .def_readonly("shape", &TensorInfo::shape)
      .def_readonly("dtype", &TensorInfo::dtype);

  py::class_<TensorBounds>(m, "TensorBounds")
      .def(py::init([](const py::tuple& t) {
        return TensorBounds{t[0].cast<std::vector<int64_t>>(),   // INT_VEC_OK
                            t[1].cast<std::vector<int64_t>>()};  // INT_VEC_OK
      }))
      .def_readonly("dynamic_dims", &TensorBounds::dynamic_dims)
      .def_readonly("upper_bounds", &TensorBounds::upper_bounds);

  py::implicitly_convertible<py::tuple, TensorInfo>();
  py::implicitly_convertible<py::tuple, TensorBounds>();

  py::class_<CompileResult>(m, "CompileResult")
      .def_readonly("module", &CompileResult::module)
      .def_readonly("executable", &CompileResult::executable);

  py::class_<torch_tpu::ContextedModule,  // NOLINT(bugprone-unused-raii)
             std::shared_ptr<torch_tpu::ContextedModule>>(m, "ContextedModule");

  py::class_<  // NOLINT(bugprone-unused-raii)
      torch_tpu::LoadedExecutableWithMetadata,
      std::shared_ptr<torch_tpu::LoadedExecutableWithMetadata>>(
      m, "LoadedExecutableWithMetadata")
      .def("get_parameter_layouts", &PyGetParameterLayoutsFromMetadata)
      .def("get_output_layouts", &PyGetOutputLayoutsFromMetadata);

  py::class_<xla::PjRtLoadedExecutable,  // NOLINT(bugprone-unused-raii)
             std::shared_ptr<xla::PjRtLoadedExecutable>>(m,
                                                         "PjRtLoadedExecutable")
      .def("get_parameter_layouts", &PyGetParameterLayouts)
      .def("get_output_layouts", &PyGetOutputLayouts);

  pybind11::class_<MultiGeneratorLocker>(
      m, "MultiGeneratorLocker",
      "A context manager for locking multiple generators' mutexes.\n\n"
      "This is necessary to prevent generator state conflicts in between\n"
      "getting the device state tensor, executing compiled executable, and\n"
      "setting the updated device state tensor.")
      .def(pybind11::init<std::vector<at::Generator>&>())
      .def("__enter__", &MultiGeneratorLocker::Enter)
      .def("__exit__", &MultiGeneratorLocker::Exit);

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
  m.def("dynamic_placeholder", PyMakeDynamicPlaceholder, py::arg("sizes"),
        py::arg("dtype"), py::arg("tensor_bounds"),
        py::arg("requires_grad") = false);
  m.def("get_device_layout_if_materialized", &PyGetDeviceLayoutIfMaterialized,
        py::arg("tensor"));
  m.def("get_default_layout", &PyGetDefaultLayout, py::arg("dtype"),
        py::arg("shape"));
  m.def("is_device_shape_dynamic", &PyIsDeviceShapeDynamic, py::arg("tensor"));
  m.def("traverse_and_compile", PyTraverseAndCompile, py::arg("result_tensors"),
        py::arg("argument_tensors"), py::arg("fast_compile") = false,
        py::arg("build_mlir_module") = false,
        py::arg("use_stablehlo_bounds") = false,
        py::arg("argument_layouts") =
            std::vector<std::vector<int64_t>>{},  // INT_VEC_OK
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
        "Returns:\n"
        "  CompileResult: The compiled module and executable.");

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
  m.def("force_strides", PyForceStrides, py::arg("tensor"),
        py::arg("target_strides"), py::arg("target_storage_offset"),
        "Returns a tensor with equivalent logical values to the input tensor, "
        "but with the given strides and storage offset, copying "
        "data as necessary.");
  m.def("get_materialize_collective_tensors_env_value",
        PyGetMaterializeCollectiveTensorsEnvValue,
        "Returns whether to materialize collective tensors.");

  py::enum_<torch_tpu::ScanDirection>(m, "ScanDirection")
      .value("kForward", torch_tpu::ScanDirection::kForward)
      .value("kReverse", torch_tpu::ScanDirection::kReverse)
      .export_values();
  m.def("create_scan_op", &PyCreateScanOp, py::arg("inits"), py::arg("inputs"),
        py::arg("body_module"), py::arg("scan_direction"), py::arg("dummy_ys"),
        py::arg("num_scan_inputs"),
        "Creates a deferred scan operation and returns the output tensors.");
}

}  // namespace torch_tpu
