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

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
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
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/compilation_cache.h"
#include "torch_tpu/common/compilation_spec.h"
#include "torch_tpu/common/compile_options_key.h"
#include "torch_tpu/common/context_manager.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
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
#include "torch_tpu/ops/view_decomposition/contiguous_to_view.h"
#include "torch_tpu/ops/view_decomposition/decomposition.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
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

  // Finally, we create a view of the base tensor with the same striding as the
  // original view, and preserve the requires_grad property.
  at::Tensor view_tensor = base_tensor.as_strided(
      arg_tensor.sizes(), arg_tensor.strides(), arg_tensor.storage_offset());
  if (arg_tensor.requires_grad()) {
    view_tensor.requires_grad_(true);
  }

  return view_tensor;
}

py::object PyGetDeviceLayoutIfMaterialized(const at::Tensor& tensor) {
  TT_ASSIGN_OR_THROW(DeviceBufferRef buffer_ref, GetBuffer(tensor));
  if (!buffer_ref.IsMaterialized()) {
    return py::none();
  }
  TT_ASSIGN_OR_THROW(auto* pjrt_buffer, buffer_ref.AwaitBuffer());
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
    const std::vector<std::pair<std::vector<int64_t>  // INT_VEC_OK
                                ,
                                at::ScalarType>>& tensor_info,
    const std::vector<std::pair<std::vector<int64_t>  // INT_VEC_OK
                                ,
                                std::vector<int64_t>  // INT_VEC_OK
                                >>& bounds_list) {
  std::vector<Shape> dynamic_shapes;
  dynamic_shapes.reserve(tensor_info.size());
  for (int i = 0; i < tensor_info.size(); ++i) {
    const auto& [dim_sizes, dtype] = tensor_info[i];
    const auto& [dynamic_dims, upper_bounds] = bounds_list[i];
    Dimensions dims = CopyIntVector(dim_sizes);
    TT_ASSIGN_OR_THROW(mlir::ElementType element_type,
                       ConvertTo<mlir::ElementType>(dtype));
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
    MlirComputationBuilder builder, const GraphKey& cache_key,
    const std::vector<Shape>& input_shapes,
    const std::vector<Shape>& output_shapes, CompilationMode compilation_mode,
    bool build_mlir_module, bool is_caching_disabled) {
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
                            compilation_mode));
  } else {
    TT_ASSIGN_OR_RETURN(CompilationSpecsByMode compilation_specs,
                        MakeCompilationSpecs(compilation_mode));
    UniqueCompileOptions compile_options =
        std::move(compilation_specs.at(compilation_mode).xla_compile_options);

    TT_ASSIGN_OR_RETURN(
        CompiledKernel compiled_kernel,
        CompilationCache::GetInstance().GetOrCompile(
            CompilationCacheKey(cache_key, CompileOptionsKey(0)), input_shapes,
            output_shapes, std::move(builder), std::move(compile_options)));

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
//   func.func @main(%arg0: tensor<1x4xi64>) -> (tensor<1x8xi64>, tensor<i32>) {
//     %c = stablehlo.constant dense<0> : tensor<i64>
//     %0 = stablehlo.pad %arg0, %c, low = [0, 0], high = [0, 4],
//       interior = [0, 0] : (tensor<1x4xi64>, tensor<i64>) -> tensor<1x8xi64>
//     %1 = stablehlo.get_dimension_size %arg0, dim = 1 :
//        (tensor<1x4xi64>) -> tensor<i32>
//     return %0, %1 : tensor<1x8xi64>, tensor<i32>
//   }
// }
CompileResult PyGetOrCompilePadModule(
    const std::vector<
        std::pair<std::vector<int64_t>, at::ScalarType>>&  // INT_VEC_OK
        tensor_info,
    const std::vector<
        std::pair<std::vector<int64_t>, std::vector<int64_t>>>&  // INT_VEC_OK
        bounds_list,
    const bool fast_compile, const bool build_mlir_module,
    const bool is_caching_disabled) {
  std::vector<Shape> dynamic_shapes =
      MakePadDynamicShapes(tensor_info, bounds_list);

  MlirComputationBuilder pad_builder = [&](mlir::MLIRContext& mlir_context) {
    return GetPadModule(mlir_context, dynamic_shapes);
  };

  auto compilation_mode = fast_compile ? CompilationMode::kFastCompile
                                       : CompilationMode::kFastRuntime;
  GraphKey key = PadModuleCacheKey(dynamic_shapes);
  std::vector<Shape> runtime_input_shapes = ToStaticShapes(dynamic_shapes);
  std::vector<Shape> cache_output_shapes = ToPaddedShapes(dynamic_shapes);

  TT_ASSIGN_OR_THROW(
      CompileResult result,
      CompileModuleWithCache(std::move(pad_builder), key, runtime_input_shapes,
                             cache_output_shapes, compilation_mode,
                             build_mlir_module, is_caching_disabled));
  return result;
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

  GraphKey key = SliceModuleCacheKey(inputs.target_dims, inputs.padded_dims,
                                     inputs.element_types);
  // For slice, the inputs to the compiled module are the padded shapes, and
  // outputs are target shapes.
  const std::vector<Shape>& runtime_input_shapes = inputs.input_shapes;
  const std::vector<Shape>& cache_output_shapes = inputs.output_shapes;

  auto compilation_mode = fast_compile ? CompilationMode::kFastCompile
                                       : CompilationMode::kFastRuntime;
  TT_ASSIGN_OR_THROW(CompileResult result,
                     CompileModuleWithCache(
                         std::move(slice_builder), key, runtime_input_shapes,
                         cache_output_shapes, compilation_mode,
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
}

}  // namespace torch_tpu
