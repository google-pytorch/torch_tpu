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

#ifndef TORCH_TPU_TESTS_COMPILED_MODE_H_
#define TORCH_TPU_TESTS_COMPILED_MODE_H_

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/compilation_spec.h"
#include "torch_tpu/common/shape.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"

// The core functions used for Dynamo compiled mode.
// Python bindings are declared in tpu_torch_compile.cc.

namespace torch_tpu {

// Creates a contiguous tensor with a new, unshared c10::Storage and a
// kPlaceholder DeviceBufferRef.
//
// We want all arguments to the FX graph to be the arguments to the compiled
// PjRtLoadedExecutable. However, the example inputs provided by torch.compile
// might be views or have deferred ops; this would change the shape of the
// deferred op graph even if the FX graph is the same, creating inconsistency
// between PyTorch's expectations and PjRt's expectations.
//
// To handle this, we create a new contiguous tensor with a kPlaceholder buffer
// matching the shape and type of the original tensor. This ensures that the
// traced graph will depend only on the shape and dtype of the example inputs
// and will use them as arguments. At execution time, we will materialize the
// inputs and pass them as PjRtBuffers (which also only have shape and dtype,
// not stride/offset/storage nbytes etc.) This aligns PyTorch Dynamo and PjRt's
// expectations for the behavior of the compiled mode output.
absl::StatusOr<at::Tensor> MakePlaceholder(absl::Span<const int64_t> sizes,
                                           at::ScalarType dtype,
                                           bool requires_grad);
absl::StatusOr<at::Tensor> MakePlaceholder(Shape shape, bool requires_grad);

// Extracts the MLIR module from the given graph without compiling or
// materializing.
//
// Args:
//   mlir_context: The MLIR context to use for the module.
//   result_tensors: The result tensors of the graph.
//   arg_tensors: The argument tensors of the graph.
//   use_stablehlo_bounds: If true, use StableHLO style bounds in the generated
//     MLIR for dynamic inputs
//
// Returns:
//   The MLIR module.
//
// If the tensors are already materialized, will return status.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ExtractMlirFromGraph(
    mlir::MLIRContext& mlir_context, const std::vector<at::Tensor>& arg_tensors,
    const std::vector<at::Tensor>& result_tensors, bool use_stablehlo_bounds);

// Result of TraverseAndCompile, containing the compiled executable and
// optionally additional compilation artifacts (like the MLIR module) if
// requested.
struct CompileResult {
  absl_nullable std::shared_ptr<ContextedModule> module = nullptr;
  SharedLoadedExecutableWithMetadata executable;
};

struct TraverseAndCompileOptions {
  // The compilation mode to use.
  CompilationMode compilation_mode = CompilationMode::kFastRuntime;
  // If true, build the MLIR module and return it in the CompileResult.
  bool build_mlir_module = false;
  // If true, use StableHLO style bounds in the generated MLIR for dynamic
  // inputs
  bool use_stablehlo_bounds = false;
  // Forced layouts for arguments.
  std::vector<Indices> argument_layouts;
};

// Traverses the graph from outputs to arguments and compiles it.
// Hides the traversal logic within its implementation.
absl::StatusOr<CompileResult> TraverseAndCompile(
    const std::vector<at::Tensor>& result_tensors,
    const std::vector<at::Tensor>& argument_tensors,
    const TraverseAndCompileOptions& options = {});

// torch.compile integration: this is called by torch.compile() to compile the
// FX graph, after it has generated the graph and its input/output tensors. This
// function does the compilation and adds the compiled executable to the
// compilation cache.
//
// Args:
//   mlir_module_bytecode: The MLIR module bytecode.
//   compile_options: XLA configurations to generate the compiled executable.
//
// Returns:
//   The compiled executable. When the executable runs, the argument tensors
//   will be converted to PjRt buffers and passed to the compiled executable,
//   and the results will be converted back to ATen tensors
//
// Important: the argument and result tensors must be the same as those used
// by the FX graph, including their order. Otherwise, the compilation might
// fail or the materialized tensors might be wrong.
absl::StatusOr<SharedLoadedExecutableWithMetadata> CompileMlirExecutable(
    std::string_view mlir_module_bytecode,
    UniqueCompileOptions compile_options);

// torch.compile integration: this is called by torch.compile() to compile the
// FX graph, after it has generated the graph and its input/output tensors. This
// function does the compilation and adds the compiled executable to the
// compilation cache.
//
// Args:
//   module: The MLIR module to compile.
//   compile_options: XLA configurations to generate the compiled executable.
//
// Returns:
//   The compiled executable. When the executable runs, the argument tensors
//   will be converted to PjRt buffers and passed to the compiled executable,
//   and the results will be converted back to ATen tensors
//
// Important: the argument and result tensors must be the same as those used
// by the FX graph, including their order. Otherwise, the compilation might
// fail or the materialized tensors might be wrong.
absl::StatusOr<SharedLoadedExecutableWithMetadata> CompileMlirExecutable(
    xla::MaybeOwningMlirModule module, UniqueCompileOptions compile_options);

// torch.compile integration: this is called to invoke the compiled executable
// returned by CompileMlirExecutable, and return new Tensors with the results.
//
// This execution will be added to an async background queue (which is shared
// with eager execution), so future calls to .cpu() may need to block to wait
// for the execution to be started and/or finished. However, the main thread
// may continue enqueuing other executions if desired.
//
// Args:
//   executable: The compiled executable to invoke.
//   argument_tensors: The argument tensors to the compiled executable.
//
// Returns:
//   The result tensors of the compiled executable.
std::vector<at::Tensor> ExecuteCompiledModel(
    const SharedLoadedExecutableWithMetadata& executable,
    absl::Span<const at::Tensor> argument_tensors,
    absl::Span<const std::vector<int64_t>>  // INT_VEC_OK
        output_shapes);

// Creates a TPU tensor that represents the constant value of the CPU tensor.
// This makes a copy of the CPU tensor data.
absl::StatusOr<at::Tensor> MakeConstantTensor(const at::Tensor& cpu_tensor);

// Updates tpu_dst_tensor to be a tensor with the constant value of the CPU
// tensor. This makes a copy of the CPU tensor data.
// Errors if the tensors are of different shapes or element types, or are not
// on the correct device respectively.
absl::Status AssignConstantTensor(const at::Tensor& cpu_src_tensor,
                                  const at::Tensor& tpu_dst_tensor);

}  // namespace torch_tpu

#endif  // TORCH_TPU_TESTS_COMPILED_MODE_H_
