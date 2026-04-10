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

#ifndef TORCH_TPU_COMMON_COMPILATION_H_
#define TORCH_TPU_COMMON_COMPILATION_H_

// Utilities for compiling PyTorch to PjRt executables.

#include <future>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {

using SharedLoadedExecutable =
    absl_nonnull std::shared_ptr<const xla::PjRtLoadedExecutable>;

using LoadedExecutablePromise =
    std::promise<absl::StatusOr<SharedLoadedExecutable>>;
using SharedLoadedExecutableFuture =
    std::shared_future<absl::StatusOr<SharedLoadedExecutable>>;

// Holds the futures for a dynamic kernel adapter. These are lightweight
// executables, e.g. padding and slicing ops.
struct DynamicKernelAdapter {
  SharedLoadedExecutableFuture preamble;
  SharedLoadedExecutableFuture postamble;
};

// Holds the futures for a compiled kernel. For static kernels, only the
// fixed_shape_kernel future will be set. For bounded dynamic kernels, both the
// fixed_shape_kernel and dynamic_kernel_adapter will be set.
struct CompiledKernel {
  SharedLoadedExecutableFuture fixed_shape_kernel;
  std::optional<DynamicKernelAdapter> dynamic_kernel_adapter;
};

// CompileOptions is a complex object with many fields. Even moving it is
// expensive. We wrap it in a unique_ptr to allow for cheap moving.
using UniqueCompileOptions = absl_nonnull std::unique_ptr<xla::CompileOptions>;

// Builder function for a loaded executable.
// It transfers ownership of the underlying MLIR module and context to the
// `PjRtClient` during compilation, and thus can only be invoked once. The `&&`
// qualifier enforces this constraint: it ensures that the builder's call
// operator may only be invoked when the builder is an rvalue, thus consuming
// the builder function and preventing it from being called more than once.
using LoadedExecutableBuilder = absl::AnyInvocable<
    absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>(
        xla::PjRtClient& client, UniqueCompileOptions compile_options) &&>;

// An MLIR module, with the MLIR context it requires.
// TODO(b/487928705): Remove this class and use `MaybeOwningMlirModule`
// directly.
class ContextedModule {
 public:
  explicit ContextedModule(std::unique_ptr<mlir::MLIRContext> context,
                           mlir::OwningOpRef<mlir::ModuleOp> module);

  // This class is move-only.
  ContextedModule(const ContextedModule&) = delete;
  ContextedModule& operator=(const ContextedModule&) = delete;
  ContextedModule(ContextedModule&&) = default;
  ContextedModule& operator=(ContextedModule&&) = default;

  // Makes a ContextedSharedModule for the given computation builder.
  static absl::StatusOr<ContextedModule> Make(
      const MlirComputationBuilder& computation_builder);

  // Returns the underlying MLIR module.
  [[nodiscard]] mlir::ModuleOp get() const { return *module_; }

  // Returns the underlying MLIR context.
  [[nodiscard]] mlir::MLIRContext& context() { return *context_; }

  // Destructively converts this ContextedModule into a MaybeOwningMlirModule.
  [[nodiscard]] xla::MaybeOwningMlirModule ToMaybeOwningMlirModule() && {
    return xla::MaybeOwningMlirModule(std::move(context_), std::move(module_));
  }

 private:
  ContextedModule() = delete;

  // The MLIR context for the module. This must outlive the module.
  // We use a unique_ptr to allow for moving, as MLIRContext is not
  // movable.
  absl_nonnull std::unique_ptr<mlir::MLIRContext> context_;
  // The module. If the context is dropped, the module will be invalidated.
  mlir::OwningOpRef<mlir::ModuleOp> module_;
};

// Mode for compiling a computation graph.
enum class CompilationMode {
  kFastCompile,  // Reduces compile time, but may result in slower execution.
  kFastRuntime,  // Produces more optimized executables, but with longer
                 // compile.
};

// Maps an XLA compiler option name to its string value. We pick this
// representation for easy interop with Python.
using CompilerOptionOverrides = std::map<std::string, std::string>;

// Lifts a MlirComputationBuilder into an ExecutableBuilder.
absl::StatusOr<LoadedExecutableBuilder>
MlirComputationBuilderToExecutableBuilder(
    const MlirComputationBuilder& computation_builder);

// Synchronously compiles a program, without any cache behavior.
absl::StatusOr<SharedLoadedExecutable> Compile(
    xla::PjRtClient& client, LoadedExecutableBuilder executable_builder,
    UniqueCompileOptions compile_options);

// Returns the compiler options for the current host/thread and the given graph
// compilation mode. Returns an error if the replica count for compilation
// cannot be determined.
//
// The options are constructed from the following sources, in order of
// precedence:
//  - compiler options set via environment variable
//    TORCH_TPU_INTERNAL_XLA_OPTIONS, which is an internal API and shouldn't be
//    used by users directly,
//  - explicitly-specified compiler options set via
//    PushCompilerOptionOverrides() (or via the custom_compiler_options context
//    manager in Python); with nested custom_compiler_options contexts, the
//    innermost context takes precedence,
//  - hard-coded default compiler options,
//  - the global device count,
//  - debug options set via environment variable XLA_FLAGS,
//
// The format of TORCH_TPU_INTERNAL_XLA_OPTIONS is a space-separated list of
// key=value pairs, e.g. "xla_optimization_level=O1
// xla_tpu_enable_deduplicated_calls=AUTO".
// Valid options for TORCH_TPU_INTERNAL_XLA_OPTIONS are documented on
// https://openxla.org/xla/flags_guidance
absl::StatusOr<UniqueCompileOptions> MakeCompilerOptions(CompilationMode mode);

// Pushes the compile option overrides for the current thread on to the
// custom compiler option stack. Thread-safe.
void PushCompilerOptionOverrides(CompilerOptionOverrides overrides);

// Pops the compile option overrides for the current thread from the
// custom compiler option stack. Thread-safe. Requires that the stack is
// non-empty.
void PopCompilerOptionOverrides();

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_COMPILATION_H_
