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
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "torch_tpu/csrc/common/compilation_spec.h"
#include "torch_tpu/csrc/common/compile_options_key.h"
#include "torch_tpu/csrc/common/shape.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"

namespace torch_tpu {

class LoadedExecutableWithMetadata;

// TODO(basioli): Rename to something more descriptive.
using SharedLoadedExecutableWithMetadata =
    absl_nonnull std::shared_ptr<const LoadedExecutableWithMetadata>;

class LoadedExecutableWithMetadata {
 public:
  static absl::StatusOr<SharedLoadedExecutableWithMetadata> MakeShared(
      std::unique_ptr<xla::PjRtLoadedExecutable> executable);

  const xla::PjRtExecutable* GetExecutable() const {
    return executable_->GetExecutable();
  }

  const xla::PjRtLoadedExecutable* GetLoadedExecutable() const {
    return executable_.get();
  }

  const std::vector<Shape>& output_shapes() const { return output_shapes_; }

 private:
  LoadedExecutableWithMetadata(
      absl_nonnull std::unique_ptr<const xla::PjRtLoadedExecutable> executable,
      std::vector<Shape> output_shapes)
      : executable_(std::move(executable)),
        output_shapes_(std::move(output_shapes)) {}

  absl_nonnull std::unique_ptr<const xla::PjRtLoadedExecutable> executable_;
  // Cached output shapes for the executable.
  std::vector<Shape> output_shapes_;
};

using LoadedExecutablePromise =
    std::promise<absl::StatusOr<SharedLoadedExecutableWithMetadata>>;
using SharedLoadedExecutableWithMetadataFuture =
    std::shared_future<absl::StatusOr<SharedLoadedExecutableWithMetadata>>;

// Holds the futures for a dynamic kernel adapter. These are lightweight
// executables, e.g. padding and slicing ops.
struct DynamicKernelAdapter {
  SharedLoadedExecutableWithMetadataFuture preamble;
  SharedLoadedExecutableWithMetadataFuture postamble;
};

// Holds the futures for a compiled kernel. For static kernels, only the
// fixed_shape_kernel future will be set. For bounded dynamic kernels, both the
// fixed_shape_kernel and dynamic_kernel_adapter will be set.
struct CompiledKernel {
  SharedLoadedExecutableWithMetadataFuture fixed_shape_kernel;
  std::optional<DynamicKernelAdapter> dynamic_kernel_adapter;
};

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

// Lifts a MlirComputationBuilder into an ExecutableBuilder.
absl::StatusOr<LoadedExecutableBuilder>
MlirComputationBuilderToExecutableBuilder(
    const MlirComputationBuilder& computation_builder);

// Creates an MLIR context and registers all required TPU dialects.
absl_nonnull std::unique_ptr<mlir::MLIRContext> MakeMlirContext();

// Synchronously compiles a program, without any cache behavior.
absl::StatusOr<SharedLoadedExecutableWithMetadata> Compile(
    xla::PjRtClient& client, LoadedExecutableBuilder executable_builder,
    UniqueCompileOptions compile_options);

// Returns resolved compilation settings for the given compilation mode.
// The returned spec includes XLA compile options (with thread-local context
// overrides applied) and its fingerprint.
[[nodiscard]] CompilationSpec GetCompilationSpec(CompilationMode mode);

// Pushes the compile option overrides for the current thread on to the
// custom compiler option stack. Thread-safe.
//
// Returns an error when it fails to make XLA compile options due to invalid
// environment settings or user-specified options. On errors, the compiler
// option overrides take no effect, i.e. the stack remains unchanged.
absl::Status PushCompilerOptionOverrides(CompilerOptionOverrides overrides);

// Pops the compile option overrides for the current thread from the
// custom compiler option stack. Thread-safe. Requires that the stack is
// non-empty.
void PopCompilerOptionOverrides();

// Returns the compile options key for the given compile options.
[[nodiscard]] CompileOptionsKey MakeCompileOptionsKey(
    std::string_view xla_flags, const xla::CompileOptions& options);

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_COMPILATION_H_
