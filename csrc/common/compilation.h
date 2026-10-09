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

#ifndef TORCH_TPU_CSRC_COMMON_COMPILATION_H_
#define TORCH_TPU_CSRC_COMMON_COMPILATION_H_

// Utilities for compiling PyTorch to PjRt executables.

#include <array>
#include <cstdint>
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
#include "absl/types/span.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/compile_options_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/shape.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/layout.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_device_dimensions.h"
#include "xla/pjrt/pjrt_executable.h"

namespace torch_tpu {

class LoadedExecutableWithMetadata;

// TODO(basioli): Rename to something more descriptive.
using SharedLoadedExecutableWithMetadata =
    absl_nonnull std::shared_ptr<const LoadedExecutableWithMetadata>;

// Metadata describing how logical runtime inputs and outputs map to physical
// PjRt executable parameters for precompiled unbounded-dynamic kernels
// (using disjoint S32[rank] runtime shape metadata parameters).
struct UnboundedDynamicExecutionMetadata {
  // The hooks below are plain function pointers because this metadata is
  // cached with the executable: a hook must not refer to a callable that can be
  // destroyed first.

  // Maps a logical output shape to the physical shape the executable's
  // parameter was compiled against, given that parameter's layout (e.g. a
  // differently-typed tiled view of the same bytes). If null, the donated
  // buffer is allocated with the logical shape and the parameter's layout.
  using PhysicalOutputShapeFunc =
      absl::StatusOr<xla::Shape> (*)(const Shape&, const xla::Layout&);

  // For dynamic outputs that do not alias an existing logical input,
  // specifies which logical output needs a pre-allocated device buffer passed
  // as a donated PjRt input parameter.
  struct DonatedOutputParam {
    int64_t logical_output_idx = 0;
    int64_t pjrt_param_idx = 0;
    PhysicalOutputShapeFunc absl_nullable physical_shape_fn = nullptr;
  };

  // Binds a physical PjRt parameter index for a disjoint S32[rank] metadata
  // buffer to the physical PjRt parameter index of the payload whose runtime
  // shape it describes (corresponding to operand_to_metadata). The payload may
  // be a logical input or a pre-allocated donated output; either way the
  // metadata reports the shape the executable actually receives, after any
  // physical shape mapping. Multiple payload operands sharing the same metadata
  // parameter require only a single binding entry.
  struct MetadataParamBinding {
    int64_t buffer_pjrt_idx = 0;
    int64_t shape_pjrt_idx = 0;
  };

  // Transforms the executable's results back into their logical representation
  // (e.g. bitcasts a differently-typed tiled view back to the logical type),
  // given the logical output shapes the caller passed to `Execute`. If null,
  // `Execute` returns the results unchanged.
  //
  // The shapes are required because a physical view is generally not
  // invertible from the result buffer alone: tile padding means several
  // logical extents map to the same physical one.
  using OutputTransform = absl::StatusOr<
      std::vector<absl_nonnull std::unique_ptr<xla::PjRtBuffer>>> (*)(
      std::vector<absl_nonnull std::unique_ptr<xla::PjRtBuffer>>,
      absl::Span<const Shape>);

  Indices logical_input_to_pjrt_param;
  std::vector<DonatedOutputParam> donated_output_params;
  std::vector<MetadataParamBinding> metadata_params;
  std::vector<xla::Shape> pjrt_param_shapes;

  // Returns whether this executable can serve a call with the given logical
  // input and output shapes. Consulted when scanning cache entries that share
  // a shapeless key.
  using CompatibilityFunc = bool (*)(absl::Span<const Shape> input_shapes,
                                     absl::Span<const Shape> output_shapes);

  OutputTransform absl_nullable output_transform = nullptr;
  CompatibilityFunc is_compatible = nullptr;
};

class LoadedExecutableWithMetadata {
 public:
  // `device_ids` are the devices the executable was built for, in device
  // assignment order; empty when the executable was built without a device
  // assignment.
  static absl::StatusOr<SharedLoadedExecutableWithMetadata> MakeShared(
      absl_nonnull std::unique_ptr<xla::PjRtLoadedExecutable> executable,
      Indices argument_indices = {},
      std::optional<UnboundedDynamicExecutionMetadata>
          unbounded_dynamic_execution_metadata = std::nullopt,
      std::vector<int64_t> device_ids = {});  // INT_VEC_OK

  static SharedLoadedExecutableWithMetadata WithArgumentIndices(
      const SharedLoadedExecutableWithMetadata& existing,
      Indices argument_indices) {
    return std::shared_ptr<const LoadedExecutableWithMetadata>(
        new LoadedExecutableWithMetadata(
            existing->executable_, existing->output_shapes_,
            std::move(argument_indices),
            existing->unbounded_dynamic_execution_metadata_,
            existing->device_ids_));
  }

  static SharedLoadedExecutableWithMetadata WithDynamicExecutionMetadata(
      const SharedLoadedExecutableWithMetadata& existing,
      UnboundedDynamicExecutionMetadata unbounded_dynamic_execution_metadata) {
    return std::shared_ptr<const LoadedExecutableWithMetadata>(
        new LoadedExecutableWithMetadata(
            existing->executable_, existing->output_shapes_,
            existing->argument_indices_,
            std::move(unbounded_dynamic_execution_metadata),
            existing->device_ids_));
  }

  const xla::PjRtExecutable* GetExecutable() const {
    return executable_->GetExecutable();
  }

  const xla::PjRtLoadedExecutable* GetLoadedExecutable() const {
    return executable_.get();
  }

  const std::vector<Shape>& output_shapes() const { return output_shapes_; }
  const Indices& argument_indices() const { return argument_indices_; }
  const std::optional<UnboundedDynamicExecutionMetadata>&
  unbounded_dynamic_execution_metadata() const {
    return unbounded_dynamic_execution_metadata_;
  }
  // The devices the executable was built for; see `MakeShared()`.
  const std::vector<int64_t>& device_ids() const {  // INT_VEC_OK
    return device_ids_;
  }

 private:
  LoadedExecutableWithMetadata(
      absl_nonnull std::shared_ptr<const xla::PjRtLoadedExecutable> executable,
      std::vector<Shape> output_shapes, Indices argument_indices = {},
      std::optional<UnboundedDynamicExecutionMetadata>
          unbounded_dynamic_execution_metadata = std::nullopt,
      std::vector<int64_t> device_ids = {})  // INT_VEC_OK
      : executable_(std::move(executable)),
        output_shapes_(std::move(output_shapes)),
        argument_indices_(std::move(argument_indices)),
        unbounded_dynamic_execution_metadata_(
            std::move(unbounded_dynamic_execution_metadata)),
        device_ids_(std::move(device_ids)) {}

  absl_nonnull std::shared_ptr<const xla::PjRtLoadedExecutable> executable_;
  // Cached output shapes for the executable.
  std::vector<Shape> output_shapes_;
  Indices argument_indices_;
  std::optional<UnboundedDynamicExecutionMetadata>
      unbounded_dynamic_execution_metadata_;
  std::vector<int64_t> device_ids_;  // INT_VEC_OK
};

// Returns the devices `compile_options` build an executable for, in device
// assignment order; empty without a device assignment.
[[nodiscard]] std::vector<int64_t> DeviceIdsOf(  // INT_VEC_OK
    const xla::CompileOptions& compile_options);

// Compiles `module` with `options` and loads the executable. A program of a
// proper subgroup of the devices with no device on the chip at the origin of
// the slice runs `ragged_all_to_all` on the TensorCores: the SparseCore
// implementation in libtpu 0.0.49 and older sends to the wrong chips from such
// a subgroup. A program of a subgroup that fills a box of devices off the
// slice origin is compiled against a sub-slice of the box's shape at the slice
// origin and loaded at the box, so its serialized form loads there in any
// process (see `LoadSerializedExecutableForDevices()`).
absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> CompileAndLoad(
    xla::PjRtClient& client, xla::MaybeOwningMlirModule module,
    xla::CompileOptions options);

// Loads `serialized`, the serialized form of an executable built for
// `device_ids`. When the devices fill a box off the slice origin (see
// `CompileAndLoad()`), loads it at the box and checks that it runs on the
// devices of this process among `device_ids`.
absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
LoadSerializedExecutableForDevices(xla::PjRtClient& client,
                                   std::string_view serialized,
                                   absl::Span<const int64_t> device_ids);

// Returns the options for loading the serialized form of an executable built
// for `device_ids`: the executable of a box of devices off the slice origin
// loads at the box's origin (see `CompileAndLoad()`).
absl::StatusOr<xla::LoadOptions> LoadOptionsForDevices(
    const xla::PjRtClient& client, absl::Span<const int64_t> device_ids);

// The chip coordinates of a device followed by its core index on the chip.
using DevicePosition = std::array<int64_t, 4>;

// A box of chips that the devices of a subgroup fill densely, counting the
// cores of each chip as a fourth axis.
struct SubsliceBox {
  // The chip of the box at the lowest coordinates.
  xla::PjRtDeviceDimensions origin;
  // The number of chips along each axis.
  xla::PjRtDeviceDimensions chips;
};

// Returns the box of the devices at `positions` when they are fewer than the
// `num_slice_devices` devices of the slice and fill a box of devices off the
// origin of the slice, and nullopt otherwise.
std::optional<SubsliceBox> OffOriginBoxOf(
    absl::Span<const DevicePosition> positions, int64_t num_slice_devices);

// Returns the chips of each process and the processes of the sub-slice that
// tiles a box of `box_chips` chips, with at most `chips_per_process` chips in
// each process.
std::pair<xla::PjRtDeviceDimensions, xla::PjRtDeviceDimensions>
SubsliceTilingOf(const xla::PjRtDeviceDimensions& box_chips,
                 const xla::PjRtDeviceDimensions& chips_per_process);

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
  // TODO(b/563604764): Rename this field as it can also hold dynamic kernels.
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

  // The devices the module's ops resolve its program to, or an empty list for
  // the default devices.
  [[nodiscard]] const std::vector<int64_t>& device_assignment()  // INT_VEC_OK
      const {
    return device_assignment_;
  }
  void set_device_assignment(
      std::vector<int64_t> device_assignment) {  // INT_VEC_OK
    device_assignment_ = std::move(device_assignment);
  }

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
  std::vector<int64_t> device_assignment_;  // INT_VEC_OK
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

// Returns a shared pointer to the active thread-local immutable
// CompilationContext (containing pre-resolved CompilationSpecs for all
// compilation modes) without deep-copying xla::CompileOptions.
//
// Because CompilationContext is immutable and PushCompilerOptionOverrides()
// always allocates a new CompilationContext instance rather than mutating an
// existing one in place, the returned shared_ptr safely keeps the resolved
// CompilationSpecs alive across scope exits or threads.
[[nodiscard]] absl_nonnull std::shared_ptr<const CompilationContext>
GetCompilationContext();

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
    const xla::CompileOptions& options);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_COMPILATION_H_
