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

#ifndef TORCH_TPU_EAGER_TRAVERSAL_H_
#define TORCH_TPU_EAGER_TRAVERSAL_H_

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/ops/python_context.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

// Creates a core abstraction for the process of traversing a graph of deferred
// ops, preparing them for compilation and execution.
//
// When a deferred operation is dispatched, we create a new node (as a
// DeviceBufferRef in the kDeferred state) in a directed graph.
// This node will retain shared pointers (as c10::Storages) to previously-
// constructed nodes. Because DeviceBufferRefs are immutable, this construction
// guarantees that the graph is acyclic.
//
// In order to compile the graph with one or more outputs, we need to identify
// its set of inputs, and the full structure of the dependencies between inputs
// and the outputs.
//
// Input nodes are leaves (i.e. they have no outgoing pointers), which includes
// these DeviceBufferRef states:
//   - kMaterialized: the DeviceBufferRef references a materialized PjRtBuffer.
//                    This PjRtBuffer must be passed to the final compiled
//                    PjRtLoadedExecutable as an argument at execution time.
//   - kZeroSize:     the DeviceBufferRef represents zero data. This means it
//                    functions as a constant. This is an input/leaf node, but
//                    is not an argument, and will be "baked in" to the final
//                    compiled executable rather than passed in as a parameter.
//   - kPlaceholder:  The DeviceBufferRef is a placeholder for a future
//                    argument. This can be used in compiled mode to create an
//                    executable without actually loading data on-device.
//                    Future executions are expected to provide a kMaterialized
//                    DeviceBufferRef for the placeholder.
//
// In eager mode, the outputs are known at traversal time, but the inputs need
// to be identified. In this case, we traverse the graph from the outputs to
// the inputs (using Create), building the list of arguments in a
// deterministic but unspecified order. Once the Traversal is complete, the
// main function can be built (using BuildMlirFunction), compiled
// (see compilation_cache.h) and then executed (filtering inputs to only the
// argument PjRtBuffers needed for execution).
//
// In compiled mode, we know both the inputs and outputs at traversal time, but
// not the graph relationship between them.
// Additionally, DAG traversal orderings are not unique. To ensure that the
// Python-level input ordering is aligned with the PjRtLoadedExecutable
// argument ordering, we need to explicitly reorder the list of arguments, using
// ValidateAndReorderArguments. Once reordered, the compilation can use
// BuildMlirFunction.
// Because compiled mode does not immediately execute the compiled executable,
// it is valid to compile using compilation placeholder inputs, and
// only later provide materialized arguments when actually executing (see
// "compiled_mode.cc" and "tpu_torch_compile.cc" for this interface).

namespace torch_tpu {

using RefToOpMap = absl::flat_hash_map<DeviceBufferRef, mlir::MlirOp>;

// A traversed graph of deferred ops, ready to be compiled and (optionally)
// executed.
// TODO(bawilson): add tests once at::Tensor dependency is removed from core
// graph initialization
class Traversal {
 public:
  // Traversal cannot be directly constructed; use Create() instead.
  Traversal() = delete;

  // Delete copy constructor and copy assignment.
  Traversal(const Traversal&) = delete;
  Traversal(Traversal&&) = default;
  Traversal& operator=(const Traversal&) = delete;
  Traversal& operator=(Traversal&&) = default;

  // Traverses the deferred operations from outputs (roots of the directed
  // acyclic graph) to inputs (leaves of the directed acyclic graph), inferring
  // arguments to be all non-constant inputs/leaves in a deterministic but
  // unspecified order. Optional parameter `stopping_points` can be used to
  // supply DeviceBufferLists that should stop the traversal and, hence, be
  // treated as inputs (even though they may be associated to deferred ops).
  // We call the union of the graph's inputs and `stopping_points` the
  // traversal's "extended inputs".
  static absl::StatusOr<Traversal> Create(
      std::vector<DeviceBufferRef> outputs,
      const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
          stopping_points = {});
  static absl::StatusOr<Traversal> Create(
      absl::Span<const SharedDeviceBufferList> output_nodes,
      const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
          stopping_points = {});

  // Creates a traversal such that:
  //
  // * the traversal execution order is the same as the ops are listed in
  //   `region`;
  //
  // * the traversal inputs are the op inputs in `region` that are not defined
  //   by any other op in `region`;
  //
  // * the traversal outputs are the op outputs in `region` that are not
  //   consumed by any other op in `region`.
  static absl::StatusOr<Traversal> CreateFromLinearRegion(
      absl::Span<const SharedDeviceBufferList> region);

  CompilationCacheKey cache_key() const {
    if (!cache_key_) {
      cache_key_ = BuildCacheKey();
    }
    return *cache_key_;
  }

  // Validates that the provided arguments are a valid reordering of the
  // Traversal's arguments and/or zero_sized_constants, and if they are,
  // overwrites the previous arguments_ with
  // these new inputs. This is used by compiled_mode.cc to align argument
  // ordering between Python and PjRt interfaces.
  // The provided arguments must include all prior non-zero-sized arguments; it
  // may or may not include zero-sized DeviceBufferRefs.
  absl::Status ValidateAndReorderArguments(
      std::vector<DeviceBufferRef> arguments);

  // Returns the graph arguments that should be used to call an executable
  // compiled from this traversal.
  [[nodiscard]] absl::Span<const DeviceBufferRef> arguments() const {
    return arguments_;
  }
  // Returns any zero-sized tensors which are baked into the compiled executable
  // as constants.
  [[nodiscard]] absl::Span<const DeviceBufferRef> zero_sized_constants() const {
    return zero_sized_constants_;
  }

  // Returns a topological sort of the deferred ops in the graph needed for
  // computing the outputs. This contains all nodes between the extended inputs
  // (not inclusive) and the outputs (inclusive).
  //
  // Invariant: all nodes in this span are deferred ops.
  [[nodiscard]] absl::Span<const SharedDeviceBufferList> execution_order()
      const {
    return execution_order_;
  }

  // Returns the graph outputs.
  [[nodiscard]] absl::Span<const DeviceBufferRef> outputs() const {
    return outputs_;
  }

  // Returns a string representation of the Traversal for debugging purposes.
  // This is extremely verbose and should not be exposed in user-facing error
  // messages.
  std::string DebugString() const;

  // Builds the MLIR module for the Traversal.
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> BuildMlirModule(
      mlir::MLIRContext& mlir_context) const;

  // Compiles the Traversal into a CompiledKernel. For static graphs, this will
  // be a single executable future. For bounded dynamic graphs, this will
  // in addition contain futures for dynamic adapters.
  absl::StatusOr<CompiledKernel> Compile(
      CompilationMode compilation_mode) const;

  // Returns true if any input to the traversal has bounded dynamic dimensions
  // marked.
  [[nodiscard]] bool IsBoundedDynamic() const;

  // The core components of a Traversal, returned by IntoParts().
  struct Parts {
    std::vector<DeviceBufferRef> arguments;
    std::vector<DeviceBufferRef> zero_sized_constants;
    std::vector<SharedDeviceBufferList> execution_order;
    std::vector<DeviceBufferRef> outputs;
  };

  // Breaks the Traversal into its core components, leaving an empty (and
  // invalid) Traversal behind. This is used once the invariants upheld by the
  // Traversal are no longer required.
  [[nodiscard]] Parts IntoParts() {
    return {std::move(arguments_), std::move(zero_sized_constants_),
            std::move(execution_order_), std::move(outputs_)};
  }

  // Returns the Python context for the Traversal.
  const PythonContext* absl_nullable GetPythonContext() const;

  // Sorts the Traversal's execution_order by op creation index.
  inline void SortByCreationOrder() {
    std::sort(
        execution_order_.begin(), execution_order_.end(),
        [](const SharedDeviceBufferList& a, const SharedDeviceBufferList& b) {
          if (const auto* a_deferred_op = a->deferred_op()) {
            if (const auto* b_deferred_op = b->deferred_op()) {
              return a_deferred_op->creation_index() <
                     b_deferred_op->creation_index();
            }
          }
          // This should never happen.
          return true;
        });
  }

 private:
  // Private constructor, only called by Traversal::Create().
  // Definitions:
  //   - arguments:            the arguments to the Traversal. At runtime, these
  //                           must be provided as materialized PjRtBuffers to
  //                           invoke the compiled executable.
  //   - zero_sized_constants: zero-sized tensors that are baked into the
  //                           compiled executable as constants.
  //   - execution_order:      deferred nodes in the graph. These may be leaf
  //                           nodes (for nullary ops), but are more commonly
  //                           branch or root nodes.
  //   - outputs:              the final outputs of the computation. These can
  //                           be any nodes in the graph, but are most commonly
  //                           branch or root nodes.
  //
  // Properties:
  //   - All tensors (if any) in arguments are unique
  //   - All tensors (if any) in zero_sized_constants are unique
  //   - No tensor appears in both arguments and zero_sized_constants.
  //   - All tensors (if any) in zero_sized_constants are zero-sized.
  //   - All tensors (if any) in execution_order are unique.
  //   - All tensors (if any) in execution_order are deferred nodes (kDeferred
  //   state).
  //   - Each deferred node in execution_order only depends on tensors in
  //     arguments, zero_sized_constants, and/or lower-indexed execution_order
  //     nodes.
  //   - All tensors in outputs are unique, and there is at least one output.
  //   - Each tensor in outputs is present in arguments, zero_sized_constants,
  //     or execution_order.
  Traversal(std::vector<DeviceBufferRef> arguments,
            std::vector<DeviceBufferRef> zero_sized_constants,
            std::vector<SharedDeviceBufferList> execution_order,
            std::vector<DeviceBufferRef> outputs)
      : arguments_(std::move(arguments)),
        zero_sized_constants_(std::move(zero_sized_constants)),
        execution_order_(std::move(execution_order)),
        outputs_(std::move(outputs)) {}

  CompilationCacheKey BuildCacheKey() const;

  // Validates that the traversal is sound.
  absl::Status Validate() const;

  // The arguments to the Traversal are the values that need to be provided
  // when invoking the compiled executable.
  // In eager mode, this will be all materialized inputs; zero-sized tensors
  // will be baked into the compiled executable as constants.
  // In compiled mode, these will either be placeholder inputs, or (in some
  // cases) zero-sized tensors provided as inputs to the FX graph.
  std::vector<DeviceBufferRef> arguments_;

  // In eager mode, zero-sized tensors are constants; their value never changes
  // (always "no data"). In compiled mode, zero-sized tensors can either be
  // constants, or they may be placeholder inputs.
  std::vector<DeviceBufferRef> zero_sized_constants_;

  // A execution_order traversal of the deferred ops in the graph.
  // Each op only depends on arguments, constants, and lower-indexed ops.
  std::vector<SharedDeviceBufferList> execution_order_;
  // The tensor outputs of the Traversal. These may be in any state, but the
  // list is non-empty and all outputs are unique.
  std::vector<DeviceBufferRef> outputs_;
  // The compilation cache key this Traversal; lazily computed.
  mutable std::optional<CompilationCacheKey> cache_key_;

  // The acceptable dynamic bounds for each dimension in the traversal's inputs.
  // Lazily computed.
  mutable std::optional<ShapeDynamismMetadata> shape_dynamism_metadata_;

  // Helper function for BuildMlirFunction.
  // Returns the MLIR op corresponding to the provided buffer ref,
  // the one corresponding to the dynamically redirected buffer ref, or an
  // error, with that order of precedence.
  absl::StatusOr<mlir::MlirOp> GetMlirOpForProcessedBuffer(
      const absl::flat_hash_map<DeviceBufferRef, mlir::MlirOp>& ref_to_op_map,
      const DeviceBufferRef& buffer_ref) const;
};

// A simple node traversal is one where the graph outputs are all the outputs of
// a single deferred op. This is the most common case for materializations.
// Exceptions of non-simple node traversals include:
// - Graphs which directly output their arguments
// - Graphs which materialize multiple deferred ops at once
// - Graphs which output subset or reordering of a single deferred op's outputs
bool IsSimpleNodeTraversal(const Traversal& traversal);

// Returns a graphviz compatible representation of the Traversal for debugging
// purposes, tries to label each buffer with the corresponding python
// variable name if possible.
//
// Returns an error if the traversal is invalid (e.g. if one of its nodes fails
// to be created).
absl::StatusOr<std::string> GetGraphviz(
    const Traversal& traversal,
    const absl::flat_hash_map<DeviceBufferRef, std::string>&
        buffer_ref_to_python_var = {});

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_TRAVERSAL_H_
