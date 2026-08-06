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
#include <memory>
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
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/compilation_spec.h"
#include "torch_tpu/common/compile_options_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/ops/python_context.h"

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
// its set of arguments, and the full structure of the dependencies between
// arguments and the outputs.
//
// Arguments nodes are any referenced DeviceBufferRef that is not in the
// kDeferred state. This includes:
//   - kMaterialized: the DeviceBufferRef references a materialized PjRtBuffer.
//   - kPlaceholder:  The DeviceBufferRef is a placeholder for a future
//                    argument. This can be used in compiled mode to create an
//                    executable without actually loading data on-device.
// Calling the compiled executable will require a kMaterialized DeviceBufferRef
// for each argument.
//
// In eager mode, the outputs are known at traversal time, but the arguments
// need to be identified. In this case, we traverse the graph from the outputs
// to the arguments (using Create), building the list of arguments in a
// deterministic but unspecified order. Once the Traversal is complete, the
// main function can be built (using BuildMlirFunction), compiled
// (see compilation_cache.h) and then executed.
//
// In compiled mode, we know both the arguments and outputs at traversal time,
// but not the graph relationship between them.
// Additionally, DAG traversal orderings are not unique. To ensure that the
// Python-level argument ordering is aligned with the PjRtLoadedExecutable
// argument ordering, we need to explicitly reorder the list of arguments, using
// ValidateAndReorderArguments. Once reordered, the compilation can use
// BuildMlirFunction.
// Because compiled mode does not immediately execute the compiled executable,
// it is valid to compile using compilation placeholder arguments, and
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

  // Delete copy and move constructors and assignment operators.
  Traversal(const Traversal&) = delete;
  Traversal(Traversal&&) = delete;
  Traversal& operator=(const Traversal&) = delete;
  Traversal& operator=(Traversal&&) = delete;

  // Traverses the deferred operations from outputs (roots of the directed
  // acyclic graph) to arguments (leaves of the directed acyclic graph),
  // recording arguments in a deterministic but unspecified order.
  // Optional parameter `stopping_points` can be used to supply
  // DeviceBufferLists that should stop the traversal and, hence, be
  // treated as arguments (even though they may be associated to deferred ops).
  static absl::StatusOr<absl_nonnull std::unique_ptr<Traversal>> Create(
      std::vector<DeviceBufferRef> outputs,
      const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
          stopping_points = {});
  static absl::StatusOr<absl_nonnull std::unique_ptr<Traversal>> Create(
      absl::Span<const SharedDeviceBufferList> output_nodes,
      const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
          stopping_points = {});

  // Creates a traversal from a given execution order and a given set of
  // outputs.
  static absl::StatusOr<absl_nonnull std::unique_ptr<Traversal>>
  CreateFromExecutionOrder(
      absl::Span<const SharedDeviceBufferList> execution_order,
      absl::Span<const SharedDeviceBufferList> nodes_to_materialize);

  // Composes a cache key for the traversal with a specific compilation setting.
  CompilationCacheKey GetCacheKey(CompileOptionsKey compile_options_key) const {
    if (graph_key_ == std::nullopt) {
      graph_key_ = BuildGraphKey();
    }
    return CompilationCacheKey(*graph_key_, compile_options_key);
  }

  // Validates that the provided arguments are a valid reordering of the
  // Traversal's arguments, and if they are, overwrites the previous arguments_
  // with the new order. This is used by compiled_mode.cc to align argument
  // ordering between Python and PjRt interfaces.
  // The provided arguments must be a superset of DeviceBufferRefs as the
  // existing arguments_, but may be in a different order, and may contain
  // additional (unused) arguments.
  absl::Status ValidateAndReorderArguments(
      std::vector<DeviceBufferRef> arguments);

  // Returns the graph arguments.
  [[nodiscard]] absl::Span<const DeviceBufferRef> arguments() const {
    return arguments_;
  }

  // Returns a topological sort of the deferred ops in the graph needed for
  // computing the outputs. This contains all nodes between the arguments
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

  // Returns an FX-style human-readable graph dump suitable for the
  // aten_graph_payload of a StructuredLogEvent.
  //
  // Format:
  //   # Graph: <N> ops, <M> inputs, reason: <reason>
  //   %0: f32[1, 2, 3] = input
  //   %1: f32[2, 3] = mm.default(%0, %2)    # file.py:25 in forward
  //   return %1
  std::string ReadableString(MaterializationReason reason) const;

  // Builds the MLIR module for the Traversal.
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> BuildMlirModule(
      mlir::MLIRContext& mlir_context, bool use_stablehlo_bounds = false) const;

  // Compiles the Traversal into a CompiledKernel. For static graphs, this will
  // be a single executable future. For bounded dynamic graphs, this will
  // in addition contain futures for dynamic adapters (unless
  // `use_stablehlo_bounds` is true).
  //
  // If `out_mlir_text` is non-null and Compile is a cache miss (i.e. the
  // MLIR module is actually built), it is populated with the textual
  // representation of that module.
  absl::StatusOr<CompiledKernel> Compile(
      CompilationSpec spec, std::string* absl_nullable out_mlir_text = nullptr,
      bool use_stablehlo_bounds = false,
      absl::Span<const Indices> argument_layouts = {}) const;

  // Returns true if any argument to the traversal has bounded dynamic
  // dimensions marked.
  [[nodiscard]] bool IsBoundedDynamic() const;

  // The core components of a Traversal, returned by IntoParts().
  struct Parts {
    std::vector<DeviceBufferRef> arguments;
    std::vector<SharedDeviceBufferList> execution_order;
    std::vector<DeviceBufferRef> outputs;
  };

  // Breaks the Traversal into its core components, leaving an empty (and
  // invalid) Traversal behind. This is used once the invariants upheld by the
  // Traversal are no longer required.
  [[nodiscard]] Parts IntoParts() {
    return {std::move(arguments_), std::move(execution_order_),
            std::move(outputs_)};
  }

  // Returns the Python context for the Traversal.
  const PythonContext* absl_nullable GetPythonContext() const;

  // Sorts the Traversal's execution_order by op creation index.
  inline void SortByCreationOrder() {
    std::sort(
        execution_order_.begin(), execution_order_.end(),
        [](const SharedDeviceBufferList& a, const SharedDeviceBufferList& b) {
          return a->creation_index() < b->creation_index();
        });
  }

 private:
  // Private constructor, only called by Traversal::Create().
  // Definitions:
  //   - arguments_:      the materialized and placeholder nodes in the graph.
  //                      When using stopping_points (see Traversal::Create),
  //                      these may include deferred nodes as well, which are
  //                      pending execution by an earlier Traversal in a
  //                      sequence of computation graphs.
  //   - execution_order: deferred nodes in the graph. These may be leaf nodes
  //                      (for nullary ops), but are more commonly branch or
  //                      root nodes.
  //   - outputs_:        the final outputs of the computation. These can be any
  //                      nodes in the graph, but are most commonly root nodes.
  //
  // Properties:
  //   - All tensors (if any) in arguments_ are unique
  //   - All tensors (if any) in execution_order are unique.
  //   - All tensors (if any) in execution_order are deferred nodes (kDeferred
  //     state).
  //   - Each deferred node in execution_order only depends on tensors in
  //     arguments_ and/or lower-indexed execution_order nodes.
  //   - All tensors in outputs are unique, and there is at least one output.
  //   - Each tensor in outputs is in either arguments_ or execution_order.
  Traversal(std::vector<DeviceBufferRef> arguments,
            std::vector<SharedDeviceBufferList> execution_order,
            std::vector<DeviceBufferRef> outputs)
      : arguments_(std::move(arguments)),
        execution_order_(std::move(execution_order)),
        outputs_(std::move(outputs)) {}

  GraphKey BuildGraphKey() const;

  // Validates that the traversal is sound.
  absl::Status Validate() const;

  // Returns true if the traversal contains any SparseCore custom operations.
  [[nodiscard]] bool HasSparseCoreOp() const;

  // The arguments to the Traversal are all DeviceBufferRefs which are in either
  // the kMaterialized or kPlaceholder state.
  std::vector<DeviceBufferRef> arguments_;
  // A execution_order traversal of the deferred ops in the graph.
  // Each op only depends on arguments and lower-indexed ops.
  // Nullary ops, which are typically either factory ops like torch.ones or
  // compiled mode constants, appear here.
  std::vector<SharedDeviceBufferList> execution_order_;
  // The tensor outputs of the Traversal. These may be in any state, but the
  // list is non-empty and all outputs are unique.
  std::vector<DeviceBufferRef> outputs_;
  // The graph key of the Traversal; lazily computed.
  mutable std::optional<GraphKey> graph_key_;

  // The acceptable dynamic bounds for each dimension in the traversal's
  // arguments.
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
