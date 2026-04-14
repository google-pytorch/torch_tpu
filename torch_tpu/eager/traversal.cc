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

#include "torch_tpu/eager/traversal.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <stack>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/compilation_cache.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/traceme.h"

namespace torch_tpu {

absl::StatusOr<Traversal> Traversal::Create(
    std::vector<DeviceBufferRef> outputs,
    const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        stopping_points) {
  ABSL_VLOG(1) << "[Traversal::Create] creating Traversal";
  tsl::profiler::TraceMe t([] { return "Traversal::Create"; });

  // This will contain the traversal inputs.
  std::vector<DeviceBufferRef> inputs;
  // This will contain a topological sort of the internal traversal nodes.
  std::vector<SharedDeviceBufferList> execution_order;
  // A stack of deferred nodes to process in depth first, by traversing reverse
  // edges.
  std::stack<SharedDeviceBufferList> stack;
  // Deferred nodes as they are being visited.  A node marked as kGray is on the
  // stack and is waiting for its inputs to be processed. A node marked as
  // kBlack has been completely processed and all of its inputs have been
  // processed too.
  enum class Color { kGray, kBlack };
  absl::flat_hash_map<const DeviceBufferList*, Color> visited_deferred;
  // Tracks inputs already visited.
  absl::flat_hash_set<DeviceBufferRef> visited_inputs;

  // Populate the stack with the nodes feeding the graph outputs.
  for (const auto& output : outputs) {
    // If the output node has a deferred op, then we must consider it for the
    // DFS traversal; otherwise, it is an input and we store it as such.
    auto& node = output.device_buffer_list();
    if (node->deferred_op()) {
      if (visited_deferred.insert(std::make_pair(node.get(), Color::kGray))
              .second) {
        stack.push(node);
      }
    } else {
      if (visited_inputs.insert(output).second) {
        inputs.push_back(output);
      }
    }
  }

  while (!stack.empty()) {
    auto node = stack.top();

    // If a node has already been fully processed, then skip it. This happens
    // for node C in the following graph if the nodes initially pushed on the
    // stack are [A, C, B]: at the time we process B, we push node C on the
    // stack again [A, C, B, C], to later rediscover the 1st instance of C on
    // the stack that by then will be marked as kBlack.
    //
    // A -> B -> C
    // |         ^
    //  \_______/
    //
    if (auto it = visited_deferred.find(node.get());
        it != visited_deferred.end() && it->second == Color::kBlack) {
      stack.pop();
      continue;
    }

    const DeferredOp* deferred_op = node->deferred_op();
    TT_RET_CHECK(deferred_op, error::kFailedPrecondition)
        << "Expected a deferred op";

    size_t prev_stack_size = stack.size();
    for (const auto& input : deferred_op->inputs()) {
      auto& input_node = input.device_buffer_list();
      // If the prev node has a deferred op, then we must consider it for the
      // DFS traversal; otherwise, it is an input and we store it as such.
      if (input_node->deferred_op()) {
        // If the input node hasn't been visited yet, then push it on the stack
        // and mark it as gray.
        if (auto it = visited_deferred.find(input_node.get());
            it == visited_deferred.end()) {
          // We found a new deferred node, however if it is a stopping point
          // then need to treat it as a traversal input.
          if (!stopping_points.contains(input_node.get())) {
            // Not a stopping point, add the newly discovered node to the stack.
            stack.push(input_node);
            visited_deferred.insert(
                std::make_pair(input_node.get(), Color::kGray));
          } else {
            // A stopping point. Treat it as a graph input.
            if (visited_inputs.insert(input).second) {
              inputs.push_back(input);
            }
          }

        } else if (it->second == Color::kGray) {
          // A rediscovered gray node is pushed on the stack, again.
          stack.push(input_node);
        }

      } else {
        // A node without a deferred op is always treated as a graph input.
        if (visited_inputs.insert(input).second) {
          inputs.push_back(input);
        }
      }
    }

    // If we are done with this node, i.e., it didn't lead to adding more (gray)
    // nodes to the stack, then we can remove it from the stack and add it to
    // the execution order.
    bool node_is_fully_processed = (stack.size() == prev_stack_size);
    if (node_is_fully_processed) {
      // Double check that node is still a reference to the top of the stack.
      ABSL_CHECK(!stack.empty());        // CRASH_OK
      ABSL_CHECK_EQ(stack.top(), node);  // CRASH_OK
      visited_deferred.find(node.get())->second = Color::kBlack;
      execution_order.push_back(node);
      stack.pop();
    }
  }
  auto traversal = Traversal(std::move(inputs), std::move(execution_order),
                             std::move(outputs));
#ifndef NDEBUG
  if (auto status = traversal.Validate(); !status.ok()) {
    LogLines(traversal.DebugString());
    return status;
  }
#endif
  return traversal;
}

absl::StatusOr<Traversal> Traversal::Create(
    absl::Span<const SharedDeviceBufferList> output_nodes,
    const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        stopping_points) {
  std::vector<DeviceBufferRef> outputs_ref;
  for (const auto& output : output_nodes) {
    for (auto i = 0; i < output->size(); ++i) {
      TT_ASSIGN_OR_RETURN(auto output_ref, DeviceBufferRef::Create(output, i));
      outputs_ref.push_back(std::move(output_ref));
    }
  }
  return Create(std::move(outputs_ref), stopping_points);
}

absl::StatusOr<Traversal> Traversal::CreateFromLinearRegion(
    absl::Span<const SharedDeviceBufferList> region) {
  ABSL_VLOG(1) << "[Traversal::CreateFromLinearRegion] Creating Traversal";
  tsl::profiler::TraceMe t([] { return "Traversal::CreateFromLinearRegion"; });

  // This will contain the traversal inputs.
  std::vector<DeviceBufferRef> inputs;
  // This will contain the traversal outputs.
  std::vector<DeviceBufferRef> outputs;
  // This will contain a sort of the internal traversal nodes.
  std::vector<SharedDeviceBufferList> execution_order;

  // View ops are not dispatched like regular ops, they appear as deferred ops
  // in the input of the other dispatched ops. In order to extract all ops that
  // must be scheduled we build a traversal from all the outputs in the region,
  // but we use only its execution order and its inputs, since its outputs are
  // not what we want.
  {
    TT_ASSIGN_OR_RETURN(Traversal tmp_traversal, Create(region));
    execution_order = std::move(tmp_traversal.execution_order_);
    inputs = std::move(tmp_traversal.inputs_);
  }

  // Set of all buffers consumed as inputs by ops within the region.
  absl::flat_hash_set<DeviceBufferRef> region_consumptions;
  for (const auto& node : execution_order) {
    const auto* deferred_op = node->deferred_op();
    ABSL_CHECK(deferred_op);  // CRASH_OK
    for (const auto& input : deferred_op->inputs()) {
      region_consumptions.insert(input);
    }
  }

  // A traversal output is an op output that is NOT consumed by any other op
  // within the region.
  //
  // NOTE: In principle we'd like to only consider as outputs the
  // DeviceBufferRefs that are not used as inputs from ops in the execution
  // order. However, when dealing with multi-output ops that would break our
  // downstream logic. In fact, when we mark a DeviceBufferList as materialized,
  // all items in the list must be materialized, i.e., we don't support
  // materialization of only some of the list items. Consequently, for
  // multi-output ops we consider all its outputs, even if they some of them
  // are used from within the region, as long as at least one is not.
  absl::flat_hash_set<const DeviceBufferList*> visited_nodes;
  for (const auto& node : execution_order) {
    bool must_insert = false;
    for (size_t i = 0; i < node->size(); ++i) {
      TT_ASSIGN_OR_RETURN(auto output, DeviceBufferRef::Create(node, i));
      if (!region_consumptions.contains(output) &&
          visited_nodes.insert(node.get()).second) {
        must_insert = true;
        break;
      }
    }
    if (must_insert) {
      for (size_t i = 0; i < node->size(); ++i) {
        TT_ASSIGN_OR_RETURN(auto output, DeviceBufferRef::Create(node, i));
        outputs.push_back(std::move(output));
      }
    }
  }

  auto traversal = Traversal(std::move(inputs), std::move(execution_order),
                             std::move(outputs));
#ifndef NDEBUG
  if (auto status = traversal.Validate(); !status.ok()) {
    LogLines(traversal.DebugString());
    return status;
  }
#endif
  return traversal;
}

CompilationCacheKey Traversal::BuildCacheKey() const {
  tsl::profiler::TraceMe t("Traversal::BuildCacheKey");
  // We will be building a GraphSignature object as a simplified
  // representation of the Traversal graph for the purposes of hashing.
  GraphSignature graph;
  // TODO(aarfaian): the concept between true and de-facto inputs (stemming from
  // the fact that eager mode always treats zero-sized constants as inputs,
  // where compile mode may not necessarily behave the same) needs to be
  // revisited with regard to variable naming here and throughout the rest of
  // the Traversal implementation.

  absl::flat_hash_map<DeviceBufferRef, int> tensor_index_map;

  // Add all inputs to tensor-indexed properties.
  for (const DeviceBufferRef& input : inputs()) {
    tensor_index_map[input] =
        graph.AddInput(input.dimensions(), input.element_type());
  }
  for (const DeviceBufferRef& zsc : non_input_zero_sized_consts()) {
    tensor_index_map[zsc] =
        graph.AddInput(zsc.dimensions(), zsc.element_type());
  }

  for (const SharedDeviceBufferList& node : execution_order()) {
    const DeferredOp* absl_nullable maybe_deferred_op = node->deferred_op();
    ABSL_VLOG(1) << "[Traversal::BuildCacheKey] node: " << node.get()
                 << " maybe_deferred_op: " << maybe_deferred_op;
    ABSL_CHECK(maybe_deferred_op != nullptr);  // CRASH_OK
    const DeferredOp& deferred_op = *maybe_deferred_op;

    graph.AddOp(deferred_op.op_name(), deferred_op.op_param_cache_keys(),
                deferred_op.donated_indices(),
                [&](GraphSignature::OpSignatureBuilder& op) {
                  for (const DeviceBufferRef& op_input : deferred_op.inputs()) {
                    op.AddInput(tensor_index_map.at(op_input));
                  }
                  for (int64_t i = 0; i < node->size(); ++i) {
                    auto output = DeviceBufferRef::Create(node, i);
                    ABSL_CHECK(output.ok())  // CRASH_OK
                        << "Failed to create DeviceBufferRef for output: " << i;
                    tensor_index_map[*output] = op.AddOutput(
                        output->dimensions(), output->element_type());
                  }
                });
  }

  for (const DeviceBufferRef& output : outputs()) {
    graph.AddGraphOutput(tensor_index_map.at(output));
  }

  return graph.cache_key();
}

absl::Status Traversal::ValidateAndReorderInputs(
    std::vector<DeviceBufferRef> inputs) {
  ABSL_VLOG(1) << "[Traversal::ValidateAndReorderInputs] validating "
                  "consistency of provided inputs";
  // Check to make sure that inputs (the argument) is just a reordering of
  // inputs_ (the previous list of inputs).
  // Build a hashmap of the previous inputs, and mark all of them as unused.
  absl::flat_hash_map<DeviceBufferRef, bool> prev_inputs;
  for (const DeviceBufferRef& prev_input : inputs_) {
    // By construction, Traversal::inputs_ should be unique.
    ABSL_CHECK(  // CRASH_OK
        prev_inputs.insert_or_assign(prev_input, false).second)
        << "Traversal::inputs_ has a duplicate input: "
        << prev_input.DebugString();
  }

  // Checks that all provided inputs are non-deferred, are non-duplicates, and
  // marks them as used.
  for (const DeviceBufferRef& input : inputs) {
    TT_RET_CHECK(input.state() != DeviceBufferRefState::kDeferred,
                 error::kInvalidArgument)
        << "found a deferred input, which is not allowed: "
        << input.DebugString();
    auto it = prev_inputs.find(input);
    if (it == prev_inputs.end()) {
      // Allow unused inputs.
      continue;
    }
    TT_RET_CHECK(!it->second, error::kInvalidArgument)
        << "identified a duplicate input: " << input.DebugString();
    it->second = true;
  }

  // Check that all previous inputs are included in the new inputs.
  for (const auto& [input, used] : prev_inputs) {
    // If the graph has any zero-sized constants the traversal will include
    // these as inputs by default. However, in compiled mode these inputs may
    // not necessarily be included in the set of inputs expected by the FX
    // graph. If we run across any zero-sized constants here that aren't already
    // marked as used then they are not "true" inputs and we can handle them
    // separately. This allows us to maintain the invariant that FX graph and
    // our own traversed inputs are always the same.
    if (!used && input.state() == DeviceBufferRefState::kZeroSize) {
      non_input_zero_sized_consts_.push_back(input);
      continue;
    }
    TT_RET_CHECK(used, error::kInvalidArgument)
        << "identified an input that was not provided: " << input.DebugString();
  }
  inputs_ = std::move(inputs);
  ABSL_VLOG(1) << "[Traversal::ValidateAndReorderInputs] New inputs are valid. "
                  "Reordering inputs to match.";
  return absl::OkStatus();
}

absl::StatusOr<mlir::MlirOp> Traversal::GetMlirOpForProcessedBuffer(
    const absl::flat_hash_map<DeviceBufferRef, mlir::MlirOp>& ref_to_op_map,
    const DeviceBufferRef& buffer_ref) const {
  auto it = ref_to_op_map.find(buffer_ref);
  TT_RET_CHECK(it != ref_to_op_map.end(), error::kInternal)
      << "DeviceBufferRef not found in ref_to_op_map: "
      << buffer_ref.DebugString();
  return it->second;
}

namespace {

mlir::MlirOp BufferToArgument(mlir::func::FunctionBuilder& fb,
                              const DeviceBufferRef& input) {
  if (input.state() == DeviceBufferRefState::kZeroSize) {
    auto type = makeTensorType(fb.getContext(), input.dimensions(),
                               input.element_type());
    // In compiled mode we can still have zero-sized tensors as explicit
    // inputs and we handle those here.
    return MakeConstant(fb, mlir::ArrayRef<int64_t>{}, type);
  }
  Dimensions dimensions = CopyIntVector(input.dimensions());
  // If input has bounded dynamic dimensions, we assume we will receive an
  // input padded to the upper bound, along with the dimension sizes.
  // We use a set_dimension_size op to
  // convert to a dynamic tensor and use that downstream.
  for (const auto& dynamic_dim : input.dynamic_dimensions()) {
    dimensions[dynamic_dim.dimension] = dynamic_dim.upper_bound;
  }
  auto type = makeTensorType(fb.getContext(), dimensions, input.element_type());
  auto dimension_size_type =
      makeTensorType(fb.getContext(), {}, mlir::ElementType::I32);
  auto result = mlir::func::Argument(fb, type);
  for (const auto& dynamic_dim : input.dynamic_dimensions()) {
    mlir::MlirOp dimension_size = mlir::func::Argument(fb, dimension_size_type);
    result = mlir::stablehlo::SetDimensionSize(result, dimension_size,
                                               dynamic_dim.dimension);
  }
  return result;
}

}  // namespace

const PythonContext* absl_nullable Traversal::GetPythonContext() const {
  if (!execution_order().empty() && execution_order().back()->deferred_op()) {
    // Use the python context of the last op in the execution order as the
    // module name. This will include the op that triggered materialization or
    // traversal split in the module name.
    return &execution_order().back()->deferred_op()->op_context();
  }
  return nullptr;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> Traversal::BuildMlirModule(
    mlir::MLIRContext& mlir_context) const {
  // Read the traversal's values.
  absl::Span<const DeviceBufferRef> inputs = this->inputs();
  absl::Span<const SharedDeviceBufferList> execution_order =
      this->execution_order();
  absl::Span<const DeviceBufferRef> outputs = this->outputs();

  // Initialize the module builder and main function builder.
  const PythonContext* absl_nullable python_context = GetPythonContext();
  std::string module_name =
      BuildModuleNameFromPyContext(mlir_context, python_context);
  mlir::ModuleBuilder mb(mlir_context, module_name);
  mlir::func::FunctionBuilder fb(mb, "main");

  // Add a function parameter for each argument DeviceBufferRef.
  RefToOpMap ref_to_op_map;
  ABSL_VLOG(2) << "[Traversal::BuildMlirModule] building MLIR ops for "
               << inputs.size() << " inputs";
  for (const DeviceBufferRef& input : inputs) {
    ref_to_op_map[input] = BufferToArgument(fb, input);
  }

  for (const DeviceBufferRef& zsc : non_input_zero_sized_consts()) {
    auto type =
        makeTensorType(mlir_context, zsc.dimensions(), zsc.element_type());
    ref_to_op_map[zsc] = MakeConstant(fb, mlir::ArrayRef<int64_t>{}, type);
  }

  // Identify which inputs are donated.
  // Deduplicate by the mlir::Value, not by the DeviceBufferRef, to avoid false
  // negatives for no-op DeferredOps that get created by CopyTpuToTpu.
  llvm::DenseSet<mlir::Value> donated_values;

  // Build an MlirOp for each deferred op in execution_order ordering, so that
  // inputs are built before their outputs.
  ABSL_VLOG(2) << "[Traversal::BuildMlirModule] building MLIR ops for "
               << execution_order.size() << " deferred ops";
  std::vector<mlir::MlirOp> deferred_inputs;
  for (const SharedDeviceBufferList& node : execution_order) {
    // Get the deferred op we need to build.
    const DeferredOp* absl_nullable maybe_deferred_op = node->deferred_op();
    TT_RET_CHECK(maybe_deferred_op, error::kInternal)
        << "DeviceBufferList in execution_order has no deferred op";
    const DeferredOp& deferred_op = *maybe_deferred_op;

    // Get the MlirOps for all inputs.
    deferred_inputs.clear();
    for (const DeviceBufferRef& input : deferred_op.inputs()) {
      TT_ASSIGN_OR_RETURN(mlir::MlirOp mlir_op,
                          GetMlirOpForProcessedBuffer(ref_to_op_map, input));
      deferred_inputs.push_back(mlir_op);
    }

    // Get the MlirOps for all donated inputs.
    for (int64_t donated_input_index : deferred_op.donated_indices()) {
      donated_values.insert(deferred_inputs[donated_input_index].getValue());
    }

    // Build the MlirOp.
    // While `context` is in scope, all MlirOp objects built by `builder` will
    // be associated with the op's Python context.
    // This includes both the result of op_builder() and
    // CastIfNeeded.
    ScopedPythonContextProvider provider(deferred_op.op_context().Copy(), &fb);
    TT_ASSIGN_OR_RETURN(
        DynamicMlirOpResults results,
        deferred_op.op_builder()(fb, absl::MakeSpan(deferred_inputs)));
    TT_RET_CHECK(results.size() == node->size(), error::kInternal)
        << "deferred op " << deferred_op.op_name() << " returned "
        << results.size() << " results, expected " << node->size();

    // Cast each output of the deferred op to the expected type.
    for (int64_t index = 0; index < node->size(); ++index) {
      TT_ASSIGN_OR_RETURN(
          mlir::MlirOp casted_op,
          CastIfNeeded(results[index], node->element_type(index)));

      TT_ASSIGN_OR_RETURN(DeviceBufferRef ref,
                          DeviceBufferRef::Create(node, index));
      ABSL_VLOG(3) << "[Traversal::BuildMlirModule] built output " << index
                   << " of deferred op " << deferred_op.op_name() << ": "
                   << mlir::debugString(casted_op.getValue());
      ref_to_op_map[std::move(ref)] = std::move(casted_op);
    }
  }

  // End the main function by returning the final results. These were already
  // casted/reshaped in the loop above.
  DynamicMlirOpResults results;
  results.reserve(outputs.size());
  for (const DeviceBufferRef& output : outputs) {
    TT_RET_CHECK(ref_to_op_map.contains(output), error::kInternal)
        << "output DeviceBufferRef was not found in ref_to_op map: "
        << output.DebugString();
    results.push_back(ref_to_op_map.at(output));
  }
  ABSL_VLOG(1) << "[Traversal::BuildMlirModule] Built a total of "
               << ref_to_op_map.size() << " MlirOps, returning "
               << results.size() << " of them as results.";

  mlir::func::Return(fb, results);
  auto module = mb.build();

  // Identify which inputs to the Traversal are donated.
  Indices donated_inputs;
  if (!donated_values.empty()) {
    for (int64_t i = 0; i < inputs.size(); ++i) {
      mlir::MlirOp input_op = ref_to_op_map[inputs[i]];
      if (donated_values.contains(input_op.getValue())) {
        donated_inputs.push_back(i);
      }
    }
  }
  if (!donated_values.empty()) {
    AnnotateBufferDonations(module.get(), donated_inputs);
  }
  return module;
}

absl::StatusOr<CompiledKernel> Traversal::Compile(
    CompilationMode compilation_mode) const {
  // Prepare a computation builder closure to be called on a cache miss.  Okay
  // to capture this here since CompilationCache::GetOrCompile() will call this
  // builder before the function returns and in the same thread it is invoked.
  MlirComputationBuilder final_op_builder =
      [this](mlir::MLIRContext& mlir_context) {
        return BuildMlirModule(mlir_context);
      };
  ABSL_VLOG(1) << "[Compile] cache_key: " << cache_key();
  std::vector<Shape> input_shapes;
  input_shapes.reserve(inputs_.size());
  for (const auto& input : inputs_) {
    Shape input_shape(CopyIntVector(input.dimensions()), input.element_type());
    for (const auto& dynamic_dim : input.dynamic_dimensions()) {
      input_shape.dynamic_dimensions().push_back(dynamic_dim);
    }
    input_shapes.push_back(std::move(input_shape));
  }

  TT_ASSIGN_OR_RETURN(UniqueCompileOptions compile_options,
                      MakeCompilerOptions(compilation_mode));

  return CompilationCache::GetInstance().GetOrCompile(
      cache_key(), input_shapes, std::move(final_op_builder),
      std::move(compile_options));
}

bool IsSimpleNodeTraversal(const Traversal& traversal) {
  absl::Span<const DeviceBufferRef> outputs = traversal.outputs();
  ABSL_CHECK(!outputs.empty());  // CRASH_OK=traversals are nonempty
  const SharedDeviceBufferList& node = outputs[0].device_buffer_list();
  const DeferredOp* absl_nullable deferred_op = node->deferred_op();
  if (deferred_op == nullptr) {
    return false;
  }
  if (node->size() != outputs.size()) {
    return false;
  }
  for (int i = 0; i < outputs.size(); ++i) {
    if (outputs[i].device_buffer_list() != node || outputs[i].index() != i) {
      return false;
    }
  }
  return true;
}

namespace {

// Helper function for Traversal::DebugString() to print one DeviceBufferRef's
// global index, dtype, and shape, like
// "#0: float32[1,2,3]"
void StreamBufferRefDebug(std::ostream& os, const DeviceBufferRef& ref,
                          const int64_t buffer_index) {
  os << "#" << buffer_index << ": " << ToString(ref.element_type())
     << ToString(ref.dimensions()) << "";
}

// Helper function for Traversal::DebugString() to print an input onto a row,
// as either:
//   #0: float32[0, 1, 2] <- zero-sized constant
//   #1: float32[1, 2, 3] <- argument 0 (materialized)
//   #2: float32[2, 3, 4] <- argument 1 (placeholder)
void StreamInputDebug(
    std::ostream& os, const DeviceBufferRef& input, size_t& arg_index,
    size_t& buffer_index,
    absl::flat_hash_map<DeviceBufferRef, size_t>& buffer_to_index) {
  StreamBufferRefDebug(os, input, buffer_index);
  switch (input.state()) {
    case DeviceBufferRefState::kZeroSize:
      os << " <- zero-sized constant";
      break;
    case DeviceBufferRefState::kMaterialized:
      os << " <- input " << arg_index++ << " (materialized)";
      break;
    case DeviceBufferRefState::kPlaceholder:
      os << " <- input " << arg_index++ << " (placeholder)";
      break;
    case DeviceBufferRefState::kDeferred:
      os << " <- input " << arg_index++;
      {
        auto* deferred_op = input.deferred_op();
        ABSL_CHECK(deferred_op);  // CRASH_OK
        os << " (deferred " << ToString(deferred_op->op_name()) << ")";
      }
      break;
  }
  buffer_to_index[input] = buffer_index++;
  os << "\n";
}

// Helper function for Traversal::DebugString() to print a deferred op onto a
// row, as {outputs} <- op_name <- {inputs}, possibly without {inputs} if the
// op is nullary.
void StreamDeferredOpDebug(
    std::ostream& os, const SharedDeviceBufferList& node, size_t& buffer_index,
    absl::flat_hash_map<DeviceBufferRef, size_t>& buffer_to_index) {
  for (int i = 0; i < node->size(); ++i) {
    if (i > 0) os << ", ";
    auto node_output = DeviceBufferRef::Create(node, i).value();
    StreamBufferRefDebug(os, node_output, buffer_index);
    buffer_to_index[node_output] = buffer_index++;
  }

  auto deferred_op = node->deferred_op();
  if (!deferred_op) {
    os << " (missing deferred op)";
    return;
  }

  os << " <- " << deferred_op->op_name();
  if (!deferred_op->inputs().empty()) {
    os << " <- ";
  }
  const auto num_inputs = deferred_op->inputs().size();
  for (auto i = 0; i < num_inputs; ++i) {
    if (i > 0) os << ", ";
    const DeviceBufferRef& input = deferred_op->inputs()[i];
    if (auto it = buffer_to_index.find(input); it != buffer_to_index.end()) {
      StreamBufferRefDebug(os, input, it->second);
    } else {
      os << "(unexpected input)";
    }
  }
  os << "\n";
}

}  // namespace

std::string Traversal::DebugString() const {
  std::ostringstream os;
  os << "Traversal:\n";
  size_t arg_index = 0;
  size_t buffer_index = 0;
  absl::flat_hash_map<DeviceBufferRef, size_t> buffer_to_index;
  os << "Inputs:\n";
  for (const auto& input : inputs_) {
    StreamInputDebug(os, input, arg_index, buffer_index, buffer_to_index);
  }
  os << "Execution Order:\n";
  for (const SharedDeviceBufferList& node : execution_order_) {
    StreamDeferredOpDebug(os, node, buffer_index, buffer_to_index);
  }
  os << "Outputs:\n";
  for (auto i = 0; i < outputs_.size(); ++i) {
    const DeviceBufferRef& output = outputs_[i];
    if (i > 0) os << ", ";
    if (auto it = buffer_to_index.find(output); it != buffer_to_index.end()) {
      StreamBufferRefDebug(os, output, it->second);
    } else {
      os << "(unexpected buffer)";
    }
  }
  return os.str();
}

namespace {

using Vertex = std::variant<const DeviceBufferRef, const DeferredOp*>;

struct Edge {
  int from_index = -1;
  int to_index = -1;
};

class GraphvizGraph {
 public:
  GraphvizGraph() = default;

  // This class is move-only.
  GraphvizGraph(GraphvizGraph&& other) = default;
  GraphvizGraph& operator=(GraphvizGraph&& other) = default;
  GraphvizGraph(const GraphvizGraph&) = delete;
  GraphvizGraph& operator=(const GraphvizGraph&) = delete;

  static absl::StatusOr<GraphvizGraph> Create(
      absl::Span<const DeviceBufferRef> inputs,
      absl::Span<const SharedDeviceBufferList> execution_order) {
    GraphvizGraph graph;
    TT_RETURN_IF_ERROR(graph.Init(inputs, execution_order));
    return graph;
  }

  const std::vector<Vertex>& vertices() const { return vertices_; }
  const std::vector<Edge>& edges() const { return edges_; }

 private:
  absl::Status Init(absl::Span<const DeviceBufferRef> inputs,
                    absl::Span<const SharedDeviceBufferList> execution_order) {
    AddInputVertices(inputs);
    TT_RETURN_IF_ERROR(AddExecutionOrderVerticesAndEdges(execution_order));
    return absl::OkStatus();
  }

  void AddInputVertices(absl::Span<const DeviceBufferRef> inputs) {
    for (const auto& input : inputs) {
      buffer_to_index_[input] = vertices_.size();
      vertices_.push_back(input);
    }
  }

  absl::Status AddExecutionOrderVerticesAndEdges(
      absl::Span<const SharedDeviceBufferList> execution_order) {
    for (const auto& node : execution_order) {
      std::optional<int> maybe_deferred_op_vertex_index = std::nullopt;
      if (const DeferredOp* deferred_op = node->deferred_op()) {
        maybe_deferred_op_vertex_index =
            AddDeferredOpVertexAndInputEdges(*deferred_op);
      }
      TT_RETURN_IF_ERROR(
          AddOutputVerticesAndEdges(node, maybe_deferred_op_vertex_index));
    }
    return absl::OkStatus();
  }

  std::optional<int> AddDeferredOpVertexAndInputEdges(
      const DeferredOp& deferred_op) {
    int deferred_op_vertex_index = vertices_.size();
    vertices_.push_back(&deferred_op);
    for (const DeviceBufferRef& input : deferred_op.inputs()) {
      if (auto it = buffer_to_index_.find(input);
          it != buffer_to_index_.end()) {
        edges_.push_back(
            {.from_index = it->second, .to_index = deferred_op_vertex_index});
      }
    }
    return deferred_op_vertex_index;
  }

  absl::Status AddOutputVerticesAndEdges(
      const SharedDeviceBufferList& node,
      std::optional<int> maybe_deferred_op_vertex_index) {
    for (int i = 0; i < node->size(); ++i) {
      TT_ASSIGN_OR_RETURN(DeviceBufferRef output,
                          DeviceBufferRef::Create(node, i));
      buffer_to_index_[output] = vertices_.size();
      vertices_.push_back(output);
      if (maybe_deferred_op_vertex_index.has_value()) {
        edges_.push_back({.from_index = *maybe_deferred_op_vertex_index,
                          .to_index = static_cast<int>(vertices_.size()) - 1});
      }
    }
    return absl::OkStatus();
  }

  absl::flat_hash_map<DeviceBufferRef, int> buffer_to_index_;
  std::vector<Vertex> vertices_;
  std::vector<Edge> edges_;
};

std::string GraphvizVertexParams(
    const Vertex& vertex,
    const absl::flat_hash_map<DeviceBufferRef, std::string>&
        buffer_ref_to_python_var) {
  std::ostringstream os;
  if (std::holds_alternative<const DeviceBufferRef>(vertex)) {
    const DeviceBufferRef& ref = std::get<const DeviceBufferRef>(vertex);
    os << "[shape=\"box\", label=\"";
    if (auto it = buffer_ref_to_python_var.find(ref);
        it != buffer_ref_to_python_var.end()) {
      os << it->second << ": ";
    } else {
      os << " ";
    }
    os << ToString(ref.element_type()) << ToString(ref.dimensions());
    switch (ref.state()) {
      case DeviceBufferRefState::kZeroSize:
        os << " (zero-sized constant)";
        break;
      case DeviceBufferRefState::kMaterialized:
        os << " (materialized)";
        break;
      case DeviceBufferRefState::kPlaceholder:
        os << " (placeholder)";
        break;
      default:
        break;
    }
    os << "\"]";
  } else {
    const DeferredOp* deferred_op = std::get<const DeferredOp*>(vertex);
    os << "[label=\"" << deferred_op->op_name() << "\"]";
  }
  return os.str();
}

}  // namespace

absl::Status Traversal::Validate() const {
  size_t buffer_index = 0;
  absl::flat_hash_map<DeviceBufferRef, size_t> buffer_to_index;

  // Get buffers from the traversal's inputs.
  for (const auto& input : inputs_) {
    buffer_to_index[input] = buffer_index++;
  }

  // Validate internal buffers.
  for (auto i = 0; i < execution_order_.size(); ++i) {
    const SharedDeviceBufferList& node = execution_order_[i];
    // Validate that all traversal's internal buffers are from deferred ops
    // using either buffer inputs or outputs from previously encountered
    // deferred ops.
    auto* deferred_op = node->deferred_op();
    TT_RET_CHECK(deferred_op, error::kFailedPrecondition)
        << "Missing deferred op at line " << i;
    const auto num_inputs = deferred_op->inputs().size();
    for (auto input_idx = 0; input_idx < num_inputs; ++input_idx) {
      const DeviceBufferRef& input = deferred_op->inputs()[input_idx];
      TT_RET_CHECK(buffer_to_index.find(input) != buffer_to_index.end(),
                   error::kFailedPrecondition)
          << "Unexpected buffer at execution order index " << i
          << ", input index " << input_idx;
    }
    // Now we can add the deferred ops' outputs to the set of known buffers.
    for (auto output_idx = 0; output_idx < node->size(); ++output_idx) {
      TT_ASSIGN_OR_RETURN(auto node_output,
                          DeviceBufferRef::Create(node, output_idx));
      buffer_to_index[std::move(node_output)] = buffer_index++;
    }
  }

  // Validate traversal outputs.
  for (auto i = 0; i < outputs_.size(); ++i) {
    const DeviceBufferRef& output = outputs_[i];
    TT_RET_CHECK(buffer_to_index.find(output) != buffer_to_index.end(),
                 error::kFailedPrecondition)
        << "Unexpected buffer at line " << i;
  }

  return absl::OkStatus();
}

bool Traversal::IsBoundedDynamic() const {
  for (const DeviceBufferRef& input : inputs_) {
    if (!input.dynamic_dimensions().empty()) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<std::string> GetGraphviz(
    const Traversal& traversal,
    const absl::flat_hash_map<DeviceBufferRef, std::string>&
        buffer_ref_to_python_var) {
  TT_ASSIGN_OR_RETURN(
      GraphvizGraph graph,
      GraphvizGraph::Create(traversal.inputs(), traversal.execution_order()));

  std::string result =
      "Graphviz string: (try pasting in http://graphviz/ to see the graph)\n"
      "digraph {\n"
      "  // Vertices:\n";

  // Stream vertices in dot format.
  for (int i = 0; i < graph.vertices().size(); ++i) {
    const auto& vertex = graph.vertices()[i];
    absl::StrAppend(&result, "  ", i, " ",
                    GraphvizVertexParams(vertex, buffer_ref_to_python_var),
                    ";\n");
  }

  // Stream edges in dot format.
  absl::StrAppend(&result, "\n  // Edges:\n  ",
                  absl::StrJoin(graph.edges(), "\n  ",
                                [](std::string* out, const Edge& edge) {
                                  absl::StrAppend(out, edge.from_index, " -> ",
                                                  edge.to_index);
                                }),
                  "\n}\n");
  return result;
}

}  // namespace torch_tpu
