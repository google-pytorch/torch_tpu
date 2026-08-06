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
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/compilation_cache.h"
#include "torch_tpu/common/compilation_spec.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/client/executable_build_options.h"
#include "xla/layout_util.h"
#include "xla/service/computation_placer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

absl::StatusOr<absl_nonnull std::unique_ptr<Traversal>> Traversal::Create(
    std::vector<DeviceBufferRef> outputs,
    const absl::flat_hash_set<const DeviceBufferList* absl_nonnull>&
        stopping_points) {
  ABSL_VLOG(1) << "[Traversal::Create] creating Traversal";
  tsl::profiler::TraceMe t([] { return "Traversal::Create"; });

  // This will contain the traversal arguments.
  std::vector<DeviceBufferRef> arguments;
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
  // Tracks arguments already visited.
  absl::flat_hash_set<DeviceBufferRef> visited_arguments;

  // Populate the stack with the nodes feeding the graph outputs.
  for (const auto& output : outputs) {
    // If the output node has a deferred op, then we must consider it for the
    // DFS traversal; otherwise, it is an input and we store it as such.
    auto& node = output.device_buffer_list();
    if (node->is_deferred()) {
      if (visited_deferred.insert(std::make_pair(node.get(), Color::kGray))
              .second) {
        stack.push(node);
      }
    } else {
      if (visited_arguments.insert(output).second) {
        arguments.push_back(output);
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

    const auto deferred_op = node->deferred_op();
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
            // A stopping point. Treat it as a graph argument.
            if (visited_arguments.insert(input).second) {
              arguments.push_back(input);
            }
          }

        } else if (it->second == Color::kGray) {
          // A rediscovered gray node is pushed on the stack, again.
          stack.push(input_node);
        }

      } else {
        // A node without a deferred op is always treated as a graph argument.
        if (visited_arguments.insert(input).second) {
          arguments.push_back(input);
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
  auto traversal = std::unique_ptr<Traversal>(new Traversal(
      std::move(arguments), std::move(execution_order), std::move(outputs)));
#ifndef NDEBUG
  if (auto status = traversal->Validate(); !status.ok()) {
    LogLines(traversal->DebugString());
    return status;
  }
#endif
  return traversal;
}

absl::StatusOr<absl_nonnull std::unique_ptr<Traversal>> Traversal::Create(
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

absl::StatusOr<absl_nonnull std::unique_ptr<Traversal>>
Traversal::CreateFromExecutionOrder(
    absl::Span<const SharedDeviceBufferList> execution_order,
    absl::Span<const SharedDeviceBufferList> nodes_to_materialize) {
  ABSL_VLOG(1) << "[Traversal::CreateFromExecutionOrder] Creating Traversal";
  tsl::profiler::TraceMe t(
      [] { return "Traversal::CreateFromExecutionOrder"; });

  // Compute the traversal inputs.
  std::vector<DeviceBufferRef> inputs;
  {
    absl::flat_hash_set<const DeviceBufferList*> defs;
    absl::flat_hash_set<DeviceBufferRef> uses;
    for (const auto& node : execution_order) {
      const auto deferred_op = node->deferred_op();
      if (deferred_op) {
        for (const auto& input : deferred_op->inputs()) {
          if (defs.find(input.device_buffer_list().get()) == defs.end()) {
            if (uses.insert(input).second) {
              inputs.push_back(input);
            }
          }
        }
      }
      defs.insert(node.get());
    }
  }

  // Compute the traversal outputs.
  std::vector<DeviceBufferRef> outputs;
  for (const auto& node : nodes_to_materialize) {
    for (auto i = 0; i < node->size(); ++i) {
      TT_ASSIGN_OR_RETURN(auto output, DeviceBufferRef::Create(node, i));
      outputs.push_back(std::move(output));
    }
  }

  auto traversal = std::unique_ptr<Traversal>(
      new Traversal(std::move(inputs),
                    std::vector<SharedDeviceBufferList>(execution_order.begin(),
                                                        execution_order.end()),
                    std::move(outputs)));
#ifndef NDEBUG
  if (auto status = traversal->Validate(); !status.ok()) {
    LogLines(traversal->DebugString());
    return status;
  }
#endif
  return traversal;
}

GraphKey Traversal::BuildGraphKey() const {
  tsl::profiler::TraceMe t("Traversal::BuildGraphKey");
  // We will be building a GraphSignature object as a simplified
  // representation of the Traversal graph for the purposes of hashing.
  GraphSignature graph;

  absl::flat_hash_map<DeviceBufferRef, int> tensor_index_map;

  // Add all arguments to tensor-indexed properties.
  for (const DeviceBufferRef& argument : arguments()) {
    tensor_index_map[argument] =
        graph.AddInput(argument.dimensions(), argument.element_type());
  }
  for (const SharedDeviceBufferList& node : execution_order()) {
    const auto maybe_deferred_op = node->deferred_op();
    ABSL_VLOG(1) << "[Traversal::BuildGraphKey] node: " << node.get()
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

  return graph.GetKey();
}

absl::Status Traversal::ValidateAndReorderArguments(
    std::vector<DeviceBufferRef> arguments) {
  ABSL_VLOG(1) << "[Traversal::ValidateAndReorderArguments] validating "
                  "consistency of provided arguments";
  // Check to make sure that the new arguments are just a reordering of
  // arguments_ (the previous list of arguments).
  // Build a hashmap of the previous arguments, and mark all of them as unused.
  absl::flat_hash_map<DeviceBufferRef, bool> prev_arguments;
  for (const DeviceBufferRef& prev_argument : arguments_) {
    // By construction, Traversal::arguments_ should be unique.
    ABSL_CHECK(  // CRASH_OK
        prev_arguments.insert_or_assign(prev_argument, false).second)
        << "Traversal::arguments_ has a duplicate argument: "
        << prev_argument.DebugString();
  }

  // Checks that all provided arguments are non-deferred, are non-duplicates,
  // and marks them as used.
  for (const DeviceBufferRef& input : arguments) {
    TT_RET_CHECK(!input.is_deferred(), error::kInvalidArgument)
        << "found a deferred argument, which is not allowed: "
        << input.DebugString();
    if (auto it = prev_arguments.find(input); it != prev_arguments.end()) {
      // The first time we see a previous argument in the new arguments, mark
      // it as used.
      // If we see it again, then the new arguments has a duplicate and is
      // invalid.
      TT_RET_CHECK(!it->second, error::kInvalidArgument)
          << "identified a duplicate input: " << input.DebugString();
      it->second = true;
    } else {
      // Found an new argument not part of the previous arguments.
      // Include it and mark it as used, so that we can detect duplicates.
      ABSL_VLOG(2) << "[Traversal::ValidateAndReorderArguments] found a new "
                      "argument not part of the previous arguments: "
                   << input.DebugString();
      prev_arguments[input] = true;
    }
  }

  // Check that there are no previous arguments missing in the new arguments.
  for (const auto& [arg, used] : prev_arguments) {
    TT_RET_CHECK(used, error::kInvalidArgument)
        << "identified an argument that was not provided: "
        << arg.DebugString();
  }
  arguments_ = std::move(arguments);
  ABSL_VLOG(1)
      << "[Traversal::ValidateAndReorderArguments] New inputs are valid. "
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

mlir::MlirOp CreateNativeDynamicArgumentOp(mlir::func::FunctionBuilder& fb,
                                           const DeviceBufferRef& argument) {
  Dimensions mlir_dims;
  mlir_dims.reserve(argument.dimensions().size());

  Dimensions bounds;
  bounds.reserve(argument.dimensions().size());

  for (int64_t dim_size : argument.dimensions()) {
    mlir_dims.push_back(dim_size);
    bounds.push_back(mlir::ShapedType::kDynamic);
  }

  for (const auto& dynamic_dim : argument.dynamic_dimensions()) {
    mlir_dims[dynamic_dim.dimension] = mlir::ShapedType::kDynamic;
    bounds[dynamic_dim.dimension] = dynamic_dim.upper_bound;
  }

  auto bounds_attr =
      mlir::stablehlo::TypeExtensionsAttr::get(&fb.getContext(), bounds);

  auto element_type =
      mlir::getElementType(fb.getContext(), argument.element_type());

  auto type = mlir::RankedTensorType::get(mlir_dims, element_type, bounds_attr);

  return mlir::func::Argument(fb, type);
}

mlir::MlirOp CreatePaddedDynamicArgumentOp(mlir::func::FunctionBuilder& fb,
                                           const DeviceBufferRef& argument) {
  Dimensions dimensions = CopyIntVector(argument.dimensions());
  // If argument has bounded dynamic dimensions, we assume we will receive an
  // input padded to the upper bound, along with the dimension sizes.
  // We use a set_dimension_size op to
  // convert to a dynamic tensor and use that downstream.
  for (const auto& dynamic_dim : argument.dynamic_dimensions()) {
    dimensions[dynamic_dim.dimension] = dynamic_dim.upper_bound;
  }
  auto type =
      makeTensorType(fb.getContext(), dimensions, argument.element_type());
  auto dimension_size_type =
      makeTensorType(fb.getContext(), {}, mlir::ElementType::I32);
  auto result = mlir::func::Argument(fb, type);
  for (const auto& dynamic_dim : argument.dynamic_dimensions()) {
    mlir::MlirOp dimension_size = mlir::func::Argument(fb, dimension_size_type);
    result = mlir::stablehlo::SetDimensionSize(result, dimension_size,
                                               dynamic_dim.dimension);
  }
  return result;
}

mlir::MlirOp CreateDynamicArgumentOp(mlir::func::FunctionBuilder& fb,
                                     const DeviceBufferRef& argument,
                                     bool use_stablehlo_bounds) {
  if (use_stablehlo_bounds) {
    return CreateNativeDynamicArgumentOp(fb, argument);
  }
  return CreatePaddedDynamicArgumentOp(fb, argument);
}

mlir::MlirOp CreateArgumentOp(mlir::func::FunctionBuilder& fb,
                              const DeviceBufferRef& argument,
                              bool use_stablehlo_bounds) {
  if (argument.dynamic_dimensions().empty()) {
    Dimensions dimensions = CopyIntVector(argument.dimensions());
    auto type =
        makeTensorType(fb.getContext(), dimensions, argument.element_type());
    return mlir::func::Argument(fb, type);
  }
  return CreateDynamicArgumentOp(fb, argument, use_stablehlo_bounds);
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
    mlir::MLIRContext& mlir_context, bool use_stablehlo_bounds) const {
  // Read the traversal's values.
  absl::Span<const DeviceBufferRef> arguments = this->arguments();
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
               << arguments.size() << " arguments";
  for (const DeviceBufferRef& argument : arguments) {
    ref_to_op_map[argument] =
        CreateArgumentOp(fb, argument, use_stablehlo_bounds);
  }

  // Identify which arguments are donated.
  llvm::DenseSet<mlir::Value> donated_values;

  // Build an MlirOp for each deferred op in execution_order ordering, so that
  // inputs are built before their outputs.
  ABSL_VLOG(2) << "[Traversal::BuildMlirModule] building MLIR ops for "
               << execution_order.size() << " deferred ops";
  std::vector<mlir::MlirOp> deferred_inputs;
  for (const SharedDeviceBufferList& node : execution_order) {
    // Get the deferred op we need to build.
    const auto maybe_deferred_op = node->deferred_op();
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

  // Identify which arguments to the Traversal are donated.
  Indices donated_arguments;
  if (!donated_values.empty()) {
    for (int64_t i = 0; i < arguments.size(); ++i) {
      mlir::MlirOp input_op = ref_to_op_map[arguments[i]];
      if (donated_values.contains(input_op.getValue())) {
        donated_arguments.push_back(i);
      }
    }
  }
  if (!donated_values.empty()) {
    AnnotateBufferDonations(module.get(), donated_arguments);
  }
  return module;
}

namespace {
std::vector<Shape> GetShapes(absl::Span<const DeviceBufferRef> buffers) {
  std::vector<Shape> shapes;
  shapes.reserve(buffers.size());
  for (const auto& buffer : buffers) {
    Shape shape(CopyIntVector(buffer.dimensions()), buffer.element_type());
    for (const auto& dynamic_dim : buffer.dynamic_dimensions()) {
      shape.dynamic_dimensions().push_back(dynamic_dim);
    }
    shapes.push_back(std::move(shape));
  }
  return shapes;
}

std::vector<xla::Shape> GetXlaShapes(absl::Span<const DeviceBufferRef> buffers,
                                     bool use_stablehlo_bounds) {
  std::vector<xla::Shape> shapes;
  shapes.reserve(buffers.size());
  for (const auto& buffer : buffers) {
    xla::PrimitiveType primitive_type =
        ConvertTo<xla::PrimitiveType>(buffer.element_type());
    xla::Shape shape =
        xla::ShapeUtil::MakeShape(primitive_type, buffer.dimensions());
    if (use_stablehlo_bounds) {
      for (const auto& dynamic_dim : buffer.dynamic_dimensions()) {
        shape.set_dynamic_dimension(dynamic_dim.dimension, true);
      }
    }
    shapes.push_back(std::move(shape));
  }
  return shapes;
}

std::string MlirModuleToString(mlir::ModuleOp module) {
  mlir::OpPrintingFlags flags;
  flags.elideLargeElementsAttrs(100);
  std::string s;
  llvm::raw_string_ostream os(s);
  module.print(os, flags);
  return os.str();
}
}  // namespace

bool Traversal::HasSparseCoreOp() const {
  for (const SharedDeviceBufferList& node : execution_order_) {
    if (node->is_deferred()) {
      const std::shared_ptr<DeferredOp> op = node->deferred_op();
      if (op != nullptr && IsSparseCoreOp(op->op_name())) {
        return true;
      }
    }
  }
  return false;
}

absl::StatusOr<CompiledKernel> Traversal::Compile(
    CompilationSpec spec, std::string* absl_nullable out_mlir_text,
    bool use_stablehlo_bounds,
    absl::Span<const Indices> argument_layouts) const {
  if (HasSparseCoreOp()) {
    absl::StatusOr<int> status =
        PjrtBackend::GetInstance().GetGlobalDeviceCount();
    if (status.ok() && *status > 1) {
      const int num_devices = *status;
      xla::ExecutableBuildOptions& exec_options =
          spec.xla_compile_options->executable_build_options;
      exec_options.set_num_replicas(num_devices);
      exec_options.set_num_partitions(1);
      xla::DeviceAssignment da(num_devices, 1);
      for (int idx = 0; idx < num_devices; ++idx) {
        da(idx, 0) = idx;
      }
      exec_options.set_device_assignment(da);
      exec_options.set_use_spmd_partitioning(false);
      exec_options.set_use_shardy_partitioner(false);
      spec.compile_options_key =
          MakeCompileOptionsKey(GetEnvOnce<kXlaFlagsEnvVar>().value_or(""),
                                *spec.xla_compile_options);
    }
  }

  // Prepare a computation builder closure to be called on a cache miss.  Okay
  // to capture this here since CompilationCache::GetOrCompile() will call this
  // builder before the function returns and in the same thread it is invoked.
  MlirComputationBuilder final_op_builder =
      [this, use_stablehlo_bounds,
       out_mlir_text](mlir::MLIRContext& mlir_context)
      -> absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> {
    TT_ASSIGN_OR_RETURN(auto module,
                        BuildMlirModule(mlir_context, use_stablehlo_bounds));
    if (out_mlir_text != nullptr) {
      *out_mlir_text = MlirModuleToString(*module);
    }
    return module;
  };
  std::vector<Shape> argument_shapes = GetShapes(arguments_);
  std::vector<Shape> output_shapes = GetShapes(outputs_);

  if (!argument_layouts.empty()) {
    std::vector<xla::Shape> xla_argument_shapes =
        GetXlaShapes(arguments_, use_stablehlo_bounds);
    TT_RET_CHECK(xla_argument_shapes.size() == argument_layouts.size(),
                 error::kInvalidArgument)
        << "argument layouts size must match, got "
        << xla_argument_shapes.size() << " layouts and "
        << argument_layouts.size() << " arguments";
    for (size_t i = 0; i < xla_argument_shapes.size(); ++i) {
      const auto& layout_indices = argument_layouts[i];
      if (!layout_indices.empty()) {
        xla_argument_shapes[i].clear_layout();
        *xla_argument_shapes[i].mutable_layout() =
            xla::LayoutUtil::MakeLayout(layout_indices);
      }
    }
    spec.xla_compile_options->argument_layouts = std::move(xla_argument_shapes);
    spec.compile_options_key = MakeCompileOptionsKey(
        GetEnvOnce<kXlaFlagsEnvVar>().value_or(""), *spec.xla_compile_options);
  }

  CompilationCacheKey compilation_cache_key =
      GetCacheKey(spec.compile_options_key);
  ABSL_VLOG(1) << "[Compile] compilation cache key: " << compilation_cache_key;

  return CompilationCache::GetInstance().GetOrCompile(
      std::move(compilation_cache_key), argument_shapes, output_shapes,
      std::move(final_op_builder), std::move(spec.xla_compile_options),
      /*use_dynamic_adapters=*/!use_stablehlo_bounds);
}

bool IsSimpleNodeTraversal(const Traversal& traversal) {
  absl::Span<const DeviceBufferRef> outputs = traversal.outputs();
  ABSL_CHECK(!outputs.empty());  // CRASH_OK=traversals are nonempty
  const SharedDeviceBufferList& node = outputs[0].device_buffer_list();
  if (!node->is_deferred()) {
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
  if (auto deferred_op = input.deferred_op()) {
    os << " <- input " << arg_index++;
    ABSL_CHECK(deferred_op);  // CRASH_OK
    os << " (deferred " << ToString(deferred_op->op_name()) << ")";
  } else if (input.is_placeholder()) {
    os << " <- input " << arg_index++ << " (placeholder)";
  } else {
    os << " <- input " << arg_index++ << " (materialized)";
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
  for (const auto& argument : arguments_) {
    StreamInputDebug(os, argument, arg_index, buffer_index, buffer_to_index);
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

// Builds the canonical {DeviceBufferRef -> index} map used by
// ReadableString. Numbers arguments first, then every
// output of every node in execution_order. Skips outputs that fail to
// construct.
absl::flat_hash_map<DeviceBufferRef, size_t> BuildBufferIndexMap(
    absl::Span<const DeviceBufferRef> arguments,
    absl::Span<const SharedDeviceBufferList> execution_order) {
  absl::flat_hash_map<DeviceBufferRef, size_t> idx;
  size_t next = 0;
  for (const DeviceBufferRef& arg : arguments) {
    if (idx.try_emplace(arg, next).second) next++;
  }
  for (const SharedDeviceBufferList& node : execution_order) {
    for (int i = 0; i < node->size(); ++i) {
      auto out_or = DeviceBufferRef::Create(node, i);
      if (out_or.ok() && idx.try_emplace(*out_or, next).second) next++;
    }
  }
  return idx;
}

// Pick a user-relevant frame from a Python traceback: skip frames inside
// torch_tpu internals and site-packages, take the first remaining frame.
// Returns nullptr if no suitable frame exists.
const PythonStackFrame* absl_nullable PickUserFrame(const PythonTraceback& tb) {
  for (const PythonStackFrame& f : tb.frames) {
    const std::string& fn = f.file_name;
    if (absl::StrContains(fn, "/torch_tpu/")) continue;
    if (absl::StrContains(fn, "/site-packages/")) continue;
    if (absl::StrContains(fn, "/torch/")) continue;
    return &f;
  }
  // Fallback: innermost frame.
  if (!tb.frames.empty()) return &tb.frames.front();
  return nullptr;
}

// Compact "f32[1, 2, 3]" form, used by ReadableString().
std::string FormatTypeCompact(const DeviceBufferRef& ref) {
  return absl::StrCat(ToShortString(ref.element_type()), "[",
                      absl::StrJoin(ref.dimensions(), ", "), "]");
}

// FX-style "%N" if ref is in idx_of, "?" otherwise.
std::string FormatRefIndex(
    const absl::flat_hash_map<DeviceBufferRef, size_t>& idx_of,
    const DeviceBufferRef& ref) {
  auto it = idx_of.find(ref);
  return it != idx_of.end() ? absl::StrCat("%", it->second) : "?";
}

// "%a, %b: f32[s], f32[s] = op_name(%c, %d)    # file:line"
void AppendFxOpRow(std::string& out, const SharedDeviceBufferList& node,
                   const DeferredOp& op,
                   const absl::flat_hash_map<DeviceBufferRef, size_t>& idx_of) {
  std::string indices_str;
  std::string types_str;

  for (int i = 0; i < node->size(); ++i) {
    auto out_or = DeviceBufferRef::Create(node, i);
    bool ok = out_or.ok();

    absl::StrAppend(&indices_str, i == 0 ? "" : ", ",
                    ok ? FormatRefIndex(idx_of, *out_or) : "?");
    absl::StrAppend(&types_str, i == 0 ? "" : ", ",
                    ok ? FormatTypeCompact(*out_or) : "?");
  }

  absl::StrAppend(&out, indices_str, ": ", types_str, " = ",
                  ToString(op.op_name()), "(");
  bool first = true;
  for (const DeviceBufferRef& operand : op.inputs()) {
    absl::StrAppend(&out, first ? "" : ", ", FormatRefIndex(idx_of, operand));
    first = false;
  }
  absl::StrAppend(&out, ")");
  if (const auto& tb = op.op_context().traceback()) {
    if (const PythonStackFrame* f = PickUserFrame(*tb)) {
      absl::StrAppend(&out, "    # ", f->file_name, ":", f->line_number, " in ",
                      f->func_name);
    }
  }
  absl::StrAppend(&out, "\n");
}

}  // namespace

std::string Traversal::ReadableString(MaterializationReason reason) const {
  const auto idx_of = BuildBufferIndexMap(arguments_, execution_order_);

  // Compute op_count and accumulate op rows in one pass over execution_order.
  std::string op_rows;
  size_t op_count = 0;
  for (const SharedDeviceBufferList& node : execution_order_) {
    const auto op = node->deferred_op();
    if (op == nullptr) continue;
    AppendFxOpRow(op_rows, node, *op, idx_of);
    ++op_count;
  }

  std::string out =
      absl::StrCat("# Graph: ", op_count, " ops, ", arguments_.size(),
                   " inputs, reason: ", ToString(reason), "\n");

  size_t input_idx = 0;
  for (const DeviceBufferRef& argument : arguments_) {
    absl::StrAppend(&out, "%", input_idx++, ": ", FormatTypeCompact(argument),
                    " = input\n");
  }
  out += op_rows;
  if (!outputs_.empty()) {
    absl::StrAppend(&out, "return ");
    bool first = true;
    for (const DeviceBufferRef& output : outputs_) {
      absl::StrAppend(&out, first ? "" : ", ", FormatRefIndex(idx_of, output));
      first = false;
    }
    absl::StrAppend(&out, "\n");
  }
  return out;
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
      if (const auto deferred_op = node->deferred_op()) {
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
    if (ref.is_placeholder()) {
      os << " (placeholder)";
    } else if (ref.is_materializing()) {
      os << " (materialized)";
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

  // Get buffers from the traversal's arguments.
  for (const auto& argument : arguments_) {
    buffer_to_index[argument] = buffer_index++;
  }

  // Validate internal buffers.
  for (auto i = 0; i < execution_order_.size(); ++i) {
    const SharedDeviceBufferList& node = execution_order_[i];
    // Validate that all traversal's internal buffers are from deferred ops
    // using either buffer inputs or outputs from previously encountered
    // deferred ops.
    const auto deferred_op = node->deferred_op();
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
  for (const DeviceBufferRef& argument : arguments_) {
    if (!argument.dynamic_dimensions().empty()) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<std::string> GetGraphviz(
    const Traversal& traversal,
    const absl::flat_hash_map<DeviceBufferRef, std::string>&
        buffer_ref_to_python_var) {
  TT_ASSIGN_OR_RETURN(GraphvizGraph graph,
                      GraphvizGraph::Create(traversal.arguments(),
                                            traversal.execution_order()));

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
