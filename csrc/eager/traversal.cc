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

#include "csrc/eager/traversal.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <stack>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
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
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_cache.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/compile_options_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fingerprint_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_assignment_pipeline.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "csrc/pjrt/pjrt_utils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/client/executable_build_options.h"
#include "xla/layout.h"
#include "xla/service/device_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"
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

GraphKey AnnotateGraphKeyWithArgumentLayouts(
    const GraphKey& graph_key,
    absl::Span<const CustomLayout> argument_layouts) {
  if (argument_layouts.empty()) return graph_key;
  FingerprintType fp = graph_key.shapeless_key().key();
  for (const CustomLayout& layout : argument_layouts) {
    if (!layout.minor_to_major.empty()) {
      fp = FingerprintCat(fp, FingerprintCat(layout.minor_to_major));
      for (const Indices& tile : layout.tiles) {
        fp = FingerprintCat(fp, FingerprintCat(tile));
      }
      fp = FingerprintCat(fp, layout.element_size_in_bits);
    } else {
      fp = FingerprintCat(fp, 0);
    }
  }
  return GraphKey(ShapelessKey(fp), graph_key.dimensions_key());
}

GraphKey AnnotateGraphKeyWithDonatedInputs(
    const GraphKey& graph_key, absl::Span<const int64_t> donated_inputs) {
  if (donated_inputs.empty()) return graph_key;
  Indices normalized_donated_inputs = CopyIntVector(donated_inputs);
  std::sort(normalized_donated_inputs.begin(), normalized_donated_inputs.end());
  normalized_donated_inputs.erase(std::unique(normalized_donated_inputs.begin(),
                                              normalized_donated_inputs.end()),
                                  normalized_donated_inputs.end());
  return GraphKey(ShapelessKey(FingerprintCat(graph_key.shapeless_key().key(),
                                              normalized_donated_inputs)),
                  graph_key.dimensions_key());
}

CompilationCacheKey Traversal::GetCacheKey(
    CompileOptionsKey compile_options_key,
    absl::Span<const CustomLayout> argument_layouts,
    absl::Span<const int64_t> donated_inputs) const {
  if (graph_key_ == std::nullopt) {
    graph_key_ = BuildGraphKey();
  }
  const GraphKey layout_key =
      AnnotateGraphKeyWithArgumentLayouts(*graph_key_, argument_layouts);
  return CompilationCacheKey(
      AnnotateGraphKeyWithDonatedInputs(layout_key, donated_inputs),
      compile_options_key);
}

GraphKey Traversal::BuildGraphKey() const {
  tsl::profiler::TraceMe t("Traversal::BuildGraphKey");
  // We will be building a GraphSignature object as a simplified
  // representation of the Traversal graph for the purposes of hashing.
  GraphSignature graph;

  graph.SetCorePinningMode(core_pinning_mode_);

  absl::flat_hash_map<DeviceBufferRef, int> tensor_index_map;

  // Add all arguments to tensor-indexed properties.
  for (const DeviceBufferRef& argument : arguments()) {
    tensor_index_map[argument] =
        graph.AddInput(argument.dimensions(), argument.element_type());
  }
  int op_idx = 0;
  for (const SharedDeviceBufferList& node : execution_order()) {
    const auto maybe_deferred_op = node->deferred_op();
    ABSL_CHECK(maybe_deferred_op != nullptr);  // CRASH_OK
    const DeferredOp& deferred_op = *maybe_deferred_op;

    ABSL_VLOG(3) << "[Traversal::BuildGraphKey] Op #" << op_idx++ << " ("
                 << ToString(deferred_op.op_name()) << ") | inputs: ["
                 << absl::StrJoin(
                        deferred_op.inputs(), ", ",
                        [&]  // REF_CAPTURE_OK=called synchronously
                        (std::string* out, const DeviceBufferRef& op_input) {
                          absl::StrAppend(out, tensor_index_map.at(op_input));
                        })
                 << "] | params: {"
                 << absl::StrJoin(deferred_op.op_param_cache_keys(), ", ",
                                  absl::PairFormatter("="))
                 << "}";
    const auto& orig_param_keys = deferred_op.op_param_cache_keys();
    std::optional<OpParamCacheKeys> cloned_param_keys;
    if (!deferred_op.device_assignment().empty()) {
      // Some ops (e.g. Pallas custom kernels) already record
      // "device_assignment" in their OpParamCacheKeys at op construction time,
      // whereas process-group collectives attach device_assignment only on the
      // DeferredOp. Since OpParamCacheKeys::SetParam CHECK-fails on duplicate
      // keys, only insert "device_assignment" when it is not already present.
      const bool has_da = std::any_of(
          orig_param_keys.begin(), orig_param_keys.end(),
          [](const auto& param) { return param.first == "device_assignment"; });
      if (!has_da) {
        cloned_param_keys = orig_param_keys.Clone();
        ABSL_CHECK_OK(  // CRASH_OK
            cloned_param_keys->SetParam("device_assignment",
                                        deferred_op.device_assignment()));
        // A program that combines process groups tiles the op's collectives
        // over the groups of its layout.
        if (deferred_op.global_device_assignment() !=
            deferred_op.device_assignment()) {
          ABSL_CHECK_OK(  // CRASH_OK
              cloned_param_keys->SetParam(
                  "global_device_assignment",
                  deferred_op.global_device_assignment()));
        }
      }
    }
    const OpParamCacheKeys& op_param_keys =
        cloned_param_keys.has_value() ? *cloned_param_keys : orig_param_keys;
    graph.AddOp(
        deferred_op.op_name(), op_param_keys, deferred_op.donated_indices(),
        [&]  // REF_CAPTURE_OK=called synchronously
        (GraphSignature::OpSignatureBuilder& op) {
          for (const DeviceBufferRef& op_input : deferred_op.inputs()) {
            op.AddInput(tensor_index_map.at(op_input));
          }
          for (int64_t i = 0; i < node->size(); ++i) {
            auto output = DeviceBufferRef::Create(node, i);
            ABSL_CHECK(output.ok())  // CRASH_OK
                << "Failed to create DeviceBufferRef for output: " << i;
            tensor_index_map[*output] =
                op.AddOutput(output->dimensions(), output->element_type());
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
        << ToString(input.shape());
    if (auto it = prev_arguments.find(input); it != prev_arguments.end()) {
      // The first time we see a previous argument in the new arguments, mark
      // it as used.
      // If we see it again, then the new arguments has a duplicate and is
      // invalid.
      TT_RET_CHECK(!it->second, error::kInvalidArgument)
          << "identified a duplicate input: " << ToString(input.shape());
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
        << ToString(arg.shape());
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

void AnnotateMlirPlacements(mlir::ModuleOp module,
                            absl::Span<const DeviceBufferRef> outputs,
                            absl::Span<const DeviceBufferRef> arguments) {
  auto func = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!func) {
    return;
  }

  mlir::MLIRContext* const context = func.getContext();
  const auto pinned_host_attr = mlir::StringAttr::get(context, "pinned_host");
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i].is_pinned_host()) {
      func.setResultAttr(i, "mhlo.memory_kind", pinned_host_attr);
    }
  }
  for (size_t i = 0; i < arguments.size(); ++i) {
    if (arguments[i].is_pinned_host()) {
      func.setArgAttr(i, "mhlo.memory_kind", pinned_host_attr);
    }
  }
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
    mlir::MLIRContext& mlir_context, bool use_stablehlo_bounds,
    absl::Span<const int64_t> donated_inputs,
    std::optional<std::string_view> module_name) const {
  TT_ASSIGN_OR_RETURN(const std::vector<int64_t> resolved_da,  // INT_VEC_OK
                      ResolveDeviceAssignment());

  // Read the traversal's values.
  absl::Span<const DeviceBufferRef> arguments = this->arguments();
  absl::Span<const SharedDeviceBufferList> execution_order =
      this->execution_order();
  absl::Span<const DeviceBufferRef> outputs = this->outputs();

  // Initialize the module builder and main function builder.
  std::string module_name_str;
  if (module_name.has_value() && !module_name->empty()) {
    module_name_str = std::string(*module_name);
  } else {
    const PythonContext* absl_nullable python_context = GetPythonContext();
    module_name_str =
        BuildModuleNameFromPyContext(mlir_context, python_context);
  }
  mlir::ModuleBuilder mb(mlir_context, module_name_str);
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
        << "in execution_order, a DeviceBufferList has no deferred op";
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

    const bool has_op_da = !deferred_op.device_assignment().empty();
    mlir::Block* insertion_block = fb.getOpBuilder().getInsertionBlock();
    mlir::Operation* prev_last_op =
        (insertion_block != nullptr && !insertion_block->empty())
            ? &insertion_block->back()
            : nullptr;
    mlir::ModuleOp module_op = GetModuleOp(fb);
    absl::flat_hash_set<std::string> existing_funcs;
    if (has_op_da) {
      for (mlir::func::FuncOp func_op :
           module_op.getOps<mlir::func::FuncOp>()) {
        existing_funcs.insert(func_op.getSymName().str());
      }
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

    if (has_op_da) {
      mlir::Operation* first_new_op = nullptr;
      if (insertion_block != nullptr && !insertion_block->empty()) {
        first_new_op = (prev_last_op != nullptr) ? prev_last_op->getNextNode()
                                                 : &insertion_block->front();
      }
      std::vector<mlir::func::FuncOp> new_func_ops;
      for (mlir::func::FuncOp func_op :
           module_op.getOps<mlir::func::FuncOp>()) {
        if (!existing_funcs.contains(func_op.getSymName().str())) {
          new_func_ops.push_back(func_op);
        }
      }
      TT_RETURN_IF_ERROR(ApplySubgroupTilingOrReplicaRewriting(
          first_new_op, new_func_ops, deferred_op, resolved_da));
    }

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
  for (int64_t donated_input : donated_inputs) {
    if (donated_input >= 0 && donated_input < arguments.size()) {
      if (absl::c_find(donated_arguments, donated_input) ==
          donated_arguments.end()) {
        donated_arguments.push_back(donated_input);
      }
    }
  }
  if (!donated_arguments.empty()) {
    AnnotateBufferDonations(module.get(), donated_arguments);
  }
  AnnotateMlirPlacements(module.get(), outputs, arguments);
  if (core_pinning_mode_ != CorePinningMode::kUnpinned) {
    auto op_builder = mlir::OpBuilder(module.get());
    std::vector<mlir::NamedAttribute> frontend_attrs;
    frontend_attrs.reserve(1);
    auto string_attr = op_builder.getStringAttr(
        core_pinning_mode_ == CorePinningMode::kSparseCore
            ? xla::kXlaComputeTypeSparseOffload
            : xla::kXlaComputeTypeDense);
    frontend_attrs.push_back(
        op_builder.getNamedAttr(xla::kXlaComputeTypeAttr, string_attr));

    module.get()->setAttr("mhlo.frontend_attributes",
                          op_builder.getDictionaryAttr(frontend_attrs));
  }

  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  mlir_context.appendDialectRegistry(registry);

  mlir::PassManager pm(&mlir_context);
  pm.addPass(mlir::createInlinerPass());
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=MLIR pass error not reachable by user
                 // input.
      mlir::succeeded(pm.run(module.get())), error::kInternal)
      << "failed to inline functions in MLIR module";

  return module;
}

namespace {
std::vector<Shape> GetShapes(absl::Span<const DeviceBufferRef> buffers) {
  std::vector<Shape> shapes;
  shapes.reserve(buffers.size());
  for (const auto& buffer : buffers) {
    shapes.push_back(buffer.shape());
  }
  return shapes;
}

void AnnotateArgumentLayouts(mlir::ModuleOp module,
                             absl::Span<const DeviceBufferRef> arguments,
                             absl::Span<const CustomLayout> argument_layouts) {
  if (argument_layouts.empty()) return;

  mlir::func::FuncOp main = module.lookupSymbol<mlir::func::FuncOp>("main");
  ABSL_CHECK(main)  // CRASH_OK: Should never call API on a malformed module.
      << "MLIR module does not contain a main function for argument layouts.";

  ABSL_CHECK_EQ(argument_layouts.size(), arguments.size())  // CRASH_OK
      << "argument_layouts size (" << argument_layouts.size()
      << ") must match arguments size (" << arguments.size() << ")";
  ABSL_CHECK_LE(argument_layouts.size(), main.getNumArguments())  // CRASH_OK
      << "argument_layouts size (" << argument_layouts.size()
      << ") exceeds MLIR main argument count (" << main.getNumArguments()
      << ")";

  mlir::Builder builder(main.getContext());
  for (size_t i = 0; i < argument_layouts.size(); ++i) {
    const CustomLayout& layout = argument_layouts[i];
    if (!layout.minor_to_major.empty()) {
      absl::Span<const int64_t> dims = arguments[i].dimensions();
      ABSL_CHECK_GE(layout.minor_to_major.size(), dims.size())  // CRASH_OK
          << "minor_to_major size (" << layout.minor_to_major.size()
          << ") is smaller than argument " << i << " rank (" << dims.size()
          << ")";
      main.setArgAttr(i, "mhlo.layout_mode",
                      builder.getStringAttr(ToXlaLayout(layout).ToString()));
    }
  }
}

void AnnotateOutputLayouts(mlir::ModuleOp module,
                           absl::Span<const DeviceBufferRef> outputs) {
  mlir::func::FuncOp main = module.lookupSymbol<mlir::func::FuncOp>("main");
  ABSL_CHECK(main)  // CRASH_OK: Should never call API on a malformed module.
      << "MLIR module does not contain a main function for output layouts.";
  ABSL_CHECK_LE(outputs.size(), main.getNumResults())  // CRASH_OK
      << "outputs size (" << outputs.size()
      << ") exceeds MLIR main result count (" << main.getNumResults() << ")";

  mlir::Builder builder(main.getContext());
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (!outputs[i].shape().layout().has_value()) continue;
    const CustomLayout& layout = *outputs[i].shape().layout();
    if (!layout.minor_to_major.empty() &&
        !IsDefaultDeviceLayout(layout, outputs[i].element_type(),
                               outputs[i].dimensions())) {
      main.setResultAttr(i, "mhlo.layout_mode",
                         builder.getStringAttr(ToXlaLayout(layout).ToString()));
    }
  }
}

std::string MlirModuleToString(mlir::ModuleOp module) {
  mlir::OpPrintingFlags flags;
  flags.elideLargeElementsAttrs(100);
  std::string s;
  llvm::raw_string_ostream os(s);
  module.print(os, flags);
  return os.str();
}

bool HasPinnedHostBuffer(absl::Span<const DeviceBufferRef> buffers) {
  return absl::c_any_of(
      buffers, [](const DeviceBufferRef& b) { return b.is_pinned_host(); });
}

std::vector<xla::Shape> GetXlaShapesWithPlacements(
    absl::Span<const DeviceBufferRef> buffers) {
  std::vector<xla::Shape> shapes;
  shapes.reserve(buffers.size());
  for (const DeviceBufferRef& buffer : buffers) {
    auto shape = xla::ShapeUtil::MakeShape(
        ConvertTo<xla::PrimitiveType>(buffer.element_type()),
        buffer.dimensions());
    if (buffer.is_pinned_host()) {
      // Keep the default layout for host buffers.
      shape.mutable_layout()->set_memory_space(xla::Layout::kHostMemorySpace);
    } else {
      // Clear layout so the compiler can assign a different layout if it wants.
      shape.clear_layout();
    }
    shapes.push_back(std::move(shape));
  }
  return shapes;
}

void ApplyPlacementsToCompileSpec(CompilationSpec& spec,
                                  absl::Span<const DeviceBufferRef> outputs,
                                  absl::Span<const DeviceBufferRef> arguments) {
  bool changed = false;
  if (HasPinnedHostBuffer(outputs)) {
    spec.xla_compile_options->executable_build_options.set_result_layout(
        xla::ShapeUtil::MakeMaybeTupleShape(
            GetXlaShapesWithPlacements(outputs)));
    changed = true;
  }

  if (HasPinnedHostBuffer(arguments)) {
    spec.xla_compile_options->argument_layouts =
        GetXlaShapesWithPlacements(arguments);
    changed = true;
  }

  if (changed) {
    spec.compile_options_key = MakeCompileOptionsKey(*spec.xla_compile_options);
  }
}

}  // namespace

absl::StatusOr<std::vector<int64_t>>  // INT_VEC_OK
Traversal::ResolveDeviceAssignment() const {
  if (!resolved_da_.has_value()) {
    TT_ASSIGN_OR_RETURN(std::vector<int64_t> resolved,  // INT_VEC_OK
                        ResolveDeviceAssignmentForTraversal(execution_order_));
    resolved_da_ = std::move(resolved);
  }
  return *resolved_da_;
}

absl::Status Traversal::ApplyDeviceAssignmentToCompileSpec(
    CompilationSpec& spec, absl::Span<const int64_t> resolved_da) const {
  // A non-empty resolved device assignment (collectives, Pallas, SparseCore)
  // pins compilation to those devices, in order, with a distinct options key.
  if (!resolved_da.empty()) {
    const int num_partitions = resolved_da.size();
    xla::ExecutableBuildOptions& exec_options =
        spec.xla_compile_options->executable_build_options;
    exec_options.set_num_replicas(1);
    exec_options.set_num_partitions(num_partitions);
    xla::DeviceAssignment da(1, num_partitions);
    for (int p = 0; p < num_partitions; ++p) {
      da(0, p) = resolved_da[p];
    }
    exec_options.set_device_assignment(da);
    spec.compile_options_key = MakeCompileOptionsKey(*spec.xla_compile_options);
  }
  return absl::OkStatus();
}

absl::Status Traversal::ApplyDeviceAssignmentToCompileSpec(
    CompilationSpec& spec) const {
  TT_ASSIGN_OR_RETURN(const std::vector<int64_t> resolved_da,  // INT_VEC_OK
                      ResolveDeviceAssignment());
  return ApplyDeviceAssignmentToCompileSpec(spec, resolved_da);
}

std::optional<CompilationSpec> Traversal::MaybeUpdateSpecWithLayout(
    const CompilationSpec& spec, absl::Span<const int64_t> resolved_da) const {
  // Avoid deep-copying `CompilationSpec` (`xla::CompileOptions` and its
  // `DebugOptions` protobuf) and re-hashing `CompileOptionsKey` on the common
  // eager compilation cache-hit fast path unless this traversal requires
  // explicit device assignment overrides or pinned-host memory space layout
  // overrides.
  const bool needs_placement_override =
      HasPinnedHostBuffer(outputs_) || HasPinnedHostBuffer(arguments_);
  if (resolved_da.empty() && !needs_placement_override) return std::nullopt;

  CompilationSpec mutable_compilation_spec = spec.Copy();
  if (!resolved_da.empty()) {
    ABSL_CHECK_OK(ApplyDeviceAssignmentToCompileSpec(  // CRASH_OK
        mutable_compilation_spec, resolved_da));
  }
  if (needs_placement_override) {
    ApplyPlacementsToCompileSpec(mutable_compilation_spec, outputs_,
                                 arguments_);
  }
  return mutable_compilation_spec;
}

absl::StatusOr<CompiledKernel> Traversal::Compile(
    const CompilationSpec& spec, std::string* absl_nullable out_mlir_text,
    bool use_stablehlo_bounds, absl::Span<const CustomLayout> argument_layouts,
    absl::Span<const int64_t> donated_inputs,
    std::optional<std::string_view> module_name,
    std::optional<CompilationCacheKey>* absl_nullable out_cache_key) const {
  tsl::profiler::TraceMe trace_await("Traversal::Compile");
  if (out_cache_key != nullptr) {
    *out_cache_key = std::nullopt;
  }
  TT_ASSIGN_OR_RETURN(const std::vector<int64_t> resolved_da,  // INT_VEC_OK
                      ResolveDeviceAssignment());
  if (!argument_layouts.empty()) {
    TT_RET_CHECK(arguments_.size() == argument_layouts.size(),
                 error::kInvalidArgument)
        << "expected the size of argument layouts to be equal to "
        << arguments_.size() << " (number of arguments), got "
        << argument_layouts.size();
  }

  // Prepare a computation builder closure to be called on a cache miss.  Okay
  // to capture this here since CompilationCache::GetOrCompile() will call this
  // builder before the function returns and in the same thread it is invoked.
  MlirComputationBuilder final_op_builder =
      [this, use_stablehlo_bounds, donated_inputs, argument_layouts,
       module_name, out_mlir_text](mlir::MLIRContext& mlir_context)
      -> absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> {
    TT_ASSIGN_OR_RETURN(auto module,
                        BuildMlirModule(mlir_context, use_stablehlo_bounds,
                                        donated_inputs, module_name));
    AnnotateArgumentLayouts(*module, arguments_, argument_layouts);
    AnnotateOutputLayouts(*module, outputs_);
    if (out_mlir_text != nullptr) {
      *out_mlir_text = MlirModuleToString(*module);
    }
    return module;
  };
  std::vector<Shape> argument_shapes = GetShapes(arguments_);
  std::vector<Shape> output_shapes = GetShapes(outputs_);

  std::optional<CompilationSpec> mutable_compilation_spec =
      MaybeUpdateSpecWithLayout(spec, resolved_da);
  const CompilationSpec& effective_compilation_spec =
      mutable_compilation_spec.has_value() ? *mutable_compilation_spec : spec;

  CompilationCacheKey compilation_cache_key =
      GetCacheKey(effective_compilation_spec.compile_options_key,
                  argument_layouts, donated_inputs);
  if (out_cache_key != nullptr) {
    out_cache_key->emplace(compilation_cache_key);
  }
  ABSL_VLOG(1) << "[Compile] compilation cache key: " << compilation_cache_key;

  if (mutable_compilation_spec.has_value()) {
    return CompilationCache::GetInstance().GetOrCompile(
        std::move(compilation_cache_key), argument_shapes, output_shapes,
        std::move(final_op_builder),
        std::move(mutable_compilation_spec->xla_compile_options),
        /*use_dynamic_adapters=*/!use_stablehlo_bounds);
  }
  return CompilationCache::GetInstance().GetOrCompile(
      std::move(compilation_cache_key), argument_shapes, output_shapes,
      std::move(final_op_builder),
      *effective_compilation_spec.xla_compile_options,
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
