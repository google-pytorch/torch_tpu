/*
 * Copyright 2026 Google LLC
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

#include "torch_tpu/_internal/compile/dispatch_invoke_subgraph.h"

#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fingerprint_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"

namespace torch_tpu {
namespace {

std::vector<DeviceBufferRef> GetInputBuffers(
    const std::vector<at::Tensor>& inputs) {
  std::vector<DeviceBufferRef> input_buffers;
  input_buffers.reserve(inputs.size());
  absl::c_transform(inputs, std::back_inserter(input_buffers),
                    [](const at::Tensor& input) -> DeviceBufferRef {
                      TT_ASSIGN_OR_THROW(const DeviceBufferRef ref,
                                         GetBuffer(input));
                      return ref;
                    });
  return input_buffers;
}

std::vector<Shape> GetOutputShapes(
    const std::vector<at::Tensor>& dummy_outputs) {
  std::vector<Shape> output_shapes;
  output_shapes.reserve(dummy_outputs.size());
  absl::c_transform(dummy_outputs, std::back_inserter(output_shapes),
                    [](const at::Tensor& output) -> Shape {
                      TT_ASSIGN_OR_THROW(
                          const auto dtype,
                          ConvertTo<mlir::ElementType>(output.scalar_type()));
                      return Shape(CopyIntVector(output.sizes()), dtype);
                    });
  return output_shapes;
}

absl::StatusOr<mlir::func::FuncOp> ImportBody(mlir::MlirBuilder& builder,
                                              std::string_view serialized_mlir,
                                              std::string_view function_name) {
  mlir::ModuleOp destination = GetModuleOp(builder);
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      serialized_mlir, mlir::ParserConfig{&builder.getContext()});
  TT_RET_CHECK(source.get() != nullptr, error::kInternal)
      << "failed to parse invoke_subgraph body MLIR module";

  mlir::func::FuncOp source_main =
      source->lookupSymbol<mlir::func::FuncOp>("main");
  TT_RET_CHECK(source_main != nullptr, error::kInternal)
      << "expected 'main' function in invoke_subgraph body MLIR module";

  if (auto existing =
          destination.lookupSymbol<mlir::func::FuncOp>(function_name)) {
    TT_RET_CHECK(existing.getFunctionType() == source_main.getFunctionType(),
                 error::kInvalidArgument)
        << "invoke_subgraph identifier was reused with an incompatible "
           "function signature";
    return existing;
  }

  mlir::SymbolTableCollection symbol_tables;
  mlir::SymbolTable& destination_symbols =
      symbol_tables.getSymbolTable(destination);
  mlir::SymbolTable& source_symbols = symbol_tables.getSymbolTable(*source);

  TT_RET_CHECK(
      mlir::succeeded(source_symbols.rename(source_main, function_name)),
      error::kInternal)
      << "failed to rename invoke_subgraph body function";
  for (mlir::func::FuncOp function : source->getOps<mlir::func::FuncOp>()) {
    if (function == source_main) continue;
    TT_RET_CHECK(mlir::succeeded(source_symbols.renameToUnique(
                     function, &destination_symbols)),
                 error::kInternal)
        << "failed to uniquely rename invoke_subgraph helper function";
  }

  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::OpBuilder::InsertionGuard guard(op_builder);
  op_builder.setInsertionPointToStart(destination.getBody());

  mlir::func::FuncOp imported_main;
  for (mlir::func::FuncOp function : source->getOps<mlir::func::FuncOp>()) {
    auto cloned = mlir::cast<mlir::func::FuncOp>(op_builder.clone(*function));
    cloned.setVisibility(mlir::SymbolTable::Visibility::Private);
    if (function == source_main) imported_main = cloned;
  }

  TT_RET_CHECK(imported_main != nullptr, error::kInternal)
      << "failed to import invoke_subgraph body function";
  return imported_main;
}

void SetCompilationBoundaryAttributes(mlir::MlirBuilder& builder,
                                      mlir::Operation* call_op,
                                      std::string_view identifier) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const auto frontend_attributes = op_builder.getDictionaryAttr({
      op_builder.getNamedAttr("compilation_unit",
                              op_builder.getStringAttr(identifier)),
      op_builder.getNamedAttr("inlineable",
                              op_builder.getStringAttr("xla_late")),
  });
  call_op->setAttr("mhlo.frontend_attributes", frontend_attributes);
}

}  // namespace

std::vector<at::Tensor> PyCreateInvokeSubgraphOp(
    const std::vector<at::Tensor>& inputs,
    const std::shared_ptr<ContextedModule>& body_module,
    const std::vector<at::Tensor>& dummy_outputs, std::string_view identifier) {
  ScopedPythonContextCapturer capturer(OpName::kCustomKernel);

  TT_CHECK_THROW(!inputs.empty(), error::kInvalidArgument)
      << "expected at least 1 input tensor";
  TT_CHECK_THROW(!dummy_outputs.empty(), error::kInvalidArgument)
      << "expected at least 1 output tensor";
  TT_CHECK_THROW(body_module != nullptr && body_module->get(),
                 error::kInvalidArgument)
      << "invalid body module";

  mlir::func::FuncOp body_main =
      body_module->get().lookupSymbol<mlir::func::FuncOp>("main");
  TT_CHECK_THROW(body_main != nullptr, error::kInternal)
      << "expected 'main' function in invoke_subgraph body MLIR module";
  TT_CHECK_THROW(body_main.getNumArguments() == inputs.size(),
                 error::kInvalidArgument)
      << "expected " << body_main.getNumArguments() << " inputs, got "
      << inputs.size();
  TT_CHECK_THROW(body_main.getNumResults() == dummy_outputs.size(),
                 error::kInvalidArgument)
      << "expected " << body_main.getNumResults() << " dummy outputs, got "
      << dummy_outputs.size();

  std::string serialized_mlir;
  {
    llvm::raw_string_ostream os(serialized_mlir);
    body_module->get().print(os);
  }

  const std::string compilation_unit =
      identifier.empty() ? absl::StrCat("invoke_subgraph_0x",
                                        absl::Hex(Fingerprint(serialized_mlir)))
                         : std::string(identifier);
  const std::string function_name = absl::StrCat(
      "invoke_subgraph_0x", absl::Hex(Fingerprint(compilation_unit)));

  MlirOpBuilder op_builder = [serialized_mlir, function_name, compilation_unit](
                                 mlir::MlirBuilder& builder,
                                 absl::Span<mlir::MlirOp> op_inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    TT_ASSIGN_OR_RETURN(mlir::func::FuncOp body,
                        ImportBody(builder, serialized_mlir, function_name));
    TT_RET_CHECK(body.getNumArguments() == op_inputs.size(),
                 error::kInvalidArgument)
        << "expected " << body.getNumArguments() << " inputs, got "
        << op_inputs.size();

    std::vector<mlir::MlirOp> call_inputs;
    call_inputs.reserve(op_inputs.size());
    for (auto [index, input] : llvm::enumerate(op_inputs)) {
      const auto actual_type =
          mlir::dyn_cast<mlir::RankedTensorType>(input.getValue().getType());
      const auto expected_type = mlir::dyn_cast<mlir::RankedTensorType>(
          body.getArgument(index).getType());
      TT_RET_CHECK(actual_type && expected_type, error::kInvalidArgument)
          << "invoke_subgraph inputs must be ranked tensors";
      if (actual_type == expected_type) {
        call_inputs.push_back(input);
        continue;
      }
      TT_RET_CHECK(actual_type.hasStaticShape() &&
                       expected_type.hasStaticShape() &&
                       actual_type.getElementType() ==
                           expected_type.getElementType() &&
                       actual_type.getNumElements() ==
                           expected_type.getNumElements(),
                   error::kInvalidArgument)
          << "invoke_subgraph input " << index
          << " cannot be reshaped to the body parameter type";
      call_inputs.push_back(mlir::stablehlo::Reshape(expected_type, input));
    }

    DynamicMlirOpResults results = mlir::func::Call(builder, body, call_inputs);
    TT_RET_CHECK(!results.empty(), error::kInternal)
        << "invoke_subgraph body must return at least one result";
    SetCompilationBoundaryAttributes(
        builder, results.front().getValue().getDefiningOp(), compilation_unit);
    return results;
  };

  TT_ASSIGN_OR_THROW(auto cache_keys,
                     *(OpParamCacheKeysBuilder()
                           .SetParam("body_mlir", serialized_mlir)
                           .SetParam("identifier", compilation_unit)));

  TT_ASSIGN_OR_THROW(
      const auto result_refs,
      DeviceBufferList::CreateDeferred(
          OpName::kCustomKernel, std::move(op_builder), GetInputBuffers(inputs),
          std::move(cache_keys), GetOutputShapes(dummy_outputs)));

  std::vector<at::Tensor> results;
  results.reserve(result_refs.size());
  absl::c_transform(result_refs, std::back_inserter(results),
                    [](const DeviceBufferRef& ref) { return MakeTensor(ref); });
  return results;
}

}  // namespace torch_tpu
