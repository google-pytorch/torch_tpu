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

#include "torch_tpu/_internal/compile/dispatch_cond.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"

namespace torch_tpu {
namespace {

struct BranchModules {
  std::string true_mlir;
  std::string false_mlir;
};

std::string SerializeModule(const ContextedModule& module) {
  std::string serialized_mlir;
  llvm::raw_string_ostream os(serialized_mlir);
  module.get().print(os);
  os.flush();
  return serialized_mlir;
}

std::string TypeString(const mlir::Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  type.print(os);
  os.flush();
  return text;
}

std::vector<DeviceBufferRef> GetInputBuffers(
    const at::Tensor& pred, const std::vector<at::Tensor>& operands) {
  std::vector<DeviceBufferRef> input_buffers;
  input_buffers.reserve(1 + operands.size());

  auto get_buffer = [](const at::Tensor& t) -> DeviceBufferRef {
    TT_ASSIGN_OR_THROW(const DeviceBufferRef ref, GetBuffer(t));
    return ref;
  };

  input_buffers.push_back(get_buffer(pred));
  absl::c_transform(operands, std::back_inserter(input_buffers), get_buffer);
  return input_buffers;
}

std::vector<Shape> GetOutputShapes(
    const std::vector<at::Tensor>& dummy_outputs) {
  std::vector<Shape> output_shapes;
  output_shapes.reserve(dummy_outputs.size());

  auto get_output_shape = [](const at::Tensor& t) -> Shape {
    TT_ASSIGN_OR_THROW(const auto dtype,
                       ConvertTo<mlir::ElementType>(t.scalar_type()));
    return Shape(CopyIntVector(t.sizes()), dtype);
  };

  absl::c_transform(dummy_outputs, std::back_inserter(output_shapes),
                    get_output_shape);
  return output_shapes;
}

mlir::func::FuncOp GetMainFuncOrThrow(
    const std::shared_ptr<ContextedModule>& module,
    const std::string& branch_name) {
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      module != nullptr && module->get(), error::kInvalidArgument)
      << "invalid " << branch_name << " branch module";

  auto main_func = module->get().lookupSymbol<mlir::func::FuncOp>("main");
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      main_func != nullptr, error::kInternal)
      << "expected 'main' function in " << branch_name << " branch MLIR module";
  return main_func;
}

void ValidateDummyOutputType(const at::Tensor& dummy_output,
                             const mlir::RankedTensorType result_type,
                             const int64_t index) {
  const Dimensions expected_shape = CopyIntVector(dummy_output.sizes());
  const Dimensions actual_shape(result_type.getShape().begin(),
                                result_type.getShape().end());
  TT_CHECK_THROW(actual_shape == expected_shape, error::kInvalidArgument)
      << "torch.cond branch output " << index
      << " shape mismatch with inferred output shape: branch has "
      << ToString(actual_shape) << ", inferred output has "
      << ToString(expected_shape);

  TT_ASSIGN_OR_THROW(const auto expected_element_type,
                     ConvertTo<mlir::ElementType>(dummy_output.scalar_type()));
  const mlir::Type expected_mlir_type =
      mlir::getElementType(*result_type.getContext(), expected_element_type);
  TT_CHECK_THROW(result_type.getElementType() == expected_mlir_type,
                 error::kInvalidArgument)
      << "torch.cond branch output " << index
      << " dtype mismatch with inferred output dtype: branch has "
      << TypeString(result_type.getElementType()) << ", inferred output has "
      << TypeString(expected_mlir_type);
}

void ValidateBranchFunctionsOrThrow(
    mlir::func::FuncOp true_main_func, mlir::func::FuncOp false_main_func,
    const int64_t num_operands, const std::vector<at::Tensor>& dummy_outputs) {
  TT_CHECK_THROW(true_main_func.getNumArguments() == num_operands,
                 error::kInvalidArgument)
      << "expected true branch to have " << num_operands
      << " tensor arguments, got " << true_main_func.getNumArguments();
  TT_CHECK_THROW(false_main_func.getNumArguments() == num_operands,
                 error::kInvalidArgument)
      << "expected false branch to have " << num_operands
      << " tensor arguments, got " << false_main_func.getNumArguments();

  const mlir::TypeRange true_result_types = true_main_func.getResultTypes();
  const mlir::TypeRange false_result_types = false_main_func.getResultTypes();
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      !dummy_outputs.empty(), error::kInvalidArgument)
      << "torch.cond TPU branches must return at least one tensor";
  TT_CHECK_THROW(true_result_types.size() == dummy_outputs.size(),
                 error::kInvalidArgument)
      << "expected true branch to have " << dummy_outputs.size()
      << " result types, got " << true_result_types.size();
  TT_CHECK_THROW(false_result_types.size() == dummy_outputs.size(),
                 error::kInvalidArgument)
      << "expected false branch to have " << dummy_outputs.size()
      << " result types, got " << false_result_types.size();

  for (int64_t i = 0; i < true_result_types.size(); ++i) {
    const mlir::Type true_type = true_result_types[i];
    const mlir::Type false_type = false_result_types[i];
    TT_CHECK_THROW(TypeString(true_type) == TypeString(false_type),
                   error::kInvalidArgument)
        << "torch.cond branch output " << i
        << " type mismatch: true branch has " << TypeString(true_type)
        << ", false branch has " << TypeString(false_type);

    const auto ranked_type = llvm::dyn_cast<mlir::RankedTensorType>(true_type);
    TT_CHECK_THROW(ranked_type, error::kInvalidArgument)
        << "torch.cond branch output " << i << " must be a ranked tensor, got "
        << TypeString(true_type);
    ValidateDummyOutputType(dummy_outputs[i], ranked_type, i);
  }
}

absl::StatusOr<mlir::Value> BuildScalarPredicate(mlir::MlirBuilder& builder,
                                                 mlir::MlirOp pred) {
  const mlir::RankedTensorType pred_type = GetTensorTypeOrDie(pred);
  TT_RET_CHECK(IsBooleanType(pred_type), error::kInvalidArgument)
      << "torch.cond predicate must be a bool tensor, got "
      << TypeString(pred_type);
  TT_RET_CHECK(pred_type.hasStaticShape(), error::kInvalidArgument)
      << "torch.cond predicate must have a static shape, got "
      << TypeString(pred_type);
  TT_RET_CHECK(pred_type.getNumElements() == 1, error::kInvalidArgument)
      << "torch.cond predicate must contain exactly one element, got "
      << TypeString(pred_type);

  if (pred_type.getRank() == 0) {
    return pred.getValue();
  }

  const mlir::RankedTensorType scalar_pred_type =
      mlir::RankedTensorType::get({}, pred_type.getElementType());
  return mlir::stablehlo::Reshape(scalar_pred_type, pred).getValue();
}

absl::Status PopulateBranchRegion(mlir::Region& region,
                                  mlir::OpBuilder& op_builder,
                                  const mlir::Location loc,
                                  mlir::func::FuncOp main_func,
                                  mlir::ValueRange operands) {
  mlir::Block* const block = op_builder.createBlock(&region);
  op_builder.setInsertionPointToStart(block);

  mlir::IRMapping mapper;
  for (const auto [arg, val] : llvm::zip(main_func.getArguments(), operands)) {
    mapper.map(arg, val);
  }

  mlir::Block& body_block = main_func.front();
  absl::c_for_each(
      llvm::make_range(body_block.begin(), std::prev(body_block.end())),
      [&op_builder, &mapper](mlir::Operation& op) {
        op_builder.clone(op, mapper);
      });

  mlir::Operation* const terminator = body_block.getTerminator();
  llvm::SmallVector<mlir::Value> returned;
  returned.reserve(terminator->getNumOperands());
  for (mlir::Value value : terminator->getOperands()) {
    returned.push_back(mapper.lookup(value));
  }
  mlir::stablehlo::ReturnOp::create(op_builder, loc, returned);
  return absl::OkStatus();
}

absl::StatusOr<DynamicMlirOpResults> BuildCondShlo(
    mlir::MlirBuilder& builder, mlir::MlirOp pred,
    absl::Span<mlir::MlirOp> operands, mlir::func::FuncOp true_main_func,
    mlir::func::FuncOp false_main_func) {
  TT_ASSIGN_OR_RETURN(mlir::Value scalar_pred,
                      BuildScalarPredicate(builder, pred));

  const mlir::TypeRange true_result_types = true_main_func.getResultTypes();
  const mlir::TypeRange false_result_types = false_main_func.getResultTypes();
  TT_RET_CHECK(true_result_types.size() == false_result_types.size(),
               error::kInvalidArgument)
      << "torch.cond branches must have the same number of result types, got "
      << true_result_types.size() << " and " << false_result_types.size();
  for (int64_t i = 0; i < true_result_types.size(); ++i) {
    const mlir::Type true_type = true_result_types[i];
    const mlir::Type false_type = false_result_types[i];
    TT_RET_CHECK(true_type == false_type, error::kInvalidArgument)
        << "torch.cond branch output " << i
        << " type mismatch after parsing: true branch has "
        << TypeString(true_type) << ", false branch has "
        << TypeString(false_type);
  }

  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::Location loc = pred.getValue().getLoc();
  mlir::OperationState state(loc, "stablehlo.if");
  state.addOperands(scalar_pred);
  state.addTypes(true_result_types);
  state.addRegion();
  state.addRegion();
  mlir::Operation* const if_op = op_builder.create(state);

  llvm::SmallVector<mlir::Value> operand_values;
  operand_values.reserve(operands.size());
  for (mlir::MlirOp operand : operands) {
    operand_values.push_back(operand.getValue());
  }

  TT_RETURN_IF_ERROR(PopulateBranchRegion(if_op->getRegion(0), op_builder, loc,
                                          true_main_func, operand_values));
  TT_RETURN_IF_ERROR(PopulateBranchRegion(if_op->getRegion(1), op_builder, loc,
                                          false_main_func, operand_values));

  op_builder.setInsertionPointAfter(if_op);
  DynamicMlirOpResults results;
  results.reserve(if_op->getNumResults());
  for (mlir::Value result : if_op->getResults()) {
    results.push_back(mlir::MlirOp(builder, result));
  }
  return results;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ParseModule(
    mlir::MlirBuilder& builder, const std::string& mlir_text,
    const std::string& branch_name) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      mlir_text, mlir::ParserConfig{&builder.getContext()});
  TT_RET_CHECK(module.get() != nullptr, error::kInternal)
      << "failed to parse " << branch_name << " branch MLIR module";
  return module;
}

}  // namespace

std::vector<at::Tensor> PyCreateCondOp(
    const at::Tensor& pred, const std::vector<at::Tensor>& operands,
    const std::shared_ptr<ContextedModule>& true_module,
    const std::shared_ptr<ContextedModule>& false_module,
    const std::vector<at::Tensor>& dummy_outputs) {
  ScopedPythonContextCapturer capturer(OpName::kCustomKernel);

  TT_CHECK_THROW(pred.scalar_type() == at::ScalarType::Bool,
                 error::kInvalidArgument)
      << "torch.cond predicate must be a bool tensor, got "
      << pred.scalar_type();
  TT_CHECK_THROW(pred.numel() == 1, error::kInvalidArgument)
      << "torch.cond predicate must contain exactly one element, got shape "
      << ToString(CopyIntVector(pred.sizes()));

  mlir::func::FuncOp true_main_func = GetMainFuncOrThrow(true_module, "true");
  mlir::func::FuncOp false_main_func =
      GetMainFuncOrThrow(false_module, "false");
  ValidateBranchFunctionsOrThrow(true_main_func, false_main_func,
                                 operands.size(), dummy_outputs);

  const BranchModules serialized{
      .true_mlir = SerializeModule(*true_module),
      .false_mlir = SerializeModule(*false_module),
  };
  const int64_t num_operands = operands.size();

  MlirOpBuilder op_builder = [serialized, num_operands](
                                 mlir::MlirBuilder& builder,
                                 absl::Span<mlir::MlirOp> op_inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    TT_RET_CHECK(op_inputs.size() == 1 + num_operands, error::kInvalidArgument)
        << "expected " << 1 + num_operands << " inputs, got "
        << op_inputs.size();

    TT_ASSIGN_OR_RETURN(auto true_module_op,
                        ParseModule(builder, serialized.true_mlir, "true"));
    TT_ASSIGN_OR_RETURN(auto false_module_op,
                        ParseModule(builder, serialized.false_mlir, "false"));

    auto true_main_func =
        true_module_op->lookupSymbol<mlir::func::FuncOp>("main");
    auto false_main_func =
        false_module_op->lookupSymbol<mlir::func::FuncOp>("main");
    TT_RET_CHECK(true_main_func != nullptr, error::kInternal)
        << "expected 'main' function in true branch MLIR module";
    TT_RET_CHECK(false_main_func != nullptr, error::kInternal)
        << "expected 'main' function in false branch MLIR module";

    mlir::MlirOp pred = op_inputs.front();
    return BuildCondShlo(builder, pred, op_inputs.subspan(1), true_main_func,
                         false_main_func);
  };

  TT_ASSIGN_OR_THROW(auto cache_keys,
                     *(OpParamCacheKeysBuilder()
                           .SetParam("true_branch_mlir", serialized.true_mlir)
                           .SetParam("false_branch_mlir", serialized.false_mlir)
                           .SetParam("num_operands", num_operands)
                           .SetParam("num_outputs", dummy_outputs.size())));

  std::vector<DeviceBufferRef> input_buffers = GetInputBuffers(pred, operands);
  std::vector<Shape> output_shapes = GetOutputShapes(dummy_outputs);
  TT_ASSIGN_OR_THROW(const auto result_refs,
                     DeviceBufferList::CreateDeferred(
                         OpName::kCustomKernel, std::move(op_builder),
                         std::move(input_buffers), std::move(cache_keys),
                         std::move(output_shapes)));

  std::vector<at::Tensor> results;
  results.reserve(result_refs.size());
  absl::c_transform(result_refs, std::back_inserter(results),
                    [](const DeviceBufferRef& ref) { return MakeTensor(ref); });
  return results;
}

}  // namespace torch_tpu
