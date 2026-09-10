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

#include "csrc/internal/compile/dispatch_scan.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "csrc/ops/scan_builder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

namespace {

std::vector<DeviceBufferRef> GetInputBuffers(
    const std::vector<at::Tensor>& inits,
    const std::vector<at::Tensor>& inputs) {
  std::vector<DeviceBufferRef> input_buffers;
  input_buffers.reserve(inits.size() + inputs.size());

  auto get_buffer = [](const at::Tensor& t) -> DeviceBufferRef {
    TT_ASSIGN_OR_THROW(const DeviceBufferRef ref, GetBuffer(t));
    return ref;
  };

  absl::c_transform(inits, std::back_inserter(input_buffers), get_buffer);
  absl::c_transform(inputs, std::back_inserter(input_buffers), get_buffer);

  return input_buffers;
}

std::vector<Shape> GetOutputShapes(
    const std::vector<at::Tensor>& inits,
    const std::vector<at::Tensor>& dummy_outputs) {
  std::vector<Shape> output_shapes;
  output_shapes.reserve(inits.size() + dummy_outputs.size());

  auto get_init_shape = [](const at::Tensor& t) -> Shape {
    TT_ASSIGN_OR_THROW(const DeviceBufferRef ref, GetBuffer(t));
    return ref.shape();
  };

  auto get_outputs_shape = [](const at::Tensor& t) -> Shape {
    TT_ASSIGN_OR_THROW(const auto dtype,
                       ConvertTo<mlir::ElementType>(t.scalar_type()));
    return Shape(CopyIntVector(t.sizes()), dtype);
  };

  absl::c_transform(inits, std::back_inserter(output_shapes), get_init_shape);
  absl::c_transform(dummy_outputs, std::back_inserter(output_shapes),
                    get_outputs_shape);

  return output_shapes;
}

llvm::SmallVector<mlir::MlirOp> InitZeroOutputAccumulators(
    mlir::MlirBuilder& builder, mlir::func::FuncOp main_func,
    const int64_t num_carries, const int64_t seq_len) {
  const mlir::TypeRange output_types =
      main_func.getResultTypes().drop_front(num_carries);
  llvm::SmallVector<mlir::MlirOp> output_inits;
  output_inits.reserve(output_types.size());

  auto create_init = [&builder, seq_len](const mlir::Type t) -> mlir::MlirOp {
    const mlir::RankedTensorType output_type =
        llvm::cast<mlir::RankedTensorType>(t);
    llvm::SmallVector<int64_t> outputs_shape = {seq_len};
    llvm::append_range(outputs_shape, output_type.getShape());
    return MakeConstant(builder, 0, output_type.getElementType(),
                        outputs_shape);
  };

  absl::c_transform(output_types, std::back_inserter(output_inits),
                    create_init);
  return output_inits;
}

// Populates the StableHLO while-loop body by cloning the pre-lowered step
// function (main_func) operations. It maps the formal step arguments to the
// active loop carries and input slices, and returns the updated loop carries
// and iteration outputs (from the loop's terminator).
absl::StatusOr<ScanBodyResults> BuildLoopBody(mlir::OpBuilder& op_builder,
                                              mlir::func::FuncOp main_func,
                                              const int64_t num_carries,
                                              const int64_t total_inputs,
                                              mlir::ValueRange slices,
                                              mlir::ValueRange carries) {
  mlir::IRMapping mapper;
  auto map_args = [&mapper](const mlir::ValueRange args,
                            const mlir::ValueRange values) {
    for (const auto [arg, val] : llvm::zip(args, values)) {
      mapper.map(arg, val);
    }
  };

  const mlir::Block::BlockArgListType func_args = main_func.getArguments();
  map_args(func_args.take_front(num_carries), carries);
  map_args(func_args.slice(num_carries, total_inputs), slices);

  // Clone all operations except the terminator. We must skip the function's
  // return terminator because loop body regions in StableHLO require a loop-
  // specific yield terminator (stablehlo.return) instead of a function return.
  // The return values are instead manually extracted and mapped below.
  mlir::Block& body_block = main_func.front();
  absl::c_for_each(
      llvm::make_range(body_block.begin(), std::prev(body_block.end())),
      [&op_builder, &mapper](mlir::Operation& op) {
        op_builder.clone(op, mapper);
      });

  auto lookup_values = [&mapper](const mlir::ValueRange range) {
    llvm::SmallVector<mlir::Value> result;
    result.reserve(range.size());
    absl::c_transform(
        range, std::back_inserter(result),
        [&mapper](const mlir::Value v) { return mapper.lookup(v); });
    return result;
  };

  mlir::Operation* const terminator = body_block.getTerminator();
  const mlir::OperandRange operands = terminator->getOperands();
  return ScanBodyResults{lookup_values(operands.take_front(num_carries)),
                         lookup_values(operands.drop_front(num_carries))};
}

}  // namespace

std::vector<at::Tensor> PyCreateScanOp(
    const std::vector<at::Tensor>& inits, const std::vector<at::Tensor>& inputs,
    const std::shared_ptr<ContextedModule>& body_module,
    ScanDirection scan_direction, const std::vector<at::Tensor>& dummy_outputs,
    int64_t num_scan_inputs) {
  ScopedPythonContextCapturer capturer(OpName::kCustomKernel);

  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      !inputs.empty(), error::kInvalidArgument)
      << "expected at least 1 input tensor";

  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      num_scan_inputs > 0, error::kInvalidArgument)
      << "expected num_scan_inputs to be greater than 0, got "
      << num_scan_inputs;
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      num_scan_inputs <= inputs.size(), error::kInvalidArgument)
      << "expected num_scan_inputs (" << num_scan_inputs
      << ") to be less than or equal to the number of inputs (" << inputs.size()
      << ")";

  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      body_module != nullptr && body_module->get(), error::kInvalidArgument)
      << "invalid body module";

  auto body_main_func =
      body_module->get().lookupSymbol<mlir::func::FuncOp>("main");
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      body_main_func != nullptr, error::kInternal)
      << "expected 'main' function in body MLIR module";

  const int64_t num_carries = inits.size();
  const int64_t num_results = body_main_func.getResultTypes().size();
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      num_results >= num_carries, error::kInvalidArgument)
      << "expected step function to have at least " << num_carries
      << " result types (corresponding to carries), but it only has "
      << num_results;

  const int64_t num_outputs = num_results - num_carries;
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      num_outputs == dummy_outputs.size(), error::kInvalidArgument)
      << "expected " << num_outputs << " dummy outputs, got "
      << dummy_outputs.size();

  // Serialize the body MLIR module to a string to act as a cache key and to
  // pass across different `MLIRContext`s.
  std::string serialized_mlir;
  {
    llvm::raw_string_ostream os(serialized_mlir);
    body_module->get().print(os);
    os.flush();
  }

  const int64_t total_inputs = inputs.size();

  MlirOpBuilder op_builder = [serialized_mlir, num_carries, total_inputs,
                              scan_direction, num_scan_inputs](
                                 mlir::MlirBuilder& builder,
                                 absl::Span<mlir::MlirOp> op_inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
        op_inputs.size() == num_carries + total_inputs, error::kInvalidArgument)
        << "expected " << num_carries + total_inputs << " inputs, got "
        << op_inputs.size();

    // Parse the captured serialized MLIR module into the lowerer's context.
    auto body_module_op = mlir::parseSourceString<mlir::ModuleOp>(
        serialized_mlir, mlir::ParserConfig{&builder.getContext()});
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
        body_module_op.get() != nullptr, error::kInternal)
        << "failed to parse body MLIR module";

    auto main_func = body_module_op->lookupSymbol<mlir::func::FuncOp>("main");

    // Split inputs into carries and scanned inputs.
    const llvm::ArrayRef<mlir::MlirOp> inputs_ref(op_inputs.data(),
                                                  op_inputs.size());
    const auto carry_inits = inputs_ref.take_front(num_carries);
    const auto scanned_inputs = inputs_ref.drop_front(num_carries);

    const mlir::RankedTensorType scan_inputs0_type =
        GetTensorTypeOrDie(scanned_inputs[0]);
    const int64_t seq_len = scan_inputs0_type.getDimSize(0);
    const llvm::SmallVector<mlir::MlirOp> output_inits =
        InitZeroOutputAccumulators(builder, main_func, num_carries, seq_len);

    // Build the while-loop body.
    auto body_builder =
        [main_func, num_carries, total_inputs](
            mlir::OpBuilder& op_builder, mlir::Location loc,
            mlir::ValueRange slices, mlir::Value index,
            mlir::ValueRange carries) -> absl::StatusOr<ScanBodyResults> {
      return BuildLoopBody(op_builder, main_func, num_carries, total_inputs,
                           slices, carries);
    };

    // PyTorch's frontend dispatcher always transposes the sequence dimension to
    // index 0. Hence, we unconditionally set dim=0 here.
    //
    // Further, should_squeeze=true because the slice returned from scanning
    // along the sequence dimension has shape [1, features...] but the step
    // function expects [features...]).
    return BuildScanShlo(
        builder, scanned_inputs, /*dim=*/0, num_scan_inputs, carry_inits,
        output_inits, std::move(body_builder),
        ScanOptions{.direction = scan_direction, .should_squeeze = true});
  };

  TT_ASSIGN_OR_THROW(auto cache_keys,
                     *(OpParamCacheKeysBuilder()
                           .SetParam("body_mlir", serialized_mlir)
                           .SetParam("scan_direction", scan_direction)
                           .SetParam("num_scan_inputs", num_scan_inputs)
                           .SetParam("num_carries", num_carries)));

  std::vector<DeviceBufferRef> input_buffers = GetInputBuffers(inits, inputs);
  std::vector<Shape> output_shapes = GetOutputShapes(inits, dummy_outputs);
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
