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

// This file implements the ATen kernel for `torch.ops.tpu._sparse_map_row`,
// which maps a user-defined row-wise computation over one or more sparse
// matrices in Compressed Sparse Row (CSR) format on TPU SparseCore.
//
// SparseMapRow lowers to a `stablehlo.custom_call` targeting the SparseCore
// hardware accelerator. The per-row computation is provided as a serialized
// MLIR module, which is cloned and embedded into the compilation module's
// `called_computations`.

#include "csrc/ops/experimental/sparse_map_row/sparse_map_row_aten_kernels.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "c10/util/ArrayRef.h"  // IWYU pragma: keep for at::IntArrayRef
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/pjrt/pjrt_utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace {

// Groups CSR tensor operands into StableHLO tuples for the SparseMapRow
// custom call.
//
// In CSR representation, each sparse matrix is represented by a group of
// tensors (e.g., row pointers, column indices, values). The XLA SparseCore
// custom call interface expects each CSR matrix as a single tuple operand,
// followed by any trailing dense arguments as individual operands.
llvm::SmallVector<mlir::Value> PackCustomCallOperands(
    mlir::OpBuilder& mlir_op_builder, mlir::Location loc,
    absl::Span<mlir::MlirOp> op_inputs, absl::Span<const int64_t> csr_lengths,
    int64_t total_csr_tensors) {
  llvm::SmallVector<mlir::Value> operands;
  int64_t csr_offset = 0;
  // Pack each CSR matrix's constituent tensors into a stablehlo::TupleOp.
  for (const int64_t len : csr_lengths) {
    llvm::SmallVector<mlir::Value> tuple_elements;
    tuple_elements.reserve(len);
    for (int64_t j = 0; j < len; ++j) {
      tuple_elements.push_back(op_inputs[csr_offset + j].getValue());
    }
    auto tuple_op =
        mlir::stablehlo::TupleOp::create(mlir_op_builder, loc, tuple_elements);
    operands.push_back(tuple_op.getResult());
    csr_offset += len;
  }
  // Append any dense arguments as individual operands.
  for (size_t i = total_csr_tensors; i < op_inputs.size(); ++i) {
    operands.push_back(op_inputs[i].getValue());
  }
  return operands;
}

// Parses the serialized MLIR module containing the per-row computation,
// assigns unique symbol names to avoid collisions in the parent module,
// clones all functions into the parent module, and ensures the entry function
// returns a single tuple type as expected by the SparseCore custom call.
absl::StatusOr<mlir::func::FuncOp> InjectClonedBodyFunction(
    mlir::MlirBuilder& builder, std::string_view serialized_mlir) {
  // Parse the serialized MLIR string containing the row-wise body module.
  auto body_module_op = mlir::parseSourceString<mlir::ModuleOp>(
      serialized_mlir, mlir::ParserConfig{&builder.getContext()});
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      body_module_op.get() != nullptr, error::kInternal)
      << "failed to parse body MLIR module";

  // Look up the entry point function ("main") in the parsed body module.
  auto main_func = body_module_op->lookupSymbol<mlir::func::FuncOp>("main");
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      main_func != nullptr, error::kInternal)
      << "failed to find the 'main' function in the body MLIR module";

  mlir::ModuleOp parent_module = GetModuleOp(builder);
  // Generate a unique symbol name for the cloned entry function to avoid
  // name collisions when multiple sparse_map_row ops are in the same module.
  static std::atomic<int64_t> sparse_map_row_id = 0;
  const int64_t id = sparse_map_row_id.fetch_add(1);
  const std::string unique_func_name = absl::StrCat("sparse_map_row_body_", id);

  mlir::SymbolTable src_st(*body_module_op);
  mlir::SymbolTable dst_st(parent_module);

  // Rename the entry function in the source module.
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      mlir::succeeded(src_st.rename(main_func, unique_func_name)),
      error::kInternal)
      << "failed to rename body function";

  // Rename any helper functions to unique names relative to the destination
  // symbol table so they do not collide with existing functions.
  llvm::SmallVector<mlir::func::FuncOp> funcs_to_clone;
  for (auto func : body_module_op->getOps<mlir::func::FuncOp>()) {
    if (func != main_func) {
      TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
          mlir::succeeded(src_st.renameToUnique(func, &dst_st)),
          error::kInternal)
          << "failed to rename helper function";
    }
    funcs_to_clone.push_back(func);
  }

  // Clone all functions into the parent module with private visibility.
  mlir::IRRewriter module_rewriter(&builder.getContext());
  module_rewriter.setInsertionPointToStart(parent_module.getBody());
  mlir::func::FuncOp cloned_main_func;
  for (auto func : funcs_to_clone) {
    auto cloned_func =
        mlir::cast<mlir::func::FuncOp>(module_rewriter.clone(*func));
    cloned_func.setVisibility(mlir::SymbolTable::Visibility::Private);
    if (func == main_func) {
      cloned_main_func = cloned_func;
    }
  }
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
      cloned_main_func != nullptr, error::kInternal)
      << "failed to clone main function";

  // SparseCore requires called_computations functions to return a single
  // Tuple. If the body function returns individual tensors, pack them into
  // a stablehlo::TupleOp and update the terminator and function type.
  if (cloned_main_func.getResultTypes().size() != 1 ||
      !mlir::isa<mlir::TupleType>(cloned_main_func.getResultTypes().front())) {
    mlir::Operation* term = cloned_main_func.front().getTerminator();
    mlir::OpBuilder term_builder(term);
    auto tuple_op = mlir::stablehlo::TupleOp::create(
        term_builder, term->getLoc(), term->getOperands());
    term_builder.create<mlir::func::ReturnOp>(term->getLoc(),
                                              tuple_op.getResult());
    term->erase();
    cloned_main_func.setType(
        term_builder.getFunctionType(cloned_main_func.getArgumentTypes(),
                                     mlir::TypeRange{tuple_op.getType()}));
  }
  return cloned_main_func;
}

// Emits the `stablehlo.custom_call` targeting SparseMapRow on SparseCore.
//
// Encodes hardware execution parameters in JSON `backend_config`:
// - `csr_config`: row sparsity padding limits (`max_non_zeroes_per_row`,
//   `pad_value`, `max_pad_per_row`).
// - `sparse_map_row_config`: parallelism and latency hiding settings
//   (`is_output_parallel`, `result_row_pointer_op_index`, `wait_threshold`).
//
// Links the cloned body function via `called_computations`.
DynamicMlirOpResults EmitSparseMapRowCustomCall(
    mlir::MlirBuilder& builder, mlir::func::FuncOp cloned_main_func,
    llvm::ArrayRef<mlir::Value> operands, absl::Span<const Shape> output_shapes,
    int64_t max_non_zeroes_per_row, int64_t pad_value, int64_t max_pad_per_row,
    double wait_threshold, int64_t result_row_pointer_op_index) {
  mlir::OpBuilder& mlir_op_builder = builder.getOpBuilder();
  const mlir::Location loc = builder.getLoc();

  // Assemble the JSON backend config required by SparseCore lowering passes.
  const std::string backend_config = absl::StrFormat(
      R"({"device_type":"DEVICE_TYPE_SPARSECORE","csr_config":{"max_non_zeroes_per_row":%d,"pad_value":%d,"max_pad_per_row":%d},"sparse_map_row_config":{"is_output_parallel":true,"result_row_pointer_op_index":%d,"wait_threshold":%f}})",
      max_non_zeroes_per_row, pad_value, max_pad_per_row,
      result_row_pointer_op_index, wait_threshold);

  const mlir::NamedAttribute backend_config_attr = mlir_op_builder.getNamedAttr(
      "backend_config", mlir_op_builder.getStringAttr(backend_config));
  const mlir::NamedAttribute call_target_attr = mlir_op_builder.getNamedAttr(
      "call_target_name", mlir_op_builder.getStringAttr("SparseMapRow"));
  const mlir::NamedAttribute has_side_effect_attr =
      mlir_op_builder.getNamedAttr("has_side_effect",
                                   mlir_op_builder.getBoolAttr(false));
  const mlir::NamedAttribute api_version_attr = mlir_op_builder.getNamedAttr(
      "api_version",
      mlir::stablehlo::CustomCallApiVersionAttr::get(
          &builder.getContext(),
          mlir::stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL));
  const mlir::NamedAttribute called_computations_attr =
      mlir_op_builder.getNamedAttr(
          "called_computations",
          mlir_op_builder.getArrayAttr(
              {mlir::FlatSymbolRefAttr::get(cloned_main_func)}));

  // Build the result types matching the expected output tensor shapes and
  // dtypes.
  llvm::SmallVector<mlir::Type> result_types;
  result_types.reserve(output_shapes.size());
  for (const auto& shape : output_shapes) {
    result_types.push_back(mlir::makeTensorType(
        builder.getContext(),
        llvm::ArrayRef(shape.dimensions().data(), shape.dimensions().size()),
        shape.dtype()));
  }

  // Create the CustomCallOp targeting SparseMapRow.
  auto custom_call = mlir::stablehlo::CustomCallOp::create(
      mlir_op_builder, loc,
      /*resultTypes=*/result_types,
      /*operands=*/operands,
      {call_target_attr, has_side_effect_attr, api_version_attr,
       backend_config_attr, called_computations_attr});

  llvm::SmallVector<mlir::MlirOp> results;
  results.reserve(custom_call.getNumResults());
  for (auto res : custom_call.getResults()) {
    results.emplace_back(builder, res);
  }
  return results;
}

}  // namespace

// Validates input tensors, parameters, and device capabilities for
// sparse_map_row.
absl::Status ValidateSparseMapRowInputs(
    const std::vector<at::Tensor>& csr_tensors, at::IntArrayRef csr_lengths,
    const std::vector<at::Tensor>& dense_args, std::string_view body_mlir,
    const std::vector<at::Tensor>& dummy_outputs,
    int64_t max_non_zeroes_per_row, int64_t result_row_pointer_op_index) {
  // Ensure at least one CSR tensor and length entry are provided.
  TT_RET_CHECK(!csr_tensors.empty(), error::kInvalidArgument)
      << "expected at least 1 CSR tensor, got 0";
  TT_RET_CHECK(!csr_lengths.empty(), error::kInvalidArgument)
      << "expected at least 1 entry in csr_lengths, got 0";

  // Verify that the sum of csr_lengths equals the number of CSR tensors passed.
  const int64_t total_csr_tensors =
      std::accumulate(csr_lengths.begin(), csr_lengths.end(), int64_t{0});
  TT_RET_CHECK(total_csr_tensors == static_cast<int64_t>(csr_tensors.size()),
               error::kInvalidArgument)
      << "expected the sum of csr_lengths (" << total_csr_tensors
      << ") to equal the number of CSR tensors, got " << csr_tensors.size();

  // Validate that result_row_pointer_op_index indexes a valid input CSR matrix.
  TT_RET_CHECK(result_row_pointer_op_index >= 0 &&
                   result_row_pointer_op_index <
                       static_cast<int64_t>(csr_lengths.size()),
               error::kInvalidArgument)
      << "expected result_row_pointer_op_index to be between 0 and "
      << (csr_lengths.size() - 1) << ", got " << result_row_pointer_op_index;

  // Max non-zeroes per row must be positive to define valid CSR buffer bounds.
  TT_RET_CHECK(max_non_zeroes_per_row > 0, error::kInvalidArgument)
      << "expected max_non_zeroes_per_row to be greater than 0, got "
      << max_non_zeroes_per_row;

  // The serialized MLIR module containing the body computation cannot be empty.
  TT_RET_CHECK(!body_mlir.empty(), error::kInvalidArgument)
      << "expected a non-empty body MLIR module, got an empty string";

  // Dummy output tensors provide shapes and dtypes for output buffer
  // allocation.
  TT_RET_CHECK(!dummy_outputs.empty(), error::kInvalidArgument)
      << "expected at least 1 dummy output tensor, got 0";

  // SparseMapRow relies on SparseCore hardware to execute. If not tracing for
  // meta shape inference, ensure that the active TPU device supports
  // SparseCore.
  if (!dummy_outputs[0].is_meta()) {
    TT_RET_CHECK(TpuDeviceSupportsSparseCore(), error::kFailedPrecondition)
        << "sparse_map_row requires a TPU device with SparseCore support";
  }

  return absl::OkStatus();
}

// ATen kernel implementation for torch.ops.tpu._sparse_map_row.
std::vector<at::Tensor> AtenSparseMapRow(
    const std::vector<at::Tensor>& csr_tensors, at::IntArrayRef csr_lengths,
    const std::vector<at::Tensor>& dense_args, std::string_view body_mlir,
    const std::vector<at::Tensor>& dummy_outputs,
    int64_t max_non_zeroes_per_row, int64_t pad_value, int64_t max_pad_per_row,
    double wait_threshold, int64_t result_row_pointer_op_index) {
  TT_KERNEL(
      OpName::kSparseMapRow, param_keys,
      (csr_tensors, csr_lengths, dense_args, body_mlir, dummy_outputs,
       max_non_zeroes_per_row, pad_value, max_pad_per_row, wait_threshold,
       result_row_pointer_op_index),
      {
        // 1. Validate inputs, CSR tensor counts, and SparseCore device support.
        TT_THROW_IF_ERROR(ValidateSparseMapRowInputs(
            csr_tensors, csr_lengths, dense_args, body_mlir, dummy_outputs,
            max_non_zeroes_per_row, result_row_pointer_op_index));

        // 2. Flatten all inputs (CSR tensors followed by dense arguments)
        // for tracking through the dispatcher.
        std::vector<at::Tensor> all_inputs;
        all_inputs.reserve(csr_tensors.size() + dense_args.size());
        all_inputs.insert(all_inputs.end(), csr_tensors.begin(),
                          csr_tensors.end());
        all_inputs.insert(all_inputs.end(), dense_args.begin(),
                          dense_args.end());

        // 3. Extract output shapes, dtypes, and dimension spans from exemplar
        // dummy outputs for buffer pre-allocation.
        std::vector<Shape> output_shapes;
        std::vector<mlir::ElementType> out_dtypes;
        std::vector<SmallInt64Vector> out_dims_storage;
        std::vector<absl::Span<const int64_t>> out_dims_spans;

        output_shapes.reserve(dummy_outputs.size());
        out_dtypes.reserve(dummy_outputs.size());
        out_dims_storage.reserve(dummy_outputs.size());
        out_dims_spans.reserve(dummy_outputs.size());

        for (const auto& dummy : dummy_outputs) {
          TT_ASSIGN_OR_THROW(const auto dtype,
                             ConvertTo<mlir::ElementType>(dummy.scalar_type()));
          out_dtypes.push_back(dtype);
          out_dims_storage.push_back(CopyIntVector(dummy.sizes()));
          output_shapes.emplace_back(CopyIntVector(dummy.sizes()), dtype);
        }
        for (const auto& dims : out_dims_storage) {
          out_dims_spans.push_back(absl::MakeConstSpan(dims));
        }

        const int64_t total_csr_tensors = csr_tensors.size();
        const int64_t total_dense_args = dense_args.size();
        const SmallInt64Vector csr_lengths_vec = CopyIntVector(csr_lengths);

        // 4. Define the MLIR graph-builder callback invoked during tracing or
        // compilation.
        auto builder_fn = [body_mlir = std::string(body_mlir), csr_lengths_vec,
                           total_csr_tensors, total_dense_args, output_shapes,
                           max_non_zeroes_per_row, pad_value, max_pad_per_row,
                           wait_threshold, result_row_pointer_op_index](
                              absl::Span<mlir::MlirOp> op_inputs,
                              mlir::MlirBuilder& builder)
            -> absl::StatusOr<DynamicMlirOpResults> {
          TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal TorchTPU error.
              op_inputs.size() ==
                  static_cast<size_t>(total_csr_tensors + total_dense_args),
              error::kInvalidArgument)
              << "expected " << (total_csr_tensors + total_dense_args)
              << " inputs, got " << op_inputs.size();

          // Pack CSR tensors into tuples and gather operands.
          llvm::SmallVector<mlir::Value> operands = PackCustomCallOperands(
              builder.getOpBuilder(), builder.getLoc(), op_inputs,
              csr_lengths_vec, total_csr_tensors);

          // Parse and clone the body function into the compilation module.
          TT_ASSIGN_OR_RETURN(mlir::func::FuncOp cloned_main_func,
                              InjectClonedBodyFunction(builder, body_mlir));

          // Emit the StableHLO CustomCall op targeting SparseMapRow.
          return EmitSparseMapRowCustomCall(
              builder, cloned_main_func, operands, output_shapes,
              max_non_zeroes_per_row, pad_value, max_pad_per_row,
              wait_threshold, result_row_pointer_op_index);
        };

        const absl::Span<const mlir::ElementType> out_dtypes_span = out_dtypes;
        const absl::Span<const absl::Span<const int64_t>> out_dims_list_span =
            out_dims_spans;

        // 5. Dispatch op execution on TPU via DispatchOp, compiling or
        // retrieving the cached executable.
        TT_ASSIGN_OR_THROW(
            auto results, (DispatchOp<kDynamicSize, kDynamicSize>(
                              std::move(builder_fn), all_inputs,
                              {.out_dtypes = out_dtypes_span,
                               .out_dims_list = out_dims_list_span,
                               .op_param_cache_keys = std::move(param_keys)})));

        // 6. Wrap device buffers into ATen tensors.
        std::vector<at::Tensor> output_tensors;
        output_tensors.reserve(results.size());
        for (const auto& ref : results) {
          output_tensors.push_back(MakeTensor(ref));
        }
        return output_tensors;
      });
}

}  // namespace torch_tpu
