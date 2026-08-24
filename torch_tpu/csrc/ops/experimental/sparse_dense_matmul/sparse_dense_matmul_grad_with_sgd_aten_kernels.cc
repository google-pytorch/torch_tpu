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

#include "torch_tpu/csrc/ops/experimental/sparse_dense_matmul/sparse_dense_matmul_grad_with_sgd_aten_kernels.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/csrc/common/dimension_types.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/fixed_size_span.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "xla/side_effect_util.h"

namespace torch_tpu {

namespace {

auto SparseDenseMatmulGradWithSgdBuilder(int64_t device_batch_size,
                                         int64_t max_ids_per_partition,
                                         int64_t max_unique_ids_per_partition,
                                         std::string_view computation_name) {
  return [max_ids_per_partition, max_unique_ids_per_partition,
          computation_name = std::string(computation_name)](
             torch_tpu::FixedSizeSpan<mlir::MlirOp, 7> inputs)
             -> absl::StatusOr<mlir::MlirOp> {
    mlir::MlirOp row_pointers = inputs[0];
    mlir::MlirOp embedding_ids = inputs[1];
    mlir::MlirOp sample_ids = inputs[2];
    mlir::MlirOp gains = inputs[3];
    mlir::MlirOp embedding_table = inputs[4];
    mlir::MlirOp activations_grad = inputs[5];
    mlir::MlirOp learning_rate = inputs[6];

    mlir::MlirBuilder& builder = row_pointers.getBuilder();
    mlir::OpBuilder& op_builder = builder.getOpBuilder();

    auto embedding_table_type = torch_tpu::GetTensorTypeOrDie(embedding_table);
    int64_t embedding_dim = embedding_table_type.getShape()[1];

    // Define the optimizer update function.

    auto tensor_type = mlir::RankedTensorType::get({1, embedding_dim},
                                                   op_builder.getF32Type());
    auto func_type = op_builder.getFunctionType(
        {tensor_type, tensor_type, tensor_type},
        {mlir::TupleType::get(op_builder.getContext(), {tensor_type})});

    mlir::ModuleOp module = torch_tpu::GetModuleOp(builder);
    if (!module.lookupSymbol<mlir::func::FuncOp>(computation_name)) {
      // Save current insertion point
      auto prev_insertion_point = op_builder.saveInsertionPoint();

      op_builder.setInsertionPointToEnd(module.getBody());
      auto func = mlir::func::FuncOp::create(op_builder, builder.getLoc(),
                                             computation_name, func_type);
      func.setVisibility(mlir::SymbolTable::Visibility::Private);

      auto* entry_block = func.addEntryBlock();
      op_builder.setInsertionPointToStart(entry_block);

      mlir::MlirOp grad(builder, entry_block->getArgument(0));
      mlir::MlirOp param(builder, entry_block->getArgument(1));
      mlir::MlirOp lr(builder, entry_block->getArgument(2));

      // lr * grad
      auto lr_grad = mlir::stablehlo::Mul(lr, grad);
      // updated_param = param - lr * grad
      auto updated_param = mlir::stablehlo::Subtract(param, lr_grad);

      auto tuple_op = mlir::stablehlo::TupleOp::create(
          op_builder, builder.getLoc(),
          mlir::ValueRange{updated_param.getValue()});
      mlir::func::ReturnOp::create(op_builder, builder.getLoc(),
                                   tuple_op.getOperation()->getResults());

      // Restore insertion point
      op_builder.restoreInsertionPoint(prev_insertion_point);
    }

    // Construct DictionaryAttr for frontend_attributes
    std::vector<mlir::NamedAttribute> frontend_attrs;
    frontend_attrs.reserve(7);
    frontend_attrs.push_back(op_builder.getNamedAttr(
        xla::kXlaComputeTypeAttr,
        op_builder.getStringAttr(xla::kXlaComputeTypeSparse)));
    frontend_attrs.push_back(op_builder.getNamedAttr(
        xla::kXlaMaxIdsPerPartitionAttr,
        op_builder.getStringAttr(std::to_string(max_ids_per_partition))));
    frontend_attrs.push_back(op_builder.getNamedAttr(
        xla::kXlaMaxUniqueIdsPerPartitionAttr,
        op_builder.getStringAttr(
            std::to_string(max_unique_ids_per_partition))));
    frontend_attrs.push_back(op_builder.getNamedAttr(
        xla::kXlaPadValueAttr, op_builder.getStringAttr("2147483647")));
    frontend_attrs.push_back(op_builder.getNamedAttr(
        xla::kXlaShardingStrategyAttr,
        op_builder.getStringAttr(xla::kXlaShardingStrategyMod)));
    frontend_attrs.push_back(op_builder.getNamedAttr(
        xla::kNumSlotVariables, op_builder.getStringAttr("0")));
    frontend_attrs.push_back(op_builder.getNamedAttr(
        xla::kNumHyperparameters, op_builder.getStringAttr("1")));

    mlir::NamedAttribute frontend_attrs_attr =
        op_builder.getNamedAttr("mhlo.frontend_attributes",
                                op_builder.getDictionaryAttr(frontend_attrs));

    std::string call_target = "SparseDenseMatmulGradOpWithOptimizerUpdate";

    mlir::NamedAttribute call_target_attr = op_builder.getNamedAttr(
        "call_target_name", op_builder.getStringAttr(call_target));
    mlir::NamedAttribute has_side_effect_attr = op_builder.getNamedAttr(
        "has_side_effect", op_builder.getBoolAttr(false));
    auto api_version_attr = op_builder.getNamedAttr(
        "api_version",
        mlir::stablehlo::CustomCallApiVersionAttr::get(
            &builder.getContext(),
            mlir::stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL));

    // called_computations attr
    auto called_computations_attr = op_builder.getNamedAttr(
        "called_computations",
        op_builder.getArrayAttr({mlir::SymbolRefAttr::get(
            op_builder.getContext(), computation_name)}));

    // Operands
    std::vector<mlir::Value> operands = {
        row_pointers.getValue(),     embedding_ids.getValue(),
        sample_ids.getValue(),       gains.getValue(),
        activations_grad.getValue(), embedding_table.getValue(),
        learning_rate.getValue()};

    auto out_type =
        mlir::TupleType::get(op_builder.getContext(), {embedding_table_type});

    auto op = mlir::stablehlo::CustomCallOp::create(
        op_builder, builder.getLoc(),
        /*resultTypes=*/{out_type},
        /*operands=*/operands,
        {call_target_attr, has_side_effect_attr, api_version_attr,
         called_computations_attr, frontend_attrs_attr});

    auto gte = mlir::stablehlo::GetTupleElementOp::create(
        op_builder, builder.getLoc(), op.getResult(0), 0);

    return mlir::MlirOp(builder, gte.getResult());
  };
}

}  // namespace

at::Tensor AtenSparseDenseMatmulGradWithSgd(
    const at::Tensor& row_pointers, const at::Tensor& embedding_ids,
    const at::Tensor& sample_ids, const at::Tensor& gains,
    const at::Tensor& embedding_table, const at::Tensor& activations_grad,
    const at::Tensor& learning_rate, int64_t device_batch_size,
    int64_t max_ids_per_partition, int64_t max_unique_ids_per_partition,
    std::string_view computation_name) {
  TT_KERNEL(
      torch_tpu::OpName::kSparseDenseMatmulGradWithSgd, param_keys,
      (row_pointers, embedding_ids, sample_ids, gains, embedding_table,
       activations_grad, learning_rate, device_batch_size,
       max_ids_per_partition, max_unique_ids_per_partition, computation_name),
      {
        std::array<torch_tpu::TensorHolder, 7> inputs = {
            row_pointers,    embedding_ids,    sample_ids,   gains,
            embedding_table, activations_grad, learning_rate};

        torch_tpu::Dimensions out_dims(embedding_table.sizes().begin(),
                                       embedding_table.sizes().end());

        auto builder_fn = SparseDenseMatmulGradWithSgdBuilder(
            device_batch_size, max_ids_per_partition,
            max_unique_ids_per_partition, computation_name);

        TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                           torch_tpu::ConvertTo<mlir::ElementType>(
                               embedding_table.scalar_type()));

        TT_ASSIGN_OR_THROW(
            auto results, (torch_tpu::DispatchOp<7, 1>(
                              builder_fn, inputs,
                              {.out_dtype = out_dtype,
                               .out_dims = out_dims,
                               .op_param_cache_keys = std::move(param_keys)})));

        return torch_tpu::MakeTensor(results);
      });
}

}  // namespace torch_tpu
