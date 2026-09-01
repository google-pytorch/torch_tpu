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

#include "csrc/ops/experimental/sparse_dense_matmul/sparse_dense_matmul_grad_with_adagrad_aten_kernels.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "xla/side_effect_util.h"

using torch_tpu::is_status_or_ref;

namespace torch_tpu {

namespace {

auto SparseDenseMatmulGradWithAdagradBuilder(
    int64_t device_batch_size, int64_t max_ids_per_partition,
    int64_t max_unique_ids_per_partition, std::string_view computation_name,
    double epsilon) {
  return [max_ids_per_partition, max_unique_ids_per_partition,
          computation_name = std::string(computation_name),
          epsilon](torch_tpu::FixedSizeSpan<mlir::MlirOp, 8> inputs)
             -> absl::StatusOr<torch_tpu::MlirOpResults<2>> {
    mlir::MlirOp row_pointers = inputs[0];
    mlir::MlirOp embedding_ids = inputs[1];
    mlir::MlirOp sample_ids = inputs[2];
    mlir::MlirOp gains = inputs[3];
    mlir::MlirOp embedding_table = inputs[4];
    mlir::MlirOp accumulator = inputs[5];
    mlir::MlirOp activations_grad = inputs[6];
    mlir::MlirOp learning_rate = inputs[7];

    mlir::MlirBuilder& builder = row_pointers.getBuilder();
    mlir::OpBuilder& op_builder = builder.getOpBuilder();

    // Create compile-time constant for hyperparameter epsilon.
    auto float_type = op_builder.getF32Type();
    auto scalar_type = mlir::RankedTensorType::get({}, float_type);
    auto epsilon_const =
        MakeConstant(builder, static_cast<float>(epsilon), scalar_type);

    auto embedding_table_type = torch_tpu::GetTensorTypeOrDie(embedding_table);
    auto accumulator_type = torch_tpu::GetTensorTypeOrDie(accumulator);
    auto accumulator_shape = accumulator_type.getShape();
    int64_t embedding_dim = embedding_table_type.getShape()[1];

    bool is_rowwise = (accumulator_shape.size() == 1);
    if (!is_rowwise) {
      TT_RET_CHECK(accumulator_shape.size() == 2, error::kInvalidArgument)
          << "expected accumulator to be 1D (row-wise) or 2D (standard), got "
          << accumulator_shape.size() << "D";
      TT_RET_CHECK(accumulator_shape[1] == embedding_dim,
                   error::kInvalidArgument)
          << "expected accumulator dimension 1 to match embedding dimension ("
          << embedding_dim << "), got " << accumulator_shape[1];
    }

    // Define the optimizer update function.

    auto param_type = mlir::RankedTensorType::get({1, embedding_dim},
                                                  op_builder.getF32Type());
    mlir::RankedTensorType acc_type =
        is_rowwise ? mlir::RankedTensorType::get({1}, op_builder.getF32Type())
                   : mlir::RankedTensorType::get({1, embedding_dim},
                                                 op_builder.getF32Type());
    auto lr_type = param_type;
    auto epsilon_type = param_type;

    auto func_type = op_builder.getFunctionType(
        {param_type, param_type, acc_type, lr_type, epsilon_type},
        {mlir::TupleType::get(op_builder.getContext(),
                              {param_type, acc_type})});

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
      mlir::MlirOp acc_arg(builder, entry_block->getArgument(2));
      mlir::MlirOp lr(builder, entry_block->getArgument(3));
      mlir::MlirOp epsilon_arg(builder, entry_block->getArgument(4));

      auto grad_sq = mlir::stablehlo::Mul(grad, grad);
      mlir::MlirOp new_acc;
      if (is_rowwise) {
        auto zero = MakeScalarConstant(builder, 0.0f, mlir::ElementType::F32);
        auto dtype = torch_tpu::GetTensorTypeOrDie(grad_sq).getElementType();
        auto reduce_builder = [&](mlir::RegionBuilder& rb) {
          mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
              dtype, rb.getRegion(), rb.getOpBuilder());
        };
        auto dim_size = mlir::stablehlo::GetDimensionSize(grad, 1);
        auto dim_size_f32 = mlir::stablehlo::ConvertElementType(
            dim_size, op_builder.getF32Type());
        TT_ASSIGN_OR_RETURN(auto dim_bcast,
                            BroadcastIfNeeded(dim_size_f32, grad_sq));
        auto grad_sq_div = mlir::stablehlo::Div(grad_sq, dim_bcast);
        auto reduced = mlir::stablehlo::Reduce(builder, grad_sq_div, zero,
                                               reduce_builder, {1})[0];
        new_acc = mlir::stablehlo::Add(acc_arg, reduced);
      } else {
        new_acc = mlir::stablehlo::Add(acc_arg, grad_sq);
      }
      auto lr_grad = mlir::stablehlo::Mul(lr, grad);
      auto sqrt_acc = mlir::stablehlo::Sqrt(new_acc);
      mlir::MlirOp sqrt_acc_bcast = sqrt_acc;
      if (is_rowwise) {
        TT_ASSIGN_OR_RETURN(sqrt_acc_bcast,
                            BroadcastIfNeeded(sqrt_acc, epsilon_arg));
      }
      auto sqrt_acc_plus_epsilon =
          mlir::stablehlo::Add(sqrt_acc_bcast, epsilon_arg);
      auto update = mlir::stablehlo::Div(lr_grad, sqrt_acc_plus_epsilon);
      auto updated_param = mlir::stablehlo::Subtract(param, update);

      auto tuple_op = mlir::stablehlo::TupleOp::create(
          op_builder, builder.getLoc(),
          mlir::ValueRange{updated_param.getValue(), new_acc.getValue()});
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
        xla::kXlaShardingStrategyAttr,
        op_builder.getStringAttr(xla::kXlaShardingStrategyMod)));
    frontend_attrs.push_back(op_builder.getNamedAttr(
        xla::kXlaPadValueAttr, op_builder.getStringAttr("2147483647")));
    frontend_attrs.push_back(op_builder.getNamedAttr(
        xla::kXlaMaxIdsPerPartitionAttr,
        op_builder.getStringAttr(std::to_string(max_ids_per_partition))));
    frontend_attrs.push_back(op_builder.getNamedAttr(
        xla::kXlaMaxUniqueIdsPerPartitionAttr,
        op_builder.getStringAttr(
            std::to_string(max_unique_ids_per_partition))));
    frontend_attrs.push_back(op_builder.getNamedAttr(
        xla::kNumSlotVariables, op_builder.getStringAttr("1")));
    frontend_attrs.push_back(op_builder.getNamedAttr(
        xla::kNumHyperparameters, op_builder.getStringAttr("2")));

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

    auto frontend_attributes_attr =
        op_builder.getNamedAttr("mhlo.frontend_attributes",
                                op_builder.getDictionaryAttr(frontend_attrs));

    auto called_computations_attr = op_builder.getNamedAttr(
        "called_computations",
        op_builder.getArrayAttr({mlir::SymbolRefAttr::get(
            op_builder.getContext(), computation_name)}));

    std::vector<mlir::Value> operands = {
        row_pointers.getValue(),     embedding_ids.getValue(),
        sample_ids.getValue(),       gains.getValue(),
        activations_grad.getValue(), embedding_table.getValue(),
        accumulator.getValue(),      learning_rate.getValue(),
        epsilon_const.getValue()};

    auto out_type = mlir::TupleType::get(
        op_builder.getContext(), {embedding_table_type, accumulator_type});

    auto op = mlir::stablehlo::CustomCallOp::create(
        op_builder, builder.getLoc(),
        /*resultTypes=*/{out_type},
        /*operands=*/operands,
        {call_target_attr, has_side_effect_attr, api_version_attr,
         frontend_attributes_attr, called_computations_attr});

    auto gte_table = mlir::stablehlo::GetTupleElementOp::create(
        op_builder, builder.getLoc(), op.getResult(0), 0);
    auto gte_accumulator = mlir::stablehlo::GetTupleElementOp::create(
        op_builder, builder.getLoc(), op.getResult(0), 1);

    return torch_tpu::MlirOpResults<2>(
        {mlir::MlirOp(builder, gte_table.getResult()),
         mlir::MlirOp(builder, gte_accumulator.getResult())});
  };
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> AtenSparseDenseMatmulGradWithAdagrad(
    const at::Tensor& row_pointers, const at::Tensor& embedding_ids,
    const at::Tensor& sample_ids, const at::Tensor& gains,
    const at::Tensor& embedding_table, const at::Tensor& accumulator,
    const at::Tensor& activations_grad, const at::Tensor& learning_rate,
    double epsilon, int64_t device_batch_size, int64_t max_ids_per_partition,
    int64_t max_unique_ids_per_partition, std::string_view computation_name) {
  TT_KERNEL(
      torch_tpu::OpName::kSparseDenseMatmulGradWithAdagrad, param_keys,
      (row_pointers, embedding_ids, sample_ids, gains, embedding_table,
       accumulator, activations_grad, learning_rate, epsilon, device_batch_size,
       max_ids_per_partition, max_unique_ids_per_partition, computation_name),
      {
        std::array<torch_tpu::TensorHolder, 8> inputs = {
            row_pointers,    embedding_ids, sample_ids,       gains,
            embedding_table, accumulator,   activations_grad, learning_rate};

        torch_tpu::Dimensions out_dims(embedding_table.sizes().begin(),
                                       embedding_table.sizes().end());
        torch_tpu::Dimensions acc_dims(accumulator.sizes().begin(),
                                       accumulator.sizes().end());

        auto builder_fn = SparseDenseMatmulGradWithAdagradBuilder(
            device_batch_size, max_ids_per_partition,
            max_unique_ids_per_partition, computation_name, epsilon);

        TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                           torch_tpu::ConvertTo<mlir::ElementType>(
                               embedding_table.scalar_type()));
        TT_ASSIGN_OR_THROW(
            mlir::ElementType acc_dtype,
            torch_tpu::ConvertTo<mlir::ElementType>(accumulator.scalar_type()));

        TT_ASSIGN_OR_THROW(auto results,
                           (torch_tpu::DispatchOp<8, 2>(
                               builder_fn, inputs,
                               {.out_dtypes = {out_dtype, acc_dtype},
                                .out_dims_list = {out_dims, acc_dims},
                                .op_param_cache_keys = std::move(param_keys),
                                .donated_indices = {4, 5}})));

        return std::make_tuple(torch_tpu::MakeTensor(std::move(results[0])),
                               torch_tpu::MakeTensor(std::move(results[1])));
      });
}

}  // namespace torch_tpu
