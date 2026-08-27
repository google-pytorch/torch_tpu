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

#include "torch_tpu/csrc/ops/experimental/sparse_dense_matmul/sparse_dense_matmul_grad_with_adam_aten_kernels.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <tuple>
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
#include "mlir/IR/Types.h"
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

using torch_tpu::is_status_or_ref;

namespace torch_tpu {

namespace {

auto SparseDenseMatmulGradWithAdamBuilder(int64_t device_batch_size,
                                          int64_t max_ids_per_partition,
                                          int64_t max_unique_ids_per_partition,
                                          double beta_1, double beta_2,
                                          double epsilon,
                                          std::string_view computation_name) {
  return [max_ids_per_partition, max_unique_ids_per_partition, beta_1, beta_2,
          epsilon, computation_name = std::string(computation_name)](
             FixedSizeSpan<mlir::MlirOp, 9> inputs)
             -> absl::StatusOr<MlirOpResults<3>> {
    mlir::MlirOp row_pointers = inputs[0];
    mlir::MlirOp embedding_ids = inputs[1];
    mlir::MlirOp sample_ids = inputs[2];
    mlir::MlirOp gains = inputs[3];
    mlir::MlirOp embedding_table = inputs[4];
    mlir::MlirOp momentum = inputs[5];
    mlir::MlirOp velocity = inputs[6];
    mlir::MlirOp activations_grad = inputs[7];
    mlir::MlirOp alpha_t = inputs[8];

    mlir::MlirBuilder& builder = row_pointers.getBuilder();
    mlir::OpBuilder& op_builder = builder.getOpBuilder();
    auto float_type = op_builder.getF32Type();

    // Hyperparameters passed from PyTorch are scalar on the outside.
    auto scalar_type = mlir::RankedTensorType::get({}, float_type);

    auto embedding_table_type = GetTensorTypeOrDie(embedding_table);
    auto momentum_type = GetTensorTypeOrDie(momentum);
    auto velocity_type = GetTensorTypeOrDie(velocity);
    int64_t embedding_dim = embedding_table_type.getShape()[1];

    bool is_partial_rowwise = (velocity_type.getShape().size() == 1);
    if (!is_partial_rowwise) {
      TT_RET_CHECK(velocity_type.getShape().size() == 2,
                   error::kInvalidArgument)
          << "expected velocity tensor to be 1D or 2D, got rank "
          << velocity_type.getShape().size();
      TT_RET_CHECK(velocity_type.getShape()[1] == embedding_dim,
                   error::kInvalidArgument)
          << "expected velocity tensor dimension 1 to be " << embedding_dim
          << ", got " << velocity_type.getShape()[1];
    }

    // Create hyperparameter constants. They are scalars conceptually,
    // and passed as operands. XLA SparseCore will broadcast them.
    auto one_minus_beta_1_const =
        MakeConstant(builder, 1.0f - static_cast<float>(beta_1), scalar_type);
    auto one_minus_beta_2_const =
        MakeConstant(builder, 1.0f - static_cast<float>(beta_2), scalar_type);
    auto epsilon_const =
        MakeConstant(builder, static_cast<float>(epsilon), scalar_type);

    // Inside the body, all hyperparameters will be broadcasted to tensor_type
    // which corresponds to `f32[1, embedding_dim]`.
    auto tensor_type =
        mlir::RankedTensorType::get({1, embedding_dim}, float_type);
    auto rowwise_type = mlir::RankedTensorType::get({1}, float_type);
    auto vel_type = is_partial_rowwise ? rowwise_type : tensor_type;

    auto func_type = op_builder.getFunctionType(
        {tensor_type, tensor_type, tensor_type, vel_type, tensor_type,
         tensor_type, tensor_type, tensor_type},
        {mlir::TupleType::get(op_builder.getContext(),
                              {tensor_type, tensor_type, vel_type})});

    mlir::ModuleOp module = GetModuleOp(builder);
    auto prev_insertion_point = op_builder.saveInsertionPoint();

    op_builder.setInsertionPointToEnd(module.getBody());
    auto func = mlir::func::FuncOp::create(op_builder, builder.getLoc(),
                                           computation_name, func_type);
    func.setVisibility(mlir::SymbolTable::Visibility::Private);

    auto* entry_block = func.addEntryBlock();
    op_builder.setInsertionPointToStart(entry_block);

    mlir::MlirOp grad(builder, entry_block->getArgument(0));   // tensor_type
    mlir::MlirOp param(builder, entry_block->getArgument(1));  // tensor_type
    mlir::MlirOp momentum_arg(builder,
                              entry_block->getArgument(2));  // tensor_type
    mlir::MlirOp velocity_arg(builder,
                              entry_block->getArgument(3));  // vel_type
    mlir::MlirOp alpha_t_arg(builder,
                             entry_block->getArgument(4));  // tensor_type
    mlir::MlirOp one_minus_beta_1_arg(
        builder, entry_block->getArgument(5));  // tensor_type
    mlir::MlirOp one_minus_beta_2_arg(
        builder,
        entry_block->getArgument(6));  // tensor_type
    mlir::MlirOp epsilon_arg(builder,
                             entry_block->getArgument(7));  // tensor_type

    // grad_square = grad * grad
    auto grad_sq = mlir::stablehlo::Mul(grad, grad);

    // grad - momentum
    auto grad_minus_momentum = mlir::stablehlo::Subtract(grad, momentum_arg);
    // (1 - beta_1) * (grad - momentum)
    auto momentum_delta =
        mlir::stablehlo::Mul(one_minus_beta_1_arg, grad_minus_momentum);
    // new_momentum = momentum + (1 - beta_1) * (grad - momentum)
    auto new_momentum = mlir::stablehlo::Add(momentum_arg, momentum_delta);

    mlir::MlirOp velocity_bcast = velocity_arg;
    if (is_partial_rowwise) {
      TT_ASSIGN_OR_RETURN(velocity_bcast,
                          BroadcastIfNeeded(velocity_arg, grad_sq));
    }

    // grad^2 - velocity
    auto grad_sq_minus_velocity =
        mlir::stablehlo::Subtract(grad_sq, velocity_bcast);
    // (1 - beta_2) * (grad^2 - velocity)
    auto velocity_delta =
        mlir::stablehlo::Mul(one_minus_beta_2_arg, grad_sq_minus_velocity);
    // new_velocity_2d = velocity + (1 - beta_2) * (grad^2 - velocity)
    auto new_velocity_2d = mlir::stablehlo::Add(velocity_bcast, velocity_delta);

    mlir::MlirOp new_velocity;
    if (is_partial_rowwise) {
      auto zero_arg = MakeScalarConstant(builder, 0.0f, mlir::ElementType::F32);
      auto dtype =
          torch_tpu::GetTensorTypeOrDie(new_velocity_2d).getElementType();
      auto reduce_builder = [&](mlir::RegionBuilder& rb) {
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
            dtype, rb.getRegion(), rb.getOpBuilder());
      };

      mlir::MlirOp dim_size = mlir::stablehlo::GetDimensionSize(grad, 1);
      mlir::MlirOp dim_size_f32 =
          mlir::stablehlo::ConvertElementType(dim_size, float_type);
      TT_ASSIGN_OR_RETURN(mlir::MlirOp dim_bcast,
                          BroadcastIfNeeded(dim_size_f32, new_velocity_2d));
      auto new_velocity_2d_div =
          mlir::stablehlo::Div(new_velocity_2d, dim_bcast);

      new_velocity = mlir::stablehlo::Reduce(builder, new_velocity_2d_div,
                                             zero_arg, reduce_builder, {1})[0];
    } else {
      new_velocity = new_velocity_2d;
    }

    // alpha_t * new_momentum
    auto lr_momentum = mlir::stablehlo::Mul(alpha_t_arg, new_momentum);

    // sqrt(new_velocity)
    auto sqrt_velocity = mlir::stablehlo::Sqrt(new_velocity);
    mlir::MlirOp sqrt_vel_bcast = sqrt_velocity;
    if (is_partial_rowwise) {
      // sqrt_velocity is [1], needs to be [1, E] to add with epsilon_arg later
      TT_ASSIGN_OR_RETURN(sqrt_vel_bcast,
                          BroadcastIfNeeded(sqrt_velocity, grad));
    }

    // sqrt(new_velocity) + epsilon
    auto denom = mlir::stablehlo::Add(sqrt_vel_bcast, epsilon_arg);
    // update = (alpha_t * new_momentum) / (sqrt(new_velocity) + epsilon)
    auto update = mlir::stablehlo::Div(lr_momentum, denom);
    // updated_param = param - update
    auto updated_param = mlir::stablehlo::Subtract(param, update);

    auto tuple_op = mlir::stablehlo::TupleOp::create(
        op_builder, builder.getLoc(),
        mlir::ValueRange{updated_param.getValue(), new_momentum.getValue(),
                         new_velocity.getValue()});
    mlir::func::ReturnOp::create(op_builder, builder.getLoc(),
                                 tuple_op.getOperation()->getResults());

    op_builder.restoreInsertionPoint(prev_insertion_point);

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
        xla::kNumSlotVariables, op_builder.getStringAttr("2")));

    // alpha_t, one_minus_beta_1, one_minus_beta_2, epsilon -> 4
    // hyperparameters
    frontend_attrs.push_back(op_builder.getNamedAttr(
        xla::kNumHyperparameters, op_builder.getStringAttr("4")));

    std::string_view call_target = "SparseDenseMatmulGradOpWithOptimizerUpdate";

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

    std::vector<mlir::Value> operands = {row_pointers.getValue(),
                                         embedding_ids.getValue(),
                                         sample_ids.getValue(),
                                         gains.getValue(),
                                         activations_grad.getValue(),
                                         embedding_table.getValue(),
                                         momentum.getValue(),
                                         velocity.getValue(),
                                         alpha_t.getValue(),
                                         one_minus_beta_1_const.getValue(),
                                         one_minus_beta_2_const.getValue(),
                                         epsilon_const.getValue()};

    auto out_type = mlir::TupleType::get(
        op_builder.getContext(),
        {embedding_table_type, momentum_type, velocity_type});

    auto op = mlir::stablehlo::CustomCallOp::create(
        op_builder, builder.getLoc(),
        /*resultTypes=*/{out_type},
        /*operands=*/operands,
        {call_target_attr, has_side_effect_attr, api_version_attr,
         frontend_attributes_attr, called_computations_attr});

    auto gte_table = mlir::stablehlo::GetTupleElementOp::create(
        op_builder, builder.getLoc(), op.getResult(0), 0);
    auto gte_momentum = mlir::stablehlo::GetTupleElementOp::create(
        op_builder, builder.getLoc(), op.getResult(0), 1);
    auto gte_velocity = mlir::stablehlo::GetTupleElementOp::create(
        op_builder, builder.getLoc(), op.getResult(0), 2);

    return MlirOpResults<3>({mlir::MlirOp(builder, gte_table.getResult()),
                             mlir::MlirOp(builder, gte_momentum.getResult()),
                             mlir::MlirOp(builder, gte_velocity.getResult())});
  };
}
}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor>
AtenSparseDenseMatmulGradWithAdam(
    const at::Tensor& row_pointers, const at::Tensor& embedding_ids,
    const at::Tensor& sample_ids, const at::Tensor& gains,
    const at::Tensor& embedding_table, const at::Tensor& momentum,
    const at::Tensor& velocity, const at::Tensor& activations_grad,
    const at::Tensor& alpha_t, double beta_1, double beta_2, double epsilon,
    int64_t device_batch_size, int64_t max_ids_per_partition,
    int64_t max_unique_ids_per_partition, std::string_view computation_name) {
  TT_KERNEL(
      OpName::kSparseDenseMatmulGradWithAdam, param_keys,
      (row_pointers, embedding_ids, sample_ids, gains, embedding_table,
       momentum, velocity, activations_grad, alpha_t, beta_1, beta_2, epsilon,
       device_batch_size, max_ids_per_partition, max_unique_ids_per_partition,
       computation_name),
      {
        std::array<TensorHolder, 9> inputs = {
            row_pointers, embedding_ids,    sample_ids,
            gains,        embedding_table,  momentum,
            velocity,     activations_grad, alpha_t};

        Dimensions out_dims(embedding_table.sizes().begin(),
                            embedding_table.sizes().end());
        Dimensions vel_dims(velocity.sizes().begin(), velocity.sizes().end());

        auto builder_fn = SparseDenseMatmulGradWithAdamBuilder(
            device_batch_size, max_ids_per_partition,
            max_unique_ids_per_partition, beta_1, beta_2, epsilon,
            computation_name);

        TT_ASSIGN_OR_THROW(
            mlir::ElementType out_dtype,
            ConvertTo<mlir::ElementType>(embedding_table.scalar_type()));
        TT_ASSIGN_OR_THROW(
            mlir::ElementType vel_dtype,
            ConvertTo<mlir::ElementType>(velocity.scalar_type()));

        TT_ASSIGN_OR_THROW(
            auto results,
            (DispatchOp<9, 3>(builder_fn, inputs,
                              {.out_dtypes = {out_dtype, out_dtype, vel_dtype},
                               .out_dims_list = {out_dims, out_dims, vel_dims},
                               .op_param_cache_keys = std::move(param_keys)})));

        return std::make_tuple(MakeTensor(std::move(results[0])),
                               MakeTensor(std::move(results[1])),
                               MakeTensor(std::move(results[2])));
      });
}

}  // namespace torch_tpu
