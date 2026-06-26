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

#include "torch_tpu/ops/experimental/sparse_dense_matmul/sparse_dense_matmul_grad_with_rowwise_adagrad_with_minibatching_aten_kernels.h"

#include <array>
#include <cstdint>
#include <string>
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
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "xla/side_effect_util.h"

namespace torch_tpu {

namespace {

// Row-wise Adagrad. Identical plumbing to the (per-element) Adagrad kernel --
// same SparseCore custom call, one accumulator slot variable, two
// hyperparameters (learning_rate, epsilon) -- but the accumulator holds a
// single scalar per embedding row, so it is a 1-D tensor of shape [vocab] (one
// D-th of the per-element Adagrad optimizer state). The libtpu grad-op
// decomposer gathers a rank-1 slot variable with a per-row scalar slice, so the
// accumulator must be 1-D [vocab] (a 2-D [vocab, 1] tensor would take the 2-D
// gather path with a width-D slice and fail shape inference).
//
// The update reduces the squared gradient across the feature dimension to a
// per-row mean before accumulating, matching TorchRec/FBGEMM RowWiseAdagrad:
//
//   acc'[r]      = acc[r] + mean_j(grad[r, j]^2)        // scalar per row
//   param'[r, j] = param[r, j] - lr * grad[r, j] / (sqrt(acc'[r]) + epsilon)
//
// The libtpu decomposer (RewriteOptimizer) rewrites this per-row body (gradient
// and embedding row [1, dim], accumulator [1]) to a per-table body ([num_rows,
// dim] / [num_rows]) by growing only the leading (row) dimension, so the
// reduction over the feature dimension and the row broadcast
// (broadcast_dimensions={0}) survive the rewrite unchanged.
auto SparseDenseMatmulGradWithRowwiseAdagradWithMinibatchingBuilder(
    int64_t device_batch_size, int64_t max_ids_per_partition,
    int64_t max_unique_ids_per_partition) {
  return
      [max_ids_per_partition, max_unique_ids_per_partition](
          torch_tpu::FixedSizeSpan<mlir::MlirOp, 10> inputs) -> absl::
                                                                 StatusOr<torch_tpu::
                                                                              MlirOpResults<2>> {
                                                                   mlir::MlirOp
                                                                       row_pointers =
                                                                           inputs
                                                                               [0];
                                                                   mlir::MlirOp
                                                                       embedding_ids =
                                                                           inputs
                                                                               [1];
                                                                   mlir::MlirOp
                                                                       sample_ids =
                                                                           inputs
                                                                               [2];
                                                                   mlir::MlirOp
                                                                       gains = inputs
                                                                           [3];
                                                                   mlir::MlirOp
                                                                       num_minibatches_per_sparse_core =
                                                                           inputs
                                                                               [4];
                                                                   mlir::MlirOp
                                                                       embedding_table =
                                                                           inputs
                                                                               [5];
                                                                   mlir::MlirOp
                                                                       accumulator =
                                                                           inputs
                                                                               [6];
                                                                   mlir::MlirOp
                                                                       activations_grad =
                                                                           inputs
                                                                               [7];
                                                                   mlir::MlirOp
                                                                       learning_rate =
                                                                           inputs
                                                                               [8];
                                                                   mlir::MlirOp
                                                                       epsilon = inputs
                                                                           [9];

                                                                   mlir::MlirBuilder&
                                                                       builder =
                                                                           row_pointers
                                                                               .getBuilder();
                                                                   mlir::OpBuilder&
                                                                       op_builder =
                                                                           builder
                                                                               .getOpBuilder();

                                                                   auto embedding_table_type =
                                                                       torch_tpu::
                                                                           GetTensorTypeOrDie(
                                                                               embedding_table);
                                                                   auto accumulator_type =
                                                                       torch_tpu::
                                                                           GetTensorTypeOrDie(
                                                                               accumulator);
                                                                   int64_t embedding_dim =
                                                                       embedding_table_type
                                                                           .getShape()
                                                                               [1];

                                                                   // Define the
                                                                   // optimizer
                                                                   // update
                                                                   // function.
                                                                   std::string
                                                                       computation_name =
                                                                           "row"
                                                                           "wis"
                                                                           "e_"
                                                                           "ada"
                                                                           "gra"
                                                                           "d_"
                                                                           "opt"
                                                                           "imi"
                                                                           "zer"
                                                                           "_up"
                                                                           "dat"
                                                                           "e";

                                                                   auto f32_type =
                                                                       op_builder
                                                                           .getF32Type();
                                                                   // Per-row
                                                                   // slices
                                                                   // seen by
                                                                   // the
                                                                   // optimizer
                                                                   // update:
                                                                   // the
                                                                   // gradient,
                                                                   // embedding
                                                                   // row, and
                                                                   // hyperparameters
                                                                   // are [1,
                                                                   // embedding_dim];
                                                                   // the
                                                                   // accumulator
                                                                   // is a
                                                                   // single
                                                                   // scalar per
                                                                   // row, [1].
                                                                   auto row_type =
                                                                       mlir::RankedTensorType::
                                                                           get({1,
                                                                                embedding_dim},
                                                                               f32_type);
                                                                   auto acc_row_type =
                                                                       mlir::RankedTensorType::
                                                                           get({1},
                                                                               f32_type);
                                                                   auto func_type =
                                                                       op_builder
                                                                           .getFunctionType(
                                                                               {row_type,
                                                                                row_type,
                                                                                acc_row_type,
                                                                                row_type,
                                                                                row_type},
                                                                               {mlir::TupleType::get(
                                                                                   op_builder
                                                                                       .getContext(),
                                                                                   {row_type,
                                                                                    acc_row_type})});

                                                                   mlir::ModuleOp module =
                                                                       torch_tpu::
                                                                           GetModuleOp(
                                                                               builder);
                                                                   // Save
                                                                   // current
                                                                   // insertion
                                                                   // point
                                                                   auto prev_insertion_point =
                                                                       op_builder
                                                                           .saveInsertionPoint();

                                                                   op_builder.setInsertionPointToEnd(
                                                                       module
                                                                           .getBody());
                                                                   auto func = mlir::
                                                                       func::FuncOp::create(
                                                                           op_builder,
                                                                           builder
                                                                               .getLoc(),
                                                                           computation_name,
                                                                           func_type);
                                                                   func.setVisibility(
                                                                       mlir::SymbolTable::
                                                                           Visibility::
                                                                               Private);

                                                                   auto* entry_block =
                                                                       func.addEntryBlock();
                                                                   op_builder.setInsertionPointToStart(
                                                                       entry_block);

                                                                   mlir::MlirOp grad(
                                                                       builder,
                                                                       entry_block
                                                                           ->getArgument(
                                                                               0));
                                                                   mlir::MlirOp param(
                                                                       builder,
                                                                       entry_block
                                                                           ->getArgument(
                                                                               1));
                                                                   mlir::MlirOp acc_arg(
                                                                       builder,
                                                                       entry_block
                                                                           ->getArgument(
                                                                               2));
                                                                   mlir::MlirOp lr(
                                                                       builder,
                                                                       entry_block
                                                                           ->getArgument(
                                                                               3));
                                                                   mlir::MlirOp epsilon_arg(
                                                                       builder,
                                                                       entry_block
                                                                           ->getArgument(
                                                                               4));

                                                                   // Per-row
                                                                   // mean of
                                                                   // squared
                                                                   // gradient:
                                                                   // mean_j(grad^2),
                                                                   // shape [1].
                                                                   // Scale
                                                                   // grad^2 by
                                                                   // the rank-0
                                                                   // scalar
                                                                   // 1/embedding_dim
                                                                   // *before*
                                                                   // reducing
                                                                   // so the
                                                                   // per-row
                                                                   // reduce
                                                                   // yields the
                                                                   // mean
                                                                   // directly.
                                                                   // We
                                                                   // deliberately
                                                                   // avoid a
                                                                   // [1]-shaped
                                                                   // constant
                                                                   // for the
                                                                   // divisor:
                                                                   // the libtpu
                                                                   // RewriteOptimizer
                                                                   // grows
                                                                   // every
                                                                   // leading-dim-1
                                                                   // instruction
                                                                   // to
                                                                   // [<=num_unique_tokens],
                                                                   // which
                                                                   // would
                                                                   // leave a
                                                                   // [1]-literal
                                                                   // constant
                                                                   // declaring
                                                                   // a [<=N]
                                                                   // shape
                                                                   // (shape/literal
                                                                   // mismatch).
                                                                   // A rank-0
                                                                   // scalar has
                                                                   // no leading
                                                                   // dim, so
                                                                   // the
                                                                   // rewriter
                                                                   // skips it.
                                                                   auto grad_sq = mlir::
                                                                       stablehlo::Mul(
                                                                           grad,
                                                                           grad);  // [1, D]
                                                                   mlir::MlirOp inv_d =
                                                                       torch_tpu::MakeScalarConstant(
                                                                           builder,
                                                                           1.0 /
                                                                               static_cast<
                                                                                   double>(
                                                                                   embedding_dim),
                                                                           f32_type);
                                                                   mlir::MlirOp inv_d_bcast =
                                                                       mlir::stablehlo::BroadcastInDim(
                                                                           row_type,
                                                                           inv_d,
                                                                           /*broadcast_dimensions=*/
                                                                           {});
                                                                   auto grad_sq_scaled =
                                                                       mlir::stablehlo::Mul(
                                                                           grad_sq,
                                                                           inv_d_bcast);  // [1, D]
                                                                   mlir::MlirOp zero =
                                                                       torch_tpu::MakeScalarConstant(
                                                                           builder,
                                                                           0.0,
                                                                           f32_type);
                                                                   auto sum_reduce_builder =
                                                                       [f32_type](
                                                                           mlir::RegionBuilder&
                                                                               rb) {
                                                                         mlir::stablehlo::buildReduceBody<
                                                                             mlir::stablehlo::
                                                                                 AddOp>(
                                                                             f32_type,
                                                                             rb.getRegion(),
                                                                             rb.getOpBuilder());
                                                                       };
                                                                   // Reduce
                                                                   // over the
                                                                   // feature
                                                                   // dimension
                                                                   // (axis 1),
                                                                   // keeping
                                                                   // the row
                                                                   // dimension.
                                                                   auto reduced = mlir::
                                                                       stablehlo::Reduce(
                                                                           builder,
                                                                           {grad_sq_scaled},
                                                                           {zero},
                                                                           sum_reduce_builder,
                                                                           /*reduction_axes=*/
                                                                           {1});
                                                                   mlir::MlirOp
                                                                       grad_sq_mean =
                                                                           reduced
                                                                               [0];  // [1]

                                                                   // acc' = acc
                                                                   // +
                                                                   // mean_j(grad^2),
                                                                   // both
                                                                   // scalar-per-row,
                                                                   // shape [1].
                                                                   auto new_acc = mlir::
                                                                       stablehlo::Add(
                                                                           acc_arg,
                                                                           grad_sq_mean);
                                                                   // Broadcast
                                                                   // the
                                                                   // per-row
                                                                   // scalar
                                                                   // back
                                                                   // across the
                                                                   // feature
                                                                   // dimension
                                                                   // for the
                                                                   // parameter
                                                                   // update:
                                                                   // [1] -> [1,
                                                                   // embedding_dim],
                                                                   // row mapped
                                                                   // to output
                                                                   // dim 0.
                                                                   mlir::MlirOp sqrt_acc =
                                                                       mlir::stablehlo::
                                                                           Sqrt(
                                                                               new_acc);
                                                                   mlir::MlirOp sqrt_acc_bcast =
                                                                       mlir::stablehlo::BroadcastInDim(
                                                                           row_type,
                                                                           sqrt_acc,
                                                                           /*broadcast_dimensions=*/
                                                                           {0});
                                                                   auto sqrt_acc_plus_epsilon =
                                                                       mlir::stablehlo::
                                                                           Add(sqrt_acc_bcast,
                                                                               epsilon_arg);
                                                                   auto lr_grad = mlir::
                                                                       stablehlo::Mul(
                                                                           lr,
                                                                           grad);
                                                                   auto update = mlir::
                                                                       stablehlo::Div(
                                                                           lr_grad,
                                                                           sqrt_acc_plus_epsilon);
                                                                   auto updated_param =
                                                                       mlir::stablehlo::
                                                                           Subtract(
                                                                               param,
                                                                               update);

                                                                   auto tuple_op = mlir::stablehlo::
                                                                       TupleOp::create(
                                                                           op_builder,
                                                                           builder
                                                                               .getLoc(),
                                                                           mlir::ValueRange{
                                                                               updated_param
                                                                                   .getValue(),
                                                                               new_acc
                                                                                   .getValue()});
                                                                   mlir::func::ReturnOp::create(
                                                                       op_builder,
                                                                       builder
                                                                           .getLoc(),
                                                                       tuple_op
                                                                           .getOperation()
                                                                           ->getResults());

                                                                   // Restore
                                                                   // insertion
                                                                   // point
                                                                   op_builder.restoreInsertionPoint(
                                                                       prev_insertion_point);

                                                                   // Construct
                                                                   // DictionaryAttr
                                                                   // for
                                                                   // frontend_attributes.
                                                                   // The
                                                                   // non-minibatching
                                                                   // grad
                                                                   // optimizer-update
                                                                   // custom
                                                                   // call is
                                                                   // consumed
                                                                   // by the
                                                                   // SparseCore
                                                                   // compiler,
                                                                   // but
                                                                   // libtpu's
                                                                   // frontend-attribute
                                                                   // conversion
                                                                   // whitelist
                                                                   // does not
                                                                   // include
                                                                   // it. Put
                                                                   // the
                                                                   // sparse-dense-matmul
                                                                   // limits in
                                                                   // backend_config
                                                                   // directly.
                                                                   std::vector<
                                                                       mlir::
                                                                           NamedAttribute>
                                                                       frontend_attrs;
                                                                   frontend_attrs
                                                                       .reserve(
                                                                           1);
                                                                   frontend_attrs
                                                                       .push_back(
                                                                           op_builder
                                                                               .getNamedAttr(
                                                                                   xla::
                                                                                       kXlaComputeTypeAttr,
                                                                                   op_builder
                                                                                       .getStringAttr(
                                                                                           xla::
                                                                                               kXlaComputeTypeSparse)));
                                                                   std::string backend_config =
                                                                       R"({"sparse_dense_matmul_config":{"max_ids_per_partition":)" +
                                                                       std::to_string(
                                                                           max_ids_per_partition) +
                                                                       R"(,"max_unique_ids_per_partition":)" +
                                                                       std::to_string(
                                                                           max_unique_ids_per_partition) +
                                                                       R"(,"sharding_strategy":"SHARDING_STRATEGY_MOD","pad_value":2147483647,"num_slot_variables":1,"num_hyperparameters":2}})";

                                                                   std::string
                                                                       call_target =
                                                                           "Spa"
                                                                           "rse"
                                                                           "Den"
                                                                           "seM"
                                                                           "atm"
                                                                           "ulG"
                                                                           "rad"
                                                                           "Opt"
                                                                           "imi"
                                                                           "zer"
                                                                           "Upd"
                                                                           "ate"
                                                                           "Wit"
                                                                           "hMi"
                                                                           "nib"
                                                                           "atc"
                                                                           "hin"
                                                                           "gO"
                                                                           "p";

                                                                   mlir::NamedAttribute call_target_attr =
                                                                       op_builder
                                                                           .getNamedAttr(
                                                                               "call_target_name",
                                                                               op_builder
                                                                                   .getStringAttr(
                                                                                       call_target));
                                                                   mlir::NamedAttribute has_side_effect_attr =
                                                                       op_builder
                                                                           .getNamedAttr(
                                                                               "has_side_effect",
                                                                               op_builder
                                                                                   .getBoolAttr(
                                                                                       false));
                                                                   auto api_version_attr =
                                                                       op_builder
                                                                           .getNamedAttr(
                                                                               "api_version",
                                                                               mlir::stablehlo::CustomCallApiVersionAttr::
                                                                                   get(&builder
                                                                                            .getContext(),
                                                                                       mlir::stablehlo::
                                                                                           CustomCallApiVersion::
                                                                                               API_VERSION_ORIGINAL));

                                                                   auto frontend_attributes_attr =
                                                                       op_builder
                                                                           .getNamedAttr(
                                                                               "mhlo.frontend_attributes",
                                                                               op_builder
                                                                                   .getDictionaryAttr(
                                                                                       frontend_attrs));
                                                                   auto backend_config_attr =
                                                                       op_builder
                                                                           .getNamedAttr(
                                                                               "backend_config",
                                                                               op_builder
                                                                                   .getStringAttr(
                                                                                       backend_config));

                                                                   auto called_computations_attr =
                                                                       op_builder
                                                                           .getNamedAttr(
                                                                               "called_computations",
                                                                               op_builder
                                                                                   .getArrayAttr(
                                                                                       {mlir::SymbolRefAttr::get(
                                                                                           op_builder
                                                                                               .getContext(),
                                                                                           computation_name)}));

                                                                   std::vector<mlir::Value> operands = {
                                                                       row_pointers
                                                                           .getValue(),
                                                                       embedding_ids
                                                                           .getValue(),
                                                                       sample_ids
                                                                           .getValue(),
                                                                       gains
                                                                           .getValue(),
                                                                       num_minibatches_per_sparse_core
                                                                           .getValue(),
                                                                       embedding_table
                                                                           .getValue(),
                                                                       accumulator
                                                                           .getValue(),
                                                                       activations_grad
                                                                           .getValue(),
                                                                       learning_rate
                                                                           .getValue(),
                                                                       epsilon
                                                                           .getValue()};

                                                                   auto out_type = mlir::
                                                                       TupleType::get(
                                                                           op_builder
                                                                               .getContext(),
                                                                           {embedding_table_type,
                                                                            accumulator_type});

                                                                   auto op = mlir::stablehlo::CustomCallOp::create(
                                                                       op_builder,
                                                                       builder
                                                                           .getLoc(),
                                                                       /*resultTypes=*/
                                                                       {out_type},
                                                                       /*operands=*/
                                                                       operands,
                                                                       {call_target_attr,
                                                                        has_side_effect_attr,
                                                                        api_version_attr,
                                                                        frontend_attributes_attr,
                                                                        backend_config_attr,
                                                                        called_computations_attr});

                                                                   auto gte_table = mlir::stablehlo::
                                                                       GetTupleElementOp::create(
                                                                           op_builder,
                                                                           builder
                                                                               .getLoc(),
                                                                           op.getResult(
                                                                               0),
                                                                           0);
                                                                   auto gte_accumulator = mlir::
                                                                       stablehlo::GetTupleElementOp::create(
                                                                           op_builder,
                                                                           builder
                                                                               .getLoc(),
                                                                           op.getResult(
                                                                               0),
                                                                           1);

                                                                   return torch_tpu::MlirOpResults<
                                                                       2>(
                                                                       {mlir::MlirOp(
                                                                            builder,
                                                                            gte_table
                                                                                .getResult()),
                                                                        mlir::MlirOp(
                                                                            builder,
                                                                            gte_accumulator
                                                                                .getResult())});
                                                                 };
}

}  // namespace

std::tuple<at::Tensor, at::Tensor>
AtenSparseDenseMatmulGradWithRowwiseAdagradWithMinibatching(
    const at::Tensor& row_pointers, const at::Tensor& embedding_ids,
    const at::Tensor& sample_ids, const at::Tensor& gains,
    const at::Tensor& num_minibatches_per_sparse_core,
    const at::Tensor& embedding_table, const at::Tensor& accumulator,
    const at::Tensor& activations_grad, const at::Tensor& learning_rate,
    const at::Tensor& epsilon, int64_t device_batch_size,
    int64_t max_ids_per_partition, int64_t max_unique_ids_per_partition) {
  TT_KERNEL(
      torch_tpu::OpName::
          kSparseDenseMatmulGradWithRowwiseAdagradWithMinibatching,
      param_keys,
      (row_pointers, embedding_ids, sample_ids, gains,
       num_minibatches_per_sparse_core, embedding_table, accumulator,
       activations_grad, learning_rate, epsilon, device_batch_size,
       max_ids_per_partition, max_unique_ids_per_partition),
      {
        std::array<torch_tpu::TensorHolder, 10> inputs = {
            row_pointers,
            embedding_ids,
            sample_ids,
            gains,
            num_minibatches_per_sparse_core,
            embedding_table,
            accumulator,
            activations_grad,
            learning_rate,
            epsilon};

        torch_tpu::Dimensions table_dims(embedding_table.sizes().begin(),
                                         embedding_table.sizes().end());
        // The row-wise accumulator is a 1-D [vocab] slot variable.
        torch_tpu::Dimensions acc_dims(accumulator.sizes().begin(),
                                       accumulator.sizes().end());

        auto builder_fn =
            SparseDenseMatmulGradWithRowwiseAdagradWithMinibatchingBuilder(
                device_batch_size, max_ids_per_partition,
                max_unique_ids_per_partition);

        TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                           torch_tpu::ConvertTo<mlir::ElementType>(
                               embedding_table.scalar_type()));

        TT_ASSIGN_OR_THROW(
            auto results, (torch_tpu::DispatchOp<10, 2>(
                              builder_fn, inputs,
                              {.out_dtypes = {out_dtype, out_dtype},
                               .out_dims_list = {table_dims, acc_dims},
                               .op_param_cache_keys = std::move(param_keys)})));

        return std::make_tuple(torch_tpu::MakeTensor(std::move(results[0])),
                               torch_tpu::MakeTensor(std::move(results[1])));
      });
}

}  // namespace torch_tpu
