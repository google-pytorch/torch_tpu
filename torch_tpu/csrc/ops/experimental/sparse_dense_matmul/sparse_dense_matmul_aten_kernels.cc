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

#include "torch_tpu/csrc/ops/experimental/sparse_dense_matmul/sparse_dense_matmul_aten_kernels.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
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

auto SparseDenseMatmulBuilder(int64_t device_batch_size,
                              int64_t max_ids_per_partition,
                              int64_t max_unique_ids_per_partition) {
  return [device_batch_size, max_ids_per_partition,
          max_unique_ids_per_partition](
             torch_tpu::FixedSizeSpan<mlir::MlirOp, 5> inputs)
             -> absl::StatusOr<mlir::MlirOp> {
    mlir::MlirOp row_pointers = inputs[0];
    mlir::MlirOp embedding_ids = inputs[1];
    mlir::MlirOp sample_ids = inputs[2];
    mlir::MlirOp gains = inputs[3];
    mlir::MlirOp embedding_table = inputs[4];
    mlir::MlirBuilder& builder = row_pointers.getBuilder();
    mlir::OpBuilder& op_builder = builder.getOpBuilder();

    // Construct DictionaryAttr for frontend_attributes
    std::vector<mlir::NamedAttribute> frontend_attrs;
    frontend_attrs.reserve(5);
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

    auto frontend_attributes_attr =
        op_builder.getNamedAttr("mhlo.frontend_attributes",
                                op_builder.getDictionaryAttr(frontend_attrs));

    std::string call_target = "SparseDenseMatmulOp";

    mlir::NamedAttribute call_target_attr = op_builder.getNamedAttr(
        "call_target_name", op_builder.getStringAttr(call_target));
    mlir::NamedAttribute has_side_effect_attr = op_builder.getNamedAttr(
        "has_side_effect", op_builder.getBoolAttr(false));
    auto api_version_attr = op_builder.getNamedAttr(
        "api_version",
        mlir::stablehlo::CustomCallApiVersionAttr::get(
            &builder.getContext(),
            mlir::stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL));

    auto embedding_table_type = torch_tpu::GetTensorTypeOrDie(embedding_table);
    int64_t embedding_dim = embedding_table_type.getShape()[1];

    // Operands
    std::vector<mlir::Value> operands = {
        row_pointers.getValue(), embedding_ids.getValue(),
        sample_ids.getValue(), gains.getValue(), embedding_table.getValue()};

    // Create activation_init operand (zero tensor of shape [device_batch_size,
    // embedding_dim])
    auto activation_init =
        torch_tpu::MakeConstant(builder, 0.0f, op_builder.getF32Type(),
                                {device_batch_size, embedding_dim});
    operands.push_back(activation_init.getValue());

    auto out_type = mlir::RankedTensorType::get(
        {device_batch_size, embedding_dim}, op_builder.getF32Type());

    auto op = mlir::stablehlo::CustomCallOp::create(
        op_builder, builder.getLoc(),
        /*resultTypes=*/{out_type},
        /*operands=*/operands,
        {call_target_attr, has_side_effect_attr, api_version_attr,
         frontend_attributes_attr});

    return mlir::MlirOp(builder, op.getResult(0));
  };
}

}  // namespace

at::Tensor AtenSparseDenseMatmul(
    const at::Tensor& row_pointers, const at::Tensor& embedding_ids,
    const at::Tensor& sample_ids, const at::Tensor& gains,
    const at::Tensor& embedding_table, int64_t device_batch_size,
    int64_t max_ids_per_partition, int64_t max_unique_ids_per_partition) {
  TT_KERNEL(
      torch_tpu::OpName::kSparseDenseMatmul, param_keys,
      (row_pointers, embedding_ids, sample_ids, gains, embedding_table,
       device_batch_size, max_ids_per_partition, max_unique_ids_per_partition),
      {
        std::array<torch_tpu::TensorHolder, 5> inputs = {
            row_pointers, embedding_ids, sample_ids, gains, embedding_table};

        torch_tpu::Dimensions out_dims = {device_batch_size,
                                          embedding_table.size(1)};

        auto builder_fn =
            SparseDenseMatmulBuilder(device_batch_size, max_ids_per_partition,
                                     max_unique_ids_per_partition);

        TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                           torch_tpu::ConvertTo<mlir::ElementType>(at::kFloat));

        TT_ASSIGN_OR_THROW(
            auto results, (torch_tpu::DispatchOp<5, 1>(
                              builder_fn, inputs,
                              {.out_dtype = out_dtype,
                               .out_dims = out_dims,
                               .op_param_cache_keys = std::move(param_keys)})));

        return torch_tpu::MakeTensor(results);
      });
}

}  // namespace torch_tpu
