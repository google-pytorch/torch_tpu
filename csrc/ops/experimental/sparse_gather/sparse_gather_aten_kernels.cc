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

#include "csrc/ops/experimental/sparse_gather/sparse_gather_aten_kernels.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

namespace {

auto SparseGatherBuilder(int64_t max_non_zeroes_per_row) {
  return [max_non_zeroes_per_row](
             torch_tpu::FixedSizeSpan<mlir::MlirOp, 3> inputs)
             -> absl::StatusOr<mlir::MlirOp> {
    mlir::MlirOp row_pointers = inputs[0];
    mlir::MlirOp indices = inputs[1];
    mlir::MlirOp operand = inputs[2];
    mlir::MlirBuilder& builder = row_pointers.getBuilder();
    mlir::OpBuilder& op_builder = builder.getOpBuilder();

    // 1. Pack CSR components (row_pointers, indices) into a tuple.
    auto tuple_op = mlir::stablehlo::TupleOp::create(
        op_builder, builder.getLoc(),
        mlir::ValueRange{row_pointers.getValue(), indices.getValue()});

    // 2. Build Backend Config JSON for SparseCore.
    std::string backend_config = absl::StrFormat(
        R"({"device_type":"DEVICE_TYPE_SPARSECORE","csr_config":{"max_non_zeroes_per_row":%d,"max_pad_per_row":8},"sparse_map_row_config":{"wait_threshold":0.5}})",
        max_non_zeroes_per_row);

    mlir::NamedAttribute backend_config_attr = op_builder.getNamedAttr(
        "backend_config", op_builder.getStringAttr(backend_config));
    mlir::NamedAttribute call_target_attr = op_builder.getNamedAttr(
        "call_target_name", op_builder.getStringAttr("SparseGather"));
    mlir::NamedAttribute has_side_effect_attr = op_builder.getNamedAttr(
        "has_side_effect", op_builder.getBoolAttr(false));
    auto api_version_attr = op_builder.getNamedAttr(
        "api_version",
        mlir::stablehlo::CustomCallApiVersionAttr::get(
            &builder.getContext(),
            mlir::stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL));

    auto operand_type = torch_tpu::GetTensorTypeOrDie(operand);
    auto indices_type = torch_tpu::GetTensorTypeOrDie(indices);
    int64_t num_gathered_elements = indices_type.getShape()[0];
    int64_t embedding_dim = operand_type.getShape()[1];

    auto out_type = mlir::RankedTensorType::get(
        {num_gathered_elements, embedding_dim}, operand_type.getElementType());

    auto op = mlir::stablehlo::CustomCallOp::create(
        op_builder, builder.getLoc(),
        /*resultTypes=*/{out_type},
        /*operands=*/
        mlir::ValueRange{operand.getValue(), tuple_op.getResult()},
        {call_target_attr, has_side_effect_attr, api_version_attr,
         backend_config_attr});

    return mlir::MlirOp(builder, op.getResult(0));
  };
}

}  // namespace

absl::Status ValidateSparseGatherInputs(const at::Tensor& row_pointers,
                                        const at::Tensor& indices,
                                        const at::Tensor& operand,
                                        int64_t max_non_zeroes_per_row) {
  TT_RET_CHECK(row_pointers.dim() == 1, error::kInvalidArgument)
      << "expected row_pointers to be a 1D tensor, got a " << row_pointers.dim()
      << "D tensor of shape " << ToString(row_pointers.sizes());

  TT_RET_CHECK(indices.dim() == 1, error::kInvalidArgument)
      << "expected indices to be a 1D tensor, got a " << indices.dim()
      << "D tensor of shape " << ToString(indices.sizes());

  TT_RET_CHECK(operand.dim() == 2, error::kInvalidArgument)
      << "expected operand to be a 2D tensor, got a " << operand.dim()
      << "D tensor of shape " << ToString(operand.sizes());

  TT_RET_CHECK(indices.size(0) == row_pointers.size(0) * max_non_zeroes_per_row,
               error::kInvalidArgument)
      << "expected indices length to match the maximum number of "
         "non-zeroes, i.e. row_pointers length * maximum number of "
         "non-zeroes per row ("
      << row_pointers.size(0) << " * " << max_non_zeroes_per_row << " = "
      << row_pointers.size(0) * max_non_zeroes_per_row << "), got "
      << indices.size(0);

  return absl::OkStatus();
}

at::Tensor AtenSparseGather(const at::Tensor& row_pointers,
                            const at::Tensor& indices,
                            const at::Tensor& operand,
                            int64_t max_non_zeroes_per_row) {
  TT_KERNEL(
      torch_tpu::OpName::kSparseGather, param_keys,
      (row_pointers, indices, operand, max_non_zeroes_per_row), {
        TT_THROW_IF_ERROR(ValidateSparseGatherInputs(
            row_pointers, indices, operand, max_non_zeroes_per_row));

        std::array<torch_tpu::TensorHolder, 3> inputs = {row_pointers, indices,
                                                         operand};

        torch_tpu::Dimensions out_dims = {indices.size(0), operand.size(1)};

        auto builder_fn = SparseGatherBuilder(max_non_zeroes_per_row);

        TT_ASSIGN_OR_THROW(
            mlir::ElementType out_dtype,
            torch_tpu::ConvertTo<mlir::ElementType>(operand.scalar_type()));

        TT_ASSIGN_OR_THROW(
            auto results, (torch_tpu::DispatchOp<3, 1>(
                              builder_fn, inputs,
                              {.out_dtype = out_dtype,
                               .out_dims = out_dims,
                               .op_param_cache_keys = std::move(param_keys)})));
        return torch_tpu::MakeTensor(results);
      });
}

}  // namespace torch_tpu
