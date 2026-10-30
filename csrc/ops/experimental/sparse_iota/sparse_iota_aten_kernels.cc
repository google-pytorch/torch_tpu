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

#include "csrc/ops/experimental/sparse_iota/sparse_iota_aten_kernels.h"

#include <cstdint>
#include <string>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
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

constexpr int64_t kPadValue = 2147483647;
constexpr int64_t kMaxPadPerRow = 8;
constexpr double kWaitThreshold = 0.5;

auto SparseIotaBuilder(int64_t max_non_zeroes, int64_t max_non_zeroes_per_row) {
  return [max_non_zeroes, max_non_zeroes_per_row](
             mlir::MlirOp row_pointers) -> absl::StatusOr<mlir::MlirOp> {
    mlir::MlirBuilder& builder = row_pointers.getBuilder();
    mlir::OpBuilder& op_builder = builder.getOpBuilder();

    // 1. Build Backend Config JSON for SparseCore.
    const std::string backend_config = absl::StrFormat(
        R"({"device_type":"DEVICE_TYPE_SPARSECORE","csr_config":{"max_non_zeroes_per_row":%d,"pad_value":%d,"max_pad_per_row":%d},"sparse_map_row_config":{"wait_threshold":%f}})",
        max_non_zeroes_per_row, kPadValue, kMaxPadPerRow, kWaitThreshold);

    const mlir::NamedAttribute backend_config_attr = op_builder.getNamedAttr(
        "backend_config", op_builder.getStringAttr(backend_config));
    const mlir::NamedAttribute call_target_attr = op_builder.getNamedAttr(
        "call_target_name", op_builder.getStringAttr("SparseIota"));
    const mlir::NamedAttribute has_side_effect_attr = op_builder.getNamedAttr(
        "has_side_effect", op_builder.getBoolAttr(false));
    const auto api_version_attr = op_builder.getNamedAttr(
        "api_version",
        mlir::stablehlo::CustomCallApiVersionAttr::get(
            &builder.getContext(),
            mlir::stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL));

    const auto out_type = mlir::RankedTensorType::get(
        {max_non_zeroes}, op_builder.getIntegerType(32));

    auto op = mlir::stablehlo::CustomCallOp::create(
        op_builder, builder.getLoc(),
        /*resultTypes=*/{out_type},
        /*operands=*/
        mlir::ValueRange{row_pointers.getValue()},
        {call_target_attr, has_side_effect_attr, api_version_attr,
         backend_config_attr});

    return mlir::MlirOp(builder, op.getResult(0));
  };
}

}  // namespace

absl::Status ValidateSparseIotaInputs(const at::Tensor& row_pointers,
                                      int64_t max_non_zeroes,
                                      int64_t max_non_zeroes_per_row) {
  TT_RET_CHECK(row_pointers.dim() == 1, error::kInvalidArgument)
      << "expected row_pointers to be a 1D tensor, got a " << row_pointers.dim()
      << "D tensor of shape " << ToString(row_pointers.sizes());

  TT_RET_CHECK(row_pointers.scalar_type() == at::kInt, error::kInvalidArgument)
      << "expected row_pointers dtype to be torch.int32, got "
      << row_pointers.scalar_type();

  TT_RET_CHECK(max_non_zeroes > 0, error::kInvalidArgument)
      << "expected max_non_zeroes to be positive, got " << max_non_zeroes;

  TT_RET_CHECK(max_non_zeroes_per_row > 0, error::kInvalidArgument)
      << "expected max_non_zeroes_per_row to be positive, got "
      << max_non_zeroes_per_row;

  return absl::OkStatus();
}

at::Tensor AtenSparseIota(const at::Tensor& row_pointers,
                          int64_t max_non_zeroes,
                          int64_t max_non_zeroes_per_row) {
  TT_KERNEL(
      torch_tpu::OpName::kSparseIota, param_keys,
      (row_pointers, max_non_zeroes, max_non_zeroes_per_row), {
        TT_THROW_IF_ERROR(ValidateSparseIotaInputs(row_pointers, max_non_zeroes,
                                                   max_non_zeroes_per_row));

        const torch_tpu::Dimensions out_dims = {max_non_zeroes};

        auto builder_fn =
            SparseIotaBuilder(max_non_zeroes, max_non_zeroes_per_row);

        TT_ASSIGN_OR_THROW(
            auto results,
            DispatchOp<1>(std::move(builder_fn), row_pointers,
                          {.out_dtype = mlir::ElementType::I32,
                           .out_dims = out_dims,
                           .op_param_cache_keys = std::move(param_keys)}));
        return MakeTensor(std::move(results));
      });
}

}  // namespace torch_tpu
