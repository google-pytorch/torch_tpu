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

#include "csrc/ops/experimental/ragged_all_to_all/ragged_all_to_all_builder.h"

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/pjrt/pjrt_utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

absl::StatusOr<mlir::MlirOp> BuildRaggedAllToAllShlo(
    mlir::MlirOp operand, mlir::MlirOp output, mlir::MlirOp input_offsets,
    mlir::MlirOp send_sizes, mlir::MlirOp output_offsets,
    mlir::MlirOp recv_sizes, mlir::DenseIntElementsAttr replica_groups_attr,
    int64_t channel_id) {
  auto& builder = operand.getBuilder();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();

  mlir::NamedAttribute call_target_attr = op_builder.getNamedAttr(
      "call_target_name", op_builder.getStringAttr("ragged_all_to_all"));
  mlir::NamedAttribute has_side_effect_attr =
      op_builder.getNamedAttr("has_side_effect", op_builder.getBoolAttr(false));
  auto api_version_attr = op_builder.getNamedAttr(
      "api_version",
      mlir::stablehlo::CustomCallApiVersionAttr::get(
          &builder.getContext(),
          mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI));

  mlir::NamedAttribute channel_id_attr = op_builder.getNamedAttr(
      "channel_id", op_builder.getI64IntegerAttr(channel_id));
  mlir::NamedAttribute replica_groups_config_attr =
      op_builder.getNamedAttr("replica_groups", replica_groups_attr);

  auto backend_config_attr = op_builder.getNamedAttr(
      "backend_config", op_builder.getDictionaryAttr(
                            {channel_id_attr, replica_groups_config_attr}));

  // SparseCore frontend attributes: on architectures supporting SparseCore,
  // set compute_type to "tpu_embedding".
  std::vector<mlir::NamedAttribute> frontend_attrs;
  frontend_attrs.reserve(2);
  if (TpuDeviceSupportsSparseCore()) {
    frontend_attrs.push_back(op_builder.getNamedAttr(
        "compute_type", op_builder.getStringAttr("tpu_embedding")));
  }
  frontend_attrs.push_back(
      op_builder.getNamedAttr("inlineable", op_builder.getStringAttr("false")));
  auto frontend_attributes_attr = op_builder.getNamedAttr(
      "mhlo.frontend_attributes", op_builder.getDictionaryAttr(frontend_attrs));

  const mlir::RankedTensorType output_type = GetTensorTypeOrDie(output);

  auto op = mlir::stablehlo::CustomCallOp::create(
      op_builder, builder.getLoc(), {output_type},
      {operand.getValue(), output.getValue(), input_offsets.getValue(),
       send_sizes.getValue(), output_offsets.getValue(), recv_sizes.getValue()},
      {call_target_attr, has_side_effect_attr, api_version_attr,
       backend_config_attr, frontend_attributes_attr});

  return mlir::MlirOp(builder, op.getResult(0));
}

}  // namespace torch_tpu
