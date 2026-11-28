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

// This shared StableHLO builder has no ATen entry point using TT_KERNEL.
#include "csrc/ops/linalg/eigh/eigh_builder_kernels.h"

#include <array>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

absl::StatusOr<MlirOpResults<2>> BuildEighShlo(mlir::MlirOp input,
                                               EighTriangle triangle,
                                               EighOrder order) {
  const mlir::RankedTensorType a_type = GetTensorTypeOrDie(input);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType a_elem_type,
                      ConvertTo<mlir::ElementType>(a_type.getElementType()));
  const mlir::ElementType w_elem_type = RealComponentOf(a_elem_type);

  mlir::MlirBuilder& builder = input.getBuilder();
  const mlir::Type w_mlir_type =
      mlir::getElementType(builder.getContext(), w_elem_type);
  const mlir::RankedTensorType w_type =
      mlir::RankedTensorType::get(a_type.getShape().drop_back(1), w_mlir_type);

  // Eigh expects lower, sort_eigenvalues, max_iter, and tolerance.
  const std::string config =
      absl::StrCat(triangle == EighTriangle::kLower ? 1 : 0, ",",
                   order == EighOrder::kAscending ? 1 : 0, ",100,",
                   w_elem_type == mlir::ElementType::F64 ? "1e-12" : "1e-6");

  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const std::array<mlir::NamedAttribute, 4> attributes = {
      op_builder.getNamedAttr("call_target_name",
                              op_builder.getStringAttr("Eigh")),
      op_builder.getNamedAttr("has_side_effect", op_builder.getBoolAttr(false)),
      op_builder.getNamedAttr(
          "api_version",
          mlir::stablehlo::CustomCallApiVersionAttr::get(
              &builder.getContext(),
              mlir::stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL)),
      op_builder.getNamedAttr("backend_config",
                              op_builder.getStringAttr(config)),
  };
  const std::array<mlir::Type, 2> result_types = {a_type, w_type};
  const std::array<mlir::Value, 1> operands = {input.getValue()};
  mlir::stablehlo::CustomCallOp eigh = mlir::stablehlo::CustomCallOp::create(
      op_builder, builder.getLoc(), result_types, operands, attributes);
  return {{mlir::MlirOp(builder, eigh.getResult(0)),
           mlir::MlirOp(builder, eigh.getResult(1))}};
}

}  // namespace torch_tpu
