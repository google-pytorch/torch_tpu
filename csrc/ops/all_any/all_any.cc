/*
 * Copyright 2025 Google LLC
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

#include "csrc/ops/all_any/all_any.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/reductions/reductions.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildAllAnyReductionShlo(
    AllAnyOp op, mlir::MlirOp input_op, absl::Span<const int64_t> reduce_dims,
    ReductionMode reduction_mode) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::RankedTensorType input_tensor_type = GetTensorTypeOrDie(input_op);
  if (!IsBooleanType(input_tensor_type)) {
    input_op =
        mlir::stablehlo::ConvertElementType(input_op, mlir::ElementType::PRED);
  }

  const bool identity_value = (op == AllAnyOp::kAll);
  auto init_value =
      MakeScalarConstant(builder, identity_value, mlir::ElementType::PRED);
  mlir::Type mlir_type =
      mlir::getElementType(builder.getContext(), mlir::ElementType::PRED);

  auto reduce_fn = [mlir_type, op](mlir::RegionBuilder& rb) {
    if (op == AllAnyOp::kAll) {
      stablehlo::buildReduceBody<stablehlo::AndOp>(mlir_type, rb.getRegion(),
                                                   rb.getOpBuilder());
      return;
    }
    stablehlo::buildReduceBody<stablehlo::OrOp>(mlir_type, rb.getRegion(),
                                                rb.getOpBuilder());
  };

  return BuildReductionShlo(input_op, reduce_dims, mlir_type, init_value,
                            reduce_fn, reduction_mode);
}

}  // namespace torch_tpu
