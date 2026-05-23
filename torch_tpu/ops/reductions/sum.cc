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

#include "torch_tpu/ops/reductions/sum.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/Optional.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"

namespace torch_tpu {
namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildSumShlo(
    mlir::MlirOp input_op, absl::Span<const int64_t> reduce_dims,
    ReductionMode reduction_mode,
    c10::optional<mlir::ElementType> element_type_opt) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  mlir::Type mlir_type;
  if (element_type_opt.has_value()) {
    mlir_type =
        mlir::getElementType(builder.getContext(), element_type_opt.value());
  } else {
    mlir_type = input_type.getElementType();
  }
  mlir::MlirOp zero_val = MakeScalarConstant(builder, 0.0, mlir_type);

  auto reduce_fn = [mlir_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<stablehlo::AddOp>(
        mlir_type, rb.getRegion(), rb.getOpBuilder());
  };

  return BuildReductionShlo(input_op, reduce_dims, mlir_type, zero_val,
                            reduce_fn, reduction_mode);
}

}  // namespace torch_tpu
