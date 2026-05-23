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

#include "torch_tpu/ops/all_any/all_any.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"

namespace torch_tpu {
namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildAllShlo(mlir::MlirOp input_op,
                                          absl::Span<const int64_t> reduce_dims,
                                          ReductionMode reduction_mode) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::RankedTensorType input_tensor_type = GetTensorTypeOrDie(input_op);
  if (!IsBooleanType(input_tensor_type)) {
    input_op =
        mlir::stablehlo::ConvertElementType(input_op, mlir::ElementType::PRED);
  }

  auto true_value = MakeScalarConstant(builder, true, mlir::ElementType::PRED);
  mlir::Type mlir_type =
      mlir::getElementType(builder.getContext(), mlir::ElementType::PRED);

  auto reduce_fn = [mlir_type](mlir::RegionBuilder& rb) {
    stablehlo::buildReduceBody<stablehlo::AndOp>(mlir_type, rb.getRegion(),
                                                 rb.getOpBuilder());
  };

  return BuildReductionShlo(input_op, reduce_dims, mlir_type, true_value,
                            reduce_fn, reduction_mode);
}

absl::StatusOr<mlir::MlirOp> BuildAnyShlo(mlir::MlirOp input_op,
                                          absl::Span<const int64_t> reduce_dims,
                                          ReductionMode reduction_mode) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::RankedTensorType input_tensor_type = GetTensorTypeOrDie(input_op);
  if (!IsBooleanType(input_tensor_type)) {
    input_op =
        mlir::stablehlo::ConvertElementType(input_op, mlir::ElementType::PRED);
  }

  auto false_value =
      MakeScalarConstant(builder, false, mlir::ElementType::PRED);
  mlir::Type mlir_type =
      mlir::getElementType(builder.getContext(), mlir::ElementType::PRED);

  auto reduce_fn = [mlir_type](mlir::RegionBuilder& rb) {
    stablehlo::buildReduceBody<stablehlo::OrOp>(mlir_type, rb.getRegion(),
                                                rb.getOpBuilder());
  };

  return BuildReductionShlo(input_op, reduce_dims, mlir_type, false_value,
                            reduce_fn, reduction_mode);
}

}  // namespace torch_tpu
