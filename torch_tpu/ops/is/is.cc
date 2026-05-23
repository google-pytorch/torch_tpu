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

#include "torch_tpu/ops/is/is.h"

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/macro_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

namespace {

// Returns a Boolean tensor with the same shape as input_type and each
// element being the given value.
mlir::MlirOp MakeBoolLike(mlir::MlirOp input_op, mlir::MlirBuilder& builder,
                          bool value) {
  return MakeConstantLike(input_op, value, mlir::ElementType::PRED);
}

mlir::MlirOp IsNanImpl(mlir::MlirOp input_op) {
  // Neither StableHLO nor CHLO has an IsNan op, but we can compose
  // IsFinite and IsInf to check for NaN as !(is_finite || is_inf)
  auto is_finite = mlir::stablehlo::IsFinite(input_op);
  auto is_inf = mlir::chlo::IsInf(input_op);
  auto is_not_nan = mlir::stablehlo::Or(is_finite, is_inf);
  return mlir::stablehlo::Not(is_not_nan);
}
}  // namespace

// Defines a function named FctName that calls FctCall on the input if it is a
// float type, or returns a Boolean tensor with the same shape as the input
// and each element being the given value.
#define TT_FLOAT_IS_OP_BUILDER_(FctName, FctCall, result)                      \
  absl::StatusOr<mlir::MlirOp> FctName(mlir::MlirOp input_op) {                \
    const mlir::RankedTensorType input_type =                                  \
        ::torch_tpu::GetTensorTypeOrDie(input_op);                             \
    if (::torch_tpu::IsFloatType(input_type)) {                                \
      return FctCall(input_op);                                                \
    }                                                                          \
    return ::torch_tpu::MakeBoolLike(input_op, input_op.getBuilder(), result); \
  }                                                                            \
  TT_REQUIRE_SEMICOLON_

// go/keep-sorted start
TT_FLOAT_IS_OP_BUILDER_(BuildIsNanShlo, IsNanImpl, false);
TT_FLOAT_IS_OP_BUILDER_(BuildIsNegInfShlo, mlir::chlo::IsNegInf, false);
TT_FLOAT_IS_OP_BUILDER_(BuildIsPosInfShlo, mlir::chlo::IsPosInf, false);
// go/keep-sorted end

#undef TT_FLOAT_IS_OP_BUILDER_

}  // namespace torch_tpu
