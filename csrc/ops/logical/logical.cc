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

#include "csrc/ops/logical/logical.h"

#include <utility>

#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {

inline absl::StatusOr<mlir::MlirOp> CastToBooleanIfNeeded(mlir::MlirOp op) {
  const mlir::RankedTensorType tensor_type = GetTensorTypeOrDie(op);
  if (!IsBooleanType(tensor_type)) {
    return stablehlo::ConvertElementType(op, mlir::ElementType::PRED);
  }
  return op;
}

struct BroadcastedOperands {
  mlir::MlirOp self;
  mlir::MlirOp other;
};

absl::StatusOr<BroadcastedOperands> BroadcastLogicalOperands(
    mlir::MlirOp self, mlir::MlirOp other) {
  TT_ASSIGN_OR_RETURN(self, CastToBooleanIfNeeded(self));
  TT_ASSIGN_OR_RETURN(other, CastToBooleanIfNeeded(other));
  TT_ASSIGN_OR_RETURN((auto [self_broadcast, other_broadcast]),
                      ApplyBroadcastIfNeeded(self, other));
  return BroadcastedOperands{.self = self_broadcast, .other = other_broadcast};
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildLogicalAndShlo(mlir::MlirOp self,
                                                 mlir::MlirOp other) {
  TT_ASSIGN_OR_RETURN(auto operands, BroadcastLogicalOperands(self, other));
  return stablehlo::And(operands.self, operands.other);
}

absl::StatusOr<mlir::MlirOp> BuildLogicalOrShlo(mlir::MlirOp self,
                                                mlir::MlirOp other) {
  TT_ASSIGN_OR_RETURN(auto operands, BroadcastLogicalOperands(self, other));
  return stablehlo::Or(operands.self, operands.other);
}

absl::StatusOr<mlir::MlirOp> BuildLogicalXorShlo(mlir::MlirOp self,
                                                 mlir::MlirOp other) {
  TT_ASSIGN_OR_RETURN(auto operands, BroadcastLogicalOperands(self, other));
  return stablehlo::Xor(operands.self, operands.other);
}

absl::StatusOr<mlir::MlirOp> BuildLogicalNotShlo(mlir::MlirOp self) {
  TT_ASSIGN_OR_RETURN(self, CastToBooleanIfNeeded(self));
  return stablehlo::Not(self);
}

}  // namespace torch_tpu
