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

#include "torch_tpu/ops/round/round.h"

#include <cmath>
#include <cstdint>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildRoundShlo(mlir::MlirOp input_op,
                                            int64_t decimals) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);

  // For integer types return the input. Decimals != 0 is not supported.
  bool is_int = input_type.getElementType().isInteger();
  if (is_int && decimals == 0) {
    return input_op;
  }
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Check is already caught by callers, or
                 // (foreach) `decimals` is always 0.
      !is_int, error::kInvalidArgument)
      << "Round not implemented for Int and non-zero decimals.";

  if (decimals == 0) return stablehlo::RoundNearestEven(input_op);

  double power = std::pow(10, decimals);
  auto k_power = MakeConstantLike(input_op, power);
  mlir::MlirOp scaled_input = stablehlo::Mul(input_op, k_power);
  mlir::MlirOp rounded_scaled_input = stablehlo::Round(scaled_input);
  return stablehlo::Div(rounded_scaled_input, k_power);
}

}  // namespace torch_tpu
