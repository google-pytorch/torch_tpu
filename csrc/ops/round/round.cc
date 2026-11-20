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

#include "csrc/ops/round/round.h"

#include <cmath>
#include <cstdint>

#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

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
      << "round not implemented for integer dtypes and non-zero decimals";

  if (decimals == 0) return stablehlo::RoundNearestEven(input_op);

  const mlir::Type orig_elem_type = input_type.getElementType();
  // Match PyTorch CPU's opmath_t (float32 for float16/bfloat16) so
  // 10^|decimals| and scaled inputs don't overflow or lose precision in reduced
  // float dtypes. Note that PyTorch's CUDA kernel
  // (`round_decimals_kernel_cuda`) currently stays in `scalar_t`, which
  // overflows float16 for decimals >= 4 and truncates 10^4 in bfloat16; we
  // intentionally follow CPU here.
  TT_ASSIGN_OR_RETURN(mlir::MlirOp promoted_input, PromoteFloatDtype(input_op));

  const bool is_negative_decimals = decimals < 0;
  // Cast to double before taking abs to avoid signed overflow for INT64_MIN.
  const double abs_decimals = std::fabs(static_cast<double>(decimals));
  const double power = std::pow(10.0, abs_decimals);
  mlir::MlirOp k_power = MakeConstantLike(promoted_input, power);
  // Match PyTorch's `nearbyint(x / 10^-d) * 10^-d` for negative `decimals` to
  // avoid reciprocal inaccuracy when unscaling. Note that XLA's
  // AlgebraicSimplifier still folds `Div(x, Const(10^-d))` into multiplication
  // by `1 / 10^-d` on the input scaling step.
  mlir::MlirOp scaled_input = is_negative_decimals
                                  ? stablehlo::Div(promoted_input, k_power)
                                  : stablehlo::Mul(promoted_input, k_power);
  mlir::MlirOp rounded_scaled_input = stablehlo::RoundNearestEven(scaled_input);
  mlir::MlirOp result = is_negative_decimals
                            ? stablehlo::Mul(rounded_scaled_input, k_power)
                            : stablehlo::Div(rounded_scaled_input, k_power);
  if (GetTensorTypeOrDie(promoted_input).getElementType() != orig_elem_type) {
    result = stablehlo::ConvertElementType(result, orig_elem_type);
  }
  return result;
}

}  // namespace torch_tpu
