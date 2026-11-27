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
#include <limits>

#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {

// The halves of a float32 value: `high` keeps the top 12 significand bits and
// `low` is the rest, so the product of any two halves is exact in float32.
struct SplitFloat32 {
  mlir::MlirOp high;
  mlir::MlirOp low;
};

[[nodiscard]] SplitFloat32 SplitSignificand(mlir::MlirOp value) {
  mlir::RankedTensorType type = GetTensorTypeOrDie(value);
  const mlir::Type i32_type = value.getBuilder().getOpBuilder().getI32Type();
  mlir::MlirOp bits = stablehlo::BitcastConvert(type.clone(i32_type), value);
  mlir::MlirOp high_mask = MakeConstantLike(bits, ~int32_t{0xFFF});
  mlir::MlirOp high_bits = stablehlo::And(bits, high_mask);
  mlir::MlirOp high = stablehlo::BitcastConvert(type, high_bits);
  return {.high = high, .low = stablehlo::Subtract(value, high)};
}

// Returns the float32 `dividend / divisor` as CPU and CUDA round it.
//
// TPU divides by multiplying with the rounded reciprocal, which can be one ULP
// off: 6500 / 1000 gives 6.5000005, which is no longer a tie, and 23 / 100
// gives 0.22999999. One Newton step on the error of that quotient,
// `quotient * divisor - dividend`, corrects it. The error keeps all its bits
// because it adds up the partial products of the split operands, each of which
// is exact (Dekker).
[[nodiscard]] mlir::MlirOp BuildDivideF32(mlir::MlirOp dividend,
                                          const double divisor) {
  if (divisor > std::numeric_limits<float>::max()) {
    // The divisor is infinite in float32, and x / inf is x * 0.
    mlir::MlirOp zero = MakeConstantLike(dividend, 0.0);
    return stablehlo::Mul(dividend, zero);
  }
  // TPU flushes subnormals to zero, which would drop the reciprocal of 10^38
  // and the correction of a quotient below 1e-31. Dividing by a 2^24 times
  // smaller divisor keeps both normal; scaling the quotient back is exact.
  const double rescale = divisor >= 0x1p25 ? 0x1p-24 : 1.0;
  const double effective_divisor =
      static_cast<double>(static_cast<float>(divisor * rescale));
  mlir::MlirOp divisor_op = MakeConstantLike(dividend, effective_divisor);
  mlir::MlirOp reciprocal = MakeConstantLike(dividend, 1.0 / effective_divisor);
  mlir::MlirOp quotient = stablehlo::Mul(dividend, reciprocal);
  SplitFloat32 split_quotient = SplitSignificand(quotient);
  SplitFloat32 split_divisor = SplitSignificand(divisor_op);
  // Largest partial product first, so that no sum is rounded. Starting from
  // the negated dividend leaves +0.0 for an exact quotient, which keeps the
  // sign of a -0.0 quotient in the subtraction below.
  mlir::MlirOp error = stablehlo::Neg(dividend);
  for (mlir::MlirOp partial_product :
       {stablehlo::Mul(split_quotient.high, split_divisor.high),
        stablehlo::Mul(split_quotient.high, split_divisor.low),
        stablehlo::Mul(split_quotient.low, split_divisor.high),
        stablehlo::Mul(split_quotient.low, split_divisor.low)}) {
    error = stablehlo::Add(error, partial_product);
  }
  mlir::MlirOp correction = stablehlo::Mul(error, reciprocal);
  mlir::MlirOp corrected = stablehlo::Subtract(quotient, correction);
  // The error is NaN for an infinite quotient, which is already what CPU
  // returns.
  mlir::MlirOp is_finite = stablehlo::IsFinite(corrected);
  mlir::MlirOp result = stablehlo::Select(is_finite, corrected, quotient);
  if (rescale == 1.0) {
    return result;
  }
  mlir::MlirOp rescale_op = MakeConstantLike(dividend, rescale);
  return stablehlo::Mul(result, rescale_op);
}

}  // namespace

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

  // Cast to double before taking abs to avoid signed overflow for INT64_MIN.
  const double power = std::pow(10.0, std::fabs(static_cast<double>(decimals)));
  mlir::MlirOp k_power = MakeConstantLike(promoted_input, power);
  const mlir::Type compute_elem_type =
      GetTensorTypeOrDie(promoted_input).getElementType();
  // float64 keeps the plain division: TPU does not compute it in IEEE double.
  const auto divide_by_power = [compute_elem_type, power,
                                &k_power](mlir::MlirOp dividend) {
    return compute_elem_type.isF32() ? BuildDivideF32(dividend, power)
                                     : stablehlo::Div(dividend, k_power);
  };
  // Match PyTorch: `nearbyint(x / 10^-d) * 10^-d` for negative `decimals` and
  // `nearbyint(x * 10^d) / 10^d` otherwise.
  mlir::MlirOp scaled_input = decimals < 0
                                  ? divide_by_power(promoted_input)
                                  : stablehlo::Mul(promoted_input, k_power);
  mlir::MlirOp rounded_scaled_input = stablehlo::RoundNearestEven(scaled_input);
  mlir::MlirOp result = decimals < 0
                            ? stablehlo::Mul(rounded_scaled_input, k_power)
                            : divide_by_power(rounded_scaled_input);
  if (compute_elem_type != orig_elem_type) {
    result = stablehlo::ConvertElementType(result, orig_elem_type);
  }
  return result;
}

}  // namespace torch_tpu
