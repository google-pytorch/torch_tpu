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

#include "csrc/ops/lcm/lcm_aten_kernels.h"

#include <cstdint>

#include "ATen/core/ATen_fwd.h"
#include "ATen/ops/result_type.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/binary_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

// Upper bounds on Euclidean algorithm iterations by Lamé's theorem:
// - 32-bit signed integers (<= 2^31 < F_47): <= 45 division steps + 1 swap.
// - 64-bit signed integers (<= 2^63 < F_93): <= 92 division steps + 1 swap.
constexpr int32_t kMaxGcdIterationsI32 = 47;
constexpr int32_t kMaxGcdIterationsI64 = 95;

absl::Status ValidateLcmInputs(const at::Tensor& self, const at::Tensor& other,
                               const at::Tensor& out) {
  const at::ScalarType common_dtype = at::result_type(self, other);
  TT_RETURN_IF_ERROR(ValidateOutDtype(out, common_dtype));
  TT_RET_CHECK(c10::isIntegralType(common_dtype, /*includeBool=*/false),
               error::kInvalidArgument)
      << "expected the promoted input dtype to be an integer type, got "
      << ToString(common_dtype);
  return absl::OkStatus();
}

// Computes the greatest common divisor of `a` and `b` (which must share the
// same shape and signed 32-bit or 64-bit integer element type).
mlir::MlirOp BuildGcdShlo(mlir::MlirOp a, mlir::MlirOp b,
                          const int32_t max_iters) {
  mlir::MlirBuilder& builder = a.getBuilder();
  auto a_abs = mlir::stablehlo::Abs(a);
  auto b_abs = mlir::stablehlo::Abs(b);
  auto zero_i32 = MakeScalarConstant(builder, 0, mlir::ElementType::I32);

  // Run the Euclidean algorithm (`(a, b) = (b, |a % b|)`) in a
  // `stablehlo.while` loop for a fixed number of iterations (`i < max_iters`).
  // The loop carries the state tuple `(i, a, b)`:
  //   - `args[0]` (`i`): scalar I32 loop counter in `[0, max_iters)`.
  //   - `args[1]` (`a`): current dividend tensor (converges to `gcd(a, b)`).
  //   - `args[2]` (`b`): current divisor/remainder tensor (converges to 0).
  auto loop = mlir::stablehlo::While(
      builder, {zero_i32, a_abs, b_abs},
      [max_iters](mlir::RegionBuilder& cond) {
        // Loop condition: continue while `i < max_iters`.
        auto args = mlir::stablehlo::Arguments(
            cond, cond.getOp<mlir::stablehlo::WhileOp>());
        auto limit =
            MakeScalarConstant(cond, max_iters, mlir::ElementType::I32);
        auto cond_val = mlir::stablehlo::Compare(
            args[0], limit, mlir::stablehlo::ComparisonDirection::LT);
        mlir::stablehlo::Return(cond, cond_val);
      },
      [](mlir::RegionBuilder& body) {
        auto args = mlir::stablehlo::Arguments(
            body, body.getOp<mlir::stablehlo::WhileOp>());
        auto i_val = args[0];
        auto a_val = args[1];
        auto b_val = args[2];

        auto one_i32 = MakeScalarConstant(body, 1, mlir::ElementType::I32);
        auto next_i = mlir::stablehlo::Add(i_val, one_i32);

        // Advance unconverged elements (`b_val != 0`) by one Euclidean step
        // `(next_a, next_b) = (b_val, |a_val % b_val|)` while keeping already
        // converged elements (`b_val == 0`) unchanged at `(a_val, 0)`.
        auto zero = MakeConstantLike(a_val, 0);
        auto one = MakeConstantLike(a_val, 1);
        auto is_nonzero = mlir::stablehlo::Compare(
            b_val, zero, mlir::stablehlo::ComparisonDirection::NE);
        // Substitute 1 for zero divisors before `Rem` to avoid division by
        // zero; the dummy remainder is discarded by `Select` below.
        auto safe_b = mlir::stablehlo::Select(is_nonzero, b_val, one);
        auto rem_signed = mlir::stablehlo::Rem(a_val, safe_b);
        // Take `Abs` of the remainder in case `a_abs` was `INT_MIN` (whose
        // initial `Abs` overflows back to negative in two's complement).
        auto rem = mlir::stablehlo::Abs(rem_signed);
        auto next_a = mlir::stablehlo::Select(is_nonzero, b_val, a_val);
        auto next_b = mlir::stablehlo::Select(is_nonzero, rem, zero);

        mlir::stablehlo::Return(body, {next_i, next_a, next_b});
      });
  // After `max_iters` steps, `b` is 0 everywhere and `loop[1]` (`a`) holds the
  // GCD.
  return loop[1];
}

absl::StatusOr<mlir::MlirOp> BuildLcmShlo(mlir::MlirOp self,
                                          mlir::MlirOp other) {
  TT_ASSIGN_OR_RETURN((auto [lhs, rhs]), ApplyBroadcastIfNeeded(self, other));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType elem_type, GetElementType(lhs));

  // Promote 8-bit and 16-bit integers (i8, ui8, i16) to i32 to match C++
  // standard integral promotion in `std::abs(a / g * b)` and avoid unsigned
  // `stablehlo::Abs` or narrow overflow before `abs`.
  const bool promote_to_i32 = (elem_type == mlir::ElementType::I8 ||
                               elem_type == mlir::ElementType::UI8 ||
                               elem_type == mlir::ElementType::I16);
  if (promote_to_i32) {
    lhs = mlir::stablehlo::ConvertElementType(lhs, mlir::ElementType::I32);
    rhs = mlir::stablehlo::ConvertElementType(rhs, mlir::ElementType::I32);
  }

  const int32_t max_iters = (elem_type == mlir::ElementType::I64)
                                ? kMaxGcdIterationsI64
                                : kMaxGcdIterationsI32;
  auto g = BuildGcdShlo(lhs, rhs, max_iters);

  auto zero = MakeConstantLike(lhs, 0);
  auto one = MakeConstantLike(lhs, 1);
  auto is_g_zero = mlir::stablehlo::Compare(
      g, zero, mlir::stablehlo::ComparisonDirection::EQ);
  auto safe_g = mlir::stablehlo::Select(is_g_zero, one, g);
  auto quotient = mlir::stablehlo::Div(lhs, safe_g);
  auto product = mlir::stablehlo::Mul(quotient, rhs);
  auto abs_product = mlir::stablehlo::Abs(product);
  auto result = mlir::stablehlo::Select(is_g_zero, zero, abs_product);

  if (promote_to_i32) {
    result = mlir::stablehlo::ConvertElementType(result, elem_type);
  }
  return result;
}

}  // namespace

at::Tensor& AtenLcmOut(const at::Tensor& self, const at::Tensor& other,
                       at::Tensor& out) {
  TT_KERNEL(OpName::kLcmOut, _, (self, other, out), {
    TT_THROW_IF_ERROR(ValidateLcmInputs(self, other, out));
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildLcmShlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

}  // namespace torch_tpu
