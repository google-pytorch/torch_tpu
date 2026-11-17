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

// Upper bound on 32-bit Euclidean GCD iterations by Lamé's theorem
// (<= 2^31 < F_47: <= 45 division steps + 1 swap).
constexpr int32_t kMaxGcdIterationsI32 = 47;
// Binary GCD bound on positive inputs in [1, 2^63] (at most 63 halving steps).
constexpr int32_t kMaxBinaryGcdIterationsI64 = 65;
constexpr int32_t kBitsPerInt64 = 64;

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
// same shape and signed 32-bit integer element type).
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

// Counts trailing zero bits of `UI64` tensor `x` via `popcnt((x & -x) - 1)`.
mlir::MlirOp BuildCtzU64(mlir::MlirOp x) {
  auto zero = MakeConstantLike(x, 0);
  auto one = MakeConstantLike(x, 1);
  auto neg_x = mlir::stablehlo::Subtract(zero, x);
  auto lsb = mlir::stablehlo::And(x, neg_x);
  auto mask = mlir::stablehlo::Subtract(lsb, one);
  return mlir::stablehlo::PopulationCount(mask);
}

// Computes unsigned 64-bit GCD on strictly positive inputs (`u0, v0 >= 1`)
// using Stein's binary GCD algorithm so TPU legalization avoids expanding
// 64-bit `stablehlo.remainder` into 64 unrolled division stages.
mlir::MlirOp BuildBinaryGcdPositiveU64(mlir::MlirOp u0, mlir::MlirOp v0) {
  mlir::MlirBuilder& builder = u0.getBuilder();
  auto uv_or = mlir::stablehlo::Or(u0, v0);
  auto shift = BuildCtzU64(uv_or);
  auto u0_ctz = BuildCtzU64(u0);
  auto u_init = mlir::stablehlo::ShiftRightLogical(u0, u0_ctz);
  auto zero_i32 = MakeScalarConstant(builder, 0, mlir::ElementType::I32);

  auto loop = mlir::stablehlo::While(
      builder, {zero_i32, u_init, v0},
      [](mlir::RegionBuilder& cond) {
        auto args = mlir::stablehlo::Arguments(
            cond, cond.getOp<mlir::stablehlo::WhileOp>());
        auto limit = MakeScalarConstant(cond, kMaxBinaryGcdIterationsI64,
                                        mlir::ElementType::I32);
        auto below_limit = mlir::stablehlo::Compare(
            args[0], limit, mlir::stablehlo::ComparisonDirection::LT);
        mlir::stablehlo::Return(cond, below_limit);
      },
      [](mlir::RegionBuilder& body) {
        auto args = mlir::stablehlo::Arguments(
            body, body.getOp<mlir::stablehlo::WhileOp>());
        auto i = args[0];
        auto u = args[1];
        auto v = args[2];

        auto one_i32 = MakeScalarConstant(body, 1, mlir::ElementType::I32);
        auto next_i = mlir::stablehlo::Add(i, one_i32);

        auto zero = MakeConstantLike(u, 0);
        auto one = MakeConstantLike(u, 1);
        auto is_nonzero = GetNonzeroMask(v);
        auto safe_v = mlir::stablehlo::Select(is_nonzero, v, one);
        auto v_ctz = BuildCtzU64(safe_v);
        auto v_odd = mlir::stablehlo::ShiftRightLogical(safe_v, v_ctz);
        auto min_uv = mlir::stablehlo::Min(u, v_odd);
        auto max_uv = mlir::stablehlo::Max(u, v_odd);
        auto diff = mlir::stablehlo::Subtract(max_uv, min_uv);
        auto next_u = mlir::stablehlo::Select(is_nonzero, min_uv, u);
        auto next_v = mlir::stablehlo::Select(is_nonzero, diff, zero);

        mlir::stablehlo::Return(body, {next_i, next_u, next_v});
      });
  auto g_odd = loop[1];
  return mlir::stablehlo::ShiftLeft(g_odd, shift);
}

// Computes unsigned 64-bit integer quotient `dividend / divisor` (`divisor >
// 0`) in a 64-step `stablehlo.while` loop instead of `stablehlo.divide` so XLA
// does not inline 64 unrolled 64-bit shift-subtract stages into the HLO graph.
mlir::MlirOp BuildUnsignedDivU64(mlir::MlirOp dividend, mlir::MlirOp divisor) {
  mlir::MlirBuilder& builder = dividend.getBuilder();
  auto zero_i32 = MakeScalarConstant(builder, 0, mlir::ElementType::I32);
  auto zero_u64 = MakeConstantLike(dividend, 0);

  auto loop = mlir::stablehlo::While(
      builder, {zero_i32, dividend, zero_u64, divisor},
      [](mlir::RegionBuilder& cond) {
        auto args = mlir::stablehlo::Arguments(
            cond, cond.getOp<mlir::stablehlo::WhileOp>());
        auto limit =
            MakeScalarConstant(cond, kBitsPerInt64, mlir::ElementType::I32);
        auto below_limit = mlir::stablehlo::Compare(
            args[0], limit, mlir::stablehlo::ComparisonDirection::LT);
        mlir::stablehlo::Return(cond, below_limit);
      },
      [](mlir::RegionBuilder& body) {
        auto args = mlir::stablehlo::Arguments(
            body, body.getOp<mlir::stablehlo::WhileOp>());
        auto i = args[0];
        auto q = args[1];
        auto r = args[2];
        auto d = args[3];

        auto one_i32 = MakeScalarConstant(body, 1, mlir::ElementType::I32);
        auto next_i = mlir::stablehlo::Add(i, one_i32);

        auto one_u64 = MakeConstantLike(q, 1);
        auto msb_shift = MakeConstantLike(q, kBitsPerInt64 - 1);
        auto q_msb = mlir::stablehlo::ShiftRightLogical(q, msb_shift);
        auto r_shl = mlir::stablehlo::ShiftLeft(r, one_u64);
        auto r_shifted = mlir::stablehlo::Or(r_shl, q_msb);
        auto q_shifted = mlir::stablehlo::ShiftLeft(q, one_u64);

        auto ge = mlir::stablehlo::Compare(
            r_shifted, d, mlir::stablehlo::ComparisonDirection::GE,
            mlir::stablehlo::ComparisonTypeAttr::get(
                &d.getContext(), mlir::stablehlo::ComparisonType::UNSIGNED));
        auto r_sub = mlir::stablehlo::Subtract(r_shifted, d);
        auto next_r = mlir::stablehlo::Select(ge, r_sub, r_shifted);
        auto q_inc = mlir::stablehlo::Or(q_shifted, one_u64);
        auto next_q = mlir::stablehlo::Select(ge, q_inc, q_shifted);

        mlir::stablehlo::Return(body, {next_i, next_q, next_r, d});
      });
  return loop[1];
}

mlir::MlirOp BuildLcmI64Shlo(mlir::MlirOp lhs, mlir::MlirOp rhs) {
  auto zero_i64 = MakeConstantLike(lhs, 0);
  auto lhs_zero = mlir::stablehlo::Compare(
      lhs, zero_i64, mlir::stablehlo::ComparisonDirection::EQ);
  auto rhs_zero = mlir::stablehlo::Compare(
      rhs, zero_i64, mlir::stablehlo::ComparisonDirection::EQ);
  auto any_zero = mlir::stablehlo::Or(lhs_zero, rhs_zero);

  // Convert `|lhs|` and `|rhs|` to `UI64` (preserving `|INT64_MIN| == 2^63` in
  // two's complement) and substitute 1 on zero lanes.
  auto u_raw = mlir::stablehlo::ConvertElementType(mlir::stablehlo::Abs(lhs),
                                                   mlir::ElementType::UI64);
  auto v_raw = mlir::stablehlo::ConvertElementType(mlir::stablehlo::Abs(rhs),
                                                   mlir::ElementType::UI64);
  auto one_u64 = MakeConstantLike(u_raw, 1);
  auto u_pos = mlir::stablehlo::Select(any_zero, one_u64, u_raw);
  auto v_pos = mlir::stablehlo::Select(any_zero, one_u64, v_raw);

  auto g_u64 = BuildBinaryGcdPositiveU64(u_pos, v_pos);
  auto quotient_u64 = BuildUnsignedDivU64(u_pos, g_u64);
  auto product_u64 = mlir::stablehlo::Mul(quotient_u64, v_pos);
  auto product_i64 =
      mlir::stablehlo::ConvertElementType(product_u64, mlir::ElementType::I64);
  auto abs_product = mlir::stablehlo::Abs(product_i64);
  return mlir::stablehlo::Select(any_zero, zero_i64, abs_product);
}

absl::StatusOr<mlir::MlirOp> BuildLcmShlo(mlir::MlirOp self,
                                          mlir::MlirOp other) {
  TT_ASSIGN_OR_RETURN((auto [lhs, rhs]), ApplyBroadcastIfNeeded(self, other));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType elem_type, GetElementType(lhs));
  if (elem_type == mlir::ElementType::I64) {
    return BuildLcmI64Shlo(lhs, rhs);
  }

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

  auto g = BuildGcdShlo(lhs, rhs, kMaxGcdIterationsI32);
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
