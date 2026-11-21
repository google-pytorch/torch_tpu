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
#include "mlir/Support/LLVM.h"
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
// Number of quadratic Hensel/Newton doubling steps to lift a 5-bit modular
// inverse `(3 * odd) ^ 2` (mod 32) to 64 bits (`5 -> 10 -> 20 -> 40 -> 80`).
constexpr int kNumModularInverseStepsU64 = 4;

absl::Status ValidateLcmInputs(const at::Tensor& self, const at::Tensor& other,
                               const at::Tensor& out) {
  const at::ScalarType common_dtype = at::result_type(self, other);
  TT_RETURN_IF_ERROR(ValidateOutDtype(out, common_dtype));
  TT_RET_CHECK(c10::isIntegralType(common_dtype, /*includeBool=*/false),
               error::kInvalidArgument)
      << "expected the promoted input dtype to be an integer type, got "
      << ToString(common_dtype);
  TT_RET_CHECK(common_dtype == at::kByte || common_dtype == at::kChar ||
                   common_dtype == at::kShort || common_dtype == at::kInt ||
                   common_dtype == at::kLong,
               error::kInvalidArgument)
      << "expected int8, uint8, int16, int32 or int64 inputs, got "
      << ToString(common_dtype);
  return absl::OkStatus();
}

// Computes the greatest common divisor of `a` and `b` (which must share the
// same shape and signed 32-bit integer element type).
mlir::MlirOp BuildGcdShlo(mlir::MlirOp a, mlir::MlirOp b) {
  mlir::MlirBuilder& builder = a.getBuilder();
  mlir::MlirOp a_abs = mlir::stablehlo::Abs(a);
  mlir::MlirOp b_abs = mlir::stablehlo::Abs(b);
  mlir::MlirOp zero_i32 =
      MakeScalarConstant(builder, 0, mlir::ElementType::I32);

  // Run the Euclidean algorithm (`(a, b) = (b, |a % b|)`) in a
  // `stablehlo.while` loop for a fixed number of iterations
  // (`i < kMaxGcdIterationsI32`).
  // The loop carries the state tuple `(i, a, b)`:
  //   - `args[0]` (`i`): scalar I32 loop counter in `[0,
  //   kMaxGcdIterationsI32)`.
  //   - `args[1]` (`a`): current dividend tensor (converges to `gcd(a, b)`).
  //   - `args[2]` (`b`): current divisor/remainder tensor (converges to 0).
  mlir::SmallVector<mlir::MlirOp> loop = mlir::stablehlo::While(
      builder, {zero_i32, a_abs, b_abs},
      [](mlir::RegionBuilder& cond) {
        // Loop condition: continue while `i < kMaxGcdIterationsI32`.
        mlir::SmallVector<mlir::MlirOp> args = mlir::stablehlo::Arguments(
            cond, cond.getOp<mlir::stablehlo::WhileOp>());
        mlir::MlirOp limit = MakeScalarConstant(cond, kMaxGcdIterationsI32,
                                                mlir::ElementType::I32);
        mlir::MlirOp cond_val = mlir::stablehlo::Compare(
            args[0], limit, mlir::stablehlo::ComparisonDirection::LT);
        mlir::stablehlo::Return(cond, cond_val);
      },
      [](mlir::RegionBuilder& body) {
        mlir::SmallVector<mlir::MlirOp> args = mlir::stablehlo::Arguments(
            body, body.getOp<mlir::stablehlo::WhileOp>());
        mlir::MlirOp i_val = args[0];
        mlir::MlirOp a_val = args[1];
        mlir::MlirOp b_val = args[2];

        mlir::MlirOp one_i32 =
            MakeScalarConstant(body, 1, mlir::ElementType::I32);
        mlir::MlirOp next_i = mlir::stablehlo::Add(i_val, one_i32);

        // Advance unconverged elements (`b_val != 0`) by one Euclidean step
        // `(next_a, next_b) = (b_val, |a_val % b_val|)` while keeping already
        // converged elements (`b_val == 0`) unchanged at `(a_val, 0)`.
        mlir::MlirOp zero = MakeConstantLike(a_val, 0);
        mlir::MlirOp one = MakeConstantLike(a_val, 1);
        mlir::MlirOp is_nonzero = mlir::stablehlo::Compare(
            b_val, zero, mlir::stablehlo::ComparisonDirection::NE);
        // Substitute 1 for zero divisors before `Rem` to avoid division by
        // zero; the dummy remainder is discarded by `Select` below.
        mlir::MlirOp safe_b = mlir::stablehlo::Select(is_nonzero, b_val, one);
        mlir::MlirOp rem_signed = mlir::stablehlo::Rem(a_val, safe_b);
        // Take `Abs` of the remainder in case `a_abs` was `INT_MIN` (whose
        // initial `Abs` overflows back to negative in two's complement).
        mlir::MlirOp rem = mlir::stablehlo::Abs(rem_signed);
        mlir::MlirOp next_a = mlir::stablehlo::Select(is_nonzero, b_val, a_val);
        mlir::MlirOp next_b = mlir::stablehlo::Select(is_nonzero, rem, zero);

        mlir::stablehlo::Return(body, {next_i, next_a, next_b});
      });
  // After `kMaxGcdIterationsI32` steps, `b` is 0 everywhere and `loop[1]` (`a`)
  // holds the GCD.
  return loop[1];
}

// Counts trailing zero bits of `UI64` tensor `x` via `popcnt((x & -x) - 1)`.
mlir::MlirOp BuildCtzU64(mlir::MlirOp x) {
  mlir::MlirOp zero = MakeConstantLike(x, 0);
  mlir::MlirOp one = MakeConstantLike(x, 1);
  mlir::MlirOp neg_x = mlir::stablehlo::Subtract(zero, x);
  mlir::MlirOp lsb = mlir::stablehlo::And(x, neg_x);
  mlir::MlirOp mask = mlir::stablehlo::Subtract(lsb, one);
  return mlir::stablehlo::PopulationCount(mask);
}

// Decomposition `gcd(u0, v0) = odd_factor << shift` for positive `UI64` inputs,
// where `odd_factor` is odd and `shift` is the shared power-of-two exponent.
struct BinaryGcdU64 {
  mlir::MlirOp odd_factor;
  mlir::MlirOp shift;
};

// Computes unsigned 64-bit GCD on strictly positive inputs (`u0, v0 >= 1`)
// using Stein's binary GCD algorithm so TPU legalization avoids expanding
// 64-bit `stablehlo.remainder` into 64 unrolled division stages.
BinaryGcdU64 BuildBinaryGcdPositiveU64(mlir::MlirOp u0, mlir::MlirOp v0) {
  mlir::MlirBuilder& builder = u0.getBuilder();
  mlir::MlirOp uv_or = mlir::stablehlo::Or(u0, v0);
  mlir::MlirOp shift = BuildCtzU64(uv_or);
  mlir::MlirOp u0_ctz = BuildCtzU64(u0);
  mlir::MlirOp u_init = mlir::stablehlo::ShiftRightLogical(u0, u0_ctz);
  mlir::MlirOp zero_i32 =
      MakeScalarConstant(builder, 0, mlir::ElementType::I32);

  mlir::SmallVector<mlir::MlirOp> loop = mlir::stablehlo::While(
      builder, {zero_i32, u_init, v0},
      [](mlir::RegionBuilder& cond) {
        mlir::SmallVector<mlir::MlirOp> args = mlir::stablehlo::Arguments(
            cond, cond.getOp<mlir::stablehlo::WhileOp>());
        mlir::MlirOp limit = MakeScalarConstant(
            cond, kMaxBinaryGcdIterationsI64, mlir::ElementType::I32);
        mlir::MlirOp below_limit = mlir::stablehlo::Compare(
            args[0], limit, mlir::stablehlo::ComparisonDirection::LT);
        mlir::stablehlo::Return(cond, below_limit);
      },
      [](mlir::RegionBuilder& body) {
        mlir::SmallVector<mlir::MlirOp> args = mlir::stablehlo::Arguments(
            body, body.getOp<mlir::stablehlo::WhileOp>());
        mlir::MlirOp i = args[0];
        mlir::MlirOp u = args[1];
        mlir::MlirOp v = args[2];

        mlir::MlirOp one_i32 =
            MakeScalarConstant(body, 1, mlir::ElementType::I32);
        mlir::MlirOp next_i = mlir::stablehlo::Add(i, one_i32);

        mlir::MlirOp zero = MakeConstantLike(u, 0);
        mlir::MlirOp one = MakeConstantLike(u, 1);
        mlir::MlirOp is_nonzero = GetNonzeroMask(v);
        mlir::MlirOp safe_v = mlir::stablehlo::Select(is_nonzero, v, one);
        mlir::MlirOp v_ctz = BuildCtzU64(safe_v);
        mlir::MlirOp v_odd = mlir::stablehlo::ShiftRightLogical(safe_v, v_ctz);
        mlir::MlirOp min_uv = mlir::stablehlo::Min(u, v_odd);
        mlir::MlirOp max_uv = mlir::stablehlo::Max(u, v_odd);
        mlir::MlirOp diff = mlir::stablehlo::Subtract(max_uv, min_uv);
        mlir::MlirOp next_u = mlir::stablehlo::Select(is_nonzero, min_uv, u);
        mlir::MlirOp next_v = mlir::stablehlo::Select(is_nonzero, diff, zero);

        mlir::stablehlo::Return(body, {next_i, next_u, next_v});
      });
  return BinaryGcdU64{.odd_factor = loop[1], .shift = shift};
}

// Computes the modular inverse of odd `UI64` tensor `odd` modulo 2^64 using
// standard Newton/Hensel lifting.
mlir::MlirOp BuildModularInverseOddU64(mlir::MlirOp odd) {
  mlir::MlirOp two = MakeConstantLike(odd, 2);
  mlir::MlirOp three = MakeConstantLike(odd, 3);
  mlir::MlirOp three_times_odd = mlir::stablehlo::Mul(three, odd);
  mlir::MlirOp inv = mlir::stablehlo::Xor(three_times_odd, two);
  for (int step = 0; step < kNumModularInverseStepsU64; ++step) {
    mlir::MlirOp odd_times_inv = mlir::stablehlo::Mul(odd, inv);
    mlir::MlirOp correction = mlir::stablehlo::Subtract(two, odd_times_inv);
    inv = mlir::stablehlo::Mul(inv, correction);
  }
  return inv;
}

// Computes signed 64-bit LCM of broadcast-matched `I64` tensors `lhs` and `rhs`
// without emitting 64-bit `stablehlo.remainder` or `stablehlo.divide`. Returns
// 0 where either input is 0, and converts the unsigned product back to `I64`
// before `Abs` to match PyTorch's two's-complement `int64` wrapping (e.g.
// `lcm(INT64_MIN, 1) == INT64_MIN`).
mlir::MlirOp BuildLcmI64Shlo(mlir::MlirOp lhs, mlir::MlirOp rhs) {
  mlir::MlirOp zero_i64 = MakeConstantLike(lhs, 0);
  mlir::MlirOp lhs_zero = mlir::stablehlo::Compare(
      lhs, zero_i64, mlir::stablehlo::ComparisonDirection::EQ);
  mlir::MlirOp rhs_zero = mlir::stablehlo::Compare(
      rhs, zero_i64, mlir::stablehlo::ComparisonDirection::EQ);
  mlir::MlirOp any_zero = mlir::stablehlo::Or(lhs_zero, rhs_zero);

  // Convert `|lhs|` and `|rhs|` to `UI64` (preserving `|INT64_MIN| == 2^63` in
  // two's complement) and substitute 1 on zero lanes.
  mlir::MlirOp u_raw = mlir::stablehlo::ConvertElementType(
      mlir::stablehlo::Abs(lhs), mlir::ElementType::UI64);
  mlir::MlirOp v_raw = mlir::stablehlo::ConvertElementType(
      mlir::stablehlo::Abs(rhs), mlir::ElementType::UI64);
  mlir::MlirOp one_u64 = MakeConstantLike(u_raw, 1);
  mlir::MlirOp u_pos = mlir::stablehlo::Select(any_zero, one_u64, u_raw);
  mlir::MlirOp v_pos = mlir::stablehlo::Select(any_zero, one_u64, v_raw);

  // `lcm(u_pos, v_pos) = ((u_pos >> shift) * inv(odd_factor)) * v_pos`.
  BinaryGcdU64 gcd = BuildBinaryGcdPositiveU64(u_pos, v_pos);
  mlir::MlirOp u_shifted = mlir::stablehlo::ShiftRightLogical(u_pos, gcd.shift);
  mlir::MlirOp inv_odd = BuildModularInverseOddU64(gcd.odd_factor);
  mlir::MlirOp quotient_u64 = mlir::stablehlo::Mul(u_shifted, inv_odd);
  mlir::MlirOp product_u64 = mlir::stablehlo::Mul(quotient_u64, v_pos);
  mlir::MlirOp product_i64 =
      mlir::stablehlo::ConvertElementType(product_u64, mlir::ElementType::I64);
  mlir::MlirOp abs_product = mlir::stablehlo::Abs(product_i64);
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

  mlir::MlirOp g = BuildGcdShlo(lhs, rhs);
  mlir::MlirOp zero = MakeConstantLike(lhs, 0);
  mlir::MlirOp one = MakeConstantLike(lhs, 1);
  mlir::MlirOp is_g_zero = mlir::stablehlo::Compare(
      g, zero, mlir::stablehlo::ComparisonDirection::EQ);
  mlir::MlirOp safe_g = mlir::stablehlo::Select(is_g_zero, one, g);
  mlir::MlirOp quotient = mlir::stablehlo::Div(lhs, safe_g);
  mlir::MlirOp product = mlir::stablehlo::Mul(quotient, rhs);
  mlir::MlirOp abs_product = mlir::stablehlo::Abs(product);
  mlir::MlirOp result = mlir::stablehlo::Select(is_g_zero, zero, abs_product);

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
