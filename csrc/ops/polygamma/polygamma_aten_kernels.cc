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

#include "csrc/ops/polygamma/polygamma_aten_kernels.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "c10/util/BFloat16.h"
#include "c10/util/Half.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/digamma/digamma_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/util/BFloat16.h"
#include "torch/headeronly/util/Half.h"

namespace torch_tpu {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Returns the machine epsilon for the given type.
double GetEpsilon(mlir::ElementType type) {
  if (type == mlir::ElementType::F64) {
    return std::numeric_limits<double>::epsilon();
  }
  if (type == mlir::ElementType::F32) {
    return std::numeric_limits<float>::epsilon();
  }
  if (type == mlir::ElementType::F16) {
    return static_cast<double>(std::numeric_limits<c10::Half>::epsilon());
  }
  if (type == mlir::ElementType::BF16) {
    return static_cast<double>(std::numeric_limits<c10::BFloat16>::epsilon());
  }
  return std::numeric_limits<float>::epsilon();  // Default fallback
}

// Raises `base` to an integer exponent `exp` via binary exponentiation,
// avoiding `stablehlo::Pow` which `X64Rewriter` expands into `exp(y * ln(|x|))`
// (~3,000 HLO instructions per call on float64).
[[nodiscard]] mlir::MlirOp BuildPosIntPower(mlir::MlirOp base, uint64_t exp) {
  ABSL_CHECK_GT(exp, uint64_t{0})  // CRASH_OK=BuildZeta passes x_int >= 2.
      << "expected positive exponent, got " << exp
      << ". This is a TorchTPU bug.";
  std::optional<mlir::MlirOp> result;
  mlir::MlirOp power = base;
  while (exp > 0) {
    if ((exp & 1) != 0) {
      result =
          result.has_value() ? mlir::stablehlo::Mul(*result, power) : power;
    }
    exp >>= 1;
    if (exp > 0) {
      power = mlir::stablehlo::Mul(power, power);
    }
  }
  return *result;
}

// Computes the scaled Hurwitz Zeta function `scale^x_int * zeta(x_int, q)` for
// integer order `x_int` (>= 2) and positive `scale` at `q`.
// Absorbing `scale = (n!)^(1 / (n + 1))` into each base `scale / (q + k)` keeps
// intermediate powers within the float32 / TPU DF64 exponent range.
// Reference: xla/hlo/builder/lib/math.cc:Zeta
[[nodiscard]] mlir::MlirOp BuildZeta(const uint64_t x_int, mlir::MlirOp q,
                                     const double scale) {
  const mlir::ElementType computation_type = GetElementTypeOrDie(q);

  // 12-term Euler-Maclaurin asymptotic expansion matching Cephes/XLA Zeta.
  static constexpr int M = 12;
  // 9 additional series terms (q, q + 1, ..., q + 9) before the tail at q + 10.
  static constexpr int N = 9;
  static constexpr int kTwoKMinusOne = 2 * M - 1;

  // (2k)! / B_{2k}, where B_{2k} are the Bernoulli numbers.
  // These are ordered in reverse, for convenience in evaluating the formula
  // by Horner's rule.
  static constexpr std::array<double, M> kZetaCoeffs{
      -7.1661652561756670113e18,
      1.8152105401943546773e17,
      -4.5979787224074726105e15,
      1.1646782814350067249e14,
      -2.950130727918164224e12,
      7.47242496e10,
      -1.8924375803183791606e9,
      47900160.0,
      -1209600.0,
      30240.0,
      -720.0,
      12.0,
  };

  mlir::MlirOp one_like_q = MakeConstantLike(q, 1.0);
  mlir::MlirOp scale_like_q =
      (scale == 1.0) ? one_like_q : MakeConstantLike(q, scale);

  // Calculating S = sum_{k=0}^{N} (scale / (q + k))^x_int using reciprocal
  // integer powers: for x_int >= 2, |scale / (q + k)| overflows only when
  // (scale / (q + k))^x_int already overflows, while avoiding intermediate
  // (q + k)^x_int overflow for large q.
  mlir::MlirOp inv_q = mlir::stablehlo::Div(scale_like_q, q);
  mlir::MlirOp s = BuildPosIntPower(inv_q, x_int);
  for (int i = 1; i <= N; ++i) {
    mlir::MlirOp i_cst = MakeConstantLike(q, static_cast<double>(i));
    mlir::MlirOp q_plus_i = mlir::stablehlo::Add(q, i_cst);
    mlir::MlirOp inv_term = mlir::stablehlo::Div(scale_like_q, q_plus_i);
    mlir::MlirOp term_pow = BuildPosIntPower(inv_term, x_int);
    s = mlir::stablehlo::Add(s, term_pow);
  }
  mlir::MlirOp tail_offset = MakeConstantLike(q, static_cast<double>(N + 1));
  mlir::MlirOp acc = mlir::stablehlo::Add(q, tail_offset);
  mlir::MlirOp inv_acc = mlir::stablehlo::Div(one_like_q, acc);
  mlir::MlirOp scaled_inv_acc =
      (scale == 1.0) ? inv_acc : mlir::stablehlo::Div(scale_like_q, acc);
  mlir::MlirOp neg_power = BuildPosIntPower(scaled_inv_acc, x_int);

  // Calculating I = neg_power * acc / (x_int - 1). Keep `neg_power * acc` so
  // q = +inf produces 0 * inf = NaN for n > 1 matching PyTorch CPU semantics.
  const double x_val = static_cast<double>(x_int);
  mlir::MlirOp i_mul = mlir::stablehlo::Mul(neg_power, acc);
  mlir::MlirOp i_op = i_mul;
  if (x_int != 2) {
    mlir::MlirOp x_minus_one_cst = MakeConstantLike(q, x_val - 1.0);
    i_op = mlir::stablehlo::Div(i_mul, x_minus_one_cst);
  }

  // Calculating T via Horner's rule.
  mlir::MlirOp a_inverse_square = mlir::stablehlo::Mul(inv_acc, inv_acc);
  mlir::MlirOp horner_sum = MakeConstantLike(acc, 0.0);

  for (int i = 0; i < M - 1; ++i) {
    const double factor_val =
        (x_val + static_cast<double>(kTwoKMinusOne - 1 - 2 * i)) *
        (x_val + static_cast<double>(kTwoKMinusOne - 2 - 2 * i));
    mlir::MlirOp factor_cst = MakeConstantLike(acc, factor_val);
    mlir::MlirOp coeff_inv_cst = MakeConstantLike(acc, 1.0 / kZetaCoeffs[i]);
    mlir::MlirOp sum_plus_coeff =
        mlir::stablehlo::Add(horner_sum, coeff_inv_cst);
    mlir::MlirOp horner_mul_1 =
        mlir::stablehlo::Mul(factor_cst, a_inverse_square);
    horner_sum = mlir::stablehlo::Mul(horner_mul_1, sum_plus_coeff);
  }

  mlir::MlirOp point_five = MakeConstantLike(neg_power, 0.5);
  mlir::MlirOp coeff_m_minus_one =
      MakeConstantLike(acc, 1.0 / kZetaCoeffs[M - 1]);
  mlir::MlirOp inner_sum = mlir::stablehlo::Add(coeff_m_minus_one, horner_sum);
  mlir::MlirOp x_cst = MakeConstantLike(acc, x_val);
  mlir::MlirOp x_div_acc = mlir::stablehlo::Mul(x_cst, inv_acc);
  mlir::MlirOp t_mul = mlir::stablehlo::Mul(x_div_acc, inner_sum);
  mlir::MlirOp t_add = mlir::stablehlo::Add(point_five, t_mul);
  mlir::MlirOp t_op = mlir::stablehlo::Mul(neg_power, t_add);

  mlir::MlirOp s_plus_i = mlir::stablehlo::Add(s, i_op);
  mlir::MlirOp accurate_result = mlir::stablehlo::Add(s_plus_i, t_op);

  const double eps_val = GetEpsilon(computation_type);
  mlir::MlirOp eps_cst = MakeConstantLike(q, eps_val);

  mlir::MlirOp abs_neg_power = mlir::stablehlo::Abs(neg_power);
  mlir::MlirOp abs_s = mlir::stablehlo::Abs(s);
  mlir::MlirOp threshold = mlir::stablehlo::Mul(abs_s, eps_cst);
  mlir::MlirOp is_accurate_enough = mlir::stablehlo::Compare(
      abs_neg_power, threshold, mlir::stablehlo::ComparisonDirection::LT);
  // Guard against overflow in both float32 and TPU DF64 emulation, where
  // TwoProd and AddF32ToF64Tuple turn finite-operand overflow above FLT_MAX
  // into NaN instead of +inf. Whenever `q` is finite and non-negative, the
  // scaled zeta sum is positive (or +inf at q == 0), so any non-`< +inf`
  // result (+inf or DF64 overflow NaN) maps to +inf, while `q == +inf` and
  // `q == NaN` preserve their `NaN` results.
  const double overflow_base = std::exp(
      std::log(static_cast<double>(std::numeric_limits<float>::max())) / x_val);
  mlir::MlirOp overflow_base_cst = MakeConstantLike(q, overflow_base);
  mlir::MlirOp leading_term_overflows = mlir::stablehlo::Compare(
      inv_q, overflow_base_cst, mlir::stablehlo::ComparisonDirection::GT);
  mlir::MlirOp zero_like_q = MakeConstantLike(q, 0.0);
  mlir::MlirOp pos_inf =
      MakeConstantLike(q, std::numeric_limits<double>::infinity());
  mlir::MlirOp finite_result =
      mlir::stablehlo::Select(is_accurate_enough, s, accurate_result);
  mlir::MlirOp is_q_ge_zero = mlir::stablehlo::Compare(
      q, zero_like_q, mlir::stablehlo::ComparisonDirection::GE);
  mlir::MlirOp is_q_lt_inf = mlir::stablehlo::Compare(
      q, pos_inf, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp is_q_nonneg_finite =
      mlir::stablehlo::And(is_q_ge_zero, is_q_lt_inf);
  mlir::MlirOp is_result_lt_inf = mlir::stablehlo::Compare(
      finite_result, pos_inf, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp result_not_lt_inf = mlir::stablehlo::Not(is_result_lt_inf);
  mlir::MlirOp is_sum_or_df64_overflow =
      mlir::stablehlo::And(is_q_nonneg_finite, result_not_lt_inf);
  mlir::MlirOp is_overflow =
      mlir::stablehlo::Or(is_sum_or_df64_overflow, leading_term_overflows);
  return mlir::stablehlo::Select(is_overflow, pos_inf, finite_result);
}

// Computes the polygamma function of order n at self_op.
// Reference: xla/hlo/builder/lib/math.cc:Polygamma
absl::StatusOr<mlir::MlirOp> BuildPolygammaShlo(
    mlir::MlirOp self_op, const int64_t n,
    const mlir::ElementType out_mlir_type) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp op,
                      ConvertIfInteger(self_op, out_mlir_type));

  if (n == 0) {  // Digamma
    return BuildDigammaShlo(op, out_mlir_type);
  }

  const mlir::ElementType input_dtype = GetElementTypeOrDie(op);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType computation_type,
                      InferComputationDtype(input_dtype));

  mlir::MlirOp input_op = op;
  if (input_dtype != computation_type) {
    input_op = mlir::stablehlo::ConvertElementType(op, computation_type);
  }

  mlir::MlirOp zero_cst = MakeConstantLike(input_op, 0.0);
  mlir::MlirOp one = MakeConstantLike(input_op, 1.0);
  mlir::MlirOp pos_inf =
      MakeConstantLike(input_op, std::numeric_limits<double>::infinity());
  mlir::MlirOp is_x_neg = mlir::stablehlo::Compare(
      input_op, zero_cst, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp one_minus_x = mlir::stablehlo::Subtract(one, input_op);

  mlir::MlirOp result;
  if (n == 1) {
    // Trigamma (n == 1):
    //   x >= 0: psi^(1)(x) = zeta(2, x)
    //   x < 0:  psi^(1)(x) = pi^2 / sin^2(pi * x) - zeta(2, 1 - x)
    // Keep `pi * input_op` without range reduction and apply reflection for all
    // x < 0 (including negative integers) to match CPU trigamma pole behavior.
    mlir::MlirOp zeta_arg =
        mlir::stablehlo::Select(is_x_neg, one_minus_x, input_op);
    mlir::MlirOp zeta_op = BuildZeta(/*x_int=*/2, zeta_arg, /*scale=*/1.0);

    mlir::MlirOp pi_cst = MakeConstantLike(input_op, kPi);
    mlir::MlirOp pix = mlir::stablehlo::Mul(pi_cst, input_op);
    mlir::MlirOp sin_pix = mlir::stablehlo::Sine(pix);
    mlir::MlirOp sin_pix_sq = mlir::stablehlo::Mul(sin_pix, sin_pix);
    mlir::MlirOp pi_sq_cst = MakeConstantLike(input_op, kPi * kPi);
    mlir::MlirOp csc_sq_term = mlir::stablehlo::Div(pi_sq_cst, sin_pix_sq);
    mlir::MlirOp reflection_op =
        mlir::stablehlo::Subtract(csc_sq_term, zeta_op);

    mlir::MlirOp selected_op =
        mlir::stablehlo::Select(is_x_neg, reflection_op, zeta_op);
    mlir::MlirOp is_pos_inf = mlir::stablehlo::Compare(
        input_op, pos_inf, mlir::stablehlo::ComparisonDirection::EQ);
    result = mlir::stablehlo::Select(is_pos_inf, zero_cst, selected_op);
  } else {
    // For n >= 2, absorb n! into BuildZeta via scale = (n!)^(1 / (n + 1)) so
    // each term is (scale / (q + k))^(n + 1). This avoids intermediate overflow
    // of n! (n >= 35) and underflow of (q + k)^-(n + 1).
    const uint64_t n_plus_one = static_cast<uint64_t>(n) + 1;
    const double n_plus_one_d = static_cast<double>(n_plus_one);
    const double scale = std::exp(std::lgamma(n_plus_one_d) / n_plus_one_d);
    const bool is_n_even = (n % 2 == 0);

    mlir::MlirOp floor_x = mlir::stablehlo::Floor(input_op);
    mlir::MlirOp is_integer = mlir::stablehlo::Compare(
        input_op, floor_x, mlir::stablehlo::ComparisonDirection::EQ);
    mlir::MlirOp is_x_not_int = mlir::stablehlo::Compare(
        input_op, floor_x, mlir::stablehlo::ComparisonDirection::NE);
    mlir::MlirOp use_reflection = mlir::stablehlo::And(is_x_neg, is_x_not_int);

    // For non-integer x < 0, decompose x = floor(x) + r with r in (0, 1) and
    // 1 - r = ceil(x) - x in (0, 1) (using ceil avoids cancellation near 0 and
    // negative integers). Differentiating pi * cot(pi * x) via its partial
    // fraction expansion gives:
    //   psi^(n)(x) = n! * ((zeta(n+1, 1-r) + (-1)^(n+1) * zeta(n+1, r))
    //                      - zeta(n+1, 1-x))
    // Grouping the r / (1-r) terms first cancels identically to 0 at
    // half-integers for even n without high-degree cotangent polynomials.
    mlir::MlirOp r = mlir::stablehlo::Subtract(input_op, floor_x);
    mlir::MlirOp ceil_x = mlir::stablehlo::Ceil(input_op);
    mlir::MlirOp one_minus_r = mlir::stablehlo::Subtract(ceil_x, input_op);

    // Reuse one BuildZeta call for x (direct path) and r (reflection path).
    mlir::MlirOp q_pos = mlir::stablehlo::Select(use_reflection, r, input_op);
    mlir::MlirOp scaled_zeta_q_pos = BuildZeta(n_plus_one, q_pos, scale);
    mlir::MlirOp scaled_zeta_one_minus_r =
        BuildZeta(n_plus_one, one_minus_r, scale);
    mlir::MlirOp scaled_zeta_one_minus_x =
        BuildZeta(n_plus_one, one_minus_x, scale);

    mlir::MlirOp direct_op =
        is_n_even ? mlir::stablehlo::Neg(scaled_zeta_q_pos) : scaled_zeta_q_pos;

    mlir::MlirOp neg_inf =
        MakeConstantLike(input_op, -std::numeric_limits<double>::infinity());
    mlir::MlirOp diff;
    if (is_n_even) {
      mlir::MlirOp raw_diff =
          mlir::stablehlo::Subtract(scaled_zeta_one_minus_r, scaled_zeta_q_pos);
      mlir::MlirOp point_five = MakeConstantLike(input_op, 0.5);
      mlir::MlirOp is_half = mlir::stablehlo::Compare(
          r, point_five, mlir::stablehlo::ComparisonDirection::EQ);
      mlir::MlirOp one_minus_r_is_inf =
          mlir::stablehlo::Compare(scaled_zeta_one_minus_r, pos_inf,
                                   mlir::stablehlo::ComparisonDirection::EQ);
      mlir::MlirOp q_pos_is_inf = mlir::stablehlo::Compare(
          scaled_zeta_q_pos, pos_inf, mlir::stablehlo::ComparisonDirection::EQ);
      mlir::MlirOp both_inf =
          mlir::stablehlo::And(one_minus_r_is_inf, q_pos_is_inf);
      mlir::MlirOp r_gt_half = mlir::stablehlo::Compare(
          r, point_five, mlir::stablehlo::ComparisonDirection::GT);
      mlir::MlirOp inf_diff =
          mlir::stablehlo::Select(r_gt_half, pos_inf, neg_inf);
      // Avoid (+inf) - (+inf) = NaN when both terms overflow, and preserve
      // exact zero cancellation at r == 0.5 even under overflow.
      mlir::MlirOp diff_no_nan =
          mlir::stablehlo::Select(both_inf, inf_diff, raw_diff);
      diff = mlir::stablehlo::Select(is_half, zero_cst, diff_no_nan);
    } else {
      // Normalize finite-operand DF64 sum overflow (NaN) to +inf.
      mlir::MlirOp raw_sum =
          mlir::stablehlo::Add(scaled_zeta_one_minus_r, scaled_zeta_q_pos);
      mlir::MlirOp is_sum_lt_inf = mlir::stablehlo::Compare(
          raw_sum, pos_inf, mlir::stablehlo::ComparisonDirection::LT);
      mlir::MlirOp sum_not_lt_inf = mlir::stablehlo::Not(is_sum_lt_inf);
      diff = mlir::stablehlo::Select(sum_not_lt_inf, pos_inf, raw_sum);
    }

    // When diff overflows to +inf, it dominates scaled_zeta(n+1, 1-x) because
    // 1-r < 1 < 1-x; guard against (+inf) - (+inf) = NaN.
    mlir::MlirOp raw_reflection =
        mlir::stablehlo::Subtract(diff, scaled_zeta_one_minus_x);
    mlir::MlirOp diff_is_pos_inf = mlir::stablehlo::Compare(
        diff, pos_inf, mlir::stablehlo::ComparisonDirection::EQ);
    mlir::MlirOp reflection_op =
        mlir::stablehlo::Select(diff_is_pos_inf, pos_inf, raw_reflection);
    if (is_n_even) {
      // When diff <= 0, subtracting scaled_zeta(n+1, 1-x) >= 0 can only
      // overflow toward -inf; map DF64 overflow NaN to -inf.
      mlir::MlirOp is_diff_le_zero = mlir::stablehlo::Compare(
          diff, zero_cst, mlir::stablehlo::ComparisonDirection::LE);
      mlir::MlirOp is_refl_gt_neg_inf = mlir::stablehlo::Compare(
          reflection_op, neg_inf, mlir::stablehlo::ComparisonDirection::GT);
      mlir::MlirOp refl_not_gt_neg_inf =
          mlir::stablehlo::Not(is_refl_gt_neg_inf);
      mlir::MlirOp neg_overflow =
          mlir::stablehlo::And(is_diff_le_zero, refl_not_gt_neg_inf);
      reflection_op =
          mlir::stablehlo::Select(neg_overflow, neg_inf, reflection_op);
    }

    mlir::MlirOp selected_op =
        mlir::stablehlo::Select(use_reflection, reflection_op, direct_op);

    // Polygamma (n > 1): returns -inf (even n) or +inf (odd n) at non-positive
    // integers and -inf. Check -inf explicitly because TPU DF64 Floor(-inf)
    // produces NaN in QuickTwoSum, making `input == Floor(input)` false.
    mlir::MlirOp is_neg_inf = mlir::stablehlo::Compare(
        input_op, neg_inf, mlir::stablehlo::ComparisonDirection::EQ);
    mlir::MlirOp is_le_zero = mlir::stablehlo::Compare(
        input_op, zero_cst, mlir::stablehlo::ComparisonDirection::LE);
    mlir::MlirOp is_nonpos_int = mlir::stablehlo::And(is_le_zero, is_integer);
    mlir::MlirOp is_pole = mlir::stablehlo::Or(is_nonpos_int, is_neg_inf);
    mlir::MlirOp pole_cst = is_n_even ? neg_inf : pos_inf;
    result = mlir::stablehlo::Select(is_pole, pole_cst, selected_op);
  }

  if (GetElementTypeOrDie(result) != out_mlir_type) {
    return mlir::stablehlo::ConvertElementType(result, out_mlir_type);
  }
  return result;
}

}  // namespace

at::Tensor& AtenPolygammaOut(int64_t n, const at::Tensor& self,
                             at::Tensor& out) {
  TT_KERNEL(OpName::kPolygammaOut, param_keys, (n, self, out), {
    TT_CHECK_THROW(n >= 0, error::kInvalidArgument)
        << "expected n to be >= 0, got " << n;
    TT_CHECK_THROW(!IsComplex(self), error::kInvalidArgument)
        << "expected the input dtype not to be complex, got "
        << ToString(self.scalar_type());

    // Enforce safe type promotion (casting to integer outputs is not safe)
    TT_CHECK_THROW(out.is_floating_point() || out.is_complex(),
                   error::kInvalidArgument)
        << "expected the output dtype to be floating point or complex, got "
        << ToString(out.scalar_type());

    TT_ASSIGN_OR_THROW(const auto out_dtype,
                       ConvertTo<mlir::ElementType>(InferOutputDtype(self)));

    auto op_builder =
        [n, out_dtype](mlir::MlirOp self_op) -> absl::StatusOr<mlir::MlirOp> {
      return BuildPolygammaShlo(self_op, n, out_dtype);
    };

    TT_THROW_IF_ERROR(UnaryOpOut(self, out, std::move(op_builder),
                                 {.op_param_cache_keys = std::move(param_keys),
                                  .out_dtype = out_dtype}));
    return out;
  });
}
}  // namespace torch_tpu
