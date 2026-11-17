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
#include <vector>

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

std::vector<double> ComputeCotDerivativeCoeffs(int64_t n) {
  if (n == 0) return {0.0, 1.0};

  std::vector<double> prev = ComputeCotDerivativeCoeffs(n - 1);
  std::vector<double> coeffs(n + 2, 0.0);

  coeffs[0] = -prev[1];
  if (prev.size() > 2) {
    coeffs[1] = -2.0 * prev[2];
  }
  for (int64_t k = 2; k <= n + 1; ++k) {
    double term1 = 0.0;
    if (k + 1 < prev.size()) {
      term1 = -(k + 1) * prev[k + 1];
    }
    double term2 = -(k - 1) * prev[k - 1];
    coeffs[k] = term1 + term2;
  }
  return coeffs;
}

mlir::MlirOp EvaluatePolynomial(mlir::MlirOp y,
                                const std::vector<double>& coeffs) {
  auto accum = MakeConstantLike(y, coeffs.back());
  for (int i = static_cast<int>(coeffs.size()) - 2; i >= 0; --i) {
    accum = mlir::stablehlo::Mul(accum, y);
    auto c = MakeConstantLike(y, coeffs[i]);
    accum = mlir::stablehlo::Add(accum, c);
  }
  return accum;
}

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

// Computes the Hurwitz Zeta function for integer order `x_int` (>= 2) at `q`.
// Reference: xla/hlo/builder/lib/math.cc:Zeta
[[nodiscard]] mlir::MlirOp BuildZeta(const uint64_t x_int, mlir::MlirOp q) {
  const mlir::ElementType computation_type = GetElementTypeOrDie(q);

  // 12-term Euler-Maclaurin asymptotic expansion matching Cephes/XLA Zeta.
  static constexpr int M = 12;
  // 9 additional series terms (q, q + 1, ..., q + 9) before the tail at q + 10.
  static constexpr int N = 9;
  static constexpr int kTwoKMinusOne = 2 * M - 1;

  // (2k)! / B_{2k}, where B_{2k} are the Bernoulli numbers.
  // These are ordered in reverse, for convenience in evaluating the formula
  // by Horner's rule.
  static const std::array<double, M> kZetaCoeffs{
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

  auto one_like_q = MakeConstantLike(q, 1.0);

  // Calculating S = sum_{k=0}^{N} (q + k)^(-x_int) using reciprocal integer
  // powers: for x_int >= 2, |1 / (q + k)| overflows only when (q + k)^(-x_int)
  // already overflows, while avoiding intermediate (q + k)^x_int overflow for
  // large q.
  auto inv_q = mlir::stablehlo::Div(one_like_q, q);
  auto s = BuildPosIntPower(inv_q, x_int);
  for (int i = 1; i <= N; ++i) {
    auto i_cst = MakeConstantLike(q, static_cast<double>(i));
    auto q_plus_i = mlir::stablehlo::Add(q, i_cst);
    auto inv_term = mlir::stablehlo::Div(one_like_q, q_plus_i);
    auto term_pow = BuildPosIntPower(inv_term, x_int);
    s = mlir::stablehlo::Add(s, term_pow);
  }
  auto tail_offset = MakeConstantLike(q, static_cast<double>(N + 1));
  auto acc = mlir::stablehlo::Add(q, tail_offset);
  auto inv_acc = mlir::stablehlo::Div(one_like_q, acc);
  auto neg_power = BuildPosIntPower(inv_acc, x_int);

  // Calculating I = neg_power * acc / (x_int - 1). Keep `neg_power * acc` so
  // q = +inf produces 0 * inf = NaN for n > 1 matching PyTorch CPU semantics.
  const double x_val = static_cast<double>(x_int);
  auto i_mul = mlir::stablehlo::Mul(neg_power, acc);
  mlir::MlirOp i_op = i_mul;
  if (x_int != 2) {
    auto x_minus_one_cst = MakeConstantLike(q, x_val - 1.0);
    i_op = mlir::stablehlo::Div(i_mul, x_minus_one_cst);
  }

  // Calculating T via Horner's rule.
  auto a_inverse_square = mlir::stablehlo::Mul(inv_acc, inv_acc);
  auto horner_sum = MakeConstantLike(acc, 0.0);

  for (int i = 0; i < M - 1; ++i) {
    const double factor_val =
        (x_val + static_cast<double>(kTwoKMinusOne - 1 - 2 * i)) *
        (x_val + static_cast<double>(kTwoKMinusOne - 2 - 2 * i));
    auto factor_cst = MakeConstantLike(acc, factor_val);
    auto coeff_inv_cst = MakeConstantLike(acc, 1.0 / kZetaCoeffs[i]);
    auto sum_plus_coeff = mlir::stablehlo::Add(horner_sum, coeff_inv_cst);
    auto horner_mul_1 = mlir::stablehlo::Mul(factor_cst, a_inverse_square);
    horner_sum = mlir::stablehlo::Mul(horner_mul_1, sum_plus_coeff);
  }

  auto point_five = MakeConstantLike(neg_power, 0.5);
  auto coeff_m_minus_one = MakeConstantLike(acc, 1.0 / kZetaCoeffs[M - 1]);
  auto inner_sum = mlir::stablehlo::Add(coeff_m_minus_one, horner_sum);
  auto x_cst = MakeConstantLike(acc, x_val);
  auto x_div_acc = mlir::stablehlo::Mul(x_cst, inv_acc);
  auto t_mul = mlir::stablehlo::Mul(x_div_acc, inner_sum);
  auto t_add = mlir::stablehlo::Add(point_five, t_mul);
  auto t_op = mlir::stablehlo::Mul(neg_power, t_add);

  auto s_plus_i = mlir::stablehlo::Add(s, i_op);
  auto accurate_result = mlir::stablehlo::Add(s_plus_i, t_op);

  const double eps_val = GetEpsilon(computation_type);
  auto eps_cst = MakeConstantLike(q, eps_val);

  auto abs_neg_power = mlir::stablehlo::Abs(neg_power);
  auto abs_s = mlir::stablehlo::Abs(s);
  auto threshold = mlir::stablehlo::Mul(abs_s, eps_cst);
  auto is_accurate_enough = mlir::stablehlo::Compare(
      abs_neg_power, threshold, mlir::stablehlo::ComparisonDirection::LT);
  return mlir::stablehlo::Select(is_accurate_enough, s, accurate_result);
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

  const uint64_t n_plus_one = static_cast<uint64_t>(n) + 1;

  // 1. Direct path (x >= 0.0)
  // psi^(n)(x) = (-1)^(n+1) * n! * zeta(n+1, x)
  const double sign_val = (n % 2 == 0) ? -1.0 : 1.0;
  const double coeff_val = sign_val * std::tgamma(n + 1);
  auto coeff_cst = MakeConstantLike(input_op, coeff_val);
  mlir::MlirOp zeta_op = BuildZeta(n_plus_one, input_op);
  auto direct_op = mlir::stablehlo::Mul(coeff_cst, zeta_op);

  // 2. Reflection path (x < 0.0)
  // psi^(n)(x) = (-1)^n * psi^(n)(1-x) - pi^(n+1) * d^n/dx^n (cot(pi x))
  auto one = MakeConstantLike(input_op, 1.0);
  auto one_minus_x = mlir::stablehlo::Subtract(one, input_op);
  mlir::MlirOp zeta_1_minus_x = BuildZeta(n_plus_one, one_minus_x);

  // First term: (-1)^n * psi^(n)(1-x) = -n! * zeta(n+1, 1-x)
  double first_term_coeff = -std::tgamma(n + 1);
  auto first_term_coeff_cst = MakeConstantLike(input_op, first_term_coeff);
  auto first_term = mlir::stablehlo::Mul(first_term_coeff_cst, zeta_1_minus_x);

  // Second term: -pi^(n+1) * poly_op
  auto pi_cst = MakeConstantLike(input_op, kPi);
  mlir::MlirOp pix;
  if (n == 1) {
    // Disabling range reduction for n=1
    pix = mlir::stablehlo::Mul(pi_cst, input_op);
  } else {
    // Shift x to [-0.5, 0.5] range to avoid precision loss in trig reduction.
    auto point_five_cst = MakeConstantLike(input_op, 0.5);
    auto x_plus_half = mlir::stablehlo::Add(input_op, point_five_cst);
    auto floor_x = mlir::stablehlo::Floor(x_plus_half);
    auto abs_floor_x = mlir::stablehlo::Abs(floor_x);
    auto reduced_x = mlir::stablehlo::Add(input_op, abs_floor_x);
    pix = mlir::stablehlo::Mul(pi_cst, reduced_x);
  }

  auto sin_pix = mlir::stablehlo::Sine(pix);
  auto cos_pix = mlir::stablehlo::Cosine(pix);
  auto cot_pix = mlir::stablehlo::Div(cos_pix, sin_pix);

  std::vector<double> coeffs = ComputeCotDerivativeCoeffs(n);
  mlir::MlirOp poly_op = EvaluatePolynomial(cot_pix, coeffs);

  double pi_pow_val = -std::pow(kPi, n + 1);
  auto pi_pow_cst = MakeConstantLike(input_op, pi_pow_val);
  auto second_term = mlir::stablehlo::Mul(pi_pow_cst, poly_op);

  auto reflection_op = mlir::stablehlo::Add(first_term, second_term);

  // We apply reflection only when x < 0 and x is not an integer.
  // For n=1, use reflection even for integers to match finite values at poles
  auto zero_cst = MakeConstantLike(input_op, 0.0);
  auto is_x_neg = mlir::stablehlo::Compare(
      input_op, zero_cst, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp use_reflection;
  if (n == 1) {
    use_reflection = is_x_neg;
  } else {
    auto floor_x_orig = mlir::stablehlo::Floor(input_op);
    auto is_x_not_int = mlir::stablehlo::Compare(
        input_op, floor_x_orig, mlir::stablehlo::ComparisonDirection::NE);
    use_reflection = mlir::stablehlo::And(is_x_neg, is_x_not_int);
  }

  auto selected_op =
      mlir::stablehlo::Select(use_reflection, reflection_op, direct_op);

  mlir::MlirOp result;
  if (n > 1) {
    // Polygamma (n > 1): returns inf/-inf for non-positive integers.
    auto floor_op = mlir::stablehlo::Floor(input_op);
    auto is_integer = mlir::stablehlo::Compare(
        input_op, floor_op, mlir::stablehlo::ComparisonDirection::EQ);
    auto is_le_zero = mlir::stablehlo::Compare(
        input_op, zero_cst, mlir::stablehlo::ComparisonDirection::LE);
    auto is_pole = mlir::stablehlo::And(is_le_zero, is_integer);

    bool is_n_even = (n % 2 == 0);
    double inf_val = is_n_even ? -std::numeric_limits<double>::infinity()
                               : std::numeric_limits<double>::infinity();
    auto inf_cst = MakeConstantLike(input_op, inf_val);
    result = mlir::stablehlo::Select(is_pole, inf_cst, selected_op);
  } else {
    // Trigamma (n=1): returns finite values at poles (using reflection) and
    // evaluates to 0.0 at +inf.
    auto pos_inf =
        MakeConstantLike(input_op, std::numeric_limits<double>::infinity());
    auto is_pos_inf = mlir::stablehlo::Compare(
        input_op, pos_inf, mlir::stablehlo::ComparisonDirection::EQ);
    result = mlir::stablehlo::Select(is_pos_inf, zero_cst, selected_op);
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
