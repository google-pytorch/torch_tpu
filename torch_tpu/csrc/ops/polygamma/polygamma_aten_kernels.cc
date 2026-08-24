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

#include "torch_tpu/csrc/ops/polygamma/polygamma_aten_kernels.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "c10/util/BFloat16.h"
#include "c10/util/Half.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch/headeronly/util/BFloat16.h"
#include "torch/headeronly/util/Half.h"
#include "torch_tpu/csrc/common/aten_utils.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/ops/digamma/digamma_aten_kernels.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/csrc/ops/unary_aten_kernels.h"

namespace torch_tpu {
namespace {

constexpr double kPi = 3.14159265358979323846;

std::vector<double> ComputeCotDerivativeCoeffs(int n) {
  if (n == 0) return {0.0, 1.0};

  std::vector<double> prev = ComputeCotDerivativeCoeffs(n - 1);
  std::vector<double> coeffs(n + 2, 0.0);

  coeffs[0] = -prev[1];
  if (prev.size() > 2) {
    coeffs[1] = -2.0 * prev[2];
  }
  for (int k = 2; k <= n + 1; ++k) {
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

// Computes the Hurwitz Zeta function of x and q.
// Reference: xla/hlo/builder/lib/math.cc:Zeta
absl::StatusOr<mlir::MlirOp> BuildZeta(mlir::MlirOp x, mlir::MlirOp q,
                                       mlir::MlirBuilder& builder) {
  mlir::ElementType computation_type = GetElementTypeOrDie(q);
  mlir::Type comp_type =
      mlir::getElementType(builder.getContext(), computation_type);

  static constexpr int M = 12;
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

  auto i32_type = builder.getOpBuilder().getI32Type();
  auto zero_i32 = MakeScalarConstant(builder, 0, i32_type);
  auto nine_i32 = MakeScalarConstant(builder, 9, i32_type);
  auto one_i32 = MakeScalarConstant(builder, 1, i32_type);

  auto one_like_q = MakeConstantLike(q, 1.0);
  auto neg_x = mlir::stablehlo::Neg(x);
  auto s_init = mlir::stablehlo::Pow(q, neg_x);

  // While loop for S: state: [i, acc, S]
  auto s_loop = mlir::stablehlo::While(
      builder, {zero_i32, q, s_init},
      [&](mlir::RegionBuilder& cond) {
        mlir::MlirBuilder cond_builder(cond.getOpBuilder(),
                                       cond.getOpBuilder().getUnknownLoc());
        auto args = mlir::stablehlo::Arguments(
            cond, cond.getOp<mlir::stablehlo::WhileOp>());
        auto cond_val = mlir::stablehlo::Compare(
            args[0], nine_i32, mlir::stablehlo::ComparisonDirection::LT);
        mlir::stablehlo::Return(cond, cond_val);
      },
      [&](mlir::RegionBuilder& body) {
        mlir::MlirBuilder body_builder(body.getOpBuilder(),
                                       body.getOpBuilder().getUnknownLoc());
        auto args = mlir::stablehlo::Arguments(
            body, body.getOp<mlir::stablehlo::WhileOp>());

        auto i_val = args[0];
        auto acc_val = args[1];
        auto s_val = args[2];

        auto next_i = mlir::stablehlo::Add(i_val, one_i32);
        auto next_acc = mlir::stablehlo::Add(acc_val, one_like_q);
        auto neg_power = mlir::stablehlo::Pow(next_acc, neg_x);
        auto next_s = mlir::stablehlo::Add(s_val, neg_power);

        mlir::stablehlo::Return(body, {next_i, next_acc, next_s});
      });

  auto acc_final = s_loop[1];
  auto s_final = s_loop[2];

  auto acc_after = mlir::stablehlo::Add(acc_final, one_like_q);
  auto neg_power_after = mlir::stablehlo::Pow(acc_after, neg_x);

  auto x_minus_one = mlir::stablehlo::Subtract(x, one_like_q);
  auto I_mul = mlir::stablehlo::Mul(neg_power_after, acc_after);
  auto I_op = mlir::stablehlo::Div(I_mul, x_minus_one);

  auto one_cst = MakeConstantLike(acc_after, 1.0);
  auto acc_square = mlir::stablehlo::Mul(acc_after, acc_after);
  auto a_inverse_square = mlir::stablehlo::Div(one_cst, acc_square);

  std::vector<double> coeffs_inv(M - 1);
  for (int i = 0; i < M - 1; ++i) {
    coeffs_inv[i] = 1.0 / kZetaCoeffs[i];
  }
  auto coeffs_inv_cst = MakeConstant(
      builder, coeffs_inv, mlir::RankedTensorType::get({M - 1}, comp_type));

  auto two_float_0d = MakeScalarConstant(builder, 2.0, comp_type);
  auto c1_float_0d = MakeScalarConstant(
      builder, static_cast<double>(kTwoKMinusOne - 1), comp_type);
  auto c2_float_0d = MakeScalarConstant(
      builder, static_cast<double>(kTwoKMinusOne - 2), comp_type);

  auto eleven_i32 = MakeScalarConstant(builder, 11, i32_type);
  auto horner_sum_init = MakeConstantLike(acc_after, 0.0);

  // While loop for T: state: [i, horner_sum]
  auto t_loop = mlir::stablehlo::While(
      builder, {zero_i32, horner_sum_init},
      [&](mlir::RegionBuilder& cond) {
        mlir::MlirBuilder cond_builder(cond.getOpBuilder(),
                                       cond.getOpBuilder().getUnknownLoc());
        auto args = mlir::stablehlo::Arguments(
            cond, cond.getOp<mlir::stablehlo::WhileOp>());
        auto cond_val = mlir::stablehlo::Compare(
            args[0], eleven_i32, mlir::stablehlo::ComparisonDirection::LT);
        mlir::stablehlo::Return(cond, cond_val);
      },
      [&](mlir::RegionBuilder& body) {
        auto args = mlir::stablehlo::Arguments(
            body, body.getOp<mlir::stablehlo::WhileOp>());

        auto i_val = args[0];
        auto horner_sum_val = args[1];

        auto coeffs_local = mlir::MlirOp(body, coeffs_inv_cst.getValue());
        auto coeff_slice =
            mlir::stablehlo::DynamicSlice(coeffs_local, {i_val}, {1});
        auto coeff_cst = mlir::stablehlo::Reshape(coeff_slice, {});

        auto i_float =
            mlir::stablehlo::ConvertElementType(i_val, computation_type);
        auto two_i = mlir::stablehlo::Mul(i_float, two_float_0d);

        auto c1_local = mlir::MlirOp(body, c1_float_0d.getValue());
        auto s1 = mlir::stablehlo::Subtract(c1_local, two_i);

        auto c2_local = mlir::MlirOp(body, c2_float_0d.getValue());
        auto s2 = mlir::stablehlo::Subtract(c2_local, two_i);

        auto x_local = mlir::MlirOp(body, x.getValue());
        auto s1_bcast = mlir::stablehlo::BroadcastInDim(
            x_local.getType(), s1,
            body.getOpBuilder().getDenseI64ArrayAttr({}));
        auto s2_bcast = mlir::stablehlo::BroadcastInDim(
            x_local.getType(), s2,
            body.getOpBuilder().getDenseI64ArrayAttr({}));

        auto term_1 = mlir::stablehlo::Add(x_local, s1_bcast);
        auto term_2 = mlir::stablehlo::Add(x_local, s2_bcast);
        auto factor = mlir::stablehlo::Mul(term_1, term_2);

        auto coeff_bcast = mlir::stablehlo::BroadcastInDim(
            horner_sum_val.getType(), coeff_cst,
            body.getOpBuilder().getDenseI64ArrayAttr({}));

        auto sum_plus_coeff = mlir::stablehlo::Add(horner_sum_val, coeff_bcast);
        auto horner_mul_1 = mlir::stablehlo::Mul(factor, a_inverse_square);
        auto next_horner_sum =
            mlir::stablehlo::Mul(horner_mul_1, sum_plus_coeff);

        auto one_local = mlir::MlirOp(body, one_i32.getValue());
        auto next_i = mlir::stablehlo::Add(i_val, one_local);

        mlir::stablehlo::Return(body, {next_i, next_horner_sum});
      });

  auto horner_sum_final = t_loop[1];

  auto point_five = MakeConstantLike(neg_power_after, 0.5);
  auto coeff_M_minus_one =
      MakeConstantLike(acc_after, 1.0 / kZetaCoeffs[M - 1]);
  auto inner_sum = mlir::stablehlo::Add(coeff_M_minus_one, horner_sum_final);
  auto x_div_acc = mlir::stablehlo::Div(x, acc_after);
  auto T_mul = mlir::stablehlo::Mul(x_div_acc, inner_sum);
  auto T_add = mlir::stablehlo::Add(point_five, T_mul);
  auto T_op = mlir::stablehlo::Mul(neg_power_after, T_add);

  auto s_plus_I = mlir::stablehlo::Add(s_final, I_op);
  auto accurate_result = mlir::stablehlo::Add(s_plus_I, T_op);

  double eps_val = GetEpsilon(computation_type);
  auto eps_cst = MakeConstantLike(q, eps_val);

  auto abs_neg_power = mlir::stablehlo::Abs(neg_power_after);
  auto abs_s = mlir::stablehlo::Abs(s_final);
  auto threshold = mlir::stablehlo::Mul(abs_s, eps_cst);
  auto is_accurate_enough = mlir::stablehlo::Compare(
      abs_neg_power, threshold, mlir::stablehlo::ComparisonDirection::LT);
  auto output =
      mlir::stablehlo::Select(is_accurate_enough, s_final, accurate_result);

  return output;
}

// Computes the polygamma function of order n at self_op.
// Reference: xla/hlo/builder/lib/math.cc:Polygamma
absl::StatusOr<mlir::MlirOp> BuildPolygammaShlo(
    mlir::MlirOp self_op, int64_t n, mlir::ElementType out_mlir_type) {
  mlir::ElementType input_dtype = GetElementTypeOrDie(self_op);
  TT_ASSIGN_OR_RETURN(mlir::ElementType computation_type,
                      InferComputationDtype(input_dtype));

  mlir::MlirOp input_op = self_op;
  if (input_dtype != computation_type) {
    input_op = mlir::stablehlo::ConvertElementType(self_op, computation_type);
  }

  if (n == 0) {  // Digamma
    return BuildDigammaShlo(self_op, out_mlir_type);
  }

  mlir::MlirBuilder& builder = input_op.getBuilder();

  // 1. Direct path (x >= 0.0)
  // psi^(n)(x) = (-1)^(n+1) * n! * zeta(n+1, x)
  double sign_val = (n % 2 == 0) ? -1.0 : 1.0;
  double coeff_val = sign_val * std::tgamma(n + 1);
  auto coeff_cst = MakeConstantLike(input_op, coeff_val);
  auto n_plus_one_cst = MakeConstantLike(input_op, static_cast<double>(n + 1));
  TT_ASSIGN_OR_RETURN(auto zeta_op,
                      BuildZeta(n_plus_one_cst, input_op, builder));
  auto direct_op = mlir::stablehlo::Mul(coeff_cst, zeta_op);

  // 2. Reflection path (x < 0.0)
  // psi^(n)(x) = (-1)^n * psi^(n)(1-x) - pi^(n+1) * d^n/dx^n (cot(pi x))
  auto one = MakeConstantLike(input_op, 1.0);
  auto one_minus_x = mlir::stablehlo::Subtract(one, input_op);
  TT_ASSIGN_OR_RETURN(auto zeta_1_minus_x,
                      BuildZeta(n_plus_one_cst, one_minus_x, builder));

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
    // Trigamma (n=1): returns finite values at poles (using reflection)
    result = selected_op;
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
        << "expected n to be non-negative, got " << n;
    TT_CHECK_THROW(!IsComplex(self), error::kInvalidArgument)
        << "expected the input dtype not to be complex, got "
        << ToString(self.scalar_type());

    auto computation_type = self.scalar_type();
    if (c10::isIntegralType(computation_type, /*includeBool=*/true)) {
      computation_type = at::ScalarType::Float;
    }
    // Enforce safe type promotion (casting to integer outputs is not safe)
    TT_CHECK_THROW(out.is_floating_point() || out.is_complex(),
                   error::kInvalidArgument)
        << "expected the output dtype to be floating point or complex, got "
        << ToString(out.scalar_type());

    TT_ASSIGN_OR_THROW(auto out_dtype,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));
    TT_ASSIGN_OR_THROW(auto computation_mlir_type,
                       ConvertTo<mlir::ElementType>(computation_type));

    auto op_builder =
        [n, out_dtype](mlir::MlirOp self_op) -> absl::StatusOr<mlir::MlirOp> {
      return BuildPolygammaShlo(self_op, n, out_dtype);
    };

    TT_THROW_IF_ERROR(UnaryOpOut(self, out, std::move(op_builder),
                                 {.op_param_cache_keys = std::move(param_keys),
                                  .out_dtype = out_dtype,
                                  .computation_dtype = computation_mlir_type}));
    return out;
  });
}
}  // namespace torch_tpu
