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

#include "csrc/ops/mish/mish_aten_kernels.h"

#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/mish_backward.h"
#include "ATen/ops/result_type.h"
#include "ATen/ops/sigmoid.h"
#include "ATen/ops/softplus.h"
#include "ATen/ops/tanh.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/GradMode.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fake_tensor_utils.h"
#include "csrc/common/op_name_stack.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/binary_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

namespace stablehlo = mlir::stablehlo;

// Shared rational terms for `tanh(softplus(x)) = num / den` using a single
// overflow-free exponential `a = exp(-|x|) in [0, 1]`:
//   tanh(ln(1 + e^x)) = ((1 + e^x)^2 - 1) / ((1 + e^x)^2 + 1)
//                     = (2 e^x + e^(2x)) / (2 + 2 e^x + e^(2x))
//   - For x >= 0 (dividing by e^(2x), with a = e^(-x)):
//       num = 1 + 2a,     den = 1 + 2a + 2a^2
//   - For x < 0 (with a = e^x):
//       num = 2a + a^2,   den = 2 + 2a + a^2
// Avoiding `log1p`, `expm1`, and a `x > 20` cutoff reduces forward and backward
// from 3-4 software transcendentals each down to a single `exp(-|x|)`.
struct MishRationalTerms {
  mlir::MlirOp a;
  mlir::MlirOp a_sq;
  mlir::MlirOp is_nonneg;
  mlir::MlirOp one;
  mlir::MlirOp num;
  mlir::MlirOp den;
};

MishRationalTerms BuildMishRationalTermsShlo(mlir::MlirOp x) {
  mlir::MlirOp zero = MakeConstantLike(x, 0.0);
  mlir::MlirOp one = MakeConstantLike(x, 1.0);
  mlir::MlirOp two = MakeConstantLike(x, 2.0);

  mlir::MlirOp abs_x = stablehlo::Abs(x);
  mlir::MlirOp neg_abs_x = stablehlo::Neg(abs_x);
  mlir::MlirOp a = stablehlo::Exp(neg_abs_x);
  mlir::MlirOp two_a = stablehlo::Mul(two, a);
  mlir::MlirOp a_sq = stablehlo::Mul(a, a);

  mlir::MlirOp is_nonneg =
      stablehlo::Compare(x, zero, stablehlo::ComparisonDirection::GE);
  mlir::MlirOp num_extra = stablehlo::Select(is_nonneg, one, a_sq);
  mlir::MlirOp den_half_extra = stablehlo::Select(is_nonneg, a_sq, one);
  mlir::MlirOp den_extra = stablehlo::Mul(two, den_half_extra);
  mlir::MlirOp num = stablehlo::Add(two_a, num_extra);
  mlir::MlirOp den = stablehlo::Add(num, den_extra);
  return {
      .a = a,
      .a_sq = a_sq,
      .is_nonneg = is_nonneg,
      .one = one,
      .num = num,
      .den = den,
  };
}

absl::StatusOr<mlir::MlirOp> BuildMishShlo(mlir::MlirOp input_op) {
  // Only defined for floating point types.
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);

  TT_ASSIGN_OR_RETURN(const at::ScalarType scalar_type,
                      ConvertTo<c10::ScalarType>(input_type.getElementType()));

  ABSL_CHECK(IsFloatType(input_type))  // CRASH_OK=checked in AtenMishOut
      << "expected the input dtype to be floating point, got "
      << ToString(scalar_type) << ". This is a TorchTPU bug.";

  // Defined as mish(x) = x * tanh(softplus(x)) = x * (num / den).
  MishRationalTerms terms = BuildMishRationalTermsShlo(input_op);
  mlir::MlirOp tanh_sp = stablehlo::Div(terms.num, terms.den);
  return stablehlo::Mul(input_op, tanh_sp);
}

absl::StatusOr<mlir::MlirOp> BuildMishBackwardShlo(mlir::MlirOp grad_output_op,
                                                   mlir::MlirOp input_op) {
  const mlir::ElementType common_type = GetElementTypeOrDie(input_op);
  const at::ScalarType common_scalar_type =
      ConvertTo<at::ScalarType>(common_type);
  const at::ScalarType comp_scalar_type = ToAccumulateType(common_scalar_type);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType comp_type,
                      ConvertTo<mlir::ElementType>(comp_scalar_type));
  // BinaryOp already casts both inputs to the promoted common_type before
  // invoking BuildMishBackwardShlo, and casts the returned MlirOp to
  // result_dtype afterward. Only upcast here when common_type is narrower than
  // comp_type (e.g., bf16 or f16) to avoid emitting duplicate
  // ConvertElementType ops.
  mlir::MlirOp comp_grad_output =
      common_type == comp_type
          ? grad_output_op
          : stablehlo::ConvertElementType(grad_output_op, comp_type);
  mlir::MlirOp comp_input =
      common_type == comp_type
          ? input_op
          : stablehlo::ConvertElementType(input_op, comp_type);

  // Mathematical formulation of the Mish backward pass:
  //   mish(x) = x * t, where t = tanh(softplus(x)) = num / den.
  //   d(mish(x)) / dx = t + x * sigmoid(x) * (1 - t^2)
  //   Using den - num = (x >= 0 ? 2*a^2 : 2), den + num = 2*(1 + a)^2, and
  //   sigmoid(x) = (x >= 0 ? 1 : a) / (1 + a), one factor of (1 + a) cancels:
  //     sigmoid(x) * (1 - t^2) = 4 * s_factor * (1 + a) / den^2
  //   where s_factor = a^2 for x >= 0 and a for x < 0, giving:
  //     d(mish(x)) / dx = (num * den + x * (4 * s_factor * (1 + a))) / den^2
  //   Evaluating weight = 4 * s_factor * (1 + a) before multiplying by x avoids
  //   4 * x overflow near FLT_MAX while preserving IEEE inf * 0 = NaN at +-inf.
  MishRationalTerms terms = BuildMishRationalTermsShlo(comp_input);
  mlir::MlirOp four = MakeConstantLike(comp_input, 4.0);
  mlir::MlirOp s_factor =
      stablehlo::Select(terms.is_nonneg, terms.a_sq, terms.a);
  mlir::MlirOp one_plus_a = stablehlo::Add(terms.one, terms.a);
  mlir::MlirOp four_s = stablehlo::Mul(four, s_factor);
  mlir::MlirOp weight = stablehlo::Mul(four_s, one_plus_a);
  mlir::MlirOp x_weight = stablehlo::Mul(comp_input, weight);
  mlir::MlirOp num_den = stablehlo::Mul(terms.num, terms.den);
  mlir::MlirOp dmish_num = stablehlo::Add(num_den, x_weight);
  mlir::MlirOp dmish_den = stablehlo::Mul(terms.den, terms.den);
  mlir::MlirOp dmish_dx = stablehlo::Div(dmish_num, dmish_den);

  TT_ASSIGN_OR_RETURN((auto [bcast_grad_output, bcast_dmish_dx]),
                      ApplyBroadcastIfNeeded(comp_grad_output, dmish_dx));
  return stablehlo::Mul(bcast_grad_output, bcast_dmish_dx);
}

absl::Status ValidateMishInputs(const at::Tensor& self) {
  TT_RET_CHECK(self.is_floating_point(), error::kPythonNotImplementedError)
      << "not implemented for " << ToString(self.scalar_type());
  TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(self));
  return absl::OkStatus();
}

// Matches PyTorch's `mish_backward` behavior, which builds a TensorIterator
// binary op with `grad_input` pre-allocated in `self.options()` and promotes
// `grad_output` and `self` to a common floating-point dtype.
absl::Status ValidateMishBackwardInputs(const at::Tensor& grad_output,
                                        const at::Tensor& self) {
  TT_RET_CHECK(IsPrivateUse1Device(grad_output), error::kInvalidArgument)
      << "expected grad_output to be on " << GetPrivateUse1DeviceDebugName()
      << ", got " << grad_output.device();
  TT_RET_CHECK(IsPrivateUse1Device(self), error::kInvalidArgument)
      << "expected self to be on " << GetPrivateUse1DeviceDebugName()
      << ", got " << self.device();
  TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(grad_output));
  TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(self));
  const at::ScalarType promoted_type = at::result_type(grad_output, self);
  TT_RETURN_IF_ERROR(ValidateOutDtype(self, promoted_type));
  TT_RET_CHECK(at::isFloatingType(promoted_type),
               error::kPythonNotImplementedError)
      << "not implemented for " << ToString(promoted_type);
  return absl::OkStatus();
}

void ValidateMishBackwardInputsOrThrow(const at::Tensor& grad_output,
                                       const at::Tensor& self) {
  internal::ScopedOpName scoped_op_name(OpName::kMishBackward);
  ScopedPythonContextCapturer capturer(OpName::kMishBackward);
  try {
    TT_THROW_IF_ERROR(ValidateMishBackwardInputs(grad_output, self));
  } catch (const TtError& e) {
    TranslateToC10ErrorAndThrow(e);
  }
}

// Composite formulation used for `FakeTensor`s, functorch transforms and
// higher-order autograd, where the fused StableHLO kernel cannot be used.
at::Tensor MishBackwardComposite(const at::Tensor& grad_output,
                                 const at::Tensor& self) {
  internal::ScopedOpName scoped_op_name(OpName::kMishBackward);
  ScopedPythonContextCapturer capturer(OpName::kMishBackward);
  const at::Tensor input_tanh_softplus = at::tanh(at::softplus(self));
  const at::Tensor input_sigmoid = at::sigmoid(self);
  const at::Tensor result = grad_output.mul(input_tanh_softplus.add(
      self.mul(input_sigmoid)
          .mul(input_tanh_softplus.mul(input_tanh_softplus).neg().add(1.0))));
  return result.scalar_type() == self.scalar_type()
             ? result
             : result.to(self.scalar_type());
}

}  // namespace

at::Tensor& AtenMishOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kMishOut, _, (self, out), {
    TT_THROW_IF_ERROR(ValidateMishInputs(self));
    const at::ScalarType comp_type = ToAccumulateType(self.scalar_type());
    TT_ASSIGN_OR_THROW(const mlir::ElementType computation_dtype,
                       ConvertTo<mlir::ElementType>(comp_type));
    TT_THROW_IF_ERROR(
        UnaryOpOut(self, out, BuildMishShlo,
                   {.op_param_cache_keys = OpParamCacheKeys::Empty(),
                    .computation_dtype = computation_dtype,
                    .allow_out_dtype_cast = false}));
    return out;
  });
}

at::Tensor AtenMishBackwardAutograd(const at::Tensor& grad_output,
                                    const at::Tensor& self) {
  ValidateMishBackwardInputsOrThrow(grad_output, self);
  // Decompose into elementary ATen ops only when higher-order autograd
  // (`gradgradcheck` / `create_graph=True`) or functorch transforms (`vmap`,
  // `jvp`, `grad`) are active. For standard first-order backward passes
  // (including `AOTAutograd` / `ProxyTorchDispatchMode` tracing under
  // `torch.compile`), redispatch below `Autograd` so `ProxyTorchDispatchMode`
  // records a single `aten.mish_backward.default` FX node that `fx_to_mlir`
  // lowers directly to `BuildMishBackwardShlo` via `PrivateUse1`.
  if (IsFunctorchActive() ||
      (c10::GradMode::is_enabled() &&
       (grad_output.requires_grad() || self.requires_grad()))) {
    return MishBackwardComposite(grad_output, self);
  }
  at::AutoDispatchBelowAutograd guard;
  return at::mish_backward(grad_output, self);
}

at::Tensor AtenMishBackward(const at::Tensor& grad_output,
                            const at::Tensor& self) {
  if (IsFakeOrFunctorchTensor(grad_output) || IsFakeOrFunctorchTensor(self)) {
    ValidateMishBackwardInputsOrThrow(grad_output, self);
    return MishBackwardComposite(grad_output, self);
  }
  TT_KERNEL(OpName::kMishBackward, _, (grad_output, self), {
    TT_THROW_IF_ERROR(ValidateMishBackwardInputs(grad_output, self));
    TT_ASSIGN_OR_THROW(const mlir::ElementType result_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_ASSIGN_OR_THROW(
        at::Tensor grad_input,
        BinaryOp(grad_output, self, BuildMishBackwardShlo,
                 {.op_param_cache_keys = OpParamCacheKeys::Empty(),
                  .result_dtype = result_dtype}));
    return grad_input;
  });
}

}  // namespace torch_tpu
