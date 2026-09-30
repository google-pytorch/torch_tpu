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

#include "csrc/ops/silu/silu_aten_kernels.h"

#include "ATen/OpMathType.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/result_type.h"
#include "ATen/ops/sigmoid.h"
#include "ATen/ops/silu_backward.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/GradMode.h"
#include "c10/core/ScalarType.h"
#include "c10/core/impl/LocalDispatchKeySet.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/binary_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildSiluShlo(mlir::MlirOp input_op) {
  // Only defined for floating point types
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);

  TT_ASSIGN_OR_RETURN(const at::ScalarType scalar_type,
                      ConvertTo<c10::ScalarType>(input_type.getElementType()));

  ABSL_CHECK(IsFloatType(input_type))  // CRASH_OK=checked in AtenSiluOut
      << "Expected the input dtype to be floating point, got "
      << ToString(scalar_type) << ". This is a TorchTPU bug.";

  // Defined as silu(x) = x / (1 + exp(-x)):
  mlir::MlirOp one = MakeConstantLike(input_op, 1.0);
  mlir::MlirOp neg_x = stablehlo::Neg(input_op);
  mlir::MlirOp exp_neg_x = stablehlo::Exp(neg_x);
  mlir::MlirOp denom = stablehlo::Add(one, exp_neg_x);
  return stablehlo::Div(input_op, denom);
}

absl::StatusOr<mlir::MlirOp> BuildSiluBackwardShlo(mlir::MlirOp grad_output_op,
                                                   mlir::MlirOp input_op) {
  const mlir::ElementType out_dtype = GetElementTypeOrDie(input_op);
  const at::ScalarType out_scalar_type = ConvertTo<at::ScalarType>(out_dtype);
  const at::ScalarType computation_scalar_type =
      at::toOpMathType(out_scalar_type);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType computation_dtype,
                      ConvertTo<mlir::ElementType>(computation_scalar_type));
  mlir::MlirOp comp_grad_output =
      stablehlo::ConvertElementType(grad_output_op, computation_dtype);
  mlir::MlirOp comp_input =
      stablehlo::ConvertElementType(input_op, computation_dtype);

  // Mathematical formulation of the SiLU backward pass:
  //   silu(x) = x * \sigma(x), where \sigma(x) = 1 / (1 + exp(-x)).
  //   d(silu(x)) / dx = \sigma(x) * (1 + x * (1 - \sigma(x)))
  //   grad_input = grad_output * d(silu(x)) / dx
  //
  // Computing this in a single StableHLO kernel in FP32 (`computation_dtype`)
  // replaces `math_silu_backward`'s 6 separate ATen kernel dispatches and 6
  // intermediate HBM tensor allocations per SwiGLU MLP layer with a single
  // VMEM-fused VPU pass.
  mlir::MlirOp one = MakeConstantLike(comp_input, 1.0);
  mlir::MlirOp neg_x = stablehlo::Neg(comp_input);
  mlir::MlirOp exp_neg_x = stablehlo::Exp(neg_x);
  mlir::MlirOp denom = stablehlo::Add(one, exp_neg_x);
  mlir::MlirOp sigmoid_op = stablehlo::Div(one, denom);
  mlir::MlirOp one_minus_sigmoid = stablehlo::Subtract(one, sigmoid_op);
  mlir::MlirOp x_times_one_minus_sigmoid =
      stablehlo::Mul(comp_input, one_minus_sigmoid);
  mlir::MlirOp one_plus_term = stablehlo::Add(one, x_times_one_minus_sigmoid);
  mlir::MlirOp dsilu_dx = stablehlo::Mul(sigmoid_op, one_plus_term);

  TT_ASSIGN_OR_RETURN((auto [bcast_grad_output, bcast_dsilu_dx]),
                      ApplyBroadcastIfNeeded(comp_grad_output, dsilu_dx));
  mlir::MlirOp grad_input = stablehlo::Mul(bcast_grad_output, bcast_dsilu_dx);
  return stablehlo::ConvertElementType(grad_input, out_dtype);
}

bool IsFunctorchActive() {
  const auto include_set = c10::impl::tls_local_dispatch_key_set().included_;
  return include_set.has(c10::DispatchKey::FuncTorchDynamicLayerFrontMode) ||
         include_set.has(c10::DispatchKey::FuncTorchDynamicLayerBackMode);
}

bool IsFakeOrFunctorchTensor(const at::Tensor& tensor) {
  if (IsFunctorchActive() || !tensor.has_storage()) {
    return true;
  }
  return tensor.storage().data_ptr() == nullptr;
}

// Matches PyTorch's `silu_backward` meta function, which builds a
// TensorIterator binary op that promotes both inputs to a common dtype. Mixed
// integral/floating inputs are therefore accepted, and only an integral (or
// bool/complex) promoted dtype is rejected.
absl::Status ValidateSiluBackwardInputs(const at::Tensor& grad_output,
                                        const at::Tensor& self) {
  const at::ScalarType promoted_type = at::result_type(grad_output, self);
  TT_RET_CHECK(at::isFloatingType(promoted_type),
               error::kPythonNotImplementedError)
      << "not implemented for " << ToString(promoted_type);
  return absl::OkStatus();
}

void ValidateSiluBackwardInputsOrThrow(const at::Tensor& grad_output,
                                       const at::Tensor& self) {
  try {
    TT_THROW_IF_ERROR(ValidateSiluBackwardInputs(grad_output, self));
  } catch (const TtError& e) {
    TranslateToC10ErrorAndThrow(e);
  }
}

// Composite formulation used for `FakeTensor`s, functorch transforms and
// higher-order autograd, where the fused StableHLO kernel cannot be used.
at::Tensor SiluBackwardComposite(const at::Tensor& grad_output,
                                 const at::Tensor& self) {
  const at::Tensor input_sigmoid = at::sigmoid(self);
  return grad_output.mul(
      input_sigmoid.mul(self.mul(input_sigmoid.neg().add(1.0)).add(1.0)));
}

}  // namespace

at::Tensor& AtenSiluOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kSiluOut, _, (self, out), {
    TT_CHECK_THROW(self.scalar_type() != at::kBool &&
                       self.scalar_type() != at::kInt &&
                       self.scalar_type() != at::kLong,
                   error::kPythonNotImplementedError)
        << "not implemented for " << ToString(self.scalar_type());
    TT_CHECK_THROW(self.is_floating_point(), error::kInvalidArgument)
        << "expected the input dtype to be floating point, got "
        << ToString(self.scalar_type());
    const auto comp_type = ToAccumulateType(self.scalar_type());
    TT_ASSIGN_OR_THROW(const auto computation_dtype,
                       ConvertTo<mlir::ElementType>(comp_type));
    TT_THROW_IF_ERROR(
        UnaryOpOut(self, out, BuildSiluShlo,
                   {.op_param_cache_keys = OpParamCacheKeys::Empty(),
                    .computation_dtype = computation_dtype,
                    .allow_out_dtype_cast = false}));
    return out;
  });
}

at::Tensor& AtenSiluBackwardGradInput(const at::Tensor& grad_output,
                                      const at::Tensor& self,
                                      at::Tensor& grad_input) {
  if (IsFakeOrFunctorchTensor(grad_output) || IsFakeOrFunctorchTensor(self) ||
      IsFakeOrFunctorchTensor(grad_input)) {
    ValidateSiluBackwardInputsOrThrow(grad_output, self);
    try {
      TT_THROW_IF_ERROR(
          ValidateOutDtype(grad_input, at::result_type(grad_output, self)));
    } catch (const TtError& e) {
      TranslateToC10ErrorAndThrow(e);
    }
    const at::Tensor result = SiluBackwardComposite(grad_output, self);
    // Follow `out=` semantics: an `out` tensor with a different shape (e.g. an
    // empty placeholder) is resized to the broadcast result shape.
    if (grad_input.sizes() != result.sizes()) {
      grad_input.resize_(result.sizes());
    }
    grad_input.copy_(result);
    return grad_input;
  }
  TT_KERNEL(
      OpName::kSiluBackwardGradInput, _, (grad_output, self, grad_input), {
        TT_THROW_IF_ERROR(ValidateSiluBackwardInputs(grad_output, self));
        // PyTorch's TensorIterator allows safely casting the promoted result
        // to the `out` dtype (e.g. float32 -> bfloat16), so keep the default
        // `allow_out_dtype_cast = true`.
        TT_THROW_IF_ERROR(
            BinaryOpOut(grad_output, self, grad_input, BuildSiluBackwardShlo,
                        {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
        return grad_input;
      });
}

at::Tensor AtenSiluBackwardAutograd(const at::Tensor& grad_output,
                                    const at::Tensor& self) {
  ValidateSiluBackwardInputsOrThrow(grad_output, self);
  // Decompose into elementary ATen ops only when higher-order autograd
  // (`gradgradcheck` / `create_graph=True`) or functorch transforms (`vmap`,
  // `jvp`, `grad`) are active. For standard first-order backward passes
  // (including `AOTAutograd` / `ProxyTorchDispatchMode` tracing under
  // `torch.compile`), redispatch below `Autograd` so `ProxyTorchDispatchMode`
  // records a single `aten.silu_backward.default` FX node that `fx_to_mlir`
  // lowers directly to `BuildSiluBackwardShlo` via `PrivateUse1`.
  if (IsFunctorchActive() ||
      (c10::GradMode::is_enabled() &&
       (grad_output.requires_grad() || self.requires_grad()))) {
    return SiluBackwardComposite(grad_output, self);
  }
  at::AutoDispatchBelowAutograd guard;
  return at::silu_backward(grad_output, self);
}

at::Tensor AtenSiluBackward(const at::Tensor& grad_output,
                            const at::Tensor& self) {
  if (IsFakeOrFunctorchTensor(grad_output) || IsFakeOrFunctorchTensor(self)) {
    ValidateSiluBackwardInputsOrThrow(grad_output, self);
    return SiluBackwardComposite(grad_output, self);
  }
  TT_KERNEL(OpName::kSiluBackward, _, (grad_output, self), {
    TT_THROW_IF_ERROR(ValidateSiluBackwardInputs(grad_output, self));
    TT_ASSIGN_OR_THROW(
        at::Tensor grad_input,
        BinaryOp(grad_output, self, BuildSiluBackwardShlo,
                 {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return grad_input;
  });
}

}  // namespace torch_tpu
