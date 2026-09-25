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

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/result_type.h"
#include "ATen/ops/sigmoid.h"
#include "ATen/ops/silu_backward.h"
#include "absl/status/status.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/GradMode.h"
#include "c10/core/ScalarType.h"
#include "c10/core/impl/LocalDispatchKeySet.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/binary_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/silu/silu.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

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
    TT_THROW_IF_ERROR(
        UnaryOpOut(self, out, BuildSiluShlo,
                   {.op_param_cache_keys = OpParamCacheKeys::Empty(),
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
