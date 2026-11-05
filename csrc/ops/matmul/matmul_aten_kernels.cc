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

#include "csrc/ops/matmul/matmul_aten_kernels.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "ATen/TensorUtils.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/_unsafe_view.h"
#include "ATen/ops/add.h"
#include "ATen/ops/conj_physical.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/matmul.h"
#include "ATen/ops/matmul_backward.h"
#include "ATen/ops/matmul_native.h"
#include "ATen/ops/mul.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/DispatchKeySet.h"
#include "c10/core/ScalarType.h"
#include "c10/core/ScalarTypeToTypeMeta.h"
#include "c10/core/SymBool.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/core/TensorImpl.h"
#include "c10/core/impl/LocalDispatchKeySet.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/DimVector.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fake_tensor_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/matmul/matmul.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/precision_context.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/csrc/autograd/VariableTypeUtils.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/function_hook.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/autograd/saved_variable.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

// Returns true for scalar types supported by `aten::matmul` on TorchTPU
// (`float16`, `bfloat16`, `float32`, `float64`, `complex64`, `complex128`).
bool IsSupportedScalarType(const at::ScalarType scalar_type) {
  return (c10::isFloatingType(scalar_type) ||
          c10::isComplexType(scalar_type)) &&
         !IsLowPrecisionFloat(scalar_type);
}

// Returns true if `tensor` is a `Meta` device tensor or a Python `FakeTensor`
// backed by `Meta` storage.
bool IsMetaOrFakeTensor(const at::Tensor& tensor) {
  return tensor.defined() &&
         (tensor.is_meta() ||
          (tensor.key_set().has(c10::DispatchKey::Python) &&
           tensor.has_storage() &&
           tensor.storage().device_type() == c10::DeviceType::Meta));
}

// Returns true if `tensor` is a Python `__torch_dispatch__` wrapper subclass
// (excluding `FakeTensor` on `Meta` storage).
bool IsPythonWrapperSubclass(const at::Tensor& tensor) {
  return tensor.defined() && tensor.key_set().has(c10::DispatchKey::Python) &&
         (!tensor.has_storage() ||
          tensor.storage().device_type() != c10::DeviceType::Meta);
}

// Determines whether `AtenMatmulAutograd` should delegate to upstream
// `at::native::matmul` decomposition (`dot` / `mm` / `bmm` / `mv`) instead of
// attaching `MatmulAutogradNode`:
//   - Outside internal FX graph compilation, when `DispatchKey::Python` is
//     active on a `Meta` / `FakeTensor`, `at::native::matmul` allows Python
//     FakeTensor rules to trace through decomposed ATen ops.
//   - When functorch transforms (`vmap`, `jvp`, `grad`), Python tensor wrapper
//     subclasses, or forward-mode AD tangents (`isFwGradDefined`) are active,
//     upstream decomposition handles those dispatch layers natively.
bool ShouldFallbackMatmulAutograd(const at::Tensor& self,
                                  const at::Tensor& other) {
  if (GetEagerMode() != EagerMode::kInternalCompileFxGraph &&
      c10::impl::tls_is_dispatch_key_included(c10::DispatchKey::Python) &&
      (IsMetaOrFakeTensor(self) || IsMetaOrFakeTensor(other))) {
    return true;
  }
  return IsFunctorchActive() || IsPythonWrapperSubclass(self) ||
         IsPythonWrapperSubclass(other) ||
         torch::autograd::isFwGradDefined(self) ||
         torch::autograd::isFwGradDefined(other);
}

// Determines whether `AtenMatmulBackwardAutograd` should decompose into
// high-level ATen ops (`DecomposeMatmulBackward`) instead of attaching
// `MatmulBackwardAutogradNode` and redispatching to `AtenMatmulBackward`.
bool ShouldFallbackMatmulBackwardAutograd(const at::Tensor& grad,
                                          const at::Tensor& self,
                                          const at::Tensor& other) {
  if (GetEagerMode() != EagerMode::kInternalCompileFxGraph &&
      c10::impl::tls_is_dispatch_key_included(c10::DispatchKey::Python) &&
      (IsMetaOrFakeTensor(grad) || IsMetaOrFakeTensor(self) ||
       IsMetaOrFakeTensor(other))) {
    return true;
  }
  return IsFunctorchActive() || IsPythonWrapperSubclass(grad) ||
         IsPythonWrapperSubclass(self) || IsPythonWrapperSubclass(other) ||
         torch::autograd::isFwGradDefined(grad) ||
         torch::autograd::isFwGradDefined(self) ||
         torch::autograd::isFwGradDefined(other);
}

// Determines whether `AtenMatmulOutAutograd` should delegate to upstream
// `at::native::matmul_out` (`dot_out` / `mm_out` / `bmm_out` / `mv_out`) so
// functorch transforms, Python wrapper subclasses, and forward-mode AD
// tangents are handled by upstream decomposed dispatch layers.
bool ShouldFallbackMatmulOutAutograd(const at::Tensor& self,
                                     const at::Tensor& other,
                                     const at::Tensor& out) {
  return IsFunctorchActive() || IsPythonWrapperSubclass(self) ||
         IsPythonWrapperSubclass(other) || IsPythonWrapperSubclass(out) ||
         torch::autograd::isFwGradDefined(self) ||
         torch::autograd::isFwGradDefined(other) ||
         torch::autograd::isFwGradDefined(out);
}

// Returns true if `tensor` carries symbolic `SymInt` dimensions or a TPU base
// buffer with bounded dynamic dimensions. Inspects `GetBaseBuffer` rather than
// `GetBuffer` so non-contiguous views (e.g. `w.t()` in linear layers) do not
// synthesize and discard deferred `kAsStrided` view instructions on every
// eager call (`MarkDynamic` materializes views into dynamic base buffers).
bool HasDynamicShape(const at::Tensor& tensor) {
  if (!tensor.defined()) {
    return false;
  }
  if (tensor.unsafeGetTensorImpl()->has_symbolic_sizes_strides()) {
    return true;
  }
  if (!IsPrivateUse1Device(tensor) || !tensor.has_storage() ||
      !tensor.storage().data_ptr()) {
    return false;
  }
  const absl::StatusOr<DeviceBufferRef> buf = GetBaseBuffer(tensor);
  return buf.ok() && (!buf->dynamic_dimensions().empty() ||
                      buf->on_device_shape_is_dynamic());
}

// Creates a lightweight zero-storage metadata carrier tensor with the given
// `sizes`, `scalar_type`, and `device`. Used by `MatmulAutogradNode` and
// `MatmulBackwardAutogradNode` when an operand does not require a gradient
// (so only its paired operand's gradient is computed), avoiding saving its
// device storage or triggering version-counter checks on in-place mutations
// when only shape/dtype/device metadata is needed.
at::Tensor MakeMetadataCarrierTensor(absl::Span<const int64_t> sizes,
                                     const at::ScalarType scalar_type,
                                     const c10::Device device) {
  auto impl = c10::make_intrusive<c10::TensorImpl>(
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
      c10::scalarTypeToTypeMeta(scalar_type), device);
  impl->set_sizes_contiguous(AsArrayRef(sizes));
  return at::Tensor::wrap_tensor_impl(std::move(impl));
}

// Constructs a `torch::autograd::Node` subclass `T`, supporting both
// `c10::intrusive_ptr_target`-based nodes and `std::shared_ptr` nodes with
// custom `deleteNode` destruction across PyTorch versions.
template <typename T, typename... Args>
auto MakeAutogradNode(Args&&... args) {
  if constexpr (std::is_base_of_v<c10::intrusive_ptr_target, T>) {
    return c10::make_intrusive<T>(std::forward<Args>(args)...);
  } else {
    return std::shared_ptr<T>(new T(std::forward<Args>(args)...),
                              [](T* ptr) { deleteNode(ptr); });
  }
}

// Validates that `self`, `other`, and any defined `out` tensor reside on the
// exact same `c10::Device` (matching both device type and device index).
absl::Status ValidateDevices(const at::Tensor& self, const at::Tensor& other,
                             const std::optional<at::Tensor>& out) {
  TT_RET_CHECK(self.device() == other.device(), error::kInvalidArgument)
      << "expected both arguments to be on the same device, got "
      << self.device() << " and " << other.device();
  if (out.has_value() && out->defined()) {
    TT_RET_CHECK(out->device() == self.device(), error::kInvalidArgument)
        << "expected out tensor to be on device " << self.device() << ", got "
        << out->device();
  }
  return absl::OkStatus();
}

// Validates that `self`, `other`, and optional `out` share the exact same
// `ScalarType` (matching upstream `aten::matmul`, which does not implicitly
// type-promote mixed dtypes).
absl::Status ValidateDtypes(const at::Tensor& self, const at::Tensor& other,
                            const std::optional<at::Tensor>& out) {
  TT_RET_CHECK(self.scalar_type() == other.scalar_type(),
               error::kInvalidArgument)
      << "expected both arguments to have the same dtype, got "
      << ToString(self.scalar_type()) << " vs "
      << ToString(other.scalar_type());
  if (out.has_value()) {
    TT_RET_CHECK(out->scalar_type() == self.scalar_type(),
                 error::kInvalidArgument)
        << "expected out tensor to have dtype " << ToString(self.scalar_type())
        << ", got " << ToString(out->scalar_type());
  }
  return absl::OkStatus();
}

// Validates contracting dimensions and computes the symbolic output shape for
// low-rank (`1D` and `2D`) operand pairs, matching the exact error messages
// produced by upstream PyTorch (`LinearAlgebra.cpp` -> `dot`, `mv`, `mm`):
//   - `1D x 1D`: `[K] x [K] -> []`
//   - `2D x 1D`: `[M, K] x [K] -> [M]`
//   - `1D x 2D`: `[K] x [K, N] -> [N]` (prepends `1` to `self` in error text)
//   - `2D x 2D`: `[M, K] x [K, N] -> [M, N]`
absl::StatusOr<c10::SymDimVector> InferLowRankMatmulShape(
    const at::Tensor& self, const at::Tensor& other) {
  const int64_t self_rank = self.dim();
  const int64_t other_rank = other.dim();
  const c10::SymIntArrayRef self_sizes = self.sym_sizes();
  const c10::SymIntArrayRef other_sizes = other.sym_sizes();
  if (self_rank == 1 && other_rank == 1) {
    TT_RET_CHECK(self_sizes[0] == other_sizes[0], error::kInvalidArgument)
        << "expected both 1D tensors to have the same number of elements, got "
        << self_sizes[0] << " and " << other_sizes[0];
    return c10::SymDimVector{};
  }
  if (self_rank == 2 && other_rank == 1) {
    TT_RET_CHECK(self_sizes[1] == other_sizes[0], error::kInvalidArgument)
        << "expected the last dimension of the first argument ("
        << self_sizes[0] << "x" << self_sizes[1]
        << ") to match the dimension of the second argument (" << other_sizes[0]
        << "), got " << self_sizes[1] << " vs " << other_sizes[0];
    return c10::SymDimVector{self_sizes[0]};
  }
  if (self_rank == 1 && other_rank == 2) {
    TT_RET_CHECK(self_sizes[0] == other_sizes[0], error::kInvalidArgument)
        << "mat1 and mat2 shapes cannot be multiplied (1x" << self_sizes[0]
        << " and " << other_sizes[0] << "x" << other_sizes[1] << ")";
    return c10::SymDimVector{other_sizes[1]};
  }
  TT_RET_CHECK(self_sizes[1] == other_sizes[0], error::kInvalidArgument)
      << "mat1 and mat2 shapes cannot be multiplied (" << self_sizes[0] << "x"
      << self_sizes[1] << " and " << other_sizes[0] << "x" << other_sizes[1]
      << ")";
  return c10::SymDimVector{self_sizes[0], other_sizes[1]};
}

absl::StatusOr<c10::SymDimVector> InferBatchSymShape(
    c10::SymIntArrayRef self_batch, c10::SymIntArrayRef other_batch) {
  const int64_t batch_ndim =
      std::max<int64_t>(self_batch.size(), other_batch.size());
  c10::SymDimVector output_dims(batch_ndim);
  for (int64_t i = batch_ndim - 1; i >= 0; --i) {
    const int64_t offset = batch_ndim - 1 - i;
    const int64_t dim_a = static_cast<int64_t>(self_batch.size()) - 1 - offset;
    const int64_t dim_b = static_cast<int64_t>(other_batch.size()) - 1 - offset;
    const c10::SymInt size_a = dim_a >= 0 ? self_batch[dim_a] : c10::SymInt(1);
    const c10::SymInt size_b = dim_b >= 0 ? other_batch[dim_b] : c10::SymInt(1);
    const bool a_is_one = size_a == 1;
    const bool b_is_one = a_is_one || size_b == 1;
    TT_RET_CHECK(b_is_one || size_a == size_b, error::kInvalidArgument)
        << "the size of tensor a (" << size_a
        << ") must match the size of tensor b (" << size_b
        << ") at non-singleton dimension " << i;
    output_dims[i] = a_is_one ? size_b : size_a;
  }
  return output_dims;
}

// Validates batch broadcasting and matrix contracting dimensions for batched
// (`rank >= 3`) operand pairs, matching upstream `LinearAlgebra.cpp` validation
// order:
//   - For `ND x 1D` (`[B..., M, K] x [K] -> [B..., M]`): validates `self_k ==
//     other_k` directly.
//   - For `1D x ND`, `2D x ND`, `ND x 2D`, and `ND x ND`: validates
//     right-aligned batch dimension broadcastability from right to left
//     (`batch_ndim - 1` down to `0`) *before* checking the matrix contracting
//     dimension `self_k == other_k`.
absl::StatusOr<c10::SymDimVector> InferBatchedMatmulShape(
    const at::Tensor& self, const at::Tensor& other) {
  const int64_t self_rank = self.dim();
  const int64_t other_rank = other.dim();
  const c10::SymIntArrayRef self_sizes = self.sym_sizes();
  const c10::SymIntArrayRef other_sizes = other.sym_sizes();
  if (self_rank >= 3 && other_rank == 1) {
    const c10::SymInt self_k = self_sizes[self_rank - 1];
    const c10::SymInt other_k = other_sizes[0];
    TT_RET_CHECK(self_k == other_k, error::kInvalidArgument)
        << "expected the last dimension of the first argument "
        << ToString(self_sizes)
        << " to match the dimension of the second argument "
        << ToString(other_sizes) << ", got " << self_k << " vs " << other_k;
    return c10::SymDimVector(self_sizes.begin(), self_sizes.end() - 1);
  }

  const c10::SymIntArrayRef self_batch =
      self_sizes.slice(0, std::max<int64_t>(0, self_rank - 2));
  const c10::SymIntArrayRef other_batch =
      other_sizes.slice(0, std::max<int64_t>(0, other_rank - 2));
  TT_ASSIGN_OR_RETURN(c10::SymDimVector output_dims,
                      InferBatchSymShape(self_batch, other_batch));
  const c10::SymInt self_m =
      self_rank == 1 ? c10::SymInt(1) : self_sizes[self_rank - 2];
  const c10::SymInt self_k = self_sizes[self_rank - 1];
  const c10::SymInt other_k = other_sizes[other_rank - 2];
  const c10::SymInt other_n = other_sizes[other_rank - 1];
  TT_RET_CHECK(self_k == other_k, error::kInvalidArgument)
      << "mat1 and mat2 shapes cannot be multiplied (" << self_m << "x"
      << self_k << " and " << other_k << "x" << other_n << ")";
  if (self_rank >= 2) {
    output_dims.push_back(self_m);
  }
  output_dims.push_back(other_n);
  return output_dims;
}

absl::Status ValidateMatmulRanks(int64_t self_rank, int64_t other_rank) {
  TT_RET_CHECK(self_rank >= 1 && other_rank >= 1, error::kInvalidArgument)
      << "both arguments to matmul need to be at least 1D, but they are "
      << self_rank << "D and " << other_rank << "D";
  return absl::OkStatus();
}

// Validates that both operands are at least 1D and infers the symbolic output
// shape of `aten::matmul(self, other)`.
absl::StatusOr<c10::SymDimVector> InferMatmulShape(const at::Tensor& self,
                                                   const at::Tensor& other) {
  const int64_t self_rank = self.dim();
  const int64_t other_rank = other.dim();
  TT_RETURN_IF_ERROR(ValidateMatmulRanks(self_rank, other_rank));
  if (self_rank <= 2 && other_rank <= 2) {
    return InferLowRankMatmulShape(self, other);
  }
  return InferBatchedMatmulShape(self, other);
}

// Validates devices, dtypes, ranks, shapes, and supported scalar types matching
// upstream PyTorch (`LinearAlgebra.cpp` + structured kernel device checks):
//   1. `ValidateDevices`: verifies `self`, `other`, and `out` share a device.
//   2. `ValidateDtypes`: verifies `self`, `other`, and `out` dtypes match.
//   3. `InferMatmulShape`: verifies rank >= 1, batch broadcastability, and
//      contracting dimension equality.
//   4. `IsSupportedScalarType`: rejects unsupported dtypes (e.g. `int32`,
//      `int64`, `bool`, FP8) *after* shape validation, except for zero-element
//      non-`1D x 1D` inputs.
absl::StatusOr<c10::SymDimVector> ValidateAndInferMatmulOutputSymShape(
    const at::Tensor& self, const at::Tensor& other,
    const std::optional<at::Tensor>& out) {
  TT_RETURN_IF_ERROR(ValidateDevices(self, other, out));
  TT_RETURN_IF_ERROR(ValidateDtypes(self, other, out));
  TT_ASSIGN_OR_RETURN(c10::SymDimVector output_dims,
                      InferMatmulShape(self, other));

  // Avoid calling numel() when either input has symbolic sizes/strides so that
  // symbolic shapes neither throw nor specialize.
  const bool has_symbolic_shapes =
      self.unsafeGetTensorImpl()->has_symbolic_sizes_strides() ||
      other.unsafeGetTensorImpl()->has_symbolic_sizes_strides();
  const bool is_empty_non_dot = !has_symbolic_shapes &&
                                (self.numel() == 0 || other.numel() == 0) &&
                                !(self.dim() == 1 && other.dim() == 1) &&
                                !IsLowPrecisionFloat(self.scalar_type());
  TT_RET_CHECK(is_empty_non_dot || IsSupportedScalarType(self.scalar_type()),
               error::kPythonNotImplementedError)
      << "not implemented for " << ToString(self.scalar_type());
  return output_dims;
}

// Validates inputs and returns the concrete `int64_t` output shape of
// `aten::matmul(self, other)`.
absl::StatusOr<Dimensions> ValidateAndInferMatmulOutputShape(
    const at::Tensor& self, const at::Tensor& other,
    const std::optional<at::Tensor>& out) {
  TT_ASSIGN_OR_RETURN(const c10::SymDimVector sym_output_dims,
                      ValidateAndInferMatmulOutputSymShape(self, other, out));
  return ConcreteSizeToDimensions(sym_output_dims);
}

// Validates that defined `grad_self_out` / `grad_other_out` tensors reside on
// the same device as `self` / `other` (unconditionally, even when masked off)
// and match the dtypes of `self` / `other` when their `mask` entries are true.
absl::Status ValidateMatmulBackwardOutTensors(
    const at::Tensor& self, const at::Tensor& other,
    const std::array<bool, 2> mask,
    const std::optional<at::Tensor>& grad_self_out,
    const std::optional<at::Tensor>& grad_other_out) {
  if (grad_self_out.has_value() && grad_self_out->defined()) {
    TT_RET_CHECK(grad_self_out->device() == self.device(),
                 error::kInvalidArgument)
        << "expected grad_self out tensor to be on device " << self.device()
        << ", got " << grad_self_out->device();
    if (mask[0]) {
      TT_RET_CHECK(grad_self_out->scalar_type() == self.scalar_type(),
                   error::kInvalidArgument)
          << "expected grad_self out tensor to have dtype "
          << ToString(self.scalar_type()) << ", got "
          << ToString(grad_self_out->scalar_type());
    }
  }
  if (grad_other_out.has_value() && grad_other_out->defined()) {
    TT_RET_CHECK(grad_other_out->device() == other.device(),
                 error::kInvalidArgument)
        << "expected grad_other out tensor to be on device " << other.device()
        << ", got " << grad_other_out->device();
    if (mask[1]) {
      TT_RET_CHECK(grad_other_out->scalar_type() == other.scalar_type(),
                   error::kInvalidArgument)
          << "expected grad_other out tensor to have dtype "
          << ToString(other.scalar_type()) << ", got "
          << ToString(grad_other_out->scalar_type());
    }
  }
  return absl::OkStatus();
}

// Validates `grad`, `self`, `other`, and optional output buffers for
// `aten::matmul_backward`, ensuring all defined inputs and outputs reside on
// the same device and `grad`'s dtype and symbolic shape match the forward
// output of `aten::matmul(self, other)`.
absl::Status ValidateMatmulBackwardInputs(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    const std::array<bool, 2> mask,
    const std::optional<at::Tensor>& grad_self_out,
    const std::optional<at::Tensor>& grad_other_out) {
  TT_RET_CHECK(
      grad.device() == self.device() && self.device() == other.device(),
      error::kInvalidArgument)
      << "expected grad, self, and other to be on the same device, got "
      << grad.device() << ", " << self.device() << ", and " << other.device();
  TT_ASSIGN_OR_RETURN(
      const c10::SymDimVector expected_grad_dims,
      ValidateAndInferMatmulOutputSymShape(self, other, /*out=*/std::nullopt));
  TT_RET_CHECK(grad.scalar_type() == self.scalar_type(),
               error::kInvalidArgument)
      << "expected grad and self to have the same dtype, got "
      << ToString(grad.scalar_type()) << " vs " << ToString(self.scalar_type());
  const c10::SymIntArrayRef expected_grad_sizes(expected_grad_dims);
  TT_RET_CHECK(grad.sym_sizes() == expected_grad_sizes, error::kInvalidArgument)
      << "expected grad to have shape " << ToString(expected_grad_sizes)
      << ", got " << ToString(grad.sym_sizes());
  return ValidateMatmulBackwardOutTensors(self, other, mask, grad_self_out,
                                          grad_other_out);
}

// Shared dispatcher for `AtenMatmul` and `AtenMatmulOut`:
//   - Records the active matmul precision setting in `param_keys` via
//     `GetAndAddPrecisionTo`.
//   - When the output tensor itself has zero elements (`output_num_elements ==
//     0`), emits an empty zero-sized tensor directly; when an input is empty
//     only along the contracting dimension (`K == 0`, so `output_num_elements >
//     0`), delegates to `BuildMatmulShlo` -> `EmitZeroElementMatmul` to
//     materialize a zero-filled output tensor.
template <typename ReturnType, typename DispatchFn>
ReturnType DispatchMatmul(const at::Tensor& self, const at::Tensor& other,
                          const std::optional<at::Tensor>& out,
                          OpParamCacheKeys param_keys,
                          DispatchFn&& dispatch_fn) {
  TT_ASSIGN_OR_RETURN(Dimensions output_dims,
                      ValidateAndInferMatmulOutputShape(self, other, out));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype_mlir,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_ASSIGN_OR_RETURN(const int64_t output_num_elements,
                      NumElements(output_dims));
  const mlir::stablehlo::Precision current_precision =
      GetAndAddPrecisionTo(param_keys);
  auto op_builder = [current_precision, output_num_elements, output_dims,
                     out_dtype_mlir](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    if (output_num_elements == 0) {
      return MakeZeroSizedTensor(inputs[0].getBuilder(), out_dtype_mlir,
                                 AsArrayRef(absl::MakeConstSpan(output_dims)));
    }
    return BuildMatmulShlo(inputs[0], inputs[1], current_precision);
  };
  return std::forward<DispatchFn>(dispatch_fn)(
      std::move(op_builder),
      DispatchOpOptions<1>{.out_dtype = out_dtype_mlir,
                           .out_dims = std::move(output_dims),
                           .op_param_cache_keys = std::move(param_keys)});
}

// Dispatches `grad_self` when `mask == {true, false}`:
//   - When shapes are static (`!has_dynamic_shape`) or `self` is a
//     storage-less metadata carrier (`!self.has_storage()`, which is guaranteed
//     to have static sizes), `grad_self` depends only on `self.sizes()`, not on
//     `self`'s values. We record `self_sizes` in `param_keys` and invoke
//     `DispatchOp<2>` with only `{grad, other}`, avoiding binding `self`'s
//     device buffer as an XLA input parameter.
//   - When bounded dynamic dimensions are present on `self`, `self` is included
//     via `DispatchOp<3>` so its dynamic dimension bounds can be propagated.
absl::StatusOr<DeviceBufferRef> DispatchSingleGradSelf(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    const mlir::ElementType self_dtype_mlir,
    const mlir::stablehlo::Precision current_precision,
    const bool has_dynamic_shape, OpParamCacheKeys param_keys) {
  if (!has_dynamic_shape || !self.has_storage()) {
    TT_RETURN_IF_ERROR(param_keys.SetParam("self_sizes", self.sizes()));
    const Dimensions self_dims(self.sizes().begin(), self.sizes().end());
    auto op_builder = [current_precision,
                       self_dims](FixedSizeSpan<mlir::MlirOp, 2> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      return BuildMatmulGradSelfShlo(inputs[0], self_dims, inputs[1],
                                     current_precision);
    };
    return DispatchOp<2>(std::move(op_builder), {grad, other},
                         {.out_dtype = self_dtype_mlir,
                          .out_dims = self.sizes(),
                          .op_param_cache_keys = std::move(param_keys)});
  }
  auto op_builder = [current_precision](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    return BuildMatmulGradSelfShlo(inputs[0], inputs[1], inputs[2],
                                   current_precision);
  };
  return DispatchOp<3>(std::move(op_builder), {grad, self, other},
                       {.out_dtype = self_dtype_mlir,
                        .out_dims = self.sizes(),
                        .op_param_cache_keys = std::move(param_keys)});
}

// Dispatches `grad_other` when `mask == {false, true}`:
//   - When shapes are static (`!has_dynamic_shape`) or `other` is a
//     storage-less metadata carrier (`!other.has_storage()`, which is
//     guaranteed to have static sizes), records `other_sizes` in `param_keys`
//     and invokes `DispatchOp<2>` with only `{grad, self}`, avoiding binding
//     `other`'s device buffer as an XLA input parameter.
//   - When bounded dynamic dimensions are present on `other`, passes
//     `{grad, self, other}` via `DispatchOp<3>` to preserve dynamic dimension
//     bounds.
absl::StatusOr<DeviceBufferRef> DispatchSingleGradOther(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    const mlir::ElementType other_dtype_mlir,
    const mlir::stablehlo::Precision current_precision,
    const bool has_dynamic_shape, OpParamCacheKeys param_keys) {
  if (!has_dynamic_shape || !other.has_storage()) {
    TT_RETURN_IF_ERROR(param_keys.SetParam("other_sizes", other.sizes()));
    const Dimensions other_dims(other.sizes().begin(), other.sizes().end());
    auto op_builder = [current_precision,
                       other_dims](FixedSizeSpan<mlir::MlirOp, 2> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      return BuildMatmulGradOtherShlo(inputs[0], inputs[1], other_dims,
                                      current_precision);
    };
    return DispatchOp<2>(std::move(op_builder), {grad, self},
                         {.out_dtype = other_dtype_mlir,
                          .out_dims = other.sizes(),
                          .op_param_cache_keys = std::move(param_keys)});
  }
  auto op_builder = [current_precision](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    return BuildMatmulGradOtherShlo(inputs[0], inputs[1], inputs[2],
                                    current_precision);
  };
  return DispatchOp<3>(std::move(op_builder), {grad, self, other},
                       {.out_dtype = other_dtype_mlir,
                        .out_dims = other.sizes(),
                        .op_param_cache_keys = std::move(param_keys)});
}

// Out-variant of `DispatchSingleGradSelf` writing into `grad_self`.
absl::Status DispatchSingleGradSelfOut(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    at::Tensor& grad_self, const mlir::ElementType self_dtype_mlir,
    const mlir::stablehlo::Precision current_precision,
    const bool has_dynamic_shape, OpParamCacheKeys param_keys) {
  if (!has_dynamic_shape || !self.has_storage()) {
    TT_RETURN_IF_ERROR(param_keys.SetParam("self_sizes", self.sizes()));
    const Dimensions self_dims(self.sizes().begin(), self.sizes().end());
    auto op_builder = [current_precision,
                       self_dims](FixedSizeSpan<mlir::MlirOp, 2> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      return BuildMatmulGradSelfShlo(inputs[0], self_dims, inputs[1],
                                     current_precision);
    };
    return DispatchOpOut<2>(std::move(op_builder), {grad, other}, grad_self,
                            {.out_dtype = self_dtype_mlir,
                             .out_dims = self.sizes(),
                             .op_param_cache_keys = std::move(param_keys)});
  }
  auto op_builder = [current_precision](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    return BuildMatmulGradSelfShlo(inputs[0], inputs[1], inputs[2],
                                   current_precision);
  };
  return DispatchOpOut<3>(std::move(op_builder), {grad, self, other}, grad_self,
                          {.out_dtype = self_dtype_mlir,
                           .out_dims = self.sizes(),
                           .op_param_cache_keys = std::move(param_keys)});
}

// Out-variant of `DispatchSingleGradOther` writing into `grad_other`.
absl::Status DispatchSingleGradOtherOut(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    at::Tensor& grad_other, const mlir::ElementType other_dtype_mlir,
    const mlir::stablehlo::Precision current_precision,
    const bool has_dynamic_shape, OpParamCacheKeys param_keys) {
  if (!has_dynamic_shape || !other.has_storage()) {
    TT_RETURN_IF_ERROR(param_keys.SetParam("other_sizes", other.sizes()));
    const Dimensions other_dims(other.sizes().begin(), other.sizes().end());
    auto op_builder = [current_precision,
                       other_dims](FixedSizeSpan<mlir::MlirOp, 2> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      return BuildMatmulGradOtherShlo(inputs[0], inputs[1], other_dims,
                                      current_precision);
    };
    return DispatchOpOut<2>(std::move(op_builder), {grad, self}, grad_other,
                            {.out_dtype = other_dtype_mlir,
                             .out_dims = other.sizes(),
                             .op_param_cache_keys = std::move(param_keys)});
  }
  auto op_builder = [current_precision](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    return BuildMatmulGradOtherShlo(inputs[0], inputs[1], inputs[2],
                                    current_precision);
  };
  return DispatchOpOut<3>(std::move(op_builder), {grad, self, other},
                          grad_other,
                          {.out_dtype = other_dtype_mlir,
                           .out_dims = other.sizes(),
                           .op_param_cache_keys = std::move(param_keys)});
}

// Implements `AtenMatmulBackward`:
//   - `mask == {false, false}`: returns two undefined tensors without
//     dispatching to XLA.
//   - `mask == {true, false}` or `{false, true}`: dispatches a single-output
//     XLA computation (`DispatchSingleGradSelf` / `DispatchSingleGradOther`).
//   - `mask == {true, true}`: fuses `grad_self` and `grad_other` into a single
//     2-output XLA computation (`DispatchOp<3, 2>`).
absl::StatusOr<std::tuple<at::Tensor, at::Tensor>> MatmulBackwardImpl(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    const std::array<bool, 2> mask, OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateMatmulBackwardInputs(
      grad, self, other, mask, /*grad_self_out=*/std::nullopt,
      /*grad_other_out=*/std::nullopt));
  if (!mask[0] && !mask[1]) {
    return std::make_tuple(at::Tensor(), at::Tensor());
  }
  TT_ASSIGN_OR_RETURN(const mlir::ElementType self_dtype_mlir,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType other_dtype_mlir,
                      ConvertTo<mlir::ElementType>(other.scalar_type()));
  const mlir::stablehlo::Precision current_precision =
      GetAndAddPrecisionTo(param_keys);
  const bool has_dynamic_shape =
      HasDynamicShape(grad) || HasDynamicShape(self) || HasDynamicShape(other);
  if (mask[0] && !mask[1]) {
    TT_ASSIGN_OR_RETURN(
        DeviceBufferRef grad_self_buf,
        DispatchSingleGradSelf(grad, self, other, self_dtype_mlir,
                               current_precision, has_dynamic_shape,
                               std::move(param_keys)));
    return std::make_tuple(MakeTensor(std::move(grad_self_buf)), at::Tensor());
  }
  if (!mask[0] && mask[1]) {
    TT_ASSIGN_OR_RETURN(
        DeviceBufferRef grad_other_buf,
        DispatchSingleGradOther(grad, self, other, other_dtype_mlir,
                                current_precision, has_dynamic_shape,
                                std::move(param_keys)));
    return std::make_tuple(at::Tensor(), MakeTensor(std::move(grad_other_buf)));
  }

  auto op_builder = [current_precision](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<MlirOpResults<2>> {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_self_op,
                        BuildMatmulGradSelfShlo(inputs[0], inputs[1], inputs[2],
                                                current_precision));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_other_op,
                        BuildMatmulGradOtherShlo(inputs[0], inputs[1],
                                                 inputs[2], current_precision));
    return MlirOpResults<2>{grad_self_op, grad_other_op};
  };
  const std::array<mlir::ElementType, 2> out_dtypes = {
      self_dtype_mlir,
      other_dtype_mlir,
  };
  const std::array<absl::Span<const int64_t>, 2> out_dims_list = {
      self.sizes(),
      other.sizes(),
  };
  TT_ASSIGN_OR_RETURN(
      auto output_bufs,
      (DispatchOp<3, 2>(std::move(op_builder), {grad, self, other},
                        {.out_dtypes = out_dtypes,
                         .out_dims_list = out_dims_list,
                         .op_param_cache_keys = std::move(param_keys)})));
  return std::make_tuple(MakeTensor(std::move(output_bufs[0])),
                         MakeTensor(std::move(output_bufs[1])));
}

// Allocates placeholder 0-sized tensors for `grad_self` / `grad_other` when an
// out-variant caller passes an undefined tensor for an active `mask` slot so
// `DispatchOpOut` can resize and populate it.
absl::Status EnsureBackwardOutTensorsDefined(const at::Tensor& self,
                                             const at::Tensor& other,
                                             const std::array<bool, 2> mask,
                                             at::Tensor& grad_self,
                                             at::Tensor& grad_other) {
  if (mask[0] && !grad_self.defined()) {
    TT_ASSIGN_OR_RETURN(
        grad_self,
        MakeEmptyTensor(/*size=*/{0}, self.scalar_type(), self.device()));
  }
  if (mask[1] && !grad_other.defined()) {
    TT_ASSIGN_OR_RETURN(
        grad_other,
        MakeEmptyTensor(/*size=*/{0}, other.scalar_type(), other.device()));
  }
  return absl::OkStatus();
}

// Implements `AtenMatmulBackwardOut`, writing active gradients into `grad_self`
// and/or `grad_other`.
absl::Status MatmulBackwardOutImpl(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    const std::array<bool, 2> mask, at::Tensor& grad_self,
    at::Tensor& grad_other, OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateMatmulBackwardInputs(grad, self, other, mask,
                                                  grad_self, grad_other));
  if (!mask[0] && !mask[1]) {
    return absl::OkStatus();
  }
  TT_RETURN_IF_ERROR(EnsureBackwardOutTensorsDefined(self, other, mask,
                                                     grad_self, grad_other));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType self_dtype_mlir,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType other_dtype_mlir,
                      ConvertTo<mlir::ElementType>(other.scalar_type()));
  const mlir::stablehlo::Precision current_precision =
      GetAndAddPrecisionTo(param_keys);
  const bool has_dynamic_shape =
      HasDynamicShape(grad) || HasDynamicShape(self) || HasDynamicShape(other);
  if (mask[0] && !mask[1]) {
    return DispatchSingleGradSelfOut(grad, self, other, grad_self,
                                     self_dtype_mlir, current_precision,
                                     has_dynamic_shape, std::move(param_keys));
  }
  if (!mask[0] && mask[1]) {
    return DispatchSingleGradOtherOut(grad, self, other, grad_other,
                                      other_dtype_mlir, current_precision,
                                      has_dynamic_shape, std::move(param_keys));
  }

  auto op_builder = [current_precision](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<MlirOpResults<2>> {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_self_op,
                        BuildMatmulGradSelfShlo(inputs[0], inputs[1], inputs[2],
                                                current_precision));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_other_op,
                        BuildMatmulGradOtherShlo(inputs[0], inputs[1],
                                                 inputs[2], current_precision));
    return MlirOpResults<2>{grad_self_op, grad_other_op};
  };
  const std::array<mlir::ElementType, 2> out_dtypes = {
      self_dtype_mlir,
      other_dtype_mlir,
  };
  const std::array<absl::Span<const int64_t>, 2> out_dims_list = {
      self.sizes(),
      other.sizes(),
  };
  return DispatchOpOut<3, 2>(std::move(op_builder), {grad, self, other},
                             {grad_self, grad_other},
                             {.out_dtypes = out_dtypes,
                              .out_dims_list = out_dims_list,
                              .op_param_cache_keys = std::move(param_keys)});
}

// Returns true when a high-rank (`dim >= 3`) operand paired with a `1D` or `2D`
// operand can be folded into a 2D matrix and dispatched to `mm` or `mv` instead
// of `bmm`. Uses `SymInt`-aware shape and stride checks (`sym_numel`,
// `sym_sizes`, `sym_strides`, `computeStride`) so symbolic `FakeTensor` shapes
// do not throw or specialize on older PyTorch versions (e.g. PyTorch 2.11).
bool ShouldFoldMatmul(const at::Tensor& tensor1, const at::Tensor& tensor2) {
  const bool tensor1_larger = tensor1.dim() >= tensor2.dim();
  const at::Tensor t1 = tensor1_larger ? tensor1 : tensor2.mT();
  const int64_t dim_t1 = t1.dim();
  const int64_t dim_t2 = tensor1_larger ? tensor2.dim() : tensor1.dim();
  if (!(dim_t1 >= 3 && dim_t2 <= 2)) {
    return false;
  }
  const bool t2_requires_grad =
      tensor1_larger ? tensor2.requires_grad() : tensor1.requires_grad();
  if (t2_requires_grad) {
    return true;
  }
  if (tensor1.dim() == 2) {
    return false;
  }
  if (TORCH_GUARD_OR_FALSE(t1.sym_numel().sym_eq(0))) {
    return true;
  }
  const c10::SymIntArrayRef t1_shape = t1.sym_sizes();
  const c10::SymDimVector leading_shape(t1_shape.begin(), t1_shape.end() - 1);
  const c10::SymDimVector folded_shape{c10::multiply_integers(leading_shape),
                                       t1_shape.back()};
  return at::detail::computeStride(t1_shape, t1.sym_strides(), folded_shape)
      .has_value();
}

// Folds the leading dimensions of the higher-rank operand into a 2D matrix via
// `reshape_symint`, executes `mm` or `mv`, and restores the output shape via
// `_unsafe_view_symint` without specializing symbolic dimensions.
at::Tensor DecomposeFoldedMatmul(const at::Tensor& tensor1,
                                 const at::Tensor& tensor2) {
  const int64_t dim_tensor1 = tensor1.dim();
  const int64_t dim_tensor2 = tensor2.dim();
  const bool transpose = dim_tensor2 > dim_tensor1;
  const at::Tensor t1 = transpose ? tensor2.mT() : tensor1;
  const at::Tensor t2 =
      !transpose ? tensor2 : (dim_tensor1 == 2 ? tensor1.t() : tensor1);
  const c10::SymIntArrayRef sizes_1 = t1.sym_sizes();
  c10::SymDimVector output_shape(sizes_1.begin(), sizes_1.end() - 1);
  const c10::SymInt folded_dim1 = c10::multiply_integers(output_shape);
  const bool t2_is_matrix = t2.dim() == 2;
  if (t2_is_matrix) {
    output_shape.push_back(t2.sym_sizes()[1]);
  }
  const c10::SymDimVector t1_folded_shape{folded_dim1, sizes_1.back()};
  const at::Tensor t1_folded = t1.reshape_symint(t1_folded_shape);
  if (!t2_is_matrix) {
    return at::_unsafe_view_symint(t1_folded.mv(t2), output_shape);
  }
  const at::Tensor output =
      at::_unsafe_view_symint(t1_folded.mm(t2), output_shape);
  return transpose ? output.mT().contiguous() : output;
}

at::Tensor ExpandMatmulRhsForBmm(const at::Tensor& tensor2,
                                 const c10::SymDimVector& batch_shape,
                                 const c10::SymInt& expand_batch_product,
                                 const c10::SymInt& m2, const c10::SymInt& p,
                                 const bool vector_rhs) {
  c10::SymDimVector expand_size(batch_shape.begin(), batch_shape.end());
  expand_size.push_back(m2);
  if (!vector_rhs) {
    expand_size.push_back(p);
  }
  const at::Tensor expanded = tensor2.expand_symint(expand_size);
  if (vector_rhs) {
    return expanded.reshape_symint({expand_batch_product, m2}).unsqueeze(2);
  }
  return expanded.reshape_symint({expand_batch_product, m2, p});
}

// Broadcast-expands and flattens leading batch dimensions of `tensor1` and
// `tensor2` using `SymInt`-aware `expand_symint` and `reshape_symint`, executes
// `bmm`, and views the output back to its broadcasted symbolic shape.
at::Tensor DecomposeBatchedMatmul(const at::Tensor& tensor1,
                                  const at::Tensor& tensor2) {
  const int64_t dim_tensor1 = tensor1.dim();
  const int64_t dim_tensor2 = tensor2.dim();
  const c10::SymIntArrayRef tensor1_sizes = tensor1.sym_sizes();
  const c10::SymIntArrayRef tensor2_sizes = tensor2.sym_sizes();
  const c10::SymInt n =
      dim_tensor1 > 1 ? tensor1_sizes[dim_tensor1 - 2] : c10::SymInt(1);
  const c10::SymInt m1 = tensor1_sizes[dim_tensor1 - 1];
  const c10::SymInt m2 =
      dim_tensor2 > 1 ? tensor2_sizes[dim_tensor2 - 2] : tensor2_sizes[0];
  const c10::SymInt p =
      dim_tensor2 > 1 ? tensor2_sizes[dim_tensor2 - 1] : c10::SymInt(1);
  const c10::SymIntArrayRef batch_tensor1 =
      tensor1_sizes.slice(0, std::max<int64_t>(dim_tensor1 - 2, 0));
  const c10::SymIntArrayRef batch_tensor2 =
      tensor2_sizes.slice(0, std::max<int64_t>(dim_tensor2 - 2, 0));
  TT_ASSIGN_OR_THROW(c10::SymDimVector output_shape,
                     InferBatchSymShape(batch_tensor1, batch_tensor2));
  const c10::SymInt expand_batch_product = c10::multiply_integers(output_shape);
  c10::SymDimVector tensor1_expand_size(output_shape.begin(),
                                        output_shape.end());
  tensor1_expand_size.push_back(n);
  tensor1_expand_size.push_back(m1);
  const at::Tensor tensor1_expanded =
      tensor1.expand_symint(tensor1_expand_size)
          .reshape_symint({expand_batch_product, n, m1});
  const bool vector_rhs = dim_tensor2 == 1;
  const at::Tensor tensor2_expanded = ExpandMatmulRhsForBmm(
      tensor2, output_shape, expand_batch_product, m2, p, vector_rhs);
  if (dim_tensor1 > 1) {
    output_shape.push_back(n);
  }
  if (dim_tensor2 > 1) {
    output_shape.push_back(p);
  }
  const at::Tensor bmm_out = tensor1_expanded.bmm(tensor2_expanded);
  return at::_unsafe_view_symint(vector_rhs ? bmm_out.squeeze(-1) : bmm_out,
                                 output_shape);
}

bool ShouldSqueeze3DBatchForGrad(const at::Tensor& tensor,
                                 const c10::SymInt& batch_size) {
  return TORCH_GUARD_OR_FALSE(batch_size.sym_eq(1)) &&
         (tensor.requires_grad() ||
          tensor.key_set().has(c10::DispatchKey::Python));
}

// Decomposes `aten::matmul` into `SymInt`-aware ATen operations (`dot`, `mv`,
// `mm`, `bmm`, `reshape_symint`, `expand_symint`, `_unsafe_view_symint`) when
// `ShouldFallbackMatmulAutograd` is true. Unlike `at::native::matmul` in
// PyTorch < 2.13 (which calls non-symbolic `numel()` and `sizes()`), this
// decomposition preserves symbolic `SymInt` shapes without throwing or guarding
// on concrete sizes.
at::Tensor DecomposeMatmul(const at::Tensor& tensor1,
                           const at::Tensor& tensor2);

at::Tensor DecomposeHighRankMatmul(const at::Tensor& tensor1,
                                   const at::Tensor& tensor2) {
  if (ShouldFoldMatmul(tensor1, tensor2)) {
    return DecomposeFoldedMatmul(tensor1, tensor2);
  }
  if (tensor1.dim() == 3 && tensor2.dim() == 3 &&
      TORCH_GUARD_OR_TRUE(
          tensor1.sym_sizes()[0].sym_ne(tensor2.sym_sizes()[0]))) {
    if (ShouldSqueeze3DBatchForGrad(tensor1, tensor1.sym_sizes()[0])) {
      return DecomposeMatmul(tensor1.squeeze(0), tensor2);
    }
    if (ShouldSqueeze3DBatchForGrad(tensor2, tensor2.sym_sizes()[0])) {
      return DecomposeMatmul(tensor1, tensor2.squeeze(0));
    }
  }
  return DecomposeBatchedMatmul(tensor1, tensor2);
}

at::Tensor DecomposeMatmul(const at::Tensor& tensor1,
                           const at::Tensor& tensor2) {
  const int64_t dim_tensor1 = tensor1.dim();
  const int64_t dim_tensor2 = tensor2.dim();
  TT_THROW_IF_ERROR(ValidateMatmulRanks(dim_tensor1, dim_tensor2));
  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    return tensor1.dot(tensor2);
  }
  if (dim_tensor1 == 2 && dim_tensor2 == 1) {
    return tensor1.mv(tensor2);
  }
  if (dim_tensor1 == 1 && dim_tensor2 == 2) {
    return tensor1.unsqueeze(0).mm(tensor2).squeeze_(0);
  }
  if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    return tensor1.mm(tensor2);
  }
  return DecomposeHighRankMatmul(tensor1, tensor2);
}

at::Tensor DecomposeMatmulGradSelf(const at::Tensor& grad,
                                   const at::Tensor& self,
                                   const at::Tensor& other) {
  const int64_t self_rank = self.dim();
  const int64_t other_rank = other.dim();
  if (self_rank == 1 && other_rank == 1) {
    return at::mul(grad, at::conj_physical(other));
  }
  if (other_rank == 1) {
    return at::mul(grad.unsqueeze(-1), at::conj_physical(other));
  }
  if (self_rank == 1) {
    return at::matmul(grad.unsqueeze(-2),
                      at::conj_physical(other).transpose(-2, -1))
        .squeeze(-2)
        .sum_to_size_symint(self.sym_sizes());
  }
  return at::matmul(grad, at::conj_physical(other).transpose(-2, -1))
      .sum_to_size_symint(self.sym_sizes());
}

at::Tensor DecomposeMatmulGradOther(const at::Tensor& grad,
                                    const at::Tensor& self,
                                    const at::Tensor& other) {
  const int64_t self_rank = self.dim();
  const int64_t other_rank = other.dim();
  if (self_rank == 1 && other_rank == 1) {
    return at::mul(at::conj_physical(self), grad);
  }
  if (other_rank == 1) {
    return at::matmul(grad.unsqueeze(-2), at::conj_physical(self))
        .squeeze(-2)
        .sum_to_size_symint(other.sym_sizes());
  }
  if (self_rank == 1) {
    return at::mul(at::conj_physical(self).unsqueeze(-1), grad.unsqueeze(-2))
        .sum_to_size_symint(other.sym_sizes());
  }
  return at::matmul(at::conj_physical(self).transpose(-2, -1), grad)
      .sum_to_size_symint(other.sym_sizes());
}

// Decomposes `aten::matmul_backward` into high-level ATen operations
// (`at::matmul`, `at::conj_physical`, `transpose`, `unsqueeze`, `squeeze`,
// `sum_to_size`) when functorch transforms (`vmap`/`jvp`/`grad`), forward-mode
// AD, Python tensor wrapper subclasses, or Python `FakeTensor` dispatch are
// active.
std::tuple<at::Tensor, at::Tensor> DecomposeMatmulBackward(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    const std::array<bool, 2> mask) {
  TT_THROW_IF_ERROR(ValidateMatmulBackwardInputs(
      grad, self, other, mask, /*grad_self_out=*/std::nullopt,
      /*grad_other_out=*/std::nullopt));
  at::Tensor grad_self =
      mask[0] ? DecomposeMatmulGradSelf(grad, self, other) : at::Tensor();
  at::Tensor grad_other =
      mask[1] ? DecomposeMatmulGradOther(grad, self, other) : at::Tensor();
  return std::make_tuple(std::move(grad_self), std::move(grad_other));
}

at::Tensor UnpackOrMakeMetadataCarrier(torch::autograd::SavedVariable& saved,
                                       const bool should_unpack,
                                       absl::Span<const int64_t> sizes,
                                       const at::ScalarType dtype,
                                       const c10::Device device) {
  return should_unpack ? saved.unpack()
                       : MakeMetadataCarrierTensor(sizes, dtype, device);
}

// Higher-order (double-backward) autograd node for `aten::matmul_backward`.
// Given incoming cotangents `(gg_self, gg_other)` with respect to
// `(grad_self, grad_other)`, computes gradients with respect to `(grad, self,
// other)`:
//   - `dL/d(grad)  = gg_self @ other + self @ gg_other` (gated by `mask_`)
//   - `dL/d(self)  = std::get<0>(matmul_backward(grad, self, gg_other, {true,
//     false}))`
//   - `dL/d(other) = std::get<1>(matmul_backward(grad, gg_self, other, {false,
//     true}))`
// When all three inputs (`grad`, `self`, `other`) have static non-Fake shapes
// (`can_use_metadata_`), only saves tensor storage for operands whose values
// can be required under `mask_`, substituting lightweight metadata carriers
// (`MakeMetadataCarrierTensor`) when only shape/dtype/device metadata is
// needed.
class MatmulBackwardAutogradNode : public torch::autograd::Node {
 public:
  MatmulBackwardAutogradNode(const at::Tensor& grad, const at::Tensor& self,
                             const at::Tensor& other,
                             const std::array<bool, 2> mask)
      : can_use_metadata_(
            !HasDynamicShape(grad) && !HasDynamicShape(self) &&
            !HasDynamicShape(other) && !IsMetaOrFakeTensor(grad) &&
            !IsMetaOrFakeTensor(self) && !IsMetaOrFakeTensor(other)),
        grad_((mask[0] || mask[1] || !can_use_metadata_)
                  ? torch::autograd::SavedVariable(grad, /*is_output=*/false)
                  : torch::autograd::SavedVariable()),
        self_((mask[1] || !can_use_metadata_)
                  ? torch::autograd::SavedVariable(self, /*is_output=*/false)
                  : torch::autograd::SavedVariable()),
        other_((mask[0] || !can_use_metadata_)
                   ? torch::autograd::SavedVariable(other, /*is_output=*/false)
                   : torch::autograd::SavedVariable()),
        grad_sizes_(can_use_metadata_
                        ? Dimensions(grad.sizes().begin(), grad.sizes().end())
                        : Dimensions{}),
        grad_dtype_(grad.scalar_type()),
        grad_device_(grad.device()),
        self_sizes_(can_use_metadata_
                        ? Dimensions(self.sizes().begin(), self.sizes().end())
                        : Dimensions{}),
        self_dtype_(self.scalar_type()),
        self_device_(self.device()),
        other_sizes_(can_use_metadata_ ? Dimensions(other.sizes().begin(),
                                                    other.sizes().end())
                                       : Dimensions{}),
        other_dtype_(other.scalar_type()),
        other_device_(other.device()),
        mask_(mask) {}

  std::string name() const override { return "MatmulBackwardBackward0"; }

  static at::Tensor ComputeGradGrad(const at::Tensor& self,
                                    const at::Tensor& other,
                                    const at::Tensor& gg_self,
                                    const at::Tensor& gg_other, bool mask0,
                                    bool mask1) {
    const bool has_first = mask0 && gg_self.defined();
    const bool has_second = mask1 && gg_other.defined();
    if (has_first && has_second) {
      return at::add(at::matmul(gg_self, other), at::matmul(self, gg_other));
    }
    if (has_first) {
      return at::matmul(gg_self, other);
    }
    if (has_second) {
      return at::matmul(self, gg_other);
    }
    return at::Tensor();
  }

  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& grads) override {
    if (grads.size() < 2) {
      return {at::Tensor(), at::Tensor(), at::Tensor()};
    }
    const at::Tensor& gg_self = grads[0];
    const at::Tensor& gg_other = grads[1];
    const bool compute_grad_grad = should_compute_output(0);
    const bool compute_grad_self =
        should_compute_output(1) && mask_[1] && gg_other.defined();
    const bool compute_grad_other =
        should_compute_output(2) && mask_[0] && gg_self.defined();
    const bool need_grad_val =
        compute_grad_self || compute_grad_other || !can_use_metadata_;
    const bool need_self_val =
        (compute_grad_grad && mask_[1] && gg_other.defined()) ||
        !can_use_metadata_;
    const bool need_other_val =
        (compute_grad_grad && mask_[0] && gg_self.defined()) ||
        !can_use_metadata_;
    const at::Tensor grad = UnpackOrMakeMetadataCarrier(
        grad_, need_grad_val, grad_sizes_, grad_dtype_, grad_device_);
    const at::Tensor self = UnpackOrMakeMetadataCarrier(
        self_, need_self_val, self_sizes_, self_dtype_, self_device_);
    const at::Tensor other = UnpackOrMakeMetadataCarrier(
        other_, need_other_val, other_sizes_, other_dtype_, other_device_);

    at::Tensor grad_grad = compute_grad_grad
                               ? ComputeGradGrad(self, other, gg_self, gg_other,
                                                 mask_[0], mask_[1])
                               : at::Tensor();

    at::Tensor grad_self = compute_grad_self
                               ? std::get<0>(at::matmul_backward(
                                     grad, self, gg_other, {true, false}))
                               : at::Tensor();

    at::Tensor grad_other = compute_grad_other
                                ? std::get<1>(at::matmul_backward(
                                      grad, gg_self, other, {false, true}))
                                : at::Tensor();

    return {std::move(grad_grad), std::move(grad_self), std::move(grad_other)};
  }

  void release_variables() override {
    grad_.reset_data();
    self_.reset_data();
    other_.reset_data();
  }

 private:
  bool can_use_metadata_;
  torch::autograd::SavedVariable grad_;
  torch::autograd::SavedVariable self_;
  torch::autograd::SavedVariable other_;
  Dimensions grad_sizes_;
  at::ScalarType grad_dtype_;
  c10::Device grad_device_;
  Dimensions self_sizes_;
  at::ScalarType self_dtype_;
  c10::Device self_device_;
  Dimensions other_sizes_;
  at::ScalarType other_dtype_;
  c10::Device other_device_;
  std::array<bool, 2> mask_;
};

// First-order autograd node for `aten::matmul(self, other)`. Dispatches to
// `at::matmul_backward` with `mask = {should_compute_output(0),
// should_compute_output(1)}`.
//
// When both `self` and `other` have static non-Fake shapes
// (`can_use_metadata_`), `grad_self = grad @ other^H` only requires `other`'s
// values (and `self`'s shape/dtype/device metadata), and `grad_other =
// self^H @ grad` only requires `self`'s values (and `other`'s metadata). Thus
// `self_` is only saved as a `SavedVariable` when `other.requires_grad()` (or
// when either input has dynamic/symbolic/Fake shapes), and `other_` is only
// saved when `self.requires_grad()` (or when either input has
// dynamic/symbolic/Fake shapes). This avoids retaining unused activation
// storage and prevents false-positive version-counter errors when an operand
// whose paired input does not require gradients is mutated in-place after the
// forward pass.
class MatmulAutogradNode : public torch::autograd::Node {
 public:
  MatmulAutogradNode(const at::Tensor& self, const at::Tensor& other)
      : can_use_metadata_(!HasDynamicShape(self) && !HasDynamicShape(other) &&
                          !IsMetaOrFakeTensor(self) &&
                          !IsMetaOrFakeTensor(other)),
        self_((other.requires_grad() || !can_use_metadata_)
                  ? torch::autograd::SavedVariable(self, /*is_output=*/false)
                  : torch::autograd::SavedVariable()),
        other_((self.requires_grad() || !can_use_metadata_)
                   ? torch::autograd::SavedVariable(other, /*is_output=*/false)
                   : torch::autograd::SavedVariable()),
        self_sizes_(can_use_metadata_
                        ? Dimensions(self.sizes().begin(), self.sizes().end())
                        : Dimensions{}),
        self_dtype_(self.scalar_type()),
        self_device_(self.device()),
        other_sizes_(can_use_metadata_ ? Dimensions(other.sizes().begin(),
                                                    other.sizes().end())
                                       : Dimensions{}),
        other_dtype_(other.scalar_type()),
        other_device_(other.device()) {}

  std::string name() const override { return "MatmulBackward0"; }

  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& grads) override {
    if (grads.empty() || !grads[0].defined()) {
      return {at::Tensor(), at::Tensor()};
    }
    const std::array<bool, 2> mask = {should_compute_output(0),
                                      should_compute_output(1)};
    if (!mask[0] && !mask[1]) {
      return {at::Tensor(), at::Tensor()};
    }
    const at::Tensor self =
        UnpackOrMakeMetadataCarrier(self_, mask[1] || !can_use_metadata_,
                                    self_sizes_, self_dtype_, self_device_);
    const at::Tensor other =
        UnpackOrMakeMetadataCarrier(other_, mask[0] || !can_use_metadata_,
                                    other_sizes_, other_dtype_, other_device_);
    auto [grad_self, grad_other] =
        at::matmul_backward(grads[0], self, other, mask);
    return {std::move(grad_self), std::move(grad_other)};
  }

  void release_variables() override {
    self_.reset_data();
    other_.reset_data();
  }

 private:
  bool can_use_metadata_;
  torch::autograd::SavedVariable self_;
  torch::autograd::SavedVariable other_;
  Dimensions self_sizes_;
  at::ScalarType self_dtype_;
  c10::Device self_device_;
  Dimensions other_sizes_;
  at::ScalarType other_dtype_;
  c10::Device other_device_;
};

}  // namespace

at::Tensor AtenMatmul(const at::Tensor& self, const at::Tensor& other) {
  TT_KERNEL(OpName::kMatmul, param_keys, (self, other), {
    TT_ASSIGN_OR_THROW(
        DeviceBufferRef result_buf,
        DispatchMatmul<absl::StatusOr<DeviceBufferRef>>(
            self, other, /*out=*/std::nullopt, std::move(param_keys),
            [&self, &other](auto op_builder, auto options) {
              return DispatchOp<2>(std::move(op_builder), {self, other},
                                   std::move(options));
            }));
    return MakeTensor(std::move(result_buf));
  });
}

// Custom `AutogradPrivateUse1` kernel for `aten::matmul`. Unlike upstream
// `at::native::matmul` (which decomposes into `dot`/`mm`/`bmm`/`mv` and loses
// unified `aten::matmul_backward` fusion), this kernel redispatches below
// `ADInplaceOrView` to `AtenMatmul` and records `MatmulAutogradNode` on the
// output. When `ShouldFallbackMatmulAutograd` is true, delegates to
// `SymInt`-aware `DecomposeMatmul`.
at::Tensor AtenMatmulAutograd(const at::Tensor& self, const at::Tensor& other) {
  if (ShouldFallbackMatmulAutograd(self, other)) {
    return DecomposeMatmul(self, other);
  }
  at::Tensor result = [&self, &other]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::matmul(self, other);
  }();
  if (torch::autograd::compute_requires_grad(self, other)) {
    auto grad_fn = MakeAutogradNode<MatmulAutogradNode>(self, other);
    grad_fn->set_next_edges(torch::autograd::collect_next_edges(self, other));
    torch::autograd::set_history(result, std::move(grad_fn));
  }
  return result;
}

at::Tensor& AtenMatmulOut(const at::Tensor& self, const at::Tensor& other,
                          at::Tensor& out) {
  TT_KERNEL(OpName::kMatmulOut, param_keys, (self, other, out), {
    TT_THROW_IF_ERROR(DispatchMatmul<absl::Status>(
        self, other, out, std::move(param_keys),
        [&self, &other, &out](auto op_builder, auto options) {
          return DispatchOpOut<2>(std::move(op_builder), {self, other}, out,
                                  std::move(options));
        }));
    return out;
  });
}

// Custom `AutogradPrivateUse1` kernel for `aten::matmul.out`. Rejects calls
// where any input or `out` requires gradients under `GradMode` (after letting
// 0D inputs fall through to the rank check, matching upstream `LinearAlgebra`),
// delegates functorch transforms, Python wrapper subclasses, and forward-mode
// AD to `at::native::matmul_out`, and otherwise redispatches below `Autograd`
// so `ADInplaceOrView` bumps `out`'s version counter.
at::Tensor& AtenMatmulOutAutograd(const at::Tensor& self,
                                  const at::Tensor& other, at::Tensor& out) {
  if (ShouldFallbackMatmulOutAutograd(self, other, out)) {
    return at::native::matmul_out(self, other, out);
  }
  if (self.dim() >= 1 && other.dim() >= 1 && out.defined() &&
      torch::autograd::compute_requires_grad(self, other, out)) {
    torch::autograd::throw_error_out_requires_grad("matmul");
  }
  at::AutoDispatchBelowAutograd guard;
  return at::matmul_out(out, self, other);
}

std::tuple<at::Tensor, at::Tensor> AtenMatmulBackward(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    const std::array<bool, 2> mask) {
  TT_KERNEL(OpName::kMatmulBackward, param_keys, (grad, self, other, mask), {
    TT_ASSIGN_OR_THROW(auto results, MatmulBackwardImpl(grad, self, other, mask,
                                                        std::move(param_keys)));
    return results;
  });
}

std::tuple<at::Tensor&, at::Tensor&> AtenMatmulBackwardOut(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    const std::array<bool, 2> mask, at::Tensor& grad_self,
    at::Tensor& grad_other) {
  TT_KERNEL(OpName::kMatmulBackwardOut, param_keys,
            (grad, self, other, mask, grad_self, grad_other), {
              TT_THROW_IF_ERROR(MatmulBackwardOutImpl(grad, self, other, mask,
                                                      grad_self, grad_other,
                                                      std::move(param_keys)));
              return std::forward_as_tuple(grad_self, grad_other);
            });
}

// Custom `AutogradPrivateUse1` kernel for `aten::matmul_backward`, attaching
// `MatmulBackwardAutogradNode` so higher-order derivatives (e.g. Hessian-vector
// products and double-backward tests) differentiate through `matmul_backward`.
std::tuple<at::Tensor, at::Tensor> AtenMatmulBackwardAutograd(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    const std::array<bool, 2> mask) {
  if (ShouldFallbackMatmulBackwardAutograd(grad, self, other)) {
    return DecomposeMatmulBackward(grad, self, other, mask);
  }
  auto [grad_self, grad_other] = [&grad, &self, &other, &mask]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::matmul_backward(grad, self, other, mask);
  }();
  if ((grad_self.defined() || grad_other.defined()) &&
      torch::autograd::compute_requires_grad(grad, self, other)) {
    auto grad_fn =
        MakeAutogradNode<MatmulBackwardAutogradNode>(grad, self, other, mask);
    grad_fn->set_next_edges(
        torch::autograd::collect_next_edges(grad, self, other));
    torch::autograd::set_history(
        torch::autograd::flatten_tensor_args(grad_self, grad_other),
        std::move(grad_fn));
  }
  return std::make_tuple(std::move(grad_self), std::move(grad_other));
}

// `Meta` dispatch kernel for `aten::matmul_backward`. `IgnoreInCacheKey` is
// used for `mask` because `TT_KERNEL` enforces that all non-tensor parameters
// are accounted for in `OpParamCacheKeys`, whereas `Meta` kernels only allocate
// `c10::kMeta` tensors with symbolic shapes (`empty_symint`) and never compile
// or look up an XLA computation cache entry.
std::tuple<at::Tensor, at::Tensor> MetaMatmulBackward(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    const std::array<bool, 2> mask) {
  TT_KERNEL(
      OpName::kMatmulBackward, _,
      (grad, self, other,
       IgnoreInCacheKey(mask, "Meta kernel does not dispatch to XLA")),
      {
        TT_THROW_IF_ERROR(
            ValidateMatmulBackwardInputs(grad, self, other, mask,
                                         /*grad_self_out=*/std::nullopt,
                                         /*grad_other_out=*/std::nullopt));
        at::Tensor grad_self =
            mask[0] ? at::empty_symint(
                          self.sym_sizes(),
                          self.options()
                              .device(c10::kMeta)
                              .memory_format(c10::MemoryFormat::Contiguous))
                    : at::Tensor();
        at::Tensor grad_other =
            mask[1] ? at::empty_symint(
                          other.sym_sizes(),
                          other.options()
                              .device(c10::kMeta)
                              .memory_format(c10::MemoryFormat::Contiguous))
                    : at::Tensor();
        return std::make_tuple(std::move(grad_self), std::move(grad_other));
      });
}

// `Meta` dispatch kernel for `aten::matmul_backward.out`. Validates inputs and
// resizes defined `grad_self` / `grad_other` meta tensors (or allocates new
// contiguous `c10::kMeta` tensors when undefined) to `self` / `other` symbolic
// sizes without XLA compilation (`IgnoreInCacheKey` on `mask`).
std::tuple<at::Tensor&, at::Tensor&> MetaMatmulBackwardOut(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    const std::array<bool, 2> mask, at::Tensor& grad_self,
    at::Tensor& grad_other) {
  TT_KERNEL(OpName::kMatmulBackwardOut, _,
            (grad, self, other,
             IgnoreInCacheKey(mask, "Meta kernel does not dispatch to XLA"),
             grad_self, grad_other),
            {
              TT_THROW_IF_ERROR(ValidateMatmulBackwardInputs(
                  grad, self, other, mask, grad_self, grad_other));
              if (mask[0]) {
                if (grad_self.defined()) {
                  grad_self.resize__symint(self.sym_sizes());
                } else {
                  grad_self = at::empty_symint(
                      self.sym_sizes(),
                      self.options()
                          .device(c10::kMeta)
                          .memory_format(c10::MemoryFormat::Contiguous));
                }
              }
              if (mask[1]) {
                if (grad_other.defined()) {
                  grad_other.resize__symint(other.sym_sizes());
                } else {
                  grad_other = at::empty_symint(
                      other.sym_sizes(),
                      other.options()
                          .device(c10::kMeta)
                          .memory_format(c10::MemoryFormat::Contiguous));
                }
              }
              return std::forward_as_tuple(grad_self, grad_other);
            });
}

}  // namespace torch_tpu
