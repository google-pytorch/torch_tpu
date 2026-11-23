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

#include "csrc/ops/masked_select/masked_select_aten_kernels.h"

#include <optional>

#include "ATen/MemoryOverlap.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/broadcast_tensors.h"
#include "absl/status/status.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/device_type.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/copy_from/copy_from_aten_kernels.h"
#include "csrc/ops/gather/gather_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/nonzero/nonzero_aten_kernels.h"
#include "csrc/ops/nullary_aten_kernels.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"

namespace torch_tpu {

namespace {

// MaskedSelectFlat implements masked_select by flattening the input tensors,
// extracting the indices of the true values in the mask using the nonzero op,
// and then using a gather operation to select the true values from the input
// tensor.
at::Tensor MaskedSelectFlat(const at::Tensor& self, const at::Tensor& mask) {
  // Flatten the tensors at all dimensions.
  const at::Tensor mask_flat = mask.flatten();
  const at::Tensor self_flat = self.flatten();

  // Get indices of true values in mask. nonzero on 1D returns (N, 1), so
  // flatten it to (N,). Note that AtenNonzero synchronizes with the host to
  // determine N.
  const at::Tensor indices = AtenNonzero(mask_flat).flatten();

  if (indices.numel() == 0) {
    return AtenEfficientZeroTensor({0}, self.scalar_type(), std::nullopt,
                                   self.device(), std::nullopt);
  }

  return AtenGather(self_flat, 0, indices, /*sparse_grad=*/false);
}

// Verifies that the mask tensor has boolean scalar type.
absl::Status ValidateMaskTensorIsBool(const at::Tensor& mask) {
  TT_RET_CHECK(IsBool(mask), error::kInvalidArgument)
      << "expected mask to be a BoolTensor, got "
      << ToString(mask.scalar_type());
  return absl::OkStatus();
}

// Validates out device, mask dtype, and out dtype before executing
// masked_select or resizing out.
absl::Status ValidateMaskedSelectOutInputs(const at::Tensor& self,
                                           const at::Tensor& mask,
                                           const at::Tensor& out) {
  TT_RET_CHECK(IsPrivateUse1Device(out), error::kInvalidArgument)
      << "expected out tensor to be on " << GetPrivateUse1DeviceDebugName()
      << ", got " << out.device().str();
  TT_RETURN_IF_ERROR(ValidateMaskTensorIsBool(mask));
  TT_RET_CHECK(self.scalar_type() == out.scalar_type(), error::kInvalidArgument)
      << "expected out tensor to have dtype " << ToString(self.scalar_type())
      << ", got " << ToString(out.scalar_type());
  return absl::OkStatus();
}

}  // namespace

// AtenMaskedSelect implements torch.masked_select by delegating to the nonzero
// op to find the selected indices and gathering them from the input.
at::Tensor AtenMaskedSelect(const at::Tensor& self, const at::Tensor& mask) {
  TT_KERNEL(OpName::kMaskedSelect, _, (self, mask), {
    TT_THROW_IF_ERROR(ValidateMaskTensorIsBool(mask));

    const auto broadcasted = at::broadcast_tensors({mask, self});
    const at::Tensor& mask_broadcasted = broadcasted[0];
    const at::Tensor& self_broadcasted = broadcasted[1];

    return MaskedSelectFlat(self_broadcasted, mask_broadcasted);
  });
}

// AtenMaskedSelectOut implements torch.masked_select.out by computing the
// selection and copying the result into the provided out tensor.
at::Tensor& AtenMaskedSelectOut(const at::Tensor& self, const at::Tensor& mask,
                                at::Tensor& out) {
  // Note: Using OpName::kMaskedSelect instead of kMaskedSelectOut so that
  // subsequent AtenCopyFrom execution is recognized under the known composite
  // op stack.
  TT_KERNEL(OpName::kMaskedSelect, _, (self, mask, out), {
    TT_THROW_IF_ERROR(ValidateMaskedSelectOutInputs(self, mask, out));
    at::assert_no_internal_overlap(out);
    at::assert_no_overlap(out, self);
    at::assert_no_overlap(out, mask);

    const at::Tensor result = AtenMaskedSelect(self, mask);
    TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, result.sizes()));
    out = AtenCopyFrom(result, out, /*non_blocking=*/true);
    return out;
  });
}

}  // namespace torch_tpu
