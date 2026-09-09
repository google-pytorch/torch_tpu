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

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/broadcast_tensors.h"
#include "absl/functional/bind_front.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/copy_from/copy_from_aten_kernels.h"
#include "csrc/ops/gather/gather_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/nonzero/nonzero.h"
#include "csrc/ops/nullary_aten_kernels.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/reduction_utils.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

// GetMaskedSelectOutputSize returns the output size of the masked_select op by
// summing the mask.
absl::StatusOr<at::Tensor> GetMaskedSelectOutputSize(const at::Tensor& mask) {
  const c10::ScalarType scalar_dtype = c10::ScalarType::Long;
  TT_ASSIGN_OR_RETURN(
      const at::Tensor result,
      ApplySumReduction(mask, std::nullopt, ReductionMode::kDropDims,
                        scalar_dtype));
  return result;
}

// MaskedSelectWithKnownOutputShape is a helper function that implements
// masked_select when the output shape is known. It does this by flattening
// the input tensors, extracting the indices of the true values in the mask
// using a Nonzero operation, and then using a gather operation to select the
// true values from the input tensors.
at::Tensor MaskedSelectWithKnownOutputShape(const at::Tensor& self,
                                            const at::Tensor& mask,
                                            const int64_t num_selected_elems) {
  if (num_selected_elems == 0) {
    return AtenEfficientZeroTensor({0}, self.scalar_type(), std::nullopt,
                                   self.device(), std::nullopt);
  }

  // Flatten the tensors at all dimensions.
  const at::Tensor mask_flat = mask.flatten();
  const at::Tensor self_flat = self.flatten();

  // Get indices of true values in mask.
  TT_ASSIGN_OR_THROW(OpParamCacheKeys param_keys,
                     TT_MAKE_OP_PARAM_CACHE_KEYS(num_selected_elems));

  // nonzero on 1D returns (N, 1)
  Dimensions out_dims = {num_selected_elems, 1};

  TT_ASSIGN_OR_THROW(
      const at::Tensor mask_flat_indices,
      UnaryOp(mask_flat, absl::bind_front(BuildNonzeroShlo, num_selected_elems),
              {.op_name = OpName::kNonzero,
               .op_param_cache_keys = std::move(param_keys),
               .out_dtype = mlir::ElementType::I64,
               .out_dims = std::move(out_dims)}));

  const at::Tensor sliced_indices = mask_flat_indices.flatten();

  TT_ASSIGN_OR_THROW(at::Tensor result, MakeEmptyTensor(sliced_indices.sizes(),
                                                        self_flat.scalar_type(),
                                                        self_flat.device()));
  return AtenGatherOut(self_flat, 0, sliced_indices, /*sparse_grad=*/false,
                       result);
}

// PrepareOutTensor validates that the out tensor is on TPU, resizes it to match
// the result tensor's shape, and checks for dtype matching.
absl::Status PrepareOutTensor(const at::Tensor& result, at::Tensor& out) {
  TT_RET_CHECK(out.device().type() == GetPrivateUse1DeviceType(),
               error::kInvalidArgument)
      << "expected out tensor to be on tpu, got " << out.device().str();

  // Make out have the same shape as result.
  out = AtenResize_(out, result.sizes(), std::nullopt);

  // Check that the out tensor dtype is the same as the result dtype.
  TT_RET_CHECK(result.scalar_type() == out.scalar_type(),
               error::kInvalidArgument)
      << "expected out tensor to have dtype " << ToString(result.scalar_type())
      << ", got " << ToString(out.scalar_type());
  return absl::OkStatus();
}

// ValidateMaskTensorIsBool verifies that the mask tensor has boolean scalar
// type.
absl::Status ValidateMaskTensorIsBool(const at::Tensor& mask) {
  TT_RET_CHECK(IsBool(mask), error::kInvalidArgument)
      << "expected mask to be a BoolTensor, got "
      << ToString(mask.scalar_type());
  return absl::OkStatus();
}

}  // namespace

// AtenMaskedSelect implements torch.masked_select by computing the number of
// selected elements via sum reduction and delegating to
// MaskedSelectWithKnownOutputShape.
at::Tensor AtenMaskedSelect(const at::Tensor& self, const at::Tensor& mask) {
  TT_KERNEL(OpName::kMaskedSelect, _, (self, mask), {
    TT_THROW_IF_ERROR(ValidateMaskTensorIsBool(mask));

    const auto broadcasted = at::broadcast_tensors({mask, self});
    const at::Tensor& mask_broadcasted = broadcasted[0];
    const at::Tensor& self_broadcasted = broadcasted[1];

    // Get the size of the mask.
    TT_ASSIGN_OR_THROW(const at::Tensor size,
                       GetMaskedSelectOutputSize(mask_broadcasted));

    // Move `size` to the CPU. This transition is necessary to
    // dispatch the op afterwards.
    const int64_t size_cpu = size.cpu().item<int64_t>();

    return MaskedSelectWithKnownOutputShape(self_broadcasted, mask_broadcasted,
                                            size_cpu);
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
    TT_THROW_IF_ERROR(ValidateMaskTensorIsBool(mask));

    const at::Tensor result = AtenMaskedSelect(self, mask);
    TT_THROW_IF_ERROR(PrepareOutTensor(result, out));

    out = AtenCopyFrom(result, out, /*non_blocking=*/true);
    return out;
  });
}

}  // namespace torch_tpu
