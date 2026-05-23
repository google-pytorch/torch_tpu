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

#include "torch_tpu/ops/masked_select/masked_select_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/broadcast_tensors.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_types.h"
#include "torch_tpu/ops/copy_from/copy_from_aten_kernels.h"
#include "torch_tpu/ops/gather/gather_aten_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/reductions/reduction_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"
#include "torch_tpu/ops/resize/resize_aten_kernels.h"
#include "torch_tpu/ops/sort/sort_aten_kernels.h"

namespace torch_tpu {

namespace {

// GetMaskedSelectOutputSize returns the output size of the masked_select op by
// summing the mask.
absl::StatusOr<at::Tensor> GetMaskedSelectOutputSize(const at::Tensor& mask) {
  c10::ScalarType scalar_dtype = c10::ScalarType::Long;
  TT_ASSIGN_OR_RETURN(
      at::Tensor result,
      ApplySumReduction(mask, std::nullopt, ReductionMode::kDropDims,
                        scalar_dtype));
  return result;
}

// TODO(b/444058028): Improve implementation by leveraging nonzeros.
// MaskedSelectWithKnownOutputShape is a helper function that implements
// masked_select when the output shape is known. It does this by flattening
// the input tensors, sorting the flattened mask, slicing the sorted indices
// to get the indices of the true values in the mask, and then using a gather
// operation to select the true values from the input tensors.
at::Tensor MaskedSelectWithKnownOutputShape(const at::Tensor& self,
                                            const at::Tensor& mask,
                                            int64_t num_selected_elems) {
  // Flatten the tensors at all dimensions.
  at::Tensor mask_flat = mask.flatten();
  at::Tensor self_flat = self.flatten();

  // Get indices of true values in mask.
  TT_ASSIGN_OR_THROW(at::Tensor mask_flat_values,
                     MakeEmptyTensor(mask_flat.sizes(), mask_flat.scalar_type(),
                                     mask_flat.device()));
  TT_ASSIGN_OR_THROW(
      at::Tensor mask_flat_indices,
      MakeEmptyTensor(mask_flat.sizes(), at::kLong, mask_flat.device()));
  AtenSortValuesStable(mask_flat, /*stable_opt=*/true, /*dim=*/0,
                       /*descending=*/true, mask_flat_values,
                       mask_flat_indices);

  // Slice the indices to get elements with true values in mask. We get the
  // first `num_selected_elems` elements.
  at::Tensor sliced_indices =
      mask_flat_indices.slice(0, 0, num_selected_elems, 1);
  // Use flattened self tensor as input to gather op.

  TT_ASSIGN_OR_THROW(at::Tensor result, MakeEmptyTensor(sliced_indices.sizes(),
                                                        self_flat.scalar_type(),
                                                        self_flat.device()));
  return AtenGatherOut(self_flat, 0, sliced_indices, false, result);
}

absl::Status PrepareOutTensor(const at::Tensor& result, at::Tensor& out) {
  TT_RET_CHECK(out.device().type() == GetPrivateUse1DeviceType(),
               error::kInvalidArgument)
      << "the out tensor is expected to be on tpu, got " << out.device().str();

  // Make out have the same shape as result.
  out = AtenResize_(out, result.sizes(), std::nullopt);

  // Check that the out tensor dtype is the same as the result dtype.
  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(out.scalar_type()));
  TT_ASSIGN_OR_RETURN(const auto result_dtype,
                      ConvertTo<mlir::ElementType>(result.scalar_type()));
  TT_RET_CHECK(result.scalar_type() == out.scalar_type(),
               error::kInvalidArgument)
      << "the out tensor dtype is expected to be " << ToString(result_dtype)
      << ", got " << ToString(out_dtype);
  return absl::OkStatus();
}

absl::Status CheckMaskTensorIsBool(const at::Tensor& mask) {
  TT_RET_CHECK(IsBool(mask), error::kInvalidArgument)
      << "expected the mask to be bool, got " << ToString(mask.scalar_type());
  return absl::OkStatus();
}

}  // namespace

at::Tensor AtenMaskedSelect(const at::Tensor& self, const at::Tensor& mask) {
  TT_KERNEL(OpName::kMaskedSelect, _, (self, mask), {
    TT_THROW_IF_ERROR(CheckMaskTensorIsBool(mask));

    auto broadcasted = at::broadcast_tensors({mask, self});
    at::Tensor& mask_broadcasted = broadcasted[0];
    at::Tensor& self_broadcasted = broadcasted[1];

    // Get the size of the mask.
    TT_ASSIGN_OR_THROW(at::Tensor size,
                       GetMaskedSelectOutputSize(mask_broadcasted));

    // Move `size` to the CPU. This transition is necessary to
    // dispatch the op afterwards.
    int64_t size_cpu = size.cpu().item<int64_t>();

    return MaskedSelectWithKnownOutputShape(self_broadcasted, mask_broadcasted,
                                            size_cpu);
  });
}

at::Tensor& AtenMaskedSelectOut(const at::Tensor& self, const at::Tensor& mask,
                                at::Tensor& out) {
  TT_KERNEL(OpName::kMaskedSelect, _, (self, mask, out), {
    TT_THROW_IF_ERROR(CheckMaskTensorIsBool(mask));

    at::Tensor result = AtenMaskedSelect(self, mask);
    TT_THROW_IF_ERROR(PrepareOutTensor(result, out));

    out = AtenCopyFrom(result, out, /*non_blocking=*/true);
    return out;
  });
}

}  // namespace torch_tpu
