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

#include "csrc/ops/gather/gather_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/MemoryOverlap.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/Storage.h"
#include "c10/core/TensorImpl.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/gather/gather.h"
#include "csrc/ops/index_utils.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

// Upstream PyTorch skips index dtype checks when `index` is empty, so callers
// must invoke this only after handling `index.numel() == 0`.
absl::Status ValidateGatherIndexDtype(const at::Tensor& index) {
  TT_RET_CHECK(index.scalar_type() == at::ScalarType::Long ||
                   index.scalar_type() == at::ScalarType::Int,
               error::kInvalidArgument)
      << "expected the index dtype to be int32 or int64, got "
      << ToString(index.scalar_type());
  return absl::OkStatus();
}

// Checks memory overlap on `out` against its post-resize layout without
// mutating `out`, so `out` preserves its shape and strides if any check throws.
void CheckGatherOutMemoryOverlap(const at::Tensor& self,
                                 const at::Tensor& index,
                                 const at::Tensor& out) {
  if (out.sizes() == index.sizes()) {
    at::assert_no_internal_overlap(out);
    at::assert_no_overlap(out, self);
    at::assert_no_partial_overlap(out, index);
    return;
  }
  // Check exact TensorImpl identity before creating `resized_out_view`: when
  // `index` is empty, `resized_out_view` has 0 elements and `assert_no_overlap`
  // below would return `No` instead of rejecting `out=self`.
  if (out.is_same(self)) {
    at::assert_no_overlap(out, self);
  }
  // ResizeTensor resets `out` to a contiguous layout at storage_offset 0 on the
  // same c10::StorageImpl, which has no internal overlap.
  c10::Storage storage_copy = out.storage();
  at::Tensor resized_out_view(c10::make_intrusive<c10::TensorImpl>(
      c10::TensorImpl::VIEW, std::move(storage_copy), out.key_set(),
      out.dtype()));
  resized_out_view.unsafeGetTensorImpl()->set_sizes_contiguous(index.sizes());
  resized_out_view.unsafeGetTensorImpl()->set_storage_offset(0);
  at::assert_no_overlap(resized_out_view, self);
  at::assert_no_partial_overlap(resized_out_view, index);
}

absl::StatusOr<DeviceBufferRef> Gather(const at::Tensor& self, int64_t dim,
                                       const at::Tensor& index,
                                       bool sparse_grad,
                                       OpParamCacheKeys&& param_keys) {
  TT_RETURN_IF_ERROR(GetBaseBuffer(self).status());
  TT_RETURN_IF_ERROR(GetBaseBuffer(index).status());
  TT_ASSIGN_OR_RETURN(dim, SafeWrapDim(dim, self.dim()));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_RETURN_IF_ERROR(ValidateGatherSparseGrad(sparse_grad));
  Dimensions output_dims = CopyIntVector(index.sizes());

  if (index.numel() == 0) {
    // Gathering with 0 indexes results in a zero-sized tensor.
    return CreateZeroSizeDeviceBufferRef(output_dims, output_dtype);
  }

  TT_RETURN_IF_ERROR(ValidateGatherIndexDtype(index));
  TT_RETURN_IF_ERROR(
      ValidateGatherInputs(self.sizes(), dim, index.sizes(), sparse_grad));

  if (auto unexpanded = TryUnexpandZeroStrideIndex(index)) {
    Dimensions squeezed_dims = std::move(unexpanded->squeezed_dims);
    Dimensions bcast_dims = std::move(unexpanded->bcast_dims);
    Dimensions logical_dims = std::move(unexpanded->logical_dims);
    TT_ASSIGN_OR_RETURN(
        auto full_param_keys,
        TT_MAKE_OP_PARAM_CACHE_KEYS(dim, sparse_grad, squeezed_dims, bcast_dims,
                                    logical_dims));
    auto gather_op_builder = [dim, sparse_grad, output_dtype, squeezed_dims,
                              bcast_dims, logical_dims](
                                 FixedSizeSpan<mlir::MlirOp, 2> inputs) {
      auto& [self_op, index_op] = inputs;
      const auto index_type = GetTensorTypeOrDie(index_op);
      if (index_type.getRank() != static_cast<int64_t>(squeezed_dims.size())) {
        index_op = mlir::stablehlo::Reshape(index_op, squeezed_dims);
      }
      const auto logical_type = mlir::RankedTensorType::get(
          logical_dims, index_type.getElementType());
      mlir::MlirOp expanded_index =
          mlir::stablehlo::BroadcastInDim(logical_type, index_op, bcast_dims);
      return BuildGatherShlo(self_op, dim, expanded_index, sparse_grad,
                             output_dtype);
    };
    return DispatchOp<2>(std::move(gather_op_builder),
                         {self, unexpanded->tensor},
                         {.out_dtype = output_dtype,
                          .out_dims = output_dims,
                          .op_param_cache_keys = std::move(full_param_keys)});
  }

  auto gather_op_builder =
      [dim, sparse_grad, output_dtype](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
        auto& [self, index] = inputs;
        return BuildGatherShlo(self, dim, index, sparse_grad, output_dtype);
      };

  return DispatchOp<2>(std::move(gather_op_builder), {self, index},
                       {.out_dtype = output_dtype,
                        .out_dims = output_dims,
                        .op_param_cache_keys = std::move(param_keys)});
}

absl::Status GatherOut(const at::Tensor& self, int64_t dim,
                       const at::Tensor& index, bool sparse_grad,
                       OpParamCacheKeys&& param_keys, at::Tensor& out) {
  TT_RET_CHECK(IsPrivateUse1Device(out), error::kInvalidArgument)
      << "expected out tensor to be on " << GetPrivateUse1DeviceDebugName()
      << ", got " << out.device().str();
  TT_RETURN_IF_ERROR(GetBaseBuffer(self).status());
  TT_RETURN_IF_ERROR(GetBaseBuffer(index).status());
  TT_ASSIGN_OR_RETURN(dim, SafeWrapDim(dim, self.dim()));
  TT_RETURN_IF_ERROR(ValidateOutDtype(out, self.scalar_type(),
                                      /*allow_out_dtype_cast=*/false));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  CheckGatherOutMemoryOverlap(self, index, out);
  TT_RETURN_IF_ERROR(ValidateGatherSparseGrad(sparse_grad));

  if (index.numel() != 0) {
    TT_RETURN_IF_ERROR(ValidateGatherIndexDtype(index));
    TT_RETURN_IF_ERROR(
        ValidateGatherInputs(self.sizes(), dim, index.sizes(), sparse_grad));
  }

  TT_RETURN_IF_ERROR(ResizeTensorIfShapeDiffers(out, index.sizes()));
  if (index.numel() == 0) {
    return absl::OkStatus();
  }

  Dimensions output_dims = CopyIntVector(index.sizes());

  if (auto unexpanded = TryUnexpandZeroStrideIndex(index)) {
    Dimensions squeezed_dims = std::move(unexpanded->squeezed_dims);
    Dimensions bcast_dims = std::move(unexpanded->bcast_dims);
    Dimensions logical_dims = std::move(unexpanded->logical_dims);
    TT_ASSIGN_OR_RETURN(
        auto full_param_keys,
        TT_MAKE_OP_PARAM_CACHE_KEYS(dim, sparse_grad, squeezed_dims, bcast_dims,
                                    logical_dims));
    auto gather_op_builder = [dim, sparse_grad, output_dtype, squeezed_dims,
                              bcast_dims, logical_dims](
                                 FixedSizeSpan<mlir::MlirOp, 2> inputs) {
      auto& [self_op, index_op] = inputs;
      const auto index_type = GetTensorTypeOrDie(index_op);
      if (index_type.getRank() != static_cast<int64_t>(squeezed_dims.size())) {
        index_op = mlir::stablehlo::Reshape(index_op, squeezed_dims);
      }
      const auto logical_type = mlir::RankedTensorType::get(
          logical_dims, index_type.getElementType());
      mlir::MlirOp expanded_index =
          mlir::stablehlo::BroadcastInDim(logical_type, index_op, bcast_dims);
      return BuildGatherShlo(self_op, dim, expanded_index, sparse_grad,
                             output_dtype);
    };
    return DispatchOpOut<2, 1>(
        std::move(gather_op_builder), {self, unexpanded->tensor}, out,
        {.out_dtype = output_dtype,
         .out_dims = output_dims,
         .op_param_cache_keys = std::move(full_param_keys)});
  }

  auto gather_op_builder =
      [dim, sparse_grad, output_dtype](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
        auto& [self, index] = inputs;
        return BuildGatherShlo(self, dim, index, sparse_grad, output_dtype);
      };

  return DispatchOpOut<2, 1>(std::move(gather_op_builder), {self, index}, out,
                             {.out_dtype = output_dtype,
                              .out_dims = output_dims,
                              .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor AtenGather(const at::Tensor& self, int64_t dim,
                      const at::Tensor& index, bool sparse_grad) {
  TT_KERNEL(OpName::kGather, param_keys, (self, dim, index, sparse_grad), {
    TT_ASSIGN_OR_THROW(auto result, Gather(self, dim, index, sparse_grad,
                                           std::move(param_keys)));
    return MakeTensor(std::move(result));
  });
}

at::Tensor& AtenGatherOut(const at::Tensor& self, int64_t dim,
                          const at::Tensor& index, bool sparse_grad,
                          at::Tensor& out) {
  TT_KERNEL(OpName::kGatherOut, param_keys,
            (self, dim, index, sparse_grad, out), {
              TT_THROW_IF_ERROR(GatherOut(self, dim, index, sparse_grad,
                                          std::move(param_keys), out));
              return out;
            });
}

}  // namespace torch_tpu
