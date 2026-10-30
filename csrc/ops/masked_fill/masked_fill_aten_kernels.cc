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

#include "csrc/ops/masked_fill/masked_fill_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/ops/empty.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/copy_from/copy_from_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/where/where.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

// Validates that `mask` has boolean data type.
absl::Status ValidateMaskIsBool(const at::Tensor& mask) {
  TT_RET_CHECK(IsBool(mask), error::kInvalidArgument)
      << "expected mask to be a boolean tensor, got "
      << ToString(mask.scalar_type());
  return absl::OkStatus();
}

// Validates that the broadcasted output shape matches `self`'s shape.
//
// Because masked_fill_ modifies `self` in place, broadcasting cannot expand
// the dimensions of `self`. Returns an invalid argument error if `mask`
// requires `self` to be expanded.
absl::Status ValidateInferredOutputShapeMatchesSelf(
    const at::Tensor& self, const Dimensions& out_dims) {
  const absl::Span<const int64_t> self_dims = self.sizes();
  TT_RET_CHECK(out_dims == self_dims, error::kInvalidArgument)
      << "expected the output shape (broadcast of self and mask) to match "
      << ToString(self_dims) << " (shape of self), got " << ToString(out_dims);
  return absl::OkStatus();
}

// Validates that `value` can be safely cast to `dtype` without overflow.
absl::Status ValidateValueCastableWithoutOverflow(const at::Scalar& value,
                                                  const at::ScalarType dtype) {
  TT_ASSIGN_OR_RETURN(const bool can_cast_without_overflow,
                      CanCastScalarWithoutOverflow(value, dtype));
  TT_RET_CHECK(can_cast_without_overflow, error::kInvalidArgument)
      << "expected value to be representable as " << ToString(dtype) << ", got "
      << ToString(value);
  return absl::OkStatus();
}

auto MakeMaskedFillBuilder(const mlir::ElementType self_mlir_type) {
  return [self_mlir_type](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
    auto& [input, mask, value] = inputs;
    // masked_fill(input, mask, value) == where(mask, value, input)
    return BuildWhereShlo(mask, value, input, self_mlir_type);
  };
}

// Dispatch logic for out-of-place masked fill op.
absl::StatusOr<at::Tensor> DispatchOutOfPlace(const at::Tensor& self,
                                              const at::Tensor& mask,
                                              const at::Tensor& value) {
  TT_RETURN_IF_ERROR(ValidateMaskIsBool(mask));
  TT_ASSIGN_OR_RETURN(const Dimensions out_dims, InferSize(self, mask));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType self_mlir_type,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef result_buf,
      DispatchOp<3>(MakeMaskedFillBuilder(self_mlir_type), {self, mask, value},
                    {.out_dtype = self_mlir_type,
                     .out_dims = out_dims,
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));
  return MakeTensor(std::move(result_buf));
}

// Dispatch logic for in-place masked fill op.
absl::Status DispatchInPlace(at::Tensor& self, const at::Tensor& mask,
                             const at::Tensor& value) {
  TT_RETURN_IF_ERROR(ValidateMaskIsBool(mask));
  TT_ASSIGN_OR_RETURN(const Dimensions out_dims, InferSize(self, mask));
  TT_RETURN_IF_ERROR(ValidateInferredOutputShapeMatchesSelf(self, out_dims));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType self_mlir_type,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  return DispatchOpOut<3>(MakeMaskedFillBuilder(self_mlir_type),
                          {self, mask, value}, self,
                          {.out_dtype = self_mlir_type,
                           .out_dims = self.sizes(),
                           .op_param_cache_keys = OpParamCacheKeys::Empty()});
}

// Validates that scalar `value` can be cast to `self_dtype` without overflow,
// materializes it as a tensor, and invokes `dispatch_fn`.
template <typename ReturnType, typename DispatchFn>
ReturnType ResolveScalarValueAndDispatch(const at::ScalarType self_dtype,
                                         PromotedScalar& value,
                                         DispatchFn&& dispatch_fn) {
  TT_RETURN_IF_ERROR(
      ValidateValueCastableWithoutOverflow(value.scalar(), self_dtype));
  TT_ASSIGN_OR_RETURN(const at::Tensor value_tensor,
                      value.GetTensor(self_dtype));
  return std::forward<DispatchFn>(dispatch_fn)(value_tensor);
}

// Validates a 0D tensor `value` (on either TPU or CPU) for scalar overflow
// and invokes `dispatch_fn`.
//
// To align with PyTorch GPU behavior, a TPU `value` is synchronously copied to
// CPU to check for scalar overflow before executing the fill, which triggers a
// host-device synchronization.
template <typename ReturnType, typename DispatchFn>
ReturnType ResolveTensorValueAndDispatch(const at::ScalarType self_dtype,
                                         const at::Tensor& value,
                                         DispatchFn&& dispatch_fn) {
  TT_RET_CHECK(value.dim() == 0, error::kInvalidArgument)
      << "expected value to be a 0D tensor, got " << value.dim()
      << "D tensor of shape " << ToString(value.sizes());

  // A CPU `value` has no device buffer to feed to the op, but PyTorch permits
  // it anyway. Read it on the host, as the GPU kernel does.
  if (IsCpuDevice(value)) {
    PromotedScalar promoted_value = PromoteScalar(value.item());
    return ResolveScalarValueAndDispatch<ReturnType>(
        self_dtype, promoted_value, std::forward<DispatchFn>(dispatch_fn));
  }

  // Force the materialization of `value` so that we can run the overflow
  // check. This aligns TorchTPU implementation with the GPU kernel.
  at::Tensor cpu_value = at::empty({}, value.options().device(c10::kCPU));
  CopyTensor(value, cpu_value, /* non_blocking= */ false);
  TT_RETURN_IF_ERROR(
      ValidateValueCastableWithoutOverflow(cpu_value.item(), self_dtype));
  return std::forward<DispatchFn>(dispatch_fn)(value);
}

}  // namespace

at::Tensor AtenMaskedFillScalar(const at::Tensor& self, const at::Tensor& mask,
                                const at::Scalar& value) {
  auto promoted_value = PromoteScalar(value);
  TT_KERNEL(OpName::kMaskedFillScalar, _, (self, mask, promoted_value), {
    TT_ASSIGN_OR_THROW(
        at::Tensor result,
        ResolveScalarValueAndDispatch<absl::StatusOr<at::Tensor>>(
            self.scalar_type(), promoted_value,
            [&]  // REF_CAPTURE_OK=called synchronously
            (const at::Tensor& value_tensor) {
              return DispatchOutOfPlace(self, mask, value_tensor);
            }));
    return result;
  });
}

at::Tensor AtenMaskedFillTensor(const at::Tensor& self, const at::Tensor& mask,
                                const at::Tensor& value) {
  TT_KERNEL(OpName::kMaskedFillTensor, _, (self, mask, value), {
    TT_ASSIGN_OR_THROW(
        at::Tensor result,
        ResolveTensorValueAndDispatch<absl::StatusOr<at::Tensor>>(
            self.scalar_type(), value,
            [&]  // REF_CAPTURE_OK=called synchronously
            (const at::Tensor& value_tensor) {
              return DispatchOutOfPlace(self, mask, value_tensor);
            }));
    return result;
  });
}

at::Tensor& AtenMaskedFill_Scalar(at::Tensor& self, const at::Tensor& mask,
                                  const at::Scalar& value) {
  auto promoted_value = PromoteScalar(value);
  TT_KERNEL(OpName::kMaskedFill_Scalar, _, (self, mask, promoted_value), {
    TT_THROW_IF_ERROR(ResolveScalarValueAndDispatch<absl::Status>(
        self.scalar_type(), promoted_value,
        [&]  // REF_CAPTURE_OK=called synchronously
        (const at::Tensor& value_tensor) {
          return DispatchInPlace(self, mask, value_tensor);
        }));
    return self;
  });
}

at::Tensor& AtenMaskedFill_Tensor(at::Tensor& self, const at::Tensor& mask,
                                  const at::Tensor& value) {
  TT_KERNEL(OpName::kMaskedFill_Tensor, _, (self, mask, value), {
    TT_THROW_IF_ERROR(ResolveTensorValueAndDispatch<absl::Status>(
        self.scalar_type(), value,
        [&]  // REF_CAPTURE_OK=called synchronously
        (const at::Tensor& value_tensor) {
          return DispatchInPlace(self, mask, value_tensor);
        }));
    return self;
  });
}

}  // namespace torch_tpu
