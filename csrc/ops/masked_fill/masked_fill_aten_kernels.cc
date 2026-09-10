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
#include "absl/types/span.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/copy_from/copy_from_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/where/where.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

namespace {

// Validates that broadcasting `mask` against `self` does not expand `self`'s
// shape.
//
// Because masked_fill_ modifies `self` in place, broadcasting cannot expand
// the dimensions of `self`. Returns an invalid argument error if `mask`
// requires `self` to be expanded.
absl::Status ValidateInferredOutputShapeMatchesSelf(const at::Tensor& self,
                                                    const at::Tensor& mask) {
  const absl::Span<const int64_t> self_dims = self.sizes();

  TT_ASSIGN_OR_RETURN(const Dimensions out_dims,
                      InferSize(self_dims, mask.sizes()));

  TT_RET_CHECK(out_dims == self_dims, error::kInvalidArgument)
      << "expected the output shape (broadcast of self and mask) to match "
      << ToString(self_dims) << " (shape of self), got " << ToString(out_dims);

  return absl::OkStatus();
}

// Validates that `value` can be safely cast to `dtype` without overflow.
absl::Status ValidateValueCastableWithoutOverflow(const at::Scalar& value,
                                                  at::ScalarType dtype) {
  TT_ASSIGN_OR_RETURN(bool can_cast_without_overflow,
                      CanCastScalarWithoutOverflow(value, dtype));
  TT_RET_CHECK(can_cast_without_overflow, error::kInvalidArgument)
      << "expected value to be representable as " << ToString(dtype) << ", got "
      << ToString(value);
  return absl::OkStatus();
}

// Dispatch logic for in-place masked fill op.
//
// Sets up the op builder using StableHLO `where` (`where(mask, value, input)`),
// and dispatches the operation.
absl::Status Dispatch(at::Tensor& self, const at::Tensor& mask,
                      const at::Tensor& value) {
  TT_RETURN_IF_ERROR(ValidateInferredOutputShapeMatchesSelf(self, mask));

  TT_ASSIGN_OR_RETURN(const auto self_mlir_type,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));

  auto op_builder = [self_mlir_type](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
    auto& [input, mask, value] = inputs;
    // masked_fill(input, mask, value) == where(mask, value, input)
    return BuildWhereShlo(mask, value, input, self_mlir_type);
  };

  return DispatchOpOut<3>(std::move(op_builder), {self, mask, value}, self,
                          {.out_dtype = self_mlir_type,
                           .out_dims = self.sizes(),
                           .op_param_cache_keys = OpParamCacheKeys::Empty()});
}

// Performs in-place masked fill using a scalar value.
//
// Checks that `value` can be safely cast to `self`'s data type without
// overflow before materializing it as a tensor and executing the fill.
absl::Status DispatchWithScalarValue(at::Tensor& self, const at::Tensor& mask,
                                     PromotedScalar& value) {
  const at::ScalarType dtype = self.scalar_type();
  TT_RETURN_IF_ERROR(  //
      ValidateValueCastableWithoutOverflow(value.scalar(), dtype));
  TT_ASSIGN_OR_RETURN(  //
      const at::Tensor value_tensor, value.GetTensor(dtype));
  return Dispatch(self, mask, std::move(value_tensor));
}

// Performs in-place masked fill using a 0D tensor value.
//
// Requires `value` to be a 0D tensor. To align with PyTorch GPU behavior,
// synchronously copies `value` to CPU to check for scalar overflow before
// executing the fill, which triggers a host-device synchronization.
absl::Status DispatchWithTensorValue(at::Tensor& self, const at::Tensor& mask,
                                     const at::Tensor& value) {
  TT_RET_CHECK(value.dim() == 0, error::kInvalidArgument)
      << "expected value to be a 0D tensor, got " << value.dim()
      << "D tensor of shape " << ToString(value.sizes());

  // Force the materialization of `value` so that we can run the overflow
  // check. This aligns TorchTPU implementation with the GPU kernel.
  at::Tensor cpu_value = at::empty({}, value.options().device(c10::kCPU));
  CopyTensor(value, cpu_value, /* non_blocking= */ false);

  TT_RETURN_IF_ERROR(  //
      ValidateValueCastableWithoutOverflow(cpu_value.item(),
                                           self.scalar_type()));

  return Dispatch(self, mask, value);
}

}  // namespace

at::Tensor& AtenMaskedFill_Scalar(at::Tensor& self, const at::Tensor& mask,
                                  const at::Scalar& value) {
  auto promoted_value = PromoteScalar(value);
  TT_KERNEL(OpName::kMaskedFill_Scalar, _, (self, mask, promoted_value), {
    TT_THROW_IF_ERROR(DispatchWithScalarValue(self, mask, promoted_value));
    return self;
  });
}

at::Tensor& AtenMaskedFill_Tensor(at::Tensor& self, const at::Tensor& mask,
                                  const at::Tensor& value) {
  TT_KERNEL(OpName::kMaskedFill_Tensor, _, (self, mask, value), {
    TT_THROW_IF_ERROR(DispatchWithTensorValue(self, mask, value));
    return self;
  });
}

}  // namespace torch_tpu
