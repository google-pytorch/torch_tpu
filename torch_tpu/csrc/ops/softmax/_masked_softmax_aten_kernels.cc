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

#include "torch_tpu/csrc/ops/softmax/_masked_softmax_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/fixed_size_span.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/csrc/ops/precision_context.h"
#include "torch_tpu/csrc/ops/resize/resize_aten_kernels.h"
#include "torch_tpu/csrc/ops/softmax/softmax.h"

namespace torch_tpu {
namespace {

absl::StatusOr<int64_t> ValidateMaskedSoftmaxInputs(
    const at::Tensor& self, const at::Tensor& mask,
    std::optional<int64_t> mask_type) {
  TT_RET_CHECK(self.is_floating_point(), error::kInvalidArgument)
      << "expected input to be a floating point tensor, got "
      << ToString(self.scalar_type());

  TT_RET_CHECK(mask.scalar_type() == c10::ScalarType::Bool,
               error::kInvalidArgument)
      << "expected mask to be a boolean tensor, got "
      << ToString(mask.scalar_type());

  // Representing the mask type:
  // 0 = Attention mask (src_mask) -> broadcasted across block dimensions (dim 1
  // and/or 2) 1 = Padding mask (src_key_padding_mask) -> broadcasted across
  // query/key block dimensions (dim 1 and 2) 2 = Default mask (generic mask) ->
  // matches full input shape
  int64_t resolved_mask_type = 2;
  if (mask_type.has_value()) {
    const int64_t requested_mask_type = mask_type.value();
    TT_RET_CHECK(requested_mask_type == 0 || requested_mask_type == 1 ||
                     requested_mask_type == 2,
                 error::kInvalidArgument)
        << "expected mask_type to be 0, 1, or 2, got " << requested_mask_type;
    if (self.dim() != 4 || mask.dim() != 2) {
      resolved_mask_type = 2;
    } else {
      resolved_mask_type = requested_mask_type;
    }
  } else {
    if (self.dim() != 4 || mask.dim() != 2) {
      resolved_mask_type = 2;
    } else if (self.size(0) == mask.size(0) && self.size(3) == mask.size(1)) {
      resolved_mask_type = 1;
    } else if (self.size(2) == mask.size(0) && self.size(3) == mask.size(1)) {
      resolved_mask_type = 0;
    } else {
      resolved_mask_type = 2;
    }
  }

  switch (resolved_mask_type) {
    case 0:
      TT_RET_CHECK(self.dim() == 4 && mask.dim() == 2 &&
                       self.size(2) == mask.size(0) &&
                       self.size(3) == mask.size(1),
                   error::kInvalidArgument)
          << "expected mask shape to be (" << self.size(2) << ", "
          << self.size(3) << ") for mask_type 0, got "
          << ToString(mask.sizes());
      break;
    case 1:
      TT_RET_CHECK(self.dim() == 4 && mask.dim() == 2 &&
                       self.size(0) == mask.size(0) &&
                       self.size(3) == mask.size(1),
                   error::kInvalidArgument)
          << "expected mask shape to be (" << self.size(0) << ", "
          << self.size(3) << ") for mask_type 1, got "
          << ToString(mask.sizes());
      break;
    case 2:
    default:
      TT_RET_CHECK(mask.sizes() == self.sizes(), error::kInvalidArgument)
          << "expected mask shape to be " << ToString(self.sizes()) << ", got "
          << ToString(mask.sizes());
      break;
  }
  return resolved_mask_type;
}

absl::Status ValidateMaskedSoftmaxBackwardInputs(const at::Tensor& grad_output,
                                                 const at::Tensor& output,
                                                 const at::Tensor& mask) {
  TT_RET_CHECK(grad_output.is_floating_point(), error::kInvalidArgument)
      << "expected grad_output to be a floating point tensor, got "
      << ToString(grad_output.scalar_type());

  TT_RET_CHECK(output.is_floating_point(), error::kInvalidArgument)
      << "expected output to be a floating point tensor, got "
      << ToString(output.scalar_type());

  TT_RET_CHECK(mask.scalar_type() == c10::ScalarType::Bool,
               error::kInvalidArgument)
      << "expected mask to be a boolean tensor, got "
      << ToString(mask.scalar_type());

  TT_RET_CHECK(grad_output.sizes() == output.sizes(), error::kInvalidArgument)
      << "expected grad_output shape to be " << ToString(output.sizes())
      << ", got " << ToString(grad_output.sizes());

  const bool same_shape = (mask.sizes() == grad_output.sizes());
  const bool is_2d_broadcastable = (grad_output.dim() == 4 && mask.dim() == 2 &&
                                    ((mask.size(0) == grad_output.size(0) &&
                                      mask.size(1) == grad_output.size(3)) ||
                                     (mask.size(0) == grad_output.size(2) &&
                                      mask.size(1) == grad_output.size(3))));

  TT_RET_CHECK(same_shape || is_2d_broadcastable, error::kInvalidArgument)
      << "expected mask shape to be " << ToString(grad_output.sizes())
      << ", got " << ToString(mask.sizes());

  return absl::OkStatus();
}

absl::Status MaskedSoftmaxInternalOut(const at::Tensor& self,
                                      const at::Tensor& mask,
                                      std::optional<int64_t> dim,
                                      std::optional<int64_t> mask_type,
                                      at::Tensor& out,
                                      OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const int64_t resolved_mask_type,
                      ValidateMaskedSoftmaxInputs(self, mask, mask_type));

  TT_RETURN_IF_ERROR(ResizeTensorIfShapeDiffers(out, self.sizes()));

  if (self.numel() == 0) {
    return absl::OkStatus();
  }

  const int64_t default_dim = self.dim() > 0 ? self.dim() - 1 : 0;
  const int64_t dim_val = dim.value_or(default_dim);
  const int64_t input_rank = self.dim() == 0 ? 1 : self.dim();
  TT_ASSIGN_OR_RETURN(const int64_t wrapped_dim,
                      SafeWrapDim(dim_val, input_rank));

  TT_ASSIGN_OR_RETURN(const mlir::ElementType computation_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));

  auto functional_builder =
      [wrapped_dim, resolved_mask_type](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<MlirOpResults<1>> {
    auto& [input_op, mask_op] = inputs;
    return BuildMaskedSoftmaxShlo(input_op, mask_op, wrapped_dim,
                                  resolved_mask_type);
  };

  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef result,
      (DispatchOp<2>(functional_builder, {self, mask},
                     {.out_dtype = computation_dtype,
                      .out_dims = self.sizes(),
                      .op_param_cache_keys = std::move(param_keys)})));

  return AssignBufferToAtTensor(std::move(result), out);
}

absl::Status MaskedSoftmaxBackwardInternalOut(const at::Tensor& grad_output,
                                              const at::Tensor& output,
                                              const at::Tensor& mask,
                                              std::optional<int64_t> dim,
                                              at::Tensor& grad_input,
                                              OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(
      ValidateMaskedSoftmaxBackwardInputs(grad_output, output, mask));

  TT_RETURN_IF_ERROR(ResizeTensorIfShapeDiffers(grad_input, output.sizes()));

  if (grad_output.numel() == 0) {
    return absl::OkStatus();
  }

  const int64_t default_dim = output.dim() > 0 ? output.dim() - 1 : 0;
  const int64_t dim_val = dim.value_or(default_dim);
  const int64_t input_rank = output.dim() == 0 ? 1 : output.dim();
  TT_ASSIGN_OR_RETURN(const int64_t wrapped_dim,
                      SafeWrapDim(dim_val, input_rank));

  const auto precision = GetAndAddPrecisionTo(param_keys);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType computation_dtype,
                      ConvertTo<mlir::ElementType>(grad_output.scalar_type()));

  auto functional_builder = [wrapped_dim,
                             precision](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<MlirOpResults<1>> {
    auto& [grad_output_op, output_op, mask_op] = inputs;
    return BuildMaskedSoftmaxBackwardDataShlo(grad_output_op, output_op,
                                              mask_op, wrapped_dim, precision);
  };

  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef result,
      (DispatchOp<3>(functional_builder, {grad_output, output, mask},
                     {.out_dtype = computation_dtype,
                      .out_dims = output.sizes(),
                      .op_param_cache_keys = std::move(param_keys)})));

  return AssignBufferToAtTensor(std::move(result), grad_input);
}

}  // namespace

at::Tensor AtenMaskedSoftmax(const at::Tensor& self, const at::Tensor& mask,
                             std::optional<int64_t> dim,
                             std::optional<int64_t> mask_type) {
  TT_KERNEL(OpName::kMaskedSoftmax, param_keys, (self, mask, dim, mask_type), {
    TT_ASSIGN_OR_THROW(
        at::Tensor out,
        MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
    TT_THROW_IF_ERROR(MaskedSoftmaxInternalOut(self, mask, dim, mask_type, out,
                                               std::move(param_keys)));
    return out;
  });
}

at::Tensor& AtenMaskedSoftmaxOut(const at::Tensor& self, const at::Tensor& mask,
                                 std::optional<int64_t> dim,
                                 std::optional<int64_t> mask_type,
                                 at::Tensor& out) {
  TT_KERNEL(OpName::kMaskedSoftmaxOut, param_keys,
            (self, mask, dim, mask_type, out), {
              TT_THROW_IF_ERROR(MaskedSoftmaxInternalOut(
                  self, mask, dim, mask_type, out, std::move(param_keys)));
              return out;
            });
}

at::Tensor AtenMaskedSoftmaxBackward(const at::Tensor& grad_output,
                                     const at::Tensor& output,
                                     const at::Tensor& mask,
                                     std::optional<int64_t> dim) {
  TT_KERNEL(
      OpName::kMaskedSoftmaxBackward, param_keys,
      (grad_output, output, mask, dim), {
        TT_ASSIGN_OR_THROW(at::Tensor grad_input,
                           MakeEmptyTensor(output.sizes(), output.scalar_type(),
                                           output.device()));
        TT_THROW_IF_ERROR(MaskedSoftmaxBackwardInternalOut(
            grad_output, output, mask, dim, grad_input, std::move(param_keys)));
        return grad_input;
      });
}

at::Tensor& AtenMaskedSoftmaxBackwardOut(const at::Tensor& grad_output,
                                         const at::Tensor& output,
                                         const at::Tensor& mask,
                                         std::optional<int64_t> dim,
                                         at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kMaskedSoftmaxBackwardOut, param_keys,
      (grad_output, output, mask, dim, grad_input), {
        TT_THROW_IF_ERROR(MaskedSoftmaxBackwardInternalOut(
            grad_output, output, mask, dim, grad_input, std::move(param_keys)));
        return grad_input;
      });
}

}  // namespace torch_tpu
