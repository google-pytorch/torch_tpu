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

#include "torch_tpu/csrc/ops/masked_fill/masked_fill_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dimension_types.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/fixed_size_span.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/common/utils.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/csrc/ops/where/where.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

namespace {

absl::StatusOr<DeviceBufferRef> MaskedFillTensor(const at::Tensor& self,
                                                 const at::Tensor& mask,
                                                 const at::Tensor& value) {
  for (const int64_t dim : value.sizes()) {
    TT_RET_CHECK(dim == 1, error::kInvalidArgument)
        << "only supports 1-element value tensors";
  }

  TT_ASSIGN_OR_RETURN(const auto output_mlir_type,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_RET_CHECK(value.scalar_type() == self.scalar_type(),
               error::kInvalidArgument)
      << "expected self and value to have the same element type, got "
      << ToString(value.scalar_type()) << " and "
      << ToString(self.scalar_type());

  const Dimensions self_dims = CopyIntVector(self.sizes());

  TT_ASSIGN_OR_RETURN(const Dimensions out_dims,
                      InferSize(self.sizes(), mask.sizes()));

  TT_RET_CHECK(out_dims == self_dims, error::kInvalidArgument)
      << "expected the output shape (broadcast of self and mask) to match "
      << ToString(self_dims) << " (shape of self), got " << ToString(out_dims);

  auto op_builder = [output_mlir_type](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
    auto& [input, mask, value] = inputs;
    // masked_fill(input, mask, value) == where(mask, value, input)
    return BuildWhereShlo(mask, value, input, output_mlir_type);
  };

  const auto elem_type = output_mlir_type;
  return DispatchOp<3>(std::move(op_builder), {self, mask, value},
                       {.out_dtype = elem_type,
                        .out_dims = out_dims,
                        .op_param_cache_keys = OpParamCacheKeys::Empty()});
}

}  // namespace

at::Tensor& AtenMaskedFill_Scalar(at::Tensor& self, const at::Tensor& mask,
                                  const at::Scalar& value) {
  auto promoted_value = PromoteScalar(value);
  TT_KERNEL(OpName::kMaskedFill_Scalar, _, (self, mask, promoted_value), {
    TT_ASSIGN_OR_THROW(const at::Tensor value_tensor,
                       promoted_value.GetTensor(self.scalar_type()));
    TT_ASSIGN_OR_THROW(DeviceBufferRef result_buf,
                       MaskedFillTensor(self, mask, value_tensor));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), self));

    return self;
  });
}

at::Tensor& AtenMaskedFill_Tensor(at::Tensor& self, const at::Tensor& mask,
                                  const at::Tensor& value) {
  TT_KERNEL(OpName::kMaskedFill_Tensor, _, (self, mask, value), {
    TT_ASSIGN_OR_THROW(DeviceBufferRef result_buf,
                       MaskedFillTensor(self, mask, value));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), self));
    return self;
  });
}

}  // namespace torch_tpu
