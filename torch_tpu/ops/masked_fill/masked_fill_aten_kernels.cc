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

#include "torch_tpu/ops/masked_fill/masked_fill_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/where/where.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

namespace {

absl::StatusOr<DeviceBufferRef> MaskedFillScalar(const at::Tensor& input,
                                                 const at::Tensor& mask,
                                                 const at::Scalar& scalar,
                                                 OpParamCacheKeys param_keys) {
  const Dimensions output_dims = CopyIntVector(input.sizes());
  TT_ASSIGN_OR_RETURN(const auto output_mlir_dtype,
                      ConvertTo<mlir::ElementType>(input.scalar_type()));

  auto op_builder = [scalar,
                     output_mlir_dtype](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [input, mask] = inputs;
    mlir::MlirBuilder& builder = input.getBuilder();
    TT_ASSIGN_OR_RETURN(auto scalar_op, MakeConstant(builder, scalar));
    // masked_fill(input, mask, value) == where(mask, value, input)
    return BuildWhereShlo(mask, scalar_op, input, output_mlir_dtype);
  };

  const auto elem_type = output_mlir_dtype;
  return DispatchOp<2>(std::move(op_builder), {input, mask},
                       {.out_dtype = elem_type,
                        .out_dims = output_dims,
                        .op_param_cache_keys = std::move(param_keys)});
}

absl::StatusOr<DeviceBufferRef> MaskedFillTensor(const at::Tensor& input,
                                                 const at::Tensor& mask,
                                                 const at::Tensor& value) {
  for (const int64_t dim : value.sizes()) {
    TT_RET_CHECK(dim == 1, error::kInvalidArgument)
        << "only supports 1-element value tensors";
  }

  TT_ASSIGN_OR_RETURN(const auto output_mlir_type,
                      ConvertTo<mlir::ElementType>(input.scalar_type()));
  const Dimensions output_dims = CopyIntVector(input.sizes());
  TT_RET_CHECK(value.scalar_type() == input.scalar_type(),
               error::kInvalidArgument)
      << "value and input must have the same element type";

  auto op_builder = [output_mlir_type](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
    auto& [input, mask, value] = inputs;
    // masked_fill(input, mask, value) == where(mask, value, input)
    return BuildWhereShlo(mask, value, input, output_mlir_type);
  };

  const auto elem_type = output_mlir_type;
  return DispatchOp<3>(std::move(op_builder), {input, mask, value},
                       {.out_dtype = elem_type,
                        .out_dims = output_dims,
                        .op_param_cache_keys = OpParamCacheKeys::Empty()});
}

}  // namespace

at::Tensor& AtenMaskedFill_Scalar(at::Tensor& self, const at::Tensor& mask,
                                  const at::Scalar& value) {
  TT_KERNEL(OpName::kMaskedFill_Scalar, param_keys, (self, mask, value), {
    TT_ASSIGN_OR_THROW(
        DeviceBufferRef result_buf,
        MaskedFillScalar(self, mask, value, std::move(param_keys)));
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
