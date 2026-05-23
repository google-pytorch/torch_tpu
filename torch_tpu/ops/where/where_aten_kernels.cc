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

#include "torch_tpu/ops/where/where_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/native/Resize.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/where/where.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

// where.self
at::Tensor AtenWhereSelf(const at::Tensor& condition, const at::Tensor& self,
                         const at::Tensor& other) {
  TT_KERNEL(OpName::kWhereSelf, _, (condition, self, other), {
    // Determine output shape and type.
    at::ScalarType output_aten_type =
        at::promoteTypes(self.scalar_type(), other.scalar_type());
    TT_ASSIGN_OR_THROW(const auto output_element_type,
                       ConvertTo<mlir::ElementType>(output_aten_type));
    TT_ASSIGN_OR_THROW(auto condition_self_dims,
                       InferSize(condition.sizes(), self.sizes()));
    TT_ASSIGN_OR_THROW(auto output_dims,
                       InferSize(condition_self_dims, other.sizes()));

    // No scalars to cache

    // Create the op builder
    auto op_builder =
        [output_element_type](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
          auto& [condition, self, other] = inputs;
          return BuildWhereShlo(condition, self, other, output_element_type);
        };

    const auto elem_type = output_element_type;
    // Dispatch the op
    TT_ASSIGN_OR_THROW(
        DeviceBufferRef result_buf,
        DispatchOp<3>(std::move(op_builder), {condition, self, other},
                      {.out_dtype = elem_type,
                       .out_dims = output_dims,
                       .op_param_cache_keys = OpParamCacheKeys::Empty()}));

    // Return the result
    return MakeTensor(std::move(result_buf));
  });
}

// where.self_out
at::Tensor& AtenWhereSelfOut(const at::Tensor& condition,
                             const at::Tensor& self, const at::Tensor& other,
                             at::Tensor& out) {
  TT_KERNEL(OpName::kWhereSelfOut, _, (condition, self, other, out), {
    // Determine output shape and type.
    at::ScalarType output_aten_type =
        at::promoteTypes(self.scalar_type(), other.scalar_type());
    TT_ASSIGN_OR_THROW(const auto output_element_type,
                       ConvertTo<mlir::ElementType>(output_aten_type));
    TT_ASSIGN_OR_THROW(auto condition_self_dims,
                       InferSize(condition.sizes(), self.sizes()));
    TT_ASSIGN_OR_THROW(auto output_dims,
                       InferSize(condition_self_dims, other.sizes()));

    // Check out-tensor conditions

    TT_CHECK_THROW(output_aten_type == out.scalar_type(),
                   error::kInvalidArgument)
        << "expected the output dtype to be " << ToString(output_aten_type)
        << " (result of promoting the dtype of the input tensors -- "
        << ToString(self.scalar_type()) << " and "
        << ToString(other.scalar_type()) << "), got "
        << ToString(out.scalar_type());
    at::native::resize_output(out, output_dims);

    // No scalars to cache

    // Create the op builder.
    auto op_builder =
        [output_element_type](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
          auto& [condition, self, other] = inputs;
          return BuildWhereShlo(condition, self, other, output_element_type);
        };

    const auto elem_type = output_element_type;
    // Dispatch the op.
    TT_ASSIGN_OR_THROW(
        DeviceBufferRef result_buf,
        DispatchOp<3>(std::move(op_builder), {condition, self, other},
                      {.out_dtype = elem_type,
                       .out_dims = output_dims,
                       .op_param_cache_keys = OpParamCacheKeys::Empty()}));

    // Assign the result.
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}

}  // namespace torch_tpu
