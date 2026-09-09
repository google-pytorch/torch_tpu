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

#include "csrc/ops/where/where_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "csrc/ops/where/where.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

namespace {

template <typename ReturnType, typename DispatchFn>
ReturnType DispatchWhere(const at::Tensor& condition, const at::Tensor& self,
                         const at::Tensor& other, DispatchFn&& dispatch_fn) {
  const at::ScalarType output_aten_type =
      at::promoteTypes(self.scalar_type(), other.scalar_type());
  TT_ASSIGN_OR_RETURN(const auto output_element_type,
                      ConvertTo<mlir::ElementType>(output_aten_type));
  TT_ASSIGN_OR_RETURN(auto condition_self_dims,
                      InferSize(condition.sizes(), self.sizes()));
  TT_ASSIGN_OR_RETURN(auto output_dims,
                      InferSize(condition_self_dims, other.sizes()));

  auto op_builder = [output_element_type](
                        FixedSizeSpan<mlir::MlirOp, 3> inputs) {
    auto& [condition_op, self_op, other_op] = inputs;
    return BuildWhereShlo(condition_op, self_op, other_op, output_element_type);
  };

  return dispatch_fn(output_aten_type, output_element_type,
                     std::move(output_dims), std::move(op_builder));
}

}  // namespace

// where.self
at::Tensor AtenWhereSelf(const at::Tensor& condition, const at::Tensor& self,
                         const at::Tensor& other) {
  TT_KERNEL(OpName::kWhereSelf, _, (condition, self, other), {
    TT_ASSIGN_OR_THROW(
        DeviceBufferRef result_buf,
        (DispatchWhere<absl::StatusOr<DeviceBufferRef>>(
            condition, self, other,
            [&](at::ScalarType /*output_aten_type*/,
                mlir::ElementType output_element_type,
                const Dimensions& output_dims, auto op_builder) {
              return DispatchOp<3>(
                  std::move(op_builder), {condition, self, other},
                  {.out_dtype = output_element_type,
                   .out_dims = output_dims,
                   .op_param_cache_keys = OpParamCacheKeys::Empty()});
            })));
    return MakeTensor(std::move(result_buf));
  });
}

// where.self_out
at::Tensor& AtenWhereSelfOut(const at::Tensor& condition,
                             const at::Tensor& self, const at::Tensor& other,
                             at::Tensor& out) {
  TT_KERNEL(OpName::kWhereSelfOut, _, (condition, self, other, out), {
    TT_THROW_IF_ERROR((DispatchWhere<absl::Status>(
        condition, self, other,
        [&](at::ScalarType output_aten_type,
            mlir::ElementType output_element_type,
            const Dimensions& output_dims, auto op_builder) -> absl::Status {
          TT_RET_CHECK(output_aten_type == out.scalar_type(),
                       error::kInvalidArgument)
              << "expected the output dtype to be "
              << ToString(output_aten_type)
              << " (result of promoting the dtype of the input tensors -- "
              << ToString(self.scalar_type()) << " and "
              << ToString(other.scalar_type()) << "), got "
              << ToString(out.scalar_type());
          TT_RETURN_IF_ERROR(ResizeTensorIfShapeDiffers(out, output_dims));
          return DispatchOpOut<3>(
              std::move(op_builder), {condition, self, other}, out,
              {.out_dtype = output_element_type,
               .out_dims = output_dims,
               .op_param_cache_keys = OpParamCacheKeys::Empty()});
        })));
    return out;
  });
}

}  // namespace torch_tpu
