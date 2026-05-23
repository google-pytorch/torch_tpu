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

#include "torch_tpu/ops/fill/fill_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

at::Tensor& AtenZero_(at::Tensor& self) {
  TT_KERNEL(OpName::kZero_, _, (self),
            { return AtenFillScalar_(self, at::Scalar(0)); });
}

at::Tensor& AtenFillTensor_(at::Tensor& self, const at::Tensor& fill_value) {
  TT_KERNEL(OpName::kFill_Tensor, _, (self, fill_value), {
    auto out_aten_type = self.scalar_type();
    // Note that we fill the tensor with its own scalar type, not the
    // fill_value's type.
    TT_ASSIGN_OR_THROW(mlir::ElementType out_mlir_element_type,
                       ConvertTo<mlir::ElementType>(out_aten_type));

    // Check size.
    int64_t fill_value_dims_size = fill_value.sizes().size();
    TT_CHECK_THROW(fill_value_dims_size == 0, error::kInvalidArgument)
        << "only supports 0-dimension value tensor but got tensor with "
        << fill_value_dims_size << " dimensions.";

    auto sizes = self.sizes();

    auto op_builder =
        [sizes = CopyIntVector(sizes), out_mlir_element_type](
            mlir::MlirOp fill_value_op) -> absl::StatusOr<mlir::MlirOp> {
      TT_ASSIGN_OR_RETURN(auto result_op,
                          CastIfNeeded(fill_value_op, out_mlir_element_type));
      TT_ASSIGN_OR_RETURN(result_op, BroadcastIfNeeded(result_op, sizes));
      return result_op;
    };

    TT_ASSIGN_OR_THROW(
        auto result_buf,
        DispatchOp<1>(std::move(op_builder),
                      /*inputs=*/{fill_value},
                      {.out_dtype = out_mlir_element_type,
                       .out_dims = sizes,
                       .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), self));
    return self;
  });
}

at::Tensor& AtenFillScalar_(at::Tensor& self, const at::Scalar& fill_value) {
  TT_KERNEL(OpName::kFill_Scalar, param_keys, (self, fill_value), {
    auto out_aten_type = self.scalar_type();
    // Note that we fill the tensor with its own scalar type, not the
    // fill_value's type.
    TT_ASSIGN_OR_THROW(mlir::ElementType out_mlir_element_type,
                       ConvertTo<mlir::ElementType>(out_aten_type));

    auto sizes = self.sizes();

    auto op_builder =
        [sizes = CopyIntVector(sizes), fill_value, out_mlir_element_type](
            mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
      TT_ASSIGN_OR_RETURN(
          auto constant,
          MakeConstant(builder, fill_value, out_mlir_element_type, sizes));
      return constant;
    };

    TT_ASSIGN_OR_THROW(
        auto result_buf,
        DispatchOp<0>(std::move(op_builder),
                      /*inputs=*/{},
                      {.out_dtype = out_mlir_element_type,
                       .out_dims = sizes,
                       .op_param_cache_keys = std::move(param_keys)}));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), self));
    return self;
  });
}

}  // namespace torch_tpu
