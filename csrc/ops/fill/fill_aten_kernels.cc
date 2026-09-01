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

#include "csrc/ops/fill/fill_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

absl::Status FillTensorHelper(at::Tensor& self, const at::Tensor& fill_value,
                              OpParamCacheKeys param_keys) {
  auto out_aten_type = self.scalar_type();
  TT_RET_CHECK(out_aten_type != at::ScalarType::ComplexDouble,
               error::kPythonNotImplementedError)
      << ToString(at::ScalarType::ComplexDouble) << " dtype is not supported";
  TT_ASSIGN_OR_RETURN(mlir::ElementType out_mlir_element_type,
                      ConvertTo<mlir::ElementType>(out_aten_type));

  int64_t fill_value_dims_size = fill_value.sizes().size();
  TT_RET_CHECK(fill_value_dims_size == 0, error::kInvalidArgument)
      << "expected value to be a 0-D tensor, got " << fill_value_dims_size
      << "-D";

  auto target_sizes = self.sizes();

  auto op_builder =
      [sizes = CopyIntVector(target_sizes), out_mlir_element_type](
          mlir::MlirOp fill_value_op) -> absl::StatusOr<mlir::MlirOp> {
    TT_ASSIGN_OR_RETURN(auto result_op,
                        CastIfNeeded(fill_value_op, out_mlir_element_type));
    TT_ASSIGN_OR_RETURN(result_op, BroadcastIfNeeded(result_op, sizes));
    return result_op;
  };

  TT_ASSIGN_OR_RETURN(
      auto result_buf,
      DispatchOp<1>(std::move(op_builder),
                    /*inputs=*/{fill_value},
                    {.out_dtype = out_mlir_element_type,
                     .out_dims = target_sizes,
                     .op_param_cache_keys = std::move(param_keys)}));
  return AssignBufferToAtTensor(std::move(result_buf), self);
}

}  // namespace

at::Tensor& AtenZero_(at::Tensor& self) {
  TT_KERNEL(OpName::kZero_, _, (self), {
    auto op_builder = [](mlir::MlirOp self_op) -> absl::StatusOr<mlir::MlirOp> {
      return MakeConstantLike(self_op, 0.0);
    };
    TT_THROW_IF_ERROR(
        UnaryOpInPlace(self, std::move(op_builder),
                       {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return self;
  });
}

at::Tensor& AtenFillTensor_(at::Tensor& self, const at::Tensor& fill_value) {
  TT_KERNEL(OpName::kFill_Tensor, _, (self, fill_value), {
    TT_THROW_IF_ERROR(
        FillTensorHelper(self, fill_value, OpParamCacheKeys::Empty()));
    return self;
  });
}

at::Tensor& AtenFillScalar_(at::Tensor& self, const at::Scalar& fill_value) {
  auto promoted_fill_value = PromoteScalar(fill_value);
  TT_KERNEL(OpName::kFill_Scalar, param_keys, (self, promoted_fill_value), {
    auto out_aten_type = self.scalar_type();
    TT_ASSIGN_OR_THROW(at::Tensor fill_value_tensor,
                       promoted_fill_value.GetTensor(out_aten_type));
    TT_THROW_IF_ERROR(
        FillTensorHelper(self, fill_value_tensor, std::move(param_keys)));
    return self;
  });
}

}  // namespace torch_tpu
