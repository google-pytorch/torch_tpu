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

#include "torch_tpu/ops/lerp/lerp_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
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

namespace torch_tpu {

namespace {

absl::Status CheckLerpSelfInput(const at::Tensor& self) {
  TT_RET_CHECK(!IsIntegral(self), error::kInvalidArgument)
      << "expected the first argument's dtype to be non-integral, got "
      << ToString(self.scalar_type());

  TT_RET_CHECK(self.scalar_type() != at::ScalarType::ComplexDouble,
               error::kUnimplemented)
      << "complex128 dtype is not yet supported";

  return absl::OkStatus();
}

absl::StatusOr<MlirOpResults<1>> BuildLerpShlo(mlir::MlirOp self,
                                               mlir::MlirOp end,
                                               mlir::MlirOp weight,
                                               mlir::ElementType common_type) {
  // Apply broadcast if needed - unclear if there is a better way to unpack
  // the return array, std::tie fails in OSS CI.
  TT_ASSIGN_OR_RETURN((auto [self_bcast, end_bcast, weight_bcast]),
                      ApplyBroadcastIfNeeded(self, end, weight));
  self = self_bcast;
  end = end_bcast;
  weight = weight_bcast;
  self = mlir::stablehlo::ConvertElementType(self, common_type);
  end = mlir::stablehlo::ConvertElementType(end, common_type);
  weight = mlir::stablehlo::ConvertElementType(weight, common_type);

  // Compute result = self + weight * (end - self)
  auto diff_op = mlir::stablehlo::Subtract(end, self);
  auto mul_op = mlir::stablehlo::Mul(weight, diff_op);
  return mlir::stablehlo::Add(self, mul_op);
}

NAryMlirOpBuilder<3> GetLerpTensorOutFunctional(mlir::ElementType common_type) {
  return [common_type](FixedSizeSpan<mlir::MlirOp, 3> inputs)
             -> absl::StatusOr<MlirOpResults<1>> {
    auto [self, end, weight] = inputs;
    return BuildLerpShlo(self, end, weight, common_type);
  };
}

}  // namespace

at::Tensor& AtenLerpTensorOut(const at::Tensor& self, const at::Tensor& end,
                              const at::Tensor& weight, at::Tensor& out) {
  TT_KERNEL(OpName::kLerpTensorOut, param_keys, (self, end, weight, out), {
    TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));
    auto promoted_dtype =
        c10::promoteTypes(self.scalar_type(), end.scalar_type());
    promoted_dtype = c10::promoteTypes(promoted_dtype, weight.scalar_type());

    // Use at least F32 for the inputs following the torch impl.
    promoted_dtype = c10::promoteTypes(promoted_dtype, at::ScalarType::Float);

    TT_ASSIGN_OR_THROW(mlir::ElementType common_type,
                       ConvertTo<mlir::ElementType>(promoted_dtype));

    TT_THROW_IF_ERROR(CheckLerpSelfInput(self));

    TT_ASSIGN_OR_THROW(
        auto result,
        DispatchOp<3>(GetLerpTensorOutFunctional(common_type),
                      /*inputs=*/{self, end, weight},
                      /*options=*/
                      {.out_dtype = output_dtype,
                       .out_dims = out.sizes(),
                       .op_param_cache_keys = std::move(param_keys)}));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(result, out));
    return out;
  });
}

at::Tensor& AtenLerpScalarOut(const at::Tensor& self, const at::Tensor& end,
                              const at::Scalar& weight, at::Tensor& out) {
  auto promoted_weight = PromoteScalar(weight);
  TT_KERNEL(OpName::kLerpScalarOut, _, (self, end, promoted_weight, out), {
    TT_ASSIGN_OR_THROW(auto weight_tensor, promoted_weight.GetTensor());
    return AtenLerpTensorOut(self, end, weight_tensor, out);
  });
}

}  // namespace torch_tpu
