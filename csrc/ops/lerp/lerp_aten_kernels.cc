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

#include "csrc/ops/lerp/lerp_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

absl::Status ValidateLerpSelfInput(const at::Tensor& self) {
  TT_RET_CHECK(!IsIntegral(self), error::kInvalidArgument)
      << "expected the first argument's dtype to be non-integral, got "
      << ToString(self.scalar_type());

  TT_RET_CHECK(self.scalar_type() != at::ScalarType::ComplexDouble,
               error::kPythonNotImplementedError)
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
    TT_ASSIGN_OR_THROW(auto end_weight_size,
                       InferSize(end.sizes(), weight.sizes()));
    TT_ASSIGN_OR_THROW(auto expected_size,
                       InferSize(self.sizes(), end_weight_size));
    TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, expected_size));

    TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));
    auto promoted_dtype =
        c10::promoteTypes(self.scalar_type(), end.scalar_type());
    promoted_dtype = c10::promoteTypes(promoted_dtype, weight.scalar_type());

    // Use at least F32 for the inputs following the torch impl.
    promoted_dtype = c10::promoteTypes(promoted_dtype, at::ScalarType::Float);

    TT_ASSIGN_OR_THROW(mlir::ElementType common_type,
                       ConvertTo<mlir::ElementType>(promoted_dtype));

    TT_THROW_IF_ERROR(ValidateLerpSelfInput(self));

    TT_THROW_IF_ERROR(
        DispatchOpOut<3>(GetLerpTensorOutFunctional(common_type),
                         /*inputs=*/{self, end, weight}, out,
                         /*options=*/
                         {.out_dtype = output_dtype,
                          .out_dims = out.sizes(),
                          .op_param_cache_keys = std::move(param_keys)}));
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
