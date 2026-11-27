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
#include "ATen/ops/result_type.h"
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
#include "stablehlo/dialect/StablehloOps.h"
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

  return absl::OkStatus();
}

absl::StatusOr<MlirOpResults<1>> BuildLerpShlo(mlir::MlirOp self,
                                               mlir::MlirOp end,
                                               mlir::MlirOp weight,
                                               mlir::ElementType common_type) {
  TT_ASSIGN_OR_RETURN(self, ComplexSafeConvertElementType(self, common_type));
  TT_ASSIGN_OR_RETURN(end, ComplexSafeConvertElementType(end, common_type));
  TT_ASSIGN_OR_RETURN(weight,
                      ComplexSafeConvertElementType(weight, common_type));

  // Apply broadcast if needed - unclear if there is a better way to unpack
  // the return array, std::tie fails in OSS CI.
  TT_ASSIGN_OR_RETURN((auto [self_bcast, end_bcast, weight_bcast]),
                      ApplyBroadcastIfNeeded(self, end, weight));
  self = self_bcast;
  end = end_bcast;
  weight = weight_bcast;

  // Like PyTorch (ATen/native/Lerp.h), interpolate from `self` for small
  // weights and from `end` otherwise. Interpolating from `self` alone loses
  // `end` at weight 1 when `end - self` rounds to `-self`.
  mlir::MlirOp diff = mlir::stablehlo::Subtract(end, self);
  mlir::MlirOp weighted_diff = mlir::stablehlo::Mul(weight, diff);
  mlir::MlirOp from_self = mlir::stablehlo::Add(self, weighted_diff);

  // The weight is small if |weight| < 0.5. PyTorch compares the squared
  // magnitude of a complex weight to avoid the square root.
  const bool is_complex = IsComplexType(GetTensorTypeOrDie(weight));
  mlir::MlirOp magnitude;
  if (is_complex) {
    mlir::MlirOp real = mlir::stablehlo::Real(weight);
    mlir::MlirOp imag = mlir::stablehlo::Imag(weight);
    mlir::MlirOp real_squared = mlir::stablehlo::Mul(real, real);
    mlir::MlirOp imag_squared = mlir::stablehlo::Mul(imag, imag);
    magnitude = mlir::stablehlo::Add(real_squared, imag_squared);
  } else {
    magnitude = mlir::stablehlo::Abs(weight);
  }
  mlir::MlirOp threshold = MakeConstantLike(magnitude, is_complex ? 0.25 : 0.5);
  mlir::MlirOp is_weight_small = mlir::stablehlo::Compare(
      magnitude, threshold, mlir::stablehlo::ComparisonDirection::LT);

  // `MakeConstantLike` cannot make a complex constant for a bounded dynamic
  // shape, so make a real one and convert it.
  mlir::MlirOp real_one = MakeConstantLike(magnitude, 1.0);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp one,
                      ComplexSafeConvertElementType(real_one, common_type));
  mlir::MlirOp one_minus_weight = mlir::stablehlo::Subtract(one, weight);
  mlir::MlirOp remaining_diff = mlir::stablehlo::Mul(diff, one_minus_weight);
  mlir::MlirOp from_end = mlir::stablehlo::Subtract(end, remaining_diff);
  return mlir::stablehlo::Select(is_weight_small, from_self, from_end);
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
    // Promote `weight` as a scalar rather than a default-float64 tensor, and
    // widen to at least F32 to match `AtenLerpTensorOut`'s compute dtype
    // without quantizing the scalar to F16/BF16.
    at::ScalarType weight_dtype =
        at::result_type(self, promoted_weight.scalar());
    weight_dtype = c10::promoteTypes(weight_dtype, at::ScalarType::Float);
    // If the promoted weight dtype is complex but the scalar argument is real,
    // demote to the matching real dtype so that ComplexSafeConvertElementType
    // in BuildLerpShlo can construct the complex operand without host
    // downcasts.
    if (IsComplex(weight_dtype) && !promoted_weight.scalar().isComplex()) {
      weight_dtype = c10::toRealValueType(weight_dtype);
    }
    TT_ASSIGN_OR_THROW(auto weight_tensor,
                       promoted_weight.GetTensor(weight_dtype));
    return AtenLerpTensorOut(self, end, weight_tensor, out);
  });
}

}  // namespace torch_tpu
