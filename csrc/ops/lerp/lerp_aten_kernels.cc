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
#include "ATen/native/TypeProperties.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

enum class OutDtypePolicy {
  kExactMatch,
  kSafeCast,
};

absl::Status ValidateLerpCommonInputs(const at::Tensor& self,
                                      const at::Tensor& end,
                                      const at::Tensor& out) {
  TT_RET_CHECK(IsPrivateUse1Device(out), error::kInvalidArgument)
      << "expected out tensor to be on " << GetPrivateUse1DeviceDebugName()
      << ", got " << out.device();
  TT_RET_CHECK(self.scalar_type() == end.scalar_type(), error::kInvalidArgument)
      << "expected end tensor to have dtype " << ToString(self.scalar_type())
      << ", got " << ToString(end.scalar_type());
  return absl::OkStatus();
}

absl::Status ValidateLerpResultAndOutDtype(
    const at::ScalarType result_dtype, const at::Tensor& out,
    const OutDtypePolicy out_dtype_policy) {
  TT_RET_CHECK(!IsIntegral(result_dtype), error::kInvalidArgument)
      << "expected the first argument's dtype to be non-integral, got "
      << ToString(result_dtype);
  TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(result_dtype));
  TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(out));
  return ValidateOutDtype(
      out, result_dtype,
      /*allow_out_dtype_cast=*/out_dtype_policy == OutDtypePolicy::kSafeCast);
}

absl::StatusOr<MlirOpResults<1>> BuildLerpShlo(
    mlir::MlirOp self, mlir::MlirOp end, mlir::MlirOp weight,
    const mlir::ElementType common_type, const mlir::ElementType output_dtype) {
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

  // Compute result = self + weight * (end - self)
  mlir::MlirOp diff_op = mlir::stablehlo::Subtract(end, self);
  mlir::MlirOp mul_op = mlir::stablehlo::Mul(weight, diff_op);
  mlir::MlirOp result = mlir::stablehlo::Add(self, mul_op);
  return ComplexSafeConvertElementType(result, output_dtype);
}

[[nodiscard]] NAryMlirOpBuilder<3> GetLerpTensorOutFunctional(
    const mlir::ElementType common_type, const mlir::ElementType output_dtype) {
  return [common_type, output_dtype](FixedSizeSpan<mlir::MlirOp, 3> inputs)
             -> absl::StatusOr<MlirOpResults<1>> {
    auto [self, end, weight] = inputs;
    return BuildLerpShlo(self, end, weight, common_type, output_dtype);
  };
}

absl::Status DispatchLerpOut(const at::Tensor& self, const at::Tensor& end,
                             const at::Tensor& weight, at::Tensor& out,
                             const at::ScalarType compute_dtype,
                             OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const Dimensions expected_size,
                      InferSize(self.sizes(), end.sizes(), weight.sizes()));
  if (out.is_alias_of(self) || out.is_alias_of(end) ||
      out.is_alias_of(weight)) {
    TT_RET_CHECK(absl::MakeConstSpan(out.sizes()) == expected_size,
                 error::kInvalidArgument)
        << "output with shape " << ToString(out.sizes())
        << " doesn't match the broadcast shape of the tensor being operated on "
           "in-place, which has shape "
        << ToString(expected_size);
  }

  TT_ASSIGN_OR_RETURN(const mlir::ElementType output_dtype,
                      ConvertTo<mlir::ElementType>(out.scalar_type()));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType common_type,
                      ConvertTo<mlir::ElementType>(compute_dtype));

  return DispatchOpOut<3>(GetLerpTensorOutFunctional(common_type, output_dtype),
                          /*inputs=*/{self, end, weight}, out,
                          /*options=*/
                          {.out_dtype = output_dtype,
                           .out_dims = expected_size,
                           .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor& AtenLerpTensorOut(const at::Tensor& self, const at::Tensor& end,
                              const at::Tensor& weight, at::Tensor& out) {
  TT_KERNEL(OpName::kLerpTensorOut, param_keys, (self, end, weight, out), {
    TT_THROW_IF_ERROR(ValidateLerpCommonInputs(self, end, out));

    // Match PyTorch's `lerp_Tensor`: non-0D `weight` requires exact dtype
    // matching on both `weight` and `out`, while 0D `weight` promotes inputs to
    // a common dtype and allows safe casts to `out`.
    const bool promote_weight = weight.dim() == 0;
    if (!promote_weight) {
      TT_CHECK_THROW(self.scalar_type() == weight.scalar_type(),
                     error::kInvalidArgument)
          << "expected weight tensor to have dtype "
          << ToString(self.scalar_type()) << ", got "
          << ToString(weight.scalar_type());
    }

    const at::ScalarType result_dtype =
        promote_weight
            ? at::native::result_type(at::ITensorListRef({self, end, weight}))
            : self.scalar_type();
    TT_THROW_IF_ERROR(ValidateLerpResultAndOutDtype(
        result_dtype, out,
        promote_weight ? OutDtypePolicy::kSafeCast
                       : OutDtypePolicy::kExactMatch));

    const at::ScalarType compute_dtype = InferComputationDtype(result_dtype);
    TT_THROW_IF_ERROR(DispatchLerpOut(self, end, weight, out, compute_dtype,
                                      std::move(param_keys)));
    return out;
  });
}

at::Tensor& AtenLerpScalarOut(const at::Tensor& self, const at::Tensor& end,
                              const at::Scalar& weight, at::Tensor& out) {
  auto promoted_weight = PromoteScalar(weight);
  TT_KERNEL(
      OpName::kLerpScalarOut, param_keys, (self, end, promoted_weight, out), {
        TT_THROW_IF_ERROR(ValidateLerpCommonInputs(self, end, out));

        const at::ScalarType result_dtype = self.scalar_type();
        TT_THROW_IF_ERROR(ValidateLerpResultAndOutDtype(
            result_dtype, out, OutDtypePolicy::kSafeCast));

        const at::ScalarType compute_dtype =
            InferComputationDtype(result_dtype);
        TT_ASSIGN_OR_THROW(const bool can_cast_weight,
                           CanCastScalarWithoutOverflow(
                               promoted_weight.scalar(), compute_dtype));
        TT_CHECK_THROW(can_cast_weight, error::kInvalidArgument)
            << "value cannot be converted to type " << ToString(compute_dtype)
            << " without overflow";

        // Demote real scalars for complex inputs to the matching real dtype so
        // `MakeBuffer` preserves float64 precision for complex128 instead of
        // downcasting ComplexDouble -> ComplexFloat; BuildLerpShlo widens to
        // `compute_dtype` via ComplexSafeConvertElementType.
        const at::ScalarType weight_dtype =
            (IsComplex(compute_dtype) && !promoted_weight.scalar().isComplex())
                ? c10::toRealValueType(compute_dtype)
                : compute_dtype;
        TT_ASSIGN_OR_THROW(const at::Tensor weight_tensor,
                           promoted_weight.GetTensor(weight_dtype));
        TT_THROW_IF_ERROR(DispatchLerpOut(self, end, weight_tensor, out,
                                          compute_dtype,
                                          std::move(param_keys)));
        return out;
      });
}

}  // namespace torch_tpu
