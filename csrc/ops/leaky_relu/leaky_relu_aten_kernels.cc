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

#include "csrc/ops/leaky_relu/leaky_relu_aten_kernels.h"

#include <array>
#include <optional>
#include <string_view>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/result_type.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

absl::Status ValidateOnPrivateUse1Device(const at::Tensor& tensor,
                                         std::string_view name) {
  TT_RET_CHECK(IsPrivateUse1Device(tensor), error::kInvalidArgument)
      << "expected " << name << " tensor to be on "
      << GetPrivateUse1DeviceDebugName() << ", got " << tensor.device();
  return absl::OkStatus();
}

absl::StatusOr<mlir::MlirOp> BuildLeakyReluShlo(
    mlir::MlirOp self_op, mlir::MlirOp negative_slope_op,
    mlir::ElementType computation_dtype) {
  auto& builder = self_op.getBuilder();
  mlir::MlirOp zero_op = MakeScalarConstant(builder, 0.0, computation_dtype);
  std::array<mlir::MlirOp, 2> broadcasted;
  TT_ASSIGN_OR_RETURN(broadcasted, ApplyBroadcastIfNeeded(self_op, zero_op));
  // As in PyTorch, 0 and -0.0 take the negative-slope branch.
  mlir::MlirOp compare_gt_zero = mlir::stablehlo::Compare(
      broadcasted[0], broadcasted[1], mlir::stablehlo::ComparisonDirection::GT);

  TT_ASSIGN_OR_RETURN(broadcasted,
                      ApplyBroadcastIfNeeded(self_op, negative_slope_op));
  mlir::MlirOp mul_op = mlir::stablehlo::Mul(broadcasted[0], broadcasted[1]);

  std::array<mlir::MlirOp, 3> select_broadcasted;
  TT_ASSIGN_OR_RETURN(select_broadcasted,
                      ApplyBroadcastIfNeeded(compare_gt_zero, self_op, mul_op));
  return mlir::stablehlo::Select(select_broadcasted[0], select_broadcasted[1],
                                 select_broadcasted[2]);
}

absl::StatusOr<mlir::MlirOp> BuildLeakyReluBackwardShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp self_or_result_op,
    mlir::MlirOp negative_slope_op, mlir::ElementType computation_dtype) {
  auto& builder = grad_output_op.getBuilder();
  mlir::MlirOp zero_op = MakeScalarConstant(builder, 0.0, computation_dtype);
  std::array<mlir::MlirOp, 2> broadcasted;
  TT_ASSIGN_OR_RETURN(broadcasted,
                      ApplyBroadcastIfNeeded(self_or_result_op, zero_op));
  // As in PyTorch, 0 and -0.0 take the negative-slope branch.
  mlir::MlirOp compare_gt_zero = mlir::stablehlo::Compare(
      broadcasted[0], broadcasted[1], mlir::stablehlo::ComparisonDirection::GT);

  TT_ASSIGN_OR_RETURN(
      broadcasted, ApplyBroadcastIfNeeded(grad_output_op, negative_slope_op));
  mlir::MlirOp mul_op = mlir::stablehlo::Mul(broadcasted[0], broadcasted[1]);

  std::array<mlir::MlirOp, 3> select_broadcasted;
  TT_ASSIGN_OR_RETURN(
      select_broadcasted,
      ApplyBroadcastIfNeeded(compare_gt_zero, grad_output_op, mul_op));
  return mlir::stablehlo::Select(select_broadcasted[0], select_broadcasted[1],
                                 select_broadcasted[2]);
}

absl::Status LeakyReluScalarHelper(const at::Tensor& self,
                                   PromotedScalar negative_slope,
                                   at::Tensor& out,
                                   OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateOnPrivateUse1Device(self, "self"));
  TT_RETURN_IF_ERROR(ValidateOnPrivateUse1Device(out, "out"));
  // `out` must match `self`'s dtype, so the checks on `self` below also cover
  // `out`.
  TT_RETURN_IF_ERROR(ValidateOutDtype(out, self.scalar_type(),
                                      /*allow_out_dtype_cast=*/false));
  TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(self));

  TT_ASSIGN_OR_RETURN(const auto dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_RET_CHECK(!IsBoolean(dtype), error::kInvalidArgument)
      << "expected non-bool dtype, got " << ToString(self.scalar_type());
  TT_RET_CHECK(!IsInteger(dtype, /*includeBool=*/false),
               error::kInvalidArgument)
      << "expected non-integer dtype, got " << ToString(self.scalar_type());
  TT_RET_CHECK(!IsComplex(dtype), error::kInvalidArgument)
      << "expected non-complex dtype, got " << ToString(self.scalar_type());

  const at::ScalarType comp_scalar_type =
      InferComputationDtype(self.scalar_type());
  TT_ASSIGN_OR_RETURN(at::Tensor negative_slope_tensor,
                      negative_slope.GetTensor(comp_scalar_type));

  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(out.scalar_type()));
  TT_ASSIGN_OR_RETURN(const auto computation_dtype,
                      ConvertTo<mlir::ElementType>(comp_scalar_type));

  auto op_builder = [computation_dtype](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    return BuildLeakyReluShlo(inputs[0], inputs[1], computation_dtype);
  };

  return DispatchOpOut<2>(std::move(op_builder), {self, negative_slope_tensor},
                          out,
                          {.out_dtype = out_dtype,
                           .out_dims = CopyIntVector(self.sizes()),
                           .computation_dtype = computation_dtype,
                           .op_param_cache_keys = std::move(param_keys)});
}

absl::Status LeakyReluBackwardScalarHelper(const at::Tensor& grad_output,
                                           const at::Tensor& self_or_result,
                                           PromotedScalar negative_slope,
                                           bool self_is_result,
                                           OpParamCacheKeys param_keys,
                                           at::Tensor& grad_input) {
  TT_RET_CHECK(!self_is_result || negative_slope.scalar().toDouble() >= 0,
               error::kInvalidArgument)
      << "expected slopes to be >= 0 for self_is_result=true, got "
      << negative_slope.scalar().toDouble();

  TT_RETURN_IF_ERROR(ValidateOnPrivateUse1Device(grad_output, "grad_output"));
  TT_RETURN_IF_ERROR(
      ValidateOnPrivateUse1Device(self_or_result, "self_or_result"));
  TT_RETURN_IF_ERROR(ValidateOnPrivateUse1Device(grad_input, "grad_input"));
  TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(grad_output));
  TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(self_or_result));

  const at::ScalarType promoted_type =
      at::result_type(grad_output, self_or_result);
  TT_ASSIGN_OR_RETURN(const auto dtype,
                      ConvertTo<mlir::ElementType>(promoted_type));
  TT_RET_CHECK(!IsBoolean(dtype), error::kInvalidArgument)
      << "bool input is unsupported";
  TT_RET_CHECK(!IsInteger(dtype, /*includeBool=*/false),
               error::kInvalidArgument)
      << "expected non-integer dtype, got " << ToString(promoted_type);
  TT_RET_CHECK(!IsComplex(dtype), error::kInvalidArgument)
      << "expected non-complex dtype, got " << ToString(promoted_type);

  TT_RETURN_IF_ERROR(ValidateOutDtype(grad_input, promoted_type,
                                      /*allow_out_dtype_cast=*/true));
  TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(grad_input));

  TT_ASSIGN_OR_RETURN(Dimensions out_dims,
                      InferSize(grad_output.sizes(), self_or_result.sizes()));
  if (grad_input.is_alias_of(grad_output) ||
      grad_input.is_alias_of(self_or_result)) {
    TT_RET_CHECK(absl::MakeConstSpan(grad_input.sizes()) == out_dims,
                 error::kInvalidArgument)
        << "output with shape " << ToString(grad_input.sizes())
        << " doesn't match the broadcast shape of the tensor being operated on "
           "in-place, which has shape "
        << ToString(out_dims);
  }

  const at::ScalarType comp_scalar_type = InferComputationDtype(promoted_type);
  TT_ASSIGN_OR_RETURN(at::Tensor negative_slope_tensor,
                      negative_slope.GetTensor(comp_scalar_type));

  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(grad_input.scalar_type()));
  TT_ASSIGN_OR_RETURN(const auto computation_dtype,
                      ConvertTo<mlir::ElementType>(comp_scalar_type));

  auto op_builder = [computation_dtype](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    return BuildLeakyReluBackwardShlo(inputs[0], inputs[1], inputs[2],
                                      computation_dtype);
  };

  return DispatchOpOut<3>(std::move(op_builder),
                          {grad_output, self_or_result, negative_slope_tensor},
                          grad_input,
                          {.out_dtype = out_dtype,
                           .out_dims = std::move(out_dims),
                           .computation_dtype = computation_dtype,
                           .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor& AtenLeakyReluOut(const at::Tensor& self,
                             const at::Scalar& negative_slope,
                             at::Tensor& out) {
  auto promoted_negative_slope = PromoteScalar(negative_slope);
  TT_KERNEL(
      OpName::kLeakyReluOut, param_keys, (self, promoted_negative_slope, out), {
        TT_THROW_IF_ERROR(
            LeakyReluScalarHelper(self, std::move(promoted_negative_slope), out,
                                  std::move(param_keys)));
        return out;
      });
}

at::Tensor& AtenLeakyReluBackwardGradInput(const at::Tensor& grad_output,
                                           const at::Tensor& self_or_result,
                                           const at::Scalar& negative_slope,
                                           bool self_is_result,
                                           at::Tensor& grad_input) {
  auto promoted_negative_slope = PromoteScalar(negative_slope);
  TT_KERNEL(
      OpName::kLeakyReluBackward, param_keys,
      (grad_output, self_or_result, promoted_negative_slope, self_is_result,
       grad_input),
      {
        TT_THROW_IF_ERROR(LeakyReluBackwardScalarHelper(
            grad_output, self_or_result, std::move(promoted_negative_slope),
            self_is_result, std::move(param_keys), grad_input));
        return grad_input;
      });
}

}  // namespace torch_tpu
