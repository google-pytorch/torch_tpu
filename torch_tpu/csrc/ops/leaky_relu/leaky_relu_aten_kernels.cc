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

#include "torch_tpu/csrc/ops/leaky_relu/leaky_relu_aten_kernels.h"

#include <array>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/fixed_size_span.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildLeakyReluShlo(mlir::MlirOp self_op,
                                                mlir::MlirOp negative_slope_op,
                                                mlir::ElementType out_dtype) {
  auto& builder = self_op.getBuilder();
  mlir::MlirOp zero_op = MakeScalarConstant(builder, 0.0, out_dtype);
  std::array<mlir::MlirOp, 2> broadcasted;
  TT_ASSIGN_OR_RETURN(broadcasted, ApplyBroadcastIfNeeded(self_op, zero_op));
  mlir::MlirOp compare_ge_zero = mlir::stablehlo::Compare(
      broadcasted[0], broadcasted[1], mlir::stablehlo::ComparisonDirection::GE);

  TT_ASSIGN_OR_RETURN(broadcasted,
                      ApplyBroadcastIfNeeded(self_op, negative_slope_op));
  mlir::MlirOp mul_op = mlir::stablehlo::Mul(broadcasted[0], broadcasted[1]);

  std::array<mlir::MlirOp, 3> select_broadcasted;
  TT_ASSIGN_OR_RETURN(select_broadcasted,
                      ApplyBroadcastIfNeeded(compare_ge_zero, self_op, mul_op));
  return mlir::stablehlo::Select(select_broadcasted[0], select_broadcasted[1],
                                 select_broadcasted[2]);
}

absl::StatusOr<mlir::MlirOp> BuildLeakyReluBackwardShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp self_or_result_op,
    mlir::MlirOp negative_slope_op, mlir::ElementType out_dtype) {
  auto& builder = grad_output_op.getBuilder();
  mlir::MlirOp zero_op = MakeScalarConstant(builder, 0.0, out_dtype);
  std::array<mlir::MlirOp, 2> broadcasted;
  TT_ASSIGN_OR_RETURN(broadcasted,
                      ApplyBroadcastIfNeeded(self_or_result_op, zero_op));
  mlir::MlirOp compare_ge_zero = mlir::stablehlo::Compare(
      broadcasted[0], broadcasted[1], mlir::stablehlo::ComparisonDirection::GE);

  TT_ASSIGN_OR_RETURN(
      broadcasted, ApplyBroadcastIfNeeded(grad_output_op, negative_slope_op));
  mlir::MlirOp mul_op = mlir::stablehlo::Mul(broadcasted[0], broadcasted[1]);

  std::array<mlir::MlirOp, 3> select_broadcasted;
  TT_ASSIGN_OR_RETURN(
      select_broadcasted,
      ApplyBroadcastIfNeeded(compare_ge_zero, grad_output_op, mul_op));
  return mlir::stablehlo::Select(select_broadcasted[0], select_broadcasted[1],
                                 select_broadcasted[2]);
}

absl::StatusOr<DeviceBufferRef> AtenLeakyReluTensorHelper(
    const at::Tensor& self, const at::Tensor& negative_slope,
    const at::Tensor& out, OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const auto dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_RET_CHECK(!IsBoolean(dtype), error::kInvalidArgument)
      << "expected non-bool dtype, got " << self.scalar_type();
  TT_RET_CHECK(!IsInteger(dtype, /*includeBool=*/false),
               error::kInvalidArgument)
      << "expected non-integer dtype, got " << self.scalar_type();
  TT_RET_CHECK(!IsComplex(dtype), error::kInvalidArgument)
      << "expected non-complex dtype, got " << self.scalar_type();

  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(out.scalar_type()));

  auto op_builder = [out_dtype](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    return BuildLeakyReluShlo(inputs[0], inputs[1], out_dtype);
  };
  return DispatchOp<2>(std::move(op_builder), {self, negative_slope},
                       {.out_dtype = out_dtype,
                        .out_dims = CopyIntVector(self.sizes()),
                        .op_param_cache_keys = std::move(param_keys)});
}

absl::StatusOr<DeviceBufferRef> AtenLeakyReluBackwardTensorHelper(
    const at::Tensor& grad_output, const at::Tensor& self_or_result,
    const at::Tensor& negative_slope, bool self_is_result,
    OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(const auto dtype,
                      ConvertTo<mlir::ElementType>(grad_output.scalar_type()));
  TT_RET_CHECK(!IsBoolean(dtype), error::kInvalidArgument)
      << "boolean dtypes are not supported, got " << grad_output.scalar_type();
  TT_RET_CHECK(!IsInteger(dtype, /*includeBool=*/false),
               error::kInvalidArgument)
      << "integer dtypes are not supported, got " << grad_output.scalar_type();
  TT_RET_CHECK(!IsComplex(dtype), error::kInvalidArgument)
      << "complex dtypes are not supported, got " << grad_output.scalar_type();

  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(grad_output.scalar_type()));

  auto op_builder = [out_dtype](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    return BuildLeakyReluBackwardShlo(inputs[0], inputs[1], inputs[2],
                                      out_dtype);
  };
  return DispatchOp<3>(std::move(op_builder),
                       {grad_output, self_or_result, negative_slope},
                       {.out_dtype = out_dtype,
                        .out_dims = CopyIntVector(grad_output.sizes()),
                        .op_param_cache_keys = std::move(param_keys)});
}

absl::StatusOr<DeviceBufferRef> LeakyReluScalarHelper(
    const at::Tensor& self, PromotedScalar negative_slope,
    const at::Tensor& out, OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(at::Tensor negative_slope_tensor,
                      negative_slope.GetTensor(self.scalar_type()));
  return AtenLeakyReluTensorHelper(self, negative_slope_tensor, out,
                                   std::move(param_keys));
}

absl::StatusOr<DeviceBufferRef> LeakyReluBackwardScalarHelper(
    const at::Tensor& grad_output, const at::Tensor& self_or_result,
    PromotedScalar negative_slope, bool self_is_result,
    OpParamCacheKeys param_keys) {
  TT_RET_CHECK(!self_is_result || negative_slope.scalar().toDouble() >= 0,
               error::kInvalidArgument)
      << "expected non-negative slopes for self_is_result=true, got "
      << negative_slope.scalar().toDouble();
  TT_ASSIGN_OR_RETURN(at::Tensor negative_slope_tensor,
                      negative_slope.GetTensor(grad_output.scalar_type()));
  return AtenLeakyReluBackwardTensorHelper(
      grad_output, self_or_result, negative_slope_tensor, self_is_result,
      std::move(param_keys));
}

}  // namespace

at::Tensor& AtenLeakyReluOut(const at::Tensor& self,
                             const at::Scalar& negative_slope,
                             at::Tensor& out) {
  auto promoted_negative_slope = PromoteScalar(negative_slope);
  TT_KERNEL(
      OpName::kLeakyReluOut, param_keys, (self, promoted_negative_slope, out), {
        TT_ASSIGN_OR_THROW(
            auto result_buf,
            LeakyReluScalarHelper(self, std::move(promoted_negative_slope), out,
                                  std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

at::Tensor& AtenLeakyReluBackwardGradInput(const at::Tensor& grad_output,
                                           const at::Tensor& self_or_result,
                                           const at::Scalar& negative_slope,
                                           bool self_is_result,
                                           at::Tensor& grad_input) {
  auto promoted_negative_slope = PromoteScalar(negative_slope);
  TT_KERNEL(OpName::kLeakyReluBackward, param_keys,
            (grad_output, self_or_result, promoted_negative_slope,
             self_is_result, grad_input),
            {
              TT_ASSIGN_OR_THROW(auto result_buf,
                                 LeakyReluBackwardScalarHelper(
                                     grad_output, self_or_result,
                                     std::move(promoted_negative_slope),
                                     self_is_result, std::move(param_keys)));
              TT_THROW_IF_ERROR(
                  AssignBufferToAtTensor(std::move(result_buf), grad_input));
              return grad_input;
            });
}

}  // namespace torch_tpu
