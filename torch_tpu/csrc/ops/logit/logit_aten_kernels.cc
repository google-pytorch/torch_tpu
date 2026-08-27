/*
 * Copyright 2026 Google LLC
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

#include "torch_tpu/csrc/ops/logit/logit_aten_kernels.h"

#include <array>
#include <limits>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Scalar.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/csrc/common/aten_utils.h"
#include "torch_tpu/csrc/common/cache_key.h"
#include "torch_tpu/csrc/common/dtype.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/fixed_size_span.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/common/utils.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/op_dispatcher.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/csrc/ops/resize/resize_aten_kernels.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildLogitShlo(mlir::MlirOp input_op,
                                            mlir::MlirOp eps_op,
                                            mlir::ElementType output_dtype) {
  TT_ASSIGN_OR_RETURN(const mlir::ElementType computation_dtype,
                      InferComputationDtype(output_dtype));

  TT_ASSIGN_OR_RETURN(const mlir::MlirOp computation_input_op,
                      CastIfNeeded(input_op, computation_dtype));
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp computation_eps_op,
                      CastIfNeeded(eps_op, computation_dtype));

  std::array<mlir::MlirOp, 2> broadcasted_inputs;
  TT_ASSIGN_OR_RETURN(
      broadcasted_inputs,
      ApplyBroadcastIfNeeded(computation_input_op, computation_eps_op));
  mlir::MlirOp self_bcst = broadcasted_inputs[0];
  mlir::MlirOp eps_bcst = broadcasted_inputs[1];

  mlir::MlirOp k_zero = MakeConstantLike(self_bcst, 0.0);
  mlir::MlirOp k_one = MakeConstantLike(self_bcst, 1.0);

  // Formula:
  //   If eps >= 0:
  //     If self < eps, y = eps
  //     Else if self > 1 - eps, y = 1 - eps
  //     Else y = self
  //   Else:
  //     y = self
  //
  //   logit(y) = log(y / (1 - y))

  // 1 - eps
  mlir::MlirOp one_minus_eps = mlir::stablehlo::Subtract(k_one, eps_bcst);
  // self > 1 - eps
  mlir::MlirOp is_greater_than_hi = mlir::stablehlo::Compare(
      self_bcst, one_minus_eps, mlir::stablehlo::ComparisonDirection::GT);
  // temp_clamped = (self > 1 - eps) ? 1 - eps : self
  mlir::MlirOp temp_clamped =
      mlir::stablehlo::Select(is_greater_than_hi, one_minus_eps, self_bcst);

  // self < eps
  mlir::MlirOp is_less_than_lo = mlir::stablehlo::Compare(
      self_bcst, eps_bcst, mlir::stablehlo::ComparisonDirection::LT);
  // clamped_input = (self < eps) ? eps : temp_clamped
  mlir::MlirOp clamped_input =
      mlir::stablehlo::Select(is_less_than_lo, eps_bcst, temp_clamped);

  // y = (eps >= 0) ? clamped_input : self
  mlir::MlirOp compare_ge_zero = mlir::stablehlo::Compare(
      eps_bcst, k_zero, mlir::stablehlo::ComparisonDirection::GE);
  mlir::MlirOp y =
      mlir::stablehlo::Select(compare_ge_zero, clamped_input, self_bcst);

  // logit(y) = log(y / (1 - y))
  mlir::MlirOp one_minus_y = mlir::stablehlo::Subtract(k_one, y);
  mlir::MlirOp fraction = mlir::stablehlo::Div(y, one_minus_y);
  mlir::MlirOp log_out = mlir::stablehlo::Log(fraction);

  return mlir::stablehlo::ConvertElementType(log_out, output_dtype);
}

absl::StatusOr<DeviceBufferRef> BuildLogitBuffer(const at::Tensor& self,
                                                 PromotedScalar& promoted_eps,
                                                 c10::ScalarType out_type,
                                                 OpParamCacheKeys param_keys) {
  const c10::ScalarType self_dtype = self.scalar_type();
  TT_RET_CHECK(!c10::isComplexType(self_dtype),
               error::kPythonNotImplementedError)
      << "expected non-complex dtype, got " << ToString(self_dtype);

  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(out_type));

  TT_ASSIGN_OR_RETURN(at::Tensor eps_tensor, promoted_eps.GetTensor(out_type));

  auto op_builder = [out_dtype](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    const auto [self_op, eps_op] = inputs;
    return BuildLogitShlo(self_op, eps_op, out_dtype);
  };

  return DispatchOp<2>(std::move(op_builder), {self, eps_tensor},
                       {.out_dtype = out_dtype,
                        .out_dims = CopyIntVector(self.sizes()),
                        .op_param_cache_keys = std::move(param_keys)});
}

absl::StatusOr<mlir::MlirOp> BuildLogitBackwardShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp self_op, mlir::MlirOp eps_op,
    mlir::ElementType output_dtype) {
  // Formula:
  //   lo = eps
  //   hi = 1 - eps
  //   grad = grad_output / (self * (1 - self))
  //
  //   If eps >= 0:
  //     If self >= lo and self <= hi, return grad
  //     Else return 0 (since forward pass clamped to constant)
  //   Else:
  //     If self >= 0 and self <= 1, return grad
  //     Else return NaN
  TT_ASSIGN_OR_RETURN(const mlir::ElementType computation_dtype,
                      InferComputationDtype(output_dtype));
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp computation_grad_output_op,
                      CastIfNeeded(grad_output_op, computation_dtype));
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp computation_self_op,
                      CastIfNeeded(self_op, computation_dtype));
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp computation_eps_op,
                      CastIfNeeded(eps_op, computation_dtype));

  std::array<mlir::MlirOp, 3> broadcasted_inputs;
  TT_ASSIGN_OR_RETURN(
      broadcasted_inputs,
      ApplyBroadcastIfNeeded(computation_grad_output_op, computation_self_op,
                             computation_eps_op));
  mlir::MlirOp grad_output_bcst = broadcasted_inputs[0];
  mlir::MlirOp self_bcst = broadcasted_inputs[1];
  mlir::MlirOp eps_bcst = broadcasted_inputs[2];

  mlir::MlirOp k_zero = MakeConstantLike(self_bcst, 0.0);
  mlir::MlirOp k_one = MakeConstantLike(self_bcst, 1.0);
  mlir::MlirOp k_nan =
      MakeConstantLike(self_bcst, std::numeric_limits<double>::quiet_NaN());

  // grad_output / (self * (1 - self))
  mlir::MlirOp one_minus_self = mlir::stablehlo::Subtract(k_one, self_bcst);
  mlir::MlirOp denom = mlir::stablehlo::Mul(self_bcst, one_minus_self);
  mlir::MlirOp div_res = mlir::stablehlo::Div(grad_output_bcst, denom);

  // eps >= 0 case
  mlir::MlirOp lo = eps_bcst;
  mlir::MlirOp hi = mlir::stablehlo::Subtract(k_one, eps_bcst);

  // Check if self is in range: lo <= self <= hi
  mlir::MlirOp ge_lo = mlir::stablehlo::Compare(
      self_bcst, lo, mlir::stablehlo::ComparisonDirection::GE);
  mlir::MlirOp le_hi = mlir::stablehlo::Compare(
      self_bcst, hi, mlir::stablehlo::ComparisonDirection::LE);
  mlir::MlirOp in_range_eps = mlir::stablehlo::And(ge_lo, le_hi);

  mlir::MlirOp res_eps = mlir::stablehlo::Select(in_range_eps, div_res, k_zero);

  // eps < 0 case
  mlir::MlirOp ge_zero = mlir::stablehlo::Compare(
      self_bcst, k_zero, mlir::stablehlo::ComparisonDirection::GE);
  mlir::MlirOp le_one = mlir::stablehlo::Compare(
      self_bcst, k_one, mlir::stablehlo::ComparisonDirection::LE);
  mlir::MlirOp in_range_no_eps = mlir::stablehlo::And(ge_zero, le_one);
  mlir::MlirOp res_no_eps =
      mlir::stablehlo::Select(in_range_no_eps, div_res, k_nan);

  // Select between eps >= 0 and eps < 0
  mlir::MlirOp compare_ge_zero = mlir::stablehlo::Compare(
      eps_bcst, k_zero, mlir::stablehlo::ComparisonDirection::GE);
  mlir::MlirOp final_res =
      mlir::stablehlo::Select(compare_ge_zero, res_eps, res_no_eps);

  return mlir::stablehlo::ConvertElementType(final_res, output_dtype);
}

}  // namespace

at::Tensor AtenLogit(const at::Tensor& self, std::optional<double> eps) {
  PromotedScalar promoted_eps = PromoteScalar(at::Scalar(eps.value_or(-1.0)));
  TT_KERNEL(OpName::kLogit, param_keys, (self, promoted_eps), {
    const c10::ScalarType out_type = InferOutputDtype(self);
    TT_ASSIGN_OR_THROW(
        DeviceBufferRef result_buf,
        BuildLogitBuffer(self, promoted_eps, out_type, std::move(param_keys)));

    return MakeTensor(std::move(result_buf));
  });
}

at::Tensor& AtenLogitOut(const at::Tensor& self, std::optional<double> eps,
                         at::Tensor& out) {
  PromotedScalar promoted_eps = PromoteScalar(at::Scalar(eps.value_or(-1.0)));
  TT_KERNEL(OpName::kLogitOut, param_keys, (self, promoted_eps, out), {
    const c10::ScalarType out_type = out.scalar_type();
    TT_ASSIGN_OR_THROW(
        DeviceBufferRef result_buf,
        BuildLogitBuffer(self, promoted_eps, out_type, std::move(param_keys)));

    TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, self.sizes()));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
    return out;
  });
}
at::Tensor& AtenLogit_(at::Tensor& self, std::optional<double> eps) {
  PromotedScalar promoted_eps = PromoteScalar(at::Scalar(eps.value_or(-1.0)));
  TT_KERNEL(OpName::kLogit_, param_keys, (self, promoted_eps), {
    const c10::ScalarType out_type = self.scalar_type();
    TT_ASSIGN_OR_THROW(
        DeviceBufferRef result_buf,
        BuildLogitBuffer(self, promoted_eps, out_type, std::move(param_keys)));

    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), self));
    return self;
  });
}

at::Tensor& AtenLogitBackwardGradInput(const at::Tensor& grad_output,
                                       const at::Tensor& self,
                                       std::optional<double> eps,
                                       at::Tensor& grad_input) {
  PromotedScalar promoted_eps = PromoteScalar(at::Scalar(eps.value_or(-1.0)));
  TT_KERNEL(
      OpName::kLogitBackwardGradInput, param_keys,
      (grad_output, self, promoted_eps, grad_input), {
        const c10::ScalarType out_type = grad_input.scalar_type();
        const c10::ScalarType self_dtype = self.scalar_type();
        TT_CHECK_THROW(!c10::isComplexType(self_dtype),
                       error::kPythonNotImplementedError)
            << "expected non-complex dtype, got " << ToString(self_dtype);

        TT_ASSIGN_OR_THROW(const auto out_dtype,
                           ConvertTo<mlir::ElementType>(out_type));

        TT_ASSIGN_OR_THROW(const at::Tensor eps_tensor,
                           promoted_eps.GetTensor(out_type));

        auto op_builder = [out_dtype](FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          const auto [grad_output_op, self_op, eps_op] = inputs;
          return BuildLogitBackwardShlo(grad_output_op, self_op, eps_op,
                                        out_dtype);
        };

        TT_ASSIGN_OR_THROW(
            DeviceBufferRef result_buf,
            DispatchOp<3>(std::move(op_builder),
                          {grad_output, self, eps_tensor},
                          {.out_dtype = out_dtype,
                           .out_dims = CopyIntVector(self.sizes()),
                           .op_param_cache_keys = std::move(param_keys)}));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result_buf), grad_input));
        return grad_input;
      });
}

}  // namespace torch_tpu
