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

#include "torch_tpu/ops/softplus/softplus_aten_kernels.h"

#include <functional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/resize/resize_aten_kernels.h"
#include "torch_tpu/ops/unary_aten_kernels.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildSoftplusShlo(mlir::MlirOp input_op,
                                               mlir::MlirOp beta_op,
                                               mlir::MlirOp threshold_op) {
  TT_ASSIGN_OR_RETURN((auto [input_bcast, beta_bcast, threshold_bcast]),
                      ApplyBroadcastIfNeeded(input_op, beta_op, threshold_op));

  auto beta_x = mlir::stablehlo::Mul(input_bcast, beta_bcast);
  auto exp_beta_x = mlir::stablehlo::Exp(beta_x);
  auto log1p_exp_beta_x = mlir::stablehlo::Log1p(exp_beta_x);
  auto result = mlir::stablehlo::Div(log1p_exp_beta_x, beta_bcast);

  // Threshold judgment: beta * x > threshold
  // If the condition is met, reverts to the linear function
  mlir::MlirOp compare_gt_threshold = mlir::stablehlo::Compare(
      beta_x, threshold_bcast, mlir::stablehlo::ComparisonDirection::GT);

  return mlir::stablehlo::Select(compare_gt_threshold, input_bcast, result);
}

absl::StatusOr<mlir::MlirOp> BuildSoftplusBackwardShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp input_op, mlir::MlirOp beta_op,
    mlir::MlirOp threshold_op) {
  TT_ASSIGN_OR_RETURN(
      (auto [grad_output_bcast, input_bcast, beta_bcast, threshold_bcast]),
      ApplyBroadcastIfNeeded(grad_output_op, input_op, beta_op, threshold_op));

  // grad_output * z / (z + 1.0), where z = (x * beta).exp()
  // Here we use StableHLO.LogisticOp, where Logistic(z) = z / (z + 1.0)
  auto beta_x = mlir::stablehlo::Mul(input_bcast, beta_bcast);
  auto sigmoid_beta_x = mlir::stablehlo::Logistic(beta_x);
  auto grad_scaled = mlir::stablehlo::Mul(grad_output_bcast, sigmoid_beta_x);

  // Threshold judgment: beta * x > threshold
  // If the condition is met, reverts to the linear function
  mlir::MlirOp condition = mlir::stablehlo::Compare(
      beta_x, threshold_bcast, mlir::stablehlo::ComparisonDirection::GT);

  return mlir::stablehlo::Select(condition, grad_output_bcast, grad_scaled);
}
}  // namespace

at::Tensor& AtenSoftplusOut(const at::Tensor& self, const at::Scalar& beta,
                            const at::Scalar& threshold, at::Tensor& out) {
  auto promoted_beta = PromoteScalar(beta);
  auto promoted_threshold = PromoteScalar(threshold);
  TT_KERNEL(
      OpName::kSoftplusOut, param_keys,
      (self, promoted_beta, promoted_threshold, out), {
        TT_CHECK_THROW(
            !isIntegralType(self.scalar_type(), /*includeBool=*/true) &&
                !IsComplex(self),
            error::kPythonNotImplementedError)
            << "expected the input dtype to be floating-point, "
            << "got " << ToString(self.scalar_type());

        TT_ASSIGN_OR_THROW(const at::Tensor beta_tensor,
                           promoted_beta.GetTensor(self.scalar_type()));
        TT_ASSIGN_OR_THROW(const at::Tensor threshold_tensor,
                           promoted_threshold.GetTensor(self.scalar_type()));

        TT_ASSIGN_OR_THROW(const mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));

        auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto [self, beta, threshold] = inputs;
          return BuildSoftplusShlo(self, beta, threshold);
        };

        TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, self.sizes()));
        TT_ASSIGN_OR_THROW(
            auto result_buf,
            DispatchOp<3>(std::move(op_builder),
                          {self, beta_tensor, threshold_tensor},
                          {.out_dtype = out_dtype,
                           .out_dims = CopyIntVector(out.sizes()),
                           .op_param_cache_keys = std::move(param_keys)}));

        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

at::Tensor& AtenSoftplusBackwardGradInput(const at::Tensor& grad_output,
                                          const at::Tensor& self,
                                          const at::Scalar& beta,
                                          const at::Scalar& threshold,
                                          at::Tensor& grad_input) {
  auto promoted_beta = PromoteScalar(beta);
  auto promoted_threshold = PromoteScalar(threshold);
  TT_KERNEL(
      OpName::kSoftplusBackwardGradInput, param_keys,
      (grad_output, self, promoted_beta, promoted_threshold, grad_input), {
        TT_CHECK_THROW(
            !isIntegralType(self.scalar_type(), /*includeBool=*/true) &&
                !IsComplex(self),
            error::kPythonNotImplementedError)
            << "expected the input dtype to be floating-point, got "
            << ToString(self.scalar_type());

        TT_CHECK_THROW(grad_output.scalar_type() == self.scalar_type() &&
                           grad_input.scalar_type() == self.scalar_type(),
                       error::kInvalidArgument)
            << "expected dtype to be the same across grad_output, self, and "
            << "grad_input, got grad_output="
            << ToString(grad_output.scalar_type())
            << ", self=" << ToString(self.scalar_type())
            << ", and grad_input=" << ToString(grad_input.scalar_type());

        TT_ASSIGN_OR_THROW(const at::Tensor beta_tensor,
                           promoted_beta.GetTensor(grad_input.scalar_type()));
        TT_ASSIGN_OR_THROW(
            const at::Tensor threshold_tensor,
            promoted_threshold.GetTensor(grad_input.scalar_type()));

        TT_ASSIGN_OR_THROW(
            const mlir::ElementType out_dtype,
            ConvertTo<mlir::ElementType>(grad_input.scalar_type()));

        auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 4> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto [grad_output, self, beta, threshold] = inputs;
          return BuildSoftplusBackwardShlo(grad_output, self, beta, threshold);
        };

        TT_ASSIGN_OR_THROW(
            auto result_buf,
            DispatchOp<4>(std::move(op_builder),
                          {grad_output, self, beta_tensor, threshold_tensor},
                          {.out_dtype = out_dtype,
                           .out_dims = CopyIntVector(grad_input.sizes()),
                           .op_param_cache_keys = std::move(param_keys)}));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result_buf), grad_input));
        return grad_input;
      });
}
}  // namespace torch_tpu
