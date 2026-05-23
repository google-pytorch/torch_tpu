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
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/unary_aten_kernels.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildSoftplusShlo(mlir::MlirOp input_op,
                                               const at::Scalar& beta,
                                               const at::Scalar& threshold) {
  auto beta_op = MakeConstantLike(input_op, beta.toDouble());
  auto threshold_op = MakeConstantLike(input_op, threshold.toDouble());

  auto beta_x = mlir::stablehlo::Mul(input_op, beta_op);
  auto exp_beta_x = mlir::stablehlo::Exp(beta_x);
  auto log1p_exp_beta_x = mlir::stablehlo::Log1p(exp_beta_x);
  auto result = mlir::stablehlo::Div(log1p_exp_beta_x, beta_op);

  // Threshold judgment: beta * x > threshold
  // If the condition is met, reverts to the linear function
  mlir::MlirOp compare_gt_threshold = mlir::stablehlo::Compare(
      beta_x, threshold_op, mlir::stablehlo::ComparisonDirection::GT);

  return mlir::stablehlo::Select(compare_gt_threshold, input_op, result);
}

// Returns an MlirUnaryOpBuilder that captures beta and threshold.
MlirUnaryOpBuilder GetSoftplusFunctional(const at::Scalar& beta,
                                         const at::Scalar& threshold) {
  return std::bind(&BuildSoftplusShlo, std::placeholders::_1, beta, threshold);
};

absl::StatusOr<mlir::MlirOp> BuildSoftplusBackwardShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp input_op, const at::Scalar& beta,
    const at::Scalar& threshold) {
  auto beta_op = MakeConstantLike(grad_output_op, beta.toDouble());
  auto threshold_op = MakeConstantLike(grad_output_op, threshold.toDouble());

  // grad_output * z / (z + 1.0), where z = (x * beta).exp()
  // Here we use StableHLO.LogisticOp, where Logistic(z) = z / (z + 1.0)
  auto beta_x = mlir::stablehlo::Mul(input_op, beta_op);
  auto sigmoid_beta_x = mlir::stablehlo::Logistic(beta_x);
  auto grad_scaled = mlir::stablehlo::Mul(grad_output_op, sigmoid_beta_x);

  // Threshold judgment: beta * x > threshold
  // If the condition is met, reverts to the linear function
  mlir::MlirOp condition = mlir::stablehlo::Compare(
      beta_x, threshold_op, mlir::stablehlo::ComparisonDirection::GT);

  return mlir::stablehlo::Select(condition, grad_output_op, grad_scaled);
}
}  // namespace

at::Tensor& AtenSoftplusOut(const at::Tensor& self, const at::Scalar& beta,
                            const at::Scalar& threshold, at::Tensor& out) {
  TT_KERNEL(OpName::kSoftplusOut, param_keys, (self, beta, threshold, out), {
    TT_CHECK_THROW(!isIntegralType(self.scalar_type(), /*includeBool=*/true) &&
                       self.scalar_type() != at::ScalarType::ComplexFloat,
                   error::kUnimplemented)
        << "expected the input dtype to be floating-point, "
        << "got " << ToString(self.scalar_type());
    TT_THROW_IF_ERROR(
        UnaryOpOut(self, out, GetSoftplusFunctional(beta, threshold),
                   {.op_param_cache_keys = std::move(param_keys)}));
    return out;
  });
}

at::Tensor& AtenSoftplusBackwardGradInput(const at::Tensor& grad_output,
                                          const at::Tensor& self,
                                          const at::Scalar& beta,
                                          const at::Scalar& threshold,
                                          at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kSoftplusBackwardGradInput, param_keys,
      (grad_output, self, beta, threshold, grad_input), {
        TT_ASSIGN_OR_THROW(
            mlir::ElementType out_dtype,
            ConvertTo<mlir::ElementType>(grad_input.scalar_type()));

        auto op_builder = [beta,
                           threshold](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          return BuildSoftplusBackwardShlo(inputs[0], inputs[1], beta,
                                           threshold);
        };

        TT_ASSIGN_OR_THROW(
            auto result_buf,
            DispatchOp<2>(std::move(op_builder), {grad_output, self},
                          {.out_dtype = out_dtype,
                           .out_dims = CopyIntVector(grad_input.sizes()),
                           .op_param_cache_keys = std::move(param_keys)}));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result_buf), grad_input));
        return grad_input;
      });
}
}  // namespace torch_tpu
