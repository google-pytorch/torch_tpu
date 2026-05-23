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

#include "torch_tpu/ops/elu/elu_aten_kernels.h"

#include <functional>
#include <string_view>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
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

absl::StatusOr<mlir::MlirOp> BuildEluShlo(mlir::MlirOp input_op,
                                          const at::Scalar& alpha,
                                          const at::Scalar& scale,
                                          const at::Scalar& input_scale) {
  // Create constants for the scalar parameters.
  auto alpha_op = MakeConstantLike(input_op, alpha.toDouble());
  auto scale_op = MakeConstantLike(input_op, scale.toDouble());
  auto input_scale_op = MakeConstantLike(input_op, input_scale.toDouble());
  auto zero_op = MakeConstantLike(input_op, 0.0);
  auto one = MakeConstantLike(input_op, 1.0);

  //  y = self * input_scale
  auto input_scaled = mlir::stablehlo::Mul(input_op, input_scale_op);
  auto pred = mlir::stablehlo::Compare(
      input_scaled, zero_op, mlir::stablehlo::ComparisonDirection::GT);

  // --- Positive path: scale * input_scaled ---
  auto positive_val = mlir::stablehlo::Mul(input_scaled, scale_op);

  // --- Negative path: scale * alpha * (exp(input_scaled) - 1) ---
  auto exp_input_scaled = mlir::stablehlo::Exp(input_scaled);
  auto exp_input_scaled_minus_1 =
      mlir::stablehlo::Subtract(exp_input_scaled, one);
  auto exp_alpha_scaled =
      mlir::stablehlo::Mul(alpha_op, exp_input_scaled_minus_1);
  auto negative_val = mlir::stablehlo::Mul(exp_alpha_scaled, scale_op);

  return mlir::stablehlo::Select(pred, /*on_true=*/positive_val,
                                 /*on_false=*/negative_val);
}

absl::StatusOr<mlir::MlirOp> BuildEluBackwardGradInputShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp self_or_result_op,
    const at::Scalar& alpha, const at::Scalar& scale,
    const at::Scalar& input_scale, bool is_result) {
  auto alpha_op = MakeConstantLike(grad_output_op, alpha.toDouble());
  auto scale_op = MakeConstantLike(grad_output_op, scale.toDouble());
  auto input_scale_op =
      MakeConstantLike(grad_output_op, input_scale.toDouble());
  auto zero_op = MakeConstantLike(grad_output_op, 0.0);

  auto negcoef = mlir::stablehlo::Mul(alpha_op, scale_op);
  auto poscoef = scale_op;
  auto negiptcoef = input_scale_op;

  // --- Positive path: grad_output * poscoef ---
  auto positive_val = mlir::stablehlo::Mul(grad_output_op, poscoef);

  // --- Negative path: ---
  mlir::MlirOp negative_val;
  if (is_result) {
    // if is_result: grad_output * negiptcoef * (self_or_result + negcoef)
    auto term_sum = mlir::stablehlo::Add(self_or_result_op, negcoef);
    auto term_mul = mlir::stablehlo::Mul(grad_output_op, negiptcoef);
    negative_val = mlir::stablehlo::Mul(term_sum, term_mul);
  } else {
    // else: grad_output * negiptcoef * negcoef *
    // torch.exp(self_or_result * negiptcoef)
    auto inner = mlir::stablehlo::Mul(self_or_result_op, negiptcoef);
    auto exp_inner = mlir::stablehlo::Exp(inner);
    auto exp_mul = mlir::stablehlo::Mul(exp_inner, negcoef);
    auto term_exp_mul = mlir::stablehlo::Mul(exp_mul, negiptcoef);
    negative_val = mlir::stablehlo::Mul(grad_output_op, term_exp_mul);
  }

  // Comparison condition: self_or_result <= 0
  auto pred = mlir::stablehlo::Compare(
      self_or_result_op, zero_op, mlir::stablehlo::ComparisonDirection::LE);

  return mlir::stablehlo::Select(pred, /*on_true=*/negative_val,
                                 /*on_false=*/positive_val);
}

// Returns an MlirUnaryOpBuilder that captures alpha, scale, and input_scale.
MlirUnaryOpBuilder GetEluFunctional(const at::Scalar& alpha,
                                    const at::Scalar& scale,
                                    const at::Scalar& input_scale) {
  return std::bind(&BuildEluShlo, std::placeholders::_1, alpha, scale,
                   input_scale);
};

absl::Status CheckIsFloatingPoint(const at::Tensor& tensor,
                                  const std::string_view name) {
  TT_RET_CHECK(IsFloatingPoint(tensor), error::kInvalidArgument)
      << "expected the " << name << " dtype to be floating point, got "
      << ToString(tensor.scalar_type());
  return absl::OkStatus();
}

}  // namespace

at::Tensor& AtenEluOut(const at::Tensor& input, const at::Scalar& alpha,
                       const at::Scalar& scale, const at::Scalar& input_scale,
                       at::Tensor& out) {
  TT_KERNEL(
      OpName::kEluOut, param_keys, (input, alpha, scale, input_scale, out), {
        TT_THROW_IF_ERROR(CheckIsFloatingPoint(input, /* name= */ "input"));
        TT_THROW_IF_ERROR(
            UnaryOpOut(input, out, GetEluFunctional(alpha, scale, input_scale),
                       {.op_param_cache_keys = std::move(param_keys)}));
        return out;
      });
}

at::Tensor& AtenEluBackwardGradInput(
    const at::Tensor& grad_output, const at::Scalar& alpha,
    const at::Scalar& scale, const at::Scalar& input_scale, bool is_result,
    const at::Tensor& self_or_result, at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kEluBackwardGradInput, param_keys,
      (grad_output, alpha, scale, input_scale, is_result, self_or_result,
       grad_input),
      {
        TT_THROW_IF_ERROR(
            CheckIsFloatingPoint(grad_output, /* name= */ "grad output"));

        auto op_builder = [alpha, scale, input_scale,
                           is_result](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [grad_output_op, self_or_result_op] = inputs;
          return BuildEluBackwardGradInputShlo(grad_output_op,
                                               self_or_result_op, alpha, scale,
                                               input_scale, is_result);
        };

        TT_ASSIGN_OR_THROW(
            const auto output_dtype,
            ConvertTo<mlir::ElementType>(grad_input.scalar_type()));
        auto output_shape = CopyIntVector(grad_input.sizes());

        TT_ASSIGN_OR_THROW(
            auto result,
            DispatchOp<2>(std::move(op_builder), {grad_output, self_or_result},
                          /*options=*/
                          {.out_dtype = output_dtype,
                           .out_dims = output_shape,
                           .op_param_cache_keys = std::move(param_keys)}));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result), grad_input));
        return grad_input;
      });
}
}  // namespace torch_tpu
