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

#include "csrc/ops/elu/elu_aten_kernels.h"

#include <string_view>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/aten_utils.h"
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

namespace torch_tpu {

namespace {

absl::StatusOr<mlir::MlirOp> BuildEluShlo(mlir::MlirOp input_op,
                                          mlir::MlirOp alpha_op,
                                          mlir::MlirOp scale_op,
                                          mlir::MlirOp input_scale_op) {
  TT_ASSIGN_OR_RETURN(
      (auto [input_bcast, alpha_bcast, scale_bcast, input_scale_bcast]),
      ApplyBroadcastIfNeeded(input_op, alpha_op, scale_op, input_scale_op));

  auto zero_op = MakeConstantLike(input_bcast, 0.0);
  auto one = MakeConstantLike(input_bcast, 1.0);

  //  y = self * input_scale
  auto input_scaled = mlir::stablehlo::Mul(input_bcast, input_scale_bcast);
  auto pred = mlir::stablehlo::Compare(
      input_scaled, zero_op, mlir::stablehlo::ComparisonDirection::GT);

  // --- Positive path: scale * input_scaled ---
  auto positive_val = mlir::stablehlo::Mul(input_scaled, scale_bcast);

  // --- Negative path: scale * alpha * (exp(input_scaled) - 1) ---
  auto exp_input_scaled = mlir::stablehlo::Exp(input_scaled);
  auto exp_input_scaled_minus_1 =
      mlir::stablehlo::Subtract(exp_input_scaled, one);
  auto exp_alpha_scaled =
      mlir::stablehlo::Mul(alpha_bcast, exp_input_scaled_minus_1);
  auto negative_val = mlir::stablehlo::Mul(exp_alpha_scaled, scale_bcast);

  return mlir::stablehlo::Select(pred, /*on_true=*/positive_val,
                                 /*on_false=*/negative_val);
}

absl::StatusOr<mlir::MlirOp> BuildEluBackwardGradInputShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp self_or_result_op,
    mlir::MlirOp alpha_op, mlir::MlirOp scale_op, mlir::MlirOp input_scale_op,
    bool is_result) {
  TT_ASSIGN_OR_RETURN(
      (auto [grad_output_bcast, self_or_result_bcast, alpha_bcast, scale_bcast,
             input_scale_bcast]),
      ApplyBroadcastIfNeeded(grad_output_op, self_or_result_op, alpha_op,
                             scale_op, input_scale_op));

  auto zero_op = MakeConstantLike(grad_output_bcast, 0.0);

  auto negcoef = mlir::stablehlo::Mul(alpha_bcast, scale_bcast);
  auto poscoef = scale_bcast;
  auto negiptcoef = input_scale_bcast;

  // --- Positive path: grad_output * poscoef ---
  auto positive_val = mlir::stablehlo::Mul(grad_output_bcast, poscoef);

  // --- Negative path: ---
  mlir::MlirOp negative_val;
  if (is_result) {
    // if is_result: grad_output * negiptcoef * (self_or_result + negcoef)
    auto term_sum = mlir::stablehlo::Add(self_or_result_bcast, negcoef);
    auto term_mul = mlir::stablehlo::Mul(grad_output_bcast, negiptcoef);
    negative_val = mlir::stablehlo::Mul(term_sum, term_mul);
  } else {
    // else: grad_output * negiptcoef * negcoef *
    // torch.exp(self_or_result * negiptcoef)
    auto inner = mlir::stablehlo::Mul(self_or_result_bcast, negiptcoef);
    auto exp_inner = mlir::stablehlo::Exp(inner);
    auto exp_mul = mlir::stablehlo::Mul(exp_inner, negcoef);
    auto term_exp_mul = mlir::stablehlo::Mul(exp_mul, negiptcoef);
    negative_val = mlir::stablehlo::Mul(grad_output_bcast, term_exp_mul);
  }

  // Comparison condition: self_or_result <= 0
  auto pred = mlir::stablehlo::Compare(
      self_or_result_bcast, zero_op, mlir::stablehlo::ComparisonDirection::LE);

  return mlir::stablehlo::Select(pred, /*on_true=*/negative_val,
                                 /*on_false=*/positive_val);
}

absl::Status ValidateIsFloatingPoint(const at::Tensor& tensor,
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
  auto promoted_alpha = PromoteScalar(alpha);
  auto promoted_scale = PromoteScalar(scale);
  auto promoted_input_scale = PromoteScalar(input_scale);
  TT_KERNEL(
      OpName::kEluOut, param_keys,
      (input, promoted_alpha, promoted_scale, promoted_input_scale, out), {
        TT_THROW_IF_ERROR(ValidateIsFloatingPoint(input, /* name= */ "input"));

        TT_ASSIGN_OR_THROW(const at::Tensor alpha_tensor,
                           promoted_alpha.GetTensor(input.scalar_type()));
        TT_ASSIGN_OR_THROW(const at::Tensor scale_tensor,
                           promoted_scale.GetTensor(input.scalar_type()));
        TT_ASSIGN_OR_THROW(const at::Tensor input_scale_tensor,
                           promoted_input_scale.GetTensor(input.scalar_type()));

        TT_ASSIGN_OR_THROW(const mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));

        auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 4> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto [input, alpha, scale, input_scale] = inputs;
          return BuildEluShlo(input, alpha, scale, input_scale);
        };

        TT_THROW_IF_ERROR(DispatchOpOut<4>(
            std::move(op_builder),
            {input, alpha_tensor, scale_tensor, input_scale_tensor}, out,
            {.out_dtype = out_dtype,
             .out_dims = CopyIntVector(out.sizes()),
             .op_param_cache_keys = std::move(param_keys)}));
        return out;
      });
}

at::Tensor& AtenEluBackwardGradInput(
    const at::Tensor& grad_output, const at::Scalar& alpha,
    const at::Scalar& scale, const at::Scalar& input_scale, bool is_result,
    const at::Tensor& self_or_result, at::Tensor& grad_input) {
  auto promoted_alpha = PromoteScalar(alpha);
  auto promoted_scale = PromoteScalar(scale);
  auto promoted_input_scale = PromoteScalar(input_scale);
  TT_KERNEL(
      OpName::kEluBackwardGradInput, param_keys,
      (grad_output, promoted_alpha, promoted_scale, promoted_input_scale,
       is_result, self_or_result, grad_input),
      {
        TT_THROW_IF_ERROR(
            ValidateIsFloatingPoint(grad_output, /* name= */ "grad output"));

        TT_ASSIGN_OR_THROW(const at::Tensor alpha_tensor,
                           promoted_alpha.GetTensor(grad_output.scalar_type()));
        TT_ASSIGN_OR_THROW(const at::Tensor scale_tensor,
                           promoted_scale.GetTensor(grad_output.scalar_type()));
        TT_ASSIGN_OR_THROW(
            const at::Tensor input_scale_tensor,
            promoted_input_scale.GetTensor(grad_output.scalar_type()));

        auto op_builder = [is_result](FixedSizeSpan<mlir::MlirOp, 5> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto [grad_output_op, self_or_result_op, alpha_op, scale_op,
                input_scale_op] = inputs;
          return BuildEluBackwardGradInputShlo(
              grad_output_op, self_or_result_op, alpha_op, scale_op,
              input_scale_op, is_result);
        };

        TT_ASSIGN_OR_THROW(
            const auto output_dtype,
            ConvertTo<mlir::ElementType>(grad_input.scalar_type()));
        auto output_shape = CopyIntVector(grad_input.sizes());

        TT_THROW_IF_ERROR(
            DispatchOpOut<5>(std::move(op_builder),
                             {grad_output, self_or_result, alpha_tensor,
                              scale_tensor, input_scale_tensor},
                             grad_input,
                             /*options=*/
                             {.out_dtype = output_dtype,
                              .out_dims = output_shape,
                              .op_param_cache_keys = std::move(param_keys)}));
        return grad_input;
      });
}
}  // namespace torch_tpu
