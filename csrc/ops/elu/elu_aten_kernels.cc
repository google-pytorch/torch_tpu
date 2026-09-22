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
#include "csrc/ops/binary.h"
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
  const auto elem_type = GetTensorTypeOrDie(input_op).getElementType();
  auto zero_op = MakeScalarConstant(input_op.getBuilder(), 0.0, elem_type);
  auto one_op = MakeScalarConstant(input_op.getBuilder(), 1.0, elem_type);

  // Precompute scalar coefficient: scale * alpha
  TT_ASSIGN_OR_RETURN(auto alpha_scale, BuildMulShlo(alpha_op, scale_op));

  // Predicate: input > 0
  TT_ASSIGN_OR_RETURN(auto pred, BuildGtShlo(input_op, zero_op));

  // --- Positive path: scale * input ---
  TT_ASSIGN_OR_RETURN(auto positive_val, BuildMulShlo(input_op, scale_op));

  // --- Negative path: (scale * alpha) * (exp(input * input_scale) - 1) ---
  TT_ASSIGN_OR_RETURN(auto input_scaled,
                      BuildMulShlo(input_op, input_scale_op));
  auto exp_input_scaled = mlir::stablehlo::Exp(input_scaled);
  TT_ASSIGN_OR_RETURN(auto exp_input_scaled_minus_1,
                      BuildSubShlo(exp_input_scaled, one_op));
  TT_ASSIGN_OR_RETURN(auto negative_val,
                      BuildMulShlo(exp_input_scaled_minus_1, alpha_scale));

  return mlir::stablehlo::Select(pred, /*on_true=*/positive_val,
                                 /*on_false=*/negative_val);
}

absl::StatusOr<mlir::MlirOp> BuildEluBackwardGradInputShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp self_or_result_op,
    mlir::MlirOp alpha_op, mlir::MlirOp scale_op, mlir::MlirOp input_scale_op,
    bool is_result) {
  TT_ASSIGN_OR_RETURN(
      (auto [grad_output_bcast, self_or_result_bcast]),
      ApplyBroadcastIfNeeded(grad_output_op, self_or_result_op));

  const auto elem_type = GetTensorTypeOrDie(grad_output_bcast).getElementType();
  auto zero_op =
      MakeScalarConstant(grad_output_bcast.getBuilder(), 0.0, elem_type);

  // Precompute scalar derivative coefficients:
  TT_ASSIGN_OR_RETURN(auto negcoef, BuildMulShlo(alpha_op, scale_op));
  auto poscoef = scale_op;
  auto negiptcoef = input_scale_op;
  TT_ASSIGN_OR_RETURN(auto neg_combined_coef,
                      BuildMulShlo(negiptcoef, negcoef));

  // --- Positive path: grad_output * poscoef ---
  TT_ASSIGN_OR_RETURN(auto positive_val,
                      BuildMulShlo(grad_output_bcast, poscoef));

  // --- Negative path: ---
  mlir::MlirOp negative_val;
  if (is_result) {
    // if is_result: grad_output * negiptcoef * (self_or_result + negcoef)
    TT_ASSIGN_OR_RETURN(auto term_sum,
                        BuildAddShlo(self_or_result_bcast, negcoef));
    TT_ASSIGN_OR_RETURN(auto term_mul,
                        BuildMulShlo(grad_output_bcast, negiptcoef));
    TT_ASSIGN_OR_RETURN(negative_val, BuildMulShlo(term_sum, term_mul));
  } else {
    // else: grad_output * (negiptcoef * negcoef) *
    // torch.exp(self_or_result * negiptcoef)
    TT_ASSIGN_OR_RETURN(auto inner,
                        BuildMulShlo(self_or_result_bcast, negiptcoef));
    auto exp_inner = mlir::stablehlo::Exp(inner);
    TT_ASSIGN_OR_RETURN(auto grad_exp,
                        BuildMulShlo(grad_output_bcast, exp_inner));
    TT_ASSIGN_OR_RETURN(negative_val,
                        BuildMulShlo(grad_exp, neg_combined_coef));
  }

  // Comparison condition: self_or_result <= 0
  TT_ASSIGN_OR_RETURN(auto pred, BuildLeShlo(self_or_result_bcast, zero_op));

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
