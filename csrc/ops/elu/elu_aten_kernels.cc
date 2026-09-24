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
namespace internal {
template <>
inline void CheckScalarInput<OpName::kEluOut, UsesScalarInput::kYes>() {}
template <>
inline void
CheckScalarInput<OpName::kEluBackwardGradInput, UsesScalarInput::kYes>() {}
}  // namespace internal

namespace {

absl::StatusOr<mlir::MlirOp> BuildEluShlo(mlir::MlirOp input_op,
                                          const at::Scalar& alpha,
                                          const at::Scalar& scale,
                                          const at::Scalar& input_scale) {
  auto type = GetTensorTypeOrDie(input_op);
  TT_ASSIGN_OR_RETURN(auto element_type,
                      ConvertTo<mlir::ElementType>(type.getElementType()));

  auto& builder = input_op.getBuilder();
  mlir::MlirOp alpha_op =
      MakeScalarConstant(builder, alpha.toDouble(), element_type);
  mlir::MlirOp scale_op =
      MakeScalarConstant(builder, scale.toDouble(), element_type);
  mlir::MlirOp input_scale_op =
      MakeScalarConstant(builder, input_scale.toDouble(), element_type);
  mlir::MlirOp zero_op = MakeScalarConstant(builder, 0.0, element_type);
  mlir::MlirOp one = MakeScalarConstant(builder, 1.0, element_type);

  TT_ASSIGN_OR_RETURN((auto [input_bcast, alpha_bcast, scale_bcast,
                             input_scale_bcast, zero_bcast, one_bcast]),
                      ApplyBroadcastIfNeeded(input_op, alpha_op, scale_op,
                                             input_scale_op, zero_op, one));

  //  y = self * input_scale
  mlir::MlirOp input_scaled =
      mlir::stablehlo::Mul(input_bcast, input_scale_bcast);
  mlir::MlirOp pred = mlir::stablehlo::Compare(
      input_scaled, zero_bcast, mlir::stablehlo::ComparisonDirection::GT);

  // --- Positive path: scale * input_scaled ---
  mlir::MlirOp positive_val = mlir::stablehlo::Mul(input_scaled, scale_bcast);

  // --- Negative path: scale * alpha * (exp(input_scaled) - 1) ---
  mlir::MlirOp exp_input_scaled = mlir::stablehlo::Exp(input_scaled);
  mlir::MlirOp exp_input_scaled_minus_1 =
      mlir::stablehlo::Subtract(exp_input_scaled, one_bcast);
  mlir::MlirOp exp_alpha_scaled =
      mlir::stablehlo::Mul(alpha_bcast, exp_input_scaled_minus_1);
  mlir::MlirOp negative_val =
      mlir::stablehlo::Mul(exp_alpha_scaled, scale_bcast);

  return mlir::stablehlo::Select(pred, /*on_true=*/positive_val,
                                 /*on_false=*/negative_val);
}

absl::StatusOr<mlir::MlirOp> BuildEluBackwardGradInputShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp self_or_result_op,
    const at::Scalar& alpha, const at::Scalar& scale,
    const at::Scalar& input_scale, bool is_result) {
  auto type = GetTensorTypeOrDie(grad_output_op);
  TT_ASSIGN_OR_RETURN(auto element_type,
                      ConvertTo<mlir::ElementType>(type.getElementType()));

  auto& builder = grad_output_op.getBuilder();
  mlir::MlirOp alpha_op =
      MakeScalarConstant(builder, alpha.toDouble(), element_type);
  mlir::MlirOp scale_op =
      MakeScalarConstant(builder, scale.toDouble(), element_type);
  mlir::MlirOp input_scale_op =
      MakeScalarConstant(builder, input_scale.toDouble(), element_type);
  mlir::MlirOp zero_op = MakeScalarConstant(builder, 0.0, element_type);

  TT_ASSIGN_OR_RETURN(
      (auto [grad_output_bcast, self_or_result_bcast, alpha_bcast, scale_bcast,
             input_scale_bcast, zero_bcast]),
      ApplyBroadcastIfNeeded(grad_output_op, self_or_result_op, alpha_op,
                             scale_op, input_scale_op, zero_op));

  mlir::MlirOp negcoef = mlir::stablehlo::Mul(alpha_bcast, scale_bcast);
  mlir::MlirOp poscoef = scale_bcast;
  mlir::MlirOp negiptcoef = input_scale_bcast;

  // --- Positive path: grad_output * poscoef ---
  mlir::MlirOp positive_val = mlir::stablehlo::Mul(grad_output_bcast, poscoef);

  // --- Negative path: ---
  mlir::MlirOp negative_val;
  if (is_result) {
    // if is_result: grad_output * negiptcoef * (self_or_result + negcoef)
    mlir::MlirOp term_sum = mlir::stablehlo::Add(self_or_result_bcast, negcoef);
    mlir::MlirOp term_mul = mlir::stablehlo::Mul(grad_output_bcast, negiptcoef);
    negative_val = mlir::stablehlo::Mul(term_sum, term_mul);
  } else {
    // else: grad_output * negiptcoef * negcoef *
    // torch.exp(self_or_result * negiptcoef)
    mlir::MlirOp inner = mlir::stablehlo::Mul(self_or_result_bcast, negiptcoef);
    mlir::MlirOp exp_inner = mlir::stablehlo::Exp(inner);
    mlir::MlirOp exp_mul = mlir::stablehlo::Mul(exp_inner, negcoef);
    mlir::MlirOp term_exp_mul = mlir::stablehlo::Mul(exp_mul, negiptcoef);
    negative_val = mlir::stablehlo::Mul(grad_output_bcast, term_exp_mul);
  }

  // Comparison condition: self_or_result <= 0
  mlir::MlirOp pred =
      mlir::stablehlo::Compare(self_or_result_bcast, zero_bcast,
                               mlir::stablehlo::ComparisonDirection::LE);

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
  TT_KERNEL(
      OpName::kEluOut, param_keys, (input, alpha, scale, input_scale, out), {
        TT_THROW_IF_ERROR(ValidateIsFloatingPoint(input, /* name= */ "input"));

        TT_ASSIGN_OR_THROW(const mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));

        auto op_builder =
            [alpha, scale, input_scale](
                mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
          return BuildEluShlo(input_op, alpha, scale, input_scale);
        };

        TT_THROW_IF_ERROR(
            DispatchOpOut<1>(std::move(op_builder), {input}, out,
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
  TT_KERNEL(
      OpName::kEluBackwardGradInput, param_keys,
      (grad_output, alpha, scale, input_scale, is_result, self_or_result,
       grad_input),
      {
        TT_THROW_IF_ERROR(
            ValidateIsFloatingPoint(grad_output, /* name= */ "grad output"));

        auto op_builder = [alpha, scale, input_scale,
                           is_result](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto [grad_output_op, self_or_result_op] = inputs;
          return BuildEluBackwardGradInputShlo(grad_output_op,
                                               self_or_result_op, alpha, scale,
                                               input_scale, is_result);
        };

        TT_ASSIGN_OR_THROW(
            const auto output_dtype,
            ConvertTo<mlir::ElementType>(grad_input.scalar_type()));
        auto output_shape = CopyIntVector(grad_input.sizes());

        TT_THROW_IF_ERROR(DispatchOpOut<2>(
            std::move(op_builder), {grad_output, self_or_result}, grad_input,
            /*options=*/
            {.out_dtype = output_dtype,
             .out_dims = output_shape,
             .op_param_cache_keys = std::move(param_keys)}));
        return grad_input;
      });
}
}  // namespace torch_tpu
