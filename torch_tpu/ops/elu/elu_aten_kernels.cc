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

#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/native/Resize.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
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
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {

enum class SelfOrResultMode {
  kSelf,
  kResult,
};

absl::StatusOr<mlir::MlirOp> BuildEluShlo(mlir::MlirOp input_op,
                                          mlir::MlirOp alpha_op,
                                          mlir::MlirOp scale_op,
                                          mlir::MlirOp input_scale_op) {
  // Dynamic parameters must be explicitly broadcasted to the input shape
  TT_ASSIGN_OR_RETURN(
      (auto [broadcasted_input_op, broadcasted_alpha_op, broadcasted_scale_op,
             broadcasted_input_scale_op]),
      ApplyBroadcastIfNeeded(input_op, alpha_op, scale_op, input_scale_op));

  mlir::MlirOp zero_op = MakeConstantLike(broadcasted_input_op, 0.0);
  mlir::MlirOp one_op = MakeConstantLike(broadcasted_input_op, 1.0);

  // Compare input against 0.0 to branch
  mlir::MlirOp pred = stablehlo::Compare(
      broadcasted_input_op, zero_op, mlir::stablehlo::ComparisonDirection::GT);

  // --- Positive path: scale * input (NO input_scale factor!) ---
  mlir::MlirOp positive_val =
      stablehlo::Mul(broadcasted_input_op, broadcasted_scale_op);

  // --- Negative path: scale * alpha * (exp(input * input_scale) - 1) ---
  mlir::MlirOp input_scaled =
      stablehlo::Mul(broadcasted_input_op, broadcasted_input_scale_op);
  mlir::MlirOp exp_input_scaled = stablehlo::Exp(input_scaled);
  mlir::MlirOp exp_input_scaled_minus_1 =
      stablehlo::Subtract(exp_input_scaled, one_op);
  mlir::MlirOp alpha_times_exp_minus_1 =
      stablehlo::Mul(broadcasted_alpha_op, exp_input_scaled_minus_1);
  mlir::MlirOp negative_val =
      stablehlo::Mul(alpha_times_exp_minus_1, broadcasted_scale_op);

  return stablehlo::Select(pred, /*on_true=*/positive_val,
                           /*on_false=*/negative_val);
}

absl::StatusOr<mlir::MlirOp> BuildEluBackwardGradInputShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp self_or_result_op,
    mlir::MlirOp alpha_op, mlir::MlirOp scale_op, mlir::MlirOp input_scale_op,
    SelfOrResultMode self_or_result_mode) {
  // Dynamic parameters must be explicitly broadcasted to the input shape
  TT_ASSIGN_OR_RETURN(
      (auto [broadcasted_grad_output_op, broadcasted_self_or_result_op,
             broadcasted_alpha_op, broadcasted_scale_op,
             broadcasted_input_scale_op]),
      ApplyBroadcastIfNeeded(grad_output_op, self_or_result_op, alpha_op,
                             scale_op, input_scale_op));

  mlir::MlirOp zero_op = MakeConstantLike(broadcasted_grad_output_op, 0.0);

  mlir::MlirOp neg_coef =
      stablehlo::Mul(broadcasted_alpha_op, broadcasted_scale_op);
  mlir::MlirOp pos_coef = broadcasted_scale_op;
  mlir::MlirOp neg_input_scale_coef = broadcasted_input_scale_op;

  // --- Positive path: grad_output * pos_coef ---
  mlir::MlirOp positive_val =
      stablehlo::Mul(broadcasted_grad_output_op, pos_coef);

  // --- Negative path: ---
  mlir::MlirOp negative_val;
  if (self_or_result_mode == SelfOrResultMode::kResult) {
    // if is_result: grad_output * neg_input_scale_coef *
    // (self_or_result + neg_coef)
    mlir::MlirOp term_sum =
        stablehlo::Add(broadcasted_self_or_result_op, neg_coef);
    mlir::MlirOp term_mul =
        stablehlo::Mul(broadcasted_grad_output_op, neg_input_scale_coef);
    negative_val = stablehlo::Mul(term_sum, term_mul);
  } else {
    // else: grad_output * neg_input_scale_coef * neg_coef *
    // torch.exp(self_or_result * neg_input_scale_coef)
    mlir::MlirOp inner =
        stablehlo::Mul(broadcasted_self_or_result_op, neg_input_scale_coef);
    mlir::MlirOp exp_inner = stablehlo::Exp(inner);
    mlir::MlirOp exp_mul = stablehlo::Mul(exp_inner, neg_coef);
    mlir::MlirOp term_exp_mul = stablehlo::Mul(exp_mul, neg_input_scale_coef);
    negative_val = stablehlo::Mul(broadcasted_grad_output_op, term_exp_mul);
  }

  // Comparison condition: self_or_result <= 0
  mlir::MlirOp pred =
      stablehlo::Compare(broadcasted_self_or_result_op, zero_op,
                         mlir::stablehlo::ComparisonDirection::LE);

  return stablehlo::Select(pred, /*on_true=*/negative_val,
                           /*on_false=*/positive_val);
}

absl::Status CheckIsFloatingPoint(const at::Tensor& tensor,
                                  std::string_view name) {
  TT_RET_CHECK(IsFloatingPoint(tensor), error::kInvalidArgument)
      << "expected the " << name << " dtype to be floating point, got "
      << ToString(tensor.scalar_type());
  return absl::OkStatus();
}

absl::StatusOr<DeviceBufferRef> EluForward(
    const at::Tensor& input, PromotedScalar& alpha, PromotedScalar& scale,
    PromotedScalar& input_scale, at::ScalarType out_dtype,
    OpParamCacheKeys op_param_cache_keys) {
  TT_RETURN_IF_ERROR(CheckIsFloatingPoint(input, "input"));

  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype_mlir,
                      ConvertTo<mlir::ElementType>(out_dtype));
  const Dimensions out_dims = CopyIntVector(input.sizes());

  // Single Tensor Allocation & Naming:
  TT_ASSIGN_OR_RETURN(at::Tensor alpha_tensor,
                      alpha.GetTensor(input.scalar_type()));
  TT_ASSIGN_OR_RETURN(at::Tensor scale_tensor,
                      scale.GetTensor(input.scalar_type()));
  TT_ASSIGN_OR_RETURN(at::Tensor input_scale_tensor,
                      input_scale.GetTensor(input.scalar_type()));

  auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 4> inputs_op)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [input_op, alpha_op, scale_op, input_scale_op] = inputs_op;
    return BuildEluShlo(input_op, alpha_op, scale_op, input_scale_op);
  };

  return DispatchOp<4>(std::move(op_builder),
                       {input, std::move(alpha_tensor), std::move(scale_tensor),
                        std::move(input_scale_tensor)},
                       {.out_dtype = out_dtype_mlir,
                        .out_dims = out_dims,
                        .op_param_cache_keys = std::move(op_param_cache_keys)});
}

absl::StatusOr<DeviceBufferRef> EluBackward(
    const at::Tensor& grad_output, const at::Tensor& self_or_result,
    PromotedScalar& alpha, PromotedScalar& scale, PromotedScalar& input_scale,
    SelfOrResultMode self_or_result_mode, at::ScalarType out_dtype,
    OpParamCacheKeys op_param_cache_keys) {
  TT_RETURN_IF_ERROR(CheckIsFloatingPoint(grad_output, "grad output"));

  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype_mlir,
                      ConvertTo<mlir::ElementType>(out_dtype));
  const Dimensions out_dims = CopyIntVector(grad_output.sizes());

  // Single Tensor Allocation & Naming:
  TT_ASSIGN_OR_RETURN(at::Tensor alpha_tensor,
                      alpha.GetTensor(grad_output.scalar_type()));
  TT_ASSIGN_OR_RETURN(at::Tensor scale_tensor,
                      scale.GetTensor(grad_output.scalar_type()));
  TT_ASSIGN_OR_RETURN(at::Tensor input_scale_tensor,
                      input_scale.GetTensor(grad_output.scalar_type()));

  auto op_builder =
      [self_or_result_mode](FixedSizeSpan<mlir::MlirOp, 5> inputs_op)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [grad_output_op, self_or_result_op, alpha_op, scale_op,
           input_scale_op] = inputs_op;
    return BuildEluBackwardGradInputShlo(grad_output_op, self_or_result_op,
                                         alpha_op, scale_op, input_scale_op,
                                         self_or_result_mode);
  };

  return DispatchOp<5>(std::move(op_builder),
                       {grad_output, self_or_result, std::move(alpha_tensor),
                        std::move(scale_tensor), std::move(input_scale_tensor)},
                       {.out_dtype = out_dtype_mlir,
                        .out_dims = out_dims,
                        .op_param_cache_keys = std::move(op_param_cache_keys)});
}

}  // namespace

at::Tensor& AtenEluOut(const at::Tensor& input, const at::Scalar& alpha,
                       const at::Scalar& scale, const at::Scalar& input_scale,
                       at::Tensor& out) {
  PromotedScalar promoted_alpha = PromoteScalar(alpha);
  PromotedScalar promoted_scale = PromoteScalar(scale);
  PromotedScalar promoted_input_scale = PromoteScalar(input_scale);
  TT_KERNEL(
      OpName::kEluOut, param_keys,
      (input, promoted_alpha, promoted_scale, promoted_input_scale, out), {
        TT_ASSIGN_OR_THROW(DeviceBufferRef result_buffer,
                           EluForward(input, promoted_alpha, promoted_scale,
                                      promoted_input_scale, out.scalar_type(),
                                      std::move(param_keys)));
        at::native::resize_output(out, input.sizes());
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result_buffer), out));
        return out;
      });
}

at::Tensor& AtenEluBackwardGradInput(
    const at::Tensor& grad_output, const at::Scalar& alpha,
    const at::Scalar& scale, const at::Scalar& input_scale, bool is_result,
    const at::Tensor& self_or_result, at::Tensor& grad_input) {
  PromotedScalar promoted_alpha = PromoteScalar(alpha);
  PromotedScalar promoted_scale = PromoteScalar(scale);
  PromotedScalar promoted_input_scale = PromoteScalar(input_scale);
  TT_KERNEL(OpName::kEluBackwardGradInput, param_keys,
            (grad_output, promoted_alpha, promoted_scale, promoted_input_scale,
             is_result, self_or_result, grad_input),
            {
              SelfOrResultMode self_or_result_mode =
                  is_result ? SelfOrResultMode::kResult
                            : SelfOrResultMode::kSelf;

              TT_ASSIGN_OR_THROW(
                  DeviceBufferRef result_buffer,
                  EluBackward(grad_output, self_or_result, promoted_alpha,
                              promoted_scale, promoted_input_scale,
                              self_or_result_mode, grad_input.scalar_type(),
                              std::move(param_keys)));

              TT_THROW_IF_ERROR(
                  AssignBufferToAtTensor(std::move(result_buffer), grad_input));
              return grad_input;
            });
}

}  // namespace torch_tpu
