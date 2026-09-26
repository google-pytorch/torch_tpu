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

#include "csrc/ops/native_batch_norm/native_batch_norm.h"

#include <array>
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/reductions/sum.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

// Computes the batch-norm inverse standard deviation: 1 / sqrt(variance + eps).
// `variance` must already be in the desired accumulation precision.
[[nodiscard]] mlir::MlirOp ComputeInvStd(mlir::MlirOp variance, double eps) {
  auto eps_op = MakeConstantLike(variance, eps);
  // Add eps to variance to avoid division by zero, similar to LayerNorm.
  auto var_plus_eps = mlir::stablehlo::Add(variance, eps_op);
  return mlir::stablehlo::Rsqrt(var_plus_eps);
}

}  // namespace

absl::StatusOr<MlirOpResults<3>> BuildBatchNormBackward(
    mlir::MlirOp grad_out, mlir::MlirOp input,
    std::optional<mlir::MlirOp> weight_opt,
    std::optional<mlir::MlirOp> running_mean_opt,
    std::optional<mlir::MlirOp> running_variance_opt,
    std::optional<mlir::MlirOp> save_mean_opt,
    std::optional<mlir::MlirOp> save_invstd_opt, bool training, double eps,
    std::array<bool, 3> output_mask, mlir::ElementType acc_dtype) {
  ABSL_VLOG(1) << "BuildBatchNormBackward "
               << " input=" << input.ToString() << " training=" << training
               << " eps=" << eps;
  mlir::MlirBuilder& builder = input.getBuilder();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const mlir::ElementType input_dtype = GetElementTypeOrDie(input);
  auto input_dims = input_type.getShape();
  int64_t num_features = input_dims[kTorchFeaturesDimensionIndex];

  mlir::MlirOp weight =
      weight_opt ? *weight_opt
                 : MakeConstant(builder, 1.0, input_dtype, {num_features});

  if (training) {
    // Promotion to accumulation precision.
    input = mlir::stablehlo::ConvertElementType(input, acc_dtype);
    grad_out = mlir::stablehlo::ConvertElementType(grad_out, acc_dtype);
    weight = mlir::stablehlo::ConvertElementType(weight, acc_dtype);

    // Compute mean and variance.
    if (!save_mean_opt || !save_invstd_opt) {
      return TT_ERROR(error::kInternal)
             << "save_mean and save_invstd must be provided when training=True";
    }

    auto mean = *save_mean_opt;
    mean = mlir::stablehlo::ConvertElementType(mean, acc_dtype);

    auto invstd = *save_invstd_opt;
    invstd = mlir::stablehlo::ConvertElementType(invstd, acc_dtype);

    // Used to reduce over all dims except the feature dim.
    Dimensions reduce_dims;
    int64_t m_val = 1;
    for (auto i = 0; i < input_dims.size(); ++i) {
      if (i != kTorchFeaturesDimensionIndex) {
        reduce_dims.push_back(i);
        m_val *= input_dims[i];
      }
    }

    bool need_sum_dy = output_mask[2] || output_mask[0];
    bool need_sum_dy_xhat = output_mask[1] || output_mask[0];

    std::optional<mlir::MlirOp> sum_dy;
    std::optional<mlir::MlirOp> sum_dy_xhat;
    std::optional<mlir::MlirOp> x_hat;

    if (need_sum_dy_xhat) {
      TT_ASSIGN_OR_RETURN(auto broadcasted_mean,
                          Broadcast(mean, input_type.getShape(),
                                    {kTorchFeaturesDimensionIndex}));
      auto x_minus_mean = mlir::stablehlo::Subtract(input, broadcasted_mean);

      TT_ASSIGN_OR_RETURN(auto broadcasted_invstd,
                          Broadcast(invstd, input_type.getShape(),
                                    {kTorchFeaturesDimensionIndex}));
      x_hat = mlir::stablehlo::Mul(x_minus_mean, broadcasted_invstd);

      auto grad_out_times_x_hat = mlir::stablehlo::Mul(grad_out, *x_hat);
      TT_ASSIGN_OR_RETURN(sum_dy_xhat,
                          BuildSumShlo(grad_out_times_x_hat, reduce_dims));
    }

    if (need_sum_dy) {
      TT_ASSIGN_OR_RETURN(sum_dy, BuildSumShlo(grad_out, reduce_dims));
    }

    mlir::MlirOp grad_input, grad_weight, grad_bias;

    if (output_mask[0]) {
      auto m_op = MakeConstantLike(invstd, static_cast<double>(m_val));

      auto mean_dy = mlir::stablehlo::Div(*sum_dy, m_op);
      TT_ASSIGN_OR_RETURN(auto broadcasted_mean_dy,
                          Broadcast(mean_dy, input_type.getShape(),
                                    {kTorchFeaturesDimensionIndex}));
      auto diff1 = mlir::stablehlo::Subtract(grad_out, broadcasted_mean_dy);

      auto mean_dy_xhat = mlir::stablehlo::Div(*sum_dy_xhat, m_op);
      TT_ASSIGN_OR_RETURN(auto broadcasted_mean_dy_xhat,
                          Broadcast(mean_dy_xhat, input_type.getShape(),
                                    {kTorchFeaturesDimensionIndex}));
      auto term3 = mlir::stablehlo::Mul(*x_hat, broadcasted_mean_dy_xhat);

      auto inner = mlir::stablehlo::Subtract(diff1, term3);

      auto multiplier = mlir::stablehlo::Mul(weight, invstd);
      TT_ASSIGN_OR_RETURN(auto broadcasted_multiplier,
                          Broadcast(multiplier, input_type.getShape(),
                                    {kTorchFeaturesDimensionIndex}));

      grad_input = mlir::stablehlo::Mul(broadcasted_multiplier, inner);
    } else {
      TT_ASSIGN_OR_RETURN(grad_input, MakeZeroSizedTensor(builder, acc_dtype));
    }

    if (output_mask[1]) {
      grad_weight = *sum_dy_xhat;
    } else {
      TT_ASSIGN_OR_RETURN(grad_weight, MakeZeroSizedTensor(builder, acc_dtype));
    }

    if (output_mask[2]) {
      grad_bias = *sum_dy;
    } else {
      TT_ASSIGN_OR_RETURN(grad_bias, MakeZeroSizedTensor(builder, acc_dtype));
    }

    // Cast results back to input precision if needed.
    grad_input = mlir::stablehlo::ConvertElementType(grad_input, input_dtype);
    grad_weight = mlir::stablehlo::ConvertElementType(grad_weight, input_dtype);
    grad_bias = mlir::stablehlo::ConvertElementType(grad_bias, input_dtype);

    return {{grad_input, grad_weight, grad_bias}};

  } else {
    // Inference mode.

    // Inference backward assumes that mean and variance are fixed constants
    // (running statistics) that do not depend on the current input batch. This
    // changes the gradient computation w.r.t training since dmean/dx and
    // dvariance/dx are 0.
    //
    // y = (x - mean) / sqrt(var + eps) * weight + bias
    // grad_out = dL/dy
    // dL/dx = grad_out * weight / sqrt(var + eps)
    // dL/dweight = grad_out * (x - mean) / sqrt(var + eps)
    // dL/dbias = grad_out

    // Promotion to accumulation precision.
    input = mlir::stablehlo::ConvertElementType(input, acc_dtype);
    grad_out = mlir::stablehlo::ConvertElementType(grad_out, acc_dtype);
    weight = mlir::stablehlo::ConvertElementType(weight, acc_dtype);

    mlir::MlirOp grad_input, grad_weight, grad_bias;
    std::optional<mlir::MlirOp> inv_std_opt;

    if (output_mask[0] || output_mask[1]) {
      mlir::MlirOp variance =
          running_variance_opt
              ? *running_variance_opt
              : MakeConstant(builder, 1.0, input_dtype, {num_features});
      variance = mlir::stablehlo::ConvertElementType(variance, acc_dtype);
      inv_std_opt = ComputeInvStd(variance, eps);
    }

    if (output_mask[0]) {
      // inv_std = 1 / sqrt(var + eps)
      // grad_input = dL/dx = grad_out * (weight * inv_std)
      auto w_times_inv_std = mlir::stablehlo::Mul(weight, *inv_std_opt);
      // Broadcast w_times_inv_std to grad_out shape
      TT_ASSIGN_OR_RETURN(auto broadcasted_w,
                          Broadcast(w_times_inv_std, input_type.getShape(),
                                    {kTorchFeaturesDimensionIndex}));
      grad_input = mlir::stablehlo::Mul(grad_out, broadcasted_w);
    } else {
      TT_ASSIGN_OR_RETURN(grad_input, MakeZeroSizedTensor(builder, acc_dtype));
    }

    // Used to reduce over all dims except the feature dim.
    Dimensions reduce_dims;
    for (auto i = 0; i < input_dims.size(); ++i) {
      if (i != kTorchFeaturesDimensionIndex) {
        reduce_dims.push_back(i);
      }
    }

    if (output_mask[1]) {
      mlir::MlirOp mean =
          running_mean_opt
              ? *running_mean_opt
              : MakeConstant(builder, 0.0, input_dtype, {num_features});
      mean = mlir::stablehlo::ConvertElementType(mean, acc_dtype);

      // grad_weight = grad_out * (x - mean) * inv_std
      TT_ASSIGN_OR_RETURN(auto broadcasted_mean,
                          Broadcast(mean, input_type.getShape(),
                                    {kTorchFeaturesDimensionIndex}));
      auto x_minus_mean = mlir::stablehlo::Subtract(input, broadcasted_mean);
      TT_ASSIGN_OR_RETURN(auto broadcasted_inv_std,
                          Broadcast(*inv_std_opt, input_type.getShape(),
                                    {kTorchFeaturesDimensionIndex}));
      auto norm = mlir::stablehlo::Mul(x_minus_mean, broadcasted_inv_std);
      auto grad_out_times_norm = mlir::stablehlo::Mul(grad_out, norm);

      // Reduce over all dims except the feature dim.
      TT_ASSIGN_OR_RETURN(grad_weight,
                          BuildSumShlo(grad_out_times_norm, reduce_dims));
    } else {
      TT_ASSIGN_OR_RETURN(grad_weight, MakeZeroSizedTensor(builder, acc_dtype));
    }

    if (output_mask[2]) {
      // grad_bias = grad_out
      TT_ASSIGN_OR_RETURN(grad_bias, BuildSumShlo(grad_out, reduce_dims));
    } else {
      TT_ASSIGN_OR_RETURN(grad_bias, MakeZeroSizedTensor(builder, acc_dtype));
    }

    // Cast results back to input precision.
    grad_input = mlir::stablehlo::ConvertElementType(grad_input, input_dtype);
    grad_weight = mlir::stablehlo::ConvertElementType(grad_weight, input_dtype);
    grad_bias = mlir::stablehlo::ConvertElementType(grad_bias, input_dtype);

    return {{grad_input, grad_weight, grad_bias}};
  }
}

absl::StatusOr<MlirOpResults<3>> BuildBatchNorm(
    mlir::MlirOp input, std::optional<mlir::MlirOp> weight_opt,
    std::optional<mlir::MlirOp> bias_opt,
    std::optional<mlir::MlirOp> running_mean_opt,
    std::optional<mlir::MlirOp> running_variance_opt, bool training,
    double momentum, double eps, mlir::ElementType acc_dtype) {
  // TODO: Parameter momentum is not used in the logic below and still
  // tests/ops_tests pass. We should figure out if/when such parameter is
  // needed. See
  // https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d.momentum
  // for more details.

  ABSL_VLOG(1) << "BuildBatchNorm "
               << " input=" << input.ToString() << " training=" << training
               << " momentum=" << momentum << " eps=" << eps;
  mlir::MlirBuilder& builder = input.getBuilder();

  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  auto input_dims = input_type.getShape();
  const mlir::ElementType input_dtype = GetElementTypeOrDie(input);
  int64_t num_features = input_dims[kTorchFeaturesDimensionIndex];
  mlir::MlirOp weight =
      weight_opt ? *weight_opt
                 : MakeConstant(builder, 1.0, input_dtype, {num_features});
  mlir::MlirOp bias =
      bias_opt ? *bias_opt
               : MakeConstant(builder, 0.0, input_dtype, {num_features});

  if (training) {
    // Promotion to accumulation precision.
    input = mlir::stablehlo::ConvertElementType(input, acc_dtype);
    weight = mlir::stablehlo::ConvertElementType(weight, acc_dtype);
    bias = mlir::stablehlo::ConvertElementType(bias, acc_dtype);

    auto outputs = mlir::stablehlo::BatchNormTraining(
        input, weight, bias, llvm::APFloat(static_cast<float>(eps)),
        kTorchFeaturesDimensionIndex);
    auto output_tensor = std::move(outputs[0]);

    // Output tensor needs to be in input precision.
    output_tensor =
        mlir::stablehlo::ConvertElementType(output_tensor, input_dtype);

    auto output_mean = std::move(outputs[1]);
    auto& output_variance = outputs[2];

    // Stats outputs need to be in acc_dtype (F32) for PyTorch compatibility.
    output_mean = mlir::stablehlo::ConvertElementType(output_mean, acc_dtype);
    output_variance =
        mlir::stablehlo::ConvertElementType(output_variance, acc_dtype);
    auto output_inverted_variance = ComputeInvStd(output_variance, eps);
    return {{output_tensor, output_mean, output_inverted_variance}};

  } else {
    // Inference mode.
    mlir::MlirOp running_mean =
        running_mean_opt
            ? *running_mean_opt
            : MakeConstant(builder, 0.0, input_dtype, {num_features});
    mlir::MlirOp running_variance =
        running_variance_opt
            ? *running_variance_opt
            : MakeConstant(builder, 0.0, input_dtype, {num_features});

    // Promotion to accumulation precision.
    input = mlir::stablehlo::ConvertElementType(input, acc_dtype);
    weight = mlir::stablehlo::ConvertElementType(weight, acc_dtype);
    bias = mlir::stablehlo::ConvertElementType(bias, acc_dtype);
    running_mean = mlir::stablehlo::ConvertElementType(running_mean, acc_dtype);
    running_variance =
        mlir::stablehlo::ConvertElementType(running_variance, acc_dtype);

    auto output_tensor = mlir::stablehlo::BatchNormInference(
        input, weight, bias, running_mean, running_variance,
        llvm::APFloat(static_cast<float>(eps)), kTorchFeaturesDimensionIndex);

    // Output tensor needs to be in input precision.
    output_tensor =
        mlir::stablehlo::ConvertElementType(output_tensor, input_dtype);

    // To match CUDA semantics, the inference mode holds the running stats in
    // accumulation precision. Instead, the CPU impl returns empty tensors.
    auto output_mean = running_mean;
    auto output_inverted_variance = ComputeInvStd(running_variance, eps);
    return {{output_tensor, output_mean, output_inverted_variance}};
  }
}

}  // namespace torch_tpu
