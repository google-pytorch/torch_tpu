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
#include <initializer_list>
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
#include "mlir/IR/Types.h"
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

bool HasZeroReductionDim(mlir::RankedTensorType input_type,
                         const Dimensions& reduce_dims) {
  for (int64_t d : reduce_dims) {
    if (input_type.getDimSize(d) == 0) {
      return true;
    }
  }
  return false;
}

mlir::ElementType GetParamElementType(
    mlir::ElementType input_dtype,
    std::initializer_list<std::optional<mlir::MlirOp>> params) {
  for (const std::optional<mlir::MlirOp>& param : params) {
    if (param.has_value()) {
      return GetElementTypeOrDie(*param);
    }
  }
  return input_dtype;
}

absl::StatusOr<MlirOpResults<3>> BuildZeroBatchTrainingGradOutputs(
    mlir::MlirOp input, mlir::MlirOp save_invstd, mlir::ElementType input_dtype,
    mlir::ElementType weight_dtype, std::array<bool, 3> output_mask) {
  mlir::MlirBuilder& builder = input.getBuilder();
  mlir::MlirOp grad_input;
  if (output_mask[0]) {
    grad_input = input;
  } else {
    TT_ASSIGN_OR_RETURN(grad_input, MakeZeroSizedTensor(builder, input_dtype));
  }

  mlir::MlirOp grad_weight;
  if (output_mask[1]) {
    grad_weight = MakeConstantLike(save_invstd, 0.0, weight_dtype);
  } else {
    TT_ASSIGN_OR_RETURN(grad_weight,
                        MakeZeroSizedTensor(builder, weight_dtype));
  }

  mlir::MlirOp grad_bias;
  if (output_mask[2]) {
    grad_bias = MakeConstantLike(save_invstd, 0.0, weight_dtype);
  } else {
    TT_ASSIGN_OR_RETURN(grad_bias, MakeZeroSizedTensor(builder, weight_dtype));
  }

  return {{grad_input, grad_weight, grad_bias}};
}

absl::StatusOr<mlir::MlirOp> ComputeReductionNormFactor(
    mlir::MlirOp input, mlir::MlirOp invstd, const Dimensions& reduce_dims) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  if (input_type.hasStaticShape()) {
    int64_t num_reduction_elements = 1;
    for (int64_t d : reduce_dims) {
      num_reduction_elements *= input_type.getDimSize(d);
    }
    return MakeConstantLike(invstd,
                            1.0 / static_cast<double>(num_reduction_elements));
  }

  mlir::MlirBuilder& builder = input.getBuilder();
  const mlir::Type invstd_elem_type =
      GetTensorTypeOrDie(invstd).getElementType();
  mlir::MlirOp num_elements =
      GetNumElements(input, invstd_elem_type, reduce_dims);
  mlir::MlirOp one = MakeScalarConstant(builder, 1.0, invstd_elem_type);
  mlir::MlirOp inv_num_elements = mlir::stablehlo::Div(one, num_elements);
  return BroadcastIfNeeded(inv_num_elements, invstd);
}

absl::StatusOr<mlir::MlirOp> ComputeTrainingGradInput(
    mlir::MlirOp grad_out, mlir::MlirOp input, mlir::MlirOp x_minus_mean,
    mlir::MlirOp weight, mlir::MlirOp invstd, mlir::MlirOp grad_output_sum,
    mlir::MlirOp dot_p, const Dimensions& reduce_dims,
    mlir::ElementType input_dtype) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp norm_const,
                      ComputeReductionNormFactor(input, invstd, reduce_dims));

  mlir::MlirOp grad_mean_1d = mlir::stablehlo::Mul(grad_output_sum, norm_const);
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp grad_mean,
      BroadcastIfNeeded(grad_mean_1d, input, {kTorchFeaturesDimensionIndex}));

  mlir::MlirOp invstd_sq = mlir::stablehlo::Mul(invstd, invstd);
  mlir::MlirOp dot_p_times_norm = mlir::stablehlo::Mul(dot_p, norm_const);
  mlir::MlirOp proj_scale_1d =
      mlir::stablehlo::Mul(dot_p_times_norm, invstd_sq);
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp proj_scale,
      BroadcastIfNeeded(proj_scale_1d, input, {kTorchFeaturesDimensionIndex}));

  mlir::MlirOp proj = mlir::stablehlo::Mul(x_minus_mean, proj_scale);

  mlir::MlirOp w_times_invstd = mlir::stablehlo::Mul(weight, invstd);
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp grad_scale,
      BroadcastIfNeeded(w_times_invstd, input, {kTorchFeaturesDimensionIndex}));

  mlir::MlirOp grad_out_minus_proj = mlir::stablehlo::Subtract(grad_out, proj);
  mlir::MlirOp diff = mlir::stablehlo::Subtract(grad_out_minus_proj, grad_mean);
  mlir::MlirOp grad_input = mlir::stablehlo::Mul(diff, grad_scale);
  return mlir::stablehlo::ConvertElementType(grad_input, input_dtype);
}

absl::StatusOr<MlirOpResults<3>> BuildBatchNormBackwardTraining(
    mlir::MlirOp grad_out, mlir::MlirOp input,
    std::optional<mlir::MlirOp> weight_opt,
    std::optional<mlir::MlirOp> running_mean_opt,
    std::optional<mlir::MlirOp> running_variance_opt,
    std::optional<mlir::MlirOp> save_mean_opt,
    std::optional<mlir::MlirOp> save_invstd_opt,
    std::array<bool, 3> output_mask, mlir::ElementType acc_dtype) {
  if (!save_mean_opt || !save_invstd_opt) {
    return TT_ERROR(error::kInternal)
           << "save_mean and save_invstd must be provided when training=True";
  }

  mlir::MlirBuilder& builder = input.getBuilder();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const mlir::ElementType input_dtype = GetElementTypeOrDie(input);
  const mlir::ElementType weight_dtype = GetParamElementType(
      input_dtype, {weight_opt, running_mean_opt, running_variance_opt,
                    save_mean_opt, save_invstd_opt});
  const Dimensions reduce_dims =
      GetBatchDimensions(input_type.getRank(), kTorchFeaturesDimensionIndex);

  if (HasZeroReductionDim(input_type, reduce_dims)) {
    return BuildZeroBatchTrainingGradOutputs(
        input, *save_invstd_opt, input_dtype, weight_dtype, output_mask);
  }

  input = mlir::stablehlo::ConvertElementType(input, acc_dtype);
  grad_out = mlir::stablehlo::ConvertElementType(grad_out, acc_dtype);
  mlir::MlirOp mean =
      mlir::stablehlo::ConvertElementType(*save_mean_opt, acc_dtype);
  mlir::MlirOp invstd =
      mlir::stablehlo::ConvertElementType(*save_invstd_opt, acc_dtype);
  mlir::MlirOp weight =
      weight_opt ? mlir::stablehlo::ConvertElementType(*weight_opt, acc_dtype)
                 : MakeConstantLike(invstd, 1.0);

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp broadcasted_mean,
      BroadcastIfNeeded(mean, input, {kTorchFeaturesDimensionIndex}));
  mlir::MlirOp x_minus_mean =
      mlir::stablehlo::Subtract(input, broadcasted_mean);

  mlir::MlirOp grad_output_sum;
  if (output_mask[0] || output_mask[2]) {
    TT_ASSIGN_OR_RETURN(grad_output_sum, BuildSumShlo(grad_out, reduce_dims));
  }

  mlir::MlirOp dot_p;
  if (output_mask[0] || output_mask[1]) {
    mlir::MlirOp grad_out_times_x_minus_mean =
        mlir::stablehlo::Mul(grad_out, x_minus_mean);
    TT_ASSIGN_OR_RETURN(dot_p,
                        BuildSumShlo(grad_out_times_x_minus_mean, reduce_dims));
  }

  mlir::MlirOp grad_input;
  if (output_mask[0]) {
    TT_ASSIGN_OR_RETURN(
        grad_input, ComputeTrainingGradInput(grad_out, input, x_minus_mean,
                                             weight, invstd, grad_output_sum,
                                             dot_p, reduce_dims, input_dtype));
  } else {
    TT_ASSIGN_OR_RETURN(grad_input, MakeZeroSizedTensor(builder, input_dtype));
  }

  mlir::MlirOp grad_weight;
  if (output_mask[1]) {
    grad_weight = mlir::stablehlo::Mul(dot_p, invstd);
    grad_weight =
        mlir::stablehlo::ConvertElementType(grad_weight, weight_dtype);
  } else {
    TT_ASSIGN_OR_RETURN(grad_weight,
                        MakeZeroSizedTensor(builder, weight_dtype));
  }

  mlir::MlirOp grad_bias;
  if (output_mask[2]) {
    grad_bias =
        mlir::stablehlo::ConvertElementType(grad_output_sum, weight_dtype);
  } else {
    TT_ASSIGN_OR_RETURN(grad_bias, MakeZeroSizedTensor(builder, weight_dtype));
  }

  return {{grad_input, grad_weight, grad_bias}};
}

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
absl::StatusOr<MlirOpResults<3>> BuildBatchNormBackwardInference(
    mlir::MlirOp grad_out, mlir::MlirOp input,
    std::optional<mlir::MlirOp> weight_opt,
    std::optional<mlir::MlirOp> running_mean_opt,
    std::optional<mlir::MlirOp> running_variance_opt, double eps,
    std::array<bool, 3> output_mask, mlir::ElementType acc_dtype) {
  mlir::MlirBuilder& builder = input.getBuilder();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const mlir::ElementType input_dtype = GetElementTypeOrDie(input);
  const mlir::ElementType weight_dtype = GetParamElementType(
      input_dtype, {weight_opt, running_mean_opt, running_variance_opt});
  const int64_t num_features =
      input_type.getDimSize(kTorchFeaturesDimensionIndex);
  const Dimensions reduce_dims =
      GetBatchDimensions(input_type.getRank(), kTorchFeaturesDimensionIndex);

  input = mlir::stablehlo::ConvertElementType(input, acc_dtype);
  grad_out = mlir::stablehlo::ConvertElementType(grad_out, acc_dtype);

  mlir::MlirOp weight =
      weight_opt ? mlir::stablehlo::ConvertElementType(*weight_opt, acc_dtype)
                 : MakeConstant(builder, 1.0, acc_dtype, {num_features});
  mlir::MlirOp mean =
      running_mean_opt
          ? mlir::stablehlo::ConvertElementType(*running_mean_opt, acc_dtype)
          : MakeConstant(builder, 0.0, acc_dtype, {num_features});
  mlir::MlirOp variance =
      running_variance_opt
          ? mlir::stablehlo::ConvertElementType(*running_variance_opt,
                                                acc_dtype)
          : MakeConstant(builder, 1.0, acc_dtype, {num_features});

  mlir::MlirOp inv_std = ComputeInvStd(variance, eps);

  mlir::MlirOp grad_input;
  if (output_mask[0]) {
    mlir::MlirOp w_times_inv_std = mlir::stablehlo::Mul(weight, inv_std);
    TT_ASSIGN_OR_RETURN(mlir::MlirOp broadcasted_w,
                        BroadcastIfNeeded(w_times_inv_std, input,
                                          {kTorchFeaturesDimensionIndex}));
    grad_input = mlir::stablehlo::Mul(grad_out, broadcasted_w);
    grad_input = mlir::stablehlo::ConvertElementType(grad_input, input_dtype);
  } else {
    TT_ASSIGN_OR_RETURN(grad_input, MakeZeroSizedTensor(builder, input_dtype));
  }

  mlir::MlirOp grad_weight;
  if (output_mask[1]) {
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp broadcasted_mean,
        BroadcastIfNeeded(mean, input, {kTorchFeaturesDimensionIndex}));
    mlir::MlirOp x_minus_mean =
        mlir::stablehlo::Subtract(input, broadcasted_mean);
    mlir::MlirOp grad_out_times_x_minus_mean =
        mlir::stablehlo::Mul(grad_out, x_minus_mean);
    TT_ASSIGN_OR_RETURN(mlir::MlirOp dot_p,
                        BuildSumShlo(grad_out_times_x_minus_mean, reduce_dims));
    grad_weight = mlir::stablehlo::Mul(dot_p, inv_std);
    grad_weight =
        mlir::stablehlo::ConvertElementType(grad_weight, weight_dtype);
  } else {
    TT_ASSIGN_OR_RETURN(grad_weight,
                        MakeZeroSizedTensor(builder, weight_dtype));
  }

  mlir::MlirOp grad_bias;
  if (output_mask[2]) {
    TT_ASSIGN_OR_RETURN(grad_bias, BuildSumShlo(grad_out, reduce_dims));
    grad_bias = mlir::stablehlo::ConvertElementType(grad_bias, weight_dtype);
  } else {
    TT_ASSIGN_OR_RETURN(grad_bias, MakeZeroSizedTensor(builder, weight_dtype));
  }

  return {{grad_input, grad_weight, grad_bias}};
}

// Normalizes `input` with its batch statistics. Returns the output in input
// precision, and the batch mean and biased batch variance in `acc_dtype`.
MlirOpResults<3> BuildBatchNormTrainingWithVariance(
    mlir::MlirOp input, mlir::MlirOp weight, mlir::MlirOp bias, double eps,
    mlir::ElementType acc_dtype) {
  const mlir::ElementType input_dtype = GetElementTypeOrDie(input);

  // Promotion to accumulation precision.
  input = mlir::stablehlo::ConvertElementType(input, acc_dtype);
  weight = mlir::stablehlo::ConvertElementType(weight, acc_dtype);
  bias = mlir::stablehlo::ConvertElementType(bias, acc_dtype);

  auto outputs = mlir::stablehlo::BatchNormTraining(
      input, weight, bias, llvm::APFloat(static_cast<float>(eps)),
      kTorchFeaturesDimensionIndex);

  // Output tensor needs to be in input precision.
  auto output_tensor =
      mlir::stablehlo::ConvertElementType(outputs[0], input_dtype);

  // Stats outputs need to be in acc_dtype (F32) for PyTorch compatibility.
  auto output_mean = mlir::stablehlo::ConvertElementType(outputs[1], acc_dtype);
  auto output_variance =
      mlir::stablehlo::ConvertElementType(outputs[2], acc_dtype);
  return {output_tensor, output_mean, output_variance};
}

// Returns (1 - momentum) * running_stat + momentum * batch_stat, computed in
// the element type of `batch_stat` and returned in that of `running_stat`.
mlir::MlirOp UpdateRunningStat(mlir::MlirOp running_stat,
                               mlir::MlirOp batch_stat, mlir::MlirOp momentum) {
  const mlir::ElementType running_dtype = GetElementTypeOrDie(running_stat);
  running_stat = mlir::stablehlo::ConvertElementType(
      running_stat, GetElementTypeOrDie(batch_stat));
  auto one = MakeConstantLike(momentum, 1.0);
  auto one_minus_momentum = mlir::stablehlo::Subtract(one, momentum);
  auto keep = mlir::stablehlo::Mul(running_stat, one_minus_momentum);
  auto add = mlir::stablehlo::Mul(batch_stat, momentum);
  auto new_running_stat = mlir::stablehlo::Add(keep, add);
  return mlir::stablehlo::ConvertElementType(new_running_stat, running_dtype);
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
  if (training) {
    return BuildBatchNormBackwardTraining(
        grad_out, input, weight_opt, running_mean_opt, running_variance_opt,
        save_mean_opt, save_invstd_opt, output_mask, acc_dtype);
  }
  return BuildBatchNormBackwardInference(grad_out, input, weight_opt,
                                         running_mean_opt, running_variance_opt,
                                         eps, output_mask, acc_dtype);
}

absl::StatusOr<MlirOpResults<3>> BuildBatchNorm(
    mlir::MlirOp input, std::optional<mlir::MlirOp> weight_opt,
    std::optional<mlir::MlirOp> bias_opt,
    std::optional<mlir::MlirOp> running_mean_opt,
    std::optional<mlir::MlirOp> running_variance_opt, bool training,
    double momentum, double eps, mlir::ElementType acc_dtype) {
  // `momentum` is not used here: it only matters when the running stats are
  // updated, which BuildBatchNormTrainingAndUpdateStats does.

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
    auto [output_tensor, output_mean, output_variance] =
        BuildBatchNormTrainingWithVariance(input, weight, bias, eps, acc_dtype);
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
            : MakeConstant(builder, 1.0, input_dtype, {num_features});

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

absl::StatusOr<MlirOpResults<5>> BuildBatchNormTrainingAndUpdateStats(
    mlir::MlirOp input, std::optional<mlir::MlirOp> weight_opt,
    std::optional<mlir::MlirOp> bias_opt, mlir::MlirOp running_mean,
    mlir::MlirOp running_variance, mlir::MlirOp momentum, double eps,
    mlir::ElementType acc_dtype) {
  ABSL_VLOG(1) << "BuildBatchNormTrainingAndUpdateStats "
               << " input=" << input.ToString() << " eps=" << eps;
  mlir::MlirBuilder& builder = input.getBuilder();

  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const mlir::ElementType input_dtype = GetElementTypeOrDie(input);
  const int64_t num_features =
      input_type.getDimSize(kTorchFeaturesDimensionIndex);
  mlir::MlirOp weight =
      weight_opt ? *weight_opt
                 : MakeConstant(builder, 1.0, input_dtype, {num_features});
  mlir::MlirOp bias =
      bias_opt ? *bias_opt
               : MakeConstant(builder, 0.0, input_dtype, {num_features});

  auto [output_tensor, output_mean, output_variance] =
      BuildBatchNormTrainingWithVariance(input, weight, bias, eps, acc_dtype);
  auto output_inverted_variance = ComputeInvStd(output_variance, eps);

  // Like PyTorch, update running_var with the unbiased batch variance,
  // var * n / (n - 1), where n is the number of elements per channel.
  const Dimensions reduce_dims =
      GetBatchDimensions(input_type.getRank(), kTorchFeaturesDimensionIndex);
  const mlir::Type stats_elem_type =
      GetTensorTypeOrDie(output_variance).getElementType();
  mlir::MlirOp n = GetNumElements(input, stats_elem_type, reduce_dims);
  mlir::MlirOp one = MakeScalarConstant(builder, 1.0, stats_elem_type);
  mlir::MlirOp n_minus_one = mlir::stablehlo::Subtract(n, one);
  mlir::MlirOp n_over_n_minus_one = mlir::stablehlo::Div(n, n_minus_one);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp unbiased_factor,
                      BroadcastIfNeeded(n_over_n_minus_one, output_variance));
  auto unbiased_variance =
      mlir::stablehlo::Mul(output_variance, unbiased_factor);

  // `momentum` is a scalar input rather than a constant, so the program does
  // not depend on its value (with `momentum=None`, nn.BatchNorm passes a new
  // value on every step).
  momentum = mlir::stablehlo::ConvertElementType(
      momentum, GetElementTypeOrDie(output_mean));
  TT_ASSIGN_OR_RETURN(momentum, BroadcastIfNeeded(momentum, output_mean));
  auto new_running_mean =
      UpdateRunningStat(running_mean, output_mean, momentum);
  auto new_running_variance =
      UpdateRunningStat(running_variance, unbiased_variance, momentum);

  return {{output_tensor, output_mean, output_inverted_variance,
           new_running_mean, new_running_variance}};
}

}  // namespace torch_tpu
