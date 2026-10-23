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

#include "csrc/ops/native_batch_norm/native_batch_norm_aten_kernels.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/native_batch_norm/native_batch_norm.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

namespace {

absl::Status ValidateIsFloating(const at::Tensor& tensor,
                                const std::string_view arg_name) {
  TT_RET_CHECK(c10::isFloatingType(tensor.scalar_type()),
               error::kInvalidArgument)
      << "expected " << arg_name << " to be floating point, got "
      << ToString(tensor.scalar_type());
  return absl::OkStatus();
}

inline std::optional<mlir::MlirOp> GetOptionalMlirOp(
    absl::Span<mlir::MlirOp> mlir_inputs, int index) {
  if (index != -1) {
    return mlir_inputs[index];
  }
  return std::nullopt;
}

struct BatchNormDispatchParams {
  std::vector<at::Tensor> inputs;
  std::array<mlir::ElementType, 3> output_dtypes;
  std::array<Dimensions, 3> output_dims;
  NAryMlirOpBuilder<kDynamicSize, 3> op_builder;
};

absl::StatusOr<BatchNormDispatchParams> PrepareBatchNormDispatch(
    const at::Tensor& input, std::optional<at::Tensor> weight,
    std::optional<at::Tensor> bias, std::optional<at::Tensor> running_mean,
    std::optional<at::Tensor> running_variance, bool training, double momentum,
    double eps) {
  TT_RETURN_IF_ERROR(ValidateIsFloating(input, /*arg_name=*/"input"));
  ABSL_VLOG(1) << "TpuBatchNorm weight: " << !!weight << ", bias: " << !!bias
               << ", running_mean: " << !!running_mean
               << ", running_variance: " << !!running_variance
               << ", training: " << training << ", momentum: " << momentum
               << ", eps: " << eps;

  absl::Span<const int64_t> input_dims = input.sizes();
  TT_ASSIGN_OR_RETURN(const auto input_dtype,
                      ConvertTo<mlir::ElementType>(input.scalar_type()));
  Dimensions features_dims = {input.size(kTorchFeaturesDimensionIndex)};

  // To match CUDA semantics, save_mean and save_invstd are per-channel, i.e.
  // dim=[C], in both training and inference modes.
  Dimensions output_mean_dims = features_dims;
  Dimensions output_variance_inverted_dims = features_dims;

  const auto acc_type = ToAccumulateType(input.scalar_type());
  TT_ASSIGN_OR_RETURN(const auto acc_dtype,
                      ConvertTo<mlir::ElementType>(acc_type));

  const std::array<Dimensions, 3> output_dims = {
      Dimensions(input_dims.begin(), input_dims.end()), output_mean_dims,
      output_variance_inverted_dims};
  const std::array<mlir::ElementType, 3> output_dtypes = {input_dtype,
                                                          acc_dtype, acc_dtype};

  std::vector<at::Tensor> inputs;
  inputs.reserve(5);
  inputs.push_back(input);

  const int weight_idx = weight ? inputs.size() : -1;
  if (weight) {
    inputs.push_back(*weight);
  }

  const int bias_idx = bias ? inputs.size() : -1;
  if (bias) {
    inputs.push_back(*bias);
  }

  const int running_mean_idx = running_mean ? inputs.size() : -1;
  if (running_mean) {
    inputs.push_back(*running_mean);
  }

  const int running_var_idx = running_variance ? inputs.size() : -1;
  if (running_variance) {
    inputs.push_back(*running_variance);
  }

  auto op_builder =
      [weight_idx, bias_idx, running_mean_idx, running_var_idx, training,
       momentum, eps, acc_dtype](
          absl::Span<mlir::MlirOp> mlir_inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<MlirOpResults<3>> {
    mlir::MlirOp input_op = mlir_inputs[0];
    std::optional<mlir::MlirOp> weight_op =
        GetOptionalMlirOp(mlir_inputs, weight_idx);
    std::optional<mlir::MlirOp> bias_op =
        GetOptionalMlirOp(mlir_inputs, bias_idx);
    std::optional<mlir::MlirOp> running_mean_op =
        GetOptionalMlirOp(mlir_inputs, running_mean_idx);
    std::optional<mlir::MlirOp> running_var_op =
        GetOptionalMlirOp(mlir_inputs, running_var_idx);

    return BuildBatchNorm(input_op, weight_op, bias_op, running_mean_op,
                          running_var_op, training, momentum, eps, acc_dtype);
  };

  return BatchNormDispatchParams{
      .inputs = std::move(inputs),
      .output_dtypes = output_dtypes,
      .output_dims = output_dims,
      .op_builder = std::move(op_builder),
  };
}

// Returns true if training-mode batch norm must update the running stats.
// PyTorch requires both running stats or neither. Like `at::batch_norm`, an
// empty input leaves them unchanged.
bool ShouldUpdateRunningStats(const at::Tensor& input,
                              const std::optional<at::Tensor>& running_mean,
                              const std::optional<at::Tensor>& running_variance,
                              bool training) {
  return training && running_mean && running_variance && input.numel() > 0;
}

// Allocates (out, save_mean, save_invstd) for the batch norm of `input`.
absl::StatusOr<std::tuple<at::Tensor, at::Tensor, at::Tensor>>
MakeBatchNormOutputs(const at::Tensor& input) {
  const auto acc_type = ToAccumulateType(input.scalar_type());
  const Dimensions features_dims = {input.size(kTorchFeaturesDimensionIndex)};
  TT_ASSIGN_OR_RETURN(
      at::Tensor out,
      MakeEmptyTensor(input.sizes(), input.scalar_type(), input.device()));
  TT_ASSIGN_OR_RETURN(at::Tensor save_mean,
                      MakeEmptyTensor(features_dims, acc_type, input.device()));
  TT_ASSIGN_OR_RETURN(at::Tensor save_invstd,
                      MakeEmptyTensor(features_dims, acc_type, input.device()));
  return std::make_tuple(std::move(out), std::move(save_mean),
                         std::move(save_invstd));
}

// Runs training-mode batch norm, and writes the updated running stats into
// `running_mean` and `running_variance` in place, as PyTorch's CPU and CUDA
// kernels do.
absl::Status TpuBatchNormTrainingUpdateStatsOut(
    const at::Tensor& input, std::optional<at::Tensor> weight,
    std::optional<at::Tensor> bias, const at::Tensor& running_mean,
    const at::Tensor& running_variance, double momentum, double eps,
    OpParamCacheKeys param_keys, at::Tensor& out, at::Tensor& save_mean,
    at::Tensor& save_invstd) {
  TT_RETURN_IF_ERROR(ValidateIsFloating(input, /*arg_name=*/"input"));
  TT_ASSIGN_OR_RETURN(const auto input_dtype,
                      ConvertTo<mlir::ElementType>(input.scalar_type()));
  const auto acc_type = ToAccumulateType(input.scalar_type());
  TT_ASSIGN_OR_RETURN(const auto acc_dtype,
                      ConvertTo<mlir::ElementType>(acc_type));
  TT_ASSIGN_OR_RETURN(const auto running_mean_dtype,
                      ConvertTo<mlir::ElementType>(running_mean.scalar_type()));
  TT_ASSIGN_OR_RETURN(
      const auto running_var_dtype,
      ConvertTo<mlir::ElementType>(running_variance.scalar_type()));
  const Dimensions features_dims = {input.size(kTorchFeaturesDimensionIndex)};

  // Pass momentum as a tensor input rather than a constant, so the program
  // does not depend on its value.
  TT_ASSIGN_OR_RETURN(const at::Tensor momentum_tensor,
                      PromoteScalar(at::Scalar(momentum)).GetTensor(acc_type));

  std::vector<at::Tensor> inputs = {input, running_mean, running_variance,
                                    momentum_tensor};
  inputs.reserve(6);

  const int weight_idx = weight ? inputs.size() : -1;
  if (weight) {
    inputs.push_back(*weight);
  }

  const int bias_idx = bias ? inputs.size() : -1;
  if (bias) {
    inputs.push_back(*bias);
  }

  auto op_builder =
      [weight_idx, bias_idx, eps, acc_dtype](
          absl::Span<mlir::MlirOp> mlir_inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<MlirOpResults<5>> {
    std::optional<mlir::MlirOp> weight_op =
        GetOptionalMlirOp(mlir_inputs, weight_idx);
    std::optional<mlir::MlirOp> bias_op =
        GetOptionalMlirOp(mlir_inputs, bias_idx);
    return BuildBatchNormTrainingAndUpdateStats(
        /*input=*/mlir_inputs[0], weight_op, bias_op,
        /*running_mean=*/mlir_inputs[1], /*running_variance=*/mlir_inputs[2],
        /*momentum=*/mlir_inputs[3], eps, acc_dtype);
  };

  return DispatchOpOut<kDynamicSize, 5>(
      std::move(op_builder), inputs,
      {out, save_mean, save_invstd, running_mean, running_variance},
      {.out_dtypes = {input_dtype, acc_dtype, acc_dtype, running_mean_dtype,
                      running_var_dtype},
       .out_dims_list = {input.sizes(), features_dims, features_dims,
                         features_dims, features_dims},
       .op_param_cache_keys = std::move(param_keys)});
}

absl::StatusOr<DeviceBufferRefArray<3>> TpuBatchNorm(
    const at::Tensor& input, std::optional<at::Tensor> weight,
    std::optional<at::Tensor> bias, std::optional<at::Tensor> running_mean,
    std::optional<at::Tensor> running_variance, bool training, double momentum,
    double eps, OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(auto p, PrepareBatchNormDispatch(
                                  input, weight, bias, running_mean,
                                  running_variance, training, momentum, eps));
  const std::array<absl::Span<const int64_t>, 3> output_dims = {
      p.output_dims[0], p.output_dims[1], p.output_dims[2]};
  std::optional<DeviceBufferRefArray<3>> results;
  TT_ASSIGN_OR_RETURN(results,
                      (DispatchOp<kDynamicSize, 3>(
                          std::move(p.op_builder), p.inputs,
                          {.out_dtypes = p.output_dtypes,
                           .out_dims_list = output_dims,
                           .op_param_cache_keys = std::move(param_keys)})));
  return std::move(*results);
}

absl::Status TpuBatchNormOut(const at::Tensor& input,
                             std::optional<at::Tensor> weight,
                             std::optional<at::Tensor> bias,
                             std::optional<at::Tensor> running_mean,
                             std::optional<at::Tensor> running_variance,
                             bool training, double momentum, double eps,
                             OpParamCacheKeys param_keys, at::Tensor& out,
                             at::Tensor& save_mean, at::Tensor& save_invstd) {
  if (ShouldUpdateRunningStats(input, running_mean, running_variance,
                               training)) {
    return TpuBatchNormTrainingUpdateStatsOut(
        input, std::move(weight), std::move(bias), *running_mean,
        *running_variance, momentum, eps, std::move(param_keys), out, save_mean,
        save_invstd);
  }
  TT_ASSIGN_OR_RETURN(auto p, PrepareBatchNormDispatch(
                                  input, weight, bias, running_mean,
                                  running_variance, training, momentum, eps));
  const std::array<absl::Span<const int64_t>, 3> output_dims = {
      p.output_dims[0], p.output_dims[1], p.output_dims[2]};
  return DispatchOpOut<kDynamicSize, 3>(
      std::move(p.op_builder), p.inputs, {out, save_mean, save_invstd},
      {.out_dtypes = p.output_dtypes,
       .out_dims_list = output_dims,
       .op_param_cache_keys = std::move(param_keys)});
}

absl::StatusOr<DeviceBufferRefArray<3>> TpuBatchNormBackward(
    const at::Tensor& grad_out, const at::Tensor& input,
    std::optional<at::Tensor> weight, std::optional<at::Tensor> running_mean,
    std::optional<at::Tensor> running_variance,
    std::optional<at::Tensor> save_mean, std::optional<at::Tensor> save_invstd,
    bool training, double eps, std::array<bool, 3> output_mask,
    OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateIsFloating(input, /*arg_name=*/"input"));
  TT_RETURN_IF_ERROR(ValidateIsFloating(grad_out, /*arg_name=*/"grad_out"));

  std::vector<at::Tensor> inputs;
  inputs.reserve(7);  // Max inputs
  inputs.push_back(grad_out);
  inputs.push_back(input);

  const int weight_idx = weight ? inputs.size() : -1;
  if (weight) {
    inputs.push_back(*weight);
  }

  const int running_mean_idx = running_mean ? inputs.size() : -1;
  if (running_mean) {
    inputs.push_back(*running_mean);
  }

  const int running_var_idx = running_variance ? inputs.size() : -1;
  if (running_variance) {
    inputs.push_back(*running_variance);
  }

  const int save_mean_idx = save_mean ? inputs.size() : -1;
  if (save_mean) {
    inputs.push_back(*save_mean);
  }

  const int save_invstd_idx = save_invstd ? inputs.size() : -1;
  if (save_invstd) {
    inputs.push_back(*save_invstd);
  }

  const auto acc_type = ToAccumulateType(input.scalar_type());
  TT_ASSIGN_OR_RETURN(const auto acc_dtype,
                      ConvertTo<mlir::ElementType>(acc_type));
  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(grad_out.scalar_type()));

  mlir::ElementType weight_dtype = output_dtype;
  for (const auto& param :
       {weight, running_mean, running_variance, save_mean, save_invstd}) {
    if (param.has_value()) {
      TT_ASSIGN_OR_RETURN(weight_dtype,
                          ConvertTo<mlir::ElementType>(param->scalar_type()));
      break;
    }
  }

  absl::Span<const int64_t> input_dims = input.sizes();
  int64_t num_features = input.size(kTorchFeaturesDimensionIndex);
  Dimensions feature_dims = {num_features};
  Dimensions empty_dims = {0};

  Dimensions grad_input_dims =
      output_mask[0] ? Dimensions(input_dims.begin(), input_dims.end())
                     : empty_dims;
  Dimensions grad_weight_dims = output_mask[1] ? feature_dims : empty_dims;
  Dimensions grad_bias_dims = output_mask[2] ? feature_dims : empty_dims;

  const std::array<absl::Span<const int64_t>, 3> output_dims_list = {
      grad_input_dims, grad_weight_dims, grad_bias_dims};
  const std::array<mlir::ElementType, 3> output_dtypes = {
      output_dtype, weight_dtype, weight_dtype};

  auto op_builder =
      [weight_idx, running_mean_idx, running_var_idx, save_mean_idx,
       save_invstd_idx, training, eps, output_mask, acc_dtype](
          absl::Span<mlir::MlirOp> mlir_inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<MlirOpResults<3>> {
    mlir::MlirOp grad_out_op = mlir_inputs[0];
    mlir::MlirOp input_op = mlir_inputs[1];
    std::optional<mlir::MlirOp> weight_op =
        GetOptionalMlirOp(mlir_inputs, weight_idx);
    std::optional<mlir::MlirOp> running_mean_op =
        GetOptionalMlirOp(mlir_inputs, running_mean_idx);
    std::optional<mlir::MlirOp> running_var_op =
        GetOptionalMlirOp(mlir_inputs, running_var_idx);
    std::optional<mlir::MlirOp> save_mean_op =
        GetOptionalMlirOp(mlir_inputs, save_mean_idx);
    std::optional<mlir::MlirOp> save_invstd_op =
        GetOptionalMlirOp(mlir_inputs, save_invstd_idx);

    return BuildBatchNormBackward(
        grad_out_op, input_op, weight_op, running_mean_op, running_var_op,
        save_mean_op, save_invstd_op, training, eps, output_mask, acc_dtype);
  };

  std::optional<DeviceBufferRefArray<3>> results;
  TT_ASSIGN_OR_RETURN(results,
                      (DispatchOp<kDynamicSize, 3>(
                          std::move(op_builder), inputs,
                          {.out_dtypes = output_dtypes,
                           .out_dims_list = output_dims_list,
                           .op_param_cache_keys = std::move(param_keys)})));

  return std::move(*results);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenNativeBatchNorm(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& running_mean,
    const std::optional<at::Tensor>& running_variance, bool training,
    double momentum, double eps) {
  TT_KERNEL(OpName::kNativeBatchNorm, param_keys,
            (input, weight, bias, running_mean, running_variance, training,
             IgnoreInCacheKey(momentum, "Doesn't affect SHLO"), eps),
            {
              if (ShouldUpdateRunningStats(
                      input, SanitizeOptionalTensor(running_mean),
                      SanitizeOptionalTensor(running_variance), training)) {
                TT_ASSIGN_OR_THROW((auto [out, save_mean, save_invstd]),
                                   MakeBatchNormOutputs(input));
                TT_THROW_IF_ERROR(TpuBatchNormTrainingUpdateStatsOut(
                    input, SanitizeOptionalTensor(weight),
                    SanitizeOptionalTensor(bias), *running_mean,
                    *running_variance, momentum, eps, std::move(param_keys),
                    out, save_mean, save_invstd));
                return {out, save_mean, save_invstd};
              }
              TT_ASSIGN_OR_THROW(
                  (auto [output, mean, variance_inverse]),
                  TpuBatchNorm(input, SanitizeOptionalTensor(weight),
                               SanitizeOptionalTensor(bias),
                               SanitizeOptionalTensor(running_mean),
                               SanitizeOptionalTensor(running_variance),
                               training, momentum, eps, std::move(param_keys)));
              return {MakeTensor(std::move(output)),
                      MakeTensor(std::move(mean)),
                      MakeTensor(std::move(variance_inverse))};
            });
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> AtenNativeBatchNormOut(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& running_mean,
    const std::optional<at::Tensor>& running_variance, bool training,
    double momentum, double eps, at::Tensor& out, at::Tensor& save_mean,
    at::Tensor& save_invstd) {
  TT_KERNEL(
      OpName::kNativeBatchNormOut, param_keys,
      (input, weight, bias, running_mean, running_variance, training,
       IgnoreInCacheKey(momentum, "Doesn't affect SHLO"), eps, out, save_mean,
       save_invstd),
      {
        TT_THROW_IF_ERROR(TpuBatchNormOut(
            input, SanitizeOptionalTensor(weight), SanitizeOptionalTensor(bias),
            SanitizeOptionalTensor(running_mean),
            SanitizeOptionalTensor(running_variance), training, momentum, eps,
            std::move(param_keys), out, save_mean, save_invstd));

        return {out, save_mean, save_invstd};
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenNativeBatchNormLegit(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, at::Tensor& running_mean,
    at::Tensor& running_variance, bool training, double momentum, double eps) {
  TT_KERNEL(
      OpName::kNativeBatchNormLegit, _,
      (input, IgnoreInCacheKey(weight, "Delegates to AtenNativeBatchNorm"),
       IgnoreInCacheKey(bias, "Delegates to AtenNativeBatchNorm"), running_mean,
       running_variance,
       IgnoreInCacheKey(training, "Delegates to AtenNativeBatchNorm"),
       IgnoreInCacheKey(momentum, "Delegates to AtenNativeBatchNorm"),
       IgnoreInCacheKey(eps, "Delegates to AtenNativeBatchNorm")),
      {
        return AtenNativeBatchNorm(input, weight, bias, running_mean,
                                   running_variance, training, momentum, eps);
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenNativeBatchNormLegitNoStats(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, bool training, double momentum,
    double eps) {
  TT_KERNEL(
      OpName::kNativeBatchNormLegitNoStats, _,
      (input, IgnoreInCacheKey(weight, "Delegates to AtenNativeBatchNorm"),
       IgnoreInCacheKey(bias, "Delegates to AtenNativeBatchNorm"),
       IgnoreInCacheKey(training, "Delegates to AtenNativeBatchNorm"),
       IgnoreInCacheKey(momentum, "Delegates to AtenNativeBatchNorm"),
       IgnoreInCacheKey(eps, "Delegates to AtenNativeBatchNorm")),
      {
        return AtenNativeBatchNorm(
            input, weight, bias, /*running_mean=*/std::nullopt,
            /*running_variance=*/std::nullopt, training, momentum, eps);
      });
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> AtenNativeBatchNormLegitOut(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, at::Tensor& running_mean,
    at::Tensor& running_variance, bool training, double momentum, double eps,
    at::Tensor& out, at::Tensor& save_mean, at::Tensor& save_invstd) {
  TT_KERNEL(
      OpName::kNativeBatchNormLegitOut, param_keys,
      (input, weight, bias, running_mean, running_variance, training,
       IgnoreInCacheKey(momentum, "Doesn't affect SHLO"), eps, out, save_mean,
       save_invstd),
      {
        TT_THROW_IF_ERROR(TpuBatchNormOut(
            input, SanitizeOptionalTensor(weight), SanitizeOptionalTensor(bias),
            SanitizeOptionalTensor(running_mean),
            SanitizeOptionalTensor(running_variance), training, momentum, eps,
            std::move(param_keys), out, save_mean, save_invstd));

        return {out, save_mean, save_invstd};
      });
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>
AtenNativeBatchNormLegitNoStatsOut(const at::Tensor& input,
                                   const std::optional<at::Tensor>& weight,
                                   const std::optional<at::Tensor>& bias,
                                   bool training, double momentum, double eps,
                                   at::Tensor& out, at::Tensor& save_mean,
                                   at::Tensor& save_invstd) {
  TT_KERNEL(
      OpName::kNativeBatchNormLegitNoStatsOut, param_keys,
      (input, weight, bias, training,
       IgnoreInCacheKey(momentum, "Doesn't affect SHLO"), eps, out, save_mean,
       save_invstd),
      {
        TT_THROW_IF_ERROR(TpuBatchNormOut(
            input, SanitizeOptionalTensor(weight), SanitizeOptionalTensor(bias),
            /*running_mean=*/std::nullopt, /*running_variance=*/std::nullopt,
            training, momentum, eps, std::move(param_keys), out, save_mean,
            save_invstd));

        return {out, save_mean, save_invstd};
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenNativeBatchNormBackward(
    const at::Tensor& grad_out, const at::Tensor& input,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& running_mean,
    const std::optional<at::Tensor>& running_variance,
    const std::optional<at::Tensor>& save_mean,
    const std::optional<at::Tensor>& save_invstd, bool training, double eps,
    std::array<bool, 3> output_mask) {
  TT_KERNEL(OpName::kNativeBatchNormBackward, param_keys,
            (grad_out, input, weight, running_mean, running_variance, save_mean,
             save_invstd, training, eps, output_mask),
            {
              TT_ASSIGN_OR_THROW(
                  (auto [grad_input_buf, grad_weight_buf, grad_bias_buf]),
                  TpuBatchNormBackward(
                      grad_out, input, SanitizeOptionalTensor(weight),
                      SanitizeOptionalTensor(running_mean),
                      SanitizeOptionalTensor(running_variance),
                      SanitizeOptionalTensor(save_mean),
                      SanitizeOptionalTensor(save_invstd), training, eps,
                      output_mask, std::move(param_keys)));

              at::Tensor grad_input =
                  output_mask[0] ? MakeTensor(std::move(grad_input_buf))
                                 : at::Tensor();
              at::Tensor grad_weight =
                  output_mask[1] ? MakeTensor(std::move(grad_weight_buf))
                                 : at::Tensor();
              at::Tensor grad_bias = output_mask[2]
                                         ? MakeTensor(std::move(grad_bias_buf))
                                         : at::Tensor();

              return {grad_input, grad_weight, grad_bias};
            });
}

}  // namespace torch_tpu
