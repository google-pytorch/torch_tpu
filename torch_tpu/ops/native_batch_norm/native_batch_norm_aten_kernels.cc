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

#include "torch_tpu/ops/native_batch_norm/native_batch_norm_aten_kernels.h"

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/AccumulateType.h"
#include "ATen/core/TensorBody.h"
#include "ATen/native/Resize.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/native_batch_norm/native_batch_norm.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

namespace {

absl::StatusOr<DeviceBufferRefArray<3>> TpuBatchNorm(
    const at::Tensor& input, std::optional<at::Tensor> weight,
    std::optional<at::Tensor> bias, std::optional<at::Tensor> running_mean,
    std::optional<at::Tensor> running_variance, bool training, double momentum,
    double eps, OpParamCacheKeys param_keys) {
  ABSL_VLOG(1) << "TpuBatchNorm weight: " << !!weight << ", bias: " << !!bias
               << ", running_mean: " << !!running_mean
               << ", running_variance: " << !!running_variance
               << ", training: " << training << ", momentum: " << momentum
               << ", eps: " << eps;

  absl::Span<const int64_t> input_dims = input.sizes();
  TT_ASSIGN_OR_RETURN(const auto input_dtype,
                      ConvertTo<mlir::ElementType>(input.scalar_type()));
  Dimensions features_dims = {input.size(kTorchFeaturesDimensionIndex)};

  const Dimensions empty_dims = {0};
  Dimensions output_mean_dims = training ? features_dims : empty_dims;
  Dimensions output_variance_inverted_dims =
      training ? features_dims : empty_dims;

  const auto acc_type =
      at::toAccumulateType(input.scalar_type(), /*is_cuda=*/true);
  TT_ASSIGN_OR_RETURN(const auto acc_dtype,
                      ConvertTo<mlir::ElementType>(acc_type));

  std::optional<DeviceBufferRefArray<3>> results;
  const std::array<absl::Span<const int64_t>, 3> output_dims = {
      input_dims, output_mean_dims, output_variance_inverted_dims};
  const std::array<mlir::ElementType, 3> output_dtypes = {input_dtype,
                                                          acc_dtype, acc_dtype};

  if (weight && bias && running_mean && running_variance) {
    auto op_builder = [training, momentum, eps,
                       acc_dtype](FixedSizeSpan<mlir::MlirOp, 5> inputs) {
      auto& [input, weight, bias, running_mean, running_variance] = inputs;
      return BuildBatchNorm(input, weight, bias, running_mean, running_variance,
                            training, momentum, eps, acc_dtype);
    };
    TT_ASSIGN_OR_RETURN(
        results, (DispatchOp<5, 3>(
                     std::move(op_builder),
                     /*inputs=*/
                     {input, *weight, *bias, *running_mean, *running_variance},
                     {.out_dtypes = output_dtypes,
                      .out_dims_list = output_dims,
                      .op_param_cache_keys = std::move(param_keys)})));
  } else if (!weight && bias && running_mean && running_variance) {
    auto op_builder = [training, momentum, eps,
                       acc_dtype](FixedSizeSpan<mlir::MlirOp, 4> inputs) {
      auto& [input, bias, running_mean, running_variance] = inputs;
      return BuildBatchNorm(input, /*weight=*/{}, bias, running_mean,
                            running_variance, training, momentum, eps,
                            acc_dtype);
    };
    TT_ASSIGN_OR_RETURN(
        results,
        (DispatchOp<4, 3>(
            std::move(op_builder),
            /*inputs=*/{input, *bias, *running_mean, *running_variance},
            {.out_dtypes = output_dtypes,
             .out_dims_list = output_dims,
             .op_param_cache_keys = std::move(param_keys)})));
  } else if (weight && !bias && running_mean && running_variance) {
    auto op_builder = [training, momentum, eps,
                       acc_dtype](FixedSizeSpan<mlir::MlirOp, 4> inputs) {
      auto& [input, weight, running_mean, running_variance] = inputs;
      return BuildBatchNorm(input, weight, /*bias=*/{}, running_mean,
                            running_variance, training, momentum, eps,
                            acc_dtype);
    };
    TT_ASSIGN_OR_RETURN(
        results,
        (DispatchOp<4, 3>(std::move(op_builder),
                          {input, *weight, *running_mean, *running_variance},
                          {.out_dtypes = output_dtypes,
                           .out_dims_list = output_dims,
                           .op_param_cache_keys = std::move(param_keys)})));
  } else if (!weight && !bias && running_mean && running_variance) {
    auto op_builder = [training, momentum, eps,
                       acc_dtype](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
      auto& [input, running_mean, running_variance] = inputs;
      return BuildBatchNorm(input, /*weight=*/{}, /*bias=*/{}, running_mean,
                            running_variance, training, momentum, eps,
                            acc_dtype);
    };
    TT_ASSIGN_OR_RETURN(
        results,
        (DispatchOp<3, 3>(std::move(op_builder),
                          /*inputs=*/{input, *running_mean, *running_variance},
                          {.out_dtypes = output_dtypes,
                           .out_dims_list = output_dims,
                           .op_param_cache_keys = std::move(param_keys)})));
  } else if (!weight && !bias && !running_mean && !running_variance) {
    auto op_builder = [training, momentum, eps, acc_dtype](mlir::MlirOp input) {
      return BuildBatchNorm(input, /*weight=*/{}, /*bias=*/{},
                            /*running_mean=*/{}, /*running_variance=*/{},
                            training, momentum, eps, acc_dtype);
    };
    TT_ASSIGN_OR_RETURN(
        results,
        (DispatchOp<1, 3>(std::move(op_builder), input,
                          {.out_dtypes = output_dtypes,
                           .out_dims_list = output_dims,
                           .op_param_cache_keys = std::move(param_keys)})));
  } else if (weight && bias && !running_mean && !running_variance) {
    auto op_builder = [training, momentum, eps,
                       acc_dtype](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
      auto& [input, weight, bias] = inputs;
      return BuildBatchNorm(input, weight, bias,
                            /*running_mean=*/{}, /*running_variance=*/{},
                            training, momentum, eps, acc_dtype);
    };
    TT_ASSIGN_OR_RETURN(
        results,
        (DispatchOp<3, 3>(std::move(op_builder), {input, *weight, *bias},
                          {.out_dtypes = output_dtypes,
                           .out_dims_list = output_dims,
                           .op_param_cache_keys = std::move(param_keys)})));
  } else {
    return TT_ERROR(error::kInternal)
           << "Not implemented yet (weight: " << !!weight
           << ", bias: " << !!bias << ", running_mean: " << !!running_mean
           << ", running_variance: " << !!running_variance << ")";
  }
  return std::move(*results);
}

inline std::optional<mlir::MlirOp> GetOptionalMlirOp(
    absl::Span<mlir::MlirOp> mlir_inputs, int index) {
  if (index != -1) {
    return mlir_inputs[index];
  }
  return std::nullopt;
}

absl::StatusOr<DeviceBufferRefArray<3>> TpuBatchNormBackward(
    const at::Tensor& grad_out, const at::Tensor& input,
    std::optional<at::Tensor> weight, std::optional<at::Tensor> running_mean,
    std::optional<at::Tensor> running_variance,
    std::optional<at::Tensor> save_mean, std::optional<at::Tensor> save_invstd,
    bool training, double eps, std::array<bool, 3> output_mask,
    OpParamCacheKeys param_keys) {
  std::vector<at::Tensor> inputs;
  inputs.reserve(7);  // Max inputs
  inputs.push_back(grad_out);
  inputs.push_back(input);

  int weight_idx = -1;
  if (weight) {
    weight_idx = inputs.size();
    inputs.push_back(*weight);
  }

  int running_mean_idx = -1;
  if (running_mean) {
    running_mean_idx = inputs.size();
    inputs.push_back(*running_mean);
  }

  int running_var_idx = -1;
  if (running_variance) {
    running_var_idx = inputs.size();
    inputs.push_back(*running_variance);
  }

  int save_mean_idx = -1;
  if (save_mean) {
    save_mean_idx = inputs.size();
    inputs.push_back(*save_mean);
  }

  int save_invstd_idx = -1;
  if (save_invstd) {
    save_invstd_idx = inputs.size();
    inputs.push_back(*save_invstd);
  }

  const auto acc_type =
      at::toAccumulateType(input.scalar_type(), /*is_cuda=*/true);
  TT_ASSIGN_OR_RETURN(const auto acc_dtype,
                      ConvertTo<mlir::ElementType>(acc_type));
  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(grad_out.scalar_type()));

  mlir::ElementType weight_dtype = output_dtype;
  if (weight) {
    TT_ASSIGN_OR_RETURN(weight_dtype,
                        ConvertTo<mlir::ElementType>(weight->scalar_type()));
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
             momentum, eps),
            {
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
      (input, weight, bias, running_mean, running_variance, training, momentum,
       eps, out, save_mean, save_invstd),
      {
        TT_ASSIGN_OR_THROW(
            (auto [output, mean, variance_inverse]),
            TpuBatchNorm(input, SanitizeOptionalTensor(weight),
                         SanitizeOptionalTensor(bias),
                         SanitizeOptionalTensor(running_mean),
                         SanitizeOptionalTensor(running_variance), training,
                         momentum, eps, std::move(param_keys)));
        auto output_tensor = MakeTensor(std::move(output));
        auto mean_tensor = MakeTensor(std::move(mean));
        auto variance_inverse_tensor = MakeTensor(std::move(variance_inverse));

        at::native::resize_output(out, output_tensor.sizes());
        out.copy_(output_tensor);

        at::native::resize_output(save_mean, mean_tensor.sizes());
        save_mean.copy_(mean_tensor);

        at::native::resize_output(save_invstd, variance_inverse_tensor.sizes());
        save_invstd.copy_(variance_inverse_tensor);

        return {out, save_mean, save_invstd};
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenNativeBatchNormLegit(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, at::Tensor& running_mean,
    at::Tensor& running_variance, bool training, double momentum, double eps) {
  TT_KERNEL(OpName::kNativeBatchNormLegit, _,
            (input, IgnoreInCacheKey(weight, "Legacy usage"),
             IgnoreInCacheKey(bias, "Legacy usage"), running_mean,
             running_variance, IgnoreInCacheKey(training, "Legacy usage"),
             IgnoreInCacheKey(momentum, "Legacy usage"),
             IgnoreInCacheKey(eps, "Legacy usage")),
            {
              return AtenNativeBatchNorm(input, weight, bias, running_mean,
                                         running_variance, training, momentum,
                                         eps);
            });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenNativeBatchNormLegitNoStats(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, bool training, double momentum,
    double eps) {
  TT_KERNEL(OpName::kNativeBatchNormLegitNoStats, _,
            (input, IgnoreInCacheKey(weight, "Legacy usage"),
             IgnoreInCacheKey(bias, "Legacy usage"),
             IgnoreInCacheKey(training, "Legacy usage"),
             IgnoreInCacheKey(momentum, "Legacy usage"),
             IgnoreInCacheKey(eps, "Legacy usage")),
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
      (input, weight, bias, running_mean, running_variance, training, momentum,
       eps, out, save_mean, save_invstd),
      {
        TT_ASSIGN_OR_THROW(
            (auto [output, mean, variance_inverse]),
            TpuBatchNorm(input, SanitizeOptionalTensor(weight),
                         SanitizeOptionalTensor(bias),
                         SanitizeOptionalTensor(running_mean),
                         SanitizeOptionalTensor(running_variance), training,
                         momentum, eps, std::move(param_keys)));
        auto output_tensor = MakeTensor(std::move(output));
        auto mean_tensor = MakeTensor(std::move(mean));
        auto variance_inverse_tensor = MakeTensor(std::move(variance_inverse));

        at::native::resize_output(out, output_tensor.sizes());
        out.copy_(output_tensor);

        at::native::resize_output(save_mean, mean_tensor.sizes());
        save_mean.copy_(mean_tensor);

        at::native::resize_output(save_invstd, variance_inverse_tensor.sizes());
        save_invstd.copy_(variance_inverse_tensor);
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
      (input, weight, bias, training, momentum, eps, out, save_mean,
       save_invstd),
      {
        TT_ASSIGN_OR_THROW(
            (auto [output, mean, variance_inverse]),
            TpuBatchNorm(input, SanitizeOptionalTensor(weight),
                         SanitizeOptionalTensor(bias),
                         /*running_mean=*/std::nullopt,
                         /*running_variance=*/std::nullopt, training, momentum,
                         eps, std::move(param_keys)));
        auto output_tensor = MakeTensor(std::move(output));
        auto mean_tensor = MakeTensor(std::move(mean));
        auto variance_inverse_tensor = MakeTensor(std::move(variance_inverse));

        at::native::resize_output(out, output_tensor.sizes());
        out.copy_(output_tensor);

        at::native::resize_output(save_mean, mean_tensor.sizes());
        save_mean.copy_(mean_tensor);

        at::native::resize_output(save_invstd, variance_inverse_tensor.sizes());
        save_invstd.copy_(variance_inverse_tensor);
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
