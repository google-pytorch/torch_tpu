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

#include "csrc/ops/fused_adamw/fused_adamw_aten_kernels.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorImpl.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

// Assigns compiled output device buffers to input tensors starting at offset.
void AssignBuffers(std::vector<DeviceBufferRef>& buffers, const size_t offset,
                   const at::TensorList tensors) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    TT_THROW_IF_ERROR(
        AssignBufferToAtTensor(std::move(buffers[offset + i]), tensors[i]));
  }
}

// Appends converted MLIR element types of input tensors to the output vector.
void AppendDtypes(const at::TensorList tensors,
                  std::vector<mlir::ElementType>& out_dtypes) {
  out_dtypes.reserve(out_dtypes.size() + tensors.size());
  for (const auto& t : tensors) {
    const auto st = c10::toRealValueType(t.scalar_type());
    TT_ASSIGN_OR_THROW(const auto dt, ConvertTo<mlir::ElementType>(st));
    out_dtypes.push_back(dt);
  }
}

// Appends dimension spans of input tensors to the output dimension vector.
void AppendDims(const at::TensorList tensors,
                std::vector<absl::Span<const int64_t>>& out_dims) {
  out_dims.reserve(out_dims.size() + tensors.size());
  for (const auto& t : tensors) {
    out_dims.push_back(t.sizes());
  }
}

// Reshapes a scalar MLIR operation to a 0-D tensor if its rank is greater than
// 0.
mlir::MlirOp ReshapeScalarTo0DIfNeeded(mlir::MlirOp op) {
  if (GetTensorTypeOrDie(op).getRank() > 0) {
    return mlir::stablehlo::Reshape(op, {});
  }
  return op;
}

struct OneParamResult {
  mlir::MlirOp param;
  std::optional<mlir::MlirOp> grad;
  mlir::MlirOp exp_avg;
  mlir::MlirOp exp_avg_sq;
  std::optional<mlir::MlirOp> max_exp_avg_sq;
};

// Precomputed scalar hyperparameters and constants shared across all parameter
// updates in an AdamW step.
struct PrecomputedAdamwScalars {
  // Learning rate converted to computation element type.
  mlir::MlirOp lr;
  // Beta1 exponential decay rate.
  mlir::MlirOp beta1;
  // Beta2 exponential decay rate.
  mlir::MlirOp beta2;
  // Weight decay factor.
  mlir::MlirOp wd;
  // Epsilon numerical stability constant.
  mlir::MlirOp eps;
  // Constant 1.0 in computation element type.
  mlir::MlirOp one_cst;
  // 1.0 - beta1.
  mlir::MlirOp one_minus_beta1;
  // 1.0 - beta2.
  mlir::MlirOp one_minus_beta2;
  // 1.0 - lr * weight_decay.
  mlir::MlirOp one_minus_decay;
  // Optional gradient scale factor for mixed precision training.
  std::optional<mlir::MlirOp> grad_scale;
  // Optional flag indicating inf/nan gradients were encountered.
  std::optional<mlir::MlirOp> is_inf;
};

// Hoisted bias correction terms derived from step count for an AdamW update.
struct StepBiasCorrection {
  // Effective step size incorporating beta1 bias correction.
  mlir::MlirOp step_size;
  // Epsilon scaled by beta2 bias correction square root.
  mlir::MlirOp scaled_eps;
};

// Reverts parameter, gradient, and moment tensors to pre step values if inf or
// nan occurred.
absl::StatusOr<OneParamResult> ApplyFoundInfRevert(
    OneParamResult current, mlir::MlirOp param_old, mlir::MlirOp grad_old,
    mlir::MlirOp exp_avg_old, mlir::MlirOp exp_avg_sq_old,
    std::optional<mlir::MlirOp> max_exp_avg_sq_old, mlir::MlirOp is_inf,
    const bool amsgrad) {
  TT_ASSIGN_OR_RETURN(auto is_inf_param,
                      BroadcastIfNeeded(is_inf, current.param));
  current.param =
      mlir::stablehlo::Select(is_inf_param, param_old, current.param);

  if (current.grad.has_value()) {
    TT_ASSIGN_OR_RETURN(auto is_inf_grad,
                        BroadcastIfNeeded(is_inf, *current.grad));
    *current.grad =
        mlir::stablehlo::Select(is_inf_grad, grad_old, *current.grad);
  }

  TT_ASSIGN_OR_RETURN(auto is_inf_exp_avg,
                      BroadcastIfNeeded(is_inf, current.exp_avg));
  current.exp_avg =
      mlir::stablehlo::Select(is_inf_exp_avg, exp_avg_old, current.exp_avg);

  TT_ASSIGN_OR_RETURN(auto is_inf_exp_avg_sq,
                      BroadcastIfNeeded(is_inf, current.exp_avg_sq));
  current.exp_avg_sq = mlir::stablehlo::Select(
      is_inf_exp_avg_sq, exp_avg_sq_old, current.exp_avg_sq);

  if (amsgrad && current.max_exp_avg_sq.has_value() &&
      max_exp_avg_sq_old.has_value()) {
    TT_ASSIGN_OR_RETURN(auto is_inf_max_sq,
                        BroadcastIfNeeded(is_inf, *current.max_exp_avg_sq));
    *current.max_exp_avg_sq = mlir::stablehlo::Select(
        is_inf_max_sq, *max_exp_avg_sq_old, *current.max_exp_avg_sq);
  }
  return current;
}

// Casts computed intermediate ops back to their respective target output buffer
// element types.
OneParamResult CastOneParamOutputs(
    OneParamResult res, mlir::ElementType param_dtype,
    mlir::ElementType grad_dtype, mlir::ElementType exp_avg_dtype,
    mlir::ElementType exp_avg_sq_dtype,
    std::optional<mlir::ElementType> max_exp_avg_sq_dtype, const bool amsgrad) {
  if (GetElementTypeOrDie(res.param) != param_dtype) {
    res.param = mlir::stablehlo::ConvertElementType(res.param, param_dtype);
  }
  if (res.grad.has_value()) {
    if (GetElementTypeOrDie(*res.grad) != grad_dtype) {
      *res.grad = mlir::stablehlo::ConvertElementType(*res.grad, grad_dtype);
    }
  }
  if (GetElementTypeOrDie(res.exp_avg) != exp_avg_dtype) {
    res.exp_avg =
        mlir::stablehlo::ConvertElementType(res.exp_avg, exp_avg_dtype);
  }
  if (GetElementTypeOrDie(res.exp_avg_sq) != exp_avg_sq_dtype) {
    res.exp_avg_sq =
        mlir::stablehlo::ConvertElementType(res.exp_avg_sq, exp_avg_sq_dtype);
  }
  if (amsgrad && res.max_exp_avg_sq.has_value() &&
      max_exp_avg_sq_dtype.has_value()) {
    if (GetElementTypeOrDie(*res.max_exp_avg_sq) != *max_exp_avg_sq_dtype) {
      *res.max_exp_avg_sq = mlir::stablehlo::ConvertElementType(
          *res.max_exp_avg_sq, *max_exp_avg_sq_dtype);
    }
  }
  return res;
}

// Builds the StableHLO graph for updating a single parameter tensor and its
// corresponding optimizer state according to AdamW using precomputed scalar
// hyperparameters and step bias corrections.
absl::StatusOr<OneParamResult> BuildAdamwStepForOneParam(
    mlir::MlirOp param, mlir::MlirOp grad, mlir::MlirOp exp_avg,
    mlir::MlirOp exp_avg_sq, std::optional<mlir::MlirOp> max_exp_avg_sq,
    mlir::ElementType comp_type, const PrecomputedAdamwScalars& scalars,
    const StepBiasCorrection& step_corr, const AmsgradMode amsgrad_mode,
    const ObjectiveMode maximize_mode, mlir::ElementType param_dtype,
    mlir::ElementType grad_dtype, mlir::ElementType exp_avg_dtype,
    mlir::ElementType exp_avg_sq_dtype,
    std::optional<mlir::ElementType> max_exp_avg_sq_dtype) {
  const bool amsgrad = (amsgrad_mode == AmsgradMode::kEnabled);
  const bool maximize = (maximize_mode == ObjectiveMode::kMaximize);

  // Convert tensor inputs to the uniform computation element type.
  param = mlir::stablehlo::ConvertElementType(param, comp_type);
  grad = mlir::stablehlo::ConvertElementType(grad, comp_type);
  exp_avg = mlir::stablehlo::ConvertElementType(exp_avg, comp_type);
  exp_avg_sq = mlir::stablehlo::ConvertElementType(exp_avg_sq, comp_type);
  if (amsgrad && max_exp_avg_sq.has_value()) {
    *max_exp_avg_sq =
        mlir::stablehlo::ConvertElementType(*max_exp_avg_sq, comp_type);
  }

  const mlir::MlirOp param_old = param;
  const mlir::MlirOp grad_old = grad;
  const mlir::MlirOp exp_avg_old = exp_avg;
  const mlir::MlirOp exp_avg_sq_old = exp_avg_sq;
  const std::optional<mlir::MlirOp> max_exp_avg_sq_old = max_exp_avg_sq;

  // Unscale gradients under AMP training and negate gradient direction if
  // maximizing objective.
  std::optional<mlir::MlirOp> out_grad;
  if (scalars.grad_scale.has_value()) {
    TT_ASSIGN_OR_RETURN(auto scale_bcast,
                        BroadcastIfNeeded(*scalars.grad_scale, grad));
    grad = mlir::stablehlo::Div(grad, scale_bcast);
    out_grad = grad;
  }
  if (maximize) {
    grad = mlir::stablehlo::Neg(grad);
  }

  // In AdamW, weight decay is decoupled from the gradient update to prevent
  // decay from being scaled by the adaptive learning rate denominator.
  TT_ASSIGN_OR_RETURN(auto decay_bcast,
                      BroadcastIfNeeded(scalars.one_minus_decay, param));
  param = mlir::stablehlo::Mul(param, decay_bcast);

  // Step 2: Update first moment exponential moving average using hoisted
  // scalar 1.0 - beta1.
  TT_ASSIGN_OR_RETURN(auto beta1_bcast,
                      BroadcastIfNeeded(scalars.beta1, exp_avg));
  TT_ASSIGN_OR_RETURN(auto one_minus_beta1_bcast,
                      BroadcastIfNeeded(scalars.one_minus_beta1, grad));
  auto term1 = mlir::stablehlo::Mul(beta1_bcast, exp_avg);
  auto term2 = mlir::stablehlo::Mul(one_minus_beta1_bcast, grad);
  exp_avg = mlir::stablehlo::Add(term1, term2);

  // Step 3: Update second moment exponential moving average using hoisted
  // scalar 1.0 - beta2.
  TT_ASSIGN_OR_RETURN(auto beta2_bcast,
                      BroadcastIfNeeded(scalars.beta2, exp_avg_sq));
  auto grad_sq = mlir::stablehlo::Mul(grad, grad);
  TT_ASSIGN_OR_RETURN(auto one_minus_beta2_bcast,
                      BroadcastIfNeeded(scalars.one_minus_beta2, grad_sq));
  auto term1_sq = mlir::stablehlo::Mul(beta2_bcast, exp_avg_sq);
  auto term2_sq = mlir::stablehlo::Mul(one_minus_beta2_bcast, grad_sq);
  exp_avg_sq = mlir::stablehlo::Add(term1_sq, term2_sq);

  // Step 4: Apply cached bias corrected step size and scaled epsilon.
  TT_ASSIGN_OR_RETURN(auto scaled_eps_bcast,
                      BroadcastIfNeeded(step_corr.scaled_eps, param));

  mlir::MlirOp denom;
  if (amsgrad && max_exp_avg_sq.has_value()) {
    *max_exp_avg_sq = mlir::stablehlo::Max(*max_exp_avg_sq, exp_avg_sq);
    auto sqrt_max_sq = mlir::stablehlo::Sqrt(*max_exp_avg_sq);
    denom = mlir::stablehlo::Add(sqrt_max_sq, scaled_eps_bcast);
  } else {
    auto sqrt_sq = mlir::stablehlo::Sqrt(exp_avg_sq);
    denom = mlir::stablehlo::Add(sqrt_sq, scaled_eps_bcast);
  }

  // Step 5: Compute final parameter update step.
  TT_ASSIGN_OR_RETURN(auto step_size_bcast,
                      BroadcastIfNeeded(step_corr.step_size, param));
  auto update = mlir::stablehlo::Mul(step_size_bcast, exp_avg);
  update = mlir::stablehlo::Div(update, denom);
  param = mlir::stablehlo::Subtract(param, update);

  // Conditionally revert updates if non finite gradients were detected during
  // AMP unscaling.
  OneParamResult res{param, out_grad, exp_avg, exp_avg_sq, max_exp_avg_sq};
  if (scalars.is_inf.has_value()) {
    TT_ASSIGN_OR_RETURN(
        res, ApplyFoundInfRevert(res, param_old, grad_old, exp_avg_old,
                                 exp_avg_sq_old, max_exp_avg_sq_old,
                                 *scalars.is_inf, amsgrad));
  }
  // Cast result tensors back to their required individual buffer dtypes before
  // returning.
  return CastOneParamOutputs(res, param_dtype, grad_dtype, exp_avg_dtype,
                             exp_avg_sq_dtype, max_exp_avg_sq_dtype, amsgrad);
}

struct AdamwInputSpans {
  absl::Span<mlir::MlirOp> params;
  absl::Span<mlir::MlirOp> grads;
  absl::Span<mlir::MlirOp> exp_avgs;
  absl::Span<mlir::MlirOp> exp_avg_sqs;
  absl::Span<mlir::MlirOp> steps;
  absl::Span<mlir::MlirOp> max_exp_avg_sqs;
  mlir::MlirOp lr;
  mlir::MlirOp beta1;
  mlir::MlirOp beta2;
  mlir::MlirOp weight_decay;
  mlir::MlirOp eps;
  std::optional<mlir::MlirOp> grad_scale;
  std::optional<mlir::MlirOp> found_inf;
};

// Slices flat input op span into parameter lists, states, and optional
// hyperparameter ops.
AdamwInputSpans ExtractAdamwInputs(absl::Span<mlir::MlirOp> inputs,
                                   const size_t num_tensors,
                                   const size_t num_unique_steps,
                                   const bool amsgrad,
                                   const bool has_grad_scale,
                                   const bool has_found_inf) {
  size_t offset = 0;
  auto params = inputs.subspan(offset, num_tensors);
  offset += num_tensors;
  auto grads = inputs.subspan(offset, num_tensors);
  offset += num_tensors;
  auto exp_avgs = inputs.subspan(offset, num_tensors);
  offset += num_tensors;
  auto exp_avg_sqs = inputs.subspan(offset, num_tensors);
  offset += num_tensors;
  auto steps = inputs.subspan(offset, num_unique_steps);
  offset += num_unique_steps;
  absl::Span<mlir::MlirOp> max_exp_avg_sqs;
  if (amsgrad) {
    max_exp_avg_sqs = inputs.subspan(offset, num_tensors);
    offset += num_tensors;
  }
  mlir::MlirOp lr = inputs[offset++];
  mlir::MlirOp beta1 = inputs[offset++];
  mlir::MlirOp beta2 = inputs[offset++];
  mlir::MlirOp weight_decay = inputs[offset++];
  mlir::MlirOp eps = inputs[offset++];
  std::optional<mlir::MlirOp> grad_scale;
  if (has_grad_scale) grad_scale = inputs[offset++];
  std::optional<mlir::MlirOp> found_inf;
  if (has_found_inf) found_inf = inputs[offset++];
  return AdamwInputSpans{
      params, grads, exp_avgs,     exp_avg_sqs, steps,      max_exp_avg_sqs, lr,
      beta1,  beta2, weight_decay, eps,         grad_scale, found_inf};
}

struct AdamwDtypeSpans {
  absl::Span<const mlir::ElementType> params;
  absl::Span<const mlir::ElementType> grads;
  absl::Span<const mlir::ElementType> exp_avgs;
  absl::Span<const mlir::ElementType> exp_avg_sqs;
  absl::Span<const mlir::ElementType> max_exp_avg_sqs;
};

// Slices flat output dtype span into individual element type spans for each
// optimizer state list.
AdamwDtypeSpans ExtractAdamwDtypes(
    absl::Span<const mlir::ElementType> out_dtypes, const size_t num_tensors,
    const bool amsgrad, const bool has_grad_scale) {
  size_t offset = 0;
  auto params = out_dtypes.subspan(offset, num_tensors);
  offset += num_tensors;
  absl::Span<const mlir::ElementType> grads = params;
  if (has_grad_scale) {
    grads = out_dtypes.subspan(offset, num_tensors);
    offset += num_tensors;
  }
  auto exp_avgs = out_dtypes.subspan(offset, num_tensors);
  offset += num_tensors;
  auto exp_avg_sqs = out_dtypes.subspan(offset, num_tensors);
  offset += num_tensors;
  absl::Span<const mlir::ElementType> max_exp_avg_sqs;
  if (amsgrad) {
    max_exp_avg_sqs = out_dtypes.subspan(offset, num_tensors);
    offset += num_tensors;
  }
  return AdamwDtypeSpans{params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs};
}

// Concatenates per tensor updated output vectors into a single flattened MLIR
// result vector.
mlir::SmallVector<mlir::MlirOp> CombineAdamwResults(
    const std::vector<mlir::MlirOp>& params,
    const std::vector<mlir::MlirOp>& grads,
    const std::vector<mlir::MlirOp>& exp_avgs,
    const std::vector<mlir::MlirOp>& exp_avg_sqs,
    const std::vector<mlir::MlirOp>& max_exp_avg_sqs, const size_t num_tensors,
    const bool amsgrad, const bool has_grad_scale) {
  const size_t out_count = (amsgrad ? 4 : 3) + (has_grad_scale ? 1 : 0);
  mlir::SmallVector<mlir::MlirOp> results;
  results.reserve(out_count * num_tensors);
  results.insert(results.end(), params.begin(), params.end());
  if (has_grad_scale) {
    results.insert(results.end(), grads.begin(), grads.end());
  }
  results.insert(results.end(), exp_avgs.begin(), exp_avgs.end());
  results.insert(results.end(), exp_avg_sqs.begin(), exp_avg_sqs.end());
  if (amsgrad) {
    results.insert(results.end(), max_exp_avg_sqs.begin(),
                   max_exp_avg_sqs.end());
  }
  return results;
}

PrecomputedAdamwScalars BuildPrecomputedAdamwScalars(
    AdamwInputSpans& in_spans, mlir::ElementType comp_type) {
  PrecomputedAdamwScalars s;
  s.lr = mlir::stablehlo::ConvertElementType(in_spans.lr, comp_type);
  s.beta1 = mlir::stablehlo::ConvertElementType(in_spans.beta1, comp_type);
  s.beta2 = mlir::stablehlo::ConvertElementType(in_spans.beta2, comp_type);
  s.wd = mlir::stablehlo::ConvertElementType(in_spans.weight_decay, comp_type);
  s.eps = mlir::stablehlo::ConvertElementType(in_spans.eps, comp_type);
  s.one_cst = MakeConstantLike(s.lr, 1.0);
  s.one_minus_beta1 = mlir::stablehlo::Subtract(s.one_cst, s.beta1);
  s.one_minus_beta2 = mlir::stablehlo::Subtract(s.one_cst, s.beta2);
  auto decay_factor = mlir::stablehlo::Mul(s.lr, s.wd);
  s.one_minus_decay = mlir::stablehlo::Subtract(s.one_cst, decay_factor);
  if (in_spans.grad_scale.has_value()) {
    s.grad_scale =
        mlir::stablehlo::ConvertElementType(*in_spans.grad_scale, comp_type);
  }
  if (in_spans.found_inf.has_value()) {
    auto found_inf_comp =
        mlir::stablehlo::ConvertElementType(*in_spans.found_inf, comp_type);
    auto zero_inf = MakeConstantLike(found_inf_comp, 0.0);
    s.is_inf = mlir::stablehlo::Compare(
        found_inf_comp, zero_inf, mlir::stablehlo::ComparisonDirection::NE);
  }
  return s;
}

absl::StatusOr<StepBiasCorrection> GetOrBuildAdamwStepCorrection(
    mlir::MlirOp step, mlir::ElementType comp_type,
    const PrecomputedAdamwScalars& s,
    absl::flat_hash_map<const void*, StepBiasCorrection>& step_cache) {
  const void* op_key = step.getValue().getAsOpaquePointer();
  if (const auto it = step_cache.find(op_key); it != step_cache.end()) {
    return it->second;
  }
  auto step_c = ReshapeScalarTo0DIfNeeded(
      mlir::stablehlo::ConvertElementType(step, comp_type));
  TT_ASSIGN_OR_RETURN(auto beta1_step, BroadcastIfNeeded(s.beta1, step_c));
  auto pow_beta1 = mlir::stablehlo::Pow(beta1_step, step_c);
  mlir::MlirOp one_cst = s.one_cst;
  auto bias_correction1 = mlir::stablehlo::Subtract(one_cst, pow_beta1);

  TT_ASSIGN_OR_RETURN(auto beta2_step, BroadcastIfNeeded(s.beta2, step_c));
  auto pow_beta2 = mlir::stablehlo::Pow(beta2_step, step_c);
  auto bias_correction2 = mlir::stablehlo::Subtract(one_cst, pow_beta2);
  auto bias_correction2_sqrt = mlir::stablehlo::Sqrt(bias_correction2);

  mlir::MlirOp lr = s.lr;
  auto step_size_unscaled = mlir::stablehlo::Div(lr, bias_correction1);
  auto step_size =
      mlir::stablehlo::Mul(step_size_unscaled, bias_correction2_sqrt);
  mlir::MlirOp eps = s.eps;
  auto scaled_eps = mlir::stablehlo::Mul(eps, bias_correction2_sqrt);

  StepBiasCorrection corr{step_size, scaled_eps};
  step_cache.emplace(op_key, corr);
  return corr;
}

// Iterates across all multi tensor optimizer lists and generates the combined
// StableHLO operations for the entire optimizer step.
absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> BuildAdamwShlo(
    absl::Span<mlir::MlirOp> inputs, const size_t num_tensors,
    absl::Span<const size_t> step_remap, const size_t num_unique_steps,
    const AmsgradMode amsgrad_mode, const ObjectiveMode maximize_mode,
    const bool has_grad_scale, const bool has_found_inf,
    absl::Span<const mlir::ElementType> out_dtypes) {
  const bool amsgrad = (amsgrad_mode == AmsgradMode::kEnabled);
  auto in_spans = ExtractAdamwInputs(inputs, num_tensors, num_unique_steps,
                                     amsgrad, has_grad_scale, has_found_inf);
  auto dt_spans =
      ExtractAdamwDtypes(out_dtypes, num_tensors, amsgrad, has_grad_scale);

  TT_ASSIGN_OR_RETURN(const auto f32_dtype,
                      ConvertTo<mlir::ElementType>(c10::ScalarType::Float));
  TT_ASSIGN_OR_RETURN(const auto f64_dtype,
                      ConvertTo<mlir::ElementType>(c10::ScalarType::Double));

  std::optional<PrecomputedAdamwScalars> f32_scalars;
  std::optional<PrecomputedAdamwScalars> f64_scalars;
  absl::flat_hash_map<const void*, StepBiasCorrection> f32_step_cache;
  absl::flat_hash_map<const void*, StepBiasCorrection> f64_step_cache;
  f32_step_cache.reserve(num_unique_steps);

  std::vector<mlir::MlirOp> new_params;
  std::vector<mlir::MlirOp> new_grads;
  std::vector<mlir::MlirOp> new_exp_avgs;
  std::vector<mlir::MlirOp> new_exp_avg_sqs;
  std::vector<mlir::MlirOp> new_max_exp_avg_sqs;
  new_params.reserve(num_tensors);
  if (has_grad_scale) new_grads.reserve(num_tensors);
  new_exp_avgs.reserve(num_tensors);
  new_exp_avg_sqs.reserve(num_tensors);
  if (amsgrad) new_max_exp_avg_sqs.reserve(num_tensors);

  for (size_t i = 0; i < num_tensors; ++i) {
    const mlir::ElementType comp_type =
        (dt_spans.params[i] == f64_dtype || dt_spans.exp_avgs[i] == f64_dtype)
            ? f64_dtype
            : f32_dtype;
    std::optional<PrecomputedAdamwScalars>& opt_scalars =
        (comp_type == f64_dtype) ? f64_scalars : f32_scalars;
    if (!opt_scalars.has_value()) {
      opt_scalars = BuildPrecomputedAdamwScalars(in_spans, comp_type);
    }
    auto& step_cache =
        (comp_type == f64_dtype) ? f64_step_cache : f32_step_cache;
    TT_ASSIGN_OR_RETURN(
        const StepBiasCorrection step_corr,
        GetOrBuildAdamwStepCorrection(in_spans.steps[step_remap[i]], comp_type,
                                      *opt_scalars, step_cache));

    std::optional<mlir::MlirOp> max_sq;
    std::optional<mlir::ElementType> max_sq_dt;
    if (amsgrad) {
      max_sq = in_spans.max_exp_avg_sqs[i];
      max_sq_dt = dt_spans.max_exp_avg_sqs[i];
    }
    TT_ASSIGN_OR_RETURN(
        auto res,
        BuildAdamwStepForOneParam(
            in_spans.params[i], in_spans.grads[i], in_spans.exp_avgs[i],
            in_spans.exp_avg_sqs[i], max_sq, comp_type, *opt_scalars, step_corr,
            amsgrad_mode, maximize_mode, dt_spans.params[i], dt_spans.grads[i],
            dt_spans.exp_avgs[i], dt_spans.exp_avg_sqs[i], max_sq_dt));
    new_params.push_back(res.param);
    if (has_grad_scale && res.grad.has_value()) {
      new_grads.push_back(*res.grad);
    }
    new_exp_avgs.push_back(res.exp_avg);
    new_exp_avg_sqs.push_back(res.exp_avg_sq);
    if (amsgrad && res.max_exp_avg_sq.has_value()) {
      new_max_exp_avg_sqs.push_back(*res.max_exp_avg_sq);
    }
  }

  return CombineAdamwResults(new_params, new_grads, new_exp_avgs,
                             new_exp_avg_sqs, new_max_exp_avg_sqs, num_tensors,
                             amsgrad, has_grad_scale);
}

// Validates that all input tensor state lists have matching lengths.
void ValidateAdamwTensorListSizes(at::TensorList self, at::TensorList grads,
                                  at::TensorList exp_avgs,
                                  at::TensorList exp_avg_sqs,
                                  at::TensorList max_exp_avg_sqs,
                                  at::TensorList state_steps,
                                  const bool amsgrad) {
  const size_t num_tensors = self.size();
  TT_CHECK_THROW(grads.size() == num_tensors, error::kInvalidArgument)
      << "expected grads to have the same number of tensors as self, got "
      << grads.size();
  TT_CHECK_THROW(exp_avgs.size() == num_tensors, error::kInvalidArgument)
      << "expected exp_avgs to have the same number of tensors as self, got "
      << exp_avgs.size();
  TT_CHECK_THROW(exp_avg_sqs.size() == num_tensors, error::kInvalidArgument)
      << "expected exp_avg_sqs to have the same number of tensors as self, got "
      << exp_avg_sqs.size();
  TT_CHECK_THROW(state_steps.size() == num_tensors, error::kInvalidArgument)
      << "expected state_steps to have the same number of tensors as self, got "
      << state_steps.size();
  if (amsgrad) {
    TT_CHECK_THROW(max_exp_avg_sqs.size() == num_tensors,
                   error::kInvalidArgument)
        << "expected max_exp_avg_sqs to have the same number of tensors as "
           "self, got "
        << max_exp_avg_sqs.size();
  }
}

// Selects 0 based argument indices of input tensors to donate to XLA for HBM
// buffer reuse across mutated tensors.
Indices ComputeAdamwDonatedIndices(const size_t num_tensors,
                                   const size_t num_unique_steps,
                                   const bool amsgrad,
                                   const bool has_grad_scale) {
  Indices donated_indices;
  donated_indices.reserve(((amsgrad ? 4 : 3) + (has_grad_scale ? 1 : 0)) *
                          num_tensors);
  const size_t max_sq_base = 4 * num_tensors + num_unique_steps;
  for (size_t i = 0; i < num_tensors; ++i) {
    donated_indices.push_back(i);  // self
    if (has_grad_scale) {
      donated_indices.push_back(1 * num_tensors + i);  // grads
    }
    donated_indices.push_back(2 * num_tensors + i);  // exp_avgs
    donated_indices.push_back(3 * num_tensors + i);  // exp_avg_sqs
    if (amsgrad) {
      donated_indices.push_back(max_sq_base + i);  // max_exp_avg_sqs
    }
  }
  return donated_indices;
}

// Assigns computed output device buffers back in place to mutated eager
// parameter and state tensors.
void AssignAdamwBuffers(std::vector<DeviceBufferRef>& result_buffers,
                        const size_t num_tensors, const bool amsgrad,
                        const bool has_grad_scale, at::TensorList self,
                        at::TensorList grads, at::TensorList exp_avgs,
                        at::TensorList exp_avg_sqs,
                        at::TensorList max_exp_avg_sqs) {
  size_t buf_offset = 0;
  AssignBuffers(result_buffers, buf_offset, self);
  buf_offset += num_tensors;
  if (has_grad_scale) {
    AssignBuffers(result_buffers, buf_offset, grads);
    buf_offset += num_tensors;
  }
  AssignBuffers(result_buffers, buf_offset, exp_avgs);
  buf_offset += num_tensors;
  AssignBuffers(result_buffers, buf_offset, exp_avg_sqs);
  buf_offset += num_tensors;
  if (amsgrad) {
    AssignBuffers(result_buffers, buf_offset, max_exp_avg_sqs);
  }
}

// Key identifying a unique step tensor buffer for AdamW step deduplication.
struct StepBufferKey {
  // Underlying device buffer if the step tensor is contiguous and unaliased.
  std::optional<DeviceBufferRef> buffer_ref;
  // Fallback pointer to tensor implementation when device buffer is
  // unavailable.
  const c10::TensorImpl* fallback_impl = nullptr;
  // Scalar type of the step tensor.
  int scalar_type = 0;
  bool operator==(const StepBufferKey&) const = default;
  template <typename H>
  friend H AbslHashValue(H h, const StepBufferKey& k) {
    return H::combine(std::move(h), k.buffer_ref, k.fallback_impl,
                      k.scalar_type);
  }
};

StepBufferKey MakeStepBufferKey(const at::Tensor& t) {
  if (t.defined() && t.is_contiguous() && t.storage_offset() == 0 &&
      !t.is_conj()) {
    auto buf_or = GetBaseBuffer(t);
    if (buf_or.ok() && buf_or->device_buffer_list() != nullptr) {
      return StepBufferKey{
          .buffer_ref = std::move(*buf_or),
          .fallback_impl = nullptr,
          .scalar_type = static_cast<int>(t.scalar_type()),
      };
    }
  }
  return StepBufferKey{
      .buffer_ref = std::nullopt,
      .fallback_impl = t.unsafeGetTensorImpl(),
      .scalar_type = static_cast<int>(t.scalar_type()),
  };
}

// Orchestrates eager execution: flattens input tensor lists, deduplicates step
// tensors by underlying device buffer identity, configures output shapes and
// dtypes, dispatches to compilation cache, and assigns result buffers back to
// mutated input tensors.
void DispatchAdamw(at::TensorList self, at::TensorList grads,
                   at::TensorList exp_avgs, at::TensorList exp_avg_sqs,
                   at::TensorList max_exp_avg_sqs, at::TensorList state_steps,
                   const at::Tensor& lr_tensor, const at::Tensor& beta1_tensor,
                   const at::Tensor& beta2_tensor,
                   const at::Tensor& weight_decay_tensor,
                   const at::Tensor& eps_tensor, const AmsgradMode amsgrad_mode,
                   const ObjectiveMode maximize_mode,
                   const std::optional<at::Tensor>& grad_scale,
                   const std::optional<at::Tensor>& found_inf,
                   OpParamCacheKeys param_keys) {
  if (self.empty()) return;

  const bool amsgrad = (amsgrad_mode == AmsgradMode::kEnabled);

  ValidateAdamwTensorListSizes(self, grads, exp_avgs, exp_avg_sqs,
                               max_exp_avg_sqs, state_steps, amsgrad);

  const size_t num_tensors = self.size();
  std::vector<at::Tensor> unique_steps;
  unique_steps.reserve(num_tensors);
  std::vector<size_t> step_remap;
  step_remap.reserve(num_tensors);
  absl::flat_hash_map<StepBufferKey, size_t> seen_step_bufs;
  seen_step_bufs.reserve(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    auto [it, inserted] = seen_step_bufs.emplace(
        MakeStepBufferKey(state_steps[i]), unique_steps.size());
    if (inserted) {
      unique_steps.push_back(state_steps[i]);
    }
    step_remap.push_back(it->second);
  }
  const size_t num_unique_steps = unique_steps.size();

  std::vector<at::Tensor> inputs;
  inputs.reserve((amsgrad ? 5 * num_tensors : 4 * num_tensors) +
                 num_unique_steps + 7);
  inputs.insert(inputs.end(), self.begin(), self.end());
  inputs.insert(inputs.end(), grads.begin(), grads.end());
  inputs.insert(inputs.end(), exp_avgs.begin(), exp_avgs.end());
  inputs.insert(inputs.end(), exp_avg_sqs.begin(), exp_avg_sqs.end());
  inputs.insert(inputs.end(), unique_steps.begin(), unique_steps.end());
  if (amsgrad) {
    inputs.insert(inputs.end(), max_exp_avg_sqs.begin(), max_exp_avg_sqs.end());
  }
  inputs.push_back(lr_tensor);
  inputs.push_back(beta1_tensor);
  inputs.push_back(beta2_tensor);
  inputs.push_back(weight_decay_tensor);
  inputs.push_back(eps_tensor);

  const bool has_grad_scale = grad_scale.has_value() && grad_scale->defined();
  if (has_grad_scale) {
    inputs.push_back(*grad_scale);
  }
  const bool has_found_inf = found_inf.has_value() && found_inf->defined();
  if (has_found_inf) {
    inputs.push_back(*found_inf);
  }

  TT_THROW_IF_ERROR(param_keys.SetParam("has_grad_scale", has_grad_scale));
  TT_THROW_IF_ERROR(param_keys.SetParam("has_found_inf", has_found_inf));
  if (num_unique_steps < num_tensors) {
    TT_THROW_IF_ERROR(
        param_keys.SetParam("step_remap", absl::MakeConstSpan(step_remap)));
  }

  const size_t out_count = (amsgrad ? 4 : 3) + (has_grad_scale ? 1 : 0);
  std::vector<mlir::ElementType> out_dtypes;
  out_dtypes.reserve(out_count * num_tensors);
  AppendDtypes(self, out_dtypes);
  if (has_grad_scale) {
    AppendDtypes(grads, out_dtypes);
  }
  AppendDtypes(exp_avgs, out_dtypes);
  AppendDtypes(exp_avg_sqs, out_dtypes);
  if (amsgrad) {
    AppendDtypes(max_exp_avg_sqs, out_dtypes);
  }

  std::vector<absl::Span<const int64_t>> out_dims_list;
  out_dims_list.reserve(out_count * num_tensors);
  AppendDims(self, out_dims_list);
  if (has_grad_scale) {
    AppendDims(grads, out_dims_list);
  }
  AppendDims(exp_avgs, out_dims_list);
  AppendDims(exp_avg_sqs, out_dims_list);
  if (amsgrad) {
    AppendDims(max_exp_avg_sqs, out_dims_list);
  }

  auto op_builder =
      [num_tensors, step_remap = std::move(step_remap), num_unique_steps,
       amsgrad_mode, maximize_mode, has_grad_scale, has_found_inf,
       out_dtypes](absl::Span<mlir::MlirOp> mlir_inputs, mlir::MlirBuilder&)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    return BuildAdamwShlo(mlir_inputs, num_tensors, step_remap,
                          num_unique_steps, amsgrad_mode, maximize_mode,
                          has_grad_scale, has_found_inf, out_dtypes);
  };

  Indices donated_indices = ComputeAdamwDonatedIndices(
      num_tensors, num_unique_steps, amsgrad, has_grad_scale);

  DispatchOpOptions<kDynamicSize> options = {
      .out_dtypes = out_dtypes,
      .out_dims_list = out_dims_list,
      .op_param_cache_keys = std::move(param_keys),
      .donated_indices = std::move(donated_indices),
  };

  // Compile and execute the multi-tensor StableHLO graph, retrieving output
  // device buffers.
  TT_ASSIGN_OR_THROW(auto result_buffers,
                     (DispatchOp<kDynamicSize, kDynamicSize>(
                         std::move(op_builder), inputs, std::move(options))));

  AssignAdamwBuffers(result_buffers, num_tensors, amsgrad, has_grad_scale, self,
                     grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs);
}

}  // namespace

// Promote scalar hyperparameters to standard eager tensors and validate input
// precision requirements.
void AtenFusedAdamw(at::TensorList self, at::TensorList grads,
                    at::TensorList exp_avgs, at::TensorList exp_avg_sqs,
                    at::TensorList max_exp_avg_sqs, at::TensorList state_steps,
                    const double lr, const double beta1, const double beta2,
                    const double weight_decay, const double eps,
                    const bool amsgrad, const bool maximize,
                    const std::optional<at::Tensor>& grad_scale,
                    const std::optional<at::Tensor>& found_inf) {
  auto promoted_lr = PromoteScalar(at::Scalar(lr));
  auto promoted_beta1 = PromoteScalar(at::Scalar(beta1));
  auto promoted_beta2 = PromoteScalar(at::Scalar(beta2));
  auto promoted_weight_decay = PromoteScalar(at::Scalar(weight_decay));
  auto promoted_eps = PromoteScalar(at::Scalar(eps));

  TT_KERNEL(
      OpName::kFusedAdamw, param_keys,
      (self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
       promoted_lr, promoted_beta1, promoted_beta2, promoted_weight_decay,
       promoted_eps, amsgrad, maximize, grad_scale, found_inf),
      {
        if (self.empty()) return;
        TT_CHECK_THROW(IsFloatingPoint(self[0]),
                       error::kPythonNotImplementedError)
            << "expected the input dtype to be floating-point, got "
            << ToString(self[0].scalar_type());
        const auto st = c10::toRealValueType(self[0].scalar_type());
        const auto hyperparam_st =
            (st == at::kDouble) ? at::kDouble : at::kFloat;
        TT_ASSIGN_OR_THROW(const at::Tensor lr_t,
                           promoted_lr.GetTensor(hyperparam_st));
        TT_ASSIGN_OR_THROW(const at::Tensor b1_t,
                           promoted_beta1.GetTensor(hyperparam_st));
        TT_ASSIGN_OR_THROW(const at::Tensor b2_t,
                           promoted_beta2.GetTensor(hyperparam_st));
        TT_ASSIGN_OR_THROW(const at::Tensor wd_t,
                           promoted_weight_decay.GetTensor(hyperparam_st));
        TT_ASSIGN_OR_THROW(const at::Tensor eps_t,
                           promoted_eps.GetTensor(hyperparam_st));

        DispatchAdamw(
            self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
            lr_t, b1_t, b2_t, wd_t, eps_t,
            amsgrad ? AmsgradMode::kEnabled : AmsgradMode::kDisabled,
            maximize ? ObjectiveMode::kMaximize : ObjectiveMode::kMinimize,
            grad_scale, found_inf, std::move(param_keys));
      });
}

// Promote scalar hyperparameters to standard eager tensors and validate input
// precision requirements.
void AtenFusedAdamwTensorLr(at::TensorList self, at::TensorList grads,
                            at::TensorList exp_avgs, at::TensorList exp_avg_sqs,
                            at::TensorList max_exp_avg_sqs,
                            at::TensorList state_steps, const at::Tensor& lr,
                            const double beta1, const double beta2,
                            const double weight_decay, const double eps,
                            const bool amsgrad, const bool maximize,
                            const std::optional<at::Tensor>& grad_scale,
                            const std::optional<at::Tensor>& found_inf) {
  auto promoted_beta1 = PromoteScalar(at::Scalar(beta1));
  auto promoted_beta2 = PromoteScalar(at::Scalar(beta2));
  auto promoted_weight_decay = PromoteScalar(at::Scalar(weight_decay));
  auto promoted_eps = PromoteScalar(at::Scalar(eps));

  TT_KERNEL(
      OpName::kFusedAdamwTensorLr, param_keys,
      (self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr,
       promoted_beta1, promoted_beta2, promoted_weight_decay, promoted_eps,
       amsgrad, maximize, grad_scale, found_inf),
      {
        if (self.empty()) return;
        TT_CHECK_THROW(IsFloatingPoint(self[0]),
                       error::kPythonNotImplementedError)
            << "expected the input dtype to be floating-point, got "
            << ToString(self[0].scalar_type());
        const auto st = c10::toRealValueType(self[0].scalar_type());
        const auto hyperparam_st =
            (st == at::kDouble) ? at::kDouble : at::kFloat;
        TT_ASSIGN_OR_THROW(const at::Tensor b1_t,
                           promoted_beta1.GetTensor(hyperparam_st));
        TT_ASSIGN_OR_THROW(const at::Tensor b2_t,
                           promoted_beta2.GetTensor(hyperparam_st));
        TT_ASSIGN_OR_THROW(const at::Tensor wd_t,
                           promoted_weight_decay.GetTensor(hyperparam_st));
        TT_ASSIGN_OR_THROW(const at::Tensor eps_t,
                           promoted_eps.GetTensor(hyperparam_st));

        DispatchAdamw(
            self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
            lr, b1_t, b2_t, wd_t, eps_t,
            amsgrad ? AmsgradMode::kEnabled : AmsgradMode::kDisabled,
            maximize ? ObjectiveMode::kMaximize : ObjectiveMode::kMinimize,
            grad_scale, found_inf, std::move(param_keys));
      });
}

}  // namespace torch_tpu
