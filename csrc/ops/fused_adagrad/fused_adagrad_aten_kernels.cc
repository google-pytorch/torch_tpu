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

#include "csrc/ops/fused_adagrad/fused_adagrad_aten_kernels.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
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

enum class AdagradObjectiveMode : bool { kMinimize = false, kMaximize = true };
enum class GradScaleMode : bool { kDisabled = false, kEnabled = true };
enum class FoundInfMode : bool { kDisabled = false, kEnabled = true };

// Determines whether gradient scaling is enabled from the optional tensor.
GradScaleMode GetGradScaleMode(const std::optional<at::Tensor>& grad_scale) {
  return (grad_scale.has_value() && grad_scale->defined())
             ? GradScaleMode::kEnabled
             : GradScaleMode::kDisabled;
}

// Determines whether inf/nan check is enabled from the optional tensor.
FoundInfMode GetFoundInfMode(const std::optional<at::Tensor>& found_inf) {
  return (found_inf.has_value() && found_inf->defined())
             ? FoundInfMode::kEnabled
             : FoundInfMode::kDisabled;
}

// Validates that param, grad, state sum, and step lists have matching sizes.
void ValidateAdagradTensorListSizes(at::TensorList self, at::TensorList grads,
                                    at::TensorList state_sums,
                                    at::TensorList state_steps) {
  TT_CHECK_THROW(self.size() == grads.size(), error::kInvalidArgument)
      << "expected grads to have size " << self.size() << ", got "
      << grads.size();
  TT_CHECK_THROW(self.size() == state_sums.size(), error::kInvalidArgument)
      << "expected state_sums to have size " << self.size() << ", got "
      << state_sums.size();
  TT_CHECK_THROW(self.size() == state_steps.size(), error::kInvalidArgument)
      << "expected state_steps to have size " << self.size() << ", got "
      << state_steps.size();
}

// Records gradient scaling and inf check flags into operation cache keys.
void SetAdagradParamCacheKeys(OpParamCacheKeys& param_keys,
                              const GradScaleMode grad_scale_mode,
                              const FoundInfMode found_inf_mode) {
  TT_THROW_IF_ERROR(param_keys.SetParam(
      "has_grad_scale", grad_scale_mode == GradScaleMode::kEnabled));
  TT_THROW_IF_ERROR(param_keys.SetParam(
      "has_found_inf", found_inf_mode == FoundInfMode::kEnabled));
}

// Assigns a slice of device buffers back to a list of PyTorch tensors.
void AssignBuffers(std::vector<DeviceBufferRef>& buffers, const size_t offset,
                   const at::TensorList tensors) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    TT_THROW_IF_ERROR(
        AssignBufferToAtTensor(std::move(buffers[offset + i]), tensors[i]));
  }
}

// Appends converted MLIR element types of tensors to the output list.
void AppendDtypes(const at::TensorList tensors,
                  std::vector<mlir::ElementType>& out_dtypes) {
  out_dtypes.reserve(out_dtypes.size() + tensors.size());
  for (const auto& t : tensors) {
    const auto st = c10::toRealValueType(t.scalar_type());
    TT_ASSIGN_OR_THROW(const auto dt, ConvertTo<mlir::ElementType>(st));
    out_dtypes.push_back(dt);
  }
}

// Appends tensor shape dimensions to the output dimensions list.
void AppendDims(const at::TensorList tensors,
                std::vector<absl::Span<const int64_t>>& out_dims) {
  out_dims.reserve(out_dims.size() + tensors.size());
  for (const auto& t : tensors) {
    out_dims.push_back(t.sizes());
  }
}

struct OneParamResult {
  mlir::MlirOp param;
  std::optional<mlir::MlirOp> grad;
  mlir::MlirOp state_sum;
};

// Reverts param, grad, and state sum to previous values if found_inf is true.
absl::StatusOr<OneParamResult> ApplyFoundInfRevert(
    OneParamResult current, mlir::MlirOp param_old, mlir::MlirOp grad_old,
    mlir::MlirOp state_sum_old, mlir::MlirOp found_inf,
    mlir::ElementType comp_type) {
  auto found_inf_comp =
      mlir::stablehlo::ConvertElementType(found_inf, comp_type);
  auto zero_inf = MakeConstantLike(found_inf_comp, 0.0);
  auto is_inf = mlir::stablehlo::Compare(
      found_inf_comp, zero_inf, mlir::stablehlo::ComparisonDirection::NE);

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

  TT_ASSIGN_OR_RETURN(auto is_inf_sum,
                      BroadcastIfNeeded(is_inf, current.state_sum));
  current.state_sum =
      mlir::stablehlo::Select(is_inf_sum, state_sum_old, current.state_sum);
  return current;
}

// Casts param, grad, and state sum MLIR ops back to target element types.
OneParamResult CastOneParamOutputs(OneParamResult res,
                                   mlir::ElementType param_dtype,
                                   mlir::ElementType grad_dtype,
                                   mlir::ElementType state_sum_dtype) {
  if (GetElementTypeOrDie(res.param) != param_dtype) {
    res.param = mlir::stablehlo::ConvertElementType(res.param, param_dtype);
  }
  if (res.grad.has_value()) {
    if (GetElementTypeOrDie(*res.grad) != grad_dtype) {
      *res.grad = mlir::stablehlo::ConvertElementType(*res.grad, grad_dtype);
    }
  }
  if (GetElementTypeOrDie(res.state_sum) != state_sum_dtype) {
    res.state_sum =
        mlir::stablehlo::ConvertElementType(res.state_sum, state_sum_dtype);
  }
  return res;
}

// Builds StableHLO MLIR computation graph for a single param Adagrad step.
absl::StatusOr<OneParamResult> BuildAdagradStepForOneParam(
    mlir::MlirOp param, mlir::MlirOp grad, mlir::MlirOp state_sum,
    mlir::MlirOp step, mlir::MlirOp lr, mlir::MlirOp lr_decay,
    mlir::MlirOp weight_decay, mlir::MlirOp eps,
    const AdagradObjectiveMode maximize_mode,
    std::optional<mlir::MlirOp> grad_scale,
    std::optional<mlir::MlirOp> found_inf, mlir::ElementType param_dtype,
    mlir::ElementType grad_dtype, mlir::ElementType state_sum_dtype) {
  const bool maximize = (maximize_mode == AdagradObjectiveMode::kMaximize);

  // Determine computation element type (f64 if any tensor is f64, else f32).
  TT_ASSIGN_OR_RETURN(const auto f32_dtype,
                      ConvertTo<mlir::ElementType>(c10::ScalarType::Float));
  TT_ASSIGN_OR_RETURN(const auto f64_dtype,
                      ConvertTo<mlir::ElementType>(c10::ScalarType::Double));
  mlir::ElementType comp_type = f32_dtype;
  if (param_dtype == f64_dtype || state_sum_dtype == f64_dtype) {
    comp_type = f64_dtype;
  }

  // Cast input tensors and hyperparameter scalars to computation element type.
  param = mlir::stablehlo::ConvertElementType(param, comp_type);
  grad = mlir::stablehlo::ConvertElementType(grad, comp_type);
  state_sum = mlir::stablehlo::ConvertElementType(state_sum, comp_type);
  step = mlir::stablehlo::ConvertElementType(step, comp_type);
  lr = mlir::stablehlo::ConvertElementType(lr, comp_type);
  lr_decay = mlir::stablehlo::ConvertElementType(lr_decay, comp_type);
  weight_decay = mlir::stablehlo::ConvertElementType(weight_decay, comp_type);
  eps = mlir::stablehlo::ConvertElementType(eps, comp_type);

  const mlir::MlirOp param_old = param;
  const mlir::MlirOp grad_old = grad;
  const mlir::MlirOp state_sum_old = state_sum;

  // Unscale gradients if a scaling factor was provided.
  std::optional<mlir::MlirOp> out_grad;
  if (grad_scale.has_value()) {
    auto scale = mlir::stablehlo::ConvertElementType(*grad_scale, comp_type);
    TT_ASSIGN_OR_RETURN(auto scale_bcast, BroadcastIfNeeded(scale, grad));
    grad = mlir::stablehlo::Div(grad, scale_bcast);
    out_grad = grad;
  }
  // Negate gradients for objective maximization if requested.
  if (maximize) {
    grad = mlir::stablehlo::Neg(grad);
  }

  // Apply coupled L2 weight decay to the gradients.
  TT_ASSIGN_OR_RETURN(auto wd_bcast, BroadcastIfNeeded(weight_decay, param));
  auto wd_term = mlir::stablehlo::Mul(wd_bcast, param);
  grad = mlir::stablehlo::Add(grad, wd_term);

  // Accumulate squared gradients into the state sum.
  auto grad_sq = mlir::stablehlo::Mul(grad, grad);
  state_sum = mlir::stablehlo::Add(state_sum, grad_sq);

  // Compute step-decayed learning rate based on current step and lr_decay.
  auto one_cst = MakeConstantLike(lr, 1.0);
  TT_ASSIGN_OR_RETURN(auto one_bcast_step, BroadcastIfNeeded(one_cst, step));
  auto step_minus_one = mlir::stablehlo::Subtract(step, one_bcast_step);
  TT_ASSIGN_OR_RETURN(auto lr_decay_bcast,
                      BroadcastIfNeeded(lr_decay, step_minus_one));
  auto decay_term = mlir::stablehlo::Mul(step_minus_one, lr_decay_bcast);
  auto denom_lr = mlir::stablehlo::Add(one_bcast_step, decay_term);
  TT_ASSIGN_OR_RETURN(auto lr_bcast, BroadcastIfNeeded(lr, denom_lr));
  auto clr = mlir::stablehlo::Div(lr_bcast, denom_lr);

  // Compute the Adagrad update step and subtract it from the parameter.
  auto sqrt_sum = mlir::stablehlo::Sqrt(state_sum);
  TT_ASSIGN_OR_RETURN(auto eps_bcast, BroadcastIfNeeded(eps, sqrt_sum));
  auto denom_std = mlir::stablehlo::Add(sqrt_sum, eps_bcast);
  TT_ASSIGN_OR_RETURN(auto clr_bcast_grad, BroadcastIfNeeded(clr, grad));
  auto update = mlir::stablehlo::Mul(clr_bcast_grad, grad);
  update = mlir::stablehlo::Div(update, denom_std);
  param = mlir::stablehlo::Subtract(param, update);

  OneParamResult res{param, out_grad, state_sum};
  // Revert parameter, gradient, and state sum if inf or nan was detected.
  if (found_inf.has_value()) {
    TT_ASSIGN_OR_RETURN(
        res, ApplyFoundInfRevert(res, param_old, grad_old, state_sum_old,
                                 *found_inf, comp_type));
  }
  return CastOneParamOutputs(res, param_dtype, grad_dtype, state_sum_dtype);
}

struct AdagradInputSpans {
  absl::Span<mlir::MlirOp> params;
  absl::Span<mlir::MlirOp> grads;
  absl::Span<mlir::MlirOp> state_sums;
  absl::Span<mlir::MlirOp> steps;
  mlir::MlirOp lr;
  mlir::MlirOp lr_decay;
  mlir::MlirOp weight_decay;
  mlir::MlirOp eps;
  std::optional<mlir::MlirOp> grad_scale;
  std::optional<mlir::MlirOp> found_inf;
};

// Extracts and slices individual input MLIR operands from the flat input span.
absl::StatusOr<AdagradInputSpans> ExtractAdagradInputs(
    absl::Span<mlir::MlirOp> inputs, const size_t num_tensors,
    const GradScaleMode grad_scale_mode, const FoundInfMode found_inf_mode) {
  const bool has_grad_scale = (grad_scale_mode == GradScaleMode::kEnabled);
  const bool has_found_inf = (found_inf_mode == FoundInfMode::kEnabled);
  size_t offset = 0;
  auto params = inputs.subspan(offset, num_tensors);
  offset += num_tensors;
  auto grads = inputs.subspan(offset, num_tensors);
  offset += num_tensors;
  auto state_sums = inputs.subspan(offset, num_tensors);
  offset += num_tensors;
  auto steps = inputs.subspan(offset, num_tensors);
  offset += num_tensors;
  auto lr = inputs[offset++];
  auto lr_decay = inputs[offset++];
  auto weight_decay = inputs[offset++];
  auto eps = inputs[offset++];
  std::optional<mlir::MlirOp> grad_scale;
  if (has_grad_scale) {
    grad_scale = inputs[offset++];
  }
  std::optional<mlir::MlirOp> found_inf;
  if (has_found_inf) {
    found_inf = inputs[offset++];
  }
  return AdagradInputSpans{params,     grads,    state_sums,   steps,
                           lr,         lr_decay, weight_decay, eps,
                           grad_scale, found_inf};
}

struct AdagradDtypeSpans {
  absl::Span<const mlir::ElementType> params;
  absl::Span<const mlir::ElementType> grads;
  absl::Span<const mlir::ElementType> state_sums;
};

// Extracts parameter, gradient, and state sum element types from output dtypes.
AdagradDtypeSpans ExtractAdagradDtypes(
    absl::Span<const mlir::ElementType> out_dtypes, const size_t num_tensors,
    const GradScaleMode grad_scale_mode) {
  const bool has_grad_scale = (grad_scale_mode == GradScaleMode::kEnabled);
  size_t offset = 0;
  auto params = out_dtypes.subspan(offset, num_tensors);
  offset += num_tensors;
  absl::Span<const mlir::ElementType> grads = params;
  if (has_grad_scale) {
    grads = out_dtypes.subspan(offset, num_tensors);
    offset += num_tensors;
  }
  auto state_sums = out_dtypes.subspan(offset, num_tensors);
  return AdagradDtypeSpans{params, grads, state_sums};
}

// Combines param, grad, and state sum results into a flat MLIR op vector.
mlir::SmallVector<mlir::MlirOp> CombineAdagradResults(
    const std::vector<mlir::MlirOp>& params,
    const std::vector<mlir::MlirOp>& grads,
    const std::vector<mlir::MlirOp>& state_sums, const size_t num_tensors,
    const GradScaleMode grad_scale_mode) {
  const bool has_grad_scale = (grad_scale_mode == GradScaleMode::kEnabled);
  const size_t out_count = 2 + (has_grad_scale ? 1 : 0);
  mlir::SmallVector<mlir::MlirOp> results;
  results.reserve(out_count * num_tensors);
  results.insert(results.end(), params.begin(), params.end());
  if (has_grad_scale) {
    results.insert(results.end(), grads.begin(), grads.end());
  }
  results.insert(results.end(), state_sums.begin(), state_sums.end());
  return results;
}

// Constructs the StableHLO computation graph for all parameter tensors.
absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> BuildAdagradShlo(
    absl::Span<mlir::MlirOp> inputs, const size_t num_tensors,
    const AdagradObjectiveMode maximize_mode,
    const GradScaleMode grad_scale_mode, const FoundInfMode found_inf_mode,
    absl::Span<const mlir::ElementType> out_dtypes) {
  const bool has_grad_scale = (grad_scale_mode == GradScaleMode::kEnabled);
  TT_ASSIGN_OR_RETURN(auto in_spans,
                      ExtractAdagradInputs(inputs, num_tensors, grad_scale_mode,
                                           found_inf_mode));
  auto dt_spans =
      ExtractAdagradDtypes(out_dtypes, num_tensors, grad_scale_mode);

  std::vector<mlir::MlirOp> new_params;
  std::vector<mlir::MlirOp> new_grads;
  std::vector<mlir::MlirOp> new_state_sums;
  new_params.reserve(num_tensors);
  if (has_grad_scale) new_grads.reserve(num_tensors);
  new_state_sums.reserve(num_tensors);

  // Build and collect Adagrad update computations for each param independently.
  for (size_t i = 0; i < num_tensors; ++i) {
    TT_ASSIGN_OR_RETURN(
        auto res,
        BuildAdagradStepForOneParam(
            in_spans.params[i], in_spans.grads[i], in_spans.state_sums[i],
            in_spans.steps[i], in_spans.lr, in_spans.lr_decay,
            in_spans.weight_decay, in_spans.eps, maximize_mode,
            in_spans.grad_scale, in_spans.found_inf, dt_spans.params[i],
            dt_spans.grads[i], dt_spans.state_sums[i]));

    new_params.push_back(res.param);
    if (has_grad_scale && res.grad.has_value()) {
      new_grads.push_back(*res.grad);
    }
    new_state_sums.push_back(res.state_sum);
  }
  return CombineAdagradResults(new_params, new_grads, new_state_sums,
                               num_tensors, grad_scale_mode);
}

// Flattens param, grad, state tensors, and scalars into one input vector.
std::vector<at::Tensor> GatherAdagradInputs(
    at::TensorList self, at::TensorList grads, at::TensorList state_sums,
    at::TensorList state_steps, const at::Tensor& lr,
    const at::Tensor& lr_decay, const at::Tensor& weight_decay,
    const at::Tensor& eps, const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf,
    const GradScaleMode grad_scale_mode, const FoundInfMode found_inf_mode) {
  const bool has_grad_scale = (grad_scale_mode == GradScaleMode::kEnabled);
  const bool has_found_inf = (found_inf_mode == FoundInfMode::kEnabled);
  const size_t num_tensors = self.size();
  const size_t scalar_count =
      4 + (has_grad_scale ? 1 : 0) + (has_found_inf ? 1 : 0);
  std::vector<at::Tensor> inputs;
  inputs.reserve(4 * num_tensors + scalar_count);

  inputs.insert(inputs.end(), self.begin(), self.end());
  inputs.insert(inputs.end(), grads.begin(), grads.end());
  inputs.insert(inputs.end(), state_sums.begin(), state_sums.end());
  inputs.insert(inputs.end(), state_steps.begin(), state_steps.end());

  inputs.push_back(lr);
  inputs.push_back(lr_decay);
  inputs.push_back(weight_decay);
  inputs.push_back(eps);

  if (has_grad_scale) {
    inputs.push_back(*grad_scale);
  }
  if (has_found_inf) {
    inputs.push_back(*found_inf);
  }
  return inputs;
}

// Gathers MLIR output element types for updated params, grads, and states.
std::vector<mlir::ElementType> GatherAdagradOutDtypes(
    at::TensorList self, at::TensorList grads, at::TensorList state_sums,
    const GradScaleMode grad_scale_mode) {
  const bool has_grad_scale = (grad_scale_mode == GradScaleMode::kEnabled);
  const size_t num_tensors = self.size();
  const size_t out_count = 2 + (has_grad_scale ? 1 : 0);
  std::vector<mlir::ElementType> out_dtypes;
  out_dtypes.reserve(out_count * num_tensors);
  AppendDtypes(self, out_dtypes);
  if (has_grad_scale) {
    AppendDtypes(grads, out_dtypes);
  }
  AppendDtypes(state_sums, out_dtypes);
  return out_dtypes;
}

// Gathers output shape dimensions for updated params, grads, and states.
std::vector<absl::Span<const int64_t>> GatherAdagradOutDims(
    at::TensorList self, at::TensorList grads, at::TensorList state_sums,
    const GradScaleMode grad_scale_mode) {
  const bool has_grad_scale = (grad_scale_mode == GradScaleMode::kEnabled);
  const size_t num_tensors = self.size();
  const size_t out_count = 2 + (has_grad_scale ? 1 : 0);
  std::vector<absl::Span<const int64_t>> out_dims_list;
  out_dims_list.reserve(out_count * num_tensors);
  AppendDims(self, out_dims_list);
  if (has_grad_scale) {
    AppendDims(grads, out_dims_list);
  }
  AppendDims(state_sums, out_dims_list);
  return out_dims_list;
}

// Assigns device output buffers back to ATen param, grad, and sum tensors.
void AssignAdagradResultBuffers(std::vector<DeviceBufferRef>& result_buffers,
                                at::TensorList self, at::TensorList grads,
                                at::TensorList state_sums,
                                const GradScaleMode grad_scale_mode) {
  const bool has_grad_scale = (grad_scale_mode == GradScaleMode::kEnabled);
  const size_t num_tensors = self.size();
  size_t buf_offset = 0;
  AssignBuffers(result_buffers, buf_offset, self);
  buf_offset += num_tensors;
  if (has_grad_scale) {
    AssignBuffers(result_buffers, buf_offset, grads);
    buf_offset += num_tensors;
  }
  AssignBuffers(result_buffers, buf_offset, state_sums);
}

// Dispatches fused Adagrad computation graph to TPU device for execution.
void DispatchAdagrad(
    at::TensorList self, at::TensorList grads, at::TensorList state_sums,
    at::TensorList state_steps, const at::Tensor& lr_tensor,
    const at::Tensor& lr_decay_tensor, const at::Tensor& weight_decay_tensor,
    const at::Tensor& eps_tensor, const AdagradObjectiveMode maximize_mode,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf, OpParamCacheKeys param_keys) {
  if (self.empty()) return;

  const GradScaleMode grad_scale_mode = GetGradScaleMode(grad_scale);
  const FoundInfMode found_inf_mode = GetFoundInfMode(found_inf);

  ValidateAdagradTensorListSizes(self, grads, state_sums, state_steps);
  SetAdagradParamCacheKeys(param_keys, grad_scale_mode, found_inf_mode);

  // Gather input tensors, output dtypes, and shape dimensions for dispatch.
  std::vector<at::Tensor> inputs = GatherAdagradInputs(
      self, grads, state_sums, state_steps, lr_tensor, lr_decay_tensor,
      weight_decay_tensor, eps_tensor, grad_scale, found_inf, grad_scale_mode,
      found_inf_mode);
  std::vector<mlir::ElementType> out_dtypes =
      GatherAdagradOutDtypes(self, grads, state_sums, grad_scale_mode);
  std::vector<absl::Span<const int64_t>> out_dims_list =
      GatherAdagradOutDims(self, grads, state_sums, grad_scale_mode);

  const size_t num_tensors = self.size();
  auto op_builder =
      [num_tensors, maximize_mode, grad_scale_mode, found_inf_mode, out_dtypes](
          absl::Span<mlir::MlirOp> mlir_inputs, mlir::MlirBuilder&)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    return BuildAdagradShlo(mlir_inputs, num_tensors, maximize_mode,
                            grad_scale_mode, found_inf_mode, out_dtypes);
  };

  const bool has_grad_scale = (grad_scale_mode == GradScaleMode::kEnabled);
  Indices donated_indices;
  donated_indices.reserve((has_grad_scale ? 3 : 2) * num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    donated_indices.push_back(i);  // self
    if (has_grad_scale) {
      donated_indices.push_back(1 * num_tensors + i);  // grads
    }
    donated_indices.push_back(2 * num_tensors + i);  // state_sums
  }

  // Dispatch op builder and input tensors to TPU device and get buffers.
  TT_ASSIGN_OR_THROW(auto result_buffers,
                     (DispatchOp<kDynamicSize, kDynamicSize>(
                         std::move(op_builder), inputs,
                         {.out_dtypes = out_dtypes,
                          .out_dims_list = out_dims_list,
                          .op_param_cache_keys = std::move(param_keys),
                          .donated_indices = std::move(donated_indices)})));

  // Assign computed output device buffers back to the input ATen tensors.
  AssignAdagradResultBuffers(result_buffers, self, grads, state_sums,
                             grad_scale_mode);
}

}  // namespace

void AtenFusedAdagrad(at::TensorList self, at::TensorList grads,
                      at::TensorList state_sums, at::TensorList state_steps,
                      double lr, double lr_decay, double weight_decay,
                      double eps, bool maximize,
                      const std::optional<at::Tensor>& grad_scale,
                      const std::optional<at::Tensor>& found_inf) {
  auto promoted_lr = PromoteScalar(at::Scalar(lr));
  auto promoted_lr_decay = PromoteScalar(at::Scalar(lr_decay));
  auto promoted_weight_decay = PromoteScalar(at::Scalar(weight_decay));
  auto promoted_eps = PromoteScalar(at::Scalar(eps));

  TT_KERNEL(
      OpName::kFusedAdagrad, param_keys,
      (self, grads, state_sums, state_steps, promoted_lr, promoted_lr_decay,
       promoted_weight_decay, promoted_eps, maximize, grad_scale, found_inf),
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
        TT_ASSIGN_OR_THROW(const at::Tensor lrd_t,
                           promoted_lr_decay.GetTensor(hyperparam_st));
        TT_ASSIGN_OR_THROW(const at::Tensor wd_t,
                           promoted_weight_decay.GetTensor(hyperparam_st));
        TT_ASSIGN_OR_THROW(const at::Tensor eps_t,
                           promoted_eps.GetTensor(hyperparam_st));

        DispatchAdagrad(self, grads, state_sums, state_steps, lr_t, lrd_t, wd_t,
                        eps_t,
                        maximize ? AdagradObjectiveMode::kMaximize
                                 : AdagradObjectiveMode::kMinimize,
                        grad_scale, found_inf, std::move(param_keys));
      });
}

void AtenFusedAdagradTensorLr(at::TensorList self, at::TensorList grads,
                              at::TensorList state_sums,
                              at::TensorList state_steps, const at::Tensor& lr,
                              double lr_decay, double weight_decay, double eps,
                              bool maximize,
                              const std::optional<at::Tensor>& grad_scale,
                              const std::optional<at::Tensor>& found_inf) {
  auto promoted_lr_decay = PromoteScalar(at::Scalar(lr_decay));
  auto promoted_weight_decay = PromoteScalar(at::Scalar(weight_decay));
  auto promoted_eps = PromoteScalar(at::Scalar(eps));

  TT_KERNEL(
      OpName::kFusedAdagradTensorLr, param_keys,
      (self, grads, state_sums, state_steps, lr, promoted_lr_decay,
       promoted_weight_decay, promoted_eps, maximize, grad_scale, found_inf),
      {
        if (self.empty()) return;
        TT_CHECK_THROW(IsFloatingPoint(self[0]),
                       error::kPythonNotImplementedError)
            << "expected the input dtype to be floating-point, got "
            << ToString(self[0].scalar_type());
        const auto st = c10::toRealValueType(self[0].scalar_type());
        const auto hyperparam_st =
            (st == at::kDouble) ? at::kDouble : at::kFloat;
        TT_ASSIGN_OR_THROW(const at::Tensor lrd_t,
                           promoted_lr_decay.GetTensor(hyperparam_st));
        TT_ASSIGN_OR_THROW(const at::Tensor wd_t,
                           promoted_weight_decay.GetTensor(hyperparam_st));
        TT_ASSIGN_OR_THROW(const at::Tensor eps_t,
                           promoted_eps.GetTensor(hyperparam_st));

        DispatchAdagrad(self, grads, state_sums, state_steps, lr, lrd_t, wd_t,
                        eps_t,
                        maximize ? AdagradObjectiveMode::kMaximize
                                 : AdagradObjectiveMode::kMinimize,
                        grad_scale, found_inf, std::move(param_keys));
      });
}

}  // namespace torch_tpu
