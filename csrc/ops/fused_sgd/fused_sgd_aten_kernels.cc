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

#include "csrc/ops/fused_sgd/fused_sgd_aten_kernels.h"

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

enum class NesterovMode : bool { kDisabled = false, kEnabled = true };
enum class SgdObjectiveMode : bool { kMinimize = false, kMaximize = true };
enum class IsFirstStepMode : bool { kFalse = false, kTrue = true };
enum class MomentumMode : bool { kDisabled = false, kEnabled = true };
enum class GradScaleMode : bool { kDisabled = false, kEnabled = true };
enum class FoundInfMode : bool { kDisabled = false, kEnabled = true };

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

struct OneParamResult {
  mlir::MlirOp param;
  std::optional<mlir::MlirOp> grad;
  std::optional<mlir::MlirOp> momentum_buffer;
};

// Reverts parameter, gradient, and moment tensors to pre-step values if inf/nan
// occurred.
absl::StatusOr<OneParamResult> ApplyFoundInfRevert(
    OneParamResult current, mlir::MlirOp param_old, mlir::MlirOp grad_old,
    std::optional<mlir::MlirOp> momentum_buffer_old, mlir::MlirOp found_inf,
    mlir::ElementType comp_type, const MomentumMode momentum_mode) {
  const bool has_momentum = (momentum_mode == MomentumMode::kEnabled);
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

  if (has_momentum && current.momentum_buffer.has_value() &&
      momentum_buffer_old.has_value()) {
    TT_ASSIGN_OR_RETURN(auto is_inf_mom,
                        BroadcastIfNeeded(is_inf, *current.momentum_buffer));
    *current.momentum_buffer = mlir::stablehlo::Select(
        is_inf_mom, *momentum_buffer_old, *current.momentum_buffer);
  }
  return current;
}

// Casts computed intermediate ops back to their respective target output buffer
// element types.
OneParamResult CastOneParamOutputs(
    OneParamResult res, mlir::ElementType param_dtype,
    mlir::ElementType grad_dtype,
    std::optional<mlir::ElementType> momentum_buffer_dtype,
    const MomentumMode momentum_mode) {
  const bool has_momentum = (momentum_mode == MomentumMode::kEnabled);
  if (GetElementTypeOrDie(res.param) != param_dtype) {
    res.param = mlir::stablehlo::ConvertElementType(res.param, param_dtype);
  }
  if (res.grad.has_value()) {
    if (GetElementTypeOrDie(*res.grad) != grad_dtype) {
      *res.grad = mlir::stablehlo::ConvertElementType(*res.grad, grad_dtype);
    }
  }
  if (has_momentum && res.momentum_buffer.has_value() &&
      momentum_buffer_dtype.has_value()) {
    if (GetElementTypeOrDie(*res.momentum_buffer) != *momentum_buffer_dtype) {
      *res.momentum_buffer = mlir::stablehlo::ConvertElementType(
          *res.momentum_buffer, *momentum_buffer_dtype);
    }
  }
  return res;
}

// Builds the StableHLO graph for updating a single parameter tensor and its
// corresponding optimizer state according to SGD.
absl::StatusOr<OneParamResult> BuildSgdStepForOneParam(
    mlir::MlirOp param, mlir::MlirOp grad,
    std::optional<mlir::MlirOp> momentum_buffer, mlir::MlirOp lr,
    mlir::MlirOp weight_decay, mlir::MlirOp momentum, mlir::MlirOp dampening,
    const NesterovMode nesterov_mode, const SgdObjectiveMode maximize_mode,
    const IsFirstStepMode is_first_step_mode,
    std::optional<mlir::MlirOp> grad_scale,
    std::optional<mlir::MlirOp> found_inf, mlir::ElementType param_dtype,
    mlir::ElementType grad_dtype,
    std::optional<mlir::ElementType> momentum_buffer_dtype,
    const MomentumMode momentum_mode) {
  const bool nesterov = (nesterov_mode == NesterovMode::kEnabled);
  const bool maximize = (maximize_mode == SgdObjectiveMode::kMaximize);
  const bool is_first_step = (is_first_step_mode == IsFirstStepMode::kTrue);
  const bool has_momentum = (momentum_mode == MomentumMode::kEnabled);

  // Determine shared computation precision
  TT_ASSIGN_OR_RETURN(const auto f32_dtype,
                      ConvertTo<mlir::ElementType>(c10::ScalarType::Float));
  TT_ASSIGN_OR_RETURN(const auto f64_dtype,
                      ConvertTo<mlir::ElementType>(c10::ScalarType::Double));
  mlir::ElementType comp_type = f32_dtype;
  if (param_dtype == f64_dtype) {
    comp_type = f64_dtype;
  }

  // Convert inputs and hyperparameters to the uniform computation element type.
  param = mlir::stablehlo::ConvertElementType(param, comp_type);
  grad = mlir::stablehlo::ConvertElementType(grad, comp_type);
  if (has_momentum && momentum_buffer.has_value()) {
    *momentum_buffer =
        mlir::stablehlo::ConvertElementType(*momentum_buffer, comp_type);
  }
  lr = mlir::stablehlo::ConvertElementType(lr, comp_type);
  weight_decay = mlir::stablehlo::ConvertElementType(weight_decay, comp_type);
  momentum = mlir::stablehlo::ConvertElementType(momentum, comp_type);
  dampening = mlir::stablehlo::ConvertElementType(dampening, comp_type);

  const mlir::MlirOp param_old = param;
  const mlir::MlirOp grad_old = grad;
  const std::optional<mlir::MlirOp> momentum_buffer_old = momentum_buffer;

  // Unscale gradients
  std::optional<mlir::MlirOp> out_grad;
  if (grad_scale.has_value()) {
    auto scale = mlir::stablehlo::ConvertElementType(*grad_scale, comp_type);
    TT_ASSIGN_OR_RETURN(auto scale_bcast, BroadcastIfNeeded(scale, grad));
    grad = mlir::stablehlo::Div(grad, scale_bcast);
    out_grad = grad;
  }
  if (maximize) {
    grad = mlir::stablehlo::Neg(grad);
  }

  // Weight decay
  // grad += weight_decay * param
  TT_ASSIGN_OR_RETURN(auto wd_bcast, BroadcastIfNeeded(weight_decay, param));
  auto wd_term = mlir::stablehlo::Mul(wd_bcast, param);
  grad = mlir::stablehlo::Add(grad, wd_term);

  // Momentum
  if (has_momentum && momentum_buffer.has_value()) {
    if (is_first_step) {
      *momentum_buffer = grad;
    } else {
      // momentum_buffer = momentum_buffer * momentum + grad * (1 - dampening)
      auto one_cst = MakeConstantLike(lr, 1.0);
      TT_ASSIGN_OR_RETURN(auto mom_bcast,
                          BroadcastIfNeeded(momentum, *momentum_buffer));
      auto mom_term1 = mlir::stablehlo::Mul(mom_bcast, *momentum_buffer);
      auto one_minus_damp = mlir::stablehlo::Subtract(one_cst, dampening);
      TT_ASSIGN_OR_RETURN(auto damp_bcast,
                          BroadcastIfNeeded(one_minus_damp, grad));
      auto mom_term2 = mlir::stablehlo::Mul(damp_bcast, grad);
      *momentum_buffer = mlir::stablehlo::Add(mom_term1, mom_term2);
    }
    if (nesterov) {
      // grad += momentum_buffer * momentum
      TT_ASSIGN_OR_RETURN(auto mom_bcast,
                          BroadcastIfNeeded(momentum, *momentum_buffer));
      auto nest_term = mlir::stablehlo::Mul(mom_bcast, *momentum_buffer);
      grad = mlir::stablehlo::Add(grad, nest_term);
    } else {
      grad = *momentum_buffer;
    }
  }

  // Final parameter update step
  // param = param - lr * grad
  TT_ASSIGN_OR_RETURN(auto lr_bcast, BroadcastIfNeeded(lr, grad));
  auto update = mlir::stablehlo::Mul(lr_bcast, grad);
  param = mlir::stablehlo::Subtract(param, update);

  OneParamResult res{param, out_grad, momentum_buffer};
  if (found_inf.has_value()) {
    TT_ASSIGN_OR_RETURN(
        res, ApplyFoundInfRevert(res, param_old, grad_old, momentum_buffer_old,
                                 *found_inf, comp_type, momentum_mode));
  }
  return CastOneParamOutputs(res, param_dtype, grad_dtype,
                             momentum_buffer_dtype, momentum_mode);
}

struct SgdInputSpans {
  absl::Span<mlir::MlirOp> params;
  absl::Span<mlir::MlirOp> grads;
  absl::Span<mlir::MlirOp> momentum_buffers;
  mlir::MlirOp weight_decay;
  mlir::MlirOp momentum;
  mlir::MlirOp lr;
  mlir::MlirOp dampening;
  std::optional<mlir::MlirOp> grad_scale;
  std::optional<mlir::MlirOp> found_inf;
};

// Extracts individual input spans and optional tensors from the flattened MLIR
// input operands.
SgdInputSpans ExtractSgdInputs(absl::Span<mlir::MlirOp> inputs,
                               const size_t num_tensors,
                               const MomentumMode momentum_mode,
                               const GradScaleMode grad_scale_mode,
                               const FoundInfMode found_inf_mode) {
  const bool has_momentum = (momentum_mode == MomentumMode::kEnabled);
  const bool has_grad_scale = (grad_scale_mode == GradScaleMode::kEnabled);
  const bool has_found_inf = (found_inf_mode == FoundInfMode::kEnabled);
  size_t offset = 0;
  auto params = inputs.subspan(offset, num_tensors);
  offset += num_tensors;
  auto grads = inputs.subspan(offset, num_tensors);
  offset += num_tensors;
  absl::Span<mlir::MlirOp> momentum_buffers;
  if (has_momentum) {
    momentum_buffers = inputs.subspan(offset, num_tensors);
    offset += num_tensors;
  }
  mlir::MlirOp weight_decay = inputs[offset++];
  mlir::MlirOp momentum = inputs[offset++];
  mlir::MlirOp lr = inputs[offset++];
  mlir::MlirOp dampening = inputs[offset++];
  std::optional<mlir::MlirOp> grad_scale;
  if (has_grad_scale) grad_scale = inputs[offset++];
  std::optional<mlir::MlirOp> found_inf;
  if (has_found_inf) found_inf = inputs[offset++];
  return SgdInputSpans{params,       grads,      momentum_buffers,
                       weight_decay, momentum,   lr,
                       dampening,    grad_scale, found_inf};
}

struct SgdDtypeSpans {
  absl::Span<const mlir::ElementType> params;
  absl::Span<const mlir::ElementType> grads;
  absl::Span<const mlir::ElementType> momentum_buffers;
};

// Extracts individual output element type spans for parameters, gradients, and
// momentum buffers.
SgdDtypeSpans ExtractSgdDtypes(absl::Span<const mlir::ElementType> out_dtypes,
                               const size_t num_tensors,
                               const MomentumMode momentum_mode,
                               const GradScaleMode grad_scale_mode) {
  const bool has_momentum = (momentum_mode == MomentumMode::kEnabled);
  const bool has_grad_scale = (grad_scale_mode == GradScaleMode::kEnabled);
  size_t offset = 0;
  auto params = out_dtypes.subspan(offset, num_tensors);
  offset += num_tensors;
  absl::Span<const mlir::ElementType> grads = params;
  if (has_grad_scale) {
    grads = out_dtypes.subspan(offset, num_tensors);
    offset += num_tensors;
  }
  absl::Span<const mlir::ElementType> momentum_buffers;
  if (has_momentum) {
    momentum_buffers = out_dtypes.subspan(offset, num_tensors);
    offset += num_tensors;
  }
  return SgdDtypeSpans{params, grads, momentum_buffers};
}

// Combines updated parameter, gradient, and momentum MLIR handles into a single
// flat result vector.
mlir::SmallVector<mlir::MlirOp> CombineSgdResults(
    const std::vector<mlir::MlirOp>& params,
    const std::vector<mlir::MlirOp>& grads,
    const std::vector<mlir::MlirOp>& momentum_buffers, const size_t num_tensors,
    const MomentumMode momentum_mode, const GradScaleMode grad_scale_mode) {
  const bool has_momentum = (momentum_mode == MomentumMode::kEnabled);
  const bool has_grad_scale = (grad_scale_mode == GradScaleMode::kEnabled);
  const size_t out_count = (has_momentum ? 2 : 1) + (has_grad_scale ? 1 : 0);
  mlir::SmallVector<mlir::MlirOp> results;
  results.reserve(out_count * num_tensors);
  results.insert(results.end(), params.begin(), params.end());
  if (has_grad_scale) {
    results.insert(results.end(), grads.begin(), grads.end());
  }
  if (has_momentum) {
    results.insert(results.end(), momentum_buffers.begin(),
                   momentum_buffers.end());
  }
  return results;
}

// Constructs the combined StableHLO graph for all tensors in the optimizer
// step, applying computation type promotion and unscaling.
absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> BuildSgdShlo(
    absl::Span<mlir::MlirOp> inputs, const size_t num_tensors,
    const MomentumMode momentum_mode, const NesterovMode nesterov_mode,
    const SgdObjectiveMode maximize_mode,
    const IsFirstStepMode is_first_step_mode,
    const GradScaleMode grad_scale_mode, const FoundInfMode found_inf_mode,
    absl::Span<const mlir::ElementType> out_dtypes) {
  const bool has_momentum = (momentum_mode == MomentumMode::kEnabled);
  const bool has_grad_scale = (grad_scale_mode == GradScaleMode::kEnabled);
  const auto in_spans = ExtractSgdInputs(inputs, num_tensors, momentum_mode,
                                         grad_scale_mode, found_inf_mode);
  const auto dt_spans =
      ExtractSgdDtypes(out_dtypes, num_tensors, momentum_mode, grad_scale_mode);

  std::vector<mlir::MlirOp> new_params;
  std::vector<mlir::MlirOp> new_grads;
  std::vector<mlir::MlirOp> new_momentum_buffers;
  new_params.reserve(num_tensors);
  if (has_grad_scale) new_grads.reserve(num_tensors);
  if (has_momentum) new_momentum_buffers.reserve(num_tensors);

  for (size_t i = 0; i < num_tensors; ++i) {
    std::optional<mlir::MlirOp> mom_buf;
    std::optional<mlir::ElementType> mom_buf_dt;
    if (has_momentum) {
      mom_buf = in_spans.momentum_buffers[i];
      mom_buf_dt = dt_spans.momentum_buffers[i];
    }
    TT_ASSIGN_OR_RETURN(
        auto res,
        BuildSgdStepForOneParam(
            in_spans.params[i], in_spans.grads[i], mom_buf, in_spans.lr,
            in_spans.weight_decay, in_spans.momentum, in_spans.dampening,
            nesterov_mode, maximize_mode, is_first_step_mode,
            in_spans.grad_scale, in_spans.found_inf, dt_spans.params[i],
            dt_spans.grads[i], mom_buf_dt, momentum_mode));
    new_params.push_back(res.param);
    if (has_grad_scale && res.grad.has_value()) {
      new_grads.push_back(*res.grad);
    }
    if (has_momentum && res.momentum_buffer.has_value()) {
      new_momentum_buffers.push_back(*res.momentum_buffer);
    }
  }

  return CombineSgdResults(new_params, new_grads, new_momentum_buffers,
                           num_tensors, momentum_mode, grad_scale_mode);
}

// Validates that gradient and momentum buffer lists match the parameter tensor
// list in size.
void ValidateSgdTensorListSizes(at::TensorList self, at::TensorList grads,
                                at::TensorList momentum_buffer_list,
                                const MomentumMode momentum_mode) {
  const bool has_momentum = (momentum_mode == MomentumMode::kEnabled);
  const size_t num_tensors = self.size();
  TT_CHECK_THROW(grads.size() == num_tensors, error::kInvalidArgument)
      << "expected grads to have the same number of tensors as self, got "
      << grads.size();
  if (has_momentum) {
    TT_CHECK_THROW(momentum_buffer_list.size() == num_tensors,
                   error::kInvalidArgument)
        << "expected momentum_buffer_list to have the same number of tensors "
           "as self, got "
        << momentum_buffer_list.size();
  }
}

// Determines whether momentum computation is enabled based on momentum buffer
// presence.
MomentumMode GetMomentumMode(at::TensorList momentum_buffer_list) {
  return momentum_buffer_list.empty() ? MomentumMode::kDisabled
                                      : MomentumMode::kEnabled;
}

// Determines whether gradient unscaling is enabled based on scale tensor
// presence.
GradScaleMode GetGradScaleMode(const std::optional<at::Tensor>& grad_scale) {
  return (grad_scale.has_value() && grad_scale->defined())
             ? GradScaleMode::kEnabled
             : GradScaleMode::kDisabled;
}

// Determines whether inf/nan checking and conditional reversion is enabled.
FoundInfMode GetFoundInfMode(const std::optional<at::Tensor>& found_inf) {
  return (found_inf.has_value() && found_inf->defined())
             ? FoundInfMode::kEnabled
             : FoundInfMode::kDisabled;
}

// Sets cache key parameters for optional optimizer features (grad scale,
// found_inf, momentum).
void SetSgdParamCacheKeys(OpParamCacheKeys& param_keys,
                          const MomentumMode momentum_mode,
                          const GradScaleMode grad_scale_mode,
                          const FoundInfMode found_inf_mode) {
  TT_THROW_IF_ERROR(param_keys.SetParam(
      "has_grad_scale", grad_scale_mode == GradScaleMode::kEnabled));
  TT_THROW_IF_ERROR(param_keys.SetParam(
      "has_found_inf", found_inf_mode == FoundInfMode::kEnabled));
  TT_THROW_IF_ERROR(param_keys.SetParam(
      "has_momentum", momentum_mode == MomentumMode::kEnabled));
}

// Gathers and flattens all input tensors and scalar hyperparameters into a
// single vector.
std::vector<at::Tensor> GatherSgdInputs(
    at::TensorList self, at::TensorList grads,
    at::TensorList momentum_buffer_list, const at::Tensor& weight_decay_tensor,
    const at::Tensor& momentum_tensor, const at::Tensor& lr_tensor,
    const at::Tensor& dampening_tensor,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf,
    const MomentumMode momentum_mode, const GradScaleMode grad_scale_mode,
    const FoundInfMode found_inf_mode) {
  const bool has_momentum = (momentum_mode == MomentumMode::kEnabled);
  const size_t num_tensors = self.size();
  std::vector<at::Tensor> inputs;
  inputs.reserve(has_momentum ? 3 * num_tensors + 6 : 2 * num_tensors + 6);
  inputs.insert(inputs.end(), self.begin(), self.end());
  inputs.insert(inputs.end(), grads.begin(), grads.end());
  if (has_momentum) {
    inputs.insert(inputs.end(), momentum_buffer_list.begin(),
                  momentum_buffer_list.end());
  }
  inputs.push_back(weight_decay_tensor);
  inputs.push_back(momentum_tensor);
  inputs.push_back(lr_tensor);
  inputs.push_back(dampening_tensor);

  if (grad_scale_mode == GradScaleMode::kEnabled && grad_scale.has_value()) {
    inputs.push_back(*grad_scale);
  }
  if (found_inf_mode == FoundInfMode::kEnabled && found_inf.has_value()) {
    inputs.push_back(*found_inf);
  }
  return inputs;
}

// Gathers output element types for parameters, gradients, and momentum buffers.
std::vector<mlir::ElementType> GatherSgdOutDtypes(
    at::TensorList self, at::TensorList grads,
    at::TensorList momentum_buffer_list, const MomentumMode momentum_mode,
    const GradScaleMode grad_scale_mode) {
  const bool has_momentum = (momentum_mode == MomentumMode::kEnabled);
  const bool has_grad_scale = (grad_scale_mode == GradScaleMode::kEnabled);
  const size_t num_tensors = self.size();
  const size_t out_count = (has_momentum ? 2 : 1) + (has_grad_scale ? 1 : 0);
  std::vector<mlir::ElementType> out_dtypes;
  out_dtypes.reserve(out_count * num_tensors);
  AppendDtypes(self, out_dtypes);
  if (has_grad_scale) {
    AppendDtypes(grads, out_dtypes);
  }
  if (has_momentum) {
    AppendDtypes(momentum_buffer_list, out_dtypes);
  }
  return out_dtypes;
}

// Gathers output tensor shapes for parameters, gradients, and momentum buffers.
std::vector<absl::Span<const int64_t>> GatherSgdOutDims(
    at::TensorList self, at::TensorList grads,
    at::TensorList momentum_buffer_list, const MomentumMode momentum_mode,
    const GradScaleMode grad_scale_mode) {
  const bool has_momentum = (momentum_mode == MomentumMode::kEnabled);
  const bool has_grad_scale = (grad_scale_mode == GradScaleMode::kEnabled);
  const size_t num_tensors = self.size();
  const size_t out_count = (has_momentum ? 2 : 1) + (has_grad_scale ? 1 : 0);
  std::vector<absl::Span<const int64_t>> out_dims_list;
  out_dims_list.reserve(out_count * num_tensors);
  AppendDims(self, out_dims_list);
  if (has_grad_scale) {
    AppendDims(grads, out_dims_list);
  }
  if (has_momentum) {
    AppendDims(momentum_buffer_list, out_dims_list);
  }
  return out_dims_list;
}

// Assigns compiled TPU device buffers back to ATen tensor inputs and optimizer
// states.
void AssignSgdResultBuffers(std::vector<DeviceBufferRef>& result_buffers,
                            at::TensorList self, at::TensorList grads,
                            at::TensorList momentum_buffer_list,
                            const MomentumMode momentum_mode,
                            const GradScaleMode grad_scale_mode) {
  const bool has_momentum = (momentum_mode == MomentumMode::kEnabled);
  const bool has_grad_scale = (grad_scale_mode == GradScaleMode::kEnabled);
  const size_t num_tensors = self.size();
  size_t buf_offset = 0;
  AssignBuffers(result_buffers, buf_offset, self);
  buf_offset += num_tensors;
  if (has_grad_scale) {
    AssignBuffers(result_buffers, buf_offset, grads);
    buf_offset += num_tensors;
  }
  if (has_momentum) {
    AssignBuffers(result_buffers, buf_offset, momentum_buffer_list);
  }
}

// Dispatches multi-tensor SGD optimizer computation to TPU runtime.
void DispatchSgd(at::TensorList self, at::TensorList grads,
                 at::TensorList momentum_buffer_list,
                 const at::Tensor& weight_decay_tensor,
                 const at::Tensor& momentum_tensor, const at::Tensor& lr_tensor,
                 const at::Tensor& dampening_tensor,
                 const NesterovMode nesterov_mode,
                 const SgdObjectiveMode maximize_mode,
                 const IsFirstStepMode is_first_step_mode,
                 const std::optional<at::Tensor>& grad_scale,
                 const std::optional<at::Tensor>& found_inf,
                 OpParamCacheKeys param_keys) {
  if (self.empty()) return;

  const MomentumMode momentum_mode = GetMomentumMode(momentum_buffer_list);
  const GradScaleMode grad_scale_mode = GetGradScaleMode(grad_scale);
  const FoundInfMode found_inf_mode = GetFoundInfMode(found_inf);

  ValidateSgdTensorListSizes(self, grads, momentum_buffer_list, momentum_mode);
  SetSgdParamCacheKeys(param_keys, momentum_mode, grad_scale_mode,
                       found_inf_mode);

  std::vector<at::Tensor> inputs = GatherSgdInputs(
      self, grads, momentum_buffer_list, weight_decay_tensor, momentum_tensor,
      lr_tensor, dampening_tensor, grad_scale, found_inf, momentum_mode,
      grad_scale_mode, found_inf_mode);
  std::vector<mlir::ElementType> out_dtypes = GatherSgdOutDtypes(
      self, grads, momentum_buffer_list, momentum_mode, grad_scale_mode);
  std::vector<absl::Span<const int64_t>> out_dims_list = GatherSgdOutDims(
      self, grads, momentum_buffer_list, momentum_mode, grad_scale_mode);

  const size_t num_tensors = self.size();
  auto op_builder = [num_tensors, momentum_mode, nesterov_mode, maximize_mode,
                     is_first_step_mode, grad_scale_mode, found_inf_mode,
                     out_dtypes](absl::Span<mlir::MlirOp> mlir_inputs,
                                 mlir::MlirBuilder&)
      -> absl::StatusOr<mlir::SmallVector<mlir::MlirOp>> {
    return BuildSgdShlo(mlir_inputs, num_tensors, momentum_mode, nesterov_mode,
                        maximize_mode, is_first_step_mode, grad_scale_mode,
                        found_inf_mode, out_dtypes);
  };

  const bool has_momentum = (momentum_mode == MomentumMode::kEnabled);
  const bool has_grad_scale = (grad_scale_mode == GradScaleMode::kEnabled);
  Indices donated_indices;
  donated_indices.reserve(
      (1 + (has_grad_scale ? 1 : 0) + (has_momentum ? 1 : 0)) * num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    donated_indices.push_back(i);  // self
    if (has_grad_scale) {
      donated_indices.push_back(1 * num_tensors + i);  // grads
    }
    if (has_momentum) {
      donated_indices.push_back(2 * num_tensors + i);  // momentum_buffer_list
    }
  }

  DispatchOpOptions<kDynamicSize> options = {
      .out_dtypes = out_dtypes,
      .out_dims_list = std::move(out_dims_list),
      .op_param_cache_keys = std::move(param_keys),
      .donated_indices = std::move(donated_indices),
  };

  TT_ASSIGN_OR_THROW(auto result_buffers,
                     (DispatchOp<kDynamicSize, kDynamicSize>(
                         std::move(op_builder), inputs, std::move(options))));

  AssignSgdResultBuffers(result_buffers, self, grads, momentum_buffer_list,
                         momentum_mode, grad_scale_mode);
}

}  // namespace

void AtenFusedSgd(at::TensorList self, at::TensorList grads,
                  at::TensorList momentum_buffer_list, double weight_decay,
                  double momentum, double lr, double dampening, bool nesterov,
                  bool maximize, bool is_first_step,
                  const std::optional<at::Tensor>& grad_scale,
                  const std::optional<at::Tensor>& found_inf) {
  auto promoted_weight_decay = PromoteScalar(at::Scalar(weight_decay));
  auto promoted_momentum = PromoteScalar(at::Scalar(momentum));
  auto promoted_lr = PromoteScalar(at::Scalar(lr));
  auto promoted_dampening = PromoteScalar(at::Scalar(dampening));

  TT_KERNEL(
      OpName::kFusedSgd, param_keys,
      (self, grads, momentum_buffer_list, promoted_weight_decay,
       promoted_momentum, promoted_lr, promoted_dampening, nesterov, maximize,
       is_first_step, grad_scale, found_inf),
      {
        if (self.empty()) return;
        TT_CHECK_THROW(IsFloatingPoint(self[0]),
                       error::kPythonNotImplementedError)
            << "expected the input dtype to be floating-point, got "
            << ToString(self[0].scalar_type());
        const auto st = c10::toRealValueType(self[0].scalar_type());
        const auto hyperparam_st =
            (st == at::kDouble) ? at::kDouble : at::kFloat;
        TT_ASSIGN_OR_THROW(const at::Tensor wd_t,
                           promoted_weight_decay.GetTensor(hyperparam_st));
        TT_ASSIGN_OR_THROW(const at::Tensor mom_t,
                           promoted_momentum.GetTensor(hyperparam_st));
        TT_ASSIGN_OR_THROW(const at::Tensor lr_t,
                           promoted_lr.GetTensor(hyperparam_st));
        TT_ASSIGN_OR_THROW(const at::Tensor damp_t,
                           promoted_dampening.GetTensor(hyperparam_st));

        DispatchSgd(
            self, grads, momentum_buffer_list, wd_t, mom_t, lr_t, damp_t,
            nesterov ? NesterovMode::kEnabled : NesterovMode::kDisabled,
            maximize ? SgdObjectiveMode::kMaximize
                     : SgdObjectiveMode::kMinimize,
            is_first_step ? IsFirstStepMode::kTrue : IsFirstStepMode::kFalse,
            grad_scale, found_inf, std::move(param_keys));
      });
}

void AtenFusedSgdTensorLr(at::TensorList self, at::TensorList grads,
                          at::TensorList momentum_buffer_list,
                          double weight_decay, double momentum,
                          const at::Tensor& lr, double dampening, bool nesterov,
                          bool maximize, bool is_first_step,
                          const std::optional<at::Tensor>& grad_scale,
                          const std::optional<at::Tensor>& found_inf) {
  auto promoted_weight_decay = PromoteScalar(at::Scalar(weight_decay));
  auto promoted_momentum = PromoteScalar(at::Scalar(momentum));
  auto promoted_dampening = PromoteScalar(at::Scalar(dampening));

  TT_KERNEL(
      OpName::kFusedSgdTensorLr, param_keys,
      (self, grads, momentum_buffer_list, promoted_weight_decay,
       promoted_momentum, lr, promoted_dampening, nesterov, maximize,
       is_first_step, grad_scale, found_inf),
      {
        if (self.empty()) return;
        TT_CHECK_THROW(IsFloatingPoint(self[0]),
                       error::kPythonNotImplementedError)
            << "expected the input dtype to be floating-point, got "
            << ToString(self[0].scalar_type());
        const auto st = c10::toRealValueType(self[0].scalar_type());
        const auto hyperparam_st =
            (st == at::kDouble) ? at::kDouble : at::kFloat;
        TT_ASSIGN_OR_THROW(const at::Tensor wd_t,
                           promoted_weight_decay.GetTensor(hyperparam_st));
        TT_ASSIGN_OR_THROW(const at::Tensor mom_t,
                           promoted_momentum.GetTensor(hyperparam_st));
        TT_ASSIGN_OR_THROW(const at::Tensor damp_t,
                           promoted_dampening.GetTensor(hyperparam_st));

        DispatchSgd(
            self, grads, momentum_buffer_list, wd_t, mom_t, lr, damp_t,
            nesterov ? NesterovMode::kEnabled : NesterovMode::kDisabled,
            maximize ? SgdObjectiveMode::kMaximize
                     : SgdObjectiveMode::kMinimize,
            is_first_step ? IsFirstStepMode::kTrue : IsFirstStepMode::kFalse,
            grad_scale, found_inf, std::move(param_keys));
      });
}

}  // namespace torch_tpu
