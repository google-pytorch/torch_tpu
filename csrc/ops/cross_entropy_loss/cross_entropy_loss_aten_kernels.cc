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

#include "csrc/ops/cross_entropy_loss/cross_entropy_loss_aten_kernels.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/Reduction.h"
#include "ATen/ops/cross_entropy_loss_native.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/GradMode.h"
#include "c10/util/Optional.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/binary.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/nll_loss/nll_loss.h"
#include "csrc/ops/nll_loss/nll_loss_aten_kernels.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/reductions/sum.h"
#include "csrc/ops/scatter/scatter.h"
#include "csrc/ops/softmax/softmax.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/csrc/autograd/custom_function.h"

namespace torch_tpu {
namespace {

enum class CrossEntropyTargetMode {
  kClassIndices,
  kProbabilities,
};

// Expands `grad_output` to match `self_type`'s shape `[N, C, d1, ..., dk]`
// after optional Mean scaling by `denom_scalar`.
absl::StatusOr<mlir::MlirOp> ExpandGradOutputToSelfShape(
    mlir::MlirOp grad_output, mlir::RankedTensorType self_type,
    int64_t reduction, std::optional<mlir::MlirOp> denom_scalar) {
  switch (static_cast<at::Reduction::Reduction>(reduction)) {
    case at::Reduction::None: {
      // grad_output has shape [N, d1, ..., dk] (or {} if unsqueezed from 1D).
      if (GetTensorTypeOrDie(grad_output).getRank() == 0) {
        TT_ASSIGN_OR_RETURN(grad_output, Unsqueeze(grad_output, 0));
      }
      TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_unsqueezed,
                          Unsqueeze(grad_output, 1));
      return BroadcastIfNeeded(grad_unsqueezed, self_type.getShape());
    }
    case at::Reduction::Sum: {
      if (GetTensorTypeOrDie(grad_output).getRank() != 0) {
        grad_output = mlir::stablehlo::Reshape(grad_output, {});
      }
      return BroadcastIfNeeded(grad_output, self_type.getShape());
    }
    case at::Reduction::Mean: {
      if (GetTensorTypeOrDie(grad_output).getRank() != 0) {
        grad_output = mlir::stablehlo::Reshape(grad_output, {});
      }
      mlir::MlirOp scaled_grad =
          mlir::stablehlo::Div(grad_output, denom_scalar.value());
      return BroadcastIfNeeded(scaled_grad, self_type.getShape());
    }
    default:
      return TT_ERROR(  // ERROR_COV_INFEASIBLE=Validated before calling
                        // builder.
                 error::kInvalidArgument)
             << "expected reduction to be one of ['none', 'mean', 'sum'], "
             << "got " << reduction;
  }
}

absl::StatusOr<mlir::MlirOp> RoundThroughOrigDtype(
    mlir::MlirOp op, mlir::ElementType orig_dtype,
    mlir::ElementType compute_dtype) {
  if (orig_dtype == compute_dtype) {
    return op;
  }
  TT_ASSIGN_OR_RETURN(op, CastIfNeeded(op, orig_dtype));
  return CastIfNeeded(op, compute_dtype);
}

absl::StatusOr<mlir::MlirOp> BuildEmptyCrossEntropyLossShlo(
    mlir::RankedTensorType orig_self_type, mlir::ElementType orig_self_dtype,
    int64_t reduction, mlir::MlirBuilder& builder) {
  switch (static_cast<at::Reduction::Reduction>(reduction)) {
    case at::Reduction::None: {
      Dimensions unreduced_shape;
      if (orig_self_type.getRank() >= 2) {
        unreduced_shape.reserve(orig_self_type.getRank() - 1);
        unreduced_shape.push_back(orig_self_type.getDimSize(0));
        for (int64_t i = 2; i < orig_self_type.getRank(); ++i) {
          unreduced_shape.push_back(orig_self_type.getDimSize(i));
        }
      }
      mlir::MlirOp zero_scalar =
          MakeScalarConstant(builder, 0.0, orig_self_dtype);
      const auto unreduced_type =
          mlir::RankedTensorType::get(AsArrayRef<int64_t>(unreduced_shape),
                                      orig_self_type.getElementType());
      return mlir::stablehlo::BroadcastInDim(unreduced_type, zero_scalar, {});
    }
    case at::Reduction::Sum:
      return MakeScalarConstant(builder, 0.0, orig_self_dtype);
    case at::Reduction::Mean:
      return MakeScalarConstant(
          builder, std::numeric_limits<double>::quiet_NaN(), orig_self_dtype);
    default:
      return TT_ERROR(  // ERROR_COV_INFEASIBLE=Validated before calling
                        // builder.
                 error::kInvalidArgument)
             << "expected reduction to be one of ['none', 'mean', 'sum'], "
             << "got " << reduction;
  }
}

absl::StatusOr<mlir::MlirOp> BuildCrossEntropyProbTargetShlo(
    mlir::MlirOp log_probs, mlir::MlirOp target,
    std::optional<mlir::MlirOp> weight, int64_t reduction,
    double label_smoothing, mlir::ElementType orig_self_dtype,
    mlir::ElementType compute_dtype, mlir::MlirBuilder& builder) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(log_probs);
  const int64_t n_classes = self_type.getDimSize(1);

  TT_ASSIGN_OR_RETURN(
      log_probs,
      RoundThroughOrigDtype(log_probs, orig_self_dtype, compute_dtype));
  TT_ASSIGN_OR_RETURN(target, CastIfNeeded(target, compute_dtype));
  if (label_smoothing > 0.0) {
    mlir::MlirOp one_minus_ls =
        MakeConstantLike(target, 1.0 - label_smoothing, compute_dtype);
    mlir::MlirOp ls_over_c = MakeConstantLike(
        target, label_smoothing / static_cast<double>(n_classes),
        compute_dtype);
    mlir::MlirOp scaled_target = mlir::stablehlo::Mul(target, one_minus_ls);
    TT_ASSIGN_OR_RETURN(
        scaled_target,
        RoundThroughOrigDtype(scaled_target, orig_self_dtype, compute_dtype));
    target = mlir::stablehlo::Add(scaled_target, ls_over_c);
    TT_ASSIGN_OR_RETURN(
        target, RoundThroughOrigDtype(target, orig_self_dtype, compute_dtype));
  }

  mlir::MlirOp term = mlir::stablehlo::Mul(log_probs, target);
  TT_ASSIGN_OR_RETURN(
      term, RoundThroughOrigDtype(term, orig_self_dtype, compute_dtype));
  if (weight.has_value()) {
    mlir::MlirOp weight_bcast =
        mlir::stablehlo::BroadcastInDim(self_type, weight.value(), {1});
    term = mlir::stablehlo::Mul(term, weight_bcast);
    TT_ASSIGN_OR_RETURN(
        term, RoundThroughOrigDtype(term, orig_self_dtype, compute_dtype));
  }

  TT_ASSIGN_OR_RETURN(mlir::MlirOp sum_class,
                      BuildSumShlo(term, {1}, ReductionMode::kDropDims));
  mlir::MlirOp loss_no_reduction = mlir::stablehlo::Neg(sum_class);

  switch (static_cast<at::Reduction::Reduction>(reduction)) {
    case at::Reduction::None:
      return loss_no_reduction;
    case at::Reduction::Sum:
      return BuildSumShlo(loss_no_reduction,
                          GetAllDimensions(loss_no_reduction),
                          ReductionMode::kDropDims);
    case at::Reduction::Mean: {
      TT_ASSIGN_OR_RETURN(
          mlir::MlirOp sum,
          BuildSumShlo(loss_no_reduction, GetAllDimensions(loss_no_reduction),
                       ReductionMode::kDropDims));
      const double num_samples = static_cast<double>(
          GetTensorTypeOrDie(loss_no_reduction).getNumElements());
      mlir::MlirOp denom =
          MakeScalarConstant(builder, num_samples, compute_dtype);
      return mlir::stablehlo::Div(sum, denom);
    }
    default:
      return TT_ERROR(  // ERROR_COV_INFEASIBLE=Validated before calling
                        // builder.
                 error::kInvalidArgument)
             << "expected reduction to be one of ['none', 'mean', 'sum'], "
             << "got " << reduction;
  }
}

absl::StatusOr<mlir::MlirOp> BuildCrossEntropyClassIndicesShlo(
    mlir::MlirOp log_probs, mlir::MlirOp target,
    std::optional<mlir::MlirOp> weight, int64_t reduction, int64_t ignore_index,
    double label_smoothing, mlir::ElementType orig_self_dtype,
    mlir::ElementType compute_dtype, mlir::MlirBuilder& builder) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(log_probs);
  const int64_t n_classes = self_type.getDimSize(1);

  TT_ASSIGN_OR_RETURN(mlir::MlirOp log_probs_orig,
                      CastIfNeeded(log_probs, orig_self_dtype));
  std::optional<mlir::MlirOp> weight_orig = weight;
  if (weight_orig.has_value()) {
    TT_ASSIGN_OR_RETURN(weight_orig,
                        CastIfNeeded(weight_orig.value(), orig_self_dtype));
  }
  TT_ASSIGN_OR_RETURN(auto nll_outputs, BuildNllLossForwardOutShlo(
                                            log_probs_orig, target, weight_orig,
                                            reduction, ignore_index, builder));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp nll_loss,
                      CastIfNeeded(nll_outputs[0], compute_dtype));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp total_weight,
                      CastIfNeeded(nll_outputs[1], compute_dtype));

  if (label_smoothing <= 0.0) {
    return nll_loss;
  }

  TT_ASSIGN_OR_RETURN(
      log_probs,
      RoundThroughOrigDtype(log_probs, orig_self_dtype, compute_dtype));
  mlir::MlirOp weighted_log_probs = log_probs;
  if (weight.has_value()) {
    mlir::MlirOp weight_bcast =
        mlir::stablehlo::BroadcastInDim(self_type, weight.value(), {1});
    TT_ASSIGN_OR_RETURN(
        weighted_log_probs,
        RoundThroughOrigDtype(mlir::stablehlo::Mul(log_probs, weight_bcast),
                              orig_self_dtype, compute_dtype));
  }
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp sum_class,
      BuildSumShlo(weighted_log_probs, {1}, ReductionMode::kDropDims));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp smooth_loss,
                      RoundThroughOrigDtype(mlir::stablehlo::Neg(sum_class),
                                            orig_self_dtype, compute_dtype));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp target_i64,
                      CastIfNeeded(target, mlir::ElementType::I64));
  mlir::MlirOp ignore_bcast = MakeConstantLike(target_i64, ignore_index);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mask, BuildNeShlo(target_i64, ignore_bcast));
  mlir::MlirOp zero_smooth = MakeConstantLike(smooth_loss, 0.0, compute_dtype);
  smooth_loss = mlir::stablehlo::Select(mask, smooth_loss, zero_smooth);

  mlir::MlirOp smooth_reduced;
  switch (static_cast<at::Reduction::Reduction>(reduction)) {
    case at::Reduction::None:
      smooth_reduced = smooth_loss;
      break;
    case at::Reduction::Sum: {
      TT_ASSIGN_OR_RETURN(
          smooth_reduced,
          BuildSumShlo(smooth_loss, GetAllDimensions(smooth_loss),
                       ReductionMode::kDropDims));
      TT_ASSIGN_OR_RETURN(smooth_reduced,
                          RoundThroughOrigDtype(smooth_reduced, orig_self_dtype,
                                                compute_dtype));
      break;
    }
    case at::Reduction::Mean: {
      TT_ASSIGN_OR_RETURN(
          mlir::MlirOp smooth_sum,
          BuildSumShlo(smooth_loss, GetAllDimensions(smooth_loss),
                       ReductionMode::kDropDims));
      TT_ASSIGN_OR_RETURN(
          smooth_sum,
          RoundThroughOrigDtype(smooth_sum, orig_self_dtype, compute_dtype));
      TT_ASSIGN_OR_RETURN(
          smooth_reduced,
          RoundThroughOrigDtype(mlir::stablehlo::Div(smooth_sum, total_weight),
                                orig_self_dtype, compute_dtype));
      break;
    }
    default:
      return TT_ERROR(  // ERROR_COV_INFEASIBLE=Validated before calling
                        // builder.
                 error::kInvalidArgument)
             << "expected reduction to be one of ['none', 'mean', 'sum'], "
             << "got " << reduction;
  }

  mlir::MlirOp one_minus_ls =
      MakeConstantLike(nll_loss, 1.0 - label_smoothing, compute_dtype);
  mlir::MlirOp ls_over_c = MakeConstantLike(
      smooth_reduced, label_smoothing / static_cast<double>(n_classes),
      compute_dtype);
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp nll_part,
      RoundThroughOrigDtype(mlir::stablehlo::Mul(nll_loss, one_minus_ls),
                            orig_self_dtype, compute_dtype));
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp smooth_part,
      RoundThroughOrigDtype(mlir::stablehlo::Mul(smooth_reduced, ls_over_c),
                            orig_self_dtype, compute_dtype));
  return mlir::stablehlo::Add(nll_part, smooth_part);
}

absl::StatusOr<mlir::MlirOp> BuildCrossEntropyProbTargetBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp self, mlir::MlirOp target,
    std::optional<mlir::MlirOp> weight, int64_t reduction,
    double label_smoothing, mlir::ElementType orig_self_dtype,
    mlir::ElementType compute_dtype, mlir::MlirBuilder& builder) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  const int64_t n_classes = self_type.getDimSize(1);

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp log_probs,
      BuildSoftmaxShlo(self, /*dim=*/1, SoftmaxMode::kLogSoftmax));
  TT_ASSIGN_OR_RETURN(
      log_probs,
      RoundThroughOrigDtype(log_probs, orig_self_dtype, compute_dtype));

  TT_ASSIGN_OR_RETURN(target, CastIfNeeded(target, compute_dtype));
  mlir::MlirOp tilde_y = target;
  if (label_smoothing > 0.0) {
    mlir::MlirOp one_minus_ls =
        MakeConstantLike(tilde_y, 1.0 - label_smoothing, compute_dtype);
    mlir::MlirOp ls_over_c = MakeConstantLike(
        tilde_y, label_smoothing / static_cast<double>(n_classes),
        compute_dtype);
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp scaled_tilde_y,
        RoundThroughOrigDtype(mlir::stablehlo::Mul(tilde_y, one_minus_ls),
                              orig_self_dtype, compute_dtype));
    TT_ASSIGN_OR_RETURN(
        tilde_y,
        RoundThroughOrigDtype(mlir::stablehlo::Add(scaled_tilde_y, ls_over_c),
                              orig_self_dtype, compute_dtype));
  }

  std::optional<mlir::MlirOp> denom_scalar;
  if (reduction == at::Reduction::Mean) {
    const double num_samples =
        static_cast<double>(self_type.getNumElements() / n_classes);
    denom_scalar = MakeScalarConstant(builder, num_samples, compute_dtype);
  }
  TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_expanded,
                      ExpandGradOutputToSelfShape(grad_output, self_type,
                                                  reduction, denom_scalar));
  TT_ASSIGN_OR_RETURN(
      grad_expanded,
      RoundThroughOrigDtype(grad_expanded, orig_self_dtype, compute_dtype));

  mlir::MlirOp grad_log_probs = mlir::stablehlo::Neg(grad_expanded);
  if (weight.has_value()) {
    mlir::MlirOp weight_bcast =
        mlir::stablehlo::BroadcastInDim(self_type, weight.value(), {1});
    TT_ASSIGN_OR_RETURN(grad_log_probs,
                        RoundThroughOrigDtype(
                            mlir::stablehlo::Mul(grad_log_probs, weight_bcast),
                            orig_self_dtype, compute_dtype));
  }
  TT_ASSIGN_OR_RETURN(
      grad_log_probs,
      RoundThroughOrigDtype(mlir::stablehlo::Mul(grad_log_probs, tilde_y),
                            orig_self_dtype, compute_dtype));

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp sum_grad,
      BuildSumShlo(grad_log_probs, {1}, ReductionMode::kKeepDims));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp sum_grad_bcast,
                      BroadcastIfNeeded(sum_grad, self_type.getShape()));
  mlir::MlirOp probs = mlir::stablehlo::Exp(log_probs);
  mlir::MlirOp probs_scaled = mlir::stablehlo::Mul(probs, sum_grad_bcast);
  return mlir::stablehlo::Subtract(grad_log_probs, probs_scaled);
}

absl::StatusOr<mlir::MlirOp> BuildCrossEntropyClassIndicesBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp probs, mlir::MlirOp target,
    std::optional<mlir::MlirOp> weight, int64_t reduction, int64_t ignore_index,
    double label_smoothing, mlir::ElementType compute_dtype) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(probs);
  const int64_t n_classes = self_type.getDimSize(1);

  TT_ASSIGN_OR_RETURN(target, CastIfNeeded(target, mlir::ElementType::I64));
  mlir::MlirOp ignore_index_bcast = MakeConstantLike(target, ignore_index);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mask,
                      BuildNeShlo(target, ignore_index_bcast));
  mlir::MlirOp zero_target = MakeConstantLike(target, 0);
  mlir::MlirOp safe_target = mlir::stablehlo::Select(mask, target, zero_target);

  mlir::MlirOp sample_weight;
  mlir::MlirOp weight_to_apply;
  if (weight.has_value()) {
    sample_weight = GatherWeightsShlo(weight.value(), safe_target);
    mlir::MlirOp zero_weight =
        MakeConstantLike(sample_weight, 0.0, compute_dtype);
    weight_to_apply = mlir::stablehlo::Select(mask, sample_weight, zero_weight);
  } else {
    sample_weight = MakeConstantLike(safe_target, 1.0, compute_dtype);
    weight_to_apply = mlir::stablehlo::ConvertElementType(mask, compute_dtype);
  }

  mlir::MlirOp grad_self = probs;
  if (weight.has_value()) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp sample_weight_unsqueeze,
                        Unsqueeze(sample_weight, 1));
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp sample_weight_bcast,
        BroadcastIfNeeded(sample_weight_unsqueeze, self_type.getShape()));
    grad_self = mlir::stablehlo::Mul(grad_self, sample_weight_bcast);
  }
  if (label_smoothing > 0.0) {
    mlir::MlirOp one_minus_ls =
        MakeConstantLike(grad_self, 1.0 - label_smoothing, compute_dtype);
    grad_self = mlir::stablehlo::Mul(grad_self, one_minus_ls);
  }

  mlir::MlirOp scatter_val = mlir::stablehlo::Neg(sample_weight);
  if (label_smoothing > 0.0) {
    mlir::MlirOp one_minus_ls =
        MakeConstantLike(scatter_val, 1.0 - label_smoothing, compute_dtype);
    scatter_val = mlir::stablehlo::Mul(scatter_val, one_minus_ls);
  }
  TT_ASSIGN_OR_RETURN(mlir::MlirOp target_unsqueeze, Unsqueeze(safe_target, 1));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp scatter_src, Unsqueeze(scatter_val, 1));
  TT_ASSIGN_OR_RETURN(
      grad_self, BuildScatterShlo(grad_self, /*dim=*/1, target_unsqueeze,
                                  scatter_src, ScatterOp::kAdd, compute_dtype));

  if (label_smoothing > 0.0) {
    mlir::MlirOp smooth_grad;
    if (weight.has_value()) {
      TT_ASSIGN_OR_RETURN(
          mlir::MlirOp weight_sum_scalar,
          BuildSumShlo(weight.value(), {0}, ReductionMode::kDropDims));
      mlir::MlirOp weight_sum_bcast =
          mlir::stablehlo::BroadcastInDim(self_type, weight_sum_scalar, {});
      mlir::MlirOp weight_bcast =
          mlir::stablehlo::BroadcastInDim(self_type, weight.value(), {1});
      mlir::MlirOp probs_weighted_sum =
          mlir::stablehlo::Mul(probs, weight_sum_bcast);
      smooth_grad = mlir::stablehlo::Subtract(probs_weighted_sum, weight_bcast);
      mlir::MlirOp ls_over_c = MakeConstantLike(
          smooth_grad, label_smoothing / static_cast<double>(n_classes),
          compute_dtype);
      smooth_grad = mlir::stablehlo::Mul(smooth_grad, ls_over_c);
    } else {
      mlir::MlirOp ls_const =
          MakeConstantLike(probs, label_smoothing, compute_dtype);
      mlir::MlirOp ls_over_c = MakeConstantLike(
          probs, label_smoothing / static_cast<double>(n_classes),
          compute_dtype);
      mlir::MlirOp probs_ls = mlir::stablehlo::Mul(probs, ls_const);
      smooth_grad = mlir::stablehlo::Subtract(probs_ls, ls_over_c);
    }
    grad_self = mlir::stablehlo::Add(grad_self, smooth_grad);
  }

  std::optional<mlir::MlirOp> denom_scalar;
  if (reduction == at::Reduction::Mean) {
    TT_ASSIGN_OR_RETURN(
        denom_scalar,
        BuildSumShlo(weight_to_apply, GetAllDimensions(weight_to_apply),
                     ReductionMode::kDropDims));
  }
  TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_expanded,
                      ExpandGradOutputToSelfShape(grad_output, self_type,
                                                  reduction, denom_scalar));
  grad_self = mlir::stablehlo::Mul(grad_self, grad_expanded);

  TT_ASSIGN_OR_RETURN(mlir::MlirOp mask_unsqueeze, Unsqueeze(mask, 1));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mask_bcast,
                      BroadcastIfNeeded(mask_unsqueeze, self_type.getShape()));
  mlir::MlirOp zero_self = MakeConstantLike(grad_self, 0.0, compute_dtype);
  return mlir::stablehlo::Select(mask_bcast, grad_self, zero_self);
}

absl::StatusOr<mlir::MlirOp> BuildCrossEntropyLossShlo(
    mlir::MlirOp self, mlir::MlirOp target, std::optional<mlir::MlirOp> weight,
    int64_t reduction, int64_t ignore_index, double label_smoothing,
    CrossEntropyTargetMode target_mode, mlir::MlirBuilder& builder) {
  const mlir::RankedTensorType orig_self_type = GetTensorTypeOrDie(self);
  const mlir::RankedTensorType orig_target_type = GetTensorTypeOrDie(target);

  TT_ASSIGN_OR_RETURN(const mlir::ElementType orig_self_dtype,
                      GetElementType(self));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(orig_self_dtype));

  if (orig_self_type.getNumElements() == 0 ||
      orig_target_type.getNumElements() == 0) {
    return BuildEmptyCrossEntropyLossShlo(orig_self_type, orig_self_dtype,
                                          reduction, builder);
  }

  bool unsqueezed = false;
  if (orig_self_type.getRank() == 1) {
    unsqueezed = true;
    TT_ASSIGN_OR_RETURN(self, Unsqueeze(self, 0));
    if (target_mode == CrossEntropyTargetMode::kProbabilities ||
        orig_target_type.getRank() == 0) {
      TT_ASSIGN_OR_RETURN(target, Unsqueeze(target, 0));
    }
  }

  TT_ASSIGN_OR_RETURN(self, CastIfNeeded(self, compute_dtype));
  if (weight.has_value()) {
    TT_ASSIGN_OR_RETURN(weight, CastIfNeeded(weight.value(), compute_dtype));
  }

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp log_probs,
      BuildSoftmaxShlo(self, /*dim=*/1, SoftmaxMode::kLogSoftmax));

  mlir::MlirOp loss_output;
  switch (target_mode) {
    case CrossEntropyTargetMode::kProbabilities: {
      TT_ASSIGN_OR_RETURN(loss_output, BuildCrossEntropyProbTargetShlo(
                                           log_probs, target, weight, reduction,
                                           label_smoothing, orig_self_dtype,
                                           compute_dtype, builder));
      break;
    }
    case CrossEntropyTargetMode::kClassIndices: {
      TT_ASSIGN_OR_RETURN(
          loss_output,
          BuildCrossEntropyClassIndicesShlo(
              log_probs, target, weight, reduction, ignore_index,
              label_smoothing, orig_self_dtype, compute_dtype, builder));
      break;
    }
  }

  if (unsqueezed && reduction == at::Reduction::None) {
    TT_ASSIGN_OR_RETURN(loss_output, Squeeze(loss_output, {0}));
  }

  return CastIfNeeded(loss_output, orig_self_dtype);
}

absl::StatusOr<mlir::MlirOp> BuildCrossEntropyLossBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp self, mlir::MlirOp target,
    std::optional<mlir::MlirOp> weight, int64_t reduction, int64_t ignore_index,
    double label_smoothing, CrossEntropyTargetMode target_mode,
    mlir::MlirBuilder& builder) {
  const mlir::RankedTensorType orig_self_type = GetTensorTypeOrDie(self);
  const mlir::RankedTensorType orig_target_type = GetTensorTypeOrDie(target);

  TT_ASSIGN_OR_RETURN(const mlir::ElementType orig_self_dtype,
                      GetElementType(self));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(orig_self_dtype));

  if (orig_self_type.getNumElements() == 0 ||
      orig_target_type.getNumElements() == 0) {
    return MakeConstantLike(self, 0.0, orig_self_dtype);
  }

  bool unsqueezed = false;
  if (orig_self_type.getRank() == 1) {
    unsqueezed = true;
    TT_ASSIGN_OR_RETURN(self, Unsqueeze(self, 0));
    if (target_mode == CrossEntropyTargetMode::kProbabilities ||
        orig_target_type.getRank() == 0) {
      TT_ASSIGN_OR_RETURN(target, Unsqueeze(target, 0));
    }
  }

  TT_ASSIGN_OR_RETURN(grad_output, CastIfNeeded(grad_output, compute_dtype));
  TT_ASSIGN_OR_RETURN(self, CastIfNeeded(self, compute_dtype));
  if (weight.has_value()) {
    TT_ASSIGN_OR_RETURN(weight, CastIfNeeded(weight.value(), compute_dtype));
  }

  mlir::MlirOp grad_self;
  switch (target_mode) {
    case CrossEntropyTargetMode::kProbabilities: {
      TT_ASSIGN_OR_RETURN(
          grad_self,
          BuildCrossEntropyProbTargetBackwardShlo(
              grad_output, self, target, weight, reduction, label_smoothing,
              orig_self_dtype, compute_dtype, builder));
      break;
    }
    case CrossEntropyTargetMode::kClassIndices: {
      TT_ASSIGN_OR_RETURN(
          mlir::MlirOp probs,
          BuildSoftmaxShlo(self, /*dim=*/1, SoftmaxMode::kSoftmax));
      TT_ASSIGN_OR_RETURN(grad_self,
                          BuildCrossEntropyClassIndicesBackwardShlo(
                              grad_output, probs, target, weight, reduction,
                              ignore_index, label_smoothing, compute_dtype));
      break;
    }
  }

  if (unsqueezed) {
    TT_ASSIGN_OR_RETURN(grad_self, Squeeze(grad_self, {0}));
  }

  return CastIfNeeded(grad_self, orig_self_dtype);
}

absl::StatusOr<mlir::MlirOp> BuildCrossEntropyLossBackwardTargetShlo(
    mlir::MlirOp grad_output, mlir::MlirOp self,
    std::optional<mlir::MlirOp> weight, int64_t reduction,
    double label_smoothing, mlir::MlirBuilder& builder) {
  const mlir::RankedTensorType orig_self_type = GetTensorTypeOrDie(self);

  TT_ASSIGN_OR_RETURN(const mlir::ElementType orig_self_dtype,
                      GetElementType(self));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(orig_self_dtype));

  if (orig_self_type.getNumElements() == 0) {
    return MakeConstantLike(self, 0.0, orig_self_dtype);
  }

  bool unsqueezed = false;
  if (orig_self_type.getRank() == 1) {
    unsqueezed = true;
    TT_ASSIGN_OR_RETURN(self, Unsqueeze(self, 0));
  }

  TT_ASSIGN_OR_RETURN(grad_output, CastIfNeeded(grad_output, compute_dtype));
  TT_ASSIGN_OR_RETURN(self, CastIfNeeded(self, compute_dtype));
  if (weight.has_value()) {
    TT_ASSIGN_OR_RETURN(weight, CastIfNeeded(weight.value(), compute_dtype));
  }

  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  const int64_t n_classes = self_type.getDimSize(1);

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp log_probs,
      BuildSoftmaxShlo(self, /*dim=*/1, SoftmaxMode::kLogSoftmax));
  TT_ASSIGN_OR_RETURN(
      log_probs,
      RoundThroughOrigDtype(log_probs, orig_self_dtype, compute_dtype));

  mlir::MlirOp grad_target = mlir::stablehlo::Neg(log_probs);
  if (label_smoothing > 0.0) {
    mlir::MlirOp one_minus_ls =
        MakeConstantLike(grad_target, 1.0 - label_smoothing, compute_dtype);
    grad_target = mlir::stablehlo::Mul(grad_target, one_minus_ls);
  }
  if (weight.has_value()) {
    mlir::MlirOp weight_bcast =
        mlir::stablehlo::BroadcastInDim(self_type, weight.value(), {1});
    grad_target = mlir::stablehlo::Mul(grad_target, weight_bcast);
  }

  std::optional<mlir::MlirOp> denom_scalar;
  if (reduction == at::Reduction::Mean) {
    const double num_samples =
        static_cast<double>(self_type.getNumElements() / n_classes);
    denom_scalar = MakeScalarConstant(builder, num_samples, compute_dtype);
  }
  TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_expanded,
                      ExpandGradOutputToSelfShape(grad_output, self_type,
                                                  reduction, denom_scalar));
  grad_target = mlir::stablehlo::Mul(grad_target, grad_expanded);

  if (unsqueezed) {
    TT_ASSIGN_OR_RETURN(grad_target, Squeeze(grad_target, {0}));
  }

  TT_ASSIGN_OR_RETURN(grad_target, CastIfNeeded(grad_target, orig_self_dtype));
  return grad_target;
}

absl::Status ValidateHardTargetShapes(const at::Tensor& self,
                                      const at::Tensor& target) {
  if (self.dim() == 1) {
    TT_RET_CHECK(target.dim() <= 1, error::kInvalidArgument)
        << "expected 0D or 1D target tensor, got " << target.dim() << "D";
    if (target.dim() == 1) {
      TT_RET_CHECK(target.size(0) == 1, error::kInvalidArgument)
          << "expected 1D target tensor to have size 1 for 1D input, got "
          << target.size(0);
    }
    return absl::OkStatus();
  }
  if (self.dim() == 2) {
    TT_RET_CHECK(target.dim() == 1, error::kInvalidArgument)
        << "expected 1D target tensor, got " << target.dim() << "D";
    TT_RET_CHECK(self.size(0) == target.size(0), error::kInvalidArgument)
        << "expected input batch_size (" << self.size(0)
        << ") to match target batch_size (" << target.size(0)
        << "), got input: " << ToString(self.sizes())
        << ", target: " << ToString(target.sizes());
    return absl::OkStatus();
  }
  TT_RET_CHECK(self.dim() == target.dim() + 1 &&
                   self.size(0) == target.size(0) &&
                   self.sizes().slice(2) == target.sizes().slice(1),
               error::kInvalidArgument)
      << "expected the shapes of the input [N, C, d1, ..., dk] and the "
         "target [N, d1, ..., dk] (k >= 1) to match, got input: "
      << ToString(self.sizes()) << ", target: " << ToString(target.sizes());
  return absl::OkStatus();
}

absl::Status ValidateCrossEntropyLoss(const at::Tensor& self,
                                      const at::Tensor& target,
                                      const c10::optional<at::Tensor>& weight,
                                      int64_t reduction, int64_t ignore_index,
                                      double label_smoothing) {
  TT_RET_CHECK(IsSupportedLossFloatType(self.scalar_type()),
               error::kInvalidArgument)
      << "unsupported input dtype: " << ToString(self.scalar_type());
  TT_RET_CHECK(self.dim() >= 1, error::kInvalidArgument)
      << "expected 1 or more dimensions for input, got " << self.dim();
  TT_RET_CHECK(reduction == at::Reduction::None ||
                   reduction == at::Reduction::Mean ||
                   reduction == at::Reduction::Sum,
               error::kInvalidArgument)
      << "expected reduction to be one of ['none', 'mean', 'sum'], got "
      << reduction;
  TT_RET_CHECK(label_smoothing >= 0.0 && label_smoothing <= 1.0,
               error::kInvalidArgument)
      << "expected label_smoothing to be between 0.0 and 1.0, got "
      << label_smoothing;

  const int64_t n_classes = self.size(self.dim() == 1 ? 0 : 1);
  if (weight.has_value() && weight->defined()) {
    TT_RET_CHECK(weight->dim() == 1 && weight->numel() == n_classes,
                 error::kInvalidArgument)
        << "expected weight tensor to be 1D with size " << n_classes << ", got "
        << ToString(weight->sizes());
  }

  const bool is_prob_target = (self.sizes() == target.sizes());
  if (is_prob_target) {
    TT_RET_CHECK(IsSupportedLossFloatType(target.scalar_type()),
                 error::kInvalidArgument)
        << "expected floating point type for target with class probabilities, "
           "got "
        << ToString(target.scalar_type());
    TT_RET_CHECK(ignore_index < 0, error::kInvalidArgument)
        << "expected ignore_index to be negative for floating point target, "
           "got "
        << ignore_index;
  } else {
    TT_RET_CHECK(IsLongOrByteTarget(target), error::kInvalidArgument)
        << "expected the target dtype to be either int64 or uint8, got "
        << ToString(target.scalar_type());
    TT_RETURN_IF_ERROR(ValidateHardTargetShapes(self, target));
  }
  return absl::OkStatus();
}

Dimensions GetCrossEntropyOutputDims(const at::Tensor& self,
                                     int64_t reduction) {
  Dimensions out_dims;
  if (reduction == at::Reduction::None && self.dim() >= 2) {
    out_dims.reserve(self.dim() - 1);
    out_dims.push_back(self.size(0));
    for (int64_t i = 2; i < self.dim(); ++i) {
      out_dims.push_back(self.size(i));
    }
  }
  return out_dims;
}

at::Tensor CrossEntropyLossBackwardSelfKernel(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const c10::optional<at::Tensor>& weight,
    int64_t reduction, int64_t ignore_index, double label_smoothing) {
  TT_KERNEL(
      OpName::kCrossEntropyLossBackward, param_keys,
      (grad_output, self, target, weight, reduction, ignore_index,
       label_smoothing),
      {
        TT_THROW_IF_ERROR(ValidateCrossEntropyLoss(
            self, target, weight, reduction, ignore_index, label_smoothing));
        const CrossEntropyTargetMode target_mode =
            (self.sizes() == target.sizes())
                ? CrossEntropyTargetMode::kProbabilities
                : CrossEntropyTargetMode::kClassIndices;
        const bool has_weight = weight.has_value() && weight->defined();

        auto op_builder =
            [reduction, ignore_index, label_smoothing, target_mode, has_weight](
                absl::Span<const mlir::MlirOp> inputs,
                mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
          mlir::MlirOp grad_output_op = inputs[0];
          mlir::MlirOp self_op = inputs[1];
          mlir::MlirOp target_op = inputs[2];
          std::optional<mlir::MlirOp> weight_op;
          if (has_weight) {
            weight_op = inputs[3];
          }
          return BuildCrossEntropyLossBackwardShlo(
              grad_output_op, self_op, target_op, weight_op, reduction,
              ignore_index, label_smoothing, target_mode, builder);
        };

        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        const Dimensions output_dims = CopyIntVector(self.sizes());

        std::vector<at::Tensor> inputs = {grad_output, self, target};
        if (has_weight) {
          inputs.push_back(weight.value());
        }

        DispatchOpOptions<1> options = {
            .out_dtype = output_dtype,
            .out_dims = output_dims,
            .op_param_cache_keys = std::move(param_keys),
        };

        TT_ASSIGN_OR_THROW(
            auto grad_self_buf,
            (DispatchOp<kDynamicSize, 1>(std::move(op_builder), inputs,
                                         std::move(options))));
        return MakeTensor(std::move(grad_self_buf));
      });
}

at::Tensor CrossEntropyLossBackwardTargetKernel(
    const at::Tensor& grad_output, const at::Tensor& self,
    const c10::optional<at::Tensor>& weight, int64_t reduction,
    double label_smoothing) {
  TT_KERNEL(
      OpName::kCrossEntropyLossBackward, param_keys,
      (grad_output, self, weight, reduction, label_smoothing), {
        const bool has_weight = weight.has_value() && weight->defined();
        auto op_builder =
            [reduction, label_smoothing, has_weight](
                absl::Span<const mlir::MlirOp> inputs,
                mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
          mlir::MlirOp grad_output_op = inputs[0];
          mlir::MlirOp self_op = inputs[1];
          std::optional<mlir::MlirOp> weight_op;
          if (has_weight) {
            weight_op = inputs[2];
          }
          return BuildCrossEntropyLossBackwardTargetShlo(
              grad_output_op, self_op, weight_op, reduction, label_smoothing,
              builder);
        };

        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        const Dimensions output_dims = CopyIntVector(self.sizes());

        std::vector<at::Tensor> inputs = {grad_output, self};
        if (has_weight) {
          inputs.push_back(weight.value());
        }

        DispatchOpOptions<1> options = {
            .out_dtype = output_dtype,
            .out_dims = output_dims,
            .op_param_cache_keys = std::move(param_keys),
        };

        TT_ASSIGN_OR_THROW(
            auto grad_target_buf,
            (DispatchOp<kDynamicSize, 1>(std::move(op_builder), inputs,
                                         std::move(options))));
        return MakeTensor(std::move(grad_target_buf));
      });
}

class CrossEntropyLossFunction
    : public torch::autograd::Function<CrossEntropyLossFunction> {
 public:
  static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                            const at::Tensor& self, const at::Tensor& target,
                            const c10::optional<at::Tensor>& weight,
                            int64_t reduction, int64_t ignore_index,
                            double label_smoothing) {
    at::AutoDispatchBelowAutograd guard;
    at::Tensor output = AtenCrossEntropyLoss(self, target, weight, reduction,
                                             ignore_index, label_smoothing);
    const at::Tensor weight_tensor =
        (weight.has_value() && weight->defined()) ? *weight : at::Tensor();
    ctx->save_for_backward({self, target, weight_tensor});
    ctx->saved_data["reduction"] = reduction;
    ctx->saved_data["ignore_index"] = ignore_index;
    ctx->saved_data["label_smoothing"] = label_smoothing;
    return output;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const at::Tensor& self = saved[0];
    const at::Tensor& target = saved[1];
    const at::Tensor& weight_tensor = saved[2];
    const int64_t reduction = ctx->saved_data["reduction"].toInt();
    const int64_t ignore_index = ctx->saved_data["ignore_index"].toInt();
    const double label_smoothing =
        ctx->saved_data["label_smoothing"].toDouble();

    const c10::optional<at::Tensor> weight =
        weight_tensor.defined() ? c10::optional<at::Tensor>(weight_tensor)
                                : c10::nullopt;
    at::Tensor grad_self;    // UNINITIALIZED_TENSOR_OK=Assigned conditionally
                             // or returned as undefined gradient.
    at::Tensor grad_target;  // UNINITIALIZED_TENSOR_OK=Assigned conditionally
                             // or returned as undefined gradient.
    if (ctx->needs_input_grad(0)) {
      grad_self = CrossEntropyLossBackwardSelfKernel(
          grad_outputs[0], self, target, weight, reduction, ignore_index,
          label_smoothing);
    }
    if (self.sizes() == target.sizes() && ctx->needs_input_grad(1)) {
      grad_target = CrossEntropyLossBackwardTargetKernel(
          grad_outputs[0], self, weight, reduction, label_smoothing);
    }
    return {grad_self,    grad_target,  at::Tensor(),
            at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

}  // namespace

at::Tensor AtenCrossEntropyLoss(const at::Tensor& self,
                                const at::Tensor& target,
                                const c10::optional<at::Tensor>& weight,
                                int64_t reduction, int64_t ignore_index,
                                double label_smoothing) {
  if (IsFakeOrFunctorchLossInputs(self, target, weight)) {
    return at::native::cross_entropy_loss_symint(
        self, target, weight, reduction, ignore_index, label_smoothing);
  }
  TT_KERNEL(
      OpName::kCrossEntropyLoss, param_keys,
      (self, target, weight, reduction, ignore_index, label_smoothing), {
        TT_THROW_IF_ERROR(ValidateCrossEntropyLoss(
            self, target, weight, reduction, ignore_index, label_smoothing));
        const CrossEntropyTargetMode target_mode =
            (self.sizes() == target.sizes())
                ? CrossEntropyTargetMode::kProbabilities
                : CrossEntropyTargetMode::kClassIndices;
        const bool has_weight = weight.has_value() && weight->defined();

        auto op_builder =
            [reduction, ignore_index, label_smoothing, target_mode, has_weight](
                absl::Span<const mlir::MlirOp> inputs,
                mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
          mlir::MlirOp self_op = inputs[0];
          mlir::MlirOp target_op = inputs[1];
          std::optional<mlir::MlirOp> weight_op;
          if (has_weight) {
            weight_op = inputs[2];
          }
          return BuildCrossEntropyLossShlo(
              self_op, target_op, weight_op, reduction, ignore_index,
              label_smoothing, target_mode, builder);
        };

        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        const Dimensions output_dims =
            GetCrossEntropyOutputDims(self, reduction);

        std::vector<at::Tensor> inputs = {self, target};
        if (has_weight) {
          inputs.push_back(weight.value());
        }

        DispatchOpOptions<1> options = {
            .out_dtype = output_dtype,
            .out_dims = output_dims,
            .op_param_cache_keys = std::move(param_keys),
        };

        TT_ASSIGN_OR_THROW(auto output_buf, (DispatchOp<kDynamicSize, 1>(
                                                std::move(op_builder), inputs,
                                                std::move(options))));
        return MakeTensor(std::move(output_buf));
      });
}

at::Tensor AtenCrossEntropyLossAutograd(const at::Tensor& self,
                                        const at::Tensor& target,
                                        const c10::optional<at::Tensor>& weight,
                                        int64_t reduction, int64_t ignore_index,
                                        double label_smoothing) {
  if (IsFakeOrFunctorchLossInputs(self, target, weight)) {
    return at::native::cross_entropy_loss_symint(
        self, target, weight, reduction, ignore_index, label_smoothing);
  }
  if (!c10::GradMode::is_enabled() ||
      (!self.requires_grad() && !target.requires_grad())) {
    at::AutoDispatchBelowAutograd guard;
    return AtenCrossEntropyLoss(self, target, weight, reduction, ignore_index,
                                label_smoothing);
  }
  return CrossEntropyLossFunction::apply(self, target, weight, reduction,
                                         ignore_index, label_smoothing);
}

}  // namespace torch_tpu
