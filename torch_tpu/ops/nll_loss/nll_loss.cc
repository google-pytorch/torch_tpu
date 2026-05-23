// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/ops/nll_loss/nll_loss.h"

#include <cstdint>
#include <optional>

#include "ATen/core/Reduction.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/binary.h"
#include "torch_tpu/ops/gather/gather.h"
#include "torch_tpu/ops/index_select/index_select.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"
#include "torch_tpu/ops/reductions/sum.h"
#include "torch_tpu/ops/scatter/scatter.h"

namespace torch_tpu {

absl::StatusOr<MlirOpResults<2>> BuildNllLossForwardOutShlo(
    mlir::MlirOp self, mlir::MlirOp target, std::optional<mlir::MlirOp> weight,
    int64_t reduction, int64_t ignore_index, mlir::MlirBuilder& builder) {
  mlir::RankedTensorType self_type_before_unsqueeze = GetTensorTypeOrDie(self);
  bool unsqueezed = false;
  if (self_type_before_unsqueeze.getRank() == 1) {
    unsqueezed = true;
    TT_ASSIGN_OR_RETURN(self, Unsqueeze(self, 0));
    TT_ASSIGN_OR_RETURN(target, Unsqueeze(target, 0));
  }

  // Now, we have the following Shapes.
  // self: [batch, num_classes]
  // target: [batch]
  // weight: [num_classes]
  // reduction: a scalar
  // ignore_index: a scalar

  TT_ASSIGN_OR_RETURN(mlir::ElementType self_dtype, GetElementType(self));

  // mask = target != ignore_index
  // mask is of shape [batch] with 0s and 1s.
  // 0s are for ignoring samples with ignore_index.
  mlir::MlirOp ignore_index_bcast = MakeConstantLike(target, ignore_index);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mask,
                      BuildNeShlo(target, ignore_index_bcast));
  mlir::MlirOp mask_float =
      mlir::stablehlo::ConvertElementType(mask, self_dtype);

  // loss = - self[target]
  TT_ASSIGN_OR_RETURN(mlir::MlirOp target_unsqueeze, Unsqueeze(target, 1));
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp gathered_unsqueeze,
      BuildGatherShlo(self, 1, target_unsqueeze, false, self_dtype));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp gathered, Squeeze(gathered_unsqueeze, {1}));
  // loss_no_reduction is of shape [batch], which contains the loss for each
  // sample.
  mlir::MlirOp loss_no_reduction = mlir::stablehlo::Neg(gathered);

  // Mask the weights and apply to loss.
  mlir::MlirOp weight_to_apply = mask_float;
  if (weight.has_value()) {
    mlir::MlirOp sample_weight =
        BuildIndexSelectShlo(weight.value(), 0, target);
    weight_to_apply = mlir::stablehlo::Mul(weight_to_apply, sample_weight);
  }
  loss_no_reduction = mlir::stablehlo::Mul(loss_no_reduction, weight_to_apply);

  mlir::MlirOp total_weight_scalar;
  TT_ASSIGN_OR_RETURN(
      total_weight_scalar,
      BuildSumShlo(weight_to_apply, GetAllDimensions(weight_to_apply),
                   ReductionMode::kDropDims,
                   /*element_type_opt=*/std::nullopt));

  mlir::MlirOp loss_output;
  if (reduction == at::Reduction::None) {
    loss_output = loss_no_reduction;
  } else {
    TT_ASSIGN_OR_RETURN(
        mlir::MlirOp sum,
        BuildSumShlo(loss_no_reduction, GetAllDimensions(loss_no_reduction),
                     ReductionMode::kDropDims,
                     /*element_type_opt=*/std::nullopt));
    if (reduction == at::Reduction::Sum) {
      loss_output = sum;
    } else if (reduction == at::Reduction::Mean) {
      loss_output = mlir::stablehlo::Div(sum, total_weight_scalar);
    } else {
      return TT_ERROR(  // ERROR_COV_INFEASIBLE=PyTorch checks this before
                        // calling this function. This is just a fall back if
                        // they changed the implementation.
                 error::kInvalidArgument)
             << "expected reduction to be one of ['none', 'mean', 'sum'], "
             << "got " << reduction;
    }
  }

  if (unsqueezed && reduction == at::Reduction::None) {
    TT_ASSIGN_OR_RETURN(loss_output, Squeeze(loss_output, {0}));
  }
  return {{loss_output, total_weight_scalar}};
}

absl::StatusOr<mlir::MlirOp> BuildNllLossBackwardGradInputShlo(
    mlir::MlirOp grad_output, mlir::MlirOp self, mlir::MlirOp target,
    std::optional<mlir::MlirOp> weight, int64_t reduction, int64_t ignore_index,
    mlir::MlirOp total_weight, mlir::MlirBuilder& builder) {
  mlir::RankedTensorType self_type_before_unsqueeze = GetTensorTypeOrDie(self);
  // Unsqueeze if it is a single sample without batch dimension.
  bool unsqueezed = false;
  if (self_type_before_unsqueeze.getRank() == 1) {
    unsqueezed = true;
    TT_ASSIGN_OR_RETURN(self, Unsqueeze(self, 0));
    TT_ASSIGN_OR_RETURN(target, Unsqueeze(target, 0));
    if (reduction == at::Reduction::None) {
      TT_ASSIGN_OR_RETURN(grad_output, Unsqueeze(grad_output, 0));
    }
  }

  // Now, we have the following Shapes.
  // grad_output: [batch] for none,or a scalar for sum and mean
  // self: [batch, num_classes]
  // target: [batch]
  // weight: [num_classes]
  // reduction: a scalar
  // ignore_index: a scalar
  // total_weight: a scalar

  // Define the types.
  TT_ASSIGN_OR_RETURN(mlir::ElementType self_dtype, GetElementType(self));
  TT_ASSIGN_OR_RETURN(mlir::ElementType grad_output_dtype,
                      GetElementType(grad_output));
  mlir::RankedTensorType target_type = GetTensorTypeOrDie(target);

  // Initialize grad_input to zeros of shape [batch, num_classes].
  // Use self shape since the grad is w.r.t. self.
  // Use grad_output dtype to keep grad dtype consistent.
  mlir::MlirOp grad_input = MakeConstantLike(self, 0, grad_output_dtype);

  // mask is of shape [batch] with 0s and 1s.
  // 0s are for ignoring samples with ignore_index.
  mlir::MlirOp ignore_index_bcast = MakeConstantLike(target, ignore_index);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mask,
                      BuildNeShlo(target, ignore_index_bcast));
  mask = mlir::stablehlo::ConvertElementType(mask, grad_output_dtype);

  // Initialize grad_val to -grad_output no matter it's a scalar or [batch].
  mlir::MlirOp grad_val = mlir::stablehlo::Neg(grad_output);
  if (reduction != at::Reduction::None) {
    // The reduction is mean or sum.
    // grad_val now is a scalar.
    if (reduction == at::Reduction::Mean) {
      grad_val = mlir::stablehlo::Div(grad_val, total_weight);
    }

    // Broadcast grad_val to [batch].
    TT_ASSIGN_OR_RETURN(grad_val,
                        BroadcastIfNeeded(grad_val, target_type.getShape()));
  }

  // Now grad_val is of shape [batch].
  // Apply the mask to grad_val.
  grad_val = mlir::stablehlo::Mul(grad_val, mask);

  // Apply sample weight to grad_val.
  if (weight.has_value()) {
    mlir::MlirOp sample_weight =
        BuildIndexSelectShlo(weight.value(), 0, target);
    grad_val = mlir::stablehlo::Mul(grad_val, sample_weight);
  }

  // Unsqueeze to prepare for scatter.
  // target and grad_val become of shape [batch, 1].
  TT_ASSIGN_OR_RETURN(target, Unsqueeze(target, 1));
  TT_ASSIGN_OR_RETURN(grad_val, Unsqueeze(grad_val, 1));

  // Apply scatter_add to grad_input.
  // grad_input[i, target[i]] += grad_val[i, 0]
  TT_ASSIGN_OR_RETURN(grad_input,
                      BuildScatterShlo(grad_input, 1, target, grad_val,
                                       ScatterOp::kAdd, self_dtype));

  // Squeeze the batch dimension if it was added earlier.
  if (unsqueezed) {
    auto grad_input_type = GetTensorTypeOrDie(grad_input);
    auto shape = grad_input_type.getShape();
    auto squeezed_shape = shape.drop_front();
    grad_input = mlir::stablehlo::Reshape(
        mlir::RankedTensorType::get(squeezed_shape,
                                    grad_input_type.getElementType()),
        grad_input);
  }

  return grad_input;
}

}  // namespace torch_tpu
