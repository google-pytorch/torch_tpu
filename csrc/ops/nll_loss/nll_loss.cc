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

#include "csrc/ops/nll_loss/nll_loss.h"

#include <cstdint>
#include <limits>
#include <optional>

#include "ATen/core/Reduction.h"
#include "absl/status/statusor.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/binary.h"
#include "csrc/ops/gather/gather.h"
#include "csrc/ops/index_select/index_select.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/reductions/sum.h"
#include "csrc/ops/scatter/scatter.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

mlir::MlirOp GatherWeightsShlo(mlir::MlirOp weight, mlir::MlirOp safe_target) {
  const mlir::RankedTensorType target_type = GetTensorTypeOrDie(safe_target);
  mlir::MlirOp flat_target = safe_target;
  if (target_type.getRank() != 1) {
    flat_target =
        mlir::stablehlo::Reshape(safe_target, {target_type.getNumElements()});
  }
  mlir::MlirOp gathered_weights = BuildIndexSelectShlo(weight, 0, flat_target);
  if (target_type.getRank() != 1) {
    gathered_weights =
        mlir::stablehlo::Reshape(gathered_weights, target_type.getShape());
  }
  return gathered_weights;
}

absl::StatusOr<MlirOpResults<2>> BuildNllLossForwardOutShlo(
    mlir::MlirOp self, mlir::MlirOp target, std::optional<mlir::MlirOp> weight,
    int64_t reduction, int64_t ignore_index, mlir::MlirBuilder& builder) {
  const mlir::RankedTensorType orig_self_type = GetTensorTypeOrDie(self);
  const mlir::RankedTensorType orig_target_type = GetTensorTypeOrDie(target);

  TT_ASSIGN_OR_RETURN(const mlir::ElementType orig_self_dtype,
                      GetElementType(self));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(orig_self_dtype));

  if (orig_self_type.getNumElements() == 0 ||
      orig_target_type.getNumElements() == 0) {
    mlir::MlirOp loss_output;
    switch (static_cast<at::Reduction::Reduction>(reduction)) {
      case at::Reduction::None:
        loss_output = MakeConstantLike(target, 0.0, orig_self_dtype);
        break;
      case at::Reduction::Sum:
        loss_output = MakeScalarConstant(builder, 0.0, orig_self_dtype);
        break;
      case at::Reduction::Mean:
        loss_output = MakeScalarConstant(
            builder, std::numeric_limits<double>::quiet_NaN(), orig_self_dtype);
        break;
      default:
        return TT_ERROR(  // ERROR_COV_INFEASIBLE=Validated before calling
                          // builder.
                   error::kInvalidArgument)
               << "expected reduction to be one of ['none', 'mean', 'sum'], "
               << "got " << reduction;
    }
    mlir::MlirOp total_weight_scalar =
        MakeScalarConstant(builder, 0.0, orig_self_dtype);
    return {{loss_output, total_weight_scalar}};
  }

  const int64_t orig_target_rank = orig_target_type.getRank();
  bool unsqueezed = false;
  if (orig_self_type.getRank() == 1) {
    unsqueezed = true;
    TT_ASSIGN_OR_RETURN(self, Unsqueeze(self, 0));
    if (orig_target_rank == 0) {
      TT_ASSIGN_OR_RETURN(target, Unsqueeze(target, 0));
    }
  }

  // Shapes after optional 1D normalization:
  //   self: [N, C, d1, ..., dk] (rank >= 2)
  //   target: [N, d1, ..., dk] (rank >= 1)
  //   weight: [C]
  TT_ASSIGN_OR_RETURN(self, CastIfNeeded(self, compute_dtype));
  if (weight.has_value()) {
    TT_ASSIGN_OR_RETURN(weight, CastIfNeeded(weight.value(), compute_dtype));
  }
  TT_ASSIGN_OR_RETURN(target, CastIfNeeded(target, mlir::ElementType::I64));

  // mask = (target != ignore_index), shape [N, d1, ..., dk]
  // Clamp ignored target indices to 0 before gathering from `self` and `weight`
  // so out-of-bounds `ignore_index` values (e.g. -100) never index into index 0
  // or last class with `-inf` and cause `-inf * 0 = NaN`.
  mlir::MlirOp ignore_index_bcast = MakeConstantLike(target, ignore_index);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mask,
                      BuildNeShlo(target, ignore_index_bcast));
  mlir::MlirOp zero_target = MakeConstantLike(target, 0);
  mlir::MlirOp safe_target = mlir::stablehlo::Select(mask, target, zero_target);

  // raw_loss = -self[safe_target] along class dimension 1.
  TT_ASSIGN_OR_RETURN(mlir::MlirOp target_unsqueeze, Unsqueeze(safe_target, 1));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp gathered_unsqueeze,
                      BuildGatherShlo(self, 1, target_unsqueeze,
                                      /*sparse_grad=*/false, compute_dtype));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp gathered, Squeeze(gathered_unsqueeze, {1}));
  mlir::MlirOp raw_loss = mlir::stablehlo::Neg(gathered);

  // Apply class weights and mask out ignored positions using Select (not Mul).
  mlir::MlirOp zero_float = MakeConstantLike(raw_loss, 0.0, compute_dtype);
  mlir::MlirOp weight_to_apply;
  if (weight.has_value()) {
    mlir::MlirOp sample_weight = GatherWeightsShlo(weight.value(), safe_target);
    raw_loss = mlir::stablehlo::Mul(raw_loss, sample_weight);
    weight_to_apply = mlir::stablehlo::Select(mask, sample_weight, zero_float);
  } else {
    weight_to_apply = mlir::stablehlo::ConvertElementType(mask, compute_dtype);
  }
  mlir::MlirOp loss_no_reduction =
      mlir::stablehlo::Select(mask, raw_loss, zero_float);

  mlir::MlirOp loss_output;
  mlir::MlirOp total_weight_scalar;
  switch (static_cast<at::Reduction::Reduction>(reduction)) {
    case at::Reduction::None: {
      if (unsqueezed) {
        TT_ASSIGN_OR_RETURN(loss_output, Squeeze(loss_no_reduction, {0}));
        TT_ASSIGN_OR_RETURN(
            total_weight_scalar,
            BuildSumShlo(weight_to_apply, GetAllDimensions(weight_to_apply),
                         ReductionMode::kDropDims));
      } else {
        loss_output = loss_no_reduction;
        total_weight_scalar = MakeScalarConstant(builder, 0.0, compute_dtype);
      }
      break;
    }
    case at::Reduction::Sum: {
      TT_ASSIGN_OR_RETURN(
          loss_output,
          BuildSumShlo(loss_no_reduction, GetAllDimensions(loss_no_reduction),
                       ReductionMode::kDropDims));
      TT_ASSIGN_OR_RETURN(
          total_weight_scalar,
          BuildSumShlo(weight_to_apply, GetAllDimensions(weight_to_apply),
                       ReductionMode::kDropDims));
      break;
    }
    case at::Reduction::Mean: {
      TT_ASSIGN_OR_RETURN(
          mlir::MlirOp sum,
          BuildSumShlo(loss_no_reduction, GetAllDimensions(loss_no_reduction),
                       ReductionMode::kDropDims));
      TT_ASSIGN_OR_RETURN(
          total_weight_scalar,
          BuildSumShlo(weight_to_apply, GetAllDimensions(weight_to_apply),
                       ReductionMode::kDropDims));
      loss_output = mlir::stablehlo::Div(sum, total_weight_scalar);
      break;
    }
    default:
      return TT_ERROR(  // ERROR_COV_INFEASIBLE=Validated before calling
                        // builder.
                 error::kInvalidArgument)
             << "expected reduction to be one of ['none', 'mean', 'sum'], "
             << "got " << reduction;
  }

  TT_ASSIGN_OR_RETURN(loss_output, CastIfNeeded(loss_output, orig_self_dtype));
  TT_ASSIGN_OR_RETURN(total_weight_scalar,
                      CastIfNeeded(total_weight_scalar, orig_self_dtype));
  return {{loss_output, total_weight_scalar}};
}

absl::StatusOr<mlir::MlirOp> BuildNllLossBackwardGradInputShlo(
    mlir::MlirOp grad_output, mlir::MlirOp self, mlir::MlirOp target,
    std::optional<mlir::MlirOp> weight, int64_t reduction, int64_t ignore_index,
    mlir::MlirOp total_weight, mlir::MlirBuilder& /*builder*/) {
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

  // Unsqueeze if it is a single 1D sample without a batch dimension.
  bool unsqueezed = false;
  if (orig_self_type.getRank() == 1) {
    unsqueezed = true;
    TT_ASSIGN_OR_RETURN(self, Unsqueeze(self, 0));
    if (orig_target_type.getRank() == 0) {
      TT_ASSIGN_OR_RETURN(target, Unsqueeze(target, 0));
    }
    if (reduction == at::Reduction::None &&
        GetTensorTypeOrDie(grad_output).getRank() == 0) {
      TT_ASSIGN_OR_RETURN(grad_output, Unsqueeze(grad_output, 0));
    }
  }

  // Shapes after optional 1D normalization:
  //   grad_output: [N, d1, ..., dk] for None, or 1-element (0D/1D) for Sum/Mean
  //   self: [N, C, d1, ..., dk] (rank >= 2)
  //   target: [N, d1, ..., dk] (rank >= 1)
  //   weight: [C]
  //   total_weight: 1-element (0D/1D)
  TT_ASSIGN_OR_RETURN(grad_output, CastIfNeeded(grad_output, compute_dtype));
  TT_ASSIGN_OR_RETURN(total_weight, CastIfNeeded(total_weight, compute_dtype));
  if (weight.has_value()) {
    TT_ASSIGN_OR_RETURN(weight, CastIfNeeded(weight.value(), compute_dtype));
  }
  TT_ASSIGN_OR_RETURN(target, CastIfNeeded(target, mlir::ElementType::I64));
  const mlir::RankedTensorType target_type = GetTensorTypeOrDie(target);

  mlir::MlirOp ignore_index_bcast = MakeConstantLike(target, ignore_index);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mask,
                      BuildNeShlo(target, ignore_index_bcast));
  mlir::MlirOp zero_target = MakeConstantLike(target, 0);
  mlir::MlirOp safe_target = mlir::stablehlo::Select(mask, target, zero_target);

  mlir::MlirOp grad_val = mlir::stablehlo::Neg(grad_output);
  switch (static_cast<at::Reduction::Reduction>(reduction)) {
    case at::Reduction::None: {
      TT_ASSIGN_OR_RETURN(grad_val,
                          BroadcastIfNeeded(grad_val, target_type.getShape()));
      break;
    }
    case at::Reduction::Sum: {
      if (GetTensorTypeOrDie(grad_val).getRank() != 0) {
        grad_val = mlir::stablehlo::Reshape(grad_val, {});
      }
      TT_ASSIGN_OR_RETURN(grad_val,
                          BroadcastIfNeeded(grad_val, target_type.getShape()));
      break;
    }
    case at::Reduction::Mean: {
      if (GetTensorTypeOrDie(grad_val).getRank() != 0) {
        grad_val = mlir::stablehlo::Reshape(grad_val, {});
      }
      if (GetTensorTypeOrDie(total_weight).getRank() != 0) {
        total_weight = mlir::stablehlo::Reshape(total_weight, {});
      }
      grad_val = mlir::stablehlo::Div(grad_val, total_weight);
      TT_ASSIGN_OR_RETURN(grad_val,
                          BroadcastIfNeeded(grad_val, target_type.getShape()));
      break;
    }
    default:
      return TT_ERROR(  // ERROR_COV_INFEASIBLE=Validated before calling
                        // builder.
                 error::kInvalidArgument)
             << "expected reduction to be one of ['none', 'mean', 'sum'], "
             << "got " << reduction;
  }

  if (weight.has_value()) {
    mlir::MlirOp sample_weight = GatherWeightsShlo(weight.value(), safe_target);
    grad_val = mlir::stablehlo::Mul(grad_val, sample_weight);
  }

  // Zero out ignored targets via Select so NaN/inf (e.g. when total_weight == 0
  // in Mean reduction) does not propagate into grad_input.
  mlir::MlirOp zero_grad_val = MakeConstantLike(grad_val, 0.0, compute_dtype);
  grad_val = mlir::stablehlo::Select(mask, grad_val, zero_grad_val);

  // Unsqueeze at class dimension 1 to [N, 1, d1, ..., dk] for BuildScatterShlo.
  TT_ASSIGN_OR_RETURN(mlir::MlirOp safe_target_unsqueeze,
                      Unsqueeze(safe_target, 1));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_val_unsqueeze, Unsqueeze(grad_val, 1));

  mlir::MlirOp grad_input = MakeConstantLike(self, 0.0, compute_dtype);
  TT_ASSIGN_OR_RETURN(
      grad_input,
      BuildScatterShlo(grad_input, 1, safe_target_unsqueeze, grad_val_unsqueeze,
                       ScatterOp::kAdd, compute_dtype));

  if (unsqueezed) {
    TT_ASSIGN_OR_RETURN(grad_input, Squeeze(grad_input, {0}));
  }

  return CastIfNeeded(grad_input, orig_self_dtype);
}

}  // namespace torch_tpu
