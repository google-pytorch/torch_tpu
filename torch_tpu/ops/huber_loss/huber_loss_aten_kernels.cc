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

#include "torch_tpu/ops/huber_loss/huber_loss_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Reduction.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/reductions/reductions.h"
#include "torch_tpu/ops/reductions/sum.h"
#include "torch_tpu/ops/resize/resize_aten_kernels.h"

namespace torch_tpu {
namespace {

absl::Status CheckHuberLossInputs(const at::Tensor& self,
                                  const at::Tensor& target, int64_t reduction,
                                  double delta) {
  TT_RET_CHECK(delta > 0, error::kInvalidArgument)
      << "expected delta > 0, got " << delta;
  TT_RET_CHECK(self.scalar_type() == target.scalar_type(),
               error::kInvalidArgument)
      << "expected self and target to have the same dtype, got "
      << torch_tpu::ToString(self.scalar_type()) << " and "
      << torch_tpu::ToString(target.scalar_type());
  TT_RET_CHECK(at::isFloatingType(self.scalar_type()), error::kInvalidArgument)
      << "expected floating point dtype, got "
      << torch_tpu::ToString(self.scalar_type());
  TT_RET_CHECK(reduction == at::Reduction::None ||
                   reduction == at::Reduction::Mean ||
                   reduction == at::Reduction::Sum,
               error::kInvalidArgument)
      << "expected reduction to be None (0), Mean (1), or Sum (2), got "
      << reduction;
  return absl::OkStatus();
}

absl::StatusOr<mlir::MlirOp> BuildHuberLossShlo(mlir::MlirOp self_op,
                                                mlir::MlirOp target_op,
                                                int64_t reduction,
                                                double delta) {
  TT_ASSIGN_OR_RETURN(Dimensions bcast_shape,
                      GetBroadcastShape({self_op, target_op}));
  TT_ASSIGN_OR_RETURN(self_op, BroadcastIfNeeded(self_op, bcast_shape));
  TT_ASSIGN_OR_RETURN(target_op, BroadcastIfNeeded(target_op, bcast_shape));

  mlir::MlirOp diff = mlir::stablehlo::Subtract(self_op, target_op);
  mlir::MlirOp abs_diff = mlir::stablehlo::Abs(diff);

  mlir::MlirOp half = MakeConstantLike(self_op, 0.5);
  mlir::MlirOp delta_const = MakeConstantLike(self_op, delta);
  mlir::MlirOp half_delta = MakeConstantLike(self_op, 0.5 * delta);

  mlir::MlirOp diff_sq = mlir::stablehlo::Mul(diff, diff);
  mlir::MlirOp loss_sq = mlir::stablehlo::Mul(diff_sq, half);

  mlir::MlirOp loss_lin_diff = mlir::stablehlo::Subtract(abs_diff, half_delta);
  mlir::MlirOp loss_lin = mlir::stablehlo::Mul(loss_lin_diff, delta_const);

  mlir::MlirOp is_less = mlir::stablehlo::Compare(
      abs_diff, delta_const, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp elem_loss = mlir::stablehlo::Select(is_less, loss_sq, loss_lin);

  if (reduction == at::Reduction::None) {
    return elem_loss;
  }

  TT_ASSIGN_OR_RETURN(mlir::MlirOp sum_loss,
                      BuildSumShlo(elem_loss, GetAllDimensions(elem_loss),
                                   ReductionMode::kDropDims,
                                   /*element_type_opt=*/std::nullopt));

  if (reduction == at::Reduction::Sum) {
    return sum_loss;
  } else {
    mlir::Type float_type = GetTensorTypeOrDie(elem_loss).getElementType();
    mlir::MlirOp numel_const = GetNumElements(elem_loss, float_type);
    TT_ASSIGN_OR_RETURN(numel_const, BroadcastIfNeeded(numel_const, sum_loss));
    return mlir::stablehlo::Div(sum_loss, numel_const);
  }
}

absl::StatusOr<mlir::MlirOp> BuildHuberLossBackwardShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp self_op, mlir::MlirOp target_op,
    const Dimensions& self_shape, int64_t reduction, double delta) {
  TT_ASSIGN_OR_RETURN(Dimensions bcast_shape,
                      GetBroadcastShape({self_op, target_op}));
  TT_ASSIGN_OR_RETURN(self_op, BroadcastIfNeeded(self_op, bcast_shape));
  TT_ASSIGN_OR_RETURN(target_op, BroadcastIfNeeded(target_op, bcast_shape));

  mlir::MlirOp diff = mlir::stablehlo::Subtract(self_op, target_op);
  mlir::MlirOp neg_delta = MakeConstantLike(self_op, -delta);
  mlir::MlirOp pos_delta = MakeConstantLike(self_op, delta);
  mlir::MlirOp clamped_diff =
      mlir::stablehlo::Clamp(neg_delta, diff, pos_delta);

  TT_ASSIGN_OR_RETURN(grad_output_op,
                      BroadcastIfNeeded(grad_output_op, self_op));
  mlir::MlirOp grad_raw = mlir::stablehlo::Mul(clamped_diff, grad_output_op);

  if (reduction == at::Reduction::Mean) {
    mlir::Type float_type = GetTensorTypeOrDie(grad_raw).getElementType();
    mlir::MlirOp numel_const = GetNumElements(grad_raw, float_type);
    TT_ASSIGN_OR_RETURN(numel_const, BroadcastIfNeeded(numel_const, grad_raw));
    grad_raw = mlir::stablehlo::Div(grad_raw, numel_const);
  }

  if (bcast_shape != self_shape) {
    const int64_t bcast_rank = bcast_shape.size();
    const int64_t self_rank = self_shape.size();
    const int64_t rank_diff = bcast_rank - self_rank;

    for (int64_t d = 0; d < rank_diff; ++d) {
      TT_ASSIGN_OR_RETURN(
          grad_raw, BuildSumShlo(grad_raw, {0}, ReductionMode::kDropDims));
    }
    for (int64_t d = 0; d < self_rank; ++d) {
      if (self_shape[d] == 1 && bcast_shape[d + rank_diff] > 1) {
        TT_ASSIGN_OR_RETURN(
            grad_raw, BuildSumShlo(grad_raw, {d}, ReductionMode::kKeepDims));
      }
    }
  }

  return grad_raw;
}

absl::StatusOr<DeviceBufferRef> AtenHuberLossImpl(const at::Tensor& self,
                                                  const at::Tensor& target,
                                                  int64_t reduction,
                                                  double delta,
                                                  OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(CheckHuberLossInputs(self, target, reduction, delta));

  auto op_builder = [reduction, delta](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [self_op, target_op] = inputs;
    return BuildHuberLossShlo(self_op, target_op, reduction, delta);
  };

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));

  TT_ASSIGN_OR_RETURN(Dimensions broadcasted_sizes, InferSize(self, target));
  Dimensions out_dims = (reduction == at::Reduction::None)
                            ? std::move(broadcasted_sizes)
                            : Dimensions{};

  return DispatchOp<2>(op_builder, {self, target},
                       {.out_dtype = output_dtype,
                        .out_dims = std::move(out_dims),
                        .op_param_cache_keys = std::move(param_keys)});
}

absl::StatusOr<DeviceBufferRef> AtenHuberLossBackwardImpl(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, int64_t reduction, double delta,
    OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(CheckHuberLossInputs(self, target, reduction, delta));
  TT_RET_CHECK(grad_output.scalar_type() == self.scalar_type(),
               error::kInvalidArgument)
      << "expected grad_output and self to have the same dtype, got "
      << torch_tpu::ToString(grad_output.scalar_type()) << " and "
      << torch_tpu::ToString(self.scalar_type());

  Dimensions self_dims(self.sizes().begin(), self.sizes().end());

  auto op_builder = [self_dims, reduction,
                     delta](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [grad_output_op, self_op, target_op] = inputs;
    return BuildHuberLossBackwardShlo(grad_output_op, self_op, target_op,
                                      self_dims, reduction, delta);
  };

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));

  return DispatchOp<3>(op_builder, {grad_output, self, target},
                       {.out_dtype = output_dtype,
                        .out_dims = std::move(self_dims),
                        .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor AtenHuberLoss(const at::Tensor& self, const at::Tensor& target,
                         int64_t reduction, double delta) {
  TT_KERNEL(OpName::kHuberLoss, param_keys, (self, target, reduction, delta), {
    TT_ASSIGN_OR_THROW(auto result_buf,
                       AtenHuberLossImpl(self, target, reduction, delta,
                                         std::move(param_keys)));
    return MakeTensor(std::move(result_buf));
  });
}

at::Tensor& AtenHuberLossOut(const at::Tensor& self, const at::Tensor& target,
                             int64_t reduction, double delta, at::Tensor& out) {
  TT_KERNEL(
      OpName::kHuberLossOut, param_keys, (self, target, reduction, delta, out),
      {
        TT_ASSIGN_OR_THROW(auto result_buf,
                           AtenHuberLossImpl(self, target, reduction, delta,
                                             std::move(param_keys)));
        TT_THROW_IF_ERROR(
            ResizeTensorIfShapeDiffers(out, result_buf.dimensions()));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

at::Tensor AtenHuberLossBackward(const at::Tensor& grad_output,
                                 const at::Tensor& self,
                                 const at::Tensor& target, int64_t reduction,
                                 double delta) {
  TT_KERNEL(OpName::kHuberLossBackward, param_keys,
            (grad_output, self, target, reduction, delta), {
              TT_ASSIGN_OR_THROW(auto result_buf,
                                 AtenHuberLossBackwardImpl(
                                     grad_output, self, target, reduction,
                                     delta, std::move(param_keys)));
              return MakeTensor(std::move(result_buf));
            });
}

at::Tensor& AtenHuberLossBackwardOut(const at::Tensor& grad_output,
                                     const at::Tensor& self,
                                     const at::Tensor& target,
                                     int64_t reduction, double delta,
                                     at::Tensor& grad_input) {
  TT_KERNEL(OpName::kHuberLossBackwardOut, param_keys,
            (grad_output, self, target, reduction, delta, grad_input), {
              TT_ASSIGN_OR_THROW(auto result_buf,
                                 AtenHuberLossBackwardImpl(
                                     grad_output, self, target, reduction,
                                     delta, std::move(param_keys)));
              TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(
                  grad_input, result_buf.dimensions()));
              TT_THROW_IF_ERROR(
                  AssignBufferToAtTensor(std::move(result_buf), grad_input));
              return grad_input;
            });
}

}  // namespace torch_tpu
