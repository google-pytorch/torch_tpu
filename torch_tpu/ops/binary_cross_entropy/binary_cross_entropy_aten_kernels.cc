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

#include "torch_tpu/ops/binary_cross_entropy/binary_cross_entropy_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Reduction.h"
#include "ATen/core/TensorBody.h"
#include "ATen/native/Resize.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/binary.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/reductions/reductions.h"
#include "torch_tpu/ops/reductions/sum.h"
#include "torch_tpu/ops/unary.h"

namespace torch_tpu {
namespace {

absl::Status ValidateBCEInputs(const at::Tensor& input,
                               const at::Tensor& target,
                               const std::optional<at::Tensor>& weight) {
  TT_RET_CHECK(IsFloatingPoint(input), error::kInvalidArgument)
      << "expected floating point for input, got '" << input.scalar_type()
      << "'";
  TT_RET_CHECK(IsFloatingPoint(target), error::kInvalidArgument)
      << "expected floating point for target, got '" << target.scalar_type()
      << "'";

  TT_RET_CHECK(input.sizes() == target.sizes(), error::kInvalidArgument)
      << "expected input and target to have the same shape, got "
      << input.sizes() << " and " << target.sizes();

  return absl::OkStatus();
}

absl::StatusOr<mlir::MlirOp> BuildBCEForwardShlo(
    mlir::MlirOp self_op, mlir::MlirOp target_op,
    std::optional<mlir::MlirOp> weight_op, int64_t reduction, int64_t numel,
    mlir::MlirBuilder& builder) {
  TT_ASSIGN_OR_RETURN(mlir::ElementType dtype, GetElementType(self_op));
  mlir::MlirOp neg_100 = MakeScalarConstant(builder, -100.0, dtype);
  mlir::MlirOp one = MakeScalarConstant(builder, 1.0, dtype);

  // log(x) clamped to -100
  TT_ASSIGN_OR_RETURN(mlir::MlirOp log_self, BuildLogShlo(self_op, dtype));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp clamped_log_self,
                      BuildMaximumShlo(log_self, neg_100));

  // log(1-x) using Log1p(-x) clamped to -100
  TT_ASSIGN_OR_RETURN(mlir::MlirOp neg_self, BuildNegShlo(self_op));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp log1p_neg_self,
                      BuildLog1pShlo(neg_self, dtype));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp clamped_log1p_neg_self,
                      BuildMaximumShlo(log1p_neg_self, neg_100));

  // (y - 1)
  TT_ASSIGN_OR_RETURN(mlir::MlirOp target_minus_one,
                      BuildSubShlo(target_op, one));

  // Element-wise Loss: (y - 1) * log(1-x) - y * log(x)
  TT_ASSIGN_OR_RETURN(mlir::MlirOp left_term,
                      BuildMulShlo(target_minus_one, clamped_log1p_neg_self));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp right_term,
                      BuildMulShlo(target_op, clamped_log_self));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp loss, BuildSubShlo(left_term, right_term));

  if (weight_op.has_value()) {
    mlir::MlirOp converted_weight = *weight_op;
    TT_ASSIGN_OR_RETURN(mlir::ElementType weight_dtype,
                        GetElementType(converted_weight));
    if (weight_dtype != dtype) {
      converted_weight =
          mlir::stablehlo::ConvertElementType(converted_weight, dtype);
    }
    TT_ASSIGN_OR_RETURN(loss, BuildMulShlo(loss, converted_weight));
  }

  if (reduction == at::Reduction::None) {
    return loss;
  }

  TT_ASSIGN_OR_RETURN(
      const mlir::MlirOp sum_loss,
      BuildSumShlo(loss, GetAllDimensions(loss), ReductionMode::kDropDims));

  if (reduction == at::Reduction::Sum) {
    return sum_loss;
  }

  // reduction == at::Reduction::Mean
  mlir::MlirOp numel_const = MakeScalarConstant(builder, numel, dtype);
  return BuildDivShlo(sum_loss, numel_const);
}

absl::StatusOr<mlir::MlirOp> BuildBCEBackwardShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp self_op, mlir::MlirOp target_op,
    std::optional<mlir::MlirOp> weight_op, int64_t reduction, int64_t numel,
    mlir::MlirBuilder& builder) {
  TT_ASSIGN_OR_RETURN(mlir::ElementType dtype, GetElementType(self_op));
  mlir::MlirOp epsilon = MakeScalarConstant(builder, 1e-12, dtype);
  mlir::MlirOp one = MakeScalarConstant(builder, 1.0, dtype);

  // Numerator: x - y
  TT_ASSIGN_OR_RETURN(mlir::MlirOp numerator, BuildSubShlo(self_op, target_op));

  // Denominator: (1 - x) * x
  TT_ASSIGN_OR_RETURN(mlir::MlirOp one_minus_self, BuildSubShlo(one, self_op));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp denominator,
                      BuildMulShlo(one_minus_self, self_op));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp clamped_denominator,
                      BuildMaximumShlo(denominator, epsilon));

  // Base Gradient: (x - y) / max(...)
  TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_base,
                      BuildDivShlo(numerator, clamped_denominator));

  // Apply grad_output (Broadcast if it's a scalar from Sum/Mean reduction)
  mlir::MlirOp final_grad_output = grad_output_op;
  if (reduction != at::Reduction::None) {
    TT_ASSIGN_OR_RETURN(
        final_grad_output,
        BroadcastIfNeeded(grad_output_op,
                          GetTensorTypeOrDie(self_op).getShape()));
  }

  TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_input,
                      BuildMulShlo(grad_base, final_grad_output));

  if (weight_op.has_value()) {
    mlir::MlirOp converted_weight = *weight_op;
    TT_ASSIGN_OR_RETURN(mlir::ElementType weight_dtype,
                        GetElementType(converted_weight));
    if (weight_dtype != dtype) {
      converted_weight =
          mlir::stablehlo::ConvertElementType(converted_weight, dtype);
    }
    TT_ASSIGN_OR_RETURN(grad_input, BuildMulShlo(grad_input, converted_weight));
  }

  if (reduction == at::Reduction::Mean) {
    mlir::MlirOp numel_const = MakeScalarConstant(builder, numel, dtype);
    TT_ASSIGN_OR_RETURN(grad_input, BuildDivShlo(grad_input, numel_const));
  }

  return grad_input;
}

absl::StatusOr<DeviceBufferRef> DispatchBCEForward(
    const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction,
    OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateBCEInputs(self, target, weight));

  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));

  const int64_t numel = self.numel();
  const bool has_weight = weight.has_value() && weight->defined();

  auto op_builder = [reduction, numel, has_weight](
                        absl::Span<const mlir::MlirOp> inputs,
                        mlir::MlirBuilder& builder) {
    mlir::MlirOp self_op = inputs[0];
    mlir::MlirOp target_op = inputs[1];
    std::optional<mlir::MlirOp> weight_op;
    if (has_weight) {
      weight_op = inputs[2];
    }
    return BuildBCEForwardShlo(self_op, target_op, weight_op, reduction, numel,
                               builder);
  };

  std::vector<at::Tensor> inputs = {self, target};
  if (has_weight) {
    inputs.push_back(*weight);
  }

  SmallInt64Vector out_dims;
  if (reduction == at::Reduction::None) {
    out_dims = CopyIntVector(self.sizes());
  } else {
    out_dims = {};  // Scalar
  }

  return DispatchOp<kDynamicSize, 1>(
      std::move(op_builder), inputs,
      {.out_dtype = out_dtype,
       .out_dims = out_dims,
       .op_param_cache_keys = std::move(param_keys)});
}

absl::StatusOr<DeviceBufferRef> DispatchBCEBackward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const std::optional<at::Tensor>& weight,
    int64_t reduction, OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateBCEInputs(self, target, weight));

  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));

  const int64_t numel = self.numel();
  const bool has_weight = weight.has_value() && weight->defined();

  auto op_builder = [reduction, numel, has_weight](
                        absl::Span<const mlir::MlirOp> inputs,
                        mlir::MlirBuilder& builder) {
    mlir::MlirOp grad_output_op = inputs[0];
    mlir::MlirOp self_op = inputs[1];
    mlir::MlirOp target_op = inputs[2];
    std::optional<mlir::MlirOp> weight_op;
    if (has_weight) {
      weight_op = inputs[3];
    }
    return BuildBCEBackwardShlo(grad_output_op, self_op, target_op, weight_op,
                                reduction, numel, builder);
  };

  std::vector<at::Tensor> inputs = {grad_output, self, target};
  if (has_weight) {
    inputs.push_back(*weight);
  }

  SmallInt64Vector out_dims = CopyIntVector(self.sizes());

  return DispatchOp<kDynamicSize, 1>(
      std::move(op_builder), inputs,
      {.out_dtype = out_dtype,
       .out_dims = out_dims,
       .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor AtenBinaryCrossEntropy(const at::Tensor& self,
                                  const at::Tensor& target,
                                  const std::optional<at::Tensor>& weight,
                                  int64_t reduction) {
  TT_KERNEL(OpName::kBinaryCrossEntropy, param_keys,
            (self, target, weight, reduction), {
              TT_ASSIGN_OR_THROW(
                  DeviceBufferRef result_buffer,
                  DispatchBCEForward(self, target, weight, reduction,
                                     std::move(param_keys)));
              return MakeTensor(result_buffer);
            });
}

at::Tensor& AtenBinaryCrossEntropyOut(const at::Tensor& self,
                                      const at::Tensor& target,
                                      const std::optional<at::Tensor>& weight,
                                      int64_t reduction, at::Tensor& out) {
  TT_KERNEL(OpName::kBinaryCrossEntropyOut, param_keys,
            (self, target, weight, reduction, out), {
              TT_ASSIGN_OR_THROW(
                  DeviceBufferRef result_buffer,
                  DispatchBCEForward(self, target, weight, reduction,
                                     std::move(param_keys)));
              at::native::resize_output(out, result_buffer.dimensions());
              TT_THROW_IF_ERROR(
                  AssignBufferToAtTensor(std::move(result_buffer), out));
              return out;
            });
}

at::Tensor AtenBinaryCrossEntropyBackward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const std::optional<at::Tensor>& weight,
    int64_t reduction) {
  TT_KERNEL(OpName::kBinaryCrossEntropyBackward, param_keys,
            (grad_output, self, target, weight, reduction), {
              TT_ASSIGN_OR_THROW(
                  DeviceBufferRef result_buffer,
                  DispatchBCEBackward(grad_output, self, target, weight,
                                      reduction, std::move(param_keys)));
              return MakeTensor(result_buffer);
            });
}

at::Tensor& AtenBinaryCrossEntropyBackwardGradInput(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const std::optional<at::Tensor>& weight,
    int64_t reduction, at::Tensor& grad_input) {
  TT_KERNEL(OpName::kBinaryCrossEntropyBackwardGradInput, param_keys,
            (grad_output, self, target, weight, reduction, grad_input), {
              TT_ASSIGN_OR_THROW(
                  DeviceBufferRef result_buffer,
                  DispatchBCEBackward(grad_output, self, target, weight,
                                      reduction, std::move(param_keys)));
              at::native::resize_output(grad_input, result_buffer.dimensions());
              TT_THROW_IF_ERROR(
                  AssignBufferToAtTensor(std::move(result_buffer), grad_input));
              return grad_input;
            });
}

}  // namespace torch_tpu
