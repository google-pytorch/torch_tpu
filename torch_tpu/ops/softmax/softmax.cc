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

#include "torch_tpu/ops/softmax/softmax.h"

#include <cstdint>
#include <limits>
#include <optional>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/reductions/sum.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildBroadcastedMaxShlo(mlir::MlirOp input_op,
                                                     int64_t dim) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  auto scalar_type =
      mlir::RankedTensorType::get({}, input_type.getElementType());
  TT_ASSIGN_OR_RETURN(mlir::DenseElementsAttr min_finite_value,
                      GetVerifiedMinFiniteValue(builder, scalar_type));
  mlir::MlirOp min_finite_const =
      mlir::stablehlo::Constant(builder, min_finite_value);

  mlir::SmallVector<mlir::MlirOp> max_val_reduced = mlir::stablehlo::Reduce(
      builder, {input_op}, {min_finite_const},
      [&input_type](mlir::RegionBuilder& rb) {
        // Scalar Max Region
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::MaxOp>(
            input_type.getElementType(), rb.getRegion(), rb.getOpBuilder());
      },
      /*dimensions_to_reduce=*/{dim});
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal OpenXLA error.
      max_val_reduced.size() == 1, error::kInternal)
      << "Expected 1 result from reduce op, got " << max_val_reduced.size();

  TT_ASSIGN_OR_RETURN(mlir::MlirOp unsqueezed_op,
                      Unsqueeze(max_val_reduced[0], dim));
  return BroadcastIfNeeded(unsqueezed_op, input_op);
}

absl::StatusOr<mlir::MlirOp> BuildBroadcastedSumShlo(mlir::MlirOp input_op,
                                                     int64_t dim) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp sum_op, BuildSumShlo(input_op, {dim}));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp unsqueezed_op, Unsqueeze(sum_op, dim));
  return BroadcastIfNeeded(unsqueezed_op, input_op);
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildSoftmaxShlo(mlir::MlirOp input_op,
                                              int64_t dim,
                                              SoftmaxMode softmax_mode) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  mlir::MlirBuilder& builder = input_op.getBuilder();
  if (input_type.getRank() == 0) {
    return MakeScalarConstant(builder,
                              softmax_mode == SoftmaxMode::kSoftmax ? 1.0 : 0.0,
                              input_type.getElementType());
  }

  if (dim < 0) {
    dim += input_type.getRank();
  }

  TT_ASSIGN_OR_RETURN(mlir::MlirOp max_val_broadcasted,
                      BuildBroadcastedMaxShlo(input_op, dim));
  mlir::MlirOp shifted_op =
      mlir::stablehlo::Subtract(input_op, max_val_broadcasted);
  mlir::MlirOp exp_op = mlir::stablehlo::Exp(shifted_op);

  TT_ASSIGN_OR_RETURN(mlir::MlirOp sum_exp_broadcasted_op,
                      BuildBroadcastedSumShlo(exp_op, dim));

  mlir::MlirOp result;
  if (softmax_mode == SoftmaxMode::kLogSoftmax) {
    mlir::MlirOp log_op = mlir::stablehlo::Log(sum_exp_broadcasted_op);
    result = mlir::stablehlo::Subtract(shifted_op, log_op);
  } else {
    result = mlir::stablehlo::Div(exp_op, sum_exp_broadcasted_op);
  }

  return result;
}

absl::StatusOr<mlir::MlirOp> BuildSoftmaxBackwardDataShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp output_op, int64_t dim,
    mlir::stablehlo::Precision precision, SoftmaxMode softmax_mode) {
  const mlir::RankedTensorType grad_output_type =
      GetTensorTypeOrDie(grad_output_op);
  const mlir::RankedTensorType output_type = GetTensorTypeOrDie(output_op);
  ABSL_VLOG(3) << "BuildSoftmaxBackwardDataShlo: grad_output_type: "
               << mlir::debugString(grad_output_type);
  ABSL_VLOG(3) << "BuildSoftmaxBackwardDataShlo: output_type: "
               << mlir::debugString(output_type);

  if (output_type.getRank() == 0) {
    mlir::MlirBuilder& builder = grad_output_op.getBuilder();
    return MakeScalarConstant(builder, 0.0, output_type.getElementType());
  }

  TT_RET_CHECK(output_type.getShape() == grad_output_type.getShape(),
               error::kInvalidArgument)
      << "expected grad_output and output arguments to have the same shape, "
         "got "
      << ToString(grad_output_type.getShape()) << " vs. "
      << ToString(output_type.getShape());
  TT_ASSIGN_OR_RETURN(dim, SafeWrapDim(dim, output_type.getRank()));
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Caught by the `SafeWrapDim()` function
                 // call above.
      0 <= dim && dim < output_type.getRank(), error::kInvalidArgument)
      << "expected the dim argument to be in the range [0, "
      << output_type.getRank() << " (rank of the output)], got " << dim;
  Dimensions all_but_dim;
  all_but_dim.reserve(output_type.getRank() - 1);
  for (int i = 0; i < output_type.getRank(); ++i) {
    if (i != dim) {
      all_but_dim.push_back(i);
    }
  }

  const mlir::ElementType out_element_type = GetElementTypeOrDie(output_op);
  TT_ASSIGN_OR_RETURN(mlir::ElementType computation_dtype,
                      InferComputationDtype(out_element_type));
  TT_ASSIGN_OR_RETURN(grad_output_op,
                      CastIfNeeded(grad_output_op, computation_dtype));
  TT_ASSIGN_OR_RETURN(output_op, CastIfNeeded(output_op, computation_dtype));
  const mlir::RankedTensorType comp_output_type = GetTensorTypeOrDie(output_op);

  mlir::MlirOp result;
  if (softmax_mode == SoftmaxMode::kSoftmax) {
    // For i a valid index in the dim-th dimension:
    //   (dL / d input)_i = (output_op)_i * ((grad_output_op)_i - dot(output_op,
    //   grad_output_op, dim))
    // where the dot is broadcasted back to the original shape.
    auto& ctx = output_op.getContext();

    // Start by computing the dot.
    auto precision_attr =
        mlir::stablehlo::PrecisionConfigAttr::get(&ctx, {precision, precision});

    auto dot_dimension_numbers = mlir::stablehlo::DotDimensionNumbersAttr::get(
        &ctx, all_but_dim, all_but_dim, {dim}, {dim});
    auto dot_op = mlir::stablehlo::DotGeneral(
        output_op, grad_output_op, dot_dimension_numbers, precision_attr);
    const mlir::RankedTensorType dot_op_type = GetTensorTypeOrDie(dot_op);
    ABSL_VLOG(3) << "BuildSoftmaxBackwardDataShlo: dot_op_type: "
                 << mlir::debugString(dot_op_type);
    auto dot_broadcasted =
        mlir::stablehlo::BroadcastInDim(comp_output_type, dot_op, all_but_dim);
    // Compute grad_output_op - dot_broadcasted
    auto sub_op = mlir::stablehlo::Subtract(grad_output_op, dot_broadcasted);

    // Compute output_op * sub_op
    result = mlir::stablehlo::Mul(output_op, sub_op);
  } else {
    // softmax_mode == SoftmaxMode::kLogSoftmax
    // For i a valid index in the dim-th dimension:
    //   (dL / d input)_i = (grad_output_op)_i - exp((output_op)_i) *
    //   sum(grad_output_op, dim)
    // where the sum is broadcasted back to the original shape.
    // Start by computing the sum
    TT_ASSIGN_OR_RETURN(auto sum_op, BuildSumShlo(grad_output_op, {dim}));
    auto sum_broadcasted =
        mlir::stablehlo::BroadcastInDim(comp_output_type, sum_op, all_but_dim);
    // Compute exp(output_op) * sum(grad_output_op, dim)
    auto exp_op = mlir::stablehlo::Exp(output_op);
    auto mul_op = mlir::stablehlo::Mul(exp_op, sum_broadcasted);
    // Compute grad_output_op - mul_op
    result = mlir::stablehlo::Subtract(grad_output_op, mul_op);
  }

  return CastIfNeeded(result, out_element_type);
}

namespace {

absl::StatusOr<mlir::MlirOp> PrepareMaskOp(mlir::MlirOp mask_op,
                                           mlir::MlirOp input_op,
                                           std::optional<int64_t> mask_type) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  const mlir::RankedTensorType mask_tensor_type = GetTensorTypeOrDie(mask_op);

  if (mask_tensor_type.getRank() == 2 && input_type.getRank() == 4) {
    int64_t resolved_type = mask_type.value_or(-1);
    if (resolved_type == -1) {
      // NOTE: There is an inherent ambiguity during the backward pass since
      // mask_type is not provided in the _masked_softmax_backward ATen
      // signature. If input shape is (B, H, L, S) where B == L (batch size
      // equals sequence length), both conditions below will match a mask of
      // shape (B, S) or (L, S). In this case, we default to resolving as a
      // padding mask (resolved_type = 1). If the forward pass actually used
      // mask_type = 0 (attention mask), the backward pass will incorrectly
      // broadcast the mask. This is a known limitation of the PyTorch ATen
      // backward operator signature.
      if (mask_tensor_type.getShape()[0] == input_type.getShape()[0] &&
          mask_tensor_type.getShape()[1] == input_type.getShape()[3]) {
        resolved_type = 1;
      } else if (mask_tensor_type.getShape()[0] == input_type.getShape()[2] &&
                 mask_tensor_type.getShape()[1] == input_type.getShape()[3]) {
        resolved_type = 0;
      }
    }
    if (resolved_type == 1) {
      Dimensions new_shape = {mask_tensor_type.getShape()[0], 1, 1,
                              mask_tensor_type.getShape()[1]};
      mask_op = mlir::stablehlo::Reshape(mask_op, new_shape);
    } else if (resolved_type == 0) {
      Dimensions new_shape = {1, 1, mask_tensor_type.getShape()[0],
                              mask_tensor_type.getShape()[1]};
      mask_op = mlir::stablehlo::Reshape(mask_op, new_shape);
    }
  }
  return BroadcastIfNeeded(mask_op, input_op);
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildMaskedSoftmaxShlo(
    mlir::MlirOp input_op, mlir::MlirOp mask_op, int64_t dim,
    std::optional<int64_t> mask_type) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  mlir::MlirBuilder& builder = input_op.getBuilder();

  if (input_type.getRank() == 0) {
    mlir::MlirOp nan =
        MakeScalarConstant(builder, std::numeric_limits<double>::quiet_NaN(),
                           input_type.getElementType());
    mlir::MlirOp one =
        MakeScalarConstant(builder, 1.0, input_type.getElementType());
    return mlir::stablehlo::Select(mask_op, nan, one);
  }

  TT_ASSIGN_OR_RETURN(mask_op, PrepareMaskOp(mask_op, input_op, mask_type));

  mlir::MlirOp neg_inf =
      MakeConstantLike(input_op, -std::numeric_limits<double>::infinity());
  mlir::MlirOp masked_input =
      mlir::stablehlo::Select(mask_op, neg_inf, input_op);

  return BuildSoftmaxShlo(masked_input, dim, SoftmaxMode::kSoftmax);
}

absl::StatusOr<mlir::MlirOp> BuildMaskedSoftmaxBackwardDataShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp output_op, mlir::MlirOp mask_op,
    int64_t dim, mlir::stablehlo::Precision precision) {
  const mlir::RankedTensorType output_type = GetTensorTypeOrDie(output_op);
  mlir::MlirBuilder& builder = grad_output_op.getBuilder();

  if (output_type.getRank() == 0) {
    return MakeScalarConstant(builder, 0.0, output_type.getElementType());
  }

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp grad_input_unmasked,
      BuildSoftmaxBackwardDataShlo(grad_output_op, output_op, dim, precision,
                                   SoftmaxMode::kSoftmax));

  TT_ASSIGN_OR_RETURN(
      mask_op, PrepareMaskOp(mask_op, output_op, /*mask_type=*/std::nullopt));

  mlir::MlirOp zero = MakeConstantLike(output_op, 0.0);
  return mlir::stablehlo::Select(mask_op, zero, grad_input_unmasked);
}

}  // namespace torch_tpu
