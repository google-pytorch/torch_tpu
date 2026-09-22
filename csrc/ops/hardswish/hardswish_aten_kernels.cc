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

#include "csrc/ops/hardswish/hardswish_aten_kernels.h"

#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

absl::StatusOr<mlir::MlirOp> BuildAddShlo(mlir::MlirOp lhs, mlir::MlirOp rhs) {
  TT_ASSIGN_OR_RETURN((auto [op1, op2]), ApplyBroadcastIfNeeded(lhs, rhs));
  return mlir::stablehlo::Add(op1, op2);
}

absl::StatusOr<mlir::MlirOp> BuildMulShlo(mlir::MlirOp lhs, mlir::MlirOp rhs) {
  TT_ASSIGN_OR_RETURN((auto [op1, op2]), ApplyBroadcastIfNeeded(lhs, rhs));
  return mlir::stablehlo::Mul(op1, op2);
}

absl::StatusOr<mlir::MlirOp> BuildLeShlo(mlir::MlirOp lhs, mlir::MlirOp rhs) {
  TT_ASSIGN_OR_RETURN((auto [op1, op2]), ApplyBroadcastIfNeeded(lhs, rhs));
  return mlir::stablehlo::Compare(op1, op2,
                                  mlir::stablehlo::ComparisonDirection::LE);
}

absl::StatusOr<mlir::MlirOp> BuildGeShlo(mlir::MlirOp lhs, mlir::MlirOp rhs) {
  TT_ASSIGN_OR_RETURN((auto [op1, op2]), ApplyBroadcastIfNeeded(lhs, rhs));
  return mlir::stablehlo::Compare(op1, op2,
                                  mlir::stablehlo::ComparisonDirection::GE);
}

absl::StatusOr<mlir::MlirOp> BuildWhereShlo(
    mlir::MlirOp condition, mlir::MlirOp input, mlir::MlirOp other,
    const mlir::ElementType output_element_type) {
  mlir::MLIRContext& ctx = condition.getContext();
  const mlir::RankedTensorType condition_type = GetTensorTypeOrDie(condition);
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const mlir::RankedTensorType other_type = GetTensorTypeOrDie(other);

  if (!IsBooleanType(condition_type)) {
    condition =
        mlir::stablehlo::ConvertElementType(condition, mlir::ElementType::PRED);
  }
  const mlir::Type output_type = mlir::getElementType(ctx, output_element_type);
  if (input_type.getElementType() != output_type) {
    input = mlir::stablehlo::ConvertElementType(input, output_type);
  }
  if (other_type.getElementType() != output_type) {
    other = mlir::stablehlo::ConvertElementType(other, output_type);
  }

  TT_ASSIGN_OR_RETURN((auto [bcast_condition, bcast_input, bcast_other]),
                      ApplyBroadcastIfNeeded(condition, input, other));

  return mlir::stablehlo::Select(bcast_condition, bcast_input, bcast_other);
}

absl::StatusOr<mlir::MlirOp> BuildHardswishShlo(mlir::MlirOp input_op) {
  const auto type = GetTensorTypeOrDie(input_op);
  TT_ASSIGN_OR_RETURN(const auto element_type,
                      ConvertTo<mlir::ElementType>(type.getElementType()));

  mlir::MlirBuilder& builder = input_op.getBuilder();
  mlir::MlirOp zero = MakeScalarConstant(builder, 0.0, element_type);
  mlir::MlirOp three = MakeScalarConstant(builder, 3.0, element_type);
  mlir::MlirOp six = MakeScalarConstant(builder, 6.0, element_type);
  mlir::MlirOp one_sixth = MakeScalarConstant(builder, 1.0 / 6.0, element_type);

  TT_ASSIGN_OR_RETURN(mlir::MlirOp x_plus_3, BuildAddShlo(input_op, three));
  mlir::MlirOp relu6 = mlir::stablehlo::Clamp(zero, x_plus_3, six);
  mlir::MlirOp num = mlir::stablehlo::Mul(input_op, relu6);
  return BuildMulShlo(num, one_sixth);
}

absl::StatusOr<mlir::MlirOp> BuildHardswishBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp self) {
  const auto type = GetTensorTypeOrDie(self);
  TT_ASSIGN_OR_RETURN(const auto element_type,
                      ConvertTo<mlir::ElementType>(type.getElementType()));

  mlir::MlirBuilder& builder = self.getBuilder();
  mlir::MlirOp neg_three = MakeScalarConstant(builder, -3.0, element_type);
  mlir::MlirOp three = MakeScalarConstant(builder, 3.0, element_type);
  mlir::MlirOp zero = MakeScalarConstant(builder, 0.0, element_type);
  mlir::MlirOp one_third = MakeScalarConstant(builder, 1.0 / 3.0, element_type);
  mlir::MlirOp half = MakeScalarConstant(builder, 0.5, element_type);

  TT_ASSIGN_OR_RETURN(mlir::MlirOp cond_less, BuildLeShlo(self, neg_three));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp cond_greater, BuildGeShlo(self, three));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp self_over_three,
                      BuildMulShlo(self, one_third));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mid_grad_factor,
                      BuildAddShlo(self_over_three, half));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp mid_grad,
                      BuildMulShlo(grad_output, mid_grad_factor));

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp res2,
      BuildWhereShlo(cond_greater, grad_output, mid_grad, element_type));
  return BuildWhereShlo(cond_less, zero, res2, element_type);
}

}  // namespace

at::Tensor AtenHardswish(const at::Tensor& self) {
  TT_KERNEL(OpName::kHardswish, _, (self), {
    // CUDA only supports floats for hardswish.
    TT_CHECK_THROW(self.is_floating_point(), error::kInvalidArgument)
        << "expected the input dtype to be floating point, got "
        << ToString(self.scalar_type());
    TT_ASSIGN_OR_THROW(
        at::Tensor result,
        UnaryOp(self, BuildHardswishShlo,
                {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return result;
  });
}

at::Tensor& AtenHardswishOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kHardswishOut, _, (self, out), {
    TT_CHECK_THROW(self.is_floating_point(), error::kInvalidArgument)
        << "expected the input dtype to be floating point, got "
        << ToString(self.scalar_type());
    TT_THROW_IF_ERROR(
        UnaryOpOut(self, out, BuildHardswishShlo,
                   {.op_param_cache_keys = OpParamCacheKeys::Empty(),
                    .allow_out_dtype_cast = false}));
    return out;
  });
}

at::Tensor& AtenHardswish_(at::Tensor& self) {
  TT_KERNEL(OpName::kHardswish_, _, (self), {
    TT_CHECK_THROW(self.is_floating_point(), error::kInvalidArgument)
        << "expected the input dtype to be floating point, got "
        << ToString(self.scalar_type());
    TT_THROW_IF_ERROR(
        UnaryOpInPlace(self, BuildHardswishShlo,
                       {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return self;
  });
}

at::Tensor AtenHardswishBackward(const at::Tensor& grad_output,
                                 const at::Tensor& self) {
  TT_KERNEL(OpName::kHardswishBackward, _, (grad_output, self), {
    TT_CHECK_THROW(self.is_floating_point(), error::kInvalidArgument)
        << "expected the input dtype to be floating point, got "
        << ToString(self.scalar_type());
    TT_ASSIGN_OR_THROW(const auto output_mlir_type,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_ASSIGN_OR_THROW(
        auto result, (DispatchOp<2>(
                         [](FixedSizeSpan<mlir::MlirOp, 2> inputs)
                             -> absl::StatusOr<mlir::MlirOp> {
                           auto& [grad_output_op, self_op] = inputs;
                           return BuildHardswishBackwardShlo(grad_output_op,
                                                             self_op);
                         },
                         {grad_output, self},
                         /*options=*/
                         {.out_dtype = output_mlir_type,
                          .out_dims = self.sizes(),
                          .op_param_cache_keys = OpParamCacheKeys::Empty()})));
    return MakeTensor(std::move(result));
  });
}

}  // namespace torch_tpu
