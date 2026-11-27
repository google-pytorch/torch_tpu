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

#include "ATen/core/TensorBody.h"
#include "ATen/ops/result_type.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/binary_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

absl::StatusOr<mlir::MlirOp> BuildHardswishShlo(mlir::MlirOp input_op) {
  auto type = GetTensorTypeOrDie(input_op);
  TT_ASSIGN_OR_RETURN(auto element_type,
                      ConvertTo<mlir::ElementType>(type.getElementType()));

  mlir::MlirOp three = MakeConstantLike(input_op, 3.0, element_type);
  mlir::MlirOp six = MakeConstantLike(input_op, 6.0, element_type);
  mlir::MlirOp zero = MakeConstantLike(input_op, 0.0, element_type);

  mlir::MlirOp x_plus_3 = mlir::stablehlo::Add(input_op, three);
  mlir::MlirOp relu6 = mlir::stablehlo::Clamp(zero, x_plus_3, six);
  mlir::MlirOp num = mlir::stablehlo::Mul(input_op, relu6);
  return mlir::stablehlo::Div(num, six);
}

absl::StatusOr<mlir::MlirOp> BuildHardswishBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp self) {
  TT_ASSIGN_OR_RETURN((auto [bcast_grad, bcast_self]),
                      ApplyBroadcastIfNeeded(grad_output, self));

  const mlir::ElementType out_dtype = GetElementTypeOrDie(bcast_self);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(out_dtype));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp grad,
                      CastIfNeeded(bcast_grad, compute_dtype));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp x, CastIfNeeded(bcast_self, compute_dtype));

  mlir::MlirOp neg_three = MakeConstantLike(x, -3.0, compute_dtype);
  mlir::MlirOp three = MakeConstantLike(x, 3.0, compute_dtype);

  mlir::MlirOp cond_less = mlir::stablehlo::Compare(
      x, neg_three, mlir::stablehlo::ComparisonDirection::LE);
  mlir::MlirOp cond_greater = mlir::stablehlo::Compare(
      x, three, mlir::stablehlo::ComparisonDirection::GE);

  mlir::MlirOp zero = MakeConstantLike(x, 0.0, compute_dtype);
  mlir::MlirOp one_third = MakeConstantLike(x, 1.0 / 3.0, compute_dtype);
  mlir::MlirOp half = MakeConstantLike(x, 0.5, compute_dtype);

  mlir::MlirOp self_over_three = mlir::stablehlo::Mul(x, one_third);
  mlir::MlirOp mid_grad_factor = mlir::stablehlo::Add(self_over_three, half);
  mlir::MlirOp mid_grad = mlir::stablehlo::Mul(grad, mid_grad_factor);

  mlir::MlirOp res2 = mlir::stablehlo::Select(cond_greater, grad, mid_grad);
  mlir::MlirOp result = mlir::stablehlo::Select(cond_less, zero, res2);
  return CastIfNeeded(result, out_dtype);
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
    const at::ScalarType promoted_type = at::result_type(grad_output, self);
    TT_CHECK_THROW(at::isFloatingType(promoted_type), error::kInvalidArgument)
        << "expected the input dtype to be floating point, got "
        << ToString(promoted_type);
    TT_ASSIGN_OR_THROW(
        at::Tensor result,
        BinaryOp(grad_output, self, BuildHardswishBackwardShlo,
                 {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return result;
  });
}

}  // namespace torch_tpu
