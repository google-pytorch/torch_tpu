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

#include "torch_tpu/ops/hardswish/hardswish_aten_kernels.h"

#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/cache_key.h"
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
#include "torch_tpu/ops/unary_aten_kernels.h"

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
  auto type = GetTensorTypeOrDie(self);
  TT_ASSIGN_OR_RETURN(auto element_type,
                      ConvertTo<mlir::ElementType>(type.getElementType()));

  mlir::MlirOp neg_three = MakeConstantLike(self, -3.0, element_type);
  mlir::MlirOp three = MakeConstantLike(self, 3.0, element_type);

  mlir::MlirOp cond_less = mlir::stablehlo::Compare(
      self, neg_three, mlir::stablehlo::ComparisonDirection::LE);
  mlir::MlirOp cond_greater = mlir::stablehlo::Compare(
      self, three, mlir::stablehlo::ComparisonDirection::GE);

  mlir::MlirOp zero = MakeConstantLike(self, 0.0, element_type);
  mlir::MlirOp one_third = MakeConstantLike(self, 1.0 / 3.0, element_type);
  mlir::MlirOp half = MakeConstantLike(self, 0.5, element_type);

  mlir::MlirOp self_over_three = mlir::stablehlo::Mul(self, one_third);
  mlir::MlirOp mid_grad_factor = mlir::stablehlo::Add(self_over_three, half);
  mlir::MlirOp mid_grad = mlir::stablehlo::Mul(grad_output, mid_grad_factor);

  mlir::MlirOp res2 =
      mlir::stablehlo::Select(cond_greater, grad_output, mid_grad);
  return mlir::stablehlo::Select(cond_less, zero, res2);
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
                   {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
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
