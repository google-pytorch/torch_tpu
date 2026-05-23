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

#include "torch_tpu/ops/hardsigmoid/hardsigmoid_aten_kernels.h"

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

absl::StatusOr<mlir::MlirOp> BuildHardsigmoidShlo(mlir::MlirOp input_op) {
  auto type = GetTensorTypeOrDie(input_op);
  TT_ASSIGN_OR_RETURN(auto element_type,
                      ConvertTo<mlir::ElementType>(type.getElementType()));

  // hardsigmoid(x) = clamp(0, x * (1/6) + 0.5, 1)
  mlir::MlirOp zero = MakeConstantLike(input_op, 0.0, element_type);
  mlir::MlirOp one_sixth = MakeConstantLike(input_op, 1.0 / 6.0, element_type);
  mlir::MlirOp half = MakeConstantLike(input_op, 0.5, element_type);
  mlir::MlirOp one = MakeConstantLike(input_op, 1.0, element_type);

  mlir::MlirOp x_scaled = mlir::stablehlo::Mul(input_op, one_sixth);
  mlir::MlirOp x_shifted = mlir::stablehlo::Add(x_scaled, half);
  return mlir::stablehlo::Clamp(zero, x_shifted, one);
}

absl::StatusOr<mlir::MlirOp> BuildHardsigmoidBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp self) {
  auto type = GetTensorTypeOrDie(self);
  TT_ASSIGN_OR_RETURN(auto element_type,
                      ConvertTo<mlir::ElementType>(type.getElementType()));

  // grad = 1/6 if -3 < x < 3 else 0
  mlir::MlirOp neg_three = MakeConstantLike(self, -3.0, element_type);
  mlir::MlirOp three = MakeConstantLike(self, 3.0, element_type);

  mlir::MlirOp cond_le = mlir::stablehlo::Compare(
      self, neg_three, mlir::stablehlo::ComparisonDirection::LE);
  mlir::MlirOp cond_ge = mlir::stablehlo::Compare(
      self, three, mlir::stablehlo::ComparisonDirection::GE);

  mlir::MlirOp zero = MakeConstantLike(self, 0.0, element_type);
  mlir::MlirOp one_sixth = MakeConstantLike(self, 1.0 / 6.0, element_type);

  mlir::MlirOp mid_grad = mlir::stablehlo::Mul(grad_output, one_sixth);

  mlir::MlirOp out_of_bounds = mlir::stablehlo::Or(cond_le, cond_ge);
  return mlir::stablehlo::Select(out_of_bounds, zero, mid_grad);
}

}  // namespace

at::Tensor& AtenHardsigmoidOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kHardsigmoidOut, _, (self, out), {
    TT_CHECK_THROW(self.is_floating_point(), error::kInvalidArgument)
        << "expected the input dtype to be floating point, got "
        << ToString(self.scalar_type());
    TT_THROW_IF_ERROR(
        UnaryOpOut(self, out, BuildHardsigmoidShlo,
                   {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

at::Tensor& AtenHardsigmoidBackwardGradInput(const at::Tensor& grad_output,
                                             const at::Tensor& self,
                                             at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kHardsigmoidBackwardGradInput, _, (grad_output, self, grad_input),
      {
        TT_CHECK_THROW(self.is_floating_point(), error::kInvalidArgument)
            << "expected the input dtype to be floating point, got "
            << ToString(self.scalar_type());
        TT_ASSIGN_OR_THROW(const auto output_mlir_type,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        TT_ASSIGN_OR_THROW(
            auto result,
            (DispatchOp<2>(
                [](FixedSizeSpan<mlir::MlirOp, 2> inputs)
                    -> absl::StatusOr<mlir::MlirOp> {
                  auto& [grad_output_op, self_op] = inputs;
                  return BuildHardsigmoidBackwardShlo(grad_output_op, self_op);
                },
                {grad_output, self},
                {.out_dtype = output_mlir_type,
                 .out_dims = self.sizes(),
                 .op_param_cache_keys = OpParamCacheKeys::Empty()})));

        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result), grad_input));
        return grad_input;
      });
}

}  // namespace torch_tpu
