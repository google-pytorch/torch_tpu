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

#include "torch_tpu/ops/log_sigmoid/log_sigmoid_aten_kernels.h"

#include <tuple>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "ATen/native/Resize.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
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

namespace torch_tpu {

namespace {

absl::StatusOr<MlirOpResults<2>> BuildLogSigmoidForwardShlo(
    mlir::MlirOp input_op) {
  TT_ASSIGN_OR_RETURN(input_op,
                      ConvertIfInteger(input_op, mlir::ElementType::F32));
  auto type = GetTensorTypeOrDie(input_op);
  auto element_type = type.getElementType();
  auto shape = type.getShape();
  auto& builder = input_op.getBuilder();

  // zero = Constant(0)
  auto zero = MakeConstant(builder, 0, element_type, shape);

  // min = stablehlo::Min(zero, input_op)
  auto min = mlir::stablehlo::Min(zero, input_op);

  // z = exp(-abs(input_op))
  auto abs_val = mlir::stablehlo::Abs(input_op);
  auto neg_abs = mlir::stablehlo::Neg(abs_val);
  auto z = mlir::stablehlo::Exp(neg_abs);

  // log1p_z = log1p(z)
  auto log1p_z = mlir::stablehlo::Log1p(z);

  // output = min - log1p_z
  auto output = mlir::stablehlo::Subtract(min, log1p_z);

  return MlirOpResults<2>{output, z};
}

absl::StatusOr<mlir::MlirOp> BuildLogSigmoidBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp self, mlir::MlirOp buffer) {
  auto type = GetTensorTypeOrDie(self);
  auto element_type = type.getElementType();
  auto shape = type.getShape();
  auto& builder = self.getBuilder();

  // Constants
  auto zero = MakeConstant(builder, 0, element_type, shape);
  auto one = MakeConstant(builder, 1, element_type, shape);
  auto neg_one = MakeConstant(builder, -1, element_type, shape);

  // in_negative = self < 0
  auto in_negative = mlir::stablehlo::Compare(
      self, zero, mlir::stablehlo::ComparisonDirection::LT);

  // max_deriv = where(in_negative, one, zero)
  auto max_deriv = mlir::stablehlo::Select(in_negative, one, zero);

  // sign = where(in_negative, one, neg_one)
  auto sign = mlir::stablehlo::Select(in_negative, one, neg_one);

  // one_plus_z = 1 + buffer
  auto one_plus_z = mlir::stablehlo::Add(one, buffer);

  // z_div_one_plus_z = buffer / one_plus_z
  auto z_div_one_plus_z = mlir::stablehlo::Div(buffer, one_plus_z);

  // deriv = max_deriv - sign * z_div_one_plus_z
  auto sign_mul = mlir::stablehlo::Mul(sign, z_div_one_plus_z);
  auto deriv = mlir::stablehlo::Subtract(max_deriv, sign_mul);

  return mlir::stablehlo::Mul(grad_output, deriv);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> AtenLogSigmoidForward(
    const at::Tensor& self) {
  TT_KERNEL(OpName::kLogSigmoidForward, param_keys, (self), {
    TT_CHECK_THROW(self.is_floating_point(), error::kInvalidArgument)
        << "expected the input dtype to be floating point, got "
        << ToString(self.scalar_type());

    TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));

    auto op_builder =
        [](mlir::MlirOp input) -> absl::StatusOr<MlirOpResults<2>> {
      return BuildLogSigmoidForwardShlo(input);
    };

    TT_ASSIGN_OR_THROW(
        auto results,
        (DispatchOp<1, 2>(std::move(op_builder), {self},
                          {.out_dtypes = {output_dtype, output_dtype},
                           .out_dims_list = {self.sizes(), self.sizes()},
                           .op_param_cache_keys = std::move(param_keys)})));

    return {MakeTensor(std::move(results[0])),
            MakeTensor(std::move(results[1]))};
  });
}

std::tuple<at::Tensor&, at::Tensor&> AtenLogSigmoidForwardOut(
    const at::Tensor& self, at::Tensor& output, at::Tensor& buffer) {
  TT_KERNEL(
      OpName::kLogSigmoidForwardOutput, param_keys, (self, output, buffer), {
        TT_CHECK_THROW(self.is_floating_point(), error::kInvalidArgument)
            << "expected the input dtype to be floating point, got "
            << ToString(self.scalar_type());

        TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        auto op_builder =
            [](mlir::MlirOp input) -> absl::StatusOr<MlirOpResults<2>> {
          return BuildLogSigmoidForwardShlo(input);
        };

        TT_ASSIGN_OR_THROW(
            auto results,
            (DispatchOp<1, 2>(std::move(op_builder), {self},
                              {.out_dtypes = {output_dtype, output_dtype},
                               .out_dims_list = {self.sizes(), self.sizes()},
                               .op_param_cache_keys = std::move(param_keys)})));
        // log_sigmoid is non-structured in PyTorch core (see
        // native_functions.yaml:12343), so we must manually support
        // out-resizing during eager test execution.
        at::native::resize_output(output, results[0].dimensions());
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(results[0]), output));

        at::native::resize_output(buffer, results[1].dimensions());
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(results[1]), buffer));

        return std::tie(output, buffer);
      });
}

at::Tensor AtenLogSigmoidBackward(const at::Tensor& grad_output,
                                  const at::Tensor& self,
                                  const at::Tensor& buffer) {
  TT_KERNEL(
      OpName::kLogSigmoidBackward, param_keys, (grad_output, self, buffer), {
        TT_CHECK_THROW(self.is_floating_point(), error::kInvalidArgument)
            << "expected the input dtype to be floating point, got "
            << ToString(self.scalar_type());

        TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          return BuildLogSigmoidBackwardShlo(inputs[0], inputs[1], inputs[2]);
        };

        TT_ASSIGN_OR_THROW(
            auto result, (DispatchOp<3, 1>(
                             std::move(op_builder), {grad_output, self, buffer},
                             {.out_dtype = output_dtype,
                              .out_dims = self.sizes(),
                              .op_param_cache_keys = std::move(param_keys)})));

        return MakeTensor(std::move(result));
      });
}

at::Tensor& AtenLogSigmoidBackwardGradInput(const at::Tensor& grad_output,
                                            const at::Tensor& self,
                                            const at::Tensor& buffer,
                                            at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kLogSigmoidBackwardGradInput, param_keys,
      (grad_output, self, buffer, grad_input), {
        TT_CHECK_THROW(self.is_floating_point(), error::kInvalidArgument)
            << "expected the input dtype to be floating point, got "
            << ToString(self.scalar_type());

        TT_ASSIGN_OR_THROW(mlir::ElementType output_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          return BuildLogSigmoidBackwardShlo(inputs[0], inputs[1], inputs[2]);
        };

        TT_ASSIGN_OR_THROW(
            auto result, (DispatchOp<3, 1>(
                             std::move(op_builder), {grad_output, self, buffer},
                             {.out_dtype = output_dtype,
                              .out_dims = self.sizes(),
                              .op_param_cache_keys = std::move(param_keys)})));

        // log_sigmoid is non-structured in PyTorch core (see
        // native_functions.yaml), so we must manually support out-resizing
        // during eager test execution for now.
        at::native::resize_output(grad_input, result.dimensions());
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result), grad_input));

        return grad_input;
      });
}

}  // namespace torch_tpu
