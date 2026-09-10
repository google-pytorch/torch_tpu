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

#include "csrc/ops/sigmoid/sigmoid_aten_kernels.h"

#include <tuple>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "csrc/ops/unary.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

// Sigmoid backward formula:
// d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
// So, grad_input = grad_output * conj(output * (1 - output)),
// where output = sigmoid(x).
absl::StatusOr<mlir::MlirOp> BuildSigmoidBackwardShlo(mlir::MlirOp grad_output,
                                                      mlir::MlirOp output) {
  auto type = GetTensorTypeOrDie(output);
  TT_ASSIGN_OR_RETURN(auto element_type,
                      ConvertTo<mlir::ElementType>(type.getElementType()));
  auto one =
      MakeConstant(grad_output.getBuilder(), 1, element_type, type.getShape());
  auto sub = mlir::stablehlo::Subtract(one, output);
  auto mul = mlir::stablehlo::Mul(output, sub);

  // For complex numbers, the gradient is grad_output * conj(f'(x))
  TT_ASSIGN_OR_RETURN(auto conj_mul, BuildConjPhysicalShlo(mul));

  return mlir::stablehlo::Mul(grad_output, conj_mul);
}

absl::Status LogSigmoidForward(const at::Tensor& self, at::Tensor& out,
                               at::Tensor& buffer) {
  TT_RET_CHECK(self.is_floating_point(), error::kInvalidArgument)
      << "expected the input dtype to be floating point, got "
      << ToString(self.scalar_type());

  TT_RETURN_IF_ERROR(ResizeTensor(out, self.sizes()));
  TT_RETURN_IF_ERROR(ResizeTensor(buffer, self.sizes()));

  auto op_builder =
      [](mlir::MlirOp self_op) -> absl::StatusOr<MlirOpResults<2>> {
    TT_ASSIGN_OR_RETURN(self_op,
                        ConvertIfInteger(self_op, mlir::ElementType::F32));

    // Stable formula for log_sigmoid:
    // log_sigmoid(x) = -softplus(-x) = -(max(-x, 0) + log1p(exp(-|x|)))
    auto neg_self_op = mlir::stablehlo::Neg(self_op);
    auto zero = MakeConstantLike(self_op, 0.0);
    auto max_val = mlir::stablehlo::Max(neg_self_op, zero);

    auto is_nan = mlir::stablehlo::Compare(
        neg_self_op, neg_self_op, mlir::stablehlo::ComparisonDirection::NE);

    auto abs_val = mlir::stablehlo::Abs(self_op);
    auto neg_abs = mlir::stablehlo::Neg(abs_val);
    auto exp_val = mlir::stablehlo::Exp(neg_abs);
    auto log1p_val = mlir::stablehlo::Log1p(exp_val);

    auto sum_val = mlir::stablehlo::Add(max_val, log1p_val);

    auto res = mlir::stablehlo::Select(is_nan, neg_self_op, sum_val);
    auto output = mlir::stablehlo::Neg(res);

    return MlirOpResults<2>{output, exp_val};
  };

  c10::ScalarType out_scalar_type = self.scalar_type();
  if (c10::isIntegralType(out_scalar_type, /*includeBool=*/true)) {
    out_scalar_type = c10::ScalarType::Float;
  }
  TT_ASSIGN_OR_RETURN(mlir::ElementType dtype,
                      ConvertTo<mlir::ElementType>(out_scalar_type));

  return DispatchOpOut<1, 2>(
      std::move(op_builder), self, {out, buffer},
      {.out_dtypes = {dtype, dtype},
       .out_dims_list = {self.sizes(), self.sizes()},
       .op_param_cache_keys = OpParamCacheKeys::Empty()});
}

absl::Status LogSigmoidBackward(const at::Tensor& grad_output,
                                const at::Tensor& self,
                                const at::Tensor& buffer,
                                at::Tensor& grad_input) {
  TT_RET_CHECK(self.is_floating_point(), error::kInvalidArgument)
      << "expected the input dtype to be floating point, got "
      << ToString(self.scalar_type());

  auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [grad_output_op, self_op, buffer_op] = inputs;

    mlir::MlirOp float_self_op = self_op;
    TT_ASSIGN_OR_RETURN(
        float_self_op, ConvertIfInteger(float_self_op, mlir::ElementType::F32));

    auto zero = MakeConstantLike(float_self_op, 0.0);
    auto one = MakeConstantLike(float_self_op, 1.0);
    auto neg_one = MakeConstantLike(float_self_op, -1.0);

    auto self_lt_zero = mlir::stablehlo::Compare(
        float_self_op, zero, mlir::stablehlo::ComparisonDirection::LT);

    auto max_deriv = mlir::stablehlo::Select(self_lt_zero, one, zero);
    auto sign = mlir::stablehlo::Select(self_lt_zero, one, neg_one);

    auto one_plus_buffer = mlir::stablehlo::Add(one, buffer_op);
    auto buffer_div = mlir::stablehlo::Div(buffer_op, one_plus_buffer);
    auto sign_mul = mlir::stablehlo::Mul(sign, buffer_div);
    auto deriv = mlir::stablehlo::Subtract(max_deriv, sign_mul);

    return mlir::stablehlo::Mul(grad_output_op, deriv);
  };

  c10::ScalarType out_scalar_type = grad_output.scalar_type();
  TT_ASSIGN_OR_RETURN(mlir::ElementType dtype,
                      ConvertTo<mlir::ElementType>(out_scalar_type));

  DispatchOpOptions<1> options = {
      .out_dtype = dtype,
      .out_dims = grad_output.sizes(),
      .op_param_cache_keys = OpParamCacheKeys::Empty(),
  };
  return DispatchOpOut<3>(std::move(op_builder), {grad_output, self, buffer},
                          grad_input, std::move(options));
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildSigmoidShlo(mlir::MlirOp input_op) {
  // TODO: We should not be converting unsupported dtypes to support certain
  // dtypes. This leads to unexpected precision loss.
  // Convert integral dtypes to float since StableHLO doesn't support
  // Logistic for integer dtypes.
  TT_ASSIGN_OR_RETURN(input_op,
                      ConvertIfInteger(input_op, mlir::ElementType::F32));
  return mlir::stablehlo::Logistic(input_op);
}

at::Tensor& AtenSigmoidOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kSigmoidOut, _, (self, out), {
    const auto comp_type = ToAccumulateType(self.scalar_type());
    TT_ASSIGN_OR_THROW(const auto computation_dtype,
                       ConvertTo<mlir::ElementType>(comp_type));
    TT_THROW_IF_ERROR(
        UnaryOpOut(self, out, BuildSigmoidShlo,
                   {.op_param_cache_keys = OpParamCacheKeys::Empty(),
                    .computation_dtype = computation_dtype}));
    return out;
  });
}

at::Tensor& AtenSigmoidBackwardGradInput(const at::Tensor& grad_output,
                                         const at::Tensor& output,
                                         at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kSigmoidBackwardGradInput, _, (grad_output, output, grad_input), {
        TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=input checked during forward
                         // pass. It is guaranteed to be floating point or
                         // complex in the backward pass.
            output.is_floating_point() || output.is_complex(),
            error::kPythonNotImplementedError)
            << "not implemented for input type "
            << ToString(output.scalar_type());
        TT_ASSIGN_OR_THROW(  // ERROR_COV_INFEASIBLE=all dtypes are supported.
            const auto output_mlir_type,
            ConvertTo<mlir::ElementType>(output.scalar_type()));
        const auto comp_type = ToAccumulateType(output.scalar_type());
        TT_ASSIGN_OR_THROW(const auto computation_dtype,
                           ConvertTo<mlir::ElementType>(comp_type));

        DispatchOpOptions<1> options = {
            .out_dtype = output_mlir_type,
            .out_dims = grad_input.sizes(),
            .computation_dtype = computation_dtype,
            .op_param_cache_keys = OpParamCacheKeys::Empty(),
        };
        TT_THROW_IF_ERROR(DispatchOpOut<2>(
            [](FixedSizeSpan<mlir::MlirOp, 2> inputs)
                -> absl::StatusOr<mlir::MlirOp> {
              auto& [grad_output_op, output_op] = inputs;
              return BuildSigmoidBackwardShlo(grad_output_op, output_op);
            },
            {grad_output, output}, grad_input, std::move(options)));
        return grad_input;
      });
}

std::tuple<at::Tensor&, at::Tensor&> AtenLogSigmoidForwardOut(
    const at::Tensor& self, at::Tensor& out, at::Tensor& buffer) {
  TT_KERNEL(OpName::kLogSigmoidForwardOut, _, (self, out, buffer), {
    TT_THROW_IF_ERROR(LogSigmoidForward(self, out, buffer));
    return {out, buffer};
  });
}

std::tuple<at::Tensor, at::Tensor> AtenLogSigmoidForward(
    const at::Tensor& self) {
  TT_KERNEL(OpName::kLogSigmoidForward, _, (self), {
    TT_ASSIGN_OR_THROW(
        at::Tensor output,
        MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
    TT_ASSIGN_OR_THROW(
        at::Tensor buffer,
        MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
    TT_THROW_IF_ERROR(LogSigmoidForward(self, output, buffer));
    return {output, buffer};
  });
}

at::Tensor& AtenLogSigmoidBackwardGradInput(const at::Tensor& grad_output,
                                            const at::Tensor& self,
                                            const at::Tensor& buffer,
                                            at::Tensor& grad_input) {
  TT_KERNEL(OpName::kLogSigmoidBackwardGradInput, _,
            (grad_output, self, buffer, grad_input), {
              TT_THROW_IF_ERROR(
                  LogSigmoidBackward(grad_output, self, buffer, grad_input));
              return grad_input;
            });
}

at::Tensor AtenLogSigmoidBackward(const at::Tensor& grad_output,
                                  const at::Tensor& self,
                                  const at::Tensor& buffer) {
  TT_KERNEL(OpName::kLogSigmoidBackward, _, (grad_output, self, buffer), {
    TT_ASSIGN_OR_THROW(
        at::Tensor grad_input,
        MakeEmptyTensor(grad_output.sizes(), grad_output.scalar_type(),
                        grad_output.device()));
    TT_THROW_IF_ERROR(
        LogSigmoidBackward(grad_output, self, buffer, grad_input));
    return grad_input;
  });
}

}  // namespace torch_tpu
