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

#include "torch_tpu/ops/sigmoid/sigmoid_aten_kernels.h"

#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
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
#include "torch_tpu/ops/unary.h"
#include "torch_tpu/ops/unary_aten_kernels.h"

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

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildSigmoidShlo(mlir::MlirOp input_op) {
  // Convert integral dtypes to float since StableHLO doesn't support
  // Logistic for integer dtypes.
  TT_ASSIGN_OR_RETURN(input_op,
                      ConvertIfInteger(input_op, mlir::ElementType::F32));
  return mlir::stablehlo::Logistic(input_op);
}

at::Tensor& AtenSigmoidOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kSigmoidOut, _, (self, out), {
    TT_THROW_IF_ERROR(
        UnaryOpOut(self, out, BuildSigmoidShlo,
                   {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
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
            error::kUnimplemented)
            << "not implemented for input type "
            << ToString(output.scalar_type());
        TT_ASSIGN_OR_THROW(  // ERROR_COV_INFEASIBLE=all dtypes are supported.
            const auto output_mlir_type,
            ConvertTo<mlir::ElementType>(output.scalar_type()));
        TT_ASSIGN_OR_THROW(  // ERROR_COV_INFEASIBLE=errors should be covered
                             // inside.
            auto result,
            (DispatchOp<2>(
                [](FixedSizeSpan<mlir::MlirOp, 2> inputs)
                    -> absl::StatusOr<mlir::MlirOp> {
                  auto& [grad_output_op, output_op] = inputs;
                  return BuildSigmoidBackwardShlo(grad_output_op, output_op);
                },
                {grad_output, output},
                {.out_dtype = output_mlir_type,
                 .out_dims = grad_input.sizes(),
                 .op_param_cache_keys = OpParamCacheKeys::Empty()})));
        TT_THROW_IF_ERROR(  // ERROR_COV_INFEASIBLE=errors should be covered
                            // inside.
            AssignBufferToAtTensor(std::move(result), grad_input));
        return grad_input;
      });
}

}  // namespace torch_tpu
