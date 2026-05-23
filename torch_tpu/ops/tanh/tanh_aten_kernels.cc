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

#include "torch_tpu/ops/tanh/tanh_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/unary.h"

namespace torch_tpu {

namespace {

// Builds the StableHLO graph for the backward pass of tanh.
// The formula is: grad_input = grad_output * conj(1 - output^2),
// where output = tanh(input).
absl::StatusOr<mlir::MlirOp> BuildTanhBackwardGradInputShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp output_op) {
  // Calculate output * output (output^2)
  auto output_squared = mlir::stablehlo::Mul(output_op, output_op);

  // Calculate 1.0 - output_squared
  auto one_op = MakeConstantLike(output_op, 1.0);
  auto one_minus_output_squared =
      mlir::stablehlo::Subtract(one_op, output_squared);

  // For complex numbers, the gradient is grad_output * conj(1 - output^2)
  TT_ASSIGN_OR_RETURN(auto conj_one_minus_output_squared,
                      BuildConjPhysicalShlo(one_minus_output_squared));

  // Calculate grad_output * conj(1.0 - output_squared)
  return mlir::stablehlo::Mul(grad_output_op, conj_one_minus_output_squared);
}
}  // namespace

at::Tensor& AtenTanhBackwardGradInput(const at::Tensor& grad_output,
                                      const at::Tensor& output,
                                      at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kTanhBackwardGradInput, param_keys,
      (grad_output, output, grad_input), {
        TT_ASSIGN_OR_THROW(const auto out_dtype,
                           ConvertTo<mlir::ElementType>(output.scalar_type()));
        auto output_shape = CopyIntVector(output.sizes());

        auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [grad_output_op, output_op] = inputs;
          return BuildTanhBackwardGradInputShlo(grad_output_op, output_op);
        };

        TT_ASSIGN_OR_THROW(
            auto result,
            DispatchOp<2>(std::move(op_builder), {grad_output, output},
                          /*options=*/
                          {.out_dtype = out_dtype,
                           .out_dims = output_shape,
                           .op_param_cache_keys = std::move(param_keys)}));
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result), grad_input));
        return grad_input;
      });
}
}  // namespace torch_tpu
