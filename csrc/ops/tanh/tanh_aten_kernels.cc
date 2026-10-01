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

#include "csrc/ops/tanh/tanh_aten_kernels.h"

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/result_type.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/binary_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/unary.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

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

  TT_ASSIGN_OR_RETURN(
      (auto [bcast_grad_output, bcast_conj_one_minus_output_squared]),
      ApplyBroadcastIfNeeded(grad_output_op, conj_one_minus_output_squared));

  // Calculate grad_output * conj(1.0 - output_squared)
  return mlir::stablehlo::Mul(bcast_grad_output,
                              bcast_conj_one_minus_output_squared);
}

absl::Status ValidateTanhBackwardInputs(const at::Tensor& grad_output,
                                        const at::Tensor& output) {
  const c10::ScalarType promoted_type = at::result_type(grad_output, output);
  TT_RET_CHECK(
      at::isFloatingType(promoted_type) || at::isComplexType(promoted_type),
      error::kPythonNotImplementedError)
      << "not implemented for " << ToString(promoted_type);
  return absl::OkStatus();
}

}  // namespace

at::Tensor& AtenTanhBackwardGradInput(const at::Tensor& grad_output,
                                      const at::Tensor& output,
                                      at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kTanhBackwardGradInput, _, (grad_output, output, grad_input), {
        TT_THROW_IF_ERROR(ValidateTanhBackwardInputs(grad_output, output));
        TT_THROW_IF_ERROR(BinaryOpOut(
            grad_output, output, grad_input, BuildTanhBackwardGradInputShlo,
            {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
        return grad_input;
      });
}
}  // namespace torch_tpu
