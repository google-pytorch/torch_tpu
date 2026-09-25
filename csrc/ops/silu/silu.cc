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

#include "csrc/ops/silu/silu.h"

#include "ATen/OpMathType.h"
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildSiluShlo(mlir::MlirOp input_op) {
  // Only defined for floating point types
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);

  TT_ASSIGN_OR_RETURN(const at::ScalarType scalar_type,
                      ConvertTo<c10::ScalarType>(input_type.getElementType()));

  ABSL_CHECK(IsFloatType(input_type))  // CRASH_OK=checked in AtenSiluOut
      << "Expected the input dtype to be floating point, got "
      << ToString(scalar_type) << ". This is a TorchTPU bug.";

  // Defined as silu(x) = x * sigmoid(x):
  mlir::MlirOp sigmoid_op = stablehlo::Logistic(input_op);
  mlir::MlirOp result_op = stablehlo::Mul(input_op, sigmoid_op);
  return result_op;
}

absl::StatusOr<mlir::MlirOp> BuildSiluBackwardShlo(mlir::MlirOp grad_output_op,
                                                   mlir::MlirOp input_op) {
  const mlir::ElementType out_dtype = GetElementTypeOrDie(input_op);
  const at::ScalarType out_scalar_type = ConvertTo<at::ScalarType>(out_dtype);
  const at::ScalarType computation_scalar_type =
      at::toOpMathType(out_scalar_type);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType computation_dtype,
                      ConvertTo<mlir::ElementType>(computation_scalar_type));
  mlir::MlirOp comp_grad_output =
      stablehlo::ConvertElementType(grad_output_op, computation_dtype);
  mlir::MlirOp comp_input =
      stablehlo::ConvertElementType(input_op, computation_dtype);

  // Mathematical formulation of the SiLU backward pass:
  //   silu(x) = x * \sigma(x), where \sigma(x) = 1 / (1 + exp(-x)).
  //   d(silu(x)) / dx = \sigma(x) * (1 + x * (1 - \sigma(x)))
  //   grad_input = grad_output * d(silu(x)) / dx
  //
  // Computing this in a single StableHLO kernel in FP32 (`computation_dtype`)
  // replaces `math_silu_backward`'s 6 separate ATen kernel dispatches and 6
  // intermediate HBM tensor allocations per SwiGLU MLP layer with a single
  // VMEM-fused VPU pass.
  mlir::MlirOp one = MakeConstantLike(comp_input, 1.0);
  mlir::MlirOp sigmoid_op = stablehlo::Logistic(comp_input);
  mlir::MlirOp one_minus_sigmoid = stablehlo::Subtract(one, sigmoid_op);
  mlir::MlirOp x_times_one_minus_sigmoid =
      stablehlo::Mul(comp_input, one_minus_sigmoid);
  mlir::MlirOp one_plus_term = stablehlo::Add(one, x_times_one_minus_sigmoid);
  mlir::MlirOp dsilu_dx = stablehlo::Mul(sigmoid_op, one_plus_term);

  TT_ASSIGN_OR_RETURN((auto [bcast_grad_output, bcast_dsilu_dx]),
                      ApplyBroadcastIfNeeded(comp_grad_output, dsilu_dx));
  mlir::MlirOp grad_input = stablehlo::Mul(bcast_grad_output, bcast_dsilu_dx);
  return stablehlo::ConvertElementType(grad_input, out_dtype);
}

}  // namespace torch_tpu
