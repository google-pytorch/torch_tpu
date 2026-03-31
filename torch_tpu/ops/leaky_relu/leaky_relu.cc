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

#include "torch_tpu/ops/leaky_relu/leaky_relu.h"

#include <utility>

#include "absl/status/statusor.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildLeakyReluShlo(mlir::MlirOp input,
                                                mlir::MlirOp negative_slope) {
  TT_ASSIGN_OR_RETURN((auto [input_bcast, negative_slope_bcast]),
                      ApplyBroadcastIfNeeded(input, negative_slope));

  auto zero = MakeConstantLike(input_bcast, 0.0);
  auto condition =
      stablehlo::Compare(input_bcast, zero, stablehlo::ComparisonDirection::GT);
  auto leaky_result = stablehlo::Mul(input_bcast, negative_slope_bcast);

  return stablehlo::Select(condition, input_bcast, leaky_result);
}

absl::StatusOr<mlir::MlirOp> BuildLeakyReluBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp self, mlir::MlirOp negative_slope) {
  TT_ASSIGN_OR_RETURN(
      (auto [grad_output_bcast, self_bcast, negative_slope_bcast]),
      ApplyBroadcastIfNeeded(grad_output, self, negative_slope));

  auto zero = MakeConstantLike(self_bcast, 0.0);
  auto condition =
      stablehlo::Compare(self_bcast, zero, stablehlo::ComparisonDirection::GT);
  auto leaky_grad = stablehlo::Mul(grad_output_bcast, negative_slope_bcast);

  return stablehlo::Select(condition, grad_output_bcast, leaky_grad);
}

}  // namespace torch_tpu
