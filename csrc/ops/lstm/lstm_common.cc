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

#include "csrc/ops/lstm/lstm_common.h"

#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

LstmGateAdjoints ComputeLstmGateAdjoints(mlir::MlirOp delta_h,
                                         mlir::MlirOp delta_c_next,
                                         mlir::MlirOp tanh_c,
                                         mlir::MlirOp c_prev, mlir::MlirOp i,
                                         mlir::MlirOp f, mlir::MlirOp g,
                                         mlir::MlirOp o, mlir::MlirOp one) {
  // Output gate adjoint: delta_o = delta_h * tanh(c)
  mlir::MlirOp delta_o = mlir::stablehlo::Mul(delta_h, tanh_c);

  // Cell state adjoint: delta_c = delta_c_next + o * dtanh(c)
  mlir::MlirOp dtanh_c = TanhBackward(delta_h, tanh_c, one);
  mlir::MlirOp delta_c_from_h = mlir::stablehlo::Mul(o, dtanh_c);
  mlir::MlirOp delta_c = mlir::stablehlo::Add(delta_c_next, delta_c_from_h);

  // Cell state adjoint propagating to previous step: delta_c_prev = delta_c * f
  mlir::MlirOp delta_c_prev = mlir::stablehlo::Mul(delta_c, f);

  // Input gate pre-activation adjoint:
  mlir::MlirOp delta_in_i = mlir::stablehlo::Mul(delta_c, g);
  mlir::MlirOp delta_pre_i = SigmoidBackward(delta_in_i, i, one);

  // Forget gate pre-activation adjoint:
  mlir::MlirOp delta_in_f = mlir::stablehlo::Mul(delta_c, c_prev);
  mlir::MlirOp delta_pre_f = SigmoidBackward(delta_in_f, f, one);

  // Output gate pre-activation adjoint:
  mlir::MlirOp delta_pre_o = SigmoidBackward(delta_o, o, one);

  // Cell / candidate gate pre-activation adjoint:
  mlir::MlirOp delta_in_g = mlir::stablehlo::Mul(delta_c, i);
  mlir::MlirOp delta_pre_g = TanhBackward(delta_in_g, g, one);

  return {delta_pre_i, delta_pre_f, delta_pre_g, delta_pre_o, delta_c_prev};
}

}  // namespace torch_tpu
