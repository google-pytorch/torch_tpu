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

#include "csrc/ops/gru/gru_common.h"

#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

GruGateAdjoints ComputeGruGateAdjoints(mlir::MlirOp delta_h, mlir::MlirOp hx,
                                       mlir::MlirOp resetgate,
                                       mlir::MlirOp updategate,
                                       mlir::MlirOp newgate, mlir::MlirOp h_n,
                                       mlir::MlirOp one) {
  // hy = (hx - n) * z + n = hx * z + n * (1 - z)
  // delta_z = delta_h * (hx - n)
  // delta_n = delta_h * (1 - z)
  // delta_hx_skip = delta_h * z
  mlir::MlirOp hx_minus_n = mlir::stablehlo::Subtract(hx, newgate);
  mlir::MlirOp d_updategate = mlir::stablehlo::Mul(delta_h, hx_minus_n);
  mlir::MlirOp one_minus_z = mlir::stablehlo::Subtract(one, updategate);
  mlir::MlirOp d_newgate = mlir::stablehlo::Mul(delta_h, one_minus_z);
  mlir::MlirOp delta_hx_skip = mlir::stablehlo::Mul(delta_h, updategate);

  // through new-gate tanh: delta_pre_n = delta_n * (1 - n^2)
  mlir::MlirOp delta_pre_n = TanhBackward(d_newgate, newgate, one);

  // reset gate: grad flows through r * h_n
  // pre_n = i_n + r * h_n
  // delta_r = delta_pre_n * h_n
  // delta_pre_r = delta_r * r * (1 - r)
  mlir::MlirOp d_reset = mlir::stablehlo::Mul(delta_pre_n, h_n);
  mlir::MlirOp delta_pre_r = SigmoidBackward(d_reset, resetgate, one);

  // update gate: delta_pre_z = delta_z * z * (1 - z)
  mlir::MlirOp delta_pre_z = SigmoidBackward(d_updategate, updategate, one);

  // hidden new-gate: delta_h_n = delta_pre_n * r (asymmetry vs input new-gate)
  mlir::MlirOp delta_h_n = mlir::stablehlo::Mul(delta_pre_n, resetgate);

  return {delta_pre_r, delta_pre_z, delta_pre_n, delta_h_n, delta_hx_skip};
}

}  // namespace torch_tpu
