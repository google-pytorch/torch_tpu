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

#ifndef TORCH_TPU_CSRC_OPS_GRU_GRU_COMMON_H_
#define TORCH_TPU_CSRC_OPS_GRU_GRU_COMMON_H_

#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

// Computes the gradient through a sigmoid activation using its analytic
// derivative:
//   d/dz sigmoid(z) = sigmoid(z) * (1 - sigmoid(z))
//   grad_z = delta * act * (1 - act)
inline mlir::MlirOp SigmoidBackward(mlir::MlirOp delta, mlir::MlirOp act,
                                    mlir::MlirOp one) {
  mlir::MlirOp one_minus_act = mlir::stablehlo::Subtract(one, act);
  mlir::MlirOp act_term = mlir::stablehlo::Mul(act, one_minus_act);
  return mlir::stablehlo::Mul(delta, act_term);
}

// Computes the gradient through a tanh activation using its analytic
// derivative:
//   d/dz tanh(z) = 1 - tanh^2(z)
//   grad_z = delta * (1 - act^2)
inline mlir::MlirOp TanhBackward(mlir::MlirOp delta, mlir::MlirOp act,
                                 mlir::MlirOp one) {
  mlir::MlirOp act_sq = mlir::stablehlo::Mul(act, act);
  mlir::MlirOp one_minus_act_sq = mlir::stablehlo::Subtract(one, act_sq);
  return mlir::stablehlo::Mul(delta, one_minus_act_sq);
}

// Encapsulates the pre-activation gate adjoints and the hidden skip adjoint
// computed during a single recurrent GRU backward step.
struct GruGateAdjoints {
  mlir::MlirOp delta_pre_r;  // Pre-activation reset gate gradient (d_r_pre)
  mlir::MlirOp delta_pre_z;  // Pre-activation update gate gradient (d_z_pre)
  mlir::MlirOp
      delta_pre_n;         // Pre-activation candidate input gradient (d_n_pre)
  mlir::MlirOp delta_h_n;  // Pre-activation candidate hidden gradient (d_hn_pre
                           // = d_n_pre * r)
  mlir::MlirOp delta_hx_skip;  // Gradient propagating through direct skip
                               // connection (delta_h * z)
};

// Computes the pre-activation gate adjoints for a single GRU step from
// incoming hidden adjoint (delta_h) and forward step activations
// (hx, resetgate, updategate, newgate, h_n).
//
// Mathematical equations:
//   hy = (hx - n) * z + n = hx * z + n * (1 - z)
//   delta_z = delta_h * (hx - n)
//   delta_n = delta_h * (1 - z)
//   delta_hx_skip = delta_h * z
//   delta_pre_n = delta_n * (1 - n^2)
//   delta_r = delta_pre_n * h_n
//   delta_pre_r = delta_r * r * (1 - r)
//   delta_pre_z = delta_z * z * (1 - z)
//   delta_h_n = delta_pre_n * r
GruGateAdjoints ComputeGruGateAdjoints(mlir::MlirOp delta_h, mlir::MlirOp hx,
                                       mlir::MlirOp resetgate,
                                       mlir::MlirOp updategate,
                                       mlir::MlirOp newgate, mlir::MlirOp h_n,
                                       mlir::MlirOp one);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_GRU_GRU_COMMON_H_
