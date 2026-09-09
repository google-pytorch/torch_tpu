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

#ifndef TORCH_TPU_CSRC_OPS_LSTM_LSTM_COMMON_H_
#define TORCH_TPU_CSRC_OPS_LSTM_LSTM_COMMON_H_

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

// Encapsulates the 4 pre-activation gate adjoints and the previous cell state
// adjoint computed during a single recurrent LSTM backward step.
struct LstmGateAdjoints {
  mlir::MlirOp delta_pre_i;  // Pre-activation input gate gradient (d_i_pre)
  mlir::MlirOp delta_pre_f;  // Pre-activation forget gate gradient (d_f_pre)
  mlir::MlirOp
      delta_pre_g;  // Pre-activation cell/candidate gate gradient (d_g_pre)
  mlir::MlirOp delta_pre_o;  // Pre-activation output gate gradient (d_o_pre)
  mlir::MlirOp
      delta_c_prev;  // Cell gradient propagating to previous timestep (d_cx)
};

// Computes the pre-activation gate adjoints and cell state adjoint for a single
// LSTM step from incoming adjoints (delta_h, delta_c_next) and forward step
// activations (tanh_c, c_prev, i, f, g, o).
//
// Mathematical equations:
//   delta_o = delta_h * tanh(c)
//   dtanh_c = (1 - tanh(c)^2) * delta_h
//   delta_c = delta_c_next + o * dtanh_c
//   delta_c_prev = delta_c * f
//   delta_pre_i = SigmoidBackward(delta_c * g, i, 1.0)
//   delta_pre_f = SigmoidBackward(delta_c * c_prev, f, 1.0)
//   delta_pre_g = TanhBackward(delta_c * i, g, 1.0)
//   delta_pre_o = SigmoidBackward(delta_o, o, 1.0)
LstmGateAdjoints ComputeLstmGateAdjoints(mlir::MlirOp delta_h,
                                         mlir::MlirOp delta_c_next,
                                         mlir::MlirOp tanh_c,
                                         mlir::MlirOp c_prev, mlir::MlirOp i,
                                         mlir::MlirOp f, mlir::MlirOp g,
                                         mlir::MlirOp o, mlir::MlirOp one);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_LSTM_LSTM_COMMON_H_
