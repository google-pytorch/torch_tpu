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

#ifndef TORCH_TPU_OPS_THNN_FUSED_LSTM_CELL_THNN_FUSED_LSTM_CELL_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_THNN_FUSED_LSTM_CELL_THNN_FUSED_LSTM_CELL_ATEN_KERNELS_H_

#include <optional>
#include <tuple>

#include "ATen/core/TensorBody.h"

namespace torch_tpu {

// Fused LSTM cell, matching aten::_thnn_fused_lstm_cell (the op that stock
// nn.LSTM lowers to on the TPU / PrivateUse1 backend; see
// pytorch aten/src/ATen/native/RNN.cpp). Computes, from the pre-activation
// input/hidden gates and the previous cell state:
//   gates = input_gates + hidden_gates (+ input_bias + hidden_bias)
//   i, f, g, o = split(gates, 4)
//   cy = sigmoid(f) * cx + sigmoid(i) * tanh(g)
//   hy = sigmoid(o) * tanh(cy)
// Returns (hy, cy, workspace); workspace holds the four activated gates for
// the fused backward.
std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenThnnFusedLstmCell(
    const at::Tensor& input_gates, const at::Tensor& hidden_gates,
    const at::Tensor& cx, const std::optional<at::Tensor>& input_bias,
    const std::optional<at::Tensor>& hidden_bias);

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> AtenThnnFusedLstmCellOut(
    const at::Tensor& input_gates, const at::Tensor& hidden_gates,
    const at::Tensor& cx, const std::optional<at::Tensor>& input_bias,
    const std::optional<at::Tensor>& hidden_bias, at::Tensor& out0,
    at::Tensor& out1, at::Tensor& out2);

// Backward of the fused LSTM cell (aten::_thnn_fused_lstm_cell_backward_impl).
// Returns (grad_gates, grad_cx, grad_bias); the fused-cell backward wrapper
// duplicates grad_gates/grad_bias to the input/hidden gate/bias grads.
std::tuple<at::Tensor, at::Tensor, at::Tensor>
AtenThnnFusedLstmCellBackwardImpl(const std::optional<at::Tensor>& grad_hy,
                                  const std::optional<at::Tensor>& grad_cy,
                                  const at::Tensor& cx, const at::Tensor& cy,
                                  const at::Tensor& workspace, bool has_bias);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_THNN_FUSED_LSTM_CELL_THNN_FUSED_LSTM_CELL_ATEN_KERNELS_H_
