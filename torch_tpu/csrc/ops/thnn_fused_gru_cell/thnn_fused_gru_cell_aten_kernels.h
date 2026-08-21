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

#ifndef TORCH_TPU_OPS_THNN_FUSED_GRU_CELL_THNN_FUSED_GRU_CELL_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_THNN_FUSED_GRU_CELL_THNN_FUSED_GRU_CELL_ATEN_KERNELS_H_

#include <optional>
#include <tuple>

#include "ATen/core/TensorBody.h"

namespace torch_tpu {

// Fused GRU cell, matching aten::_thnn_fused_gru_cell (the op stock nn.GRU
// lowers to on the TPU/PrivateUse1 backend; see pytorch RNN.cpp GRUCell).
// From the input/hidden gate pre-activations and previous hidden state hx:
//   r = sigmoid(i_r + h_r);  z = sigmoid(i_z + h_z)
//   n = tanh(i_n + r * h_n)                    # reset gates only the hidden n
//   hy = (hx - n) * z + n
// Returns (hy, workspace); workspace = concat(r, z, n, hx, h_n) for backward.
std::tuple<at::Tensor, at::Tensor> AtenThnnFusedGruCell(
    const at::Tensor& input_gates, const at::Tensor& hidden_gates,
    const at::Tensor& hx, const std::optional<at::Tensor>& input_bias,
    const std::optional<at::Tensor>& hidden_bias);

std::tuple<at::Tensor&, at::Tensor&> AtenThnnFusedGruCellOut(
    const at::Tensor& input_gates, const at::Tensor& hidden_gates,
    const at::Tensor& hx, const std::optional<at::Tensor>& input_bias,
    const std::optional<at::Tensor>& hidden_bias, at::Tensor& out0,
    at::Tensor& out1);

// Backward of the fused GRU cell (aten::_thnn_fused_gru_cell_backward).
// Returns (grad_input_gates, grad_hidden_gates, grad_hx, grad_input_bias,
// grad_hidden_bias). Unlike LSTM the input/hidden gate grads differ (the reset
// gate multiplies the hidden new-gate), so both are returned explicitly.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
AtenThnnFusedGruCellBackward(const at::Tensor& grad_hy,
                             const at::Tensor& workspace, bool has_bias);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_THNN_FUSED_GRU_CELL_THNN_FUSED_GRU_CELL_ATEN_KERNELS_H_
