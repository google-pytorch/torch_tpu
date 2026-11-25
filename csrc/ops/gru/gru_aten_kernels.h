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

#ifndef TORCH_TPU_CSRC_OPS_GRU_GRU_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_GRU_GRU_ATEN_KERNELS_H_

#include <cstdint>
#include <tuple>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "c10/util/ArrayRef.h"

namespace torch_tpu {

// Implements aten::gru.input on PrivateUse1 via fused StableHLO recurrence.
[[nodiscard]] std::tuple<at::Tensor, at::Tensor> AtenGruInput(
    const at::Tensor& input, const at::Tensor& hx, at::TensorList params,
    bool has_biases, int64_t num_layers, double dropout, bool train,
    bool bidirectional, bool batch_first);

// AutogradPrivateUse1 entry point for aten::gru.input across eager and
// torch.compile.
[[nodiscard]] std::tuple<at::Tensor, at::Tensor> AtenGruInputAutogradKernel(
    const at::Tensor& input, const at::Tensor& hx, at::TensorList params,
    bool has_biases, int64_t num_layers, double dropout, bool train,
    bool bidirectional, bool batch_first);

// Internal tpu::_gru_input* custom op implementations and Meta kernels.
[[nodiscard]] std::tuple<at::Tensor, at::Tensor> TpuGruInput(
    const at::Tensor& input, const at::Tensor& hx, at::TensorList params,
    bool has_biases, int64_t num_layers, double dropout, bool train,
    bool bidirectional, bool batch_first);

[[nodiscard]] std::tuple<at::Tensor, at::Tensor> MetaTpuGruInput(
    const at::Tensor& input, const at::Tensor& hx, at::TensorList params,
    bool has_biases, int64_t num_layers, double dropout, bool train,
    bool bidirectional, bool batch_first);

[[nodiscard]] std::vector<at::Tensor> TpuGruInputForwardCached(
    const at::Tensor& input, const at::Tensor& hx, at::TensorList params,
    bool has_biases, int64_t num_layers, double dropout, bool train,
    bool bidirectional, bool batch_first);

[[nodiscard]] std::vector<at::Tensor> MetaTpuGruInputForwardCached(
    const at::Tensor& input, const at::Tensor& hx, at::TensorList params,
    bool has_biases, int64_t num_layers, double dropout, bool train,
    bool bidirectional, bool batch_first);

[[nodiscard]] std::vector<at::Tensor> TpuGruInputBackward(
    const at::Tensor& grad_output, const at::Tensor& grad_hy,
    const at::Tensor& input, const at::Tensor& hx, at::TensorList params,
    bool has_biases, int64_t num_layers, double dropout, bool train,
    bool bidirectional, bool batch_first, at::TensorList cached_activations);

[[nodiscard]] std::vector<at::Tensor> MetaTpuGruInputBackward(
    const at::Tensor& grad_output, const at::Tensor& grad_hy,
    const at::Tensor& input, const at::Tensor& hx, at::TensorList params,
    bool has_biases, int64_t num_layers, double dropout, bool train,
    bool bidirectional, bool batch_first, at::TensorList cached_activations);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_GRU_GRU_ATEN_KERNELS_H_
