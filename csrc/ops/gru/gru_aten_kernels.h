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

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "c10/util/ArrayRef.h"

namespace torch_tpu {

// Implements aten::gru.input on TPU via pure StableHLO builder operations.
// Fuses batched input projections and the recurrence loop on-chip.
// Seamlessly delegates to AtenGruInputAutograd when gradients are required.
std::tuple<at::Tensor, at::Tensor> AtenGruInput(
    const at::Tensor& input, const at::Tensor& hx, at::TensorList params,
    bool has_biases, int64_t num_layers, double dropout, bool train,
    bool bidirectional, bool batch_first);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_GRU_GRU_ATEN_KERNELS_H_
