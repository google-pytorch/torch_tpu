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

#ifndef TORCH_TPU_CSRC_OPS_SELECT_SCATTER_SELECT_SCATTER_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_SELECT_SCATTER_SELECT_SCATTER_ATEN_KERNELS_H_

#include <cstdint>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

// Implements aten::select_scatter on TPU by lowering directly to StableHLO:
//   - Reshapes `src_op` to unsqueeze `dim` (shape `[..., 1, ...]`).
//   - If `total_len == 1`, returns `reshaped_src` directly.
//   - Otherwise emits `stablehlo::DynamicUpdateSlice(self_op, reshaped_src,
//     start_indices)` where `start_indices[dim] = index`.
at::Tensor AtenSelectScatter(const at::Tensor& self, const at::Tensor& src,
                             int64_t dim, int64_t index);

// Implements aten::select_scatter.out on TPU with direct StableHLO lowerings
// and buffer donation via DispatchOpOut.
at::Tensor& AtenSelectScatterOut(const at::Tensor& self, const at::Tensor& src,
                                 int64_t dim, int64_t index, at::Tensor& out);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_SELECT_SCATTER_SELECT_SCATTER_ATEN_KERNELS_H_
