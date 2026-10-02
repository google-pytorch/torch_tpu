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

#ifndef TORCH_TPU_CSRC_OPS_SLICE_SCATTER_SLICE_SCATTER_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_SLICE_SCATTER_SLICE_SCATTER_ATEN_KERNELS_H_

#include <cstdint>
#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

// Implements aten::slice_scatter on TPU with direct StableHLO lowerings and
// algebraic strength reductions:
//   - Fast-path 1 (Full replacement): If `step == 1 && slice_len == total_len`,
//     returns `src.clone()` or `src.to(self.options())` directly without
//     running DynamicUpdateSlice.
//   - Fast-path 2 (Boundary-anchored concatenate): For unit-stride slices
//     (`step == 1`) anchored at `start == 0` or `end == total_len` where
//     `slice_len * 2 >= total_len` (e.g. Causal LM logit slicing, KV-cache
//     prefix/suffix updates), emits `stablehlo::Concatenate` joining `src_op`
//     with `stablehlo::Slice(self_op, ...)` along `dim`. This eliminates
//     DynamicUpdateSlice index tensor creation and enables fusion in XLA.
//   - Fallback for interior slices (`step == 1`): Emits
//     `stablehlo::DynamicUpdateSlice(self_op, src_op, start_indices)`.
at::Tensor AtenSliceScatter(const at::Tensor& self, const at::Tensor& src,
                            int64_t dim, std::optional<int64_t> start,
                            std::optional<int64_t> end, int64_t step);

// Implements aten::slice_scatter.out on TPU with direct StableHLO lowerings and
// buffer donation via DispatchOpOut.
at::Tensor& AtenSliceScatterOut(const at::Tensor& self, const at::Tensor& src,
                                int64_t dim, std::optional<int64_t> start,
                                std::optional<int64_t> end, int64_t step,
                                at::Tensor& out);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_SLICE_SCATTER_SLICE_SCATTER_ATEN_KERNELS_H_
