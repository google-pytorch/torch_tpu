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

#ifndef TORCH_TPU_CSRC_OPS_SLICE_BACKWARD_SLICE_BACKWARD_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_SLICE_BACKWARD_SLICE_BACKWARD_ATEN_KERNELS_H_

#include <cstdint>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

// Implements aten::select_backward on TPU by lowering directly to a single
// StableHLO DynamicUpdateSlice into a zero-broadcast tensor (or an identity
// reshape when the selected dimension has size 1). This avoids PyTorch's
// default composite decomposition (zeros + select + copy_) while preserving
// XLA's AlgebraicSimplifier ability to fold step-by-step recurrence gradient
// accumulations into in-place buffer updates.
at::Tensor AtenSelectBackward(const at::Tensor& grad_output,
                              at::IntArrayRef input_sizes, int64_t dim,
                              int64_t index);

// Implements aten::slice_backward on TPU by lowering directly to StableHLO:
//   - 2-way `stablehlo.concatenate` with a single zero-broadcast block for
//     large boundary-anchored unit-stride slices (`step == 1`, `start == 0` or
//     `end == total_len`, and `slice_len * 2 >= total_len`), such as Causal LM
//     logit shifts (`logits[:, :-1, :]`) or RoPE half-slices.
//   - `stablehlo.dynamic_update_slice` into a zero-broadcast tensor for
//     interior or small unit-stride slices (`step == 1`), allowing XLA's
//     AlgebraicSimplifier to fold recurrence accumulation loops in-place.
//   - `stablehlo.pad` with `interior_pad = step - 1` for strided slices (`step
//   > 1`).
at::Tensor AtenSliceBackward(const at::Tensor& grad_output,
                             at::IntArrayRef input_sizes, int64_t dim,
                             int64_t start, int64_t end, int64_t step);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_SLICE_BACKWARD_SLICE_BACKWARD_ATEN_KERNELS_H_
