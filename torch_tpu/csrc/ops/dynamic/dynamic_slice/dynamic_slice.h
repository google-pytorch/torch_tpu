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

#ifndef TORCH_TPU_OPS_DYNAMIC_DYNAMIC_SLICE_H_
#define TORCH_TPU_OPS_DYNAMIC_DYNAMIC_SLICE_H_

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

// Slices a dynamic sub-array from the input tensor at the given runtime start
// indices with static slice sizes.
//
// This operator is used in torch.compile() mode to support dynamic start
// indexing on TPU. It lowers to stablehlo.dynamic_slice.
//
// Args:
//   input: The input tensor to slice.
//   start_indices: List of 0-D (scalar) int32 or int64 tensors (all of the same
//     dtype) containing the runtime start indices for each dimension.
//   slice_sizes: Integer array containing the slice sizes for each dimension.
//
// Returns:
//   The sliced tensor with shape `slice_sizes`.
at::Tensor DynamicSlice(const at::Tensor& input, at::TensorList start_indices,
                        at::IntArrayRef slice_sizes);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_DYNAMIC_DYNAMIC_SLICE_H_
