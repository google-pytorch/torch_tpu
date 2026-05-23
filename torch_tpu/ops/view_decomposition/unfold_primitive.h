/*
 * Copyright 2025 Google LLC
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

#ifndef TORCH_TPU_OPS_VIEW_DECOMPOSITION_UNFOLD_PRIMITIVE_H_
#define TORCH_TPU_OPS_VIEW_DECOMPOSITION_UNFOLD_PRIMITIVE_H_

#include <cstdint>
#include <ostream>

#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"

// In CUDA, it is possible through the use of as_strided() or unfold() to
// create a view where some elements appear more than once at different
// indexes, but without any stride being 0 (a broadcast). For example,
// `torch.arange(3).as_strided_(x, sizes=(2, 2), strides=(1, 1))`
// will have the values of [[0, 1], [1, 2]]; the original tensor element 1
// can be accessed by either (0, 1) or (1, 0) index tuples.
//
// StableHLO does not support this sort of view manipulation. However, we
// can achieve the same logical effect with a concatenation (a copying
// operation):
// ```
//  x = torch.arange(3)
//  y = torch.cat([x[0:2], x[1:3]], dim=0)
// ```

namespace torch_tpu {

// An unfold primitive converts a tensor of shape (..., 1, M) with strides
// (..., ?, T) and offset O into a new tensor of shape (..., N, window_size),
// strides (..., window_stride * T, T) and offset O + start_index * T, where
// N = ceil((limit_index - start_index - window_size + 1) / window_stride).
//
// It is assumed that window_size > window_stride, meaning that the resulting
// tensor will typically have more elements than the original tensor, with
// elements along the dimensions being copied into repeated overlapping windows.
//
// If start_index == 0, limit_index == input_dim[-1], and
// window_size == input_dim[-1], then this is a no-op.
struct UnfoldPrimitive {
  int64_t start_index = 0;
  int64_t limit_index = 0;
  int64_t window_stride = 1;
  int64_t window_size = 1;
};
static_assert(sizeof(UnfoldPrimitive) == 32, "");

std::ostream& operator<<(std::ostream& os, const UnfoldPrimitive& unfold);

// Updates the layout to reflect the effect of applying the given unfold.
//
// Crashes if the input layout is not shaped like (..., 1, M) with
// stride 1 in the last dimension, or if the unfold dimensions would access
// out-of-bounds elements.
//
// Returns true if the layout was modified, or false if the unfold is a
// no-op.
bool UpdateLayout(StridedLayout& layout, const UnfoldPrimitive& unfold);

// An overload for the ViewPrimitiveShlo function that handles unfold
// primitives. This uses stablehlo::Slice and stablehlo::Concatenate.
absl::StatusOr<mlir::MlirOp> ViewPrimitiveShlo(mlir::MlirOp input,
                                               const UnfoldPrimitive& unfold);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_VIEW_DECOMPOSITION_UNFOLD_PRIMITIVE_H_
