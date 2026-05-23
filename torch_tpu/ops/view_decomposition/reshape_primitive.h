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

#ifndef TORCH_TPU_OPS_VIEW_DECOMPOSITION_RESHAPE_PRIMITIVE_H_
#define TORCH_TPU_OPS_VIEW_DECOMPOSITION_RESHAPE_PRIMITIVE_H_

#include <ostream>

#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"

// Reshape (also called "torch.view" in PyTorch) is a view primitive. It
// reinterprets a tensor shape as another tensor shape with the same number of
// elements.

namespace torch_tpu {

// A reshape primitive converts an N-dimensional tensor into an
// M-dimensional tensor with shape given by new_sizes.
//
// For this to be a valid view, the total number of elements in the shapes must
// match, and the strides must only be reshaped within "contiguity-like" blocks,
// as described in
// https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html.
// A "contiguity-like" block is one where size[i] == size[i+1] * stride[i+1] for
// some sequence of k dimensions (trivially true for k == 1).
//
// There must be an equal number of contiguity-like blocks in the input and
// output, and the element count of each block must be equal in the 1:1 mapping
// of the input and output.
//
// If this condition holds, then the resulting strides of the new shapes are:
//   * new_strides[j] == input_strides[i] if i, j are the last indices of
//     their respective contiguity-like blocks, and
//   * new_strides[j] == new_strides[j+1] * new_sizes[j+1] otherwise.
//
// Simplifications:
//   * If new_sizes == input_sizes, then this is a no-op and can be skipped.
//   * Back-to-back reshapes are redundant, only the second reshape is needed.
struct ReshapePrimitive {
  Dimensions base_sizes;
  Dimensions new_sizes;
};
static_assert(sizeof(ReshapePrimitive) == 112, "");

// Formats the reshape like "reshape[1, 2, 3]".
std::ostream& operator<<(std::ostream& os, const ReshapePrimitive& reshape);

// Updates the layout to reflect the effect of applying the given reshape
// primitive.
//
// Crashes if the reshape is not valid for the layout, because it
// does not reshape with matching element counts, or violates the
// contiguity-like criterion.
//
// Returns true if the layout was modified, or false if the reshape is a no-op.
bool UpdateLayout(StridedLayout& layout, const ReshapePrimitive& reshape);

// Merges two sequential reshapes into a single reshape.
[[nodiscard]] ReshapePrimitive Merge(ReshapePrimitive first,
                                     ReshapePrimitive second);

// An overload for the ViewPrimitiveShlo function that handles reshape
// primitives. This is a direct call into the stablehlo::Reshape function.
absl::StatusOr<mlir::MlirOp> ViewPrimitiveShlo(mlir::MlirOp input,
                                               const ReshapePrimitive& reshape);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_VIEW_DECOMPOSITION_RESHAPE_PRIMITIVE_H_
