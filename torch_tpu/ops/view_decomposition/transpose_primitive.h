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

#ifndef TORCH_TPU_OPS_VIEW_DECOMPOSITION_TRANSPOSE_PRIMITIVE_H_
#define TORCH_TPU_OPS_VIEW_DECOMPOSITION_TRANSPOSE_PRIMITIVE_H_

#include <ostream>

#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"

// Transpose is a view primitive. It reorders the axes of a tensor, preserving
// the size and stride of each axis.
//
// Note: stablehlo::Transpose is the equivalent of torch.permute.
// torch.transpose is a special case of torch.permute where exactly 2 axes are
// swapped; stablehlo::Transpose is the more general reordering.
//
// We align to the StableHLO naming here, which is "Transpose" for the operation
// and "permutation" for the argument.

namespace torch_tpu {

// A transpose primitive converts an N-dimensionsal tensor into another
// N-dimensionsal tensor, where result_dim[i] == input_dim[permutation[i]] for
// all i.
//
// Similarly, result_stride[i] == input_strides[permutation[i]] for all i.
//
// Simplifications:
//   * If permutation == [0, 1, ..., N-1], then this is a no-op and can be
//     skipped. Note that this is always true for scalars.
//   * Back-to-back transposes can be combined into a new transpose with
//     merged_permutation[i] = first_permutation[second_permutation[i]]
struct TransposePrimitive {
  Indices permutation;
};
static_assert(sizeof(TransposePrimitive) == 56, "");

// Formats the transpose like "transpose([1, 2, 0])".
std::ostream& operator<<(std::ostream& os, const TransposePrimitive& transpose);

// Updates the layout to reflect the effect of applying the given transpose.
//
// Crashes if number of axes does not match, or if the permutation
// is not one-to-one with permuted axes.
//
// Returns true if the layout was modified, or false if the transpose is a
// no-op.
bool UpdateLayout(StridedLayout& layout, const TransposePrimitive& transpose);

// An overload for the ViewPrimitiveShlo function that handles transpose
// primitives. This is a direct call into the stablehlo::Transpose function.
absl::StatusOr<mlir::MlirOp> ViewPrimitiveShlo(
    mlir::MlirOp input, const TransposePrimitive& transpose);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_VIEW_DECOMPOSITION_TRANSPOSE_PRIMITIVE_H_
