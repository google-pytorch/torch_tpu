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

#ifndef TORCH_TPU_OPS_VIEW_DECOMPOSITION_BROADCAST_PRIMITIVE_H_
#define TORCH_TPU_OPS_VIEW_DECOMPOSITION_BROADCAST_PRIMITIVE_H_

#include <stdint.h>

#include <ostream>

#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"

// Broadcast (full name: stablehlo::BroadcastInDim) is a view primitive. It
// inserts dimensions into a tensor, reorders them, and/or increases dimensions
// with size = 1 to arbitrary size.
//
// Note that this is a primitive in StableHLO; in torch, these three operations
// would need to be implemented either separately as torch.unsqueeze,
// torch.expand, and torch.permute, or as a single as_strided operation.

namespace torch_tpu {

// A broadcast primitive converts an M-dimensional tensor into an N-dimensional
// tensor by:
//   - Mapping input dimension i to output dimension j as indicated by
//     broadcast_dimensions[i] = j
//   - Inserting dimensions of size new_sizes[j] for all output dimensions j not
//     mapped to by broadcast_dimensions.
//   - Expanding all mapped dimensions j to new_sizes[j], provided that either
//     input_sizes[i] == new_sizes[j] or input_sizes[i] == 1.
//
// Any inserted or expanded dimension will have stride 0, while any
// non-expanded dimension j will have the same stride as the corresponding
// input dimension.
struct BroadcastPrimitive {
  // The shape of the input tensor.
  Dimensions base_shape;
  // The shape of the output tensor.
  Dimensions new_sizes;
  // A mapping from input dimensions to output dimensions.
  // broadcast_dimensions[i] == j means that input dimension i is mapped to
  // output dimension j.
  Dimensions broadcast_dimensions;
};
// We use Dimensions instead of std::vector to avoid heap allocations for
// small ranks.
static_assert(sizeof(BroadcastPrimitive) == 168);

// Formats the broadcast like "broadcast(new_sizes=[1, 2, 3],
// broadcast_dimensions=[1, 2])".
std::ostream& operator<<(std::ostream& os, const BroadcastPrimitive& broadcast);

// Updates the layout to reflect the effect of applying the given broadcast
// primitive.
//
// Crashes if broadcast_dimensions is not a valid permutation of the
// input dimensions, or if any expanded dimension is not 1.
//
// Returns true if the layout was modified.
bool UpdateLayout(StridedLayout& layout, const BroadcastPrimitive& broadcast);

// An overload for the ViewPrimitiveShlo function that handles broadcast
// primitives. This is a direct call into the stablehlo::BroadcastInDim
// function.
absl::StatusOr<mlir::MlirOp> ViewPrimitiveShlo(
    mlir::MlirOp input, const BroadcastPrimitive& broadcast);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_VIEW_DECOMPOSITION_BROADCAST_PRIMITIVE_H_
