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

#ifndef TORCH_TPU_OPS_VIEW_DECOMPOSITION_SLICE_PRIMITIVE_H_
#define TORCH_TPU_OPS_VIEW_DECOMPOSITION_SLICE_PRIMITIVE_H_

#include <cstdint>
#include <ostream>
#include <vector>

#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"

// Slice is a view primitive. It reduces the size of a tensor by removing
// elements at the low indexes, high indexes, and evenly-spaced interior
// intervals on each axis.The stride of each axis is multiplied by the
// slice stride on that axis.

namespace torch_tpu {

// One dimension of a slice primitive.
// Provided that 0 <= start_index < limit_index <= input_sizes[i], the
// result tensor will have:
//   result_size[i] = floor((limit_index- start_index) / stride)
//   result_strides[i] = input_strides[i] * stride
// for the dimension i that this SliceDimension is assigned to.
struct SliceDimension {
  // The low index of the slice on the input axis (inclusive).
  int64_t start_index = 0;
  // The high index of the slice on the input axis (exclusive).
  int64_t limit_index = 0;
  // The number of input rows to increment for each output index (min 1).
  int64_t stride = 1;
};
static_assert(sizeof(SliceDimension) == 24, "");

bool operator==(const SliceDimension& lhs, const SliceDimension& rhs);

// Formats the slice dimension like "(start=1, limit=2, stride=3)".
std::ostream& operator<<(std::ostream& os, const SliceDimension& dim);

// A slice primitive converts an N-dimensional tensor into another
// N-dimensional tensor, where:
//   result_size[i] = floor((limit_index[i] - start_index[i]) / stride[i])
//   result_strides[i] = input_strides[i] * stride[i]
// each dimension i, provided that 0 <= start_index[i] < limit_index[i] <=
// input_sizes[i].
//
// Simplifications:
//   * If start_index[i] == 0, limit_index[i] == input_sizes[i], and
//     stride[i] == 1 for all i, then this is a no-op and can be skipped.
//   * Back-to-back slices can be combined into a single slice.
struct SlicePrimitive {
  std::vector<SliceDimension> slice_dims;
};
// Using a std::vector instead of absl::InlinedVector to keep the size of the
// SlicePrimitive below 64 bytes (one cache line on most x86 CPUs).
static_assert(sizeof(SlicePrimitive) == 24, "");

// Formats the slice like
// "slice[(start=1, limit=2, stride=3), (start=4, limit=5, stride=6)]".
std::ostream& operator<<(std::ostream& os, const SlicePrimitive& slice);

// Updates the layout to reflect the effect of applying the given slice
// primitive.
//
// Crashes if the number of dimensions does not match, or if the
// slice indexes are invalid.
//
// Returns true if the layout was modified.
bool UpdateLayout(StridedLayout& layout, const SlicePrimitive& slice);

// Merges two sequential slices into a single slice.
[[nodiscard]] SlicePrimitive Merge(SlicePrimitive first, SlicePrimitive second);

// An overload for the ViewPrimitiveShlo function that handles slice
// primitives. This is a direct call into the stablehlo::Slice function; the
// pad_type argument is ignored (but kept to allow for std::visit).
absl::StatusOr<mlir::MlirOp> ViewPrimitiveShlo(mlir::MlirOp input,
                                               const SlicePrimitive& slice);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_VIEW_DECOMPOSITION_SLICE_PRIMITIVE_H_
