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

#ifndef TORCH_TPU_OPS_STRIDE_STRIDE_HELP_H_
#define TORCH_TPU_OPS_STRIDE_STRIDE_HELP_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/dimension_types.h"

namespace torch_tpu {

// Calculates the memory strides for a tensor of a given shape, assuming a
// row-major contiguous memory layout.
//
// Stride defines the number of elements to skip in a contiguous memory block
// to advance one index in a specific dimension. For example, a tensor with
// shape {2, 3, 4} will have strides {12, 4, 1}.
//
// Args:
//   shape: A span of integers representing the dimensions of the tensor.
//
// Returns:
//   A vector of integers containing the calculated stride for each dimension.
//   Returns an empty vector if the input shape is empty.
Strides CalculateStridesContiguous(absl::Span<const int64_t> shape);

// Checks if the given set of sizes and strides is overlapping.
// If allow_expanded is true, then dimensions with stride=0 are considered
// non-overlapping; otherwise, they are considered overlapping.
// Crashes if the sizes and strides do not have the same length.
bool IsOverlapping(absl::Span<const int64_t> sizes,
                   absl::Span<const int64_t> strides,
                   bool allow_expanded = false);

// Checks that the provided sizes, strides, and storage offset fit within the
// provided number of storage bytes.
absl::Status CheckProvidedLayoutDataFitsInStorage(
    int64_t storage_numel, mlir::ElementType storage_element_type,
    Dimensions sizes, Strides strides, int64_t storage_offset,
    mlir::ElementType layout_element_type);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_STRIDE_STRIDE_HELP_H_
