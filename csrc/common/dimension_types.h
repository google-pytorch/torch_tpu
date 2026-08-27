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

#ifndef TORCH_TPU_COMMON_DIMENSION_TYPES_H_
#define TORCH_TPU_COMMON_DIMENSION_TYPES_H_

#include <cstdint>

#include "absl/container/inlined_vector.h"

namespace torch_tpu {

// Compared with std::vector, this avoids a heap allocation when the vector has
// <= 6 elements. The threshold 6 is chosen to match XLA:
// See https://github.com/openxla/xla/blob/main/xla/util.h, DimensionVector
constexpr int kNumInlinedDimensions = 6;
using SmallInt64Vector = absl::InlinedVector<int64_t, kNumInlinedDimensions>;

// Holds the size array of a tensor efficiently.
using Dimensions = absl::InlinedVector<int64_t, kNumInlinedDimensions>;

// Holds the strides array of a tensor efficiently.
using Strides = absl::InlinedVector<int64_t, kNumInlinedDimensions>;

// Holds an array of indices efficiently.
using Indices = absl::InlinedVector<int64_t, kNumInlinedDimensions>;

// Compact device layout struct to represent minor_to_major, tiles, and element
// size without coupling core headers to xla/layout.h.
struct CustomLayout {
  Indices minor_to_major;
  absl::InlinedVector<Indices, 1> tiles;
  int64_t element_size_in_bits = 0;

  friend bool operator==(const CustomLayout& lhs, const CustomLayout& rhs) {
    return lhs.minor_to_major == rhs.minor_to_major && lhs.tiles == rhs.tiles &&
           lhs.element_size_in_bits == rhs.element_size_in_bits;
  }
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_DIMENSION_TYPES_H_
