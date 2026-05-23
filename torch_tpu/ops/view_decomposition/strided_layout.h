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

#ifndef TORCH_TPU_OPS_VIEW_DECOMPOSITION_STRIDED_LAYOUT_H_
#define TORCH_TPU_OPS_VIEW_DECOMPOSITION_STRIDED_LAYOUT_H_

#include <stdint.h>

#include <ostream>

#include "ATen/core/TensorBody.h"
#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "torch_tpu/common/dimension_types.h"

// A DeviceBufferRef, or a PjRtBuffer, contains information about the shape
// and element type of a tensor.
//
// However, to correctly interpret the logical behavior of an as_strided
//  operation, we also need two additional pieces of information:
//   * The storage offset of the tensor in the 1D storage array.
//   * The stride of each dimension.
//
// A StridedLayout is a struct that contains this additional information, for
// use in the view decomposition algorithm.

namespace torch_tpu {

// One axis of a tensor, including both the dimension size and the stride.
// Special cases of note:
//   * stride == 0: the dimension is a broadcasted dimension; because
//                  index * 0 == 0 for any index, the view is overlapping unless
//                  the size is 1.
//   * size == 1:   because the only valid index is 0, the stride is arbitrary,
//                  as the only element index will be 0 * stride == 0.
struct StridedDimension {
  // The number of valid indices in this dimension; i in [0, size) is valid.
  int64_t size = 1;  // avoid uninitialized values
  // The number of linear elements to skip for each increment in this dimension.
  int64_t stride = 0;  // avoid uninitialized values
};
static_assert(sizeof(StridedDimension) == 16, "");

bool operator==(const StridedDimension& lhs, const StridedDimension& rhs);

// A struct representing a strided layout of a tensor, with a size and stride
// for 0 or more dimensions.
// The logical behavior of this on CPU/CUDA would be that for some index vector
// x := [x0, x1, ..., xN-1], the element index into a 1D storage array would be:
//   storage_offset + x[0]*stride[0] + x[1]*stride[1] + ... + x[N-1]*stride[N-1]
// provided that all 0 <= x[i] < size[i] for all i in [0, N)
struct StridedLayout {
  // Constructs a StridedLayout from a PyTorch tensor.
  static StridedLayout FromTensor(const at::Tensor& tensor);

  // The dimensions of the tensor in strided layout.
  absl::InlinedVector<StridedDimension, 3> strided_dims;
  // The offset of the first element in the tensor in the 1D storage array.
  int64_t storage_offset;
};
// Using a smaller inlined vector than normal to keep the size of the
// StridedLayout at 64 bytes (one cache line on most x86 CPUs).
static_assert(sizeof(StridedLayout) == 64, "wrong size");

bool operator==(const StridedLayout& lhs, const StridedLayout& rhs);

inline bool operator!=(const StridedLayout& lhs, const StridedLayout& rhs) {
  return !(lhs == rhs);
}

std::ostream& operator<<(std::ostream& os, const StridedLayout& layout);

// Given a shape, returns the unique contiguous base layout with that shape.
// The contiguous base layout is row-major, non-overlapping, dense, with
// storage offset 0.
StridedLayout MakeContiguousBaseLayout(absl::Span<const int64_t> shape);

// Given a StridedLayout, returns a vector with all dimension sizes gathered.
Dimensions GetSizes(const StridedLayout& layout);

// Given a StridedLayout, returns a vector with all dimension strides gathered.
Strides GetStrides(const StridedLayout& layout);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_VIEW_DECOMPOSITION_STRIDED_LAYOUT_H_
