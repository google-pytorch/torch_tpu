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

#include "torch_tpu/ops/view_decomposition/strided_layout.h"

#include <cstddef>
#include <cstdint>
#include <ostream>

#include "ATen/core/TensorBody.h"
#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/to_string.h"

namespace torch_tpu {

bool operator==(const StridedDimension& lhs, const StridedDimension& rhs) {
  return lhs.size == rhs.size && lhs.stride == rhs.stride;
}

StridedLayout MakeContiguousBaseLayout(absl::Span<const int64_t> shape) {
  // Base == storage_offset of 0
  StridedLayout layout{.storage_offset = 0};

  // Initialize the layout with the given shape
  layout.strided_dims.reserve(shape.size());
  for (int64_t dim_size : shape) {
    layout.strided_dims.push_back(
        StridedDimension{.size = dim_size, .stride = 0});
  }

  // Contiguous == the stride of each dimension is the product of the sizes o
  // all dimensions to the right, with 1 for the last dimension.
  int64_t right_num_elements = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    layout.strided_dims[i].stride = right_num_elements;
    right_num_elements *= shape[i];
  }

  return layout;
}

StridedLayout StridedLayout::FromTensor(const at::Tensor& tensor) {
  auto sizes = tensor.sizes();
  auto strides = tensor.strides();
  StridedLayout layout{.storage_offset = tensor.storage_offset()};
  layout.strided_dims.reserve(tensor.dim());
  for (int i = 0; i < tensor.dim(); ++i) {
    layout.strided_dims.push_back(
        StridedDimension{.size = sizes[i], .stride = strides[i]});
  }
  return layout;
}

std::ostream& operator<<(std::ostream& os, const StridedLayout& layout) {
  Dimensions shape;
  shape.reserve(layout.strided_dims.size());
  Strides strides;
  strides.reserve(layout.strided_dims.size());
  for (const auto& dim : layout.strided_dims) {
    shape.push_back(dim.size);
    strides.push_back(dim.stride);
  }
  os << "shape: " << ToString(shape) << " strides: " << ToString(strides)
     << " storage_offset: " << layout.storage_offset;
  return os;
}

bool operator==(const StridedLayout& lhs, const StridedLayout& rhs) {
  if (lhs.strided_dims.size() != rhs.strided_dims.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.strided_dims.size(); ++i) {
    if (lhs.strided_dims[i].size != rhs.strided_dims[i].size) {
      return false;
    }
    if (lhs.strided_dims[i].size == 1) {
      // Strides for dimensions of size 1 are arbitrary and don't need to match
      // for two layouts to be equal.
      continue;
    }
    if (lhs.strided_dims[i].stride != rhs.strided_dims[i].stride) {
      return false;
    }
  }
  return lhs.storage_offset == rhs.storage_offset;
}

Dimensions GetSizes(const StridedLayout& layout) {
  Dimensions dimensions(layout.strided_dims.size());
  absl::c_transform(layout.strided_dims, dimensions.begin(),
                    [](const auto& dim) { return dim.size; });
  return dimensions;
}

Strides GetStrides(const StridedLayout& layout) {
  Strides strides(layout.strided_dims.size());
  absl::c_transform(layout.strided_dims, strides.begin(),
                    [](const auto& dim) { return dim.stride; });
  return strides;
}

}  // namespace torch_tpu
