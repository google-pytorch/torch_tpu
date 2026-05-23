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

#include "torch_tpu/ops/stride/stride_helper.h"

#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"

namespace torch_tpu {

Strides CalculateStridesContiguous(absl::Span<const int64_t> shape) {
  if (shape.empty()) {
    return Strides();
  }

  Strides strides(shape.size());
  strides.back() = 1;  // The stride for the last dimension is always 1

  // Calculate strides by iterating backwards from the second to last dimension
  for (int i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  return strides;
}

namespace {

// Size and stride pair for one dimension of a tensor. Used by IsOverlapping
// to sort axes to row-major order.
struct SizeAndStride {
  int64_t size;
  int64_t stride;
};

}  // namespace

bool IsOverlapping(absl::Span<const int64_t> sizes,
                   absl::Span<const int64_t> strides, bool allow_expanded) {
  ABSL_CHECK_EQ(sizes.size(), strides.size());  // CRASH_OK

  // Scalars are non-overlapping.
  if (sizes.empty()) return false;

  std::vector<SizeAndStride> size_and_strides;
  size_and_strides.reserve(sizes.size());  // Maximum size if no size-1 dims.

  bool is_expanded = false;
  for (int64_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] == 0) {
      // Zero-sized views are always non-overlapping; it's impossible to have
      // multiple elements pointing to the same offset, if you have no elements.
      return false;
    }

    // Skip size=1 dimensions--they never overlap.
    if (sizes[i] == 1) continue;

    // Size > 1, stride=0 dimensions are overlapping, but can be supported in
    // some cases where complex overlapping is not supported (e.g. as_strided).
    // In that case, they behave like size=1 dimensions (broadcasting).
    if (strides[i] == 0) {
      is_expanded = true;
      continue;
    }

    size_and_strides.push_back({.size = sizes[i], .stride = strides[i]});
  }
  if (is_expanded && !allow_expanded) return true;

  // Scalars and 1D tensors (after removing size-1 dims) are non-overlapping.
  if (size_and_strides.size() < 2) return false;

  // 2+D tensors may or may not be overlapping.
  // Sort into a row-major order (descending order of strides)
  absl::c_sort(size_and_strides,
               [](const SizeAndStride& a, const SizeAndStride& b) {
                 return a.stride > b.stride;
               });

  // If any dimension's maximum inner offset exceeds or equals the stride of the
  // next-most-major dimension, then the tensor is overlapping.
  int64_t max_inner_offset = 0;
  for (int64_t i = size_and_strides.size() - 1; i >= 0; --i) {
    if (i < size_and_strides.size() - 1) {
      if (size_and_strides[i].stride <= max_inner_offset) return true;
    }
    max_inner_offset +=
        (size_and_strides[i].size - 1) * size_and_strides[i].stride;
  }
  return false;
}

absl::Status CheckProvidedLayoutDataFitsInStorage(
    int64_t storage_numel, mlir::ElementType storage_element_type,
    Dimensions sizes, Strides strides, int64_t storage_offset,
    mlir::ElementType layout_element_type) {
  TT_RET_CHECK(storage_offset >= 0, error::kInvalidArgument)
      << "expected the given storage offset to be >= 0, got " << storage_offset;

  TT_RET_CHECK(sizes.size() == strides.size(), error::kInvalidArgument)
      << "expected the given sizes " << ToString(sizes) << " and strides "
      << ToString(strides) << " to have the same length, got " << sizes.size()
      << " vs. " << strides.size();

  const int64_t storage_element_bitwidth =
      TorchEquivalentBitwidth(storage_element_type);
  int64_t storage_nbytes = -1;
  if (storage_element_bitwidth < 8) {
    // Round up to the nearest byte.
    storage_nbytes = (storage_numel * storage_element_bitwidth + 7) / 8;
  } else {
    storage_nbytes = storage_numel * (storage_element_bitwidth / 8);
  }

  int64_t required_numel = storage_offset + 1;
  for (auto i = 0; i < sizes.size(); ++i) {
    if (sizes[i] == 0) {
      // Zero-sized views are always valid.
      required_numel = 0;
      break;
    }
    required_numel += (sizes[i] - 1) * strides[i];
  }
  const int64_t layout_element_bitwidth =
      TorchEquivalentBitwidth(layout_element_type);
  int64_t required_nbytes = -1;
  if (layout_element_bitwidth < 8) {
    // Round up to the nearest byte.
    required_nbytes = (required_numel * layout_element_bitwidth + 7) / 8;
  } else {
    required_nbytes = required_numel * (layout_element_bitwidth / 8);
  }

  TT_RET_CHECK(required_nbytes <= storage_nbytes, error::kInvalidArgument)
      << "expected the number of bytes required by the given arguments to be "
         "<= "
      << storage_nbytes << " (actual storage size), got " << required_nbytes;

  return absl::OkStatus();
}

}  // namespace torch_tpu
