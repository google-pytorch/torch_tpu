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

#include "torch_tpu/ops/view_decomposition/contiguous_to_view.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/view_decomposition/decomposition.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"

namespace torch_tpu {

namespace {

// Helper struct to sort dimensions by stride while preserving the original
// index and size of each dimension.
struct IndexedStridedDim {
  int64_t index;
  int64_t size;
  int64_t stride;
};

// Identifies which dimensions (axes) of a tensor are overlapping, given its
// dimensions and the target strides.
//
// Returns a list of unique indices (0 <= index < dimensions.size()), ordered
// with the dimensions having the smallest strides first.
//
// If dimensions/target_strides imply a non-overlapping view (or a simple
// broadcast), then this will return an empty list.
//
// Precondition: dimensions.size() == target_strides.size() (unchecked).
Indices GetOverlappingAxes(absl::Span<const int64_t> dimensions,
                           absl::Span<const int64_t> target_strides) {
  // Sort the dimensions in descending order of stride (row-major), preserving
  // the original index of each dimension.
  std::vector<IndexedStridedDim> indexed_strided_dims;
  indexed_strided_dims.reserve(dimensions.size());
  for (int64_t index = 0; index < dimensions.size(); ++index) {
    indexed_strided_dims.push_back({.index = index,
                                    .size = dimensions[index],
                                    .stride = target_strides[index]});
  }
  std::sort(indexed_strided_dims.begin(), indexed_strided_dims.end(),
            [](const IndexedStridedDim& lhs, const IndexedStridedDim& rhs) {
              return lhs.stride > rhs.stride;
            });

  // Return value.
  Indices overlapping_indices;

  // If the maximum index to the right of a dimension is greater than or equal
  // to the stride of that dimension, then the dimension is overlapping.
  int64_t max_right_index = 0;

  // Iterate from the back of the sorted list (smallest stride) to the front
  // (largest stride).
  for (auto it = indexed_strided_dims.rbegin();
       it != indexed_strided_dims.rend(); ++it) {
    const auto& indexed_dim = *it;
    // Trivial dimensions are not overlapping, and do not increase the max right
    // index, because (1 - 1) * stride == (size - 1) * 0 == 0.
    if (indexed_dim.size == 1 || indexed_dim.stride == 0) continue;

    if (indexed_dim.stride <= max_right_index) {
      // This dimension is overlapping. It will be sliced to one element at
      // a time, so we don't count it towards the right index.
      overlapping_indices.push_back(indexed_dim.index);
    } else {
      // This dimension is not overlapping. It will be sliced to its full
      // size on each iteration.
      max_right_index += (indexed_dim.size - 1) * indexed_dim.stride;
    }
  }
  if (!overlapping_indices.empty()) {
    ABSL_VLOG(1) << "[ContiguousToView] Found overlapping axes "
                 << ToString(overlapping_indices) << " for dimensions "
                 << ToString(dimensions) << " and target strides "
                 << ToString(target_strides);
  }
  return overlapping_indices;
}

// Returns the shape of the largest non-overlapping window that can be read from
// a contiguous tensor of shape `dimensions`, and written to a tensor with the
// given `target_strides`, factoring in the already-identified
// `overlapping_axes`.
//
// This requires slicing all broadcasted and overlapping dimensions to 1 to
// ensure that no two elements in a window refer to the same element in a view
// tensor with target strides.
Dimensions GetNonOverlappingWindowShape(
    absl::Span<const int64_t> dimensions,
    absl::Span<const int64_t> target_strides, const Indices& overlapping_axes) {
  // Initialize assuming we slice to the entire base shape.
  Dimensions non_overlapping_window_shape = CopyIntVector(dimensions);
  // Slice away any broadcasted dimensions.
  for (int64_t i = 0; i < dimensions.size(); ++i) {
    if (target_strides[i] == 0) {
      non_overlapping_window_shape[i] = 1;
    }
  }
  // Slice away any overlapping dimensions.
  for (const int64_t axis : overlapping_axes) {
    non_overlapping_window_shape[axis] = 1;
  }
  ABSL_VLOG(2) << "[ContiguousToView] Non-overlapping window shape: "
               << ToString(non_overlapping_window_shape) << " for dimensions "
               << ToString(dimensions) << " and target strides "
               << ToString(target_strides);
  return non_overlapping_window_shape;
}

// One window of data that needs to be written to the output tensor.
struct NonOverlappingWindow {
  // The logical values that will be written into the output tensor.
  DeviceBufferRef window_value;
  // The storage offset in the output tensor to write this window to.
  // This also requires knowing the `target_strides` used to compute this
  // window.
  int64_t write_storage_offset;
};

// Returns the full set of non-overlapping windows that can be read from the
// `contiguous_buffer`, and written to a tensor with the
// given `target_strides`, so that the final tensor will have the same logical
// values as the `contiguous_buffer` but with the given `target_strides` and
// `target_storage_offset`.
//
// WARNING: Each window is internally non-overlapping, allowing for a
// non-ambiguous write, but 2 different windows may write to the same output
// tensor element. It is expected that all locations in `contiguous_buffer`
// which are overlapping have the same value; if they do not, the result will
// have a "last write wins" behavior.
//
// Precondition: contiguous_buffer.dimensions().size() == target_strides.size().
absl::StatusOr<std::vector<NonOverlappingWindow>> GetNonOverlappingWindows(
    DeviceBufferRef contiguous_buffer, absl::Span<const int64_t> target_strides,
    int64_t target_storage_offset) {
  // Set up the properties which are constant across all windows.
  // Find which specific dimensions are overlapping.
  const Indices overlapping_axes =
      GetOverlappingAxes(contiguous_buffer.dimensions(), target_strides);
  const Dimensions non_overlapping_window_shape = GetNonOverlappingWindowShape(
      contiguous_buffer.dimensions(), target_strides, overlapping_axes);
  const Strides read_strides =
      GetStrides(MakeContiguousBaseLayout(contiguous_buffer.dimensions()));
  const at::Tensor duplicated_tensor = MakeTensor(std::move(contiguous_buffer));

  // Output value.
  std::vector<NonOverlappingWindow> non_overlapping_windows;

  // Iterate over all unique combinations of overlapping element indices.
  // This is similar to an i,j,k, ... loop but where we have a variable number
  // of nested loops.
  // So our element tuples are like (0, 0, ...), (1, 0, ...) ... (n, 0, ...),
  // then (0, 1, ...) through (0, m, ...  ), then (1, 0, ... ) to (1, m, ...),
  // etc up until (n, m, ...) when we are done.
  Indices overlapping_elements;
  overlapping_elements.resize(overlapping_axes.size(), 0);
  while (true) {
    ABSL_VLOG(3) << "Indexing with " << ToString(overlapping_elements)
                 << " for overlapping axes " << ToString(overlapping_axes);
    // For each window, we read a slice out of a contiguous / flattened value
    // tensor, and then we write it into a slice of a differently-strided
    // output tensor.
    int64_t read_storage_offset = 0;
    int64_t write_storage_offset = target_storage_offset;
    for (auto i = 0; i < overlapping_axes.size(); ++i) {
      const auto axis = overlapping_axes[i];
      read_storage_offset += read_strides[axis] * overlapping_elements[i];
      write_storage_offset += target_strides[axis] * overlapping_elements[i];
    }
    // Create a new at::TensorImpl that points to the same c10::Storage,
    // then update the layout metadata to represent the window.
    at::Tensor window_tensor = duplicated_tensor.alias();
    window_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
        non_overlapping_window_shape, read_strides, read_storage_offset);
    ABSL_VLOG(3) << "Reading from window of shape "
                 << ToString(window_tensor.sizes()) << ", strides "
                 << ToString(window_tensor.strides()) << ", at offset "
                 << read_storage_offset;
    TT_ASSIGN_OR_RETURN(DeviceBufferRef window_value, GetBuffer(window_tensor));
    non_overlapping_windows.push_back(
        NonOverlappingWindow{.window_value = std::move(window_value),
                             .write_storage_offset = write_storage_offset});

    // Advance to the next set of overlapping elements.
    bool done = true;
    for (auto i = 0; i < overlapping_axes.size(); ++i) {
      const int64_t axis = overlapping_axes[i];
      if (++overlapping_elements[i] < duplicated_tensor.size(axis)) {
        done = false;
        break;
      }
      overlapping_elements[i] = 0;
    }
    if (done) break;
  }
  ABSL_VLOG(1) << "Found " << non_overlapping_windows.size()
               << " non-overlapping windows";
  return non_overlapping_windows;
}

// Repeatedly writes the values of each non-overlapping window to the output
// tensor.
//
// The output tensor will be modified in place.
absl::Status WriteWindowsToTensor(
    const at::Tensor& output_tensor, absl::Span<const int64_t> target_strides,
    std::vector<NonOverlappingWindow> non_overlapping_windows) {
  for (auto& window : non_overlapping_windows) {
    // Take a view of the correct destination slice in the output tensor.
    at::Tensor writable_window = output_tensor;  // intentional copy
    writable_window.unsafeGetTensorImpl()->set_sizes_and_strides(
        window.window_value.dimensions(), target_strides,
        window.write_storage_offset);

    // Copy the window values into the output tensor slice.
    ABSL_VLOG(3) << "Writing into window of shape "
                 << ToString(window.window_value.dimensions()) << ", strides "
                 << ToString(target_strides) << ", at offset "
                 << window.write_storage_offset;
    TT_RETURN_IF_ERROR(AssignBufferToAtTensor(std::move(window.window_value),
                                              writable_window));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<at::Tensor> ContiguousToView(
    DeviceBufferRef contiguous_buffer, absl::Span<const int64_t> target_strides,
    int64_t target_storage_offset) {
  // Check argument validity.
  TT_RET_CHECK(contiguous_buffer.dimensions().size() == target_strides.size(),
               error::kInvalidArgument)
      << "must specify the same number of strides as dimensions in the tensor";
  TT_RET_CHECK(target_storage_offset >= 0, error::kInvalidArgument)
      << "target_storage_offset must be non-negative";

  if (target_storage_offset == 0 &&
      GetStrides(MakeContiguousBaseLayout(contiguous_buffer.dimensions())) ==
          target_strides) {
    ABSL_VLOG(2) << "[ContiguousToView] Buffer is already contiguous, no op";
    // The target view is a contiguous base. Lift directly into output tensor.
    return MakeTensor(std::move(contiguous_buffer));
  }

  // Create an empty contiguous base tensor to serve as the new storage.
  StridedLayout target_view_layout{
      .storage_offset = target_storage_offset,
  };
  for (auto i = 0; i < contiguous_buffer.dimensions().size(); ++i) {
    target_view_layout.strided_dims.push_back(
        {.size = contiguous_buffer.dimensions()[i],
         .stride = target_strides[i]});
  }
  TT_ASSIGN_OR_RETURN(Dimensions target_contiguous_base_shape,
                      GetContiguousBaseShape(target_view_layout));
  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef empty_contiguous_base,
      DeviceBufferList::CreateEmpty(target_contiguous_base_shape,
                                    contiguous_buffer.element_type()));
  at::Tensor output_tensor = MakeTensor(std::move(empty_contiguous_base));

  // Get the non-overlapping windows to write to the output tensor.
  // If the output tensor is non-overlapping, or only has simple broadcasts,
  // there will only be one window.
  TT_ASSIGN_OR_RETURN(
      std::vector<NonOverlappingWindow> non_overlapping_windows,
      GetNonOverlappingWindows(contiguous_buffer, target_strides,
                               target_storage_offset));

  // Write the non-overlapping windows to the output tensor.
  TT_RETURN_IF_ERROR(WriteWindowsToTensor(output_tensor, target_strides,
                                          std::move(non_overlapping_windows)));

  // Apply the target view to the output tensor.
  output_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
      contiguous_buffer.dimensions(), target_strides, target_storage_offset);
  ABSL_VLOG(1) << "[ContiguousToView] Successfully created view of shape "
               << ToString(output_tensor.sizes()) << ", strides "
               << ToString(output_tensor.strides()) << ", at offset "
               << output_tensor.storage_offset();
  return output_tensor;
}

}  // namespace torch_tpu
