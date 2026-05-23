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

#include "torch_tpu/ops/view_decomposition/slice_primitive.h"

#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "torch_tpu/ops/view_decomposition/view_primitive_error_utils.h"

namespace torch_tpu {

namespace {

int64_t ComputeNewSize(const SliceDimension& slice_dim) {
  // ceil((limit_index - start_index) / stride)
  return (slice_dim.limit_index - slice_dim.start_index + slice_dim.stride -
          1) /
         slice_dim.stride;
}

template <typename GetErrorMessageSuffix>
void CheckSlice(const SlicePrimitive& slice, absl::Span<const int64_t> sizes,
                const GetErrorMessageSuffix& get_error_message_suffix) {
  ABSL_CHECK_EQ(  // CRASH_OK=Internal error
                  // on view decomposition.
      slice.slice_dims.size(), sizes.size())
      << "expected the SlicePrimitive input rank to be "
      << slice.slice_dims.size()
      << ", which is the number of SliceDimensions, got " << sizes.size()
      << get_error_message_suffix();

  for (int i = 0; i < slice.slice_dims.size(); ++i) {
    const auto& slice_dim = slice.slice_dims[i];
    const int64_t input_dim = sizes[i];

    ABSL_CHECK_GE(slice_dim.start_index,  // CRASH_OK=Internal error on view
                                          // decomposition.
                  0)
        << "expected the SlicePrimitive start index at dimension " << i
        << " to be >= 0, got " << slice_dim.start_index
        << get_error_message_suffix();
    ABSL_CHECK_LT(  // CRASH_OK=Internal error on view
                    // decomposition.
        slice_dim.start_index, input_dim)
        << "expected the SlicePrimitive start index at dimension " << i
        << " to be < " << input_dim
        << ", which is the size of the input dimension " << i << ", got "
        << slice_dim.start_index << get_error_message_suffix();
    ABSL_CHECK_LT(  // CRASH_OK=Internal error on view
                    // decomposition.
        slice_dim.start_index, slice_dim.limit_index)
        << "expected the SlicePrimitive start index at dimension " << i
        << " to be < " << slice_dim.limit_index
        << ", which is the limit index at dimension " << i << ", got "
        << slice_dim.start_index << get_error_message_suffix();
    ABSL_CHECK_GT(slice_dim.stride,  // CRASH_OK=Internal error on view
                                     // decomposition.
                  0)
        << "expected the SlicePrimitive stride at dimension " << i
        << " to be > 0, got " << slice_dim.stride << get_error_message_suffix();

    const int64_t new_size = ComputeNewSize(slice_dim);

    ABSL_CHECK_LE(  // CRASH_OK=Guaranteed by checks above.
        new_size, input_dim)
        << "expected the SlicePrimitive slice size to be <= " << input_dim
        << ", which is the size of the input at dimension " << i << ", got "
        << new_size << get_error_message_suffix();
  }
}

void CheckSliceMerge(const SlicePrimitive& first,
                     const SlicePrimitive& second) {
  Dimensions dimensions(first.slice_dims.size());
  absl::c_transform(first.slice_dims, dimensions.begin(), ComputeNewSize);

  CheckSlice(second, dimensions, /* get_error_message_suffix= */ [&]() {
    return absl::StrCat(
        "; calling Merge() with first=", ToString(first),
        " and second=", ToString(second),
        GetViewPrimitiveErrorSuffix(second, {.leading_semicolon = false}));
  });
}

}  // namespace

bool operator==(const SliceDimension& lhs, const SliceDimension& rhs) {
  return lhs.start_index == rhs.start_index &&
         lhs.limit_index == rhs.limit_index && lhs.stride == rhs.stride;
}

std::ostream& operator<<(std::ostream& os, const SliceDimension& dim) {
  os << "(start=" << dim.start_index << ", limit=" << dim.limit_index
     << ", stride=" << dim.stride << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const SlicePrimitive& slice) {
  os << "slice" << ToString(slice.slice_dims);
  return os;
}

bool UpdateLayout(StridedLayout& layout, const SlicePrimitive& slice) {
  CheckSlice(slice, GetSizes(layout),
             /* get_error_message_suffix= */ [&]() {
               return GetUpdateLayoutBugSuffix(slice, layout);
             });
  bool updated = false;

  // Storage layout offset may be increased, but always starts at the input's
  // storage offset.
  StridedLayout new_layout{.storage_offset = layout.storage_offset};
  new_layout.strided_dims.reserve(slice.slice_dims.size());

  for (int i = 0; i < slice.slice_dims.size(); ++i) {
    const auto& slice_dim = slice.slice_dims[i];
    const auto& input_dim = layout.strided_dims[i];

    if (slice_dim.start_index != 0 || slice_dim.limit_index != input_dim.size ||
        slice_dim.stride != 1) {
      updated = true;
    };

    const int64_t new_size = ComputeNewSize(slice_dim);

    new_layout.storage_offset += slice_dim.start_index * input_dim.stride;
    new_layout.strided_dims.push_back(StridedDimension{
        .size = new_size, .stride = input_dim.stride * slice_dim.stride});
  }
  if (updated) {
    layout = std::move(new_layout);
  }
  return updated;
}

SlicePrimitive Merge(SlicePrimitive first, const SlicePrimitive second) {
  CheckSliceMerge(first, second);

  for (int i = 0; i < second.slice_dims.size(); ++i) {
    // The second slice is relative to the first slice; we need to account for
    // the stride on the first slice to determine the combined slice relative
    // to the unsliced base.
    // Formula:
    //   dimension[a:b:c][d:e:f] -> dimension[a+d*c:a+e*c:c*f]
    // Example:
    //   arange(10) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    //   arange(10)[1:9:2] == [1, 3, 5, 7]
    //   arange(10)[1:9:2][1:4:2] == [3, 7]
    //   arange(10)[3:9:4] == [3, 7] as desired
    const int64_t new_start_index =
        first.slice_dims[i].start_index +
        second.slice_dims[i].start_index * first.slice_dims[i].stride;
    const int64_t new_limit_index =
        first.slice_dims[i].start_index +
        second.slice_dims[i].limit_index * first.slice_dims[i].stride;
    const int64_t new_stride =
        first.slice_dims[i].stride * second.slice_dims[i].stride;

    // Overwrite `first` in-place to avoid extra allocation
    first.slice_dims[i] = SliceDimension{.start_index = new_start_index,
                                         .limit_index = new_limit_index,
                                         .stride = new_stride};
  }
  return first;
}

absl::StatusOr<mlir::MlirOp> ViewPrimitiveShlo(mlir::MlirOp input,
                                               const SlicePrimitive& slice) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  CheckSlice(slice, input_type.getShape(),
             /* get_error_message_suffix= */ [&]() {
               return GetViewPrimitiveShloErrorSuffix(slice,
                                                      input_type.getShape());
             });

  // Restructure from list-of-tuples to tuple-of-lists to match StableHLO API.
  Indices start_indices;
  start_indices.reserve(slice.slice_dims.size());
  Indices limit_indices;
  limit_indices.reserve(slice.slice_dims.size());
  Indices strides;
  strides.reserve(slice.slice_dims.size());
  for (const auto& slice_dim : slice.slice_dims) {
    start_indices.push_back(slice_dim.start_index);
    limit_indices.push_back(slice_dim.limit_index);
    strides.push_back(slice_dim.stride);
  }
  return mlir::stablehlo::Slice(input, start_indices, limit_indices, strides);
}

}  // namespace torch_tpu
