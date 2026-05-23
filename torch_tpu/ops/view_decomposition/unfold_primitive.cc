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

#include "torch_tpu/ops/view_decomposition/unfold_primitive.h"

#include <cstdint>
#include <ostream>
#include <string_view>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "torch_tpu/ops/view_decomposition/view_primitive_error_utils.h"

namespace torch_tpu {
namespace {

void CheckUnfold(const UnfoldPrimitive& unfold, absl::Span<const int64_t> sizes,
                 const std::string_view error_message_suffix) {
  const int64_t rank = sizes.size();

  ABSL_CHECK_GT(  // CRASH_OK=Guaranteed by the previous reshape
                  // call in the caller.
      rank, 1)
      << "expected the UnfoldPrimitive input to have at least 2 dimensions, "
         "got "
      << rank << error_message_suffix;

  ABSL_CHECK_EQ(  // CRASH_OK=Guaranteed by the previous reshape
                  // call in the caller.
      sizes[rank - 2], 1)
      << "expected the UnfoldPrimitive input second-to-last dimension size, "
         "i.e. size of dimension "
      << rank - 2 << " to be 1, got " << sizes[rank - 2]
      << error_message_suffix;

  ABSL_CHECK(  // CRASH_OK=Internal error on view decomposition.
      0 <= unfold.start_index && unfold.start_index < sizes.back())
      << "expected the UnfoldPrimitive start index to be within range [0, "
      << sizes.back() << "), got " << unfold.start_index
      << error_message_suffix;

  ABSL_CHECK_LT(  // CRASH_OK=Internal error on view decomposition.
      unfold.start_index, unfold.limit_index)
      << "expected the UnfoldPrimitive start index to be < "
      << unfold.limit_index << " , which is its limit index, got "
      << unfold.start_index << error_message_suffix;

  ABSL_CHECK_LE(  // CRASH_OK=Internal error on view decomposition.
      unfold.limit_index, sizes.back())
      << "expected the UnfoldPrimitive limit index to be < " << sizes.back()
      << " , which is the size of the input's last dimension, got "
      << unfold.limit_index << error_message_suffix;

  const int64_t max_window_size = unfold.limit_index - unfold.start_index;
  ABSL_CHECK_LE(  // CRASH_OK=Internal error on view decomposition.
      unfold.window_size, max_window_size)
      << "expected the UnfoldPrimitive window size to be <= " << max_window_size
      << ", which is the number of elements from index " << unfold.start_index
      << " to " << unfold.limit_index << " (exclusive), got "
      << unfold.window_size << error_message_suffix;

  ABSL_CHECK_GT(  // CRASH_OK=Internal error on view decomposition.
      unfold.window_stride, 0)
      << "expected the UnfoldPrimitive window stride to be > 0, got "
      << unfold.window_stride << error_message_suffix;
}

void CheckUnfold(const StridedLayout& layout, const UnfoldPrimitive& unfold) {
  CheckUnfold(
      unfold, GetSizes(layout),
      /* error_message_suffix= */ GetUpdateLayoutBugSuffix(unfold, layout));
}

void CheckUnfold(mlir::MlirOp input, const UnfoldPrimitive& unfold) {
  const mlir::RankedTensorType type = GetTensorTypeOrDie(input);
  CheckUnfold(unfold, type.getShape(),
              /* error_message_suffix= */
              GetViewPrimitiveShloErrorSuffix(unfold, type.getShape()));
}

}  // namespace

std::ostream& operator<<(std::ostream& os, const UnfoldPrimitive& unfold) {
  os << "unfold(start_index=" << unfold.start_index
     << ", limit_index=" << unfold.limit_index
     << ", window_stride=" << unfold.window_stride
     << ", window_size=" << unfold.window_size << ")";
  return os;
}

bool UpdateLayout(StridedLayout& layout, const UnfoldPrimitive& unfold) {
  CheckUnfold(layout, unfold);

  // The last slice will be over the indexes
  // [start_index + (N-1) * window_stride,
  //  start_index + (N-1) * window_stride + window_size).
  // To keep the upper bound <= limit_index, we set
  // N = 1 + floor((limit_index - start_index - window_size) / window_stride).
  const int64_t num_windows =
      1 + (unfold.limit_index - unfold.start_index - unfold.window_size) /
              unfold.window_stride;
  const bool changed = num_windows != 1 ||
                       unfold.window_size != layout.strided_dims.back().size ||
                       unfold.start_index > 0;

  const int64_t rank = layout.strided_dims.size();

  layout.strided_dims[rank - 2].size = num_windows;
  // This will almost always be 1 during the decomposition process, except when
  // there's also a real() or imag() that creates a non-zero trailing stride.
  layout.strided_dims[rank - 2].stride =
      unfold.window_stride * layout.strided_dims[rank - 1].stride;
  layout.strided_dims[rank - 1].size = unfold.window_size;
  layout.storage_offset +=
      unfold.start_index * layout.strided_dims[rank - 1].stride;

  return changed;
}

absl::StatusOr<mlir::MlirOp> ViewPrimitiveShlo(mlir::MlirOp input,
                                               const UnfoldPrimitive& unfold) {
  CheckUnfold(input, unfold);

  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const int64_t rank = input_type.getRank();

  Indices start_indices;
  start_indices.resize(rank, 0);
  start_indices[rank - 1] = unfold.start_index;

  Indices limit_indices(input_type.getShape().begin(),
                        input_type.getShape().end());
  limit_indices[rank - 1] = start_indices[rank - 1] + unfold.window_size;

  Strides slice_strides(rank, 1);
  std::vector<mlir::MlirOp> slices;

  while (limit_indices[rank - 1] <= unfold.limit_index) {
    slices.push_back(mlir::stablehlo::Slice(input, start_indices, limit_indices,
                                            slice_strides));
    start_indices[rank - 1] += unfold.window_stride;
    limit_indices[rank - 1] += unfold.window_stride;
  }

  return mlir::stablehlo::Concatenate(input.getBuilder(), slices, rank - 2);
}

}  // namespace torch_tpu
