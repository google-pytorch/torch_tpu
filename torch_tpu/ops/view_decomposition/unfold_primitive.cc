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
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

std::ostream& operator<<(std::ostream& os, const UnfoldPrimitive& unfold) {
  os << "unfold(start_index=" << unfold.start_index
     << ", limit_index=" << unfold.limit_index
     << ", window_stride=" << unfold.window_stride
     << ", window_size=" << unfold.window_size << ")";
  return os;
}

absl::StatusOr<bool> UpdateLayoutImpl(StridedLayout& layout,
                                      const UnfoldPrimitive& unfold) {
  TT_RET_CHECK(layout.strided_dims.size() > 1, error::kInvalidArgument)
      << "expected input to have at least 2 dimensions, but got "
      << layout.strided_dims.size();
  const int64_t rank = layout.strided_dims.size();
  TT_RET_CHECK(layout.strided_dims[rank - 2].size == 1, error::kInvalidArgument)
      << "expected input to have size 1 in the second-to-last dimension, but "
         "got "
      << layout.strided_dims[rank - 2].size;
  TT_RET_CHECK(0 <= unfold.start_index &&
                   unfold.start_index < layout.strided_dims.back().size,
               error::kInvalidArgument)
      << "start index " << unfold.start_index
      << " is out of bounds for dimension of size "
      << layout.strided_dims.back().size;
  TT_RET_CHECK(unfold.start_index < unfold.limit_index, error::kInvalidArgument)
      << "limit index cannot be less than start index, got limit index "
      << unfold.limit_index << " and start index " << unfold.start_index;
  TT_RET_CHECK(unfold.limit_index <= layout.strided_dims.back().size,
               error::kInvalidArgument)
      << "limit index " << unfold.limit_index
      << " is out of bounds for dimension of size "
      << layout.strided_dims.back().size;
  const int64_t max_window_size = unfold.limit_index - unfold.start_index;
  TT_RET_CHECK(unfold.window_size <= max_window_size, error::kInvalidArgument)
      << "window size " << unfold.window_size << " is larger than the range ["
      << unfold.start_index << ", " << unfold.limit_index << ")";
  TT_RET_CHECK(unfold.window_stride > 0, error::kInvalidArgument)
      << "window stride must be positive, got " << unfold.window_stride;
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
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const int64_t rank = input_type.getRank();
  TT_RET_CHECK(rank > 1, error::kInvalidArgument)
      << "expected input to have at least 2 dimensions, but got " << rank;
  TT_RET_CHECK(input_type.getDimSize(rank - 2) == 1, error::kInvalidArgument)
      << "expected input to have trailing dimensions of size (..., 1, M), but "
         "got (..., "
      << input_type.getDimSize(rank - 2) << ", "
      << input_type.getDimSize(rank - 1) << ")";
  std::vector<mlir::MlirOp> slices;
  Indices start_indices;
  start_indices.resize(rank, 0);
  start_indices[rank - 1] = unfold.start_index;
  Indices limit_indices(input_type.getShape().begin(),
                        input_type.getShape().end());
  limit_indices[rank - 1] = start_indices[rank - 1] + unfold.window_size;
  Strides slice_strides(rank, 1);
  while (limit_indices[rank - 1] <= unfold.limit_index) {
    slices.push_back(mlir::stablehlo::Slice(input, start_indices, limit_indices,
                                            slice_strides));
    start_indices[rank - 1] += unfold.window_stride;
    limit_indices[rank - 1] += unfold.window_stride;
  }
  return mlir::stablehlo::Concatenate(input.getBuilder(), slices, rank - 2);
}

}  // namespace torch_tpu
