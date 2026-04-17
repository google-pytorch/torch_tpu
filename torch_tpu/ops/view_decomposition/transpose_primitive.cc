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

#include "torch_tpu/ops/view_decomposition/transpose_primitive.h"

#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

absl::Status ValidatePermutation(const TransposePrimitive& transpose,
                                 size_t expected_num_axes) {
  TT_RET_CHECK(transpose.permutation.size() == expected_num_axes,
               error::kInvalidArgument)
      << "transpose has wrong number of axes. Expected: " << expected_num_axes
      << " Actual: " << transpose.permutation.size();
  std::vector<bool> dim_used(expected_num_axes, false);
  for (int i = 0; i < transpose.permutation.size(); ++i) {
    bool index_in_bounds = 0 <= transpose.permutation[i] &&
                           transpose.permutation[i] < expected_num_axes;
    TT_RET_CHECK(index_in_bounds, error::kInvalidArgument)
        << "permutation dimension index is out of bounds. There are only "
        << expected_num_axes << " axes but permutation index is "
        << transpose.permutation[i];
    TT_RET_CHECK(!dim_used[transpose.permutation[i]], error::kInvalidArgument)
        << "transpose has duplicate axis dimension: " << transpose;
    dim_used[transpose.permutation[i]] = true;
  }
  return absl::OkStatus();
}

}  // namespace

std::ostream& operator<<(std::ostream& os,
                         const TransposePrimitive& transpose) {
  os << "transpose" << ToString(transpose.permutation);
  return os;
}

absl::StatusOr<bool> UpdateLayoutImpl(StridedLayout& layout,
                                      const TransposePrimitive& transpose) {
  TT_RETURN_IF_ERROR(
      ValidatePermutation(transpose, layout.strided_dims.size()));

  bool updated = false;

  // Storage offset is unchanged.
  StridedLayout new_layout{.storage_offset = layout.storage_offset};
  new_layout.strided_dims.reserve(transpose.permutation.size());

  for (int i = 0; i < transpose.permutation.size(); ++i) {
    if (transpose.permutation[i] != i) {
      updated = true;
    }
    new_layout.strided_dims.push_back(StridedDimension{
        .size = layout.strided_dims[transpose.permutation[i]].size,
        .stride = layout.strided_dims[transpose.permutation[i]].stride});
  }
  if (updated) {
    layout = std::move(new_layout);
  }
  return updated;
}

absl::StatusOr<mlir::MlirOp> ViewPrimitiveShlo(
    mlir::MlirOp input, const TransposePrimitive& transpose) {
  return mlir::stablehlo::Transpose(input, transpose.permutation);
}

}  // namespace torch_tpu
