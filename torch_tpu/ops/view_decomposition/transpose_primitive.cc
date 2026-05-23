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
#include <cstdint>
#include <ostream>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "torch_tpu/ops/view_decomposition/view_primitive_error_utils.h"

namespace torch_tpu {

namespace {

void CheckTranspose(const TransposePrimitive& transpose,
                    const StridedLayout& layout) {
  const size_t rank = layout.strided_dims.size();

  ABSL_CHECK_EQ(  // CRASH_OK=Internal error on view decomposition.
      transpose.permutation.size(), rank)
      << "expected the number of elements in the TransposePrimitive "
         "permutation to be of size "
      << rank << " (rank of the layout input), got "
      << transpose.permutation.size()
      << GetUpdateLayoutBugSuffix(transpose, layout);

  Indices last_use_of(rank, -1);
  for (int i = 0; i < rank; ++i) {
    const int64_t permuted_index = transpose.permutation[i];

    ABSL_CHECK(  // CRASH_OK=Internal error on view decomposition.
        0 <= permuted_index && permuted_index < rank)
        << "expected each element in the TransposePrimitive permutation to be "
           "within the range [0, "
        << rank << "), got " << permuted_index
        << GetUpdateLayoutBugSuffix(transpose, layout);

    ABSL_CHECK_EQ(  // CRASH_OK=Internal error on view decomposition.
        last_use_of[permuted_index], -1)
        << "expected the TransposePrimitive permutation elements to be unique, "
           "got element "
        << permuted_index << " at indices " << i << " and "
        << last_use_of[permuted_index]
        << GetUpdateLayoutBugSuffix(transpose, layout);

    last_use_of[permuted_index] = i;
  }
}

}  // namespace

std::ostream& operator<<(std::ostream& os,
                         const TransposePrimitive& transpose) {
  os << "transpose" << ToString(transpose.permutation);
  return os;
}

bool UpdateLayout(StridedLayout& layout, const TransposePrimitive& transpose) {
  CheckTranspose(transpose, layout);

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
