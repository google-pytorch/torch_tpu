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

#include "csrc/common/layout_utils.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace torch_tpu {
namespace {

absl::Status AdjustFp4Layout(Dimensions& sizes, Strides& strides,
                             int64_t& storage_offset) {
  // FP4 values are packed 2-per-byte in PyTorch storage (represented by the
  // float4_e2m1fn_x2 dtype). For StableHLO, we need to expose the layout
  // in terms of individual unpacked FP4 elements.
  //
  // 1. Identify the packed dimension (contiguous, stride 1). Default to the
  //    last dimension. If the last dimension is not stride 1 (e.g.,
  //    transposed tensor), search for the dimension with stride 1 (preferring
  //    sizes[d] > 1).
  // 2. Double the size of the packed dimension because each packed element
  //    contains 2 FP4 values.
  // 3. Double the strides of all other dimensions to maintain correct byte
  //    offsets now that the packed dimension size has doubled.
  // 4. Double the storage offset to convert it from packed element offset
  //    to unpacked element offset.
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch core rejects 0D FP4 tensors
                 // generally on all devices.
      !sizes.empty(), error::kInvalidArgument)
      << "expected float4_e2m1fn_x2 tensors to be at least 1-dimensional, "
         "got 0-dimensional";
  size_t packed_dim = strides.size() - 1;
  if (strides.back() != 1) {
    std::optional<size_t> found_dim;
    for (size_t d = 0; d < strides.size(); ++d) {
      if (strides[d] == 1) {
        if (!found_dim.has_value() || sizes[d] > 1) {
          found_dim = d;
        }
      }
    }
    TT_RET_CHECK(found_dim.has_value(), error::kInvalidArgument)
        << "expected packed dimension to be contiguous (stride 1) for "
           "float4_e2m1fn_x2 view, got stride "
        << strides.back();
    packed_dim = *found_dim;
  }
  sizes[packed_dim] *= 2;
  for (size_t i = 0; i < strides.size(); ++i) {
    if (i != packed_dim) {
      strides[i] *= 2;
    }
  }
  storage_offset *= 2;
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<TpuLayout> ResolveTpuLayout(const at::Tensor& tensor) {
  const auto scalar_type = tensor.scalar_type();
  TT_ASSIGN_OR_RETURN(const auto dtype,
                      ConvertTo<mlir::ElementType>(scalar_type));

  Dimensions sizes(tensor.sizes().begin(), tensor.sizes().end());
  Strides strides(tensor.strides().begin(), tensor.strides().end());
  int64_t offset = tensor.storage_offset();

  if (scalar_type == at::kFloat4_e2m1fn_x2) {
    TT_RETURN_IF_ERROR(AdjustFp4Layout(sizes, strides, offset));
  }
  return TpuLayout{std::move(sizes), std::move(strides), offset, dtype};
}

}  // namespace torch_tpu
