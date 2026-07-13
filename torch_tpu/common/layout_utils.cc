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

#include "torch_tpu/common/layout_utils.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"

namespace torch_tpu {

absl::StatusOr<TpuLayout> ResolveTpuLayout(const at::Tensor& tensor) {
  const auto scalar_type = tensor.scalar_type();
  TT_ASSIGN_OR_RETURN(const auto dtype,
                      ConvertTo<mlir::ElementType>(scalar_type));

  Dimensions sizes(tensor.sizes().begin(), tensor.sizes().end());
  Strides strides(tensor.strides().begin(), tensor.strides().end());
  int64_t offset = tensor.storage_offset();

  if (scalar_type == at::kFloat4_e2m1fn_x2) {
    // FP4 values are packed 2-per-byte in PyTorch storage (represented by the
    // float4_e2m1fn_x2 dtype). For StableHLO, we need to expose the layout
    // in terms of individual unpacked FP4 elements.
    //
    // 1. Double the size of the last (packed) dimension because each packed
    //    element contains 2 FP4 values.
    // 2. Double the strides of all outer dimensions to maintain correct
    //    byte offsets now that the inner dimension size has doubled.
    // 3. Double the storage offset to convert it from packed element offset
    //    to unpacked element offset.
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch core rejects 0D FP4 tensors
                   // generally on all devices.
        !sizes.empty(), error::kInvalidArgument)
        << "expected float4_e2m1fn_x2 tensors to be at least 1-dimensional, "
           "got 0-dimensional";
    sizes.back() *= 2;
    for (size_t i = 0; i + 1 < strides.size(); ++i) {
      strides[i] *= 2;
    }
    TT_RET_CHECK(
        strides.back() == 1,
        error::kInvalidArgument)  // ERROR_COV_INFEASIBLE=Utility not yet
                                  // reachable from Python.
        << "expected packed dimension to be "
           "contiguous (stride 1) for "
           "float4_e2m1fn_x2 view, got stride "
        << strides.back();
    offset *= 2;
  }
  return TpuLayout{std::move(sizes), std::move(strides), offset, dtype};
}

}  // namespace torch_tpu
