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

#include "torch_tpu/_internal/dynamism/dynamism.h"

#include <algorithm>
#include <cstdint>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ATen/core/TensorBody.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/tensor_to_buffer.h"

namespace torch_tpu {

absl::StatusOr<at::Tensor> MarkDynamic(const at::Tensor& tensor,
                                       int64_t dimension, int64_t lower_bound,
                                       int64_t upper_bound) {
  ABSL_VLOG(1) << "[MarkDynamic] Marking dynamic tensor with dim sizes "
               << tensor.sizes() << " on dimension " << dimension
               << " lower_bound " << lower_bound << " upper_bound "
               << upper_bound;
  TT_ASSIGN_OR_RETURN(DeviceBufferRef base_buffer_ref,
                      GetBaseBufferFromAtTensor(tensor));

  TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer_ref,
                      GetBufferFromAtTensor(tensor));

  auto dynamic_dimensions = buffer_ref.dynamic_dimensions();
  auto it_find = std::find_if(
      dynamic_dimensions.begin(), dynamic_dimensions.end(),
      [dimension](const BoundedDynamicDimension& dynamic_dimension) {
        return dynamic_dimension.dimension == dimension;
      });
  TT_RET_CHECK(
      it_find != dynamic_dimensions.end() || dynamic_dimensions.empty(),
      error::kInvalidArgument)
      << "only one dynamic dimension is supported per tensor";
  TT_RETURN_IF_ERROR(
      buffer_ref.MarkDynamic(dimension, lower_bound, upper_bound));

  if (base_buffer_ref == buffer_ref) {
    // The input tensor is not a view tensor, so we can return the input tensor.
    return tensor;
  }
  // The tensor is a view tensor. Create a new tensor holding the new
  // DeviceBufferRef.
  at::Tensor view_tensor = MakeTensor(buffer_ref);
  return view_tensor;
}

absl::StatusOr<absl::Span<const BoundedDynamicDimension>> GetDynamismInfo(
    const at::Tensor& tensor) {
  TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer_ref,
                      GetBaseBufferFromAtTensor(tensor));
  return buffer_ref.dynamic_dimensions();
}

}  // namespace torch_tpu
