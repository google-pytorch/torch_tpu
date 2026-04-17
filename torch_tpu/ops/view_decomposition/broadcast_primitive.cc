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

#include "torch_tpu/ops/view_decomposition/broadcast_primitive.h"

#include <cstdint>
#include <ostream>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/DebugStringHelper.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

namespace {

absl::Status ValidateBroadcastImpl(const BroadcastPrimitive& broadcast,
                                   absl::Span<const int64_t> input_sizes,
                                   mlir::RankedTensorType input_type) {
  TT_RET_CHECK(broadcast.broadcast_dimensions.size() == input_sizes.size(),
               error::kInvalidArgument)
      << "broadcast_dimensions must have the same "
         "size as the input tensor, got input size "
      << input_sizes.size() << " and broadcast_dimensions size "
      << broadcast.broadcast_dimensions.size();
  std::vector<bool> dim_used(broadcast.new_sizes.size(), false);
  for (auto i = 0; i < input_sizes.size(); ++i) {
    auto j = broadcast.broadcast_dimensions[i];
    TT_RET_CHECK(j >= 0 && j < broadcast.new_sizes.size(),
                 error::kInvalidArgument)
        << "broadcast_dimensions[" << i << "] = " << j
        << " which is out of bounds for new_sizes of size "
        << broadcast.new_sizes.size();
    TT_RET_CHECK(!dim_used[j], error::kInvalidArgument)
        << "broadcast_dimensions has a duplicate input index: "
        << ToString(broadcast.broadcast_dimensions);
    dim_used[j] = true;
    const bool is_dynamic_dim = input_type && input_type.isDynamicDim(i);
    TT_RET_CHECK(input_sizes[i] == 1 || is_dynamic_dim ||
                     input_sizes[i] == broadcast.new_sizes[j],
                 error::kInvalidArgument)
        << "cannot broadcast input dimension " << i << " of size "
        << input_sizes[i] << " to output dimension " << j << " of size "
        << broadcast.new_sizes[j];
  }
  return absl::OkStatus();
}

absl::Status ValidateBroadcast(const BroadcastPrimitive& broadcast,
                               absl::Span<const int64_t> input_sizes) {
  return ValidateBroadcastImpl(broadcast, input_sizes, nullptr /*input_type*/);
}

absl::Status ValidateBroadcast(const BroadcastPrimitive& broadcast,
                               mlir::RankedTensorType input_type) {
  return ValidateBroadcastImpl(broadcast, input_type.getShape(), input_type);
}

absl::Status ValidateBroadcast(const BroadcastPrimitive& broadcast,
                               const StridedLayout& layout) {
  Dimensions input_dims;
  input_dims.reserve(layout.strided_dims.size());
  for (const auto& dim : layout.strided_dims) {
    input_dims.push_back(dim.size);
  }
  return ValidateBroadcast(broadcast, input_dims);
}

}  // namespace

std::ostream& operator<<(std::ostream& os,
                         const BroadcastPrimitive& broadcast) {
  os << "broadcast(base_shape=" << ToString(broadcast.base_shape)
     << ", new_sizes=" << ToString(broadcast.new_sizes)
     << ", broadcast_dimensions=" << ToString(broadcast.broadcast_dimensions)
     << ")";
  return os;
}

absl::StatusOr<bool> UpdateLayoutImpl(StridedLayout& layout,
                                      const BroadcastPrimitive& broadcast) {
  TT_RETURN_IF_ERROR(ValidateBroadcast(broadcast, layout));

  // Storage offset is unchanged.
  StridedLayout new_layout = {.storage_offset = layout.storage_offset};
  new_layout.strided_dims.reserve(broadcast.new_sizes.size());
  for (int64_t new_size : broadcast.new_sizes) {
    new_layout.strided_dims.push_back({.size = new_size, .stride = 0});
  }
  for (auto i = 0; i < layout.strided_dims.size(); ++i) {
    auto j = broadcast.broadcast_dimensions[i];
    if (layout.strided_dims[i].size == broadcast.new_sizes[j]) {
      // If a broadcast preserves a size, we also preserve the stride.
      new_layout.strided_dims[j].stride = layout.strided_dims[i].stride;
    }
  }

  bool updated = new_layout != layout;
  layout = new_layout;
  return updated;
}

absl::StatusOr<mlir::MlirOp> ViewPrimitiveShlo(
    mlir::MlirOp input, const BroadcastPrimitive& broadcast) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  ABSL_VLOG(3) << "ViewPrimitiveShlo: input_type: "
               << mlir::debugString(input_type) << " broadcast dims: "
               << ToString(broadcast.broadcast_dimensions)
               << " new sizes: " << ToString(broadcast.new_sizes);
  TT_RETURN_IF_ERROR(ValidateBroadcast(broadcast, input_type));

  return Broadcast(input, broadcast.new_sizes, broadcast.broadcast_dimensions);
}

}  // namespace torch_tpu
