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
#include <string_view>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "torch_tpu/ops/view_decomposition/view_primitive_error_utils.h"

namespace torch_tpu {

namespace {

template <typename IsDynamicDim>
void CheckBroadcast(const BroadcastPrimitive& broadcast,
                    absl::Span<const int64_t> input_sizes,
                    const std::string_view error_message_suffix,
                    const IsDynamicDim& is_dynamic_dim) {
  ABSL_CHECK_EQ(  // CRASH_OK=Internal error on view decomposition.
      broadcast.broadcast_dimensions.size(), input_sizes.size())
      << "expected the BroadcastPrimitive broadcast dimensions size to be "
      << input_sizes.size() << ", which is the rank of the input layout, got "
      << broadcast.broadcast_dimensions.size() << error_message_suffix;

  Indices last_use_of(broadcast.new_sizes.size(), -1);

  for (auto i = 0; i < input_sizes.size(); ++i) {
    auto j = broadcast.broadcast_dimensions[i];
    ABSL_CHECK(  // CRASH_OK=Internal error on view decomposition.
        j >= 0 && j < broadcast.new_sizes.size())
        << "expected the BroadcastPrimitive to map the input dimension " << i
        << " to a dimension within the range [0, " << broadcast.new_sizes.size()
        << "), got " << j << error_message_suffix;

    ABSL_CHECK_EQ(  // CRASH_OK=Internal error on view decomposition.
        last_use_of[j], -1)
        << "expected the BroadcastPrimitive dimensions to map each input "
           "dimension to a unique dimension, got "
        << j << " at indices " << i << " and " << last_use_of[j]
        << error_message_suffix;
    last_use_of[j] = i;

    ABSL_CHECK(  // CRASH_OK=Internal error on view decomposition.
        is_dynamic_dim(i) || input_sizes[i] == 1 ||
        input_sizes[i] == broadcast.new_sizes[j])
        << "expected the BroadcastPrimitive input dimension at index " << i
        << " to be either dynamic, 1, or equal " << broadcast.new_sizes[j]
        << " (broadcasted dimension at index " << j
        << ") so that it is considered broadcastable, "
           "got "
        << input_sizes[i] << error_message_suffix;
  }
}

void CheckBroadcast(const BroadcastPrimitive& broadcast,
                    mlir::RankedTensorType type) {
  CheckBroadcast(broadcast, type.getShape(),
                 /* error_message_suffix= */
                 GetViewPrimitiveShloErrorSuffix(broadcast, type.getShape()),
                 /* is_dynamic_dim= */ [type](const int64_t i) {
                   return type.isDynamicDim(i);
                 });
};

void CheckBroadcast(const BroadcastPrimitive& broadcast,
                    const StridedLayout& layout) {
  CheckBroadcast(
      broadcast, GetSizes(layout),
      /* error_message_suffix= */ GetUpdateLayoutBugSuffix(broadcast, layout),
      /* is_dynamic_dim= */ [](const int64_t _) { return false; });
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

bool UpdateLayout(StridedLayout& layout, const BroadcastPrimitive& broadcast) {
  CheckBroadcast(broadcast, layout);

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
  CheckBroadcast(broadcast, input_type);

  return Broadcast(input, broadcast.new_sizes, broadcast.broadcast_dimensions);
}

}  // namespace torch_tpu
