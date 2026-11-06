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
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/pjrt/pjrt_state.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

int64_t FindFp4PackedDim(const at::Tensor& tensor) {
  const int64_t rank = tensor.dim();
  if (rank == 0) {
    return -1;
  }
  if (tensor.stride(rank - 1) != 1) {
    for (int64_t d = 0; d < rank - 1; ++d) {
      if (tensor.stride(d) == 1 && tensor.size(d) > 1) {
        return d;
      }
    }
    return -1;
  }
  if (tensor.size(rank - 1) == 1) {
    return rank - 1;
  }
  for (int64_t d = 0; d < rank - 1; ++d) {
    if (tensor.size(d) == 1 && tensor.stride(d) == 1) {
      return d;
    }
  }
  return rank - 1;
}

// Applies any overrides from `shape.layout()` onto `layout`.
void ApplyCustomLayout(const Shape& shape, xla::Layout& layout) {
  if (!shape.layout().has_value()) {
    return;
  }
  const CustomLayout& custom_layout = *shape.layout();
  if (!custom_layout.minor_to_major.empty()) {
    *layout.mutable_minor_to_major() =
        xla::DimensionVector(custom_layout.minor_to_major.begin(),
                             custom_layout.minor_to_major.end());
  }
  // Dynamic on-device buffers have physical tiling determined by their upper
  // bounds rather than their runtime logical dimensions, so static custom
  // tiles do not apply.
  if (!shape.on_device_shape_is_dynamic() && !custom_layout.tiles.empty()) {
    layout.clear_tiles();
    for (const Indices& tile_dims : custom_layout.tiles) {
      layout.mutable_tiles()->emplace_back(tile_dims);
    }
  }
  if (custom_layout.element_size_in_bits != 0 &&
      custom_layout.element_size_in_bits !=
          XlaEquivalentBitwidth(shape.dtype())) {
    layout.set_element_size_in_bits(custom_layout.element_size_in_bits);
  }
}

}  // namespace

absl::StatusOr<TpuLayout> ResolveTpuLayout(const at::Tensor& tensor) {
  const auto scalar_type = tensor.scalar_type();
  TT_ASSIGN_OR_RETURN(const auto dtype,
                      ConvertTo<mlir::ElementType>(scalar_type));

  Dimensions sizes = CopyIntVector(tensor.sizes());
  Strides strides = CopyIntVector(tensor.strides());
  int64_t offset = tensor.storage_offset();

  if (scalar_type == at::kFloat4_e2m1fn_x2) {
    // FP4 values are packed 2-per-byte in PyTorch storage (represented by the
    // float4_e2m1fn_x2 dtype). For StableHLO, we need to expose the layout
    // in terms of individual unpacked FP4 elements.
    //
    // 1. Double the size of the packed (stride-1) dimension because each
    //    packed element contains 2 FP4 values.
    // 2. Double the strides of all other dimensions to maintain correct
    //    element offsets now that the packed dimension size has doubled.
    // 3. Double the storage offset to convert it from packed element offset
    //    to unpacked element offset.
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch core rejects 0D FP4 tensors
                   // generally on all devices.
        !sizes.empty(), error::kInvalidArgument)
        << "expected float4_e2m1fn_x2 tensors to be at least 1-dimensional, "
           "got 0-dimensional";
    const int64_t packed_dim = FindFp4PackedDim(tensor);
    TT_RET_CHECK(packed_dim >= 0, error::kInvalidArgument)
        << "expected packed dimension to be contiguous (stride 1) for "
           "float4_e2m1fn_x2 view, got strides "
        << ToString(strides);
    sizes[packed_dim] *= 2;
    for (size_t i = 0; i < strides.size(); ++i) {
      if (static_cast<int64_t>(i) != packed_dim) {
        strides[i] *= 2;
      }
    }
    offset *= 2;
  }
  return TpuLayout{std::move(sizes), std::move(strides), offset, dtype};
}

absl::StatusOr<std::vector<xla::Layout>> ResolveXlaLayouts(
    absl::Span<const Shape> shapes) {
  if (shapes.empty()) {
    return std::vector<xla::Layout>{};
  }
  TT_ASSIGN_OR_RETURN(xla::PjRtClient* const client,
                      PjrtBackend::GetInstance().GetClient());
  std::vector<xla::Layout> layouts;
  layouts.reserve(shapes.size());
  for (const Shape& shape : shapes) {
    TT_ASSIGN_OR_RETURN(
        xla::Layout layout,
        AdaptXlaError(client->GetDefaultLayout(
                          ConvertTo<xla::PrimitiveType>(shape.dtype()),
                          shape.dimensions()),
                      /*context=*/"failed to get default layout"));
    ApplyCustomLayout(shape, layout);
    layouts.push_back(std::move(layout));
  }
  return layouts;
}

absl::StatusOr<std::optional<CustomLayout>>
ExtractCustomLayoutIfDifferentFromDefault(const Shape& shape) {
  if (!shape.layout().has_value()) {
    return std::nullopt;
  }
  TT_ASSIGN_OR_RETURN(xla::PjRtClient* const client,
                      PjrtBackend::GetInstance().GetClient());
  TT_ASSIGN_OR_RETURN(
      const xla::Layout default_layout,
      AdaptXlaError(
          client->GetDefaultLayout(ConvertTo<xla::PrimitiveType>(shape.dtype()),
                                   shape.dimensions()),
          /*context=*/"failed to get default layout"));
  xla::Layout resolved_layout = default_layout;
  ApplyCustomLayout(shape, resolved_layout);
  if (resolved_layout == default_layout) {
    return std::nullopt;
  }
  ABSL_VLOG(3) << "[ExtractCustomLayoutIfDifferentFromDefault] Layout "
               << resolved_layout.ToString()
               << " differs from PjRT default layout "
               << default_layout.ToString() << " for shape ("
               << ToString(shape.dimensions()) << ")";
  CustomLayout result = *shape.layout();
  if (shape.on_device_shape_is_dynamic()) {
    // Dynamic on-device buffers (e.g. outputs of compiled models with bounded
    // dynamism) have physical tiling determined by their upper bounds rather
    // than their runtime logical dimensions. Propagate minor_to_major but omit
    // tiling so downstream eager kernels do not force static tile padding onto
    // the dynamic buffer.
    result.tiles.clear();
  }
  return result;
}

}  // namespace torch_tpu
