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

#include "csrc/common/shape.h"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/layout.h"
#include "xla/shape.h"

namespace torch_tpu {

absl::StatusOr<Shape> MakeShape(const xla::Shape& xla_shape) {
  TT_ASSIGN_OR_RETURN(mlir::ElementType result_dtype,
                      ConvertTo<mlir::ElementType>(xla_shape.element_type()));
  Shape shape(CopyIntVector(xla_shape.dimensions()), result_dtype);
  if (xla_shape.has_layout()) {
    CustomLayout layout;
    layout.minor_to_major = CopyIntVector(xla_shape.layout().minor_to_major());
    // Preserve layout tiles so downstream consumers have complete on-device
    // layout information.
    for (const auto& tile : xla_shape.layout().tiles()) {
      layout.tiles.push_back(CopyIntVector(tile.dimensions()));
    }
    layout.element_size_in_bits = xla_shape.layout().element_size_in_bits();
    shape.set_layout(std::move(layout));
  }
  return shape;
}

xla::Layout ToXlaLayout(const CustomLayout& layout) {
  std::vector<xla::Tile> tiles;
  tiles.reserve(layout.tiles.size());
  for (const Indices& tile_indices : layout.tiles) {
    tiles.push_back(xla::Tile(tile_indices));
  }
  return xla::Layout(layout.minor_to_major, tiles, layout.element_size_in_bits);
}

std::string ToString(const Shape& s) {
  std::ostringstream os;
  for (auto d : s.dimensions()) {
    os << d << "x";
  }
  os << ToString(s.dtype());
  if (!s.dynamic_dimensions().empty()) {
    os << "[";
    for (auto d : s.dynamic_dimensions()) {
      os << d.dimension << ":" << d.lower_bound << ":" << d.upper_bound << ",";
    }
    os << "]";
  }
  return os.str();
}

bool TilesMatchDefault(absl::Span<const Indices> layout_tiles,
                       absl::Span<const xla::Tile> default_tiles) {
  if (layout_tiles.empty()) {
    return true;
  }
  if (layout_tiles.size() != default_tiles.size()) {
    return false;
  }
  for (size_t t = 0; t < layout_tiles.size(); ++t) {
    if (absl::MakeConstSpan(layout_tiles[t]) != default_tiles[t].dimensions()) {
      return false;
    }
  }
  return true;
}

bool TilesMatchDefault(absl::Span<const xla::Tile> layout_tiles,
                       absl::Span<const xla::Tile> default_tiles) {
  if (layout_tiles.empty()) {
    return true;
  }
  if (layout_tiles.size() != default_tiles.size()) {
    return false;
  }
  for (size_t t = 0; t < layout_tiles.size(); ++t) {
    if (layout_tiles[t] != default_tiles[t]) {
      return false;
    }
  }
  return true;
}

bool MinorToMajorMatchesIgnoringDegenerate(
    absl::Span<const int64_t> mtm_a, absl::Span<const int64_t> mtm_b,
    absl::Span<const int64_t> dimensions) {
  if (mtm_a == mtm_b) {
    return true;
  }
  if (mtm_a.size() != mtm_b.size() || mtm_a.size() != dimensions.size()) {
    return false;
  }
  bool has_degenerate = false;
  for (int64_t dim : dimensions) {
    if (dim <= 1) {
      has_degenerate = true;
      break;
    }
  }
  if (!has_degenerate) {
    return false;
  }
  absl::InlinedVector<int64_t, 6> filtered_a;
  absl::InlinedVector<int64_t, 6> filtered_b;
  for (int64_t axis : mtm_a) {
    if (axis >= 0 && axis < static_cast<int64_t>(dimensions.size()) &&
        dimensions[axis] > 1) {
      filtered_a.push_back(axis);
    }
  }
  for (int64_t axis : mtm_b) {
    if (axis >= 0 && axis < static_cast<int64_t>(dimensions.size()) &&
        dimensions[axis] > 1) {
      filtered_b.push_back(axis);
    }
  }
  return filtered_a == filtered_b;
}

}  // namespace torch_tpu
