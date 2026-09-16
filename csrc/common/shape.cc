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

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/layout.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace torch_tpu {

absl::StatusOr<Shape> MakeShape(const xla::Shape& xla_shape) {
  TT_ASSIGN_OR_RETURN(mlir::ElementType result_dtype,
                      ConvertTo<mlir::ElementType>(xla_shape.element_type()));
  Shape shape(CopyIntVector(xla_shape.dimensions()), result_dtype);
  shape.set_on_device_shape_is_dynamic(xla_shape.is_dynamic());
  if (xla_shape.has_layout()) {
    CustomLayout layout;
    layout.minor_to_major = CopyIntVector(xla_shape.layout().minor_to_major());
    // Preserve layout tiles so downstream consumers have complete on-device
    // layout information.
    for (const auto& tile : xla_shape.layout().tiles()) {
      layout.tiles.push_back(CopyIntVector(tile.dimensions()));
    }
    layout.element_size_in_bits = XlaEquivalentBitwidth(result_dtype);
    shape.set_layout(std::move(layout));
  }
  return shape;
}

absl::StatusOr<xla::Shape> MakeXlaShape(const Shape& shape) {
  const xla::PrimitiveType xla_type =
      ConvertTo<xla::PrimitiveType>(shape.dtype());
  TT_ASSIGN_OR_RETURN(xla::Shape xla_shape,
                      xla::ShapeUtil::MakeValidatedShapeWithDescendingLayout(
                          xla_type, shape.dimensions()));
  if (shape.layout().has_value()) {
    const CustomLayout& layout = *shape.layout();
    std::vector<xla::Tile> tiles;
    tiles.reserve(layout.tiles.size());
    for (const Indices& tile_dims : layout.tiles) {
      tiles.push_back(xla::Tile(tile_dims));
    }
    *xla_shape.mutable_layout() =
        xla::Layout(layout.minor_to_major, tiles, layout.element_size_in_bits);
    TT_RETURN_IF_ERROR(xla::ShapeUtil::ValidateShape(xla_shape));
  }
  return xla_shape;
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

}  // namespace torch_tpu
