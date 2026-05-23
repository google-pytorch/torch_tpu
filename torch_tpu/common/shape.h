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

#ifndef TORCH_TPU_COMMON_SHAPE_H_
#define TORCH_TPU_COMMON_SHAPE_H_

#include <cstdint>
#include <string>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/dimension_types.h"
#include "xla/shape.h"

namespace torch_tpu {

// A dimension that should be interpreted as dynamic.
// Tensors always have a concrete shape, but we might want to build operations
// they are involved in with a dynamic dimension so they can be reused.
struct BoundedDynamicDimension {
  int64_t dimension;
  int64_t lower_bound;
  int64_t upper_bound;

  friend bool operator==(const BoundedDynamicDimension& lhs,
                         const BoundedDynamicDimension& rhs) {
    return lhs.dimension == rhs.dimension &&
           lhs.lower_bound == rhs.lower_bound &&
           lhs.upper_bound == rhs.upper_bound;
  }
};

// A tensor shape, consisting of dimensions and element type.
// This is roughly equivalent to xla::Shape.
class Shape {
 public:
  Shape() = default;
  Shape(Dimensions dimensions, mlir::ElementType dtype)
      : Shape(std::move(dimensions), dtype, {}) {}
  Shape(Dimensions dimensions, mlir::ElementType dtype,
        absl::InlinedVector<BoundedDynamicDimension, 1> dynamic_dimensions)
      : dimensions_(std::move(dimensions)),
        dtype_(dtype),
        dynamic_dimensions_(std::move(dynamic_dimensions)) {}

  const Dimensions& dimensions() const { return dimensions_; }
  Dimensions& dimensions() { return dimensions_; }

  mlir::ElementType dtype() const { return dtype_; }
  void set_dtype(mlir::ElementType dtype) { dtype_ = dtype; }

  const absl::InlinedVector<BoundedDynamicDimension, 1>& dynamic_dimensions()
      const {
    return dynamic_dimensions_;
  }
  absl::InlinedVector<BoundedDynamicDimension, 1>& dynamic_dimensions() {
    return dynamic_dimensions_;
  }

  friend bool operator==(const Shape& lhs, const Shape& rhs) {
    return lhs.dtype_ == rhs.dtype_ && lhs.dimensions_ == rhs.dimensions_ &&
           lhs.dynamic_dimensions_ == rhs.dynamic_dimensions_;
  }

 private:
  Dimensions dimensions_;
  mlir::ElementType dtype_;
  // We choose 1 as the inlined size because most Shapes are not dynamic, and
  // most dynamic Shapes are dynamic in 1 dimension only, so this allows for
  // no heap allocation in the common case.
  absl::InlinedVector<BoundedDynamicDimension, 1> dynamic_dimensions_;
};
static_assert(sizeof(Shape) == 96);

// Converts an XLA shape to a torch_tpu shape.
absl::StatusOr<Shape> MakeShape(const xla::Shape& xla_shape);

std::string ToString(const Shape& s);

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_SHAPE_H_
