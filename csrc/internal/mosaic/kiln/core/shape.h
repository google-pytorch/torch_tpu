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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_SHAPE_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_SHAPE_H_

#include <concepts>
#include <cstdint>
#include <limits>
#include <ostream>
#include <string>

#include "absl/base/attributes.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace torch_tpu::kiln {

using Dimensions = llvm::SmallVector<int64_t, 4>;

// Sentinel for a runtime-sized (dynamic) dimension in `Dimensions` / `Shape`
// (`?` in MLIR), matching `mlir::ShapedType::kDynamic`. Only valid in
// `Dimensions` / `Shape` specifications (for dynamic slice sizes, pass a
// runtime `Scalar` to `Slice` or use `all`).
inline constexpr int64_t kDynamic = std::numeric_limits<int64_t>::min();

// Combines tensor dimensions with an element data type.
class Shape {
 public:
  Shape(llvm::ArrayRef<int64_t> dims, DType dtype)
      : dims_(dims), dtype_(dtype) {}

  Shape(const Shape&) = default;
  Shape& operator=(const Shape&) = default;
  Shape(Shape&&) noexcept = default;
  Shape& operator=(Shape&&) noexcept = default;

  llvm::ArrayRef<int64_t> dims() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return dims_;
  }
  DType dtype() const { return dtype_; }
  int64_t rank() const { return dims_.size(); }

  friend bool operator==(const Shape& lhs, const Shape& rhs) {
    return lhs.dims_ == rhs.dims_ && lhs.dtype_ == rhs.dtype_;
  }
  friend bool operator!=(const Shape& lhs, const Shape& rhs) {
    return !(lhs == rhs);
  }

  std::string ToString() const {
    return absl::StrCat(dtype_, "[",
                        absl::StrJoin(dims_, ", ",
                                      [](std::string* out, int64_t dim) {
                                        if (dim == kDynamic) {
                                          out->append("?");
                                        } else {
                                          absl::StrAppend(out, dim);
                                        }
                                      }),
                        "]");
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Shape& shape) {
    sink.Append(shape.ToString());
  }

 private:
  Dimensions dims_;
  DType dtype_;
};

inline std::ostream& operator<<(std::ostream& os, const Shape& shape) {
  return os << shape.ToString();
}

inline Shape DType::operator()(llvm::ArrayRef<int64_t> dims) const {
  return Shape(dims, *this);
}

template <std::convertible_to<int64_t>... Dims>
inline Shape DType::operator()(Dims... dims) const {
  return Shape(llvm::ArrayRef<int64_t>{static_cast<int64_t>(dims)...}, *this);
}

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_SHAPE_H_
