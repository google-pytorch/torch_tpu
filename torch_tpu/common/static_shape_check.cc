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

#include "torch_tpu/common/static_shape_check.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ATen/core/TensorBody.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"

namespace torch_tpu {

namespace {

// Return a string representation of the shape of `type`.
//
// This is similar to calling `ToString(type.getShape())`, with the difference
// that it will replace every dynamic dimension by the word 'dyn'.
//
// For example, given the input tensor:
//
//   ```py
//   tensor = torch.rand(2, 3, 4, 5)
//   dynamism.mark_dynamic(tensor, 0, 1, 10)
//   dynamism.mark_dynamic(tensor, 2, 1, 10)
//   ```
//
// This function will return the string below:
//
//   [dyn, 3, dyn, 5]
//
std::string GetShapeStringWithDynamicDimensions(mlir::RankedTensorType type) {
  std::vector<std::string> str(type.getRank());

  absl::c_transform(type.getShape(), str.begin(), [](const int64_t size) {
    // String representation of dimension `i`.
    //   - "dyn", if `i` is dynamic
    //   - the size of dimension `i`, otherwise
    return mlir::ShapedType::isDynamic(size) ? "dyn" : std::to_string(size);
  });

  return absl::StrCat("[", absl::StrJoin(str, /* separator= */ ", "), "]");
}

// Return a string representation of the shape of `sizes`.
//
// This is similar to calling `ToString(sizes)`, with the difference
// that it will append every dynamic dimension with its upper bound.
//
// For example, given the input tensor:
//
//   ```py
//   tensor = torch.rand(2, 3, 4, 5)
//   dynamism.mark_dynamic(tensor, 0, 1, 10)
//   dynamism.mark_dynamic(tensor, 2, 1, 10)
//   ```
//
// This function will return the string below:
//
//   [2 (up to 10), 3, 4 (up to 10), 5]
//
std::string GetShapeStringWithDynamicDimensionsUpperBound(
    absl::Span<const int64_t> sizes,
    absl::Span<const BoundedDynamicDimension> dynamic_dimensions_info) {
  // Upper bound of each dynamic dimension, or `nullopt` otherwise.
  std::vector<std::optional<int64_t>> dimensions_upper_bound(sizes.size(),
                                                             std::nullopt);
  // Populate from `dynamic_dimensions_info`.
  for (const auto& info : dynamic_dimensions_info) {
    dimensions_upper_bound[info.dimension] = info.upper_bound;
  }

  // Convert each dimension size to its string representation.
  std::vector<std::string> str(sizes.size());

  for (int64_t i = 0; i < sizes.size(); ++i) {
    const auto& upper_bound = dimensions_upper_bound[i];
    str[i] = upper_bound.has_value()
                 ? absl::StrCat(sizes[i], " (up to ", *upper_bound, ")")
                 : std::to_string(sizes[i]);
  }

  return absl::StrCat("[", absl::StrJoin(str, /* separator= */ ", "), "]");
}

// Runs the actual static shape check.
template <typename GetExtraErrorMessage>
absl::Status CheckStaticShapeImpl(
    const int64_t dynamic_dimensions_number, const std::string_view name,
    const GetExtraErrorMessage& get_extra_error_message) {
  TT_RET_CHECK(dynamic_dimensions_number == 0, error::kInvalidArgument)
      << "expected all dimensions of the " << name
      << " tensor to be static, got " << dynamic_dimensions_number
      << " dynamic dimension" << (dynamic_dimensions_number == 1 ? "" : "s")
      << get_extra_error_message();

  return absl::OkStatus();
}

}  // namespace

absl::Status CheckStaticShape(mlir::RankedTensorType type,
                              const std::string_view arg_name) {
  auto get_extra_error_message = [type]() {
    return absl::StrCat(" within shape ",
                        GetShapeStringWithDynamicDimensions(type));
  };

  return CheckStaticShapeImpl(type.getNumDynamicDims(), arg_name,
                              get_extra_error_message);
}

absl::Status CheckStaticShape(const at::Tensor& tensor,
                              const DeviceBufferRef& buffer_ref,
                              const std::string_view arg_name) {
  absl::Span<const BoundedDynamicDimension> dynamic_dimensions_info =
      buffer_ref.dynamic_dimensions();

  auto get_extra_error_message = [&]() {
    if (tensor.sizes() != buffer_ref.dimensions()) {
      // TODO: make sure that the BoundedDynamicDimension information gets
      // propagated into view tensors.
      //
      // Since `tensor` is a view, we can't get the exact dynamic dimensions
      // information for it.
      return absl::StrCat(" in the underlying tensor behind a view of shape ",
                          ToString(tensor.sizes()));
    }
    return absl::StrCat(" within shape ",
                        GetShapeStringWithDynamicDimensionsUpperBound(
                            tensor.sizes(), dynamic_dimensions_info));
  };

  return CheckStaticShapeImpl(dynamic_dimensions_info.size(), arg_name,
                              get_extra_error_message);
}

}  // namespace torch_tpu
