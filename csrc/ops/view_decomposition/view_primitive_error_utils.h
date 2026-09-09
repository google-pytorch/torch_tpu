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

#ifndef TORCH_TPU_CSRC_OPS_VIEW_DECOMPOSITION_VIEW_PRIMITIVE_ERROR_UTILS_H_
#define TORCH_TPU_CSRC_OPS_VIEW_DECOMPOSITION_VIEW_PRIMITIVE_ERROR_UTILS_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/view_decomposition/strided_layout.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

// Utilities for reporting errors for view decomposition.
//
// Template vs. Using the `ViewPrimitive` Type
// ===========================================
//
// We intentionally don't use the `ViewPrimitive` type. The reason is that it
// creates the following dependency cyle:
//
//   - `view_primitive_error_utils` needs the `ViewPrimitive` type definition in
//     `view_sequence` library
//
//   - `view_sequence` needs each view primitive type definition in their
//     respective libraries
//
//   - Each view primitive library (e.g. `bitcast_primitive`) needs the error
//     suffix functions in the `view_primitive_error_utils` library

namespace torch_tpu {

// Settings for controlling the bug suffix.
//
// This enum is used to decide whether to append the TorchTPU bug to the error
// message: "; this is a TorchTPU bug".
enum class ViewPrimitiveBugSuffix {
  kHide,
  kAppend,
};

// Options to the `GetViewPrimitiveErrorSuffix()` function.
struct ViewPrimitiveErrorOptions {
  // Adds a "; " before the error message.
  bool leading_semicolon = true;

  // Appends the bug suffix at the end of the error message.
  ViewPrimitiveBugSuffix bug_suffix = ViewPrimitiveBugSuffix::kAppend;
};

inline bool ShouldAppendBugSuffix(const ViewPrimitiveBugSuffix bug_suffix) {
  return bug_suffix == ViewPrimitiveBugSuffix::kAppend;
}

// Returns a string to be appended to checks in the view decomposition logic,
// describing the primitive that raised the error.
//
// `ViewPrimitiveT` should be a `ViewPrimitive` type.
template <typename ViewPrimitiveT>
std::string GetViewPrimitiveErrorSuffix(
    const ViewPrimitiveT& view_primitive,
    const ViewPrimitiveErrorOptions options = ViewPrimitiveErrorOptions{}) {
  constexpr std::string_view kBugSuffix = "; this is a TorchTPU bug";

  // Add leading phrase separator for better readability.
  const std::string_view maybe_leading_semicolon =
      options.leading_semicolon ? "; " : "";
  const std::string_view maybe_bug_suffix =
      ShouldAppendBugSuffix(options.bug_suffix) ? kBugSuffix : "";

  return absl::StrCat(maybe_leading_semicolon,
                      "primitive=", ToString(view_primitive), maybe_bug_suffix);
}

// Returns a string to be appended to checks in the view decomposition logic,
// specifically when calling `UpdateLayout()`.
//
// `ViewPrimitiveT` should be a `ViewPrimitive` type.
template <typename ViewPrimitiveT>
std::string GetUpdateLayoutBugSuffix(const ViewPrimitiveT& view_primitive,
                                     const StridedLayout& layout) {
  return absl::StrCat("; calling UpdateLayout() with layout=", ToString(layout),
                      " and ",
                      GetViewPrimitiveErrorSuffix(
                          view_primitive, {.leading_semicolon = false}));
}

// Returns a string to be appended to checks in the view decomposition logic,
// specifically when calling `ViewPrimitiveShlo()`.
//
// This function handles dynamic shapes by replacing them with the word "dyn",
// before returning the concatenated string.
//
// The `mode` parameter controls whether the "; this is a TorchTPU bug" error
// message should be appended to the result. By default, it is
// `kAppendBugSuffix`, which appends the bug suffix.
//
// `ViewPrimitiveT` should be a `ViewPrimitive` type.
template <typename ViewPrimitiveT>
std::string GetViewPrimitiveShloErrorSuffix(
    const ViewPrimitiveT& view_primitive, const absl::Span<const int64_t> shape,
    const ViewPrimitiveBugSuffix bug_suffix = ViewPrimitiveBugSuffix::kAppend) {
  // Extra step for handling dynamic shapes.
  std::vector<std::string> dynamic_shape_to_str(shape.size());

  // Replace every dynamic size by the "dyn" word.
  absl::c_transform(shape, dynamic_shape_to_str.begin(), [](int64_t size) {
    return mlir::ShapedType::isDynamic(size) ? "dyn" : ToString(size);
  });

  return absl::StrCat(
      "; calling ViewPrimitiveShlo() with input shape=",
      ToString(dynamic_shape_to_str), " and ",
      GetViewPrimitiveErrorSuffix(view_primitive, {.leading_semicolon = false,
                                                   .bug_suffix = bug_suffix}));
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_VIEW_DECOMPOSITION_VIEW_PRIMITIVE_ERROR_UTILS_H_
