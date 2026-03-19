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

#ifndef TORCH_TPU_COMMON_MACRO_UTILS_H_
#define TORCH_TPU_COMMON_MACRO_UTILS_H_

#include <cstddef>
#include <vector>

#include "absl/strings/str_split.h"  // IWYU pragma: keep for macro
#include "absl/strings/string_view.h"  // IWYU pragma: keep for macro

// Preprocessor utilities for making macro implementations more readable.

// TT_REQUIRE_SEMICOLON_ requires a semicolon after its expansion. Useful for
// enforcing that a macro must be used with a trailing semicolon.
//
// Example:
//   #define TT_FOO(x) \
//     void foo##x() { ... } \
//     TT_REQUIRE_SEMICOLON_
//
//   This ensures that TT_FOO() is used with a trailing semicolon, e.g.
//     TT_FOO(bar);
//   but not:
//     TT_FOO(bar)
//
// NOTE: This macro is not needed often. If we define a macro that expands to
// C++ statements or type definitions, we just need to omit the trailing
// semicolon from the macro definition. For example, in:
//
//   #define TT_BAR(x) \
//     class Bar##x { ... }
//   #define TT_BAZ(x) \
//     int x = ::torch_tpu::Baz(...)
//
// since the trailing semicolon is omitted in the macro definitions, users of
// TT_BAR() and TT_BAZ() will be forced to add a trailing semicolon. Therefore
// we don't need TT_REQUIRE_SEMICOLON_ in these macro definitions.
#define TT_REQUIRE_SEMICOLON_ \
  static_assert(true, "A semicolon is needed after this macro.")

// Converts a variadic list of identifiers into a vector of string_views.
//
// The arguments must be a list of single identifiers only. If the list is
// empty, the result will be an empty vector.
//
// Example:
//   std::vector<std::string_view> arg_names = TT_ARG_AS_STRINGS_(x, y, z);
//   // arg_names == {"x", "y", "z"}
//   std::vector<std::string_view> arg_names = TT_ARG_AS_STRINGS_();
//   // arg_names == {}
#define TT_ARGS_AS_STRINGS_(...) \
  ::torch_tpu::internal::ArgsAsStrings(#__VA_ARGS__)

// Path normalized __FILE__ macro replacement.
//
// Prefer this macro over __FILE__ mainly if it will get dumped to the user.
// This macro calls `NormalizeIfInsideBuildDirectory()` function with
// `__FILE__` as argument.
//
// See `NormalizeIfInsideBuildDirectory()` for more details.
#define TT_NORMALIZED_FILE \
  ::torch_tpu::internal::NormalizeIfInsideBuildDirectory(__FILE__)

namespace torch_tpu {
namespace internal {

#ifdef NDEBUG
inline constexpr bool kDebugMode = false;
#else
inline constexpr bool kDebugMode = true;
#endif  // NDEBUG

// Checks for whitespace at compile time.
[[nodiscard]] inline constexpr bool IsWhitespace(char c) noexcept {
  switch (c) {
    case ' ':
    case '\t':
    case '\n':
    case '\v':
    case '\f':
    case '\r':
      return true;
    default:
      return false;
  }
}

// Checks for identifier characters at compile time.
[[nodiscard]] inline constexpr bool IsIdentifierChar(char c) noexcept {
  return ('0' <= c && c <= '9') || ('a' <= c && c <= 'z') ||
         ('A' <= c && c <= 'Z') || c == '_';
}

// Checks if a string is a valid identifier.
[[nodiscard]] inline constexpr bool IsValidIdentifier(
    std::string_view s) noexcept {
  if (s.empty()) return false;
  for (char c : s) {
    if (!IsIdentifierChar(c)) return false;
  }
  return true;
}

// Returns true if the string is a comma-separated list where each item is an
// identifier name or "IgnoreInCacheKey(identifier)".
// This must be evaluated at compile time as we don't want to incur any
// runtime overhead.
[[nodiscard]] inline constexpr bool ArgsAreIdentifiers(
    const std::string_view csv_string) noexcept {
  if (csv_string.empty()) return true;

  size_t start = 0;
  while (start <= csv_string.size()) {
    size_t comma_pos = csv_string.find(',', start);
    std::string_view element =
        (comma_pos == std::string_view::npos)
            ? csv_string.substr(start)
            : csv_string.substr(start, comma_pos - start);

    while (!element.empty() && IsWhitespace(element.front())) {
      element.remove_prefix(1);
    }
    while (!element.empty() && IsWhitespace(element.back())) {
      element.remove_suffix(1);
    }

    if (element.empty()) {
      // Allow a trailing comma (e.g., "a,") to match original behavior.
      if (comma_pos == std::string_view::npos) {
        break;
      }
      return false;
    }

    bool valid = IsValidIdentifier(element);
    if (!valid) {
      constexpr std::string_view kPrefix = "IgnoreInCacheKey(";
      if (element.size() > kPrefix.size() &&
          element.substr(0, kPrefix.size()) == kPrefix &&
          element.back() == ')') {
        std::string_view inner =
            element.substr(kPrefix.size(), element.size() - kPrefix.size() - 1);
        while (!inner.empty() && IsWhitespace(inner.front())) {
          inner.remove_prefix(1);
        }
        while (!inner.empty() && IsWhitespace(inner.back())) {
          inner.remove_suffix(1);
        }
        valid = IsValidIdentifier(inner);
      }
    }

    if (!valid) return false;

    if (comma_pos == std::string_view::npos) break;
    start = comma_pos + 1;
  }
  return true;
}

// Splits a comma-separated string into a vector of items, each trimmed of
// whitespace.
std::vector<std::string_view> ArgsAsStrings(std::string_view csv_string);

// Normalize `path`, if it's inside the build directory (e.g. starts with
// "bazel-out/"), to be relative to whichever of the 2 is present in the path:
//
//   - "third_party/py/torch_tpu/" directory
//   - "torch_tpu/" directory
//
// As an example, this function translates (1) into (2):
//
//   1.
//   bazel-out/k8-fastbuild/bin/torch_tpu/ops/_virtual_includes/op_builder_utils/torch_tpu/ops/op_builder_utils.h
//
//   2.
//   torch_tpu/ops/op_builder_utils.h
//
// If `path` does not starts with "bazel-out" directory, and contains either
// "third_party/py/torch_tpu/" or "torch_tpu/" directory, this function returns
// `path` untouched.
//
// In some situations the build system uses a combination of -I and symlinks to
// allow the use of shorter paths to header files. In these cases, the
// `__FILE__` macro inside a header will expand to a path similar to the above
// format.
//
// To make it easier for developers to find the file in different build
// environments, we normalize the header location to be relative to the repo
// root instead.
const char* NormalizeIfInsideBuildDirectory(const char* path);

}  // namespace internal
}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_MACRO_UTILS_H_
