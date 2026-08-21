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

#include "torch_tpu/csrc/ops/macros/logging.h"

#include <cctype>
#include <regex>  // NOLINT
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/strings/strip.h"

namespace torch_tpu {
namespace internal {

// Given a string of comma-separated types, returns the index of the end of the
// first type.
[[nodiscard]] static int FindTypeEnd(std::string_view args_str) {
  int open_brackets = 0;
  int open_parentheses = 0;
  int i = 0;
  for (; i < args_str.size(); ++i) {
    const char c = args_str[i];
    if (c == '<') {
      ++open_brackets;
    } else if (c == '>') {
      --open_brackets;
    } else if (c == '(') {
      ++open_parentheses;
    } else if (c == ')') {
      --open_parentheses;
    } else if (c == ',' && open_brackets == 0 && open_parentheses == 0) {
      // The type ends on a comma that is not inside any brackets or
      // parentheses. Or it ends on the end of the string.
      break;
    }
  }
  return i;
}

// Normalizes a type name parsed from __PRETTY_FUNCTION__.
// __PRETTY_FUNCTION__ doesn't normalize the type names, so we may get
// different strings depending on how the function definition is written.
[[nodiscard]] static std::string NormalizePrettyFunctionTypeName(
    std::string_view type_name) {
  // Ignore the "const " prefix if any, as we only care about the underlying
  // type.
  type_name = absl::StripPrefix(type_name, "const ");
  // Ignore the " &" suffix if any, as we only care about the underlying type.
  type_name = absl::StripSuffix(type_name, " &");

  // Replace type aliases with their underlying types.

  using Regex = const absl::NoDestructor<std::regex>;

  static Regex kOptional(R"(\bc10::optional\b)");
  std::string type_name_str =
      std::regex_replace(std::string(type_name), *kOptional, "std::optional");

  static Regex kStringView(R"(\bc10::string_view\b)");
  type_name_str =
      std::regex_replace(type_name_str, *kStringView, "std::string_view");

  static Regex kLongLong(R"(\blong long\b)");
  type_name_str = std::regex_replace(type_name_str, *kLongLong, "int64_t");

  static Regex kLong(R"(\blong\b)");
  type_name_str = std::regex_replace(type_name_str, *kLong, "int64_t");

  static Regex kSymInt(R"(\bc10::SymInt\b)");
  type_name_str = std::regex_replace(type_name_str, *kSymInt, "at::SymInt");

  static Regex kUnsignedInt64(R"(\bunsigned int64_t\b|\bsize_t\b)");
  type_name_str =
      std::regex_replace(type_name_str, *kUnsignedInt64, "uint64_t");

  static Regex kArrayRef(R"(\bc10::ArrayRef\b)");
  type_name_str = std::regex_replace(type_name_str, *kArrayRef, "at::ArrayRef");

  static Regex kScalarType(R"(\bc10::ScalarType\b)");
  type_name_str =
      std::regex_replace(type_name_str, *kScalarType, "at::ScalarType");

  static Regex kOptionalIntArrayRef(R"(\bc10::OptionalIntArrayRef\b)");
  type_name_str = std::regex_replace(type_name_str, *kOptionalIntArrayRef,
                                     "at::OptionalIntArrayRef");

  static Regex kOptionalArrayRef(R"(\bc10::OptionalArrayRef\b)");
  type_name_str = std::regex_replace(type_name_str, *kOptionalArrayRef,
                                     "at::OptionalArrayRef");

  static Regex kOptionalInt64ArrayRef(
      R"(\bat::OptionalArrayRef\s*<\s*int64_t\s*>)");
  type_name_str = std::regex_replace(type_name_str, *kOptionalInt64ArrayRef,
                                     "at::OptionalIntArrayRef");

  static Regex kIntArrayRef(
      R"(\bc10::IntArrayRef\b|\bat::ArrayRef\s*<\s*int64_t\s*>)");
  type_name_str =
      std::regex_replace(type_name_str, *kIntArrayRef, "at::IntArrayRef");

  static Regex kDevice(R"(\bc10::Device\b)");
  type_name_str = std::regex_replace(type_name_str, *kDevice, "at::Device");

  static Regex kLayout(R"(\bc10::Layout\b)");
  type_name_str = std::regex_replace(type_name_str, *kLayout, "at::Layout");

  static Regex kMemoryFormat(R"(\bc10::MemoryFormat\b)");
  type_name_str =
      std::regex_replace(type_name_str, *kMemoryFormat, "at::MemoryFormat");

  static Regex kStorage(R"(\bc10::Storage\b)");
  type_name_str = std::regex_replace(type_name_str, *kStorage, "at::Storage");

  static Regex kSymIntArrayRef(
      R"(\bc10::SymIntArrayRef\b|\bat::ArrayRef\s*<\s*at::SymInt\s*>)");
  type_name_str =
      std::regex_replace(type_name_str, *kSymIntArrayRef, "at::SymIntArrayRef");

  static Regex kOptionalSymIntArrayRef(
      R"(\bat::OptionalArrayRef\s*<\s*at::SymInt\s*>)");
  type_name_str = std::regex_replace(type_name_str, *kOptionalSymIntArrayRef,
                                     "at::OptionalSymIntArrayRef");

  static Regex kTensorListRef(R"(\bc10::IListRef<at::Tensor>)");
  type_name_str =
      std::regex_replace(type_name_str, *kTensorListRef, "at::ITensorListRef");

  return type_name_str;
}

std::vector<std::string> ParseArgTypesOrEmpty(const std::string_view func_sig) {
  std::vector<std::string> arg_types;
  // Parse the argument types from the function signature, e.g.
  // "void foo(int a, const char* b, double c)"
  // -> {"int", "const char*", "double"}.

  // Find the last '(' and ')' character. We skip earlier ones because they may
  // be part of the function name, e.g.
  // "void torch_tpu::(anonymous namespace)::NullaryFunc()".
  const int start_idx = func_sig.find_last_of('(');
  if (start_idx == std::string_view::npos) {
    return arg_types;
  }
  const int end_idx = func_sig.find_last_of(')');
  if (end_idx == std::string_view::npos) {
    return arg_types;
  }
  std::string_view args_str =
      func_sig.substr(start_idx + 1, end_idx - start_idx - 1);
  while (!args_str.empty()) {
    const int type_end_idx = FindTypeEnd(args_str);
    arg_types.push_back(
        NormalizePrettyFunctionTypeName(args_str.substr(0, type_end_idx)));
    args_str = args_str.substr(type_end_idx < args_str.size() ? type_end_idx + 1
                                                              : type_end_idx);
    // Skip leading whitespace.
    while (!args_str.empty() && std::isspace(args_str.front())) {
      args_str = args_str.substr(1);
    }
  }
  return arg_types;
}

}  // namespace internal
}  // namespace torch_tpu
