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

// This file is needed to make sure the header file can be compiled on its own.
#include "torch_tpu/common/macro_utils.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_split.h"

namespace torch_tpu {
namespace internal {

std::vector<std::string_view> ArgsAsStrings(const std::string_view csv_string) {
  std::vector<std::string_view> split_args =
      absl::StrSplit(csv_string, absl::ByChar(','), absl::SkipWhitespace());
  for (auto& arg : split_args) {
    arg = absl::StripAsciiWhitespace(arg);
  }
  return split_args;
}

const char* NormalizeIfInsideBuildDirectory(const char* path) {
  // Wrap `path` in a `string_view` for easier path manipulation.
  const std::string_view sv_path = path;

  constexpr std::string_view kBlazeOutDir = "blaze-out/";
  constexpr std::string_view kBazelOutDir = "bazel-out/";
  constexpr size_t kBuildDirectoryLen = kBlazeOutDir.length();

  // No normalization if:
  if (
      // 1. `path` can't be inside the build directory, since it's shorter than
      //    the build directory name.
      sv_path.length() <= kBuildDirectoryLen ||
      // 2. `path` doesn't start with neither of the 2 possible build directory
      //    names: 'blaze-out/' and 'bazel-out/'.
      (sv_path.substr(0, kBuildDirectoryLen) != kBlazeOutDir &&
       sv_path.substr(0, kBuildDirectoryLen) != kBazelOutDir)) {
    return path;
  }

  // Return the relative path starting from `dir`, if it exists in `path`.
  //
  // More specifically, it returns the right-most substring of `path` that
  // starts with `dir`.
  auto get_relative_path_starting_at =
      [sv_path](const std::string_view dir) -> std::optional<std::string_view> {
    const auto pos = sv_path.rfind(dir);
    return (pos == std::string::npos)
               ? std::nullopt
               : std::make_optional(sv_path.substr(pos + 1));
  };

  constexpr std::string_view kThirdPartyDir = "/third_party/py/torch_tpu/";
  constexpr std::string_view kTorchTpuDir = "/torch_tpu/";

  if (auto r = get_relative_path_starting_at(kThirdPartyDir); r.has_value()) {
    return r.value().data();
  }
  if (auto r = get_relative_path_starting_at(kTorchTpuDir); r.has_value()) {
    return r.value().data();
  }

  return path;
}

}  // namespace internal
}  // namespace torch_tpu
