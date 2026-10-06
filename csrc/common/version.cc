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

#include "csrc/common/version.h"

#include <cstdint>
#include <string_view>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "csrc/common/error_utils.h"
#include "re2/re2.h"

namespace torch_tpu {
namespace {

// Parses `digits` into `*value`, leaving `*value` unchanged if `digits` is
// empty (i.e. the optional regex group didn't participate in the match).
template <typename T>
absl::Status ParseOptionalComponent(std::string_view digits,
                                    std::string_view version, T* value) {
  if (digits.empty()) {
    return absl::OkStatus();
  }
  TT_RET_CHECK(absl::SimpleAtoi(digits, value), error::kInvalidArgument)
      << "version component '" << digits << "' overflows in '" << version
      << "'";
  return absl::OkStatus();
}

}  // namespace

void Version::DieOnNegativeComponent(int major, int minor, int patch,
                                     int64_t dev) {
  // Versions are built from constants or parsed digits, so a negative
  // component is a torch_tpu bug rather than a user error.
  ABSL_LOG(FATAL)  // CRASH_OK
      << "expected version components to be non-negative, got major=" << major
      << ", minor=" << minor << ", patch=" << patch << ", dev=" << dev;
}

absl::StatusOr<Version> Version::Parse(std::string_view version) {
  static const LazyRE2 kPattern = {
      R"((\d+)(?:\.(\d+))?(?:\.(\d+))?(?:\.dev(\d+))?(?:[^\d.].*)?)"};
  std::string_view major_str, minor_str, patch_str, dev_str;
  TT_RET_CHECK(RE2::FullMatch(version, *kPattern, &major_str, &minor_str,
                              &patch_str, &dev_str),
               error::kInvalidArgument)
      << "expected version string to be of the format "
         "<major>[.<minor>[.<patch>]][.dev<build>][<suffix>], got '"
      << version << "'";
  Version result(0, 0, 0, kStable);
  TT_RETURN_IF_ERROR(
      ParseOptionalComponent(major_str, version, &result.major_));
  TT_RETURN_IF_ERROR(
      ParseOptionalComponent(minor_str, version, &result.minor_));
  TT_RETURN_IF_ERROR(
      ParseOptionalComponent(patch_str, version, &result.patch_));
  TT_RETURN_IF_ERROR(ParseOptionalComponent(dev_str, version, &result.dev_));
  return result;
}

}  // namespace torch_tpu
