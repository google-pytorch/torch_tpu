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

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "csrc/common/error_utils.h"
#include "re2/re2.h"

namespace torch_tpu {
namespace {

// Parses `digits` into `*value`, leaving `*value` unchanged if `digits` is
// empty (i.e. the optional regex group didn't participate in the match).
template <typename T>
absl::Status ParseOptionalComponent(const std::string_view digits,
                                    const std::string_view version, T* value) {
  if (digits.empty()) {
    return absl::OkStatus();
  }
  TT_RET_CHECK(absl::SimpleAtoi(digits, value), error::kInvalidArgument)
      << "version component '" << digits << "' overflows in '" << version
      << "'";
  return absl::OkStatus();
}

absl::Status ParseReleaseComponents(const std::string_view release_str,
                                    const std::string_view version,
                                    const std::array<int*, 8>& targets) {
  size_t idx = 0;
  for (const std::string_view part : absl::StrSplit(release_str, '.')) {
    if (idx >= targets.size()) {
      break;
    }
    TT_RETURN_IF_ERROR(ParseOptionalComponent(part, version, targets[idx++]));
  }
  return absl::OkStatus();
}

}  // namespace

void Version::DieOnNegativeComponent(const int major, const int minor,
                                     const int patch, const int subpatch,
                                     const int64_t dev) {
  // Versions are built from constants or parsed digits, so a negative
  // component is a torch_tpu bug rather than a user error.
  ABSL_LOG(FATAL)  // CRASH_OK
      << "expected version components to be non-negative, got major=" << major
      << ", minor=" << minor << ", patch=" << patch << ", subpatch=" << subpatch
      << ", dev=" << dev;
}

absl::StatusOr<Version> Version::Parse(const std::string_view version) {
  // Match the full canonical PEP 440 grammar in C++ rather than an ad-hoc
  // subset: wheel versions may use any PEP 440 segment (4+ release components,
  // `a`/`b`/`rc` pre-releases, `.post`, `.dev`, `+local`), and parsing in C++
  // avoids a runtime `packaging` dependency and Python/GIL coupling.
  static const LazyRE2 kPep440Pattern = {
      R"((?i)^[[:space:]]*v?)"
      R"((?:([0-9]+)!)?)"
      R"(([0-9]+(?:\.[0-9]+){0,7}(?:\.0+)*))"
      R"((?:[-_.]?(alpha|a|beta|b|preview|pre|rc|c)[-_.]?([0-9]+)?)?)"
      R"((?:(?:-([0-9]+))|(?:[-_.]?(post|rev|r)[-_.]?([0-9]+)?))?)"
      R"((?:[-_.]?(dev)[-_.]?([0-9]+)?)?)"
      R"((?:\+[a-z0-9]+(?:[-_.][a-z0-9]+)*)?[[:space:]]*$)"};
  std::string_view epoch_str;
  std::string_view release_str;
  std::string_view pre_label;
  std::string_view pre_num_str;
  std::string_view implicit_post_str;
  std::string_view post_label;
  std::string_view post_num_str;
  std::string_view dev_label;
  std::string_view dev_num_str;
  TT_RET_CHECK(
      RE2::FullMatch(version, *kPep440Pattern, &epoch_str, &release_str,
                     &pre_label, &pre_num_str, &implicit_post_str, &post_label,
                     &post_num_str, &dev_label, &dev_num_str),
      error::kInvalidArgument)
      << "expected a valid PEP 440 version string, got '" << version << "'";

  Version result(0, 0, 0);
  TT_RETURN_IF_ERROR(
      ParseOptionalComponent(epoch_str, version, &result.epoch_));
  TT_RETURN_IF_ERROR(ParseReleaseComponents(
      release_str, version,
      {&result.major_, &result.minor_, &result.patch_, &result.subpatch_,
       &result.extra_release_[0], &result.extra_release_[1],
       &result.extra_release_[2], &result.extra_release_[3]}));

  if (!pre_label.empty()) {
    const char first =
        absl::ascii_tolower(static_cast<unsigned char>(pre_label.front()));
    if (first == 'a') {
      result.pre_stage_ = PreStage::kAlpha;
    } else if (first == 'b') {
      result.pre_stage_ = PreStage::kBeta;
    } else {
      // 'c', 'rc', 'pre', and 'preview' all normalize to 'rc' in PEP 440.
      result.pre_stage_ = PreStage::kRc;
    }
    TT_RETURN_IF_ERROR(
        ParseOptionalComponent(pre_num_str, version, &result.pre_num_));
  }

  if (!implicit_post_str.empty()) {
    result.has_post_ = true;
    TT_RETURN_IF_ERROR(
        ParseOptionalComponent(implicit_post_str, version, &result.post_));
  } else if (!post_label.empty()) {
    result.has_post_ = true;
    TT_RETURN_IF_ERROR(
        ParseOptionalComponent(post_num_str, version, &result.post_));
  }

  if (!dev_label.empty()) {
    result.has_no_dev_ = false;
    if (pre_label.empty() && !result.has_post_) {
      result.pre_stage_ = PreStage::kPureDev;
    }
    TT_RETURN_IF_ERROR(
        ParseOptionalComponent(dev_num_str, version, &result.dev_));
  }

  return result;
}

}  // namespace torch_tpu
