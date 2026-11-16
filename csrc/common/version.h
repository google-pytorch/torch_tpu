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

#ifndef TORCH_TPU_CSRC_COMMON_VERSION_H_
#define TORCH_TPU_CSRC_COMMON_VERSION_H_

#include <array>
#include <compare>  // IWYU pragma: keep
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"

namespace torch_tpu {

// A PEP 440 version, e.g. `2.3.15`, `0.0.50.1`, `0.0.48rc1`, or
// `0.0.47.dev20260824`.
//
// Versions are ordered per PEP 440: epoch, release components (padded with
// trailing zeros), pre-release (`a` < `b` < `rc`), post-release (`post`), and
// dev-release (`dev`). Local version labels (`+<local>`, e.g. `+nightly`) are
// validated by `Parse` and ignored for comparison. For example:
//   Version(0, 0, 46) < Version(0, 0, 47, Version::Dev(20260824))
//                     < Version(0, 0, 47, Version::Dev(20260825))
//                     < *Version::Parse("0.0.47a1")
//                     < *Version::Parse("0.0.47rc1")
//                     < Version(0, 0, 47)
//                     < *Version::Parse("0.0.47.post1")
//                     < Version(0, 0, 47, 1, Version::Dev(20260824))
//                     < Version(0, 0, 47, 1)
class Version {
 public:
  // Tag type distinguishing a `.dev<build>` pre-release build number from an
  // integer `subpatch` release component in `Version` constructors.
  struct Dev {
    explicit constexpr Dev(int64_t build) : value(build) {}
    int64_t value;
  };

  // Returns a version that is greater than or equal to every other version.
  [[nodiscard]] static constexpr Version Max() {
    constexpr int kMaxInt = std::numeric_limits<int>::max();
    constexpr int64_t kMaxInt64 = std::numeric_limits<int64_t>::max();
    Version v(0, 0, 0);
    v.epoch_ = kMaxInt;
    v.major_ = kMaxInt;
    v.minor_ = kMaxInt;
    v.patch_ = kMaxInt;
    v.subpatch_ = kMaxInt;
    v.extra_release_.fill(kMaxInt);
    v.pre_stage_ = PreStage::kFinal;
    v.has_post_ = true;
    v.post_ = kMaxInt64;
    v.has_no_dev_ = true;
    return v;
  }

  // Parses a PEP 440 version string
  // (`[v][<epoch>!]<release>[<pre>][<post>][<dev>][+<local>]`), e.g. "2.3",
  // "2.3.15", "0.0.50.1", "0.0.48rc1", or "0.0.47.dev20260824+nightly".
  // Missing release components default to 0, and local version labels
  // (`+<local>`) are ignored after validation. Returns an error if `version` is
  // not a valid PEP 440 version string or a numeric component overflows.
  static absl::StatusOr<Version> Parse(std::string_view version);

  // Constructs a stable release `<major>.<minor>.<patch>[.<subpatch>]`.
  // All components must be non-negative: a negative component is a compile
  // error in a constant expression (e.g. a `constexpr Version`) and crashes
  // otherwise.
  constexpr Version(int major, int minor, int patch, int subpatch = 0)
      : major_(major), minor_(minor), patch_(patch), subpatch_(subpatch) {
    if (major < 0 || minor < 0 || patch < 0 || subpatch < 0) {
      DieOnNegativeComponent(major, minor, patch, subpatch, /*dev=*/0);
    }
  }

  // Constructs a dev pre-release `<major>.<minor>.<patch>.dev<build>`.
  constexpr Version(int major, int minor, int patch, Dev dev)
      : Version(major, minor, patch, /*subpatch=*/0, dev) {}

  // Constructs a dev pre-release
  // `<major>.<minor>.<patch>.<subpatch>.dev<build>`.
  constexpr Version(int major, int minor, int patch, int subpatch, Dev dev)
      : major_(major),
        minor_(minor),
        patch_(patch),
        subpatch_(subpatch),
        pre_stage_(PreStage::kPureDev),
        has_no_dev_(false),
        dev_(dev.value) {
    if (major < 0 || minor < 0 || patch < 0 || subpatch < 0 || dev.value < 0) {
      DieOnNegativeComponent(major, minor, patch, subpatch, dev.value);
    }
  }

  friend constexpr auto operator<=>(const Version&, const Version&) = default;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Version& version) {
    if (version.epoch_ > 0) {
      absl::Format(&sink, "%d!", version.epoch_);
    }
    absl::Format(&sink, "%d.%d.%d", version.major_, version.minor_,
                 version.patch_);
    size_t extra_len = version.extra_release_.size();
    while (extra_len > 0 && version.extra_release_[extra_len - 1] == 0) {
      --extra_len;
    }
    if (version.subpatch_ > 0 || extra_len > 0) {
      absl::Format(&sink, ".%d", version.subpatch_);
    }
    for (size_t i = 0; i < extra_len; ++i) {
      absl::Format(&sink, ".%d", version.extra_release_[i]);
    }
    switch (version.pre_stage_) {
      case PreStage::kAlpha:
        absl::Format(&sink, "a%d", version.pre_num_);
        break;
      case PreStage::kBeta:
        absl::Format(&sink, "b%d", version.pre_num_);
        break;
      case PreStage::kRc:
        absl::Format(&sink, "rc%d", version.pre_num_);
        break;
      case PreStage::kPureDev:
      case PreStage::kFinal:
        break;
    }
    if (version.has_post_) {
      absl::Format(&sink, ".post%d", version.post_);
    }
    if (!version.has_no_dev_) {
      absl::Format(&sink, ".dev%d", version.dev_);
    }
  }

 private:
  // Ordered to match PEP 440's comparison key: pure `.devN` (no pre- or
  // post-release) sorts before alpha (`kPureDev`), and final/post-releases
  // (no pre-release) sort after rc (`kFinal`).
  enum class PreStage : uint8_t {
    kPureDev = 0,
    kAlpha = 1,
    kBeta = 2,
    kRc = 3,
    kFinal = 4,
  };

  // Crashes. Deliberately not `constexpr`, so that reaching it during constant
  // evaluation is a compile error.
  [[noreturn]] static void DieOnNegativeComponent(int major, int minor,
                                                  int patch, int subpatch,
                                                  int64_t dev);

  // The declaration order determines the ordering of `operator<=>`.
  int epoch_ = 0;
  int major_ = 0;
  int minor_ = 0;
  int patch_ = 0;
  int subpatch_ = 0;
  // 5th through 8th release components (zero-padded per PEP 440). Using
  // `std::array` rather than `std::vector` keeps `Version` a literal type under
  // OSS `libstdc++` 10 (which lacks `constexpr std::vector`).
  std::array<int, 4> extra_release_ = {};
  PreStage pre_stage_ = PreStage::kFinal;
  int64_t pre_num_ = 0;
  bool has_post_ = false;
  int64_t post_ = 0;
  bool has_no_dev_ = true;
  // `int64_t` because timestamp-based dev builds use 12 digits (YYYYMMDDHHMM),
  // which overflow `int`.
  int64_t dev_ = 0;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_VERSION_H_
