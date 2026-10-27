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

#include <compare>
#include <cstdint>
#include <limits>

#include "absl/strings/str_format.h"

namespace torch_tpu {

// A release version of the form
// `<major>.<minor>.<patch>[.<subpatch>][.dev<build>]`, e.g. `2.3.15`,
// `0.0.50.1`, or `0.0.47.dev20260824`.
//
// Versions are ordered component-by-component. A stable release ranks above
// all pre-release (dev) builds of the same
// `<major>.<minor>.<patch>.<subpatch>`. For example:
//   Version(0, 0, 46) < Version(0, 0, 47, 0, 20260824)
//                     < Version(0, 0, 47, 0, 20260825)
//                     < Version(0, 0, 47)
//                     < Version(0, 0, 47, 1)
class Version {
 public:
  // The `dev` value of a stable release.
  static constexpr int64_t kStable = std::numeric_limits<int64_t>::max();

  // Returns a version that is greater than or equal to every other version.
  [[nodiscard]] static constexpr Version Max() {
    constexpr int kMax = std::numeric_limits<int>::max();
    return Version(kMax, kMax, kMax, kMax, kStable);
  }

  // `subpatch` is the fourth numeric release component (e.g. `1` in
  // `0.0.50.1`); omit it (or pass 0) for a three-component release. `dev` is
  // the build number of a pre-release; omit it for a stable release.
  // All components must be non-negative: a negative component is a compile
  // error in a constant expression (e.g. a `constexpr Version`) and crashes
  // otherwise.
  constexpr Version(int major, int minor, int patch, int subpatch = 0,
                    int64_t dev = kStable)
      : major_(major),
        minor_(minor),
        patch_(patch),
        subpatch_(subpatch),
        dev_(dev) {
    if (major < 0 || minor < 0 || patch < 0 || subpatch < 0 || dev < 0) {
      DieOnNegativeComponent(major, minor, patch, subpatch, dev);
    }
  }

  friend constexpr auto operator<=>(const Version&, const Version&) = default;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Version& version) {
    absl::Format(&sink, "%d.%d.%d", version.major_, version.minor_,
                 version.patch_);
    if (version.subpatch_ != 0) {
      absl::Format(&sink, ".%d", version.subpatch_);
    }
    if (version.dev_ != kStable) {
      absl::Format(&sink, ".dev%d", version.dev_);
    }
  }

 private:
  // Crashes. Deliberately not `constexpr`, so that reaching it during constant
  // evaluation is a compile error.
  [[noreturn]] static void DieOnNegativeComponent(int major, int minor,
                                                  int patch, int subpatch,
                                                  int64_t dev);

  // The declaration order determines the ordering of `operator<=>`.
  int major_ = 0;
  int minor_ = 0;
  int patch_ = 0;
  int subpatch_ = 0;
  // `int64_t` because timestamp-based dev builds use 12 digits (YYYYMMDDHHMM),
  // which overflow `int`.
  int64_t dev_ = 0;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_VERSION_H_
