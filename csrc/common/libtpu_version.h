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

#ifndef TORCH_TPU_CSRC_COMMON_LIBTPU_VERSION_H_
#define TORCH_TPU_CSRC_COMMON_LIBTPU_VERSION_H_

#include <optional>
#include <string>
#include <string_view>

#include "csrc/common/version.h"

namespace torch_tpu {

// Sets the libtpu version string.
// Calling SetLibtpuVersion with a different version after it has already been
// set will trigger a check failure. Calling it with the exact same version is
// a no-op.
void SetLibtpuVersion(std::string_view version);

// Returns the libtpu version string set via SetLibtpuVersion, or std::nullopt
// if never set. If libtpu was not found, the returned optional contains an
// empty string.
[[nodiscard]] std::optional<std::string> GetLibtpuVersionString();

// Returns the libtpu version set via SetLibtpuVersion, for gating features on
// it, e.g. `if (GetLibtpuVersion() >= Version(0, 0, 48)) { ... }`.
//
// If no libtpu version has been set (e.g. in-tree builds) or the version is
// empty, returns `Version::Max()`, so every feature is treated as available.
//
// Stable versions are given higher precedence than nightly/pre-release builds
// of the same release version (see `Version`). For example:
// - Stable "0.0.47" satisfies requirements for 0.0.47, 0.0.46, and nightly
//   versions like 0.0.47.dev20260824.
// - A nightly build "0.0.47.dev20260824+nightly" satisfies requirements for
//   0.0.46 or older nightlies like 0.0.47.dev20260823, but does NOT satisfy a
//   requirement for the stable release 0.0.47 or newer nightlies like
//   0.0.47.dev20260825.
[[nodiscard]] Version GetLibtpuVersion();

// Resets the libtpu version state to uninitialized (std::nullopt).
// For test isolation only.
void ResetLibtpuVersionForTesting();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_LIBTPU_VERSION_H_
