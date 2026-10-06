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

#include "csrc/common/libtpu_version.h"

#include <optional>
#include <string>
#include <string_view>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/absl_log.h"
#include "absl/synchronization/mutex.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/version.h"

namespace torch_tpu {
namespace {

struct LibtpuVersionState {
  absl::Mutex mutex;
  std::optional<std::string> version ABSL_GUARDED_BY(mutex);
  Version parsed_version ABSL_GUARDED_BY(mutex) = Version::Max();
};

LibtpuVersionState& GetState() {
  static absl::NoDestructor<LibtpuVersionState> state;
  return *state;
}

}  // namespace

void SetLibtpuVersion(std::string_view version) {
  LibtpuVersionState& state = GetState();
  absl::MutexLock lock(state.mutex);
  if (state.version.has_value()) {
    TT_CHECK_THROW(*state.version == version, error::kInvalidArgument)
        << "expected libtpu version to match previously set version '"
        << *state.version << "', got '" << version << "'";
    return;
  }
  if (version.empty()) {
    state.version = std::string(version);
    ABSL_LOG(INFO) << "libtpu not installed.";
    return;
  }
  TT_ASSIGN_OR_THROW(const Version parsed_version, Version::Parse(version));
  state.version = std::string(version);
  state.parsed_version = parsed_version;
  ABSL_LOG(INFO) << "Setting libtpu version to: " << *state.version;
}

std::optional<std::string> GetLibtpuVersionString() {
  LibtpuVersionState& state = GetState();
  absl::MutexLock lock(state.mutex);
  return state.version;
}

Version GetLibtpuVersion() {
  LibtpuVersionState& state = GetState();
  absl::MutexLock lock(state.mutex);
  return state.parsed_version;
}

void ResetLibtpuVersionForTesting() {
  LibtpuVersionState& state = GetState();
  absl::MutexLock lock(state.mutex);
  state.version.reset();
  state.parsed_version = Version::Max();
}

}  // namespace torch_tpu
