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

#include "torch_tpu/common/excess_precision.h"

#include <atomic>

#include "absl/base/no_destructor.h"
#include "xla/client/executable_build_options.h"

namespace torch_tpu {
namespace {

enum class ExcessPrecisionState {
  // Use XLA_FLAGS or fall back to the default.
  kUnset,
  // Explicitly allow excess precision.
  kAllow,
  // Explicitly disallow excess precision.
  kDisallow,
};

static std::atomic<ExcessPrecisionState> g_allow_excess_precision{
    ExcessPrecisionState::kUnset};

}  // namespace

const xla::ExecutableBuildOptions& GetDefaultExecutableBuildOptions() {
  static const absl::NoDestructor<xla::ExecutableBuildOptions>
      default_executable_build_options([] {
        xla::ExecutableBuildOptions options;
        // Calling mutable_debug_options() triggers parsing XLA_FLAGS.
        options.mutable_debug_options();
        return options;
      }());
  return *default_executable_build_options;
}

void SetAllowExcessPrecision(bool allow) {
  g_allow_excess_precision.store(allow ? ExcessPrecisionState::kAllow
                                       : ExcessPrecisionState::kDisallow);
}

bool GetAllowExcessPrecision() {
  switch (g_allow_excess_precision.load()) {
    case ExcessPrecisionState::kAllow:
      return true;
    case ExcessPrecisionState::kDisallow:
      return false;
    case ExcessPrecisionState::kUnset:
      // Fall back to XLA_FLAGS or default to true.
      // Use static to memoize the value to avoid reading the debug options
      // multiple times.
      static const bool allow_excess_precision = [] {
        const xla::DebugOptions& debug_options =
            GetDefaultExecutableBuildOptions().debug_options();
        if (debug_options.has_xla_allow_excess_precision()) {
          return debug_options.xla_allow_excess_precision();
        }
        // TODO: b/502610173 - Set to False when XLA_FLAGS is not set.
        return true;
      }();
      return allow_excess_precision;
  }
}

}  // namespace torch_tpu
