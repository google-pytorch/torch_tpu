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

#include "csrc/common/excess_precision.h"

#include <atomic>
#include <string>

#include "absl/base/no_destructor.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"
#include "xla/client/executable_build_options.h"

namespace torch_tpu {
namespace {

enum class ExcessPrecisionState {
  // Use XLA_FLAGS or fall back to the default (unlocked).
  kUnset,
  // Explicitly allow excess precision (unlocked).
  kAllow,
  // Explicitly disallow excess precision (unlocked).
  kDisallow,
  // Locked to allow excess precision.
  kLockedAllow,
  // Locked to disallow excess precision.
  kLockedDisallow,
};

static std::atomic<ExcessPrecisionState> g_allow_excess_precision{
    ExcessPrecisionState::kUnset};

bool IsLocked(ExcessPrecisionState state) {
  return state == ExcessPrecisionState::kLockedAllow ||
         state == ExcessPrecisionState::kLockedDisallow;
}

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

// Locks excess precision. If it was already set to allow or disallow, it
// transitions to kLockedAllow or kLockedDisallow respectively. If it was unset,
// it resolves to the default setting and locks it. Uses compare_exchange_strong
// to ensure thread safety.
void LockAllowExcessPrecision() {
  // Relaxed is safe because if this initial read is stale,
  // compare_exchange_strong will return false, update `current`, and the loop
  // will retry or exit.
  ExcessPrecisionState current =
      g_allow_excess_precision.load(std::memory_order_relaxed);
  while (!IsLocked(current)) {
    ExcessPrecisionState locked_state;
    switch (current) {
      case ExcessPrecisionState::kAllow:
        locked_state = ExcessPrecisionState::kLockedAllow;
        break;
      case ExcessPrecisionState::kDisallow:
        locked_state = ExcessPrecisionState::kLockedDisallow;
        break;
      case ExcessPrecisionState::kUnset:
        [[fallthrough]];
      default:
        locked_state = GetAllowExcessPrecision()
                           ? ExcessPrecisionState::kLockedAllow
                           : ExcessPrecisionState::kLockedDisallow;
        break;
    }

    // On success: acquire synchronizes with prior SetAllowExcessPrecision()
    // updates; release publishes the new state to GetAllowExcessPrecision() and
    // IsAllowExcessPrecisionLocked().
    // On failure: we just reload `current` and try again; relaxed matches the
    // initial load above.
    if (g_allow_excess_precision.compare_exchange_strong(
            current, locked_state, std::memory_order_acq_rel,
            std::memory_order_relaxed)) {
      break;
    }
  }
}

// Returns whether excess precision is locked (i.e. whether compilation has
// already started).
bool IsAllowExcessPrecisionLocked() {
  // memory_order_acquire synchronizes with the release/acq_rel CAS in
  // LockAllowExcessPrecision().
  return IsLocked(g_allow_excess_precision.load(std::memory_order_acquire));
}

// Sets the excess precision setting. If the setting has already been locked
// by an ongoing or completed XLA compilation, throws an error unless the
// value matches the current setting. Atomically checks the lock bit and writes
// the new setting using compare_exchange_strong.
void SetAllowExcessPrecision(bool allow) {
  // Relaxed is safe because if this initial read is stale,
  // compare_exchange_strong will return false, update `current`, and the loop
  // will retry or exit.
  ExcessPrecisionState current =
      g_allow_excess_precision.load(std::memory_order_relaxed);
  const ExcessPrecisionState desired =
      allow ? ExcessPrecisionState::kAllow : ExcessPrecisionState::kDisallow;
  while (true) {
    if (IsLocked(current)) {
      TT_CHECK_THROW(allow == GetAllowExcessPrecision(),
                     error::kFailedPrecondition)
          << "cannot change allow_excess_precision after XLA compilation has "
             "already started";
      return;
    }
    // On success: acquire synchronizes with prior SetAllowExcessPrecision()
    // updates; release publishes the new state to GetAllowExcessPrecision() and
    // IsAllowExcessPrecisionLocked().
    // On failure: we just reload `current` and try again; relaxed matches the
    // initial load above.
    if (g_allow_excess_precision.compare_exchange_strong(
            current, desired, std::memory_order_acq_rel,
            std::memory_order_relaxed)) {
      return;
    }
  }
}

bool GetAllowExcessPrecision() {
  // memory_order_acquire synchronizes with the release/acq_rel writes in
  // SetAllowExcessPrecision() and LockAllowExcessPrecision(), ensuring the
  // calling thread observes all state initialized prior to setting or locking
  // the precision state.
  switch (g_allow_excess_precision.load(std::memory_order_acquire)) {
    case ExcessPrecisionState::kAllow:
    case ExcessPrecisionState::kLockedAllow:
      return true;
    case ExcessPrecisionState::kDisallow:
    case ExcessPrecisionState::kLockedDisallow:
      return false;
    case ExcessPrecisionState::kUnset:
      // Fall back to XLA_FLAGS or default to false.
      // Use static to memoize the value to avoid reading the debug options
      // multiple times.
      static const bool allow_excess_precision = [] {
        // We manually check if the flag is present in XLA_FLAGS string because
        // xla::DebugOptions always provides a value for
        // xla_allow_excess_precision (defaulting to true), making it impossible
        // to distinguish between a user-set value and XLA's internal default.
        const auto& xla_flags = ExpandAndGetXlaFlagsEnvVarOnce();
        if (xla_flags.has_value() &&
            xla_flags->find("xla_allow_excess_precision") !=
                std::string::npos) {
          const xla::DebugOptions& debug_options =
              GetDefaultExecutableBuildOptions().debug_options();
          return debug_options.xla_allow_excess_precision();
        }
        return false;
      }();
      return allow_excess_precision;
  }
}

}  // namespace torch_tpu
