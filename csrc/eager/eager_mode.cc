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

#include "csrc/eager/eager_mode.h"

#include <atomic>

#include "absl/log/absl_log.h"
#include "csrc/common/context_manager.h"
#include "csrc/common/context_states.h"
#include "csrc/common/env_vars.h"

namespace torch_tpu {

[[nodiscard]] static EagerMode GetDefaultEagerMode() {
  const auto& launch_blocking_public =
      GetEnvOnce<kTorchTpuLaunchBlockingEnvVar>();
  const auto& launch_blocking = launch_blocking_public.has_value()
                                    ? launch_blocking_public
                                    : GetEnvOnce<kTpuLaunchBlocking>();
  if (launch_blocking == "1") {
    return EagerMode::kDeferNeverAndLaunchBlocking;
  }
  const auto& defer_and_fuse_public = GetEnvOnce<kTorchTpuDeferAndFuseEnvVar>();
  const auto& defer_and_fuse = defer_and_fuse_public.has_value()
                                   ? defer_and_fuse_public
                                   : GetEnvOnce<kTpuDeferAndFuse>();
  if (defer_and_fuse == "1") {
    return EagerMode::kDeferAndFuse;
  }
  return EagerMode::kDeferNever;
}

// Returns the global base eager mode.
static std::atomic<EagerMode>& GetMutableGlobalEagerMode() {
  static std::atomic<EagerMode> eager_mode = GetDefaultEagerMode();
  return eager_mode;
}

EagerMode GetEagerMode() {
  return GetContextState<EagerModeContextState>(
      [] { return GetMutableGlobalEagerMode().load(); });
}

void SetEagerMode(const EagerMode mode) {
  ABSL_VLOG(1) << "SetEagerMode " << static_cast<int>(mode);
  GetMutableGlobalEagerMode() = mode;
}

[[nodiscard]] static bool GetDefaultInplaceBufferDonation() {
  const auto& env_var =
      GetEnvOnce<kTorchTpuInternalDisableInplaceBufferDonationEnvVar>();
  if (env_var.has_value() && (*env_var == "1" || *env_var == "true")) {
    return false;
  }
  return true;
}

// Returns the global base in-place buffer donation setting.
static std::atomic<bool>& GetMutableGlobalInplaceBufferDonation() {
  static std::atomic<bool> donation_enabled = GetDefaultInplaceBufferDonation();
  return donation_enabled;
}

bool IsInplaceBufferDonationEnabled() {
  return GetMutableGlobalInplaceBufferDonation().load();
}

void EnableInplaceBufferDonation(const bool enabled) {
  ABSL_VLOG(1) << "EnableInplaceBufferDonation " << enabled;
  GetMutableGlobalInplaceBufferDonation().store(enabled);
}

}  // namespace torch_tpu
