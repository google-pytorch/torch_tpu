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

#include "torch_tpu/csrc/eager/eager_mode.h"

#include <atomic>

#include "absl/log/absl_log.h"
#include "torch_tpu/csrc/common/context_manager.h"
#include "torch_tpu/csrc/common/context_states.h"
#include "torch_tpu/csrc/common/env_vars.h"

namespace torch_tpu {

[[nodiscard]] static EagerMode GetDefaultEagerMode() {
  if (GetEnvOnce<kTpuLaunchBlocking>() == "1") {
    return EagerMode::kDeferNeverAndLaunchBlocking;
  } else if (GetEnvOnce<kTpuDeferAndFuse>() == "1") {
    return EagerMode::kDeferAndFuse;
  } else {
    return EagerMode::kDeferNever;
  }
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

}  // namespace torch_tpu
