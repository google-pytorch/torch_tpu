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

#include "csrc/common/process.h"

#include <sys/types.h>
#include <unistd.h>

#include <cstdlib>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "csrc/common/env_vars.h"

namespace torch_tpu {
namespace {

// Returns the PID of the earliest TorchTPU process in the current process'
// lineage, which is the current process itself if it has no TorchTPU ancestor
// process.
//
// The earliest TorchTPU process records its PID in
// `TORCH_TPU_INTERNAL_TOPMOST_PID` on the first call, and its descendant
// processes inherit it through their environment. So a process has a TorchTPU
// ancestor process iff it finds this env var already set to a PID other than
// its own. Launchers that don't load TorchTPU never set the env var, and
// descendants launched without the inherited environment never see it.
pid_t GetTopmostTorchTpuProcessPid() {
  static const pid_t topmost_pid = []() -> pid_t {
    const char* const env_str =
        std::getenv(  // GETENV_OK=Internal implementation env var.
            kTorchTpuInternalTopmostPidEnvVar);
    pid_t env_pid = 0;
    if (env_str != nullptr && absl::SimpleAtoi(env_str, &env_pid) &&
        env_pid > 0) {
      return env_pid;
    }
    const pid_t self_pid = getpid();
    SetEnv<kTorchTpuInternalTopmostPidEnvVar>(absl::StrCat(self_pid));
    return self_pid;
  }();
  return topmost_pid;
}

}  // namespace

bool HasTorchTpuAncestorProcess() {
  return getpid() != GetTopmostTorchTpuProcessPid();
}

}  // namespace torch_tpu
