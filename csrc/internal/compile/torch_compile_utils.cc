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

#include "csrc/internal/compile/torch_compile_utils.h"

#include "absl/strings/match.h"
#include "csrc/common/env_vars.h"

namespace torch_tpu {

bool PyGetMaterializeCollectiveTensorsEnvVarOnce() {
  static const bool env_value = []() {
    const auto& raw_env_value =
        GetEnvOnce<kTorchTpuInternalMaterializeCollectiveTensorsEnvVar>();
    return !raw_env_value.has_value() ||
           (*raw_env_value != "0" &&
            !absl::EqualsIgnoreCase(*raw_env_value, "false"));
  }();
  return env_value;
}

}  // namespace torch_tpu
