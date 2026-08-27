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

#include "torch_tpu/csrc/distributed/handshake.h"

#include "absl/log/absl_log.h"
#include "absl/strings/numbers.h"
#include "torch_tpu/csrc/common/env_vars.h"

namespace torch_tpu {

HandshakeStage GetHandshakeStageEnvVarOnce() {
  static const HandshakeStage env_value = []() {
    const auto& raw_env_value =
        GetEnvOnce<kTorchTpuInternalHandshakeStageEnvVar>();
    if (!raw_env_value.has_value() || *raw_env_value == "OFF") {
      return HandshakeStage::kOff;
    }
    if (*raw_env_value == "COMPILE_STAGE") {
      return HandshakeStage::kCompileStage;
    }
    if (*raw_env_value == "DISPATCH_STAGE") {
      return HandshakeStage::kDispatchStage;
    }
    ABSL_LOG(FATAL) << "Invalid value for "  // CRASH_OK
                    << kTorchTpuInternalHandshakeStageEnvVar << ": "
                    << *raw_env_value
                    << ". Expected OFF, COMPILE_STAGE, or DISPATCH_STAGE.";
  }();
  return env_value;
}

int GetHandshakePortEnvVarOnce() {
  static const int env_value = []() {
    constexpr int kDefaultHandshakePort = 36423;
    const auto& raw_env_value = GetEnvOnce<kTorchTpuHandshakePortEnvVar>();
    if (!raw_env_value.has_value()) {
      return kDefaultHandshakePort;
    }
    int port;
    if (absl::SimpleAtoi(*raw_env_value, &port)) {
      return port;
    }
    ABSL_LOG(FATAL) << "Invalid value for "  // CRASH_OK
                    << kTorchTpuHandshakePortEnvVar << ": " << *raw_env_value
                    << ". Expected an integer.";
  }();
  return env_value;
}

}  // namespace torch_tpu
