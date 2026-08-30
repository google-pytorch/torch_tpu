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

#include "csrc/distributed/handshake.h"

#include "csrc/common/env_vars.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

TEST(HandshakeDeathTest, FatalOnInvalidHandshakeStage) {
  EXPECT_DEATH(
      {
        setenv(kTorchTpuInternalHandshakeStageEnvVar, "INVALID_VALUE", 1);
        GetHandshakeStageEnvVarOnce();
      },
      "Invalid value for TORCH_TPU_INTERNAL_HANDSHAKE_STAGE: INVALID_VALUE");
}

}  // namespace
}  // namespace torch_tpu
