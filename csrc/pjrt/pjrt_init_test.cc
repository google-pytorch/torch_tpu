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

#include <cstdlib>
#include <string>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "csrc/common/env_vars.h"
#include "csrc/pjrt/pjrt_state.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

class PjRtInitTest : public testing::Test {
 protected:
  void TearDown() override { PjrtBackend::GetInstance().Shutdown(); }
};

TEST_F(PjRtInitTest, InitializePjRtIsLazy) {
  ASSERT_FALSE(PjrtBackend::GetInstance().IsInitialized());

  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "tpu"});
  EXPECT_FALSE(PjrtBackend::GetInstance().IsInitialized());

  ASSERT_NE(PjrtBackend::GetInstance().GetClient(), nullptr);
  EXPECT_TRUE(PjrtBackend::GetInstance().IsInitialized());
}

TEST_F(PjRtInitTest, InitializePjRtSetsCorrectDeviceCount) {
  ASSERT_FALSE(PjrtBackend::GetInstance().IsInitialized());

  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "tpu"});

  auto* client = PjrtBackend::GetInstance().GetClient();
  ASSERT_NE(client, nullptr);
  EXPECT_TRUE(PjrtBackend::GetInstance().IsInitialized());

  auto device_count_or = PjrtBackend::GetInstance().GetGlobalDeviceCount();
  ASSERT_EQ(device_count_or.status(), absl::OkStatus());
  EXPECT_EQ(device_count_or.value(), 1);
}

TEST_F(PjRtInitTest, InitializeExpandsRankInEnvVars) {
  auto cleanup = absl::MakeCleanup([] {
    unsetenv(kXlaFlagsEnvVar);
    unsetenv(kLibtpuInitArgsEnvVar);
    unsetenv(kRankEnvVar);
  });

  // Use a valid XLA flag to avoid parsing errors in XLA initialization.
  SetEnv<kXlaFlagsEnvVar>("--xla_disable_hlo_passes=some_pass_${RANK}");
  SetEnv<kLibtpuInitArgsEnvVar>("some_args_${RANK}");
  SetEnv<kRankEnvVar>("1234");

  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "tpu"});
  auto* client = PjrtBackend::GetInstance().GetClient();
  ASSERT_NE(client, nullptr);

  const char* xla_flags = std::getenv(kXlaFlagsEnvVar);
  EXPECT_NE(xla_flags, nullptr);
  EXPECT_STREQ(xla_flags, "--xla_disable_hlo_passes=some_pass_1234");

  const char* libtpu_args = std::getenv(kLibtpuInitArgsEnvVar);
  EXPECT_NE(libtpu_args, nullptr);
  EXPECT_STREQ(libtpu_args, "some_args_1234");
}

}  // namespace
}  // namespace torch_tpu
