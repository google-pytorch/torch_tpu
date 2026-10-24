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

#include <unistd.h>

#include <cstdlib>
#include <string>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/pjrt/pjrt_state.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {
namespace {

using absl_testing::StatusIs;
using testing::ExitedWithCode;
using testing::HasSubstr;

class PjRtInitTest : public testing::Test {
 protected:
  void TearDown() override {
    // Hooks outlive Shutdown(), and Shutdown() is a no-op in a "bad fork";
    // clear them first.
    PjrtBackend::GetInstance().SetPjRtInitializationOptions({});
    PjrtBackend::GetInstance().Shutdown();
  }
};

// Initializes the PJRT TPU backend in single-device mode, verifies that the
// runtime sees 1 addressable and 1 global device, and shuts down the backend.
void VerifySingleDevicePjRtInit() {
  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "tpu"});
  ASSERT_EQ(PjrtBackend::GetInstance().EnsureInitialized(), absl::OkStatus());
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtClient * client,
                          PjrtBackend::GetInstance().GetClient());
  ASSERT_NE(client, nullptr);
  EXPECT_EQ(client->addressable_device_count(), 1);
  TT_ASSERT_OK_AND_ASSIGN(int device_count,
                          PjrtBackend::GetInstance().GetGlobalDeviceCount());
  EXPECT_EQ(device_count, 1);
  PjrtBackend::GetInstance().Shutdown();
}

TEST(PjRtInitDeathTest,
     EnsureInitializedSucceedsWhenWorldSizeIsOneWithoutDistEnvVars) {
  EXPECT_EXIT(
      {
        SetEnv<kWorldSizeEnvVar>("1");
        unsetenv(kTorchTpuSlicebuilderAddressesEnvVar);
        unsetenv(kTorchTpuTopologyEnvVar);
        VerifySingleDevicePjRtInit();
        _exit(testing::Test::HasFailure() ? 1 : 0);
      },
      ExitedWithCode(0), "");
}

TEST(PjRtInitDeathTest,
     EnsureInitializedSucceedsWhenWorldSizeIsUnsetWithoutDistEnvVars) {
  EXPECT_EXIT(
      {
        unsetenv(kWorldSizeEnvVar);
        unsetenv(kTorchTpuSlicebuilderAddressesEnvVar);
        unsetenv(kTorchTpuTopologyEnvVar);
        VerifySingleDevicePjRtInit();
        _exit(testing::Test::HasFailure() ? 1 : 0);
      },
      ExitedWithCode(0), "");
}

TEST_F(PjRtInitTest, InitializePjRtIsLazy) {
  ASSERT_FALSE(PjrtBackend::GetInstance().IsInitialized());

  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "tpu"});
  EXPECT_FALSE(PjrtBackend::GetInstance().IsInitialized());

  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtClient * client,
                          PjrtBackend::GetInstance().GetClient());
  EXPECT_NE(client, nullptr);
  EXPECT_TRUE(PjrtBackend::GetInstance().IsInitialized());
  EXPECT_STREQ(std::getenv(kTpuPremappedBufferSizeEnvVar), "0");
}

TEST_F(PjRtInitTest, GetClientAndGetDeviceSucceedAfterInit) {
  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "tpu"});

  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtClient * client,
                          PjrtBackend::GetInstance().GetClient());
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtDevice * device,
                          PjrtBackend::GetInstance().GetDevice());
  ASSERT_NE(client, nullptr);
  ASSERT_NE(device, nullptr);
  EXPECT_EQ(device->client(), client);
}

TEST_F(PjRtInitTest, GetClientAndGetDeviceFailForUnsupportedDeviceType) {
  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "unsupported"});

  EXPECT_FALSE(PjrtBackend::GetInstance().GetClient().ok());
  EXPECT_FALSE(PjrtBackend::GetInstance().GetDevice().ok());
  EXPECT_FALSE(PjrtBackend::GetInstance().IsInitialized());
}

TEST_F(PjRtInitTest, InitializePjRtSetsCorrectDeviceCount) {
  ASSERT_FALSE(PjrtBackend::GetInstance().IsInitialized());

  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "tpu"});

  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtClient * client,
                          PjrtBackend::GetInstance().GetClient());
  EXPECT_NE(client, nullptr);
  EXPECT_TRUE(PjrtBackend::GetInstance().IsInitialized());

  TT_ASSERT_OK_AND_ASSIGN(int device_count,
                          PjrtBackend::GetInstance().GetGlobalDeviceCount());
  EXPECT_EQ(device_count, 1);
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
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtClient * client,
                          PjrtBackend::GetInstance().GetClient());
  EXPECT_NE(client, nullptr);

  const char* xla_flags = std::getenv(kXlaFlagsEnvVar);
  EXPECT_NE(xla_flags, nullptr);
  EXPECT_STREQ(xla_flags, "--xla_disable_hlo_passes=some_pass_1234");

  const char* libtpu_args = std::getenv(kLibtpuInitArgsEnvVar);
  EXPECT_NE(libtpu_args, nullptr);
  EXPECT_STREQ(libtpu_args, "some_args_1234");
}

TEST_F(PjRtInitTest, BadForkCheckerPreventsAccessAfterInit) {
  ASSERT_FALSE(PjrtBackend::GetInstance().IsInitialized());

  static bool in_bad_fork = false;
  in_bad_fork = false;
  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "tpu", .bad_fork_checker = []() { return in_bad_fork; }});

  ASSERT_EQ(PjrtBackend::GetInstance().EnsureInitialized(), absl::OkStatus());
  EXPECT_TRUE(PjrtBackend::GetInstance().IsInitialized());

  // Simulate fork in child after parent already initialized.
  in_bad_fork = true;
  EXPECT_TRUE(PjrtBackend::GetInstance().IsInBadFork());
  EXPECT_FALSE(PjrtBackend::GetInstance().IsInitialized());
  EXPECT_FALSE(PjrtBackend::GetInstance().GetClient().ok());
  EXPECT_FALSE(PjrtBackend::GetInstance().GetDevice().ok());
  EXPECT_THAT(
      PjrtBackend::GetInstance().EnsureInitialized(),
      StatusIs(error::kFailedPrecondition, HasSubstr("forked subprocess")));
}

TEST_F(PjRtInitTest, OnInitializedHookIsInvoked) {
  ASSERT_FALSE(PjrtBackend::GetInstance().IsInitialized());

  static bool hook_called = false;
  hook_called = false;
  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "tpu",
       .bad_fork_checker = nullptr,
       .on_initialized_hook = []() { hook_called = true; }});

  EXPECT_FALSE(hook_called);
  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtClient * client,
                          PjrtBackend::GetInstance().GetClient());
  EXPECT_NE(client, nullptr);
  EXPECT_TRUE(hook_called);
  EXPECT_TRUE(PjrtBackend::GetInstance().IsInitialized());
}

TEST_F(PjRtInitTest, SingleSliceInitDoesNotSetSkipMegascaleEnvVar) {
  unsetenv(kSkipMegascalePjrtClientEnvVar);
  PjrtBackend::GetInstance().SetPjRtInitializationOptions(
      {.device_type = "tpu"});

  TT_ASSERT_OK_AND_ASSIGN(xla::PjRtClient * client,
                          PjrtBackend::GetInstance().GetClient());
  ASSERT_NE(client, nullptr);
  EXPECT_EQ(std::getenv(kSkipMegascalePjrtClientEnvVar), nullptr);
}

}  // namespace
}  // namespace torch_tpu
