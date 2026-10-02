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

#include "csrc/common/env_vars.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/absl_check.h"
#include "c10/util/Exception.h"
#include "csrc/common/env_vars_testing.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

using testing::ContainsRegex;
using testing::ElementsAre;

// RAII helper struct that intercepts and records c10/PyTorch warnings (e.g.,
// from TORCH_WARN or TORCH_WARN_ONCE) within a test scope.
//
// How it works:
// 1. On construction, it saves the existing active c10 warning handler
//    (`prev_`) and registers `this` as the current handler via
//    `c10::WarningUtils::set_warning_handler(this)`.
// 2. When any warning macro (such as `TORCH_WARN_ONCE`) is triggered, c10
//    invokes `process(const c10::Warning& warning)`. Instead of logging to
//    stderr, `WarningCapture` appends the warning string (`warning.msg()`) to
//    its internal `messages_` vector.
// 3. On destruction, it restores the original warning handler (`prev_`),
//    preventing any side effects from leaking to subsequent tests.
class WarningCapture : public c10::WarningHandler {
 public:
  WarningCapture() : prev_(c10::WarningUtils::get_warning_handler()) {
    c10::WarningUtils::set_warning_handler(this);
  }

  ~WarningCapture() override { c10::WarningUtils::set_warning_handler(prev_); }

  // Called by c10 when a warning is emitted. We record the message string for
  // test assertion inspection.
  void process(const c10::Warning& warning) override {
    messages_.push_back(warning.msg());
  }

  const std::vector<std::string>& messages() const { return messages_; }

 private:
  // The warning handler that was active before this capture scope was created.
  c10::WarningHandler* prev_ = nullptr;

  // All captured warning messages in the order they were emitted.
  std::vector<std::string> messages_;
};

// Tests that only the first read of TORCH_TPU_TIER2_COMPILATION_CACHE triggers
// a warning.
TEST(EnvVarsTest, Tier2CompilationCacheWarnsOnce) {
  setenv(kTorchTpuTier2CompilationCacheEnvVar, "my_cache", 1);

  WarningCapture warnings;
  EXPECT_TRUE(warnings.messages().empty());

  // First call should trigger the warning.
  EXPECT_TRUE(GetEnvOnce<kTorchTpuTier2CompilationCacheEnvVar>().has_value());
  EXPECT_THAT(warnings.messages(),
              ElementsAre(ContainsRegex(
                  "TORCH_TPU_TIER2_COMPILATION_CACHE .* experimental")));

  // Second call should NOT trigger another warning.
  EXPECT_TRUE(GetEnvOnce<kTorchTpuTier2CompilationCacheEnvVar>().has_value());
  EXPECT_EQ(warnings.messages().size(), 1);
}

TEST(EnvVarsTest, InternalDisableInplaceBufferDonationNoWarning) {
  setenv(kTorchTpuInternalDisableInplaceBufferDonationEnvVar, "1", 1);

  WarningCapture warnings;
  EXPECT_TRUE(warnings.messages().empty());

  const auto val =
      GetBooleanEnvOnce<kTorchTpuInternalDisableInplaceBufferDonationEnvVar>();
  EXPECT_TRUE(val.has_value());
  EXPECT_TRUE(*val);
  EXPECT_TRUE(warnings.messages().empty());
}

TEST(EnvVarsTest, GetBooleanEnvOnceUnsetReturnsNullopt) {
  unsetenv(kTorchTpuInternalSplitRngStateUpdate);
  EXPECT_EQ(GetBooleanEnvOnce<kTorchTpuInternalSplitRngStateUpdate>(),
            std::nullopt);
}

TEST(EnvVarsTest, GetBooleanEnvOnceParsesFalseFromZero) {
  setenv(kTorchTpuEnableDebugChecksEnvVar, "0", 1);
  EXPECT_EQ(GetBooleanEnvOnce<kTorchTpuEnableDebugChecksEnvVar>(), false);
}

TEST(EnvVarsTest, GetBooleanEnvOnceParsesTrueFromString) {
  setenv(kTorchTpuInternalEnableReassociateNormWeightsEnvVar, "true", 1);
  EXPECT_EQ(
      GetBooleanEnvOnce<kTorchTpuInternalEnableReassociateNormWeightsEnvVar>(),
      true);
}

TEST(EnvVarsTest, GetBooleanEnvOnceParsesFalseFromString) {
  setenv(kTorchTpuDeferAndFuseEnvVar, "false", 1);
  EXPECT_EQ(GetBooleanEnvOnce<kTorchTpuDeferAndFuseEnvVar>(), false);
}

TEST(EnvVarsTest, GetBooleanEnvOnceReturnsNulloptOnInvalidValue) {
  setenv(kTorchTpuLaunchBlockingEnvVar, "invalid_boolean", 1);
  EXPECT_EQ(GetBooleanEnvOnce<kTorchTpuLaunchBlockingEnvVar>(), std::nullopt);
}

TEST(EnvVarsTest, GetIntegerEnvOnceUnsetReturnsNullopt) {
  unsetenv(kMasterPortEnvVar);
  EXPECT_EQ((GetIntegerEnvOnce<kMasterPortEnvVar>()), std::nullopt);
}

TEST(EnvVarsTest, GetIntegerEnvOnceParsesValidInt) {
  setenv(kLocalRankEnvVar, "42", 1);
  EXPECT_EQ((GetIntegerEnvOnce<kLocalRankEnvVar>()), 42);
}

TEST(EnvVarsTest, GetIntegerEnvOnceParsesNegativeInt) {
  setenv(kRankEnvVar, "-1", 1);
  EXPECT_EQ((GetIntegerEnvOnce<kRankEnvVar>()), -1);
}

TEST(EnvVarsTest, GetIntegerEnvOnceParsesInt64) {
  setenv(kTpuPremappedBufferSizeEnvVar, "10737418240", 1);
  EXPECT_EQ((GetIntegerEnvOnce<kTpuPremappedBufferSizeEnvVar>()),
            10737418240LL);
}

TEST(EnvVarsTest, GetIntegerEnvOnceReturnsNulloptOnInvalidValue) {
  setenv(kTorchTpuHandshakePortEnvVar, "invalid_int", 1);
  EXPECT_EQ((GetIntegerEnvOnce<kTorchTpuHandshakePortEnvVar>()), std::nullopt);
}

// Sets the env vars that reference ${RANK}, runs `read_first`, and checks that
// the rank is expanded both in the memoized values and in the process
// environment that libtpu and XLA read.
template <typename ReadFirst>
void CheckRankIsExpandedOnFirstRead(ReadFirst read_first) {
  SafeSetEnv<kXlaFlagsEnvVar>("--xla_disable_hlo_passes=some_pass_${RANK}");
  SafeSetEnv<kLibtpuInitArgsEnvVar>("some_args_${RANK}");
  SafeSetEnv<kRankEnvVar>("1234");

  read_first();

  ABSL_CHECK_EQ(ExpandAndGetXlaFlagsEnvVarOnce().value_or(""),
                "--xla_disable_hlo_passes=some_pass_1234");
  ABSL_CHECK_EQ(ExpandAndGetLibtpuInitArgsEnvVarOnce().value_or(""),
                "some_args_1234");
  const char* const xla_flags = std::getenv(kXlaFlagsEnvVar);
  ABSL_CHECK_NE(xla_flags, nullptr);
  ABSL_CHECK_EQ(std::string(xla_flags),
                "--xla_disable_hlo_passes=some_pass_1234");
  const char* const libtpu_init_args = std::getenv(kLibtpuInitArgsEnvVar);
  ABSL_CHECK_NE(libtpu_init_args, nullptr);
  ABSL_CHECK_EQ(std::string(libtpu_init_args), "some_args_1234");
}

// The rank is expanded and memoized at most once per process, and other tests
// in this binary read RANK, so the tests below run in a subprocess where the
// env vars are set before anything reads them. They use the "threadsafe" style,
// which re-executes the test binary, since the "fast" style forks the current
// process and inherits its memoized env vars.

TEST(EnvVarsDeathTest, ExpandAndGetXlaFlagsEnvVarOnceExpandsRank) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        CheckRankIsExpandedOnFirstRead(
            [] { static_cast<void>(ExpandAndGetXlaFlagsEnvVarOnce()); });
        std::exit(0);
      },
      testing::ExitedWithCode(0), "");
}

TEST(EnvVarsDeathTest, ExpandAndGetLibtpuInitArgsEnvVarOnceExpandsRank) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        CheckRankIsExpandedOnFirstRead(
            [] { static_cast<void>(ExpandAndGetLibtpuInitArgsEnvVarOnce()); });
        std::exit(0);
      },
      testing::ExitedWithCode(0), "");
}

TEST(EnvVarsDeathTest, EnsureRankIsExpandedInEnvVarsExpandsRank) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        CheckRankIsExpandedOnFirstRead([] { EnsureRankIsExpandedInEnvVars(); });
        std::exit(0);
      },
      testing::ExitedWithCode(0), "");
}

TEST(EnvVarsTest, SetEnvBeforeGetEnvOnceSucceeds) {
  SafeSetEnv<kTorchTpuInternalDumpMlirPassesToEnvVar>("print");
  EXPECT_EQ(GetEnvOnce<kTorchTpuInternalDumpMlirPassesToEnvVar>(), "print");
}

TEST(EnvVarsTest, UnsetEnvBeforeGetEnvOnceSucceeds) {
  SafeSetEnv<kNnodesEnvVar>("2");
  SafeUnsetEnv<kNnodesEnvVar>();
  EXPECT_EQ(GetEnvOnce<kNnodesEnvVar>(), std::nullopt);
}

TEST(EnvVarsDeathTest, SetEnvAfterGetEnvOnceCrashes) {
  EXPECT_DEATH(
      {
        static_cast<void>(GetEnvOnce<kTorchTpuSpmdHandshakeStageEnvVar>());
        SafeSetEnv<kTorchTpuSpmdHandshakeStageEnvVar>("OFF");
      },
      "TORCH_TPU_SPMD_HANDSHAKE_STAGE.*GetEnvOnce");
}

TEST(EnvVarsDeathTest, UnsetEnvAfterGetIntegerEnvOnceCrashes) {
  EXPECT_DEATH(
      {
        static_cast<void>(GetIntegerEnvOnce<kTorchcommRankEnvVar>());
        SafeUnsetEnv<kTorchcommRankEnvVar>();
      },
      "TORCHCOMM_RANK.*GetEnvOnce");
}

TEST(EnvVarsTest, ResetEnvCacheForTestingRereadsEnvVars) {
  setenv(kMasterAddrEnvVar, "worker0", 1);
  setenv(kTorchTpuInternalSkinnyScOffloadEnvVar, "1", 1);
  setenv(kTorchTpuInternalHandshakeTimeoutSEnvVar, "10", 1);
  ResetEnvCacheForTesting();
  const std::optional<std::string>& master_addr =
      GetEnvOnce<kMasterAddrEnvVar>();
  EXPECT_EQ(master_addr, "worker0");
  EXPECT_EQ(GetBooleanEnvOnce<kTorchTpuInternalSkinnyScOffloadEnvVar>(), true);
  EXPECT_EQ((GetIntegerEnvOnce<kTorchTpuInternalHandshakeTimeoutSEnvVar>()),
            10);

  // Without a reset, the memoized values are kept.
  setenv(kMasterAddrEnvVar, "worker1", 1);
  setenv(kTorchTpuInternalSkinnyScOffloadEnvVar, "0", 1);
  setenv(kTorchTpuInternalHandshakeTimeoutSEnvVar, "20", 1);
  EXPECT_EQ(GetEnvOnce<kMasterAddrEnvVar>(), "worker0");
  EXPECT_EQ(GetBooleanEnvOnce<kTorchTpuInternalSkinnyScOffloadEnvVar>(), true);
  EXPECT_EQ((GetIntegerEnvOnce<kTorchTpuInternalHandshakeTimeoutSEnvVar>()),
            10);

  // After a reset, the updated values are read.
  ResetEnvCacheForTesting();
  EXPECT_EQ(GetEnvOnce<kMasterAddrEnvVar>(), "worker1");
  EXPECT_EQ(master_addr, "worker1");
  EXPECT_EQ(GetBooleanEnvOnce<kTorchTpuInternalSkinnyScOffloadEnvVar>(), false);
  EXPECT_EQ((GetIntegerEnvOnce<kTorchTpuInternalHandshakeTimeoutSEnvVar>()),
            20);

  unsetenv(kMasterAddrEnvVar);
  unsetenv(kTorchTpuInternalSkinnyScOffloadEnvVar);
  unsetenv(kTorchTpuInternalHandshakeTimeoutSEnvVar);
  ResetEnvCacheForTesting();
  EXPECT_EQ(GetEnvOnce<kMasterAddrEnvVar>(), std::nullopt);
  ResetEnvCacheForTesting();
}

TEST(EnvVarsTest, GetBooleanEnvOnceWarnsOncePerReset) {
  setenv(kTorchTpuInternalMaterializeCollectiveTensorsEnvVar, "invalid_boolean",
         1);
  ResetEnvCacheForTesting();

  WarningCapture warnings;
  EXPECT_EQ(
      GetBooleanEnvOnce<kTorchTpuInternalMaterializeCollectiveTensorsEnvVar>(),
      std::nullopt);
  EXPECT_THAT(
      warnings.messages(),
      ElementsAre(ContainsRegex(
          "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS.*Boolean")));

  // Reading the same variable again before a reset does not warn.
  EXPECT_EQ(
      GetBooleanEnvOnce<kTorchTpuInternalMaterializeCollectiveTensorsEnvVar>(),
      std::nullopt);
  EXPECT_EQ(warnings.messages().size(), 1);

  // After a reset, an invalid value warns once again.
  ResetEnvCacheForTesting();
  EXPECT_EQ(
      GetBooleanEnvOnce<kTorchTpuInternalMaterializeCollectiveTensorsEnvVar>(),
      std::nullopt);
  EXPECT_EQ(warnings.messages().size(), 2);
  EXPECT_EQ(
      GetBooleanEnvOnce<kTorchTpuInternalMaterializeCollectiveTensorsEnvVar>(),
      std::nullopt);
  EXPECT_EQ(warnings.messages().size(), 2);

  // A valid value does not warn.
  setenv(kTorchTpuInternalMaterializeCollectiveTensorsEnvVar, "true", 1);
  ResetEnvCacheForTesting();
  EXPECT_EQ(
      GetBooleanEnvOnce<kTorchTpuInternalMaterializeCollectiveTensorsEnvVar>(),
      true);
  EXPECT_EQ(warnings.messages().size(), 2);

  unsetenv(kTorchTpuInternalMaterializeCollectiveTensorsEnvVar);
  ResetEnvCacheForTesting();
}

TEST(EnvVarsTest, GetIntegerEnvOnceWarnsOncePerReset) {
  setenv(kTorchTpuDevicesPerHostEnvVar, "invalid_int", 1);
  ResetEnvCacheForTesting();

  WarningCapture warnings;
  EXPECT_EQ((GetIntegerEnvOnce<kTorchTpuDevicesPerHostEnvVar>()), std::nullopt);
  // TORCH_TPU_DEVICES_PER_HOST is experimental, so setting it also warns.
  EXPECT_THAT(
      warnings.messages(),
      ElementsAre(ContainsRegex("TORCH_TPU_DEVICES_PER_HOST .* experimental"),
                  ContainsRegex("TORCH_TPU_DEVICES_PER_HOST.*integer")));

  // Reading the same variable again before a reset does not warn.
  EXPECT_EQ((GetIntegerEnvOnce<kTorchTpuDevicesPerHostEnvVar>()), std::nullopt);
  EXPECT_EQ(warnings.messages().size(), 2);

  // After a reset, an invalid value warns once again.
  ResetEnvCacheForTesting();
  EXPECT_EQ((GetIntegerEnvOnce<kTorchTpuDevicesPerHostEnvVar>()), std::nullopt);
  EXPECT_EQ(warnings.messages().size(), 4);

  // A valid value does not warn, other than about the experimental variable.
  setenv(kTorchTpuDevicesPerHostEnvVar, "4", 1);
  ResetEnvCacheForTesting();
  EXPECT_EQ((GetIntegerEnvOnce<kTorchTpuDevicesPerHostEnvVar>()), 4);
  EXPECT_EQ(warnings.messages().size(), 5);

  unsetenv(kTorchTpuDevicesPerHostEnvVar);
  ResetEnvCacheForTesting();
}

TEST(EnvVarsDeathTest, SafeSetEnvAfterResetSucceedsUntilReread) {
  setenv(kTorchTpuInternalXlaOptionsEnvVar, "a", 1);
  EXPECT_EQ(GetEnvOnce<kTorchTpuInternalXlaOptionsEnvVar>(), "a");

  ResetEnvCacheForTesting();
  SafeSetEnv<kTorchTpuInternalXlaOptionsEnvVar>("b");
  EXPECT_EQ(GetEnvOnce<kTorchTpuInternalXlaOptionsEnvVar>(), "b");
  EXPECT_DEATH(SafeSetEnv<kTorchTpuInternalXlaOptionsEnvVar>("c"),
               "TORCH_TPU_INTERNAL_XLA_OPTIONS.*GetEnvOnce");

  ResetEnvCacheForTesting();
  SafeUnsetEnv<kTorchTpuInternalXlaOptionsEnvVar>();
  ResetEnvCacheForTesting();
}

}  // namespace
}  // namespace torch_tpu
