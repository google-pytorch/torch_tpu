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

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "c10/util/Exception.h"
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
  setenv(kTpuDeferAndFuse, "0", 1);
  EXPECT_EQ(GetBooleanEnvOnce<kTpuDeferAndFuse>(), false);
}

TEST(EnvVarsTest, GetBooleanEnvOnceParsesTrueFromString) {
  setenv(kTpuLaunchBlocking, "true", 1);
  EXPECT_EQ(GetBooleanEnvOnce<kTpuLaunchBlocking>(), true);
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
  EXPECT_EQ((GetIntegerEnvOnce<int, kMasterPortEnvVar>()), std::nullopt);
}

TEST(EnvVarsTest, GetIntegerEnvOnceParsesValidInt) {
  setenv(kLocalRankEnvVar, "42", 1);
  EXPECT_EQ((GetIntegerEnvOnce<int, kLocalRankEnvVar>()), 42);
}

TEST(EnvVarsTest, GetIntegerEnvOnceParsesNegativeInt) {
  setenv(kRankEnvVar, "-1", 1);
  EXPECT_EQ((GetIntegerEnvOnce<int, kRankEnvVar>()), -1);
}

TEST(EnvVarsTest, GetIntegerEnvOnceParsesInt64) {
  setenv(kTpuPremappedBufferSizeEnvVar, "10737418240", 1);
  EXPECT_EQ((GetIntegerEnvOnce<int64_t, kTpuPremappedBufferSizeEnvVar>()),
            10737418240LL);
}

TEST(EnvVarsTest, GetIntegerEnvOnceReturnsNulloptOnInvalidValue) {
  setenv(kTorchTpuHandshakePortEnvVar, "invalid_int", 1);
  EXPECT_EQ((GetIntegerEnvOnce<int, kTorchTpuHandshakePortEnvVar>()),
            std::nullopt);
}

TEST(EnvVarsTest, GetEnvSinceResetForTestingMemoizesUntilNextReset) {
  setenv(kNodeRankEnvVar, "1", 1);
  ResetEnvCacheForTesting();
  EXPECT_TRUE(EnvCacheWasResetForTesting());
  EXPECT_EQ(GetEnvSinceResetForTesting(kNodeRankEnvVar), "1");

  setenv(kNodeRankEnvVar, "2", 1);
  EXPECT_EQ(GetEnvSinceResetForTesting(kNodeRankEnvVar), "1");

  ResetEnvCacheForTesting();
  EXPECT_EQ(GetEnvSinceResetForTesting(kNodeRankEnvVar), "2");

  unsetenv(kNodeRankEnvVar);
  ResetEnvCacheForTesting();
  EXPECT_EQ(GetEnvSinceResetForTesting(kNodeRankEnvVar), std::nullopt);
}

TEST(EnvVarsTest, ResetEnvCacheForTestingDoesNotResetGetEnvOnce) {
  setenv(kMasterAddrEnvVar, "worker0", 1);
  EXPECT_EQ(GetEnvOnce<kMasterAddrEnvVar>(), "worker0");

  setenv(kMasterAddrEnvVar, "worker1", 1);
  ResetEnvCacheForTesting();
  EXPECT_EQ(GetEnvOnce<kMasterAddrEnvVar>(), "worker0");
  EXPECT_EQ(GetEnvSinceResetForTesting(kMasterAddrEnvVar), "worker1");
}

}  // namespace
}  // namespace torch_tpu
