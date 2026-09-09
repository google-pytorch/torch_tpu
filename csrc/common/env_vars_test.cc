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
  GetEnvOnce<kTorchTpuTier2CompilationCacheEnvVar>();
  EXPECT_THAT(warnings.messages(),
              ElementsAre(ContainsRegex(
                  "TORCH_TPU_TIER2_COMPILATION_CACHE .* experimental")));

  // Second call should NOT trigger another warning.
  GetEnvOnce<kTorchTpuTier2CompilationCacheEnvVar>();
  EXPECT_EQ(warnings.messages().size(), 1);
}

TEST(EnvVarsTest, InternalDisableInplaceBufferDonationNoWarning) {
  setenv(kTorchTpuInternalDisableInplaceBufferDonationEnvVar, "1", 1);

  WarningCapture warnings;
  EXPECT_TRUE(warnings.messages().empty());

  const auto& val =
      GetEnvOnce<kTorchTpuInternalDisableInplaceBufferDonationEnvVar>();
  EXPECT_TRUE(val.has_value());
  EXPECT_EQ(*val, "1");
  EXPECT_TRUE(warnings.messages().empty());
}

}  // namespace
}  // namespace torch_tpu
