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
#include "csrc/common/discovery.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "c10/util/Exception.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

using testing::ExitedWithCode;

// Since GetPremappedBufferSizeFromEnvOnce() memoizes the environment variable,
// we need to use EXPECT_EXIT() to test the behavior of the function when
// setting the environment variable to different values (EXPECT_EXIT() runs
// the function in a subprocess, so side effects are not shared between tests).

TEST(GetPremappedBufferSizeFromEnvOnceDeathTest,
     SetsAndReturnsDefaultWhenUnset) {
  EXPECT_EXIT(
      {
        unsetenv(kTpuPremappedBufferSizeEnvVar);
        TT_ASSIGN_OR_CRASH(const int64_t size,
                           GetPremappedBufferSizeFromEnvOnce());
        const char* const env_str = getenv(kTpuPremappedBufferSizeEnvVar);
        ABSL_CHECK_NE(env_str, nullptr);
        int64_t parsed = -1;
        ABSL_CHECK(absl::SimpleAtoi(env_str, &parsed));
        ABSL_CHECK_EQ(parsed, size);
        ABSL_CHECK_GE(size, 0);
        exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(GetPremappedBufferSizeFromEnvOnceDeathTest, ReturnsEnvVarValueIfSet) {
  EXPECT_EXIT(
      {
        setenv(kTpuPremappedBufferSizeEnvVar, "123", 1);
        const auto& status_or_size = GetPremappedBufferSizeFromEnvOnce();
        if (!status_or_size.ok()) {
          exit(1);  // Failed to getenv.
        }
        exit(*status_or_size);
      },
      // Verifies that *status_or_size is 123.
      ExitedWithCode(123), "");
}

TEST(GetPremappedBufferSizeFromEnvOnceDeathTest, ReturnsErrorIfInvalid) {
  EXPECT_EXIT(
      {
        setenv(kTpuPremappedBufferSizeEnvVar, "not_an_int", 1);
        const auto& status_or_size = GetPremappedBufferSizeFromEnvOnce();
        exit(status_or_size.ok() ? 1 : 0);
      },
      ExitedWithCode(0), "");
}

// Records the c10 warnings (e.g. from TT_WARN) emitted while in scope.
class WarningCapture : public c10::WarningHandler {
 public:
  WarningCapture() : prev_(c10::WarningUtils::get_warning_handler()) {
    c10::WarningUtils::set_warning_handler(this);
  }

  ~WarningCapture() override { c10::WarningUtils::set_warning_handler(prev_); }

  void process(const c10::Warning& warning) override {
    messages_.push_back(warning.msg());
  }

  const std::vector<std::string>& messages() const { return messages_; }

 private:
  c10::WarningHandler* prev_ = nullptr;
  std::vector<std::string> messages_;
};

TEST(GetPremappedBufferSizeFromEnvOnceDeathTest, WarnsIfDefaultIsZero) {
  EXPECT_EXIT(
      {
        unsetenv(kTpuPremappedBufferSizeEnvVar);
        // So many local processes that no host can give each of them enough
        // memory for a non-zero default.
        setenv(kWorldSizeEnvVar, "1048576", 1);
        setenv(kTorchTpuDevicesPerHostEnvVar, "1048576", 1);
        WarningCapture warnings;
        TT_ASSIGN_OR_CRASH(const int64_t size,
                           GetPremappedBufferSizeFromEnvOnce());
        ABSL_CHECK_EQ(size, 0);
        ABSL_CHECK_EQ(warnings.messages().size(), 1);
        ABSL_CHECK(absl::StrContains(warnings.messages()[0],
                                     "//TorchTpu/Performance// "
                                     "TPU_PREMAPPED_BUFFER_SIZE is unset and "
                                     "its automatic default is 0"));
        exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(GetPremappedBufferSizeFromEnvOnceDeathTest, DoesNotWarnIfSetToZero) {
  EXPECT_EXIT(
      {
        setenv(kTpuPremappedBufferSizeEnvVar, "0", 1);
        WarningCapture warnings;
        TT_ASSIGN_OR_CRASH(const int64_t size,
                           GetPremappedBufferSizeFromEnvOnce());
        ABSL_CHECK_EQ(size, 0);
        ABSL_CHECK(warnings.messages().empty());
        exit(0);
      },
      ExitedWithCode(0), "");
}

}  // namespace
}  // namespace torch_tpu
