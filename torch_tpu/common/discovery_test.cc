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
#include "torch_tpu/common/discovery.h"

#include <cstdlib>
#include <string>

#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/error_utils.h"

namespace torch_tpu {
namespace {

using testing::ExitedWithCode;

// Since GetPremappedBufferSizeFromEnvOnce() memoizes the environment variable,
// we need to use EXPECT_EXIT() to test the behavior of the function when
// setting the environment variable to different values (EXPECT_EXIT() runs
// the function in a subprocess, so side effects are not shared between tests).

TEST(GetPremappedBufferSizeFromEnvOnceDeathTest, Returns0ByDefault) {
  EXPECT_EXIT(
      {
        unsetenv(kTpuPremappedBufferSizeEnvVar);
        TT_ASSIGN_OR_CRASH(auto size, GetPremappedBufferSizeFromEnvOnce());
        exit(size);
      },
      // Verifies that *status_or_size is 0.
      ExitedWithCode(0), "");
}

TEST(GetPremappedBufferSizeFromEnvOnceDeathTest, SetsEnvVarIfNotFound) {
  EXPECT_EXIT(
      {
        unsetenv(kTpuPremappedBufferSizeEnvVar);
        const auto& status_or_size = GetPremappedBufferSizeFromEnvOnce();
        ABSL_CHECK_OK(status_or_size.status());
        // This should be true.
        const bool env_var_is_0 =
            std::string(getenv(kTpuPremappedBufferSizeEnvVar)) == "0";
        exit(env_var_is_0 ? 0 : 1);
      },
      // Verifies that env_var_is_0 is true.
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

}  // namespace
}  // namespace torch_tpu
