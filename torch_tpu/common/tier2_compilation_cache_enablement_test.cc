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

#include <string>

#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "gtest/gtest.h"
#include "torch_tpu/common/flags.h"
#include "torch_tpu/common/tier2_compilation_cache.h"
#include "torch_tpu/pjrt/pjrt_state.h"

ABSL_FLAG(std::string, expected_tier2_cache_name, "",
          "The expected name of the tier-2 compilation cache, or empty if the "
          "cache is expected to be disabled.");

namespace torch_tpu {
namespace {

// A test environment that initializes the PjRt client, which is required for
// GetFromTier2Cache() to work.
class TpuTestEnvironment : public testing::Environment {
 public:
  void SetUp() override {
    // Use xla_cpu for testing to allow mocking multiple devices in a single
    // process without needing real TPU hardware or multiple workers.
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = "xla_cpu"});
    ABSL_CHECK_OK(PjrtBackend::GetInstance().EnsureInitialized());
  }
};

// Installs the test environment.
auto* const test_env =
    testing::AddGlobalTestEnvironment(new TpuTestEnvironment);

// Since GetTier2CacheName() is memoized, we can only test one scenario per
// test program.
TEST(Tier2CompilationCacheEnablementTest, Tier2CacheName) {
  EXPECT_EQ(GetTier2CacheName(),
            (GetFlagOnce<std::string, &FLAGS_expected_tier2_cache_name>()));
}

}  // namespace
}  // namespace torch_tpu
