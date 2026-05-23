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

#include "torch_tpu/common/environment.h"

#include <cstdlib>
#include <string>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/distributed/slicebuilder/discovery.h"

namespace torch_tpu {
namespace {

TEST(EnvironmentTest,
     InitializeDistributedEnvironmentSetsAllowMultipleLibtpuLoad) {
  unsetenv(kAllowMultipleLibtpuLoadEnvVar);
  unsetenv(kTpuProcessAddressesEnvVar);
  unsetenv(kTpuHostBoundsEnvVar);
  unsetenv(kTpuChipsPerHostBoundsEnvVar);
  unsetenv(kTpuProcessBoundsEnvVar);
  unsetenv(kTpuChipsPerProcessBoundsEnvVar);
  DistributedWorkerConfiguration config;
  config.rank = 0;
  config.local_rank = 0;
  config.sb_port = "1234";
  config.sb_addrs = "localhost:1234";
  config.topology = "1,1,1";

  EXPECT_EQ(InitializeDistributedEnvironment(config), absl::OkStatus());

  const char* env_val = std::getenv(kAllowMultipleLibtpuLoadEnvVar);
  ASSERT_NE(env_val, nullptr);
  EXPECT_STREQ(env_val, "1");

  const char* addr_val = std::getenv(kTpuProcessAddressesEnvVar);
  ASSERT_NE(addr_val, nullptr);
  EXPECT_STREQ(addr_val, "localhost:1234");

  const char* host_bounds_val = std::getenv(kTpuHostBoundsEnvVar);
  ASSERT_NE(host_bounds_val, nullptr);
  EXPECT_STREQ(host_bounds_val, "1,1,1");

  const char* host_chips_val = std::getenv(kTpuChipsPerHostBoundsEnvVar);
  ASSERT_NE(host_chips_val, nullptr);
  EXPECT_STREQ(host_chips_val, "1,1,1");

  const char* proc_bounds_val = std::getenv(kTpuProcessBoundsEnvVar);
  ASSERT_NE(proc_bounds_val, nullptr);
  EXPECT_STREQ(proc_bounds_val, "1,1,1");

  const char* proc_chips_val = std::getenv(kTpuChipsPerProcessBoundsEnvVar);
  ASSERT_NE(proc_chips_val, nullptr);
  EXPECT_STREQ(proc_chips_val, "1,1,1");
}

TEST(EnvironmentTest,
     InitializeDistributedEnvironmentSetsChipsBoundsFor4DTopology) {
  unsetenv(kAllowMultipleLibtpuLoadEnvVar);
  unsetenv(kTpuProcessAddressesEnvVar);
  unsetenv(kTpuHostBoundsEnvVar);
  unsetenv(kTpuChipsPerHostBoundsEnvVar);
  unsetenv(kTpuProcessBoundsEnvVar);
  unsetenv(kTpuChipsPerProcessBoundsEnvVar);
  DistributedWorkerConfiguration config;
  config.rank = 0;
  config.local_rank = 0;
  config.sb_port = "1234";
  config.sb_addrs = "localhost:1234";
  config.topology = "2,2,1,2";

  EXPECT_EQ(InitializeDistributedEnvironment(config), absl::OkStatus());

  const char* host_chips_val = std::getenv(kTpuChipsPerHostBoundsEnvVar);
  ASSERT_NE(host_chips_val, nullptr);
  EXPECT_STREQ(host_chips_val, "1,1,1,1");

  const char* proc_chips_val = std::getenv(kTpuChipsPerProcessBoundsEnvVar);
  ASSERT_NE(proc_chips_val, nullptr);
  EXPECT_STREQ(proc_chips_val, "1,1,1,1");
}

TEST(EnvironmentTest, InitializeDistributedEnvironmentUnsetsAddressesFirst) {
  setenv(kTpuProcessAddressesEnvVar, "old_address", 1);
  DistributedWorkerConfiguration config;
  config.rank = 0;
  config.local_rank = 0;
  config.sb_port = "1234";
  config.sb_addrs = "new_address";
  config.topology = "1,1,1";

  EXPECT_EQ(InitializeDistributedEnvironment(config), absl::OkStatus());

  const char* addr_val = std::getenv(kTpuProcessAddressesEnvVar);
  ASSERT_NE(addr_val, nullptr);
  EXPECT_STREQ(addr_val, "new_address");
}

}  // namespace
}  // namespace torch_tpu
