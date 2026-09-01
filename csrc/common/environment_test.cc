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

#include "csrc/common/environment.h"

#include <unistd.h>

#include <cstdlib>
#include <string>

#include "absl/status/status.h"
#include "csrc/common/env_vars.h"
#include "csrc/distributed/slicebuilder/discovery.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

using testing::ExitedWithCode;

TEST(EnvironmentDeathTest,
     InitializeDistributedEnvironment_PopulatesSlicebuilderAddresses) {
  EXPECT_EXIT(
      {
        unsetenv(kTpuProcessAddressesEnvVar);
        unsetenv(kTpuProcessPortEnvVar);
        DistributedWorkerConfiguration config;
        config.rank = 0;
        config.local_rank = 0;
        config.sb_port = "1234";
        config.sb_addrs = "host0:1234,host1:1234";
        config.topology = "1,1,1";

        EXPECT_TRUE(InitializeDistributedEnvironment(config).ok());
        const char* addrs = std::getenv(kTpuProcessAddressesEnvVar);
        ASSERT_NE(addrs, nullptr);
        EXPECT_STREQ(addrs, "host0:1234,host1:1234");
        const char* port = std::getenv(kTpuProcessPortEnvVar);
        ASSERT_NE(port, nullptr);
        EXPECT_STREQ(port, "1234");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitializeDistributedEnvironment_PopulatesSingleDeviceProcessAddresses) {
  EXPECT_EXIT(
      {
        unsetenv(kTpuProcessAddressesEnvVar);
        unsetenv(kTpuProcessPortEnvVar);
        DistributedWorkerConfiguration config;
        config.rank = 0;
        config.local_rank = 0;
        config.sb_port = "5678";
        config.sb_addrs = "host0:5678";
        config.topology = "1,1,1";

        EXPECT_TRUE(InitializeDistributedEnvironment(config).ok());
        const char* addrs = std::getenv(kTpuProcessAddressesEnvVar);
        ASSERT_NE(addrs, nullptr);
        EXPECT_STREQ(addrs, "host0:5678");
        const char* port = std::getenv(kTpuProcessPortEnvVar);
        ASSERT_NE(port, nullptr);
        EXPECT_STREQ(port, "5678");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest, InitializeDistributedEnvironment_ResolvesBnsPort) {
  EXPECT_EXIT(
      {
        unsetenv(kTpuProcessPortEnvVar);
        DistributedWorkerConfiguration config;
        config.rank = 0;
        config.local_rank = 0;
        config.sb_port = "1234";
        config.sb_addrs = "localhost:1234";
        config.topology = "1,1,1";

        EXPECT_TRUE(InitializeDistributedEnvironment(config).ok());
        const char* port = std::getenv(kTpuProcessPortEnvVar);
        ASSERT_NE(port, nullptr);
        EXPECT_STREQ(port, "1234");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitializeDistributedEnvironment_FailsOnEmptySlicebuilderAddresses) {
  EXPECT_EXIT(
      {
        DistributedWorkerConfiguration config;
        config.rank = 0;
        config.local_rank = 0;
        config.sb_port = "1234";
        config.sb_addrs = "";
        config.topology = "1,1,1";

        EXPECT_FALSE(InitializeDistributedEnvironment(config).ok());
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitializeDistributedEnvironment_FailsOnNegativeRank) {
  EXPECT_EXIT(
      {
        DistributedWorkerConfiguration config;
        config.rank = -1;
        config.local_rank = 0;
        config.sb_port = "1234";
        config.sb_addrs = "localhost:1234";
        config.topology = "1,1,1";

        EXPECT_FALSE(InitializeDistributedEnvironment(config).ok());
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitializeDistributedEnvironment_OverridesXlaBarrierFlags) {
  EXPECT_EXIT(
      {
        setenv(kLibtpuInitArgsEnvVar,
               "--xla_tpu_use_enhanced_launch_barrier=true", 1);
        DistributedWorkerConfiguration config;
        config.rank = 0;
        config.local_rank = 0;
        config.sb_port = "1234";
        config.sb_addrs = "localhost:1234";
        config.topology = "1,1,1";

        EXPECT_TRUE(InitializeDistributedEnvironment(config).ok());
        const char* val = std::getenv(kLibtpuInitArgsEnvVar);
        ASSERT_NE(val, nullptr);
        EXPECT_NE(std::string(val).find(
                      "--xla_tpu_use_enhanced_launch_barrier=false"),
                  std::string::npos);
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitializeDistributedEnvironment_SetsAllowMultipleLibtpuLoad) {
  EXPECT_EXIT(
      {
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

        EXPECT_TRUE(InitializeDistributedEnvironment(config).ok());
        EXPECT_STREQ(std::getenv(kAllowMultipleLibtpuLoadEnvVar), "1");
        EXPECT_STREQ(std::getenv(kTpuProcessAddressesEnvVar), "localhost:1234");
        EXPECT_STREQ(std::getenv(kTpuHostBoundsEnvVar), "1,1,1");
        EXPECT_STREQ(std::getenv(kTpuChipsPerHostBoundsEnvVar), "1,1,1");
        EXPECT_STREQ(std::getenv(kTpuProcessBoundsEnvVar), "1,1,1");
        EXPECT_STREQ(std::getenv(kTpuChipsPerProcessBoundsEnvVar), "1,1,1");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitializeDistributedEnvironmentSetsChipsBoundsFor4DTopology) {
  EXPECT_EXIT(
      {
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

        EXPECT_TRUE(InitializeDistributedEnvironment(config).ok());
        EXPECT_STREQ(std::getenv(kTpuChipsPerHostBoundsEnvVar), "1,1,1,1");
        EXPECT_STREQ(std::getenv(kTpuChipsPerProcessBoundsEnvVar), "1,1,1,1");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitializeDistributedEnvironmentUnsetsAddressesFirst) {
  EXPECT_EXIT(
      {
        setenv(kTpuProcessAddressesEnvVar, "old_address", 1);
        DistributedWorkerConfiguration config;
        config.rank = 0;
        config.local_rank = 0;
        config.sb_port = "1234";
        config.sb_addrs = "new_address";
        config.topology = "1,1,1";

        EXPECT_TRUE(InitializeDistributedEnvironment(config).ok());
        EXPECT_STREQ(std::getenv(kTpuProcessAddressesEnvVar), "new_address");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitializeDistributedEnvironmentSetsTaskIdUsingSliceRankForMultislice) {
  EXPECT_EXIT(
      {
        unsetenv(kAllowMultipleLibtpuLoadEnvVar);
        unsetenv(kTpuProcessAddressesEnvVar);
        unsetenv(kTpuHostBoundsEnvVar);
        unsetenv(kTpuChipsPerHostBoundsEnvVar);
        unsetenv(kTpuProcessBoundsEnvVar);
        unsetenv(kTpuChipsPerProcessBoundsEnvVar);
        unsetenv(kCloudTpuTaskIdEnvVar);

        DistributedWorkerConfiguration config;
        config.rank = 5;
        config.local_rank = 0;
        config.sb_port = "1234";
        config.sb_addrs = "host0:1234,host1:1234";
        config.topology = "1,1,1";

        EXPECT_TRUE(InitializeDistributedEnvironment(config).ok());
        EXPECT_STREQ(std::getenv(kCloudTpuTaskIdEnvVar), "1");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitDistEnv_FallsBackOnMalformedTpuVisibleDevicesEnvVar) {
  EXPECT_EXIT(
      {
        setenv(kTpuVisibleDevicesEnvVar, "invalid_dev", 1);
        DistributedWorkerConfiguration config;
        config.rank = 0;
        config.local_rank = 1;
        config.sb_port = "1234";
        config.sb_addrs = "localhost:1234";
        config.topology = "1,1,1";
        EXPECT_TRUE(InitializeDistributedEnvironment(config).ok());
        const char* val = std::getenv(kTpuVisibleDevicesEnvVar);
        EXPECT_NE(val, nullptr);
        EXPECT_STREQ(val, "1");
        const char* chips_val = std::getenv(kTpuVisibleChipsEnvVar);
        EXPECT_NE(chips_val, nullptr);
        EXPECT_STREQ(chips_val, "1");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitDistEnv_FallsBackOnNegativeTpuVisibleDevicesEnvVar) {
  EXPECT_EXIT(
      {
        setenv(kTpuVisibleDevicesEnvVar, "-5", 1);
        DistributedWorkerConfiguration config;
        config.rank = 0;
        config.local_rank = 2;
        config.sb_port = "1234";
        config.sb_addrs = "localhost:1234";
        config.topology = "1,1,1";
        EXPECT_TRUE(InitializeDistributedEnvironment(config).ok());
        const char* val = std::getenv(kTpuVisibleDevicesEnvVar);
        EXPECT_NE(val, nullptr);
        EXPECT_STREQ(val, "2");
        const char* chips_val = std::getenv(kTpuVisibleChipsEnvVar);
        EXPECT_NE(chips_val, nullptr);
        EXPECT_STREQ(chips_val, "2");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitDistEnv_RespectsExistingSingleTpuVisibleDevicesEnvVar) {
  EXPECT_EXIT(
      {
        setenv(kTpuVisibleDevicesEnvVar, "3", 1);
        DistributedWorkerConfiguration config;
        config.rank = 0;
        config.local_rank = 0;
        config.sb_port = "1234";
        config.sb_addrs = "localhost:1234";
        config.topology = "1,1,1";

        EXPECT_TRUE(InitializeDistributedEnvironment(config).ok());
        EXPECT_STREQ(std::getenv(kTpuVisibleDevicesEnvVar), "3");
        EXPECT_STREQ(std::getenv(kTpuVisibleChipsEnvVar), "3");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitDistEnv_OverwritesMultiDeviceTpuVisibleDevicesEnvVar) {
  EXPECT_EXIT(
      {
        setenv(kTpuVisibleDevicesEnvVar, "0,1,2,3", 1);
        DistributedWorkerConfiguration config;
        config.rank = 0;
        config.local_rank = 1;
        config.sb_port = "1234";
        config.sb_addrs = "localhost:1234";
        config.topology = "1,1,1";

        EXPECT_TRUE(InitializeDistributedEnvironment(config).ok());
        EXPECT_STREQ(std::getenv(kTpuVisibleDevicesEnvVar), "1");
        EXPECT_STREQ(std::getenv(kTpuVisibleChipsEnvVar), "1");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitDistEnv_OverwritesExistingTpuVisibleChipsEnvVar) {
  EXPECT_EXIT(
      {
        setenv(kTpuVisibleChipsEnvVar, "0,1,2,3", 1);
        unsetenv(kTpuVisibleDevicesEnvVar);
        DistributedWorkerConfiguration config;
        config.rank = 0;
        config.local_rank = 0;
        config.sb_port = "1234";
        config.sb_addrs = "localhost:1234";
        config.topology = "1,1,1";

        EXPECT_TRUE(InitializeDistributedEnvironment(config).ok());
        EXPECT_STREQ(std::getenv(kTpuVisibleDevicesEnvVar), "0");
        EXPECT_STREQ(std::getenv(kTpuVisibleChipsEnvVar), "0");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitSingleDevEnv_RespectsExistingSingleTpuVisibleDevicesEnvVar) {
  EXPECT_EXIT(
      {
        setenv(kTpuVisibleDevicesEnvVar, "3", 1);
        EXPECT_TRUE(InitializeSingleDeviceEnvironment().ok());
        EXPECT_STREQ(std::getenv(kTpuVisibleDevicesEnvVar), "3");
        EXPECT_STREQ(std::getenv(kTpuVisibleChipsEnvVar), "3");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitSingleDevEnv_OverwritesMultiDeviceTpuVisibleDevicesEnvVar) {
  EXPECT_EXIT(
      {
        setenv(kTpuVisibleDevicesEnvVar, "0,1", 1);
        EXPECT_TRUE(InitializeSingleDeviceEnvironment().ok());
        EXPECT_STREQ(std::getenv(kTpuVisibleDevicesEnvVar), "0");
        EXPECT_STREQ(std::getenv(kTpuVisibleChipsEnvVar), "0");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitSingleDevEnv_FallsBackOnMalformedTpuVisibleDevicesEnvVar) {
  EXPECT_EXIT(
      {
        setenv(kTpuVisibleDevicesEnvVar, "invalid_dev", 1);
        EXPECT_TRUE(InitializeSingleDeviceEnvironment().ok());
        EXPECT_STREQ(std::getenv(kTpuVisibleDevicesEnvVar), "0");
        EXPECT_STREQ(std::getenv(kTpuVisibleChipsEnvVar), "0");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitSingleDevEnv_OverwritesExistingTpuVisibleChipsEnvVar) {
  EXPECT_EXIT(
      {
        setenv(kTpuVisibleChipsEnvVar, "2", 1);
        unsetenv(kTpuVisibleDevicesEnvVar);
        EXPECT_TRUE(InitializeSingleDeviceEnvironment().ok());
        EXPECT_STREQ(std::getenv(kTpuVisibleDevicesEnvVar), "0");
        EXPECT_STREQ(std::getenv(kTpuVisibleChipsEnvVar), "0");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

}  // namespace
}  // namespace torch_tpu
