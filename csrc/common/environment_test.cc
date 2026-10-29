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

#include <cstdio>
#include <cstdlib>

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
        DistributedWorkerConfiguration config{};
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
        DistributedWorkerConfiguration config{};
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
        DistributedWorkerConfiguration config{};
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
        DistributedWorkerConfiguration config{};
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
        DistributedWorkerConfiguration config{};
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
        DistributedWorkerConfiguration config{};
        config.rank = 0;
        config.local_rank = 0;
        config.sb_port = "1234";
        config.sb_addrs = "localhost:1234";
        config.topology = "1,1,1";

        EXPECT_TRUE(InitializeDistributedEnvironment(config).ok());
        const char* val = std::getenv(kLibtpuInitArgsEnvVar);
        ASSERT_NE(val, nullptr);
        EXPECT_STREQ(val,
                     "--xla_tpu_use_enhanced_launch_barrier=false "
                     "--slicebuilder_use_insecure_grpc=true "
                     "--undefok=slicebuilder_use_insecure_grpc");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

// Returns the configuration of a single-process job whose SliceBuilder address
// is `localhost:1234`.
DistributedWorkerConfiguration LocalhostConfig() {
  DistributedWorkerConfiguration config{};
  config.rank = 0;
  config.local_rank = 0;
  config.sb_port = "1234";
  config.sb_addrs = "localhost:1234";
  config.topology = "1,1,1";
  return config;
}

// Runs InitializeDistributedEnvironment(LocalhostConfig()) with
// `LIBTPU_INIT_ARGS` set to `libtpu_init_args` in a death test child and
// prints the resulting `LIBTPU_INIT_ARGS` to stderr as
// "libtpu_init_args=<value>" for EXPECT_EXIT to match.
void InitAndPrintLibtpuInitArgs(const char* const libtpu_init_args) {
  setenv(kLibtpuInitArgsEnvVar, libtpu_init_args, 1);
  if (!InitializeDistributedEnvironment(LocalhostConfig()).ok()) _exit(1);
  const char* const args = std::getenv(kLibtpuInitArgsEnvVar);
  std::fprintf(stderr, "libtpu_init_args=%s\n", args ? args : "unset");
  _exit(0);
}

TEST(EnvironmentDeathTest,
     InitializeDistributedEnvironment_PreservesSlicebuilderInsecureGrpcFlag) {
  EXPECT_EXIT(
      InitAndPrintLibtpuInitArgs(
          "--undefok=existing_flag --slicebuilder_use_insecure_grpc=false"),
      ExitedWithCode(0),
      "libtpu_init_args=--undefok=existing_flag,slicebuilder_use_insecure_grpc "
      "--slicebuilder_use_insecure_grpc=false "
      "--xla_tpu_use_enhanced_launch_barrier=false\n");
  EXPECT_EXIT(InitAndPrintLibtpuInitArgs("--noslicebuilder_use_insecure_grpc"),
              ExitedWithCode(0),
              "libtpu_init_args=--noslicebuilder_use_insecure_grpc "
              "--xla_tpu_use_enhanced_launch_barrier=false "
              "--undefok=slicebuilder_use_insecure_grpc\n");
  EXPECT_EXIT(
      InitAndPrintLibtpuInitArgs("-slicebuilder_use_insecure_grpc=false"),
      ExitedWithCode(0),
      "libtpu_init_args=-slicebuilder_use_insecure_grpc=false "
      "--xla_tpu_use_enhanced_launch_barrier=false "
      "--undefok=slicebuilder_use_insecure_grpc\n");
  EXPECT_EXIT(InitAndPrintLibtpuInitArgs("--slicebuilder_use_insecure_grpc"),
              ExitedWithCode(0),
              "libtpu_init_args=--slicebuilder_use_insecure_grpc "
              "--xla_tpu_use_enhanced_launch_barrier=false "
              "--undefok=slicebuilder_use_insecure_grpc\n");
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
        DistributedWorkerConfiguration config{};
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
        DistributedWorkerConfiguration config{};
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
        DistributedWorkerConfiguration config{};
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

        DistributedWorkerConfiguration config{};
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

// SliceBuilder addresses of one slice made of two hosts with four processes
// each.
constexpr char kTwoHostSliceAddresses[] =
    "h0:8070,h0:8071,h0:8072,h0:8073,h1:8070,h1:8071,h1:8072,h1:8073";

// Returns the configuration of local rank `local_rank` on the first host of
// the second slice in a job of two `kTwoHostSliceAddresses` slices.
DistributedWorkerConfiguration MultisliceConfig(const int local_rank) {
  DistributedWorkerConfiguration config{};
  config.rank = 8 + local_rank;
  config.local_rank = local_rank;
  config.world_size = 16;
  config.sb_port = "8070";
  config.sb_addrs = kTwoHostSliceAddresses;
  config.topology = "1,1,1";
  return config;
}

// Runs InitializeDistributedEnvironment(config) in a death test child and
// prints the resulting MegaScale coordinator address and ports to stderr as
// "coord=<MEGASCALE_COORDINATOR_ADDRESS>
// ports=<MEGASCALE_PORT>,<MEGASCALE_DEBUG_PORT>" for EXPECT_EXIT to match.
void InitAndPrintMegascalePorts(
    const DistributedWorkerConfiguration& config,
    const char* const coordinator_address = nullptr,
    const char* const initial_megascale_port = nullptr) {
  if (coordinator_address != nullptr) {
    setenv(kMegascaleCoordinatorAddressEnvVar, coordinator_address, 1);
  } else {
    unsetenv(kMegascaleCoordinatorAddressEnvVar);
  }
  if (initial_megascale_port != nullptr) {
    setenv(kMegascalePortEnvVar, initial_megascale_port, 1);
  } else {
    unsetenv(kMegascalePortEnvVar);
  }
  unsetenv(kMegascaleDebugPortEnvVar);
  if (!InitializeDistributedEnvironment(config).ok()) _exit(1);
  const char* const coord = std::getenv(kMegascaleCoordinatorAddressEnvVar);
  const char* const port = std::getenv(kMegascalePortEnvVar);
  const char* const debug_port = std::getenv(kMegascaleDebugPortEnvVar);
  std::fprintf(stderr, "coord=%s ports=%s,%s\n", coord ? coord : "unset",
               port ? port : "unset", debug_port ? debug_port : "unset");
  _exit(0);
}

TEST(EnvironmentDeathTest,
     InitDistEnvSetsMegascalePortsPerLocalRankForMultislice) {
  EXPECT_EXIT(InitAndPrintMegascalePorts(MultisliceConfig(/*local_rank=*/3)),
              ExitedWithCode(0), "ports=8086,8087");
}

TEST(EnvironmentDeathTest,
     InitDistEnvKeepsDefaultMegascalePortsForLocalRankZero) {
  EXPECT_EXIT(InitAndPrintMegascalePorts(MultisliceConfig(/*local_rank=*/0)),
              ExitedWithCode(0), "ports=8080,8081");
}

TEST(EnvironmentDeathTest,
     InitDistEnvDerivesMegascaleBasePortFromCoordinatorAddress) {
  EXPECT_EXIT(InitAndPrintMegascalePorts(MultisliceConfig(/*local_rank=*/0),
                                         /*coordinator_address=*/"slice0:8081",
                                         /*initial_megascale_port=*/"8081"),
              ExitedWithCode(0), "coord=slice0:8081 ports=8081,8082");
  EXPECT_EXIT(InitAndPrintMegascalePorts(MultisliceConfig(/*local_rank=*/3),
                                         /*coordinator_address=*/"slice0:8081",
                                         /*initial_megascale_port=*/"8081"),
              ExitedWithCode(0), "coord=slice0:8081 ports=8087,8088");
  // When a startup hook pre-sets MEGASCALE_PORT to `base_port + local_rank`
  // (8084 for local_rank=3), the coordinator address port still takes
  // precedence as the base port.
  EXPECT_EXIT(InitAndPrintMegascalePorts(MultisliceConfig(/*local_rank=*/3),
                                         /*coordinator_address=*/"slice0:8081",
                                         /*initial_megascale_port=*/"8084"),
              ExitedWithCode(0), "coord=slice0:8081 ports=8087,8088");
  EXPECT_EXIT(
      InitAndPrintMegascalePorts(MultisliceConfig(/*local_rank=*/2),
                                 /*coordinator_address=*/"[2001:db8::1]:8081"),
      ExitedWithCode(0), "coord=\\[2001:db8::1\\]:8081 ports=8085,8086");
}

TEST(EnvironmentDeathTest,
     InitDistEnvPinsPortlessCoordinatorAddressToMegascaleBasePort) {
  EXPECT_EXIT(InitAndPrintMegascalePorts(MultisliceConfig(/*local_rank=*/0),
                                         /*coordinator_address=*/"10.8.0.5",
                                         /*initial_megascale_port=*/"8081"),
              ExitedWithCode(0), "coord=10.8.0.5:8081 ports=8081,8082");
  EXPECT_EXIT(InitAndPrintMegascalePorts(MultisliceConfig(/*local_rank=*/3),
                                         /*coordinator_address=*/"10.8.0.5",
                                         /*initial_megascale_port=*/"8081"),
              ExitedWithCode(0), "coord=10.8.0.5:8081 ports=8087,8088");
  EXPECT_EXIT(InitAndPrintMegascalePorts(MultisliceConfig(/*local_rank=*/3),
                                         /*coordinator_address=*/"slice0"),
              ExitedWithCode(0), "coord=slice0:8080 ports=8086,8087");
  EXPECT_EXIT(InitAndPrintMegascalePorts(MultisliceConfig(/*local_rank=*/0),
                                         /*coordinator_address=*/"slice0:bad",
                                         /*initial_megascale_port=*/"-1"),
              ExitedWithCode(0), "coord=slice0:bad ports=8080,8081");
}

TEST(EnvironmentDeathTest, InitDistEnvLeavesMegascalePortsUnsetForSingleSlice) {
  DistributedWorkerConfiguration config = MultisliceConfig(/*local_rank=*/3);
  config.world_size = 8;
  EXPECT_EXIT(InitAndPrintMegascalePorts(config), ExitedWithCode(0),
              "ports=unset,unset");
}

// Runs InitializeDistributedEnvironment(config) in a death test child and
// prints the resulting SKIP_MEGASCALE_PJRT_CLIENT to stderr as
// "skip=<SKIP_MEGASCALE_PJRT_CLIENT>" for EXPECT_EXIT to match.
void InitAndPrintSkipMegascale(const DistributedWorkerConfiguration& config) {
  unsetenv(kSkipMegascalePjrtClientEnvVar);
  if (!InitializeDistributedEnvironment(config).ok()) _exit(1);
  const char* const skip = std::getenv(kSkipMegascalePjrtClientEnvVar);
  std::fprintf(stderr, "skip=%s\n", skip ? skip : "unset");
  _exit(0);
}

TEST(EnvironmentDeathTest, InitDistEnvSkipsMegascaleClientForSingleSlice) {
  DistributedWorkerConfiguration config = MultisliceConfig(/*local_rank=*/3);
  config.world_size = 8;
  EXPECT_EXIT(InitAndPrintSkipMegascale(config), ExitedWithCode(0),
              "skip=true");
}

TEST(EnvironmentDeathTest, InitDistEnvKeepsMegascaleClientForMultislice) {
  EXPECT_EXIT(InitAndPrintSkipMegascale(MultisliceConfig(/*local_rank=*/3)),
              ExitedWithCode(0), "skip=unset");
}

TEST(EnvironmentDeathTest,
     InitDistEnv_FallsBackOnMalformedTpuVisibleDevicesEnvVar) {
  EXPECT_EXIT(
      {
        setenv(kTpuVisibleDevicesEnvVar, "invalid_dev", 1);
        DistributedWorkerConfiguration config{};
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
        DistributedWorkerConfiguration config{};
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
        DistributedWorkerConfiguration config{};
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
        DistributedWorkerConfiguration config{};
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
        DistributedWorkerConfiguration config{};
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

// Unsets the env vars read by HasCustomTpuDevicePaths().
void ClearTpuDevicePathEnvVars() {
  unsetenv(kTpuVisibleDevicePathsEnvVar);
  unsetenv(kLibtpuInitArgsEnvVar);
}

TEST(HasCustomTpuDevicePathsTest, ReturnsFalseByDefault) {
  ClearTpuDevicePathEnvVars();
  EXPECT_FALSE(HasCustomTpuDevicePaths());
}

TEST(HasCustomTpuDevicePathsTest, DetectsTpuVisibleDevicePaths) {
  ClearTpuDevicePathEnvVars();
  setenv(kTpuVisibleDevicePathsEnvVar, "/dev/vfio/132", /*overwrite=*/1);
  EXPECT_TRUE(HasCustomTpuDevicePaths());
  setenv(kTpuVisibleDevicePathsEnvVar, "", /*overwrite=*/1);
  EXPECT_FALSE(HasCustomTpuDevicePaths());
  ClearTpuDevicePathEnvVars();
}

TEST(HasCustomTpuDevicePathsTest, DetectsIncludedDevicePathsFlag) {
  ClearTpuDevicePathEnvVars();
  for (const char* const args : {
           "--deepsea_hal_included_devicepaths=/dev/vfio/132",
           "--xla_foo=1 -deepsea_hal_included_devicepaths=/dev/accel0",
           "--deepsea_hal_included_devicepaths /dev/vfio/132",
       }) {
    setenv(kLibtpuInitArgsEnvVar, args, /*overwrite=*/1);
    EXPECT_TRUE(HasCustomTpuDevicePaths()) << args;
  }
  for (const char* const args : {
           "",
           "--xla_foo=1",
           "--deepsea_hal_included_devicepaths=",
           "--deepsea_hal_included_devs=0",
           "--deepsea_hal_included_devicepaths_foo=1",
       }) {
    setenv(kLibtpuInitArgsEnvVar, args, /*overwrite=*/1);
    EXPECT_FALSE(HasCustomTpuDevicePaths()) << args;
  }
  ClearTpuDevicePathEnvVars();
}

}  // namespace
}  // namespace torch_tpu
