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

#include <ftw.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <string_view>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "csrc/common/env_vars.h"
#include "csrc/distributed/slicebuilder/discovery.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

using testing::ExitedWithCode;

class ScopedFakePciDir {
 public:
  ScopedFakePciDir() {
    std::string tmpl = absl::StrCat(testing::TempDir(), "/fake_pci_XXXXXX");
    char* const created = mkdtemp(tmpl.data());
    ABSL_CHECK(created != nullptr);
    dir_ = created;
  }

  ~ScopedFakePciDir() {
    nftw(
        dir_.c_str(),
        [](const char* fpath, const struct stat*, int, struct FTW*) {
          return remove(fpath);
        },
        /*nopenfd=*/16, FTW_DEPTH | FTW_PHYS);
  }

  ScopedFakePciDir(const ScopedFakePciDir&) = delete;
  ScopedFakePciDir& operator=(const ScopedFakePciDir&) = delete;

  const std::string& path() const { return dir_; }

  void AddTpuDevice(std::string_view slot, std::string_view device_id) {
    const std::string dev_dir = absl::StrCat(dir_, "/", slot);
    ABSL_CHECK_EQ(mkdir(dev_dir.c_str(), 0700), 0);
    std::ofstream(absl::StrCat(dev_dir, "/vendor")) << "0x1ae0\n";
    std::ofstream(absl::StrCat(dev_dir, "/device")) << device_id << "\n";
  }

 private:
  std::string dir_;
};

// Returns a configuration of a single-process job with a fake slice builder.
[[nodiscard]] DistributedWorkerConfiguration GetDefaultDistWorkerConfig() {
  return {
      .rank = 0,
      .local_rank = 0,
      .world_size = 0,
      .sb_addrs = "localhost:1234",
      .sb_port = "1234",
      .topology = "1,1,1",
  };
}

TEST(EnvironmentDeathTest,
     InitializeDistributedEnvironment_PopulatesSlicebuilderAddresses) {
  EXPECT_EXIT(
      {
        unsetenv(kTpuProcessAddressesEnvVar);
        unsetenv(kTpuProcessPortEnvVar);
        DistributedWorkerConfiguration config = GetDefaultDistWorkerConfig();
        config.sb_addrs = "host0:1234,host1:1234";

        ABSL_CHECK_OK(InitializeDistributedEnvironment(config));
        const char* addrs = std::getenv(kTpuProcessAddressesEnvVar);
        ABSL_CHECK_NE(addrs, nullptr);
        ABSL_CHECK_STREQ(addrs, "host0:1234,host1:1234");
        const char* port = std::getenv(kTpuProcessPortEnvVar);
        ABSL_CHECK_NE(port, nullptr);
        ABSL_CHECK_STREQ(port, "1234");
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
        DistributedWorkerConfiguration config = GetDefaultDistWorkerConfig();
        config.sb_addrs = "host0:5678";
        config.sb_port = "5678";

        ABSL_CHECK_OK(InitializeDistributedEnvironment(config));
        const char* addrs = std::getenv(kTpuProcessAddressesEnvVar);
        ABSL_CHECK_NE(addrs, nullptr);
        ABSL_CHECK_STREQ(addrs, "host0:5678");
        const char* port = std::getenv(kTpuProcessPortEnvVar);
        ABSL_CHECK_NE(port, nullptr);
        ABSL_CHECK_STREQ(port, "5678");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest, InitializeDistributedEnvironment_ResolvesBnsPort) {
  EXPECT_EXIT(
      {
        unsetenv(kTpuProcessPortEnvVar);
        ABSL_CHECK_OK(
            InitializeDistributedEnvironment(GetDefaultDistWorkerConfig()));
        const char* port = std::getenv(kTpuProcessPortEnvVar);
        ABSL_CHECK_NE(port, nullptr);
        ABSL_CHECK_STREQ(port, "1234");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitializeDistributedEnvironment_FailsOnEmptySlicebuilderAddresses) {
  EXPECT_EXIT(
      {
        DistributedWorkerConfiguration config = GetDefaultDistWorkerConfig();
        config.sb_addrs = "";

        ABSL_CHECK(!InitializeDistributedEnvironment(config).ok());
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitializeDistributedEnvironment_FailsOnNegativeRank) {
  EXPECT_EXIT(
      {
        DistributedWorkerConfiguration config = GetDefaultDistWorkerConfig();
        config.rank = -1;

        ABSL_CHECK(!InitializeDistributedEnvironment(config).ok());
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
        ABSL_CHECK_OK(
            InitializeDistributedEnvironment(GetDefaultDistWorkerConfig()));
        const char* val = std::getenv(kLibtpuInitArgsEnvVar);
        ABSL_CHECK_NE(val, nullptr);
        ABSL_CHECK_STREQ(val,
                         "--xla_tpu_use_enhanced_launch_barrier=false "
                         "--slicebuilder_use_insecure_grpc=true "
                         "--undefok=slicebuilder_use_insecure_grpc");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

// Runs InitializeDistributedEnvironment(GetDefaultDistWorkerConfig()) with
// `LIBTPU_INIT_ARGS` set to `libtpu_init_args` in a death test child and
// prints the resulting `LIBTPU_INIT_ARGS` to stderr as
// "libtpu_init_args=<value>" for EXPECT_EXIT to match.
void InitAndPrintLibtpuInitArgs(const char* const libtpu_init_args) {
  setenv(kLibtpuInitArgsEnvVar, libtpu_init_args, 1);
  ABSL_CHECK_OK(InitializeDistributedEnvironment(GetDefaultDistWorkerConfig()));
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
        ABSL_CHECK_OK(
            InitializeDistributedEnvironment(GetDefaultDistWorkerConfig()));
        ABSL_CHECK_STREQ(std::getenv(kAllowMultipleLibtpuLoadEnvVar), "1");
        ABSL_CHECK_STREQ(std::getenv(kTpuProcessAddressesEnvVar),
                         "localhost:1234");
        ABSL_CHECK_STREQ(std::getenv(kTpuHostBoundsEnvVar), "1,1,1");
        ABSL_CHECK_STREQ(std::getenv(kTpuChipsPerHostBoundsEnvVar), "1,1,1");
        ABSL_CHECK_STREQ(std::getenv(kTpuProcessBoundsEnvVar), "1,1,1");
        ABSL_CHECK_STREQ(std::getenv(kTpuChipsPerProcessBoundsEnvVar), "1,1,1");
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
        DistributedWorkerConfiguration config = GetDefaultDistWorkerConfig();
        config.topology = "2,2,1,2";

        ABSL_CHECK_OK(InitializeDistributedEnvironment(config));
        ABSL_CHECK_STREQ(std::getenv(kTpuChipsPerHostBoundsEnvVar), "1,1,1,1");
        ABSL_CHECK_STREQ(std::getenv(kTpuChipsPerProcessBoundsEnvVar),
                         "1,1,1,1");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitializeDistributedEnvironmentUnsetsAddressesFirst) {
  EXPECT_EXIT(
      {
        setenv(kTpuProcessAddressesEnvVar, "old_address", 1);
        DistributedWorkerConfiguration config = GetDefaultDistWorkerConfig();
        config.sb_addrs = "new_address";

        ABSL_CHECK_OK(InitializeDistributedEnvironment(config));
        ABSL_CHECK_STREQ(std::getenv(kTpuProcessAddressesEnvVar),
                         "new_address");
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

        DistributedWorkerConfiguration config = GetDefaultDistWorkerConfig();
        config.rank = 5;
        config.sb_addrs = "host0:1234,host1:1234";

        ABSL_CHECK_OK(InitializeDistributedEnvironment(config));
        ABSL_CHECK_STREQ(std::getenv(kCloudTpuTaskIdEnvVar), "1");
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
[[nodiscard]] DistributedWorkerConfiguration MultisliceConfig(
    const int local_rank) {
  return {
      .rank = 8 + local_rank,
      .local_rank = local_rank,
      .world_size = 16,
      .sb_addrs = kTwoHostSliceAddresses,
      .sb_port = "8070",
      .topology = "1,1,1",
  };
}

// Runs InitializeDistributedEnvironment(config) in a death test child and
// prints the resulting MegaScale ports to stderr as
// "ports=<MEGASCALE_PORT>,<MEGASCALE_DEBUG_PORT>" for EXPECT_EXIT to match.
void InitAndPrintMegascalePorts(const DistributedWorkerConfiguration& config) {
  ABSL_CHECK_OK(InitializeDistributedEnvironment(config));
  const char* const port = std::getenv(kMegascalePortEnvVar);
  const char* const debug_port = std::getenv(kMegascaleDebugPortEnvVar);
  std::fprintf(stderr, "ports=%s,%s\n", port ? port : "unset",
               debug_port ? debug_port : "unset");
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
  ABSL_CHECK_OK(InitializeDistributedEnvironment(config));
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
     InitDistEnv_FailsOnMalformedTpuVisibleDevicesEnvVar) {
  EXPECT_EXIT(
      {
        setenv(kTpuVisibleDevicesEnvVar, "invalid_dev", 1);
        DistributedWorkerConfiguration config = GetDefaultDistWorkerConfig();
        config.local_rank = 1;
        ABSL_CHECK(
            absl::IsInvalidArgument(InitializeDistributedEnvironment(config)));
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest, InitDistEnv_FailsOnNegativeTpuVisibleDevicesEnvVar) {
  EXPECT_EXIT(
      {
        setenv(kTpuVisibleDevicesEnvVar, "-5", 1);
        DistributedWorkerConfiguration config = GetDefaultDistWorkerConfig();
        config.local_rank = 2;
        ABSL_CHECK(
            absl::IsInvalidArgument(InitializeDistributedEnvironment(config)));
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitDistEnv_RespectsExistingSingleTpuVisibleDevicesEnvVar) {
  EXPECT_EXIT(
      {
        setenv(kTpuVisibleDevicesEnvVar, "3", 1);
        ABSL_CHECK_OK(
            InitializeDistributedEnvironment(GetDefaultDistWorkerConfig()));
        ABSL_CHECK_STREQ(std::getenv(kTpuVisibleDevicesEnvVar), "3");
        ABSL_CHECK_STREQ(std::getenv(kTpuVisibleChipsEnvVar), "3");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitDistEnv_OverwritesMultiDeviceTpuVisibleDevicesEnvVar) {
  EXPECT_EXIT(
      {
        setenv(kTpuVisibleDevicesEnvVar, "4,5,6,7", 1);
        DistributedWorkerConfiguration config = GetDefaultDistWorkerConfig();
        config.local_rank = 1;

        ABSL_CHECK_OK(InitializeDistributedEnvironment(config));
        ABSL_CHECK_STREQ(std::getenv(kTpuVisibleDevicesEnvVar), "5");
        ABSL_CHECK_STREQ(std::getenv(kTpuVisibleChipsEnvVar), "5");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitDistEnv_FailsIfTooFewDevicesInTpuVisibleDevicesEnvVar) {
  EXPECT_EXIT(
      {
        setenv(kTpuVisibleDevicesEnvVar, "4,5", 1);
        DistributedWorkerConfiguration config = GetDefaultDistWorkerConfig();
        config.local_rank = 2;
        ABSL_CHECK(
            absl::IsInvalidArgument(InitializeDistributedEnvironment(config)));
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
        ABSL_CHECK_OK(
            InitializeDistributedEnvironment(GetDefaultDistWorkerConfig()));
        ABSL_CHECK_STREQ(std::getenv(kTpuVisibleDevicesEnvVar), "0");
        ABSL_CHECK_STREQ(std::getenv(kTpuVisibleChipsEnvVar), "0");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitSingleDevEnv_RespectsExistingSingleTpuVisibleDevicesEnvVar) {
  ScopedFakePciDir pci;
  EXPECT_EXIT(
      {
        unsetenv(kAllowMultipleLibtpuLoadEnvVar);
        unsetenv(kTorchTpuTopologyEnvVar);
        unsetenv(kTpuHostBoundsEnvVar);
        unsetenv(kTpuChipsPerHostBoundsEnvVar);
        unsetenv(kTpuProcessBoundsEnvVar);
        unsetenv(kTpuChipsPerProcessBoundsEnvVar);
        setenv(kTpuVisibleDevicesEnvVar, "3", 1);
        ABSL_CHECK_OK(InitializeSingleDeviceEnvironment(pci.path()));
        ABSL_CHECK_STREQ(std::getenv(kTpuVisibleDevicesEnvVar), "3");
        ABSL_CHECK_STREQ(std::getenv(kTpuVisibleChipsEnvVar), "3");
        ABSL_CHECK_EQ(std::getenv(kTorchTpuTopologyEnvVar), nullptr);
        ABSL_CHECK_STREQ(std::getenv(kTpuHostBoundsEnvVar), "1,1,1");
        ABSL_CHECK_STREQ(std::getenv(kTpuChipsPerHostBoundsEnvVar), "1,1,1");
        ABSL_CHECK_STREQ(std::getenv(kTpuProcessBoundsEnvVar), "1,1,1");
        ABSL_CHECK_STREQ(std::getenv(kTpuChipsPerProcessBoundsEnvVar), "1,1,1");
        ABSL_CHECK_STREQ(std::getenv(kAllowMultipleLibtpuLoadEnvVar), "1");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitSingleDevEnv_OverwritesMultiDeviceTpuVisibleDevicesEnvVar) {
  ScopedFakePciDir pci;
  EXPECT_EXIT(
      {
        setenv(kTpuVisibleDevicesEnvVar, "4,5", 1);
        ABSL_CHECK_OK(InitializeSingleDeviceEnvironment(pci.path()));
        ABSL_CHECK_STREQ(std::getenv(kTpuVisibleDevicesEnvVar), "4");
        ABSL_CHECK_STREQ(std::getenv(kTpuVisibleChipsEnvVar), "4");
        ABSL_CHECK_STREQ(std::getenv(kTpuHostBoundsEnvVar), "1,1,1");
        ABSL_CHECK_STREQ(std::getenv(kTpuChipsPerHostBoundsEnvVar), "1,1,1");
        ABSL_CHECK_STREQ(std::getenv(kTpuProcessBoundsEnvVar), "1,1,1");
        ABSL_CHECK_STREQ(std::getenv(kTpuChipsPerProcessBoundsEnvVar), "1,1,1");
        ABSL_CHECK_STREQ(std::getenv(kAllowMultipleLibtpuLoadEnvVar), "1");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitSingleDevEnv_FailsOnMalformedTpuVisibleDevicesEnvVar) {
  ScopedFakePciDir pci;
  EXPECT_EXIT(
      {
        setenv(kTpuVisibleDevicesEnvVar, "invalid_dev", 1);
        ABSL_CHECK(absl::IsInvalidArgument(
            InitializeSingleDeviceEnvironment(pci.path())));
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest,
     InitSingleDevEnv_OverwritesExistingTpuVisibleChipsEnvVar) {
  ScopedFakePciDir pci;
  EXPECT_EXIT(
      {
        unsetenv(kAllowMultipleLibtpuLoadEnvVar);
        unsetenv(kTorchTpuTopologyEnvVar);
        unsetenv(kTpuHostBoundsEnvVar);
        unsetenv(kTpuChipsPerHostBoundsEnvVar);
        unsetenv(kTpuProcessBoundsEnvVar);
        unsetenv(kTpuChipsPerProcessBoundsEnvVar);
        setenv(kTpuVisibleChipsEnvVar, "2", 1);
        unsetenv(kTpuVisibleDevicesEnvVar);
        ABSL_CHECK_OK(InitializeSingleDeviceEnvironment(pci.path()));
        ABSL_CHECK_STREQ(std::getenv(kTpuVisibleDevicesEnvVar), "0");
        ABSL_CHECK_STREQ(std::getenv(kTpuVisibleChipsEnvVar), "0");
        ABSL_CHECK_EQ(std::getenv(kTorchTpuTopologyEnvVar), nullptr);
        ABSL_CHECK_STREQ(std::getenv(kTpuHostBoundsEnvVar), "1,1,1");
        ABSL_CHECK_STREQ(std::getenv(kTpuChipsPerHostBoundsEnvVar), "1,1,1");
        ABSL_CHECK_STREQ(std::getenv(kTpuProcessBoundsEnvVar), "1,1,1");
        ABSL_CHECK_STREQ(std::getenv(kTpuChipsPerProcessBoundsEnvVar), "1,1,1");
        ABSL_CHECK_STREQ(std::getenv(kAllowMultipleLibtpuLoadEnvVar), "1");
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(EnvironmentDeathTest, InitSingleDevEnv_SetsBoundsForAllSupportedTpuSpecs) {
  for (std::string_view device_id : {"0x0062", "0x0063", "0x006f", "0x007c"}) {
    ScopedFakePciDir pci;
    pci.AddTpuDevice("0000:00:04.0", device_id);
    EXPECT_EXIT(
        {
          ABSL_CHECK_OK(InitializeSingleDeviceEnvironment(pci.path()));
          ABSL_CHECK_STREQ(std::getenv(kTpuHostBoundsEnvVar), "1,1,1");
          ABSL_CHECK_STREQ(std::getenv(kTpuChipsPerHostBoundsEnvVar), "1,1,1");
          ABSL_CHECK_STREQ(std::getenv(kTpuProcessBoundsEnvVar), "1,1,1");
          ABSL_CHECK_STREQ(std::getenv(kTpuChipsPerProcessBoundsEnvVar),
                           "1,1,1");
          _exit(0);
        },
        ExitedWithCode(0), "")
        << device_id;
  }
  for (std::string_view device_id : {"0x0076", "0x0083"}) {
    ScopedFakePciDir pci;
    pci.AddTpuDevice("0000:00:04.0", device_id);
    EXPECT_EXIT(
        {
          ABSL_CHECK_OK(InitializeSingleDeviceEnvironment(pci.path()));
          ABSL_CHECK_STREQ(std::getenv(kTpuHostBoundsEnvVar), "1,1,1,1");
          ABSL_CHECK_STREQ(std::getenv(kTpuChipsPerHostBoundsEnvVar),
                           "1,1,1,1");
          ABSL_CHECK_STREQ(std::getenv(kTpuProcessBoundsEnvVar), "1,1,1,1");
          ABSL_CHECK_STREQ(std::getenv(kTpuChipsPerProcessBoundsEnvVar),
                           "1,1,1,1");
          _exit(0);
        },
        ExitedWithCode(0), "")
        << device_id;
  }
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
