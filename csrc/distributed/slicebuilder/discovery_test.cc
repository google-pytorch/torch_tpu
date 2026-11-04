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

#include "csrc/distributed/slicebuilder/discovery.h"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <string>

#include "absl/log/absl_log.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

using absl_testing::StatusIs;
using testing::AllOf;
using testing::ExitedWithCode;
using testing::HasSubstr;
using testing::Not;

TEST(DiscoveryDeathTest, GetDistributedWorkerConfigurationMissingAllVars) {
  EXPECT_EXIT(
      {
        unsetenv(kRankEnvVar);
        unsetenv(kLocalRankEnvVar);
        unsetenv(kWorldSizeEnvVar);
        unsetenv(kMasterAddrEnvVar);
        unsetenv(kMasterPortEnvVar);
        unsetenv(kTorchTpuSlicebuilderAddressesEnvVar);
        unsetenv(kTorchTpuTopologyEnvVar);

        EXPECT_THAT(
            GetDistributedWorkerConfiguration().status(),
            StatusIs(error::kFailedPrecondition,
                     AllOf(HasSubstr("missing required environment variables "
                                     "for distributed training: RANK, "
                                     "LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, "
                                     "MASTER_PORT, "
                                     "TORCH_TPU_SLICEBUILDER_ADDRESSES, "
                                     "TORCH_TPU_TOPOLOGY"),
                           HasSubstr(
                               "properly and call "
                               "torch.tpu.distributed.set_tpu_launch_env()"))));
        _exit(testing::Test::HasFailure() ? 1 : 0);
      },
      ExitedWithCode(0), "");
}

TEST(DiscoveryDeathTest, GetDistributedWorkerConfigurationMissingTorchrunVars) {
  EXPECT_EXIT(
      {
        unsetenv(kRankEnvVar);
        unsetenv(kLocalRankEnvVar);
        unsetenv(kWorldSizeEnvVar);
        unsetenv(kMasterAddrEnvVar);
        unsetenv(kMasterPortEnvVar);
        setenv(kTorchTpuSlicebuilderAddressesEnvVar, "host0:54321,host1:54322",
               1);
        setenv(kTorchTpuTopologyEnvVar, "1x1x1", 1);

        EXPECT_THAT(
            GetDistributedWorkerConfiguration().status(),
            StatusIs(error::kFailedPrecondition,
                     AllOf(HasSubstr("missing required environment variables "
                                     "for distributed training: RANK, "
                                     "LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, "
                                     "MASTER_PORT; please run the program via "
                                     "torchrun"),
                           Not(HasSubstr(kTorchTpuSlicebuilderAddressesEnvVar)),
                           Not(HasSubstr("set_tpu_launch_env")))));
        _exit(testing::Test::HasFailure() ? 1 : 0);
      },
      ExitedWithCode(0), "");
}

TEST(DiscoveryDeathTest, GetDistributedWorkerConfigurationMissingTpuVars) {
  EXPECT_EXIT(
      {
        setenv(kRankEnvVar, "0", 1);
        setenv(kLocalRankEnvVar, "0", 1);
        setenv(kWorldSizeEnvVar, "2", 1);
        setenv(kMasterAddrEnvVar, "localhost", 1);
        setenv(kMasterPortEnvVar, "12345", 1);
        unsetenv(kTorchTpuSlicebuilderAddressesEnvVar);
        unsetenv(kTorchTpuTopologyEnvVar);

        EXPECT_THAT(
            GetDistributedWorkerConfiguration().status(),
            StatusIs(error::kFailedPrecondition,
                     AllOf(HasSubstr("missing required environment variables "
                                     "for distributed training: "
                                     "TORCH_TPU_SLICEBUILDER_ADDRESSES, "
                                     "TORCH_TPU_TOPOLOGY; please call "
                                     "torch.tpu.distributed."
                                     "set_tpu_launch_env()"),
                           Not(HasSubstr("torchrun")))));
        _exit(testing::Test::HasFailure() ? 1 : 0);
      },
      ExitedWithCode(0), "");
}

TEST(DiscoveryDeathTest, GetDistributedWorkerConfiguration) {
  EXPECT_EXIT(
      {
        setenv(kRankEnvVar, "0", 1);
        setenv(kLocalRankEnvVar, "0", 1);
        setenv(kWorldSizeEnvVar, "2", 1);
        setenv(kMasterAddrEnvVar, "localhost", 1);
        setenv(kMasterPortEnvVar, "12345", 1);
        setenv(kTorchTpuSlicebuilderAddressesEnvVar, "host0:54321,host1:54322",
               1);
        setenv(kTorchTpuTopologyEnvVar, "1x1x1", 1);

        const absl::StatusOr<DistributedWorkerConfiguration> config =
            GetDistributedWorkerConfiguration();

        if (!config.ok()) {
          ABSL_LOG(ERROR) << "Expected success, but got failure: "
                          << config.status();
          _exit(1);
        }
        if (config->rank != 0 || config->sb_port != "54321" ||
            config->sb_addrs != "host0:54321,host1:54322") {
          ABSL_LOG(ERROR) << "Unexpected config values.";
          _exit(2);
        }
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(DiscoveryDeathTest, GetDistributedWorkerConfigurationMultislice) {
  EXPECT_EXIT(
      {
        setenv(kRankEnvVar, "3", 1);
        setenv(kLocalRankEnvVar, "0", 1);
        setenv(kWorldSizeEnvVar, "4", 1);
        setenv(kMasterAddrEnvVar, "localhost", 1);
        setenv(kMasterPortEnvVar, "12345", 1);
        setenv(kTorchTpuSlicebuilderAddressesEnvVar, "host0:54321,host1:54322",
               1);
        setenv(kTorchTpuTopologyEnvVar, "1x1x1", 1);

        const absl::StatusOr<DistributedWorkerConfiguration> config =
            GetDistributedWorkerConfiguration();

        if (!config.ok()) {
          ABSL_LOG(ERROR) << "Expected success, but got failure: "
                          << config.status();
          _exit(1);
        }
        if (config->rank != 3 || config->sb_port != "54322" ||
            config->sb_addrs != "host0:54321,host1:54322") {
          ABSL_LOG(ERROR) << "Unexpected config values.";
          _exit(2);
        }
        _exit(0);
      },
      ExitedWithCode(0), "");
}

TEST(DiscoveryDeathTest, GetDistributedWorkerConfigurationIPv6) {
  EXPECT_EXIT(
      {
        setenv(kRankEnvVar, "0", 1);
        setenv(kLocalRankEnvVar, "0", 1);
        setenv(kWorldSizeEnvVar, "2", 1);
        setenv(kMasterAddrEnvVar, "::1", 1);
        setenv(kMasterPortEnvVar, "12345", 1);
        setenv(kTorchTpuSlicebuilderAddressesEnvVar,
               "[2002:a00:1::1]:54321,[2002:a00:1::2]:54322", 1);
        setenv(kTorchTpuTopologyEnvVar, "1x1x1", 1);

        const absl::StatusOr<DistributedWorkerConfiguration> config_or =
            GetDistributedWorkerConfiguration();

        if (!config_or.ok()) {
          ABSL_LOG(ERROR) << "Expected success, but got failure: "
                          << config_or.status();
          _exit(1);
        }
        if (config_or->rank != 0 || config_or->sb_port != "54321" ||
            config_or->sb_addrs !=
                "[2002:a00:1::1]:54321,[2002:a00:1::2]:54322") {
          ABSL_LOG(ERROR) << "Unexpected config values.";
          _exit(2);
        }
        _exit(0);
      },
      ExitedWithCode(0), "");
}

}  // namespace
}  // namespace torch_tpu
