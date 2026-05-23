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

#include "torch_tpu/distributed/slicebuilder/discovery.h"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <string>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/error_utils.h"

namespace torch_tpu {
namespace {

using testing::ExitedWithCode;

TEST(DiscoveryDeathTest, GetDistributedWorkerConfigurationMissingVars) {
  EXPECT_EXIT(
      {
        unsetenv(kRankEnvVar);
        unsetenv(kLocalRankEnvVar);
        unsetenv(kWorldSizeEnvVar);
        unsetenv(kMasterAddrEnvVar);
        unsetenv(kMasterPortEnvVar);
        unsetenv(kTpuSlicebuilderAddressesEnvVar);
        unsetenv(kTpuTopologyEnvVar);

        const absl::StatusOr<DistributedWorkerConfiguration> config_or =
            GetDistributedWorkerConfiguration();

        if (config_or.ok()) {
          ABSL_LOG(ERROR) << "Expected failure, but got success.";
          _exit(1);
        }
        if (config_or.status().code() != error::kFailedPrecondition) {
          ABSL_LOG(ERROR) << "Expected FailedPrecondition, but got: "
                          << config_or.status().code();
          _exit(2);
        }
        if (!absl::StrContains(config_or.status().message(),
                               "missing required environment variables "
                               "for distributed training")) {
          ABSL_LOG(ERROR) << "Unexpected error message: "
                          << config_or.status().message();
          _exit(3);
        }
        _exit(0);
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
        setenv(kTpuSlicebuilderAddressesEnvVar, "host0:54321,host1:54322", 1);
        setenv(kTpuTopologyEnvVar, "1x1x1", 1);

        const absl::StatusOr<DistributedWorkerConfiguration> config_or =
            GetDistributedWorkerConfiguration();

        if (!config_or.ok()) {
          ABSL_LOG(ERROR) << "Expected success, but got failure: "
                          << config_or.status();
          _exit(1);
        }
        if (config_or->rank != 0 || config_or->sb_port != "54321" ||
            config_or->sb_addrs != "host0:54321,host1:54322") {
          ABSL_LOG(ERROR) << "Unexpected config values.";
          _exit(2);
        }
        _exit(0);
      },
      ExitedWithCode(0), "");
}

}  // namespace
}  // namespace torch_tpu
