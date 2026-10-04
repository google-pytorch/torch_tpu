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

#include <cstddef>
#include <string>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"

namespace torch_tpu {

static absl::Status ValidateRequiredDistributedEnvVars() {
  std::vector<std::string> missing_vars;
  if (!GetEnvOnce<kRankEnvVar>().has_value()) {
    missing_vars.push_back(kRankEnvVar);
  }
  if (!GetEnvOnce<kLocalRankEnvVar>().has_value()) {
    missing_vars.push_back(kLocalRankEnvVar);
  }
  if (!GetEnvOnce<kWorldSizeEnvVar>().has_value()) {
    missing_vars.push_back(kWorldSizeEnvVar);
  }
  if (!GetEnvOnce<kMasterAddrEnvVar>().has_value()) {
    missing_vars.push_back(kMasterAddrEnvVar);
  }
  if (!GetEnvOnce<kMasterPortEnvVar>().has_value()) {
    missing_vars.push_back(kMasterPortEnvVar);
  }
  if (!GetEnvOnce<kTorchTpuSlicebuilderAddressesEnvVar>().has_value()) {
    missing_vars.push_back(kTorchTpuSlicebuilderAddressesEnvVar);
  }
  if (!GetEnvOnce<kTorchTpuTopologyEnvVar>().has_value()) {
    missing_vars.push_back(kTorchTpuTopologyEnvVar);
  }

  TT_RET_CHECK(missing_vars.empty(), error::kFailedPrecondition)
      << "missing required environment variables for distributed training: "
      << absl::StrJoin(missing_vars, ", ") << "; "
      << "please run the program via torchrun or similar tools so that "
         "the environment is set up properly";
  return absl::OkStatus();
}

absl::StatusOr<DistributedWorkerConfiguration>
GetDistributedWorkerConfiguration() {
  TT_RETURN_IF_ERROR(ValidateRequiredDistributedEnvVars());

  TT_ASSIGN_OR_RETURN(const int rank,
                      (GetRequiredIntegerEnvOnce<int, kRankEnvVar>()));
  TT_ASSIGN_OR_RETURN(const int local_rank,
                      (GetRequiredIntegerEnvOnce<int, kLocalRankEnvVar>()));

  // Get the master address and port from the environment variables.
  std::string master_addr;
  TT_ASSIGN_OR_RETURN(master_addr, GetRequiredEnvOnce<kMasterAddrEnvVar>());
  TT_ASSIGN_OR_RETURN(const int master_port,
                      (GetRequiredIntegerEnvOnce<int, kMasterPortEnvVar>()));
  (void)master_port;  // VOID_CAST_OK=master_port is validated but not used in
                      // DistributedWorkerConfiguration.

  // Get the world size from the environment variables.
  TT_ASSIGN_OR_RETURN(const int world_size,
                      (GetRequiredIntegerEnvOnce<int, kWorldSizeEnvVar>()));

  // Get the slice builder addresses from the environment variables.
  TT_ASSIGN_OR_RETURN(
      std::string sb_addrs,
      GetRequiredEnvOnce<kTorchTpuSlicebuilderAddressesEnvVar>());

  if (sb_addrs.empty()) {
    return TT_ERROR(error::kFailedPrecondition)
           << kTorchTpuSlicebuilderAddressesEnvVar << " is empty.";
  }
  if (rank < 0) {
    return TT_ERROR(error::kFailedPrecondition)
           << "RANK " << rank << " is out of bounds (negative)";
  }
  std::vector<std::string> addresses = absl::StrSplit(sb_addrs, ',');
  int slice_rank = rank % static_cast<int>(addresses.size());

  // Get the local port (use rfind(':') so IPv6 addresses work).
  std::string my_addr = addresses[slice_rank];
  const size_t colon_pos = my_addr.rfind(':');
  if (colon_pos == std::string::npos || colon_pos == 0 ||
      colon_pos + 1 >= my_addr.size()) {
    return TT_ERROR(error::kFailedPrecondition)
           << "Invalid address format in "
           << kTorchTpuSlicebuilderAddressesEnvVar
           << " for current rank: " << my_addr;
  }
  std::string sb_port = my_addr.substr(colon_pos + 1);

  // Get the topology from the environment variables.
  TT_ASSIGN_OR_RETURN(std::string topology,
                      GetRequiredEnvOnce<kTorchTpuTopologyEnvVar>());

  auto distributed_worker_config = DistributedWorkerConfiguration{
      .rank = rank,
      .local_rank = local_rank,
      .world_size = world_size,
      .sb_addrs = sb_addrs,
      .sb_port = sb_port,
      .topology = topology,
  };

  ABSL_LOG(INFO) << "DistributedWorkerConfiguration: "
                 << distributed_worker_config.rank << " "
                 << distributed_worker_config.local_rank << " "
                 << distributed_worker_config.world_size << " "
                 << distributed_worker_config.sb_addrs << " "
                 << distributed_worker_config.sb_port << " "
                 << distributed_worker_config.topology;

  return distributed_worker_config;
}

}  // namespace torch_tpu
