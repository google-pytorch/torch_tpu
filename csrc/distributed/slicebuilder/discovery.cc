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

#include "torch_tpu/csrc/distributed/slicebuilder/discovery.h"

#include <string>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "torch_tpu/csrc/common/env_vars.h"
#include "torch_tpu/csrc/common/error_utils.h"

namespace torch_tpu {

// Helper function to get a required environment variable or return an error.
template <const char* env_name>
const absl::StatusOr<std::string>& GetRequiredEnvOnce() {
  static const absl::NoDestructor<absl::StatusOr<std::string>> env_var(
      []() -> absl::StatusOr<std::string> {
        const auto& env_val = GetEnvOnce<env_name>();
        TT_RET_CHECK(env_val.has_value(), error::kFailedPrecondition)
            << "the " << env_name << " environment variable is not set";
        return *env_val;
      }());
  return *env_var;
}

static absl::Status CheckRequiredDistributedEnvVars() {
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
  if (!GetEnvOnce<kTpuSlicebuilderAddressesEnvVar>().has_value()) {
    missing_vars.push_back(kTpuSlicebuilderAddressesEnvVar);
  }
  if (!GetEnvOnce<kTpuTopologyEnvVar>().has_value()) {
    missing_vars.push_back(kTpuTopologyEnvVar);
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
  TT_RETURN_IF_ERROR(CheckRequiredDistributedEnvVars());

  int rank = -1;
  TT_ASSIGN_OR_RETURN(std::string env_rank, GetRequiredEnvOnce<kRankEnvVar>());
  if (!absl::SimpleAtoi(env_rank, &rank)) {
    return TT_ERROR(error::kFailedPrecondition)
           << "Failed to parse RANK: " << env_rank;
  }

  int local_rank = -1;
  TT_ASSIGN_OR_RETURN(std::string env_local_rank,
                      GetRequiredEnvOnce<kLocalRankEnvVar>());
  if (!absl::SimpleAtoi(env_local_rank, &local_rank)) {
    return TT_ERROR(error::kFailedPrecondition)
           << "Failed to parse LOCAL_RANK: " << env_local_rank;
  }

  // Get the master address and port from the environment variables.
  std::string master_addr;
  int master_port;
  TT_ASSIGN_OR_RETURN(master_addr, GetRequiredEnvOnce<kMasterAddrEnvVar>());
  TT_ASSIGN_OR_RETURN(std::string env_master_port,
                      GetRequiredEnvOnce<kMasterPortEnvVar>());
  if (!absl::SimpleAtoi(env_master_port, &master_port)) {
    return TT_ERROR(error::kFailedPrecondition)
           << "Failed to parse MASTER_PORT: " << env_master_port;
  }

  // Get the world size from the environment variables.
  TT_ASSIGN_OR_RETURN(const int world_size, GetWorldSizeFromEnvOnce());

  // Get the slice builder addresses from the environment variables.
  TT_ASSIGN_OR_RETURN(std::string sb_addrs,
                      GetRequiredEnvOnce<kTpuSlicebuilderAddressesEnvVar>());

  if (sb_addrs.empty()) {
    return TT_ERROR(error::kFailedPrecondition)
           << kTpuSlicebuilderAddressesEnvVar << " is empty.";
  }
  if (rank < 0) {
    return TT_ERROR(error::kFailedPrecondition)
           << "RANK " << rank << " is out of bounds (negative)";
  }
  std::vector<std::string> addresses = absl::StrSplit(sb_addrs, ',');
  int slice_rank = rank % static_cast<int>(addresses.size());

  // Get the local port.
  std::string my_addr = addresses[slice_rank];
  std::vector<std::string> my_parts = absl::StrSplit(my_addr, ':');
  if (my_parts.size() != 2) {
    return TT_ERROR(error::kFailedPrecondition)
           << "Invalid address format in " << kTpuSlicebuilderAddressesEnvVar
           << " for current rank: " << my_addr;
  }
  std::string sb_port = my_parts[1];

  // Get the topology from the environment variables.
  TT_ASSIGN_OR_RETURN(std::string topology,
                      GetRequiredEnvOnce<kTpuTopologyEnvVar>());

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

const absl::StatusOr<int>& GetWorldSizeFromEnvOnce() {
  static const absl::NoDestructor<absl::StatusOr<int>> world_size(
      []() -> absl::StatusOr<int> {
        const auto& env_world_size = GetEnvOnce<kWorldSizeEnvVar>();
        TT_RET_CHECK(env_world_size.has_value(), error::kFailedPrecondition)
            << "the " << kWorldSizeEnvVar << " environment variable is not set";
        int world_size;
        TT_RET_CHECK(absl::SimpleAtoi(*env_world_size, &world_size),
                     error::kFailedPrecondition)
            << "the " << kWorldSizeEnvVar
            << " environment variable is not a valid integer: "
            << *env_world_size;
        return world_size;
      }());
  return *world_size;
}

}  // namespace torch_tpu
