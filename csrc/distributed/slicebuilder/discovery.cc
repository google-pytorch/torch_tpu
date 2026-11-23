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

#include <optional>
#include <string>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"

namespace torch_tpu {

static absl::Status ValidateRequiredDistributedEnvVars(
    const std::optional<int>& explicit_slice_rank) {
  std::vector<std::string> missing_torchrun_vars;
  const std::optional<std::string>& sb_addrs =
      GetEnvOnce<kTorchTpuSlicebuilderAddressesEnvVar>();
  if (explicit_slice_rank.has_value()) {
    const std::optional<int> world_size = GetIntegerEnvOnce<kWorldSizeEnvVar>();
    if (world_size.has_value() && sb_addrs.has_value() && !sb_addrs->empty()) {
      const std::vector<std::string> addresses = absl::StrSplit(*sb_addrs, ',');
      if (*world_size > static_cast<int>(addresses.size()) &&
          !GetIntegerEnvOnce<kLocalRankEnvVar>().has_value()) {
        missing_torchrun_vars.push_back(kLocalRankEnvVar);
      }
    }
  } else {
    if (!GetIntegerEnvOnce<kRankEnvVar>().has_value()) {
      missing_torchrun_vars.push_back(kRankEnvVar);
    }
    if (!GetIntegerEnvOnce<kLocalRankEnvVar>().has_value()) {
      missing_torchrun_vars.push_back(kLocalRankEnvVar);
    }
    if (!GetIntegerEnvOnce<kWorldSizeEnvVar>().has_value()) {
      missing_torchrun_vars.push_back(kWorldSizeEnvVar);
    }
    if (!GetEnvOnce<kMasterAddrEnvVar>().has_value()) {
      missing_torchrun_vars.push_back(kMasterAddrEnvVar);
    }
    if (!GetIntegerEnvOnce<kMasterPortEnvVar>().has_value()) {
      missing_torchrun_vars.push_back(kMasterPortEnvVar);
    }
  }

  std::vector<std::string> missing_tpu_vars;
  if (!sb_addrs.has_value()) {
    missing_tpu_vars.push_back(kTorchTpuSlicebuilderAddressesEnvVar);
  }
  if (!GetEnvOnce<kTorchTpuTopologyEnvVar>().has_value()) {
    missing_tpu_vars.push_back(kTorchTpuTopologyEnvVar);
  }

  if (missing_torchrun_vars.empty() && missing_tpu_vars.empty()) {
    return absl::OkStatus();
  }

  std::vector<std::string> all_missing_vars = missing_torchrun_vars;
  all_missing_vars.insert(all_missing_vars.end(), missing_tpu_vars.begin(),
                          missing_tpu_vars.end());

  std::vector<std::string> hints;
  if (!missing_torchrun_vars.empty()) {
    hints.push_back(
        "run the program via torchrun or similar tools so that the "
        "distributed environment is set up properly");
  }
  if (!missing_tpu_vars.empty()) {
    hints.push_back(
        "call torch.tpu.distributed.set_tpu_launch_env() before spawning "
        "workers or export eval \"$(python3 -m torch.tpu.distributed)\" "
        "before launching");
  }

  return TT_ERROR(error::kFailedPrecondition)
         << "missing or invalid required environment variables for "
            "distributed training: "
         << absl::StrJoin(all_missing_vars, ", ") << "; please "
         << SequenceToReadableStr(absl::MakeConstSpan(hints));
}

absl::StatusOr<DistributedWorkerConfiguration>
GetDistributedWorkerConfiguration() {
  const std::optional<int> explicit_slice_rank =
      GetIntegerEnvOnce<kTorchTpuSliceRankEnvVar>();
  TT_RETURN_IF_ERROR(ValidateRequiredDistributedEnvVars(explicit_slice_rank));

  // Get the slice builder addresses from the environment variables.
  TT_ASSIGN_OR_RETURN(
      std::string sb_addrs,
      GetRequiredEnvOnce<kTorchTpuSlicebuilderAddressesEnvVar>());

  if (sb_addrs.empty()) {
    return TT_ERROR(error::kFailedPrecondition)
           << kTorchTpuSlicebuilderAddressesEnvVar << " is empty.";
  }
  std::vector<std::string> addresses = absl::StrSplit(sb_addrs, ',');

  int rank = 0;
  int local_rank = 0;
  int world_size = 0;
  int slice_rank = 0;
  if (explicit_slice_rank.has_value()) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=covered in C++ discovery_test.
        *explicit_slice_rank >= 0 &&
            *explicit_slice_rank < static_cast<int>(addresses.size()),
        error::kFailedPrecondition)
        << "expected " << kTorchTpuSliceRankEnvVar << " to be in [0, "
        << addresses.size() << "), got " << *explicit_slice_rank;
    rank = GetIntegerEnvOnce<kRankEnvVar>().value_or(*explicit_slice_rank);
    if (rank < 0) {
      return TT_ERROR(error::kFailedPrecondition)
             << "RANK " << rank << " is out of bounds (negative)";
    }
    local_rank =
        GetIntegerEnvOnce<kLocalRankEnvVar>().value_or(*explicit_slice_rank);
    world_size = GetIntegerEnvOnce<kWorldSizeEnvVar>().value_or(
        static_cast<int>(addresses.size()));
    slice_rank = *explicit_slice_rank;
  } else {
    TT_ASSIGN_OR_RETURN(rank, (GetRequiredIntegerEnvOnce<kRankEnvVar>()));
    TT_ASSIGN_OR_RETURN(local_rank,
                        (GetRequiredIntegerEnvOnce<kLocalRankEnvVar>()));

    // Get the master address from the environment variables.
    std::string master_addr;
    TT_ASSIGN_OR_RETURN(master_addr, GetRequiredEnvOnce<kMasterAddrEnvVar>());

    // Get the world size from the environment variables.
    TT_ASSIGN_OR_RETURN(world_size,
                        (GetRequiredIntegerEnvOnce<kWorldSizeEnvVar>()));

    if (rank < 0) {
      return TT_ERROR(error::kFailedPrecondition)
             << "RANK " << rank << " is out of bounds (negative)";
    }
    slice_rank = rank % static_cast<int>(addresses.size());
  }

  // Get the local port.
  std::string my_addr = addresses[slice_rank];
  std::vector<std::string> my_parts = absl::StrSplit(my_addr, ':');
  if (my_parts.size() != 2) {
    return TT_ERROR(error::kFailedPrecondition)
           << "Invalid address format in "
           << kTorchTpuSlicebuilderAddressesEnvVar
           << " for current rank: " << my_addr;
  }
  std::string sb_port = my_parts[1];

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
      .slice_rank = explicit_slice_rank,
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
