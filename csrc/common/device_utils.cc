// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "csrc/common/device_utils.h"

#include <string>
#include <string_view>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "csrc/common/env_vars.h"

namespace torch_tpu {

bool IsSingleDeviceSpecified(std::string_view device_indices) {
  if (device_indices.empty()) {
    return false;
  }
  if (device_indices.find_first_of(",; ") != std::string_view::npos) {
    return false;
  }
  int device_id = 0;
  if (!absl::SimpleAtoi(device_indices, &device_id) || device_id < 0) {
    return false;
  }
  return true;
}

std::string GetChipsPerHostBounds(std::string_view topology) {
  std::vector<std::string_view> topology_dims = absl::StrSplit(topology, ',');
  return (topology_dims.size() == 4) ? "1,1,1,1" : "1,1,1";
}

std::string ConfigureVisibleDevice(std::string_view fallback_device) {
  const auto& visible_devices_env = GetEnvOnce<kTpuVisibleDevicesEnvVar>();
  std::string target_dev = (visible_devices_env.has_value() &&
                            IsSingleDeviceSpecified(*visible_devices_env))
                               ? *visible_devices_env
                               : std::string(fallback_device);
  SetEnv<kTpuVisibleDevicesEnvVar>(target_dev);
  SetEnv<kTpuVisibleChipsEnvVar>(target_dev);
  return target_dev;
}

void SetLibtpuInitFlag(std::string* args, std::string_view flag_name,
                       std::string_view value) {
  std::string prefix = absl::StrCat("--", flag_name);
  std::string target = absl::StrCat(prefix, "=", value);
  if (!absl::StrContains(*args, prefix)) {
    absl::StrAppend(args, " ", target);
    return;
  }
  std::string opposite =
      absl::StrCat(prefix, "=", value == "true" ? "false" : "true");
  if (absl::StrContains(*args, opposite)) {
    *args = absl::StrReplaceAll(*args, {{opposite, target}});
  }
}

void ConfigureDistributedTopologyEnv(int rank, int local_rank,
                                     const std::string& topology,
                                     const std::string& sb_addrs,
                                     const std::string& sb_port) {
  std::vector<std::string_view> addresses = absl::StrSplit(sb_addrs, ',');
  int slice_rank = rank % static_cast<int>(addresses.size());
  SetEnv<kCloudTpuTaskIdEnvVar>(absl::StrCat(slice_rank));

  ConfigureVisibleDevice(absl::StrCat(local_rank));

  std::string chips_bounds = GetChipsPerHostBounds(topology);
  SetEnv<kTpuHostBoundsEnvVar>(topology);
  SetEnv<kTpuChipsPerHostBoundsEnvVar>(chips_bounds);
  SetEnv<kTpuProcessBoundsEnvVar>(topology);
  SetEnv<kTpuChipsPerProcessBoundsEnvVar>(chips_bounds);
  SetEnv<kTpuProcessPortEnvVar>(sb_port);
  SetEnv<kTpuProcessAddressesEnvVar>(sb_addrs);
  SetEnv<kAllowMultipleLibtpuLoadEnvVar>("1");
}

}  // namespace torch_tpu
