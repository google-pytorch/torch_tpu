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

#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "csrc/common/env_vars.h"

namespace torch_tpu {
namespace {

// If `flag` sets the libtpu flag `flag_name` (`-<flag_name>` or
// `--<flag_name>`, optionally followed by `=<value>`), returns the rest of
// `flag` after the flag name (empty, or starting with '='). Returns
// std::nullopt otherwise.
[[nodiscard]] std::optional<std::string_view> MatchLibtpuFlag(
    std::string_view flag, std::string_view flag_name) {
  if (!absl::ConsumePrefix(&flag, "-")) return std::nullopt;
  absl::ConsumePrefix(&flag, "-");
  if (!absl::ConsumePrefix(&flag, flag_name)) return std::nullopt;
  if (!flag.empty() && flag.front() != '=') return std::nullopt;
  return flag;
}

}  // namespace

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

void ConfigureVisibleDevice(std::string_view default_device) {
  const char* const visible_devices =
      std::getenv(kTpuVisibleDevicesEnvVar);  // GETENV_OK=env var is set below.
  const std::string target_dev = GetSingleDeviceOrDefault(
      visible_devices != nullptr ? visible_devices : "", default_device);
  SafeSetEnv<kTpuVisibleDevicesEnvVar>(target_dev);
  SafeSetEnv<kTpuVisibleChipsEnvVar>(target_dev);
}

void SetTpuBoundsEnv(const std::string& host_bounds,
                     const std::string& chips_bounds) {
  SafeSetEnv<kTpuHostBoundsEnvVar>(host_bounds);
  SafeSetEnv<kTpuChipsPerHostBoundsEnvVar>(chips_bounds);
  SafeSetEnv<kTpuProcessBoundsEnvVar>(host_bounds);
  SafeSetEnv<kTpuChipsPerProcessBoundsEnvVar>(chips_bounds);

  // Avoid multi-process libtpu lock file collisions across distributed workers
  // or independent single-device processes on the same host, see b/487769788.
  SafeSetEnv<kAllowMultipleLibtpuLoadEnvVar>("1");
}

void SetLibtpuInitFlag(std::string* args, std::string_view flag_name,
                       const bool value) {
  const std::string target =
      absl::StrCat("--", flag_name, "=", value ? "true" : "false");
  std::vector<std::string> flags =
      absl::StrSplit(*args, absl::ByAsciiWhitespace(), absl::SkipEmpty());
  bool found = false;
  for (std::string& flag : flags) {
    std::optional<std::string_view> rest = MatchLibtpuFlag(flag, flag_name);
    if (!rest.has_value()) {
      // The negated form of a boolean flag, e.g. `--noname`.
      rest = MatchLibtpuFlag(flag, absl::StrCat("no", flag_name));
    }
    if (rest.has_value()) {
      flag = target;
      found = true;
    }
  }
  if (!found) {
    flags.push_back(target);
  }
  *args = absl::StrJoin(flags, " ");
}

bool HasCustomTpuDevicePaths() {
  // Read with std::getenv() rather than GetEnvOnce<>(), which caches the value:
  // these env vars belong to libtpu, which reads them when it initializes.
  const char* const device_paths =
      std::getenv(kTpuVisibleDevicePathsEnvVar);  // GETENV_OK=libtpu's env
  if (device_paths != nullptr && device_paths[0] != '\0') {
    return true;
  }
  const char* const libtpu_init_args =
      std::getenv(kLibtpuInitArgsEnvVar);  // GETENV_OK=libtpu's env
  if (libtpu_init_args == nullptr) {
    return false;
  }
  const std::vector<std::string_view> flags = absl::StrSplit(
      libtpu_init_args, absl::ByAsciiWhitespace(), absl::SkipEmpty());
  return absl::c_any_of(flags, [](std::string_view flag) {
    const std::optional<std::string_view> rest =
        MatchLibtpuFlag(flag, "deepsea_hal_included_devicepaths");
    // `--name value` (value in the next token) or `--name=value`.
    return rest.has_value() && (rest->empty() || rest->size() > 1);
  });
}

void ConfigureDistributedTopologyEnv(int rank, std::string_view default_device,
                                     const std::string& topology,
                                     const std::string& sb_addrs,
                                     const std::string& sb_port) {
  const std::vector<std::string_view> addresses = absl::StrSplit(sb_addrs, ',');
  const int slice_rank = rank % static_cast<int>(addresses.size());
  SafeSetEnv<kCloudTpuTaskIdEnvVar>(absl::StrCat(slice_rank));

  ConfigureVisibleDevice(default_device);

  SetTpuBoundsEnv(topology, GetChipsPerHostBounds(topology));

  // The free slicebuilder port of this process.
  SafeSetEnv<kTpuProcessPortEnvVar>(sb_port);
  // The addresses of all workers in the slice.
  SafeSetEnv<kTpuProcessAddressesEnvVar>(sb_addrs);
}

}  // namespace torch_tpu
