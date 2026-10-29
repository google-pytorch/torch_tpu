/*
 * Copyright 2025 Google LLC
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

#include <cstddef>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "csrc/common/device_utils.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"
#include "csrc/distributed/slicebuilder/discovery.h"

namespace torch_tpu {
namespace {

// libtpu's default MegaScale port when neither MEGASCALE_COORDINATOR_ADDRESS
// nor MEGASCALE_PORT specifies a base port.
constexpr int kMegascaleDefaultBasePort = 8080;
constexpr int kMaxTcpPort = 65535;

// Returns the `:port` suffix of `address` (after any IPv6 `]`), or
// std::nullopt if `address` has no port suffix.
[[nodiscard]] std::optional<std::string_view> CoordinatorAddressPort(
    std::string_view address) {
  if (absl::StartsWith(address, "[")) {
    const size_t bracket_pos = address.find(']');
    if (bracket_pos == std::string_view::npos ||
        bracket_pos + 1 >= address.size() || address[bracket_pos + 1] != ':') {
      return std::nullopt;
    }
    return address.substr(bracket_pos + 2);
  }
  const size_t colon_pos = address.find(':');
  if (colon_pos == std::string_view::npos ||
      address.find(':', colon_pos + 1) != std::string_view::npos) {
    return std::nullopt;
  }
  return address.substr(colon_pos + 1);
}

// Resolves local rank 0's MegaScale port from `coordinator_address` first (so
// hooks that pre-set MEGASCALE_PORT to `base_port + local_rank` still anchor on
// rank 0's coordinator port), then from `MEGASCALE_PORT` (when pre-set at the
// container level, e.g. 8081 on GKE KubeRay), falling back to 8080.
[[nodiscard]] int ResolveMegascaleBasePort(
    std::string_view coordinator_address) {
  int port = 0;
  if (const std::optional<std::string_view> addr_port =
          CoordinatorAddressPort(coordinator_address);
      addr_port.has_value() && absl::SimpleAtoi(*addr_port, &port) &&
      port > 0 && port <= kMaxTcpPort) {
    return port;
  }
  const char* const env_port =
      std::getenv(kMegascalePortEnvVar);  // GETENV_OK=libtpu's env
  if (env_port != nullptr && absl::SimpleAtoi(env_port, &port) && port > 0 &&
      port <= kMaxTcpPort) {
    return port;
  }
  return kMegascaleDefaultBasePort;
}

// Gives the process with `local_rank` its own MegaScale ports in a multi-slice
// job. libtpu defaults the MegaScale port to 8080 and the MegaScale debug port
// to 8081 in every process, so the processes on a host would collide. The
// ports are interleaved (`base_port` + 2k, `base_port` + 2k + 1) so that no
// process's MegaScale port is another process's debug port, while local rank 0
// keeps `base_port` to match the MegaScale coordinator address.
void AssignMegascalePorts(const int local_rank) {
  const char* const raw_coord =
      std::getenv(kMegascaleCoordinatorAddressEnvVar);  // GETENV_OK=libtpu
  const std::string_view coord_addr =
      raw_coord != nullptr ? raw_coord : std::string_view();
  const int base_port = ResolveMegascaleBasePort(coord_addr);
  if (!coord_addr.empty() && !absl::StrContains(coord_addr, ':')) {
    // Pin a portless coordinator address to `base_port` before overwriting
    // MEGASCALE_PORT, since libtpu otherwise appends each process's own
    // MEGASCALE_PORT (`base_port + 2 * local_rank`).
    SetEnv<kMegascaleCoordinatorAddressEnvVar>(
        absl::StrCat(coord_addr, ":", base_port));
  }
  const int megascale_port = base_port + 2 * local_rank;
  const int megascale_debug_port = megascale_port + 1;
  ABSL_LOG(INFO) << "Multi-slice job detected, setting " << kMegascalePortEnvVar
                 << "=" << megascale_port << " and "
                 << kMegascaleDebugPortEnvVar << "=" << megascale_debug_port;
  SetEnv<kMegascalePortEnvVar>(absl::StrCat(megascale_port));
  SetEnv<kMegascaleDebugPortEnvVar>(absl::StrCat(megascale_debug_port));
}

// Keeps libtpu from wrapping the TPU PJRT client in a MegaScale client, which a
// single-slice world has no use for. libtpu creates one whenever
// MEGASCALE_COORDINATOR_ADDRESS, MEGASCALE_SLICE_ID and MEGASCALE_NUM_SLICES
// are set, including when each slice of a multi-slice JobSet runs as its own
// world, and its slice discovery can then wait forever on the coordinator.
void SkipMegascaleClient() {
  ABSL_LOG(INFO) << "Single-slice job detected, setting "
                 << kSkipMegascalePjrtClientEnvVar << "=true";
  SetEnv<kSkipMegascalePjrtClientEnvVar>("true");
}

// Returns true if `flags` sets the boolean libtpu flag `flag_name` in any
// spelling the flag parser accepts: `--<name>`, `--<name>=<value>` or the
// negated `--no<name>`, each with one or two leading dashes.
[[nodiscard]] bool HasLibtpuBoolFlag(std::string_view flag_name,
                                     const std::vector<std::string>& flags) {
  return absl::c_any_of(flags, [flag_name](std::string_view flag) {
    if (!absl::ConsumePrefix(&flag, "-")) return false;
    absl::ConsumePrefix(&flag, "-");
    if (!absl::ConsumePrefix(&flag, flag_name)) {
      absl::ConsumePrefix(&flag, "no");
      if (!absl::ConsumePrefix(&flag, flag_name)) return false;
    }
    return flag.empty() || flag.front() == '=';
  });
}

// Ensures `flags` sets the boolean flag matching `flag_prefix` to
// `target_flag`. If `bad_flag` is present, logs `warning_message` and rewrites
// it in place to `target_flag`; if no flag matching `flag_prefix` is present,
// appends `target_flag`.
void EnsureLibtpuBoolFlag(std::string_view flag_prefix,
                          std::string_view bad_flag,
                          std::string_view target_flag,
                          std::string_view warning_message,
                          std::vector<std::string>& flags) {
  bool found = false;
  for (std::string& flag : flags) {
    if (flag == bad_flag) {
      ABSL_LOG(WARNING) << warning_message;
      flag = std::string(target_flag);
      found = true;
    } else if (absl::StartsWith(flag, flag_prefix)) {
      found = true;
    }
  }
  if (!found) {
    flags.emplace_back(target_flag);
  }
}

// Appends `item` to the comma-separated list in `undefok_flag` (which starts
// with `--undefok=` or `-undefok=`) if `item` is not already present.
void AppendUndefokItem(std::string_view item, std::string& undefok_flag) {
  const size_t eq_pos = undefok_flag.find('=');
  const std::string_view existing =
      std::string_view(undefok_flag).substr(eq_pos + 1);
  std::vector<std::string_view> items =
      absl::StrSplit(existing, ',', absl::SkipEmpty());
  if (absl::c_find(items, item) == items.end()) {
    items.push_back(item);
  }
  undefok_flag = absl::StrCat("--undefok=", absl::StrJoin(items, ","));
}

// Ensures `flag_name` is listed in `--undefok` within `flags`, either by
// appending it to an existing `--undefok=`/`-undefok=` flag or by adding a new
// `--undefok=<flag_name>` entry.
void EnsureUndefokFlag(std::string_view flag_name,
                       std::vector<std::string>& flags) {
  for (std::string& flag : flags) {
    if (absl::StartsWith(flag, "--undefok=") ||
        absl::StartsWith(flag, "-undefok=")) {
      AppendUndefokItem(flag_name, flag);
      return;
    }
  }
  flags.emplace_back(absl::StrCat("--undefok=", flag_name));
}

// Normalizes `LIBTPU_INIT_ARGS` (`raw_args`) for distributed execution by
// disabling the enhanced launch barrier and defaulting to insecure SliceBuilder
// gRPC credentials along with `--undefok=slicebuilder_use_insecure_grpc`.
[[nodiscard]] std::string ConfigureLibtpuInitArgs(std::string_view raw_args) {
  std::vector<std::string> flags =
      absl::StrSplit(raw_args, absl::ByAsciiWhitespace(), absl::SkipEmpty());

  // Preventing distributed hangs, see b/477673365.
  EnsureLibtpuBoolFlag(
      "--xla_tpu_use_enhanced_launch_barrier",
      "--xla_tpu_use_enhanced_launch_barrier=true",
      "--xla_tpu_use_enhanced_launch_barrier=false",
      "libtpu_init_args contains "
      "--xla_tpu_use_enhanced_launch_barrier=true, "
      "this can cause distributed hangs. Rewriting environment "
      "variable to false.",
      flags);

  // Preventing SliceBuilder ALTS gRPC handshake hangs on GKE/GCE, see
  // b/565342785 and b/526788926. This is only a default: an explicit
  // `--slicebuilder_use_insecure_grpc` setting is preserved so users can opt
  // back into ALTS credentials, see b/571936195.
  if (!HasLibtpuBoolFlag("slicebuilder_use_insecure_grpc", flags)) {
    flags.emplace_back("--slicebuilder_use_insecure_grpc=true");
  }
  // Mark `slicebuilder_use_insecure_grpc` in `--undefok` so environments where
  // the flag is not registered (e.g. standalone C++ tests) never fail command
  // line flag parsing.
  EnsureUndefokFlag("slicebuilder_use_insecure_grpc", flags);
  return absl::StrJoin(flags, " ");
}

// Returns true if `libtpu_init_args` sets libtpu's
// `--deepsea_hal_included_devicepaths` flag, to which libtpu also maps
// TPU_VISIBLE_DEVICE_PATHS.
[[nodiscard]] bool SetsIncludedDevicePathsFlag(
    std::string_view libtpu_init_args) {
  const std::vector<std::string_view> flags = absl::StrSplit(
      libtpu_init_args, absl::ByAsciiWhitespace(), absl::SkipEmpty());
  return absl::c_any_of(flags, [](std::string_view flag) {
    if (!absl::ConsumePrefix(&flag, "-")) return false;
    absl::ConsumePrefix(&flag, "-");
    if (!absl::ConsumePrefix(&flag, "deepsea_hal_included_devicepaths")) {
      return false;
    }
    // `--name value` (value in the next token) or `--name=value`.
    return flag.empty() || (flag.front() == '=' && flag.size() > 1);
  });
}

}  // namespace

absl::Status InitializeDistributedEnvironment(
    const DistributedWorkerConfiguration& config) {
  if (config.sb_addrs.empty()) {
    return TT_ERROR(error::kFailedPrecondition)
           << kTorchTpuSlicebuilderAddressesEnvVar << " is empty.";
  }
  if (config.rank < 0) {
    return TT_ERROR(error::kFailedPrecondition)
           << "RANK " << config.rank << " is out of bounds (negative)";
  }
  std::vector<std::string> addresses = absl::StrSplit(config.sb_addrs, ',');
  const int slice_rank = config.rank % static_cast<int>(addresses.size());
  SetEnv<kCloudTpuTaskIdEnvVar>(absl::StrCat(slice_rank));

  // TPU_VISIBLE_DEVICES is our source of truth for device visibility.
  // Because libtpu prioritizes TPU_VISIBLE_CHIPS over TPU_VISIBLE_DEVICES when
  // non-empty, we explicitly overwrite TPU_VISIBLE_CHIPS to match
  // TPU_VISIBLE_DEVICES so that external settings do not override device
  // selection.
  const auto& visible_devices_env = GetEnvOnce<kTpuVisibleDevicesEnvVar>();
  const std::string target_dev = (visible_devices_env.has_value() &&
                                  IsSingleDeviceSpecified(*visible_devices_env))
                                     ? *visible_devices_env
                                     : absl::StrCat(config.local_rank);
  SetEnv<kTpuVisibleDevicesEnvVar>(target_dev);
  SetEnv<kTpuVisibleChipsEnvVar>(target_dev);

  const std::vector<std::string> topology_dims =
      absl::StrSplit(config.topology, ',');
  const std::string chips_bounds =
      (topology_dims.size() == 4) ? "1,1,1,1" : "1,1,1";

  SetEnv<kTpuHostBoundsEnvVar>(config.topology);
  SetEnv<kTpuChipsPerHostBoundsEnvVar>(chips_bounds);
  SetEnv<kTpuProcessBoundsEnvVar>(config.topology);
  SetEnv<kTpuChipsPerProcessBoundsEnvVar>(chips_bounds);

  // The free slicebuilder port of this process.
  SetEnv<kTpuProcessPortEnvVar>(config.sb_port);

  // `addresses` lists the processes of one slice, so a larger world spans
  // several slices, which libtpu connects over MegaScale (DCN).
  if (config.world_size > static_cast<int>(addresses.size())) {
    AssignMegascalePorts(config.local_rank);
  } else {
    SkipMegascaleClient();
  }

  // The addresses of all other workers in the slice.
  SetEnv<kTpuProcessAddressesEnvVar>(absl::StrJoin(addresses, ","));

  // Avoid multi-process libtpu lock in GCP init, see b/487769788.
  SetEnv<kAllowMultipleLibtpuLoadEnvVar>("1");

  const std::string libtpu_init_args_str =
      ConfigureLibtpuInitArgs(GetEnvOnce<kLibtpuInitArgsEnvVar>().value_or(""));
  SetEnv<kLibtpuInitArgsEnvVar>(libtpu_init_args_str);
  return absl::OkStatus();
}

absl::Status InitializeSingleDeviceEnvironment() {
  // TPU_VISIBLE_DEVICES is our source of truth for device visibility.
  // Because libtpu prioritizes TPU_VISIBLE_CHIPS over TPU_VISIBLE_DEVICES when
  // non-empty, we explicitly overwrite TPU_VISIBLE_CHIPS to match
  // TPU_VISIBLE_DEVICES so that external settings do not override device
  // selection.
  const auto& visible_devices_env = GetEnvOnce<kTpuVisibleDevicesEnvVar>();
  const std::string target_dev = (visible_devices_env.has_value() &&
                                  IsSingleDeviceSpecified(*visible_devices_env))
                                     ? *visible_devices_env
                                     : "0";
  SetEnv<kTpuVisibleDevicesEnvVar>(target_dev);
  SetEnv<kTpuVisibleChipsEnvVar>(target_dev);
  return absl::OkStatus();
}

// In OSS, libtpu is loaded dynamically, so its absl flags (such as
// --deepsea_hal_included_devicepaths) aren't linked into this process. Read
// the env vars that libtpu reads instead. Internally, libtpu is linked
// statically and google/csrc/common/environment.cc reads the flag directly.
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
  return libtpu_init_args != nullptr &&
         SetsIncludedDevicePathsFlag(libtpu_init_args);
}

}  // namespace torch_tpu
