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

#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "csrc/common/device_utils.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/libtpu_version.h"
#include "csrc/distributed/slicebuilder/discovery.h"

namespace torch_tpu {
namespace {

constexpr std::string_view kMinLibtpuVersionWithSlicebuilderInsecureGrpc =
    "0.0.48";

// Returns the local hostname with any domain or port suffix (starting at '.'
// or ':') stripped, matching SliceBuilder's `GetAndSanitizeHostName()`. Falls
// back to "localhost" if querying or resolving the sanitized hostname fails.
[[nodiscard]] std::string GetSanitizedHostName() {
  char hostname[1024];
  if (gethostname(hostname, sizeof(hostname)) != 0) {
    return "localhost";
  }
  hostname[sizeof(hostname) - 1] = '\0';
  std::string_view host_view(hostname);
  const size_t strip_from = host_view.find_first_of(".:");
  if (strip_from != std::string_view::npos) {
    host_view = host_view.substr(0, strip_from);
  }
  if (host_view.empty()) {
    return "localhost";
  }
  const std::string sanitized(host_view);
  struct addrinfo* res = nullptr;
  if (getaddrinfo(sanitized.c_str(), nullptr, nullptr, &res) != 0) {
    return "localhost";
  }
  freeaddrinfo(res);
  return sanitized;
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
// disabling the enhanced launch barrier and, when
// `use_slicebuilder_insecure_grpc` is true, enabling insecure SliceBuilder gRPC
// credentials along with `--undefok=slicebuilder_use_insecure_grpc`.
[[nodiscard]] std::string ConfigureLibtpuInitArgs(
    std::string_view raw_args, const bool use_slicebuilder_insecure_grpc) {
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

  if (use_slicebuilder_insecure_grpc) {
    // Preventing SliceBuilder ALTS gRPC handshake hangs on GKE/GCE, see
    // b/565342785 and b/526788926.
    EnsureLibtpuBoolFlag(
        "--slicebuilder_use_insecure_grpc",
        "--slicebuilder_use_insecure_grpc=false",
        "--slicebuilder_use_insecure_grpc=true",
        "libtpu_init_args contains "
        "--slicebuilder_use_insecure_grpc=false, "
        "this can cause SliceBuilder gRPC handshake hangs on GKE/GCE. "
        "Rewriting environment variable to true.",
        flags);
    // `slicebuilder_use_insecure_grpc` was introduced in libtpu 0.0.48; mark
    // it in `--undefok` so environments where SetLibtpuVersion was not called
    // (e.g. standalone C++ tests) never fail command line flag parsing.
    EnsureUndefokFlag("slicebuilder_use_insecure_grpc", flags);
  }
  return absl::StrJoin(flags, " ");
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
  const bool use_slicebuilder_insecure_grpc =
      IsLibtpuVersionAtLeast(kMinLibtpuVersionWithSlicebuilderInsecureGrpc);
  if (use_slicebuilder_insecure_grpc) {
    // Under LIBTPU_ON_GCE (libtpu >= 0.0.48), SliceBuilder WorkerService binds
    // to `<GetAndSanitizeHostName()>:<port>` rather than `[::]:<port>`, which
    // resolves to the host/pod network interface IP rather than loopback
    // (127.0.0.1). Rewrite `localhost:<port>` addresses to `<hostname>:<port>`
    // so MasterFactory connects to the exact address WorkerService listens on.
    const std::string sanitized_hostname = GetSanitizedHostName();
    for (std::string& addr : addresses) {
      std::string_view addr_view(addr);
      if (absl::ConsumePrefix(&addr_view, "localhost:")) {
        addr = absl::StrCat(sanitized_hostname, ":", addr_view);
      }
    }
  }

  ConfigureDistributedTopologyEnv(
      config.rank, config.local_rank, config.topology,
      absl::StrJoin(addresses, ","), config.sb_port);

  const std::string libtpu_init_args_str =
      ConfigureLibtpuInitArgs(GetEnvOnce<kLibtpuInitArgsEnvVar>().value_or(""),
                              use_slicebuilder_insecure_grpc);
  SetEnv<kLibtpuInitArgsEnvVar>(libtpu_init_args_str);
  return absl::OkStatus();
}

absl::Status InitializeSingleDeviceEnvironment() {
  ConfigureVisibleDevice("0");
  return absl::OkStatus();
}

}  // namespace torch_tpu
