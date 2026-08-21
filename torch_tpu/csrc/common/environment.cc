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

#include "torch_tpu/csrc/common/environment.h"

#include <string>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "torch_tpu/csrc/common/device_utils.h"
#include "torch_tpu/csrc/common/env_vars.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/distributed/slicebuilder/discovery.h"

namespace torch_tpu {

absl::Status InitializeDistributedEnvironment(
    const DistributedWorkerConfiguration& config) {
  if (config.sb_addrs.empty()) {
    return TT_ERROR(error::kFailedPrecondition)
           << kTpuSlicebuilderAddressesEnvVar << " is empty.";
  }
  if (config.rank < 0) {
    return TT_ERROR(error::kFailedPrecondition)
           << "RANK " << config.rank << " is out of bounds (negative)";
  }
  std::vector<std::string> addresses = absl::StrSplit(config.sb_addrs, ',');
  int slice_rank = config.rank % static_cast<int>(addresses.size());
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

  std::vector<std::string> topology_dims = absl::StrSplit(config.topology, ',');
  std::string chips_bounds = (topology_dims.size() == 4) ? "1,1,1,1" : "1,1,1";

  SetEnv<kTpuHostBoundsEnvVar>(config.topology);
  SetEnv<kTpuChipsPerHostBoundsEnvVar>(chips_bounds);
  SetEnv<kTpuProcessBoundsEnvVar>(config.topology);
  SetEnv<kTpuChipsPerProcessBoundsEnvVar>(chips_bounds);

  // The free slicebuilder port of this process.
  SetEnv<kTpuProcessPortEnvVar>(config.sb_port);

  // The addresses of all other workers in the slice.
  SetEnv<kTpuProcessAddressesEnvVar>(config.sb_addrs);

  // Avoid multi-process libtpu lock in GCP init, see b/487769788.
  SetEnv<kAllowMultipleLibtpuLoadEnvVar>("1");

  std::string libtpu_init_args_str =
      GetEnvOnce<kLibtpuInitArgsEnvVar>().value_or("");
  if (absl::StrContains(libtpu_init_args_str,
                        "--xla_tpu_use_enhanced_launch_barrier=true")) {
    ABSL_LOG(WARNING)
        << "libtpu_init_args contains "
           "--xla_tpu_use_enhanced_launch_barrier=true, "
           "this can cause distributed hangs. Rewriting environment "
           "variable to false.";
    libtpu_init_args_str =
        absl::StrReplaceAll(libtpu_init_args_str,
                            {{"--xla_tpu_use_enhanced_launch_barrier=true",
                              "--xla_tpu_use_enhanced_launch_barrier=false"}});
  } else if (!absl::StrContains(libtpu_init_args_str,
                                "--xla_tpu_use_enhanced_launch_barrier")) {
    // Preventing distributed hangs, see b/477673365.
    absl::StrAppend(&libtpu_init_args_str,
                    " --xla_tpu_use_enhanced_launch_barrier=false");
  }
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

}  // namespace torch_tpu
