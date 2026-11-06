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

#ifndef TORCH_TPU_CSRC_COMMON_DEVICE_UTILS_H_
#define TORCH_TPU_CSRC_COMMON_DEVICE_UTILS_H_

#include <string>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "csrc/common/error_utils.h"

namespace torch_tpu {

// Returns true if `val` represents a single device index (e.g. "0" or "2").
// Returns false for empty strings, multi-device strings ("0,1"), or
// non-integers.
[[nodiscard]] inline bool IsSingleDeviceSpecified(std::string_view val) {
  if (val.empty()) {
    return false;
  }
  if (val.find_first_of(",; ") != std::string_view::npos) {
    return false;
  }
  int device_id = 0;
  if (!absl::SimpleAtoi(val, &device_id) || device_id < 0) {
    return false;
  }
  return true;
}

// Returns the device of the process with local rank `local_rank` (0 if not
// distributed), given the visible devices, e.g. from `TPU_VISIBLE_DEVICES`:
// - `local_rank`, if `visible_devices` is empty, i.e. all devices are visible.
// - The listed device, if only one is listed, e.g. "3" -> "3" for any local
//   rank.
// - The `local_rank`-th listed device, if several are listed, like
//   `CUDA_VISIBLE_DEVICES`, e.g. "4,5,6,7" -> "5" for local rank 1.
// Fails with InvalidArgument if `local_rank` is negative, `visible_devices` is
// malformed, or it lists several devices but fewer than `local_rank + 1`.
inline absl::StatusOr<std::string> SelectLocalDevice(
    const std::string_view visible_devices, const int local_rank) {
  TT_RET_CHECK(local_rank >= 0, error::kInvalidArgument)
      << "expected local rank to be >= 0, got " << local_rank;
  if (visible_devices.empty()) {  // All devices are visible.
    return absl::StrCat(local_rank);
  }

  const std::vector<std::string_view> devices =
      absl::StrSplit(visible_devices, ',');
  TT_RET_CHECK(absl::c_all_of(devices, IsSingleDeviceSpecified),
               error::kInvalidArgument)
      << "expected visible devices to be comma-separated integers >= 0, got '"
      << visible_devices << "'";
  if (devices.size() == 1) {
    return std::string(devices[0]);
  }

  TT_RET_CHECK(local_rank < static_cast<int>(devices.size()),
               error::kInvalidArgument)
      << "local rank " << local_rank << " has no device in visible devices '"
      << visible_devices << "'";

  return std::string(devices[local_rank]);
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_DEVICE_UTILS_H_
