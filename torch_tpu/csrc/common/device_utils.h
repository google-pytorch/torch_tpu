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

#ifndef TORCH_TPU_COMMON_DEVICE_UTILS_H_
#define TORCH_TPU_COMMON_DEVICE_UTILS_H_

#include <string_view>

#include "absl/strings/numbers.h"

namespace torch_tpu {

// Returns true if `val` represents a single device index (e.g. "0" or "2").
// Returns false for empty strings, multi-device strings ("0,1"), or
// non-integers.
inline bool IsSingleDeviceSpecified(std::string_view val) {
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

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_DEVICE_UTILS_H_
