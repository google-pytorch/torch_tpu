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

#include "csrc/pjrt/world_config.h"

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"

namespace torch_tpu {

absl::StatusOr<WorldConfig> ResolveWorldConfig(int world_size, int device_count,
                                               bool virtual_multihost) {
  TT_RET_CHECK(device_count > 0, error::kInvalidArgument)
      << "invalid PJRT device count: " << device_count;
  WorldConfig config = {.world_size = world_size,
                        .global_device_count = world_size,
                        .virtual_multihost = false};
  if (world_size == 1 || world_size == device_count) {
    return config;
  }
  if (virtual_multihost && world_size > device_count &&
      world_size % device_count == 0) {
    config.global_device_count = device_count;
    config.virtual_multihost = true;
    return config;
  }
  return TT_ERROR(error::kInvalidArgument)
         << "invalid world_size configuration: expected 1 or " << device_count
         << (virtual_multihost ? " or a larger multiple of it" : "")
         << ", but got " << world_size
         << ". Please check your environment variables"
         << (virtual_multihost || world_size <= device_count ||
                     world_size % device_count != 0
                 ? ""
                 : absl::StrCat(" (set ", kTorchTpuVirtualMultiHostEnvVar,
                                "=1 to emulate ", world_size, " ranks on this ",
                                device_count, "-device slice)"));
}

}  // namespace torch_tpu
