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

#ifndef TORCH_TPU_CSRC_PJRT_WORLD_CONFIG_H_
#define TORCH_TPU_CSRC_PJRT_WORLD_CONFIG_H_

#include "absl/status/statusor.h"

namespace torch_tpu {

// How the torch world maps onto the PJRT client's devices.
struct WorldConfig {
  // The torch world size (WORLD_SIZE, or 1 when unset).
  int world_size = 1;
  // How many devices the XLA computations of this process are compiled for:
  // the world size, or the PJRT slice's device count in virtual multi-host
  // mode.
  int global_device_count = 1;
  // Virtual multi-host mode (TORCH_TPU_VIRTUAL_MULTIHOST): the world spans
  // `world_size / global_device_count` per-host slices that cannot talk to
  // each other; cross-slice collectives are emulated on the local slice
  // (distributed/utils.h). Compile / profile only.
  bool virtual_multihost = false;

  bool operator==(const WorldConfig&) const = default;
};

// Resolves the world configuration from WORLD_SIZE and the PJRT device count.
//
// The world size must be 1 or the device count; with `virtual_multihost`
// enabled it may also be any larger multiple of the device count, in which
// case the computations are compiled for the local slice only. Anything else
// is an invalid-argument error naming both numbers (and, when the opt-in is
// off, the environment variable that would enable the emulation).
absl::StatusOr<WorldConfig> ResolveWorldConfig(int world_size, int device_count,
                                               bool virtual_multihost);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_PJRT_WORLD_CONFIG_H_
