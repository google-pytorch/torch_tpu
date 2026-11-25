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
#ifndef TORCH_TPU_CSRC_COMMON_DISCOVERY_H_
#define TORCH_TPU_CSRC_COMMON_DISCOVERY_H_

#include <cstdint>

#include "absl/status/statusor.h"

namespace torch_tpu {

// Returns the premapped buffer size from the TPU_PREMAPPED_BUFFER_SIZE
// environment variable. If the environment variable is not set, computes a
// memory-aware default, writes it to TPU_PREMAPPED_BUFFER_SIZE, and returns it.
//
// The default divides the host or container memory limit (the minimum of
// physical RAM and any cgroup memory limit) by the number of local TorchTPU
// worker processes. It is 0 when that per-worker share is below 16 GiB, and
// otherwise min(4 GiB, per-worker share / 16) rounded down to a power of two.
// A performance warning is emitted if the default is 0.
// Expected defaults in common environments:
//
//   Environment                    | Memory limit | Workers | Default/worker
//   -------------------------------+--------------+---------+---------------
//   Small container                |  <= 15 GiB   |    1    | 0
//   24 GiB container               |    24 GiB    |    1    | 1 GiB
//   24 GiB container               |    24 GiB    |  2 - 8  | 0
//   Cloud TPU v5e, 1 chip          |    48 GiB    |    1    | 2 GiB
//   Cloud TPU v5e, 8 chips         |   384 GiB    |    8    | 2 GiB
//   Cloud TPU v5p, 4 chips         |   448 GiB    |    4    | 4 GiB
//   Cloud TPU v6e, 1 / 4 / 8 chips | 176-1440 GiB |  1 - 8  | 4 GiB
//   Cloud TPU v7x, 4 chips         |   960 GiB    |    8    | 4 GiB
//
// This function is memoized, so the environment variable is only read once.
[[nodiscard]] const absl::StatusOr<int64_t>&
GetPremappedBufferSizeFromEnvOnce();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_DISCOVERY_H_
