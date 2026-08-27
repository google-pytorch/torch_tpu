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
#ifndef TORCH_TPU_COMMON_DISCOVERY_H_
#define TORCH_TPU_COMMON_DISCOVERY_H_

#include <cstdint>

#include "absl/status/statusor.h"

namespace torch_tpu {

// Returns the premapped buffer size from the TPU_PREMAPPED_BUFFER_SIZE
// environment variable. If the environment variable is not set, it is set to
// "0" and 0 is returned.
//
// This function is memoized, so the environment variable is only read once.
const absl::StatusOr<int64_t>& GetPremappedBufferSizeFromEnvOnce();

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_DISCOVERY_H_
