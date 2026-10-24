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

#ifndef TORCH_TPU_CSRC_COMMON_ENVIRONMENT_H_
#define TORCH_TPU_CSRC_COMMON_ENVIRONMENT_H_

#include "absl/status/status.h"
#include "csrc/distributed/slicebuilder/discovery.h"

namespace torch_tpu {

// This function handles the environment initialization for distributed TPU
// environment.
absl::Status InitializeDistributedEnvironment(
    const DistributedWorkerConfiguration& config);

// Configures the local environment for single-device execution, including
// single-device visibility and topology/bounds settings.
absl::Status InitializeSingleDeviceEnvironment();

// Returns whether the TPU device paths that libtpu should use are configured
// explicitly. Such paths may name device nodes outside of the default
// locations checked by HasTpuDeviceNodes() in tpu_hardware.h. Does not
// initialize the TPU runtime and does not cache the result.
[[nodiscard]] bool HasCustomTpuDevicePaths();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_ENVIRONMENT_H_
