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

#ifndef TORCH_TPU_CSRC_PJRT_PJRT_CLIENT_H_
#define TORCH_TPU_CSRC_PJRT_PJRT_CLIENT_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {

// Initializes and returns a PjRtClient based on the provided device type.
absl::StatusOr<absl_nonnull std::unique_ptr<xla::PjRtClient>> GetPjRtClient(
    const std::string& device_type, int64_t tpu_premapped_buffer_size);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_PJRT_PJRT_CLIENT_H_
