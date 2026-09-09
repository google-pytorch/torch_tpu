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

#include "csrc/pjrt/pjrt_client.h"

#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/distributed/slicebuilder/discovery.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {

absl::StatusOr<absl_nonnull std::unique_ptr<xla::PjRtClient>> GetPjRtClient(
    const std::string& device_type, int64_t tpu_premapped_buffer_size) {
  if (device_type == "tpu") {
    return xla::GetCApiClient(
        "tpu", {{"premapped_buffer_size", tpu_premapped_buffer_size}});
  }
  if (device_type == "xla_cuda") {
    TT_ASSIGN_OR_RETURN(
        auto client, xla::GetCApiClient("gpu", {{"use_tfrt_gpu_client", true}}),
        _.SetPrepend()
            << "failed to create the XLA:GPU client (add a dependency on the "
               "'gpu_static_registration' build target to use the XLA:GPU "
               "backend): ");
    return client;
  }
  if (device_type == "xla_cpu") {
    int64_t cpu_device_count = 1;
    if (auto world_size_or = GetWorldSizeFromEnvOnce(); world_size_or.ok()) {
      cpu_device_count = *world_size_or;
    }
    return xla::GetCApiClient("cpu", {{"cpu_device_count", cpu_device_count}});
  }
  return TT_ERROR(error::kInvalidArgument)
         << "unable to get PJRT client for unsupported device type: "
         << device_type;
}

}  // namespace torch_tpu
