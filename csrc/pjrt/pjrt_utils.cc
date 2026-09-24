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

#include "csrc/pjrt/pjrt_utils.h"

#include <sstream>
#include <string>
#include <string_view>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/pjrt/pjrt_state.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

absl::StatusOr<xla::PjRtClient*> GetPjRtClient() {
  xla::PjRtClient* const client = PjrtBackend::GetInstance().GetClient();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=XLA PjRt client invariants ensure
                 // PjRtClient is always initialized during runtime.
      client != nullptr, error::kFailedPrecondition)
      << "pjrt client not initialized";
  return client;
}

absl::StatusOr<xla::PjRtDevice*> GetPjRtDevice() {
  xla::PjRtDevice* const device = PjrtBackend::GetInstance().GetDevice();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=XLA PjRt device invariants ensure
                 // PjRtDevice is always initialized during runtime.
      device != nullptr, error::kFailedPrecondition)
      << "pjrt device not initialized";
  return device;
}

absl::StatusOr<xla::PjRtMemorySpace*> GetDefaultMemorySpace(
    xla::PjRtDevice* const device) {
  TT_ASSIGN_OR_RETURN(xla::PjRtMemorySpace* const memory_space,
                      device->default_memory_space());
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=XLA PjRt memory space invariants ensure
                 // default memory space is always non-null.
      memory_space != nullptr, error::kInternal)
      << "default memory space is null";
  return memory_space;
}

std::string ToString(const xla::PjRtBuffer& buffer) {
  std::ostringstream os;
  os << "PjRtBuffer[shape=" << buffer.on_device_shape().ToString() << "]";
  return os.str();
}

bool TpuDeviceSupportsSparseCore() {
  xla::PjRtDevice* const device = PjrtBackend::GetInstance().GetDevice();
  if (device == nullptr) {
    return false;
  }
  std::string_view kind = device->device_kind();
  if (kind == "TPU v5 lite" || kind == "TPU v5e" || kind == "TPU v4" ||
      kind == "TPU v4 lite" || kind == "TPU v3" || kind == "TPU v2" ||
      kind == "cpu" || kind.empty()) {
    return false;
  }
  return true;
}

}  // namespace torch_tpu
