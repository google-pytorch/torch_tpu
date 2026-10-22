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

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>

#include "absl/base/nullability.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/pjrt/pjrt_state.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

absl::StatusOr<xla::PjRtClient* absl_nonnull> GetPjRtClient() {
  return PjrtBackend::GetInstance().GetClient();
}

absl::StatusOr<xla::PjRtDevice* absl_nonnull> GetPjRtDevice() {
  return PjrtBackend::GetInstance().GetDevice();
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
  const absl::StatusOr<xla::PjRtDevice* absl_nonnull> device_or =
      PjrtBackend::GetInstance().GetDevice();
  if (!device_or.ok()) {
    ABSL_LOG_FIRST_N(WARNING, 1)
        << "TpuDeviceSupportsSparseCore: PjRt backend unavailable: "
        << device_or.status();
    return false;
  }
  std::string_view kind = (*device_or)->device_kind();
  if (kind == "TPU v5 lite" || kind == "TPU v5e" || kind == "TPU v4" ||
      kind == "TPU v4 lite" || kind == "TPU v3" || kind == "TPU v2" ||
      kind == "cpu" || kind.empty()) {
    return false;
  }
  return true;
}

bool IsDefaultDeviceLayout(const CustomLayout& layout,
                           const mlir::ElementType element_type,
                           const absl::Span<const int64_t> dimensions) {
  const absl::StatusOr<xla::PjRtClient* absl_nonnull> client = GetPjRtClient();
  if (!client.ok()) {
    ABSL_LOG_FIRST_N(WARNING, 1)
        << "IsDefaultDeviceLayout: PjRt backend unavailable: "
        << client.status();
    return false;
  }
  const absl::StatusOr<xla::Layout> default_layout =
      (*client)->GetDefaultLayout(ConvertTo<xla::PrimitiveType>(element_type),
                                  dimensions);
  if (!default_layout.ok()) {
    return false;
  }
  if (layout.minor_to_major != default_layout->minor_to_major()) {
    return false;
  }
  if (layout.tiles.size() != default_layout->tiles().size()) {
    return false;
  }
  for (size_t i = 0; i < layout.tiles.size(); ++i) {
    if (layout.tiles[i] != default_layout->tiles()[i].dimensions()) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<xla::Layout> GetDefaultDeviceLayout(
    const mlir::ElementType element_type,
    const absl::Span<const int64_t> dimensions) {
  TT_ASSIGN_OR_RETURN(xla::PjRtClient* const client, GetPjRtClient());
  return client->GetDefaultLayout(ConvertTo<xla::PrimitiveType>(element_type),
                                  dimensions);
}

}  // namespace torch_tpu
