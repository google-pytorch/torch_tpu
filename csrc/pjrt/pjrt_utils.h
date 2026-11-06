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

#ifndef TORCH_TPU_CSRC_PJRT_PJRT_UTILS_H_
#define TORCH_TPU_CSRC_PJRT_PJRT_UTILS_H_

#include <cstdint>
#include <string>

// clang-format off
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/dimension_types.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/xla_data.pb.h"
// clang-format on

namespace torch_tpu {

absl::StatusOr<xla::PjRtClient* absl_nonnull> GetPjRtClient();
absl::StatusOr<xla::PjRtDevice* absl_nonnull> GetPjRtDevice();
absl::StatusOr<xla::PjRtMemorySpace*> GetDefaultMemorySpace(
    xla::PjRtDevice* device);

std::string ToString(const xla::PjRtBuffer& buffer);

// Returns true if the active TPU device has SparseCore hardware on chip and
// supports SparseCore offloading. Returns false on architectures without
// SparseCore (e.g., TPU v2, v3, v4, v5e).
bool TpuDeviceSupportsSparseCore();

// Returns whether `layout` has the dimension order and tiles of the default
// layout for `element_type` and `dimensions`. Returns false if the default
// layout is unknown.
bool IsDefaultDeviceLayout(const CustomLayout& layout,
                           mlir::ElementType element_type,
                           absl::Span<const int64_t> dimensions);

// Returns the layout that PjRt gives a buffer of `element_type` and
// `dimensions` by default.
absl::StatusOr<xla::Layout> GetDefaultDeviceLayout(
    mlir::ElementType element_type, absl::Span<const int64_t> dimensions);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_PJRT_PJRT_UTILS_H_
