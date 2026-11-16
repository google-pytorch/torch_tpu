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

#ifndef TORCH_TPU_CSRC_COMMON_LAYOUT_UTILS_H_
#define TORCH_TPU_CSRC_COMMON_LAYOUT_UTILS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/shape.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/layout.h"

namespace torch_tpu {

// Represents the layout metrics of a TPU tensor mapped from PyTorch.
struct TpuLayout {
  // The sizes of the tensor dimensions. Must have the same size (rank) as
  // `strides`.
  Dimensions sizes;
  // The strides of the tensor dimensions. Must have the same size (rank) as
  // `sizes`.
  Strides strides;
  // The offset in elements from the start of the storage. Defaults to -1 (an
  // obviously incorrect offset value) to force the caller to explicitly specify
  // it.
  int64_t storage_offset = -1;
  // The element type on the TPU/StableHLO side. Defaults to an invalid element
  // type (static_cast<mlir::ElementType>(-1)) to force the caller to explicitly
  // specify it.
  mlir::ElementType element_type = static_cast<mlir::ElementType>(-1);
};

// Resolves the TPU layout for a PyTorch tensor. This maps the PyTorch tensor's
// logical layout (sizes, strides, offset, dtype) to a TPU/StableHLO-compatible
// logical layout. For packed formats like FP4, this includes applying morphing
// configurations to expose them as unpacked 1-byte elements with doubled
// physical dimensions, adjusting sizes, strides, and offsets accordingly.
absl::StatusOr<TpuLayout> ResolveTpuLayout(const at::Tensor& tensor);

// Resolves the on-device `xla::Layout` for each `Shape` in `shapes`: starts
// from `PjRtClient::GetDefaultLayout` for the shape's dtype and dimensions, and
// applies any overrides from `shape.layout()` (custom tiles are ignored for
// dynamic on-device shapes).
absl::StatusOr<std::vector<xla::Layout>> ResolveXlaLayouts(
    absl::Span<const Shape> shapes);

// Returns `shape.layout()` if it differs from the device's default layout for
// `shape`, or `std::nullopt` if `shape` uses the default layout. For dynamic
// on-device shapes whose `minor_to_major` differs from the default, tiles are
// omitted from the returned `CustomLayout`.
absl::StatusOr<std::optional<CustomLayout>>
ExtractCustomLayoutIfDifferentFromDefault(const Shape& shape);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_LAYOUT_UTILS_H_
