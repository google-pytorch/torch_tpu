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

#ifndef TORCH_TPU_CSRC_OPS_TO_COPY_TO_COPY_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_TO_COPY_TO_COPY_ATEN_KERNELS_H_

#include <optional>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/Device.h"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

// Copies self into a new TPU tensor.
//
// If memory_format is Preserve, the strides and storage offset
// are preserved if self is non-overlapping and dense. If memory_format is
// Preserve but self is overlapping or non-dense, the returned tensor will have
// the same dimensions as self, but the stride pattern will be dense and
// non-overlapping, and the offset will be 0.
absl::StatusOr<at::Tensor> CopyToTpuTensor(
    const at::Tensor& self, std::optional<at::ScalarType> dtype = std::nullopt,
    bool non_blocking = false,
    std::optional<at::MemoryFormat> memory_format = std::nullopt,
    std::optional<c10::DeviceIndex> tpu_device_index = std::nullopt);

[[nodiscard]] at::Tensor AtenToCopy(
    const at::Tensor& self, std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout, std::optional<at::Device> device,
    std::optional<bool> pin_memory, bool non_blocking,
    std::optional<at::MemoryFormat> memory_format);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_TO_COPY_TO_COPY_ATEN_KERNELS_H_
