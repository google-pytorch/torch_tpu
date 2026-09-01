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

#ifndef TORCH_TPU_OPS_TRIL_INDICES_TRIL_INDICES_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_TRIL_INDICES_TRIL_INDICES_ATEN_KERNELS_H_

#include <cstdint>
#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "c10/core/Device.h"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

// aten::tril_indices
at::Tensor AtenTrilIndices(int64_t row, int64_t col, int64_t offset,
                           std::optional<at::ScalarType> dtype_opt,
                           std::optional<at::Layout> layout_opt,
                           std::optional<at::Device> device_opt,
                           std::optional<bool> pin_memory_opt);
}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_TRIL_INDICES_TRIL_INDICES_ATEN_KERNELS_H_
