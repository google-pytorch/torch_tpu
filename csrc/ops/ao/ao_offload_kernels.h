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

#ifndef TORCH_TPU_CSRC_OPS_AO_AO_OFFLOAD_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_AO_AO_OFFLOAD_KERNELS_H_

#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "c10/core/Device.h"

namespace torch_tpu {

// Activation offload kernel. Emits annotate_device_placement with placement
// "pinned_host".
at::Tensor AtenAoOffload(const at::Tensor& tensor);

// Activation reload kernel. Emits annotate_device_placement with placement
// "device".
at::Tensor AtenAoReload(const at::Tensor& tensor, at::Device device,
                        at::OptionalIntArrayRef original_size = std::nullopt,
                        at::OptionalIntArrayRef original_stride = std::nullopt);

// Activation wait_tensor kernel (no-op pass-through).
at::Tensor AtenAoWaitTensor(
    const at::Tensor& tensor,
    const std::optional<at::Tensor>& keepalive = std::nullopt,
    const std::optional<at::Tensor>& last_use_of_storage = std::nullopt);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_AO_AO_OFFLOAD_KERNELS_H_
