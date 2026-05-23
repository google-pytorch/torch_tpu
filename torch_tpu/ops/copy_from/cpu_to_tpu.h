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

#ifndef TORCH_TPU_OPS_COPY_CPU_TO_TPU_H_
#define TORCH_TPU_OPS_COPY_CPU_TO_TPU_H_

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "torch_tpu/eager/device_buffer.h"

namespace torch_tpu {

// Copies the value out of the source CPU tensor to a materialized
// DeviceBufferRef.
absl::StatusOr<DeviceBufferRef> CopyCpuToTpuBuffer(const at::Tensor& src,
                                                   bool non_blocking);

// Copies the value out of the source CPU tensor and assigns it to the
// destination TPU tensor.
absl::Status CopyCpuToTpu(const at::Tensor& src, const at::Tensor& dest,
                          bool non_blocking);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_COPY_CPU_TO_TPU_H_
