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

#ifndef TORCH_TPU_CSRC_OPS_COPY_FROM_TPU_TO_CPU_H_
#define TORCH_TPU_CSRC_OPS_COPY_FROM_TPU_TO_CPU_H_

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/ops/copy_from/copy_common.h"

namespace torch_tpu {

// Writes the values of src (on TPU) to dest (on CPU).
// src and dest may have different dtypes.
// src and dest may have different shapes as long as src is broadcastable to
// dest.
// If transfer_sync_mode is kEagerNonBlocking, this will return without blocking
// the caller if possible; however if dest is non-dense, a blocking copy will be
// required.
// If dest is in pinned memory, the destination buffer will be kept
// alive until the copy completes; if dest is not in pinned memory, the caller
// must ensure that the destination buffer is not dropped before the copy
// completes.
absl::Status CopyTpuToCpu(const at::Tensor& src, const at::Tensor& dest,
                          TransferSyncMode transfer_sync_mode);

// Writes the values of src_buffer_ref (on TPU) to a new, contiguous CPU tensor.
absl::StatusOr<at::Tensor> CopyDeviceBufferRefToCpuTensor(
    const DeviceBufferRef& src_buffer_ref, TransferSyncMode transfer_sync_mode);

// Writes the values of src_buffer_ref (on TPU) to the given CPU pointer.
// The caller is responsible for allocating sufficient memory for dest_ptr, and
// for ensuring that the pointer is not dropped before the copy completes if
// using eager non-blocking mode.
absl::Status CopyDeviceBufferRefToCpuPtr(const DeviceBufferRef& src_buffer_ref,
                                         void* dest_ptr,
                                         TransferSyncMode transfer_sync_mode);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_COPY_FROM_TPU_TO_CPU_H_
