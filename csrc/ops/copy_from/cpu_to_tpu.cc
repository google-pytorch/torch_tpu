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

#include "csrc/ops/copy_from/cpu_to_tpu.h"

#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "c10/core/TensorOptions.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/layout_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/copy_from/copy_common.h"
#include "csrc/pjrt/pjrt_transfer.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

absl::Status CopyCpuToTpu(const at::Tensor& src, const at::Tensor& dest,
                          TransferSyncMode transfer_sync_mode) {
  ABSL_VLOG(1) << "[AtenCopyFrom] CPU -> TPU copy path for "
               << ToString(src, "src");
  if (src.numel() == 0) {
    // Copying nothing is a no-op.
    return absl::OkStatus();
  }

  // Ensure the source tensor has the correct dtype and is contiguous.
  at::Tensor effective_src = src;
  if (src.scalar_type() != dest.scalar_type() || !src.is_contiguous()) {
    effective_src = src.to(at::TensorOptions()
                               .dtype(dest.scalar_type())
                               .memory_format(at::MemoryFormat::Contiguous),
                           /*non_blocking=*/false, /*copy=*/true);
  }
  if (transfer_sync_mode == TransferSyncMode::kEagerNonBlocking &&
      effective_src.data_ptr() != src.data_ptr()) {
    // If to() created a copy, then effective_src will be
    // dropped at the end of this function, invalidating the non-blocking
    // copy. We must do a blocking copy to avoid this.
    // TODO(bawilson): implement a non-blocking copy using a CPU->TPU->TPU path
    transfer_sync_mode = TransferSyncMode::kIsolated;
  }

  TT_ASSIGN_OR_RETURN(TpuLayout tpu_layout, ResolveTpuLayout(effective_src));

  // NOTE: we are intentionally using Tensor::data_ptr() here,
  // and **not** Tensor::storage().data_ptr().get().
  // On CPU, Tensor::data_ptr() will account for the storage offset, while
  // Tensor::storage().data_ptr().get() will not.
  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef tpu_buf,
      TpuMallocAndMemcpyHtoD(effective_src.data_ptr(), tpu_layout.element_type,
                             tpu_layout.sizes));
  if (transfer_sync_mode == TransferSyncMode::kIsolated) {
    TT_RETURN_IF_ERROR(AdaptXlaError(
        tpu_buf.Synchronize(),
        /* context= */ "failed to copy buffer from host to device"));
  } else {
    RecordAsyncHostToDevice(tpu_buf);
  }
  TT_RETURN_IF_ERROR(AssignBufferToAtTensor(tpu_buf, dest));
  return absl::OkStatus();
}

}  // namespace torch_tpu
