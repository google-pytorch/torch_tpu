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

#include <memory>
#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/TensorOptions.h"
#include "csrc/common/context_manager.h"
#include "csrc/common/context_states.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/layout_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/copy_from/copy_common.h"
#include "csrc/pjrt/pjrt_transfer.h"
#include "csrc/pjrt/pjrt_utils.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "xla/layout.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

// Returns `tensor`'s layout on the TPU if it isn't the default for its shape,
// e.g. the SparseCore layout of an embedding table.
static std::optional<xla::Layout> GetNonDefaultLayout(
    const at::Tensor& tensor) {
  const absl::StatusOr<DeviceBufferRef> buffer = GetBuffer(tensor);
  // Buffers in pinned host memory have host layouts.
  if (!buffer.ok() || buffer->is_pinned_host() ||
      !buffer->shape().layout().has_value() ||
      IsDefaultDeviceLayout(*buffer->shape().layout(), buffer->element_type(),
                            buffer->dimensions())) {
    return std::nullopt;
  }
  const absl::StatusOr<xla::Layout> default_layout =
      GetDefaultDeviceLayout(buffer->element_type(), buffer->dimensions());
  if (!default_layout.ok()) return std::nullopt;
  // The buffer's layout records the dtype's width as its element size. Like
  // LayoutContext, use the default element size instead.
  xla::Layout layout = ToXlaLayout(*buffer->shape().layout());
  layout.set_element_size_in_bits(default_layout->element_size_in_bits());
  return layout;
}

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

  // The copy gives `dest` a new buffer. Unless a LayoutContext is active,
  // create it in `dest`'s current layout, so that e.g. load_state_dict() keeps
  // custom layouts.
  std::optional<xla::Layout> dest_layout;
  if (!GetContextState<LayoutContextState>().has_value()) {
    dest_layout = GetNonDefaultLayout(dest);
  }
  if (dest_layout.has_value()) {
    PushContextState<LayoutContextState>(
        std::make_shared<const xla::Layout>(*dest_layout));
  }
  absl::Cleanup exit_layout_context = [&dest_layout] {
    if (dest_layout.has_value()) PopContextState<LayoutContextState>();
  };

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
