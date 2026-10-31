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
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/context_manager.h"
#include "csrc/common/context_states.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/layout_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/host_allocator.h"
#include "csrc/eager/materialize.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/copy_from/copy_common.h"
#include "csrc/ops/copy_from/tpu_to_tpu.h"
#include "csrc/pjrt/pjrt_transfer.h"
#include "csrc/pjrt/pjrt_utils.h"
#include "xla/layout.h"

namespace torch_tpu {
namespace {

// Returns `tensor`'s layout on the TPU if it isn't the default for its shape,
// e.g. the SparseCore layout of an embedding table.
std::optional<xla::Layout> GetNonDefaultLayout(const at::Tensor& tensor) {
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

// Returns whether to convert `src` to `dest`'s dtype on the host, so that the
// copy keeps `dest`'s layout. Otherwise, a TPU-to-TPU copy converts it, which
// gives `dest` a buffer in the default layout.
bool ShouldConvertOnHostToKeepLayout(const at::Tensor& src,
                                     const at::Tensor& dest) {
  // FP4 can't be converted on the host.
  return src.sizes() == dest.sizes() &&
         src.scalar_type() != dest.scalar_type() &&
         src.scalar_type() != at::kFloat4_e2m1fn_x2 &&
         dest.scalar_type() != at::kFloat4_e2m1fn_x2 &&
         !GetContextState<LayoutContextState>().has_value() &&
         GetNonDefaultLayout(dest).has_value();
}

}  // namespace

absl::Status CopyCpuToTpu(const at::Tensor& src, const at::Tensor& dest,
                          TransferSyncMode transfer_sync_mode) {
  ABSL_VLOG(1) << "[AtenCopyFrom] CPU -> TPU copy path for "
               << ToString(src, "src");
  if (src.numel() == 0) {
    // Copying nothing is a no-op.
    return absl::OkStatus();
  }

  if (ShouldConvertOnHostToKeepLayout(src, dest)) {
    return CopyCpuToTpu(src.to(dest.scalar_type()), dest, transfer_sync_mode);
  }

  // In the most common case of `.to("tpu")`, src and dest will be the same size
  // and dtype. However, `copy_()` allows broadcasting and dtype conversion.
  // In these cases, we do a CPU->TPU copy with no broadcasting or dtype
  // conversion, and then do a TPU->TPU copy to the target shape and dtype.
  // The TPU->TPU copy will handle broadcasting and dtype conversion in a
  // non-blocking fashion.
  if (src.sizes() != dest.sizes() || src.scalar_type() != dest.scalar_type()) {
    TT_ASSIGN_OR_RETURN(
        at::Tensor src_tpu,
        MakeEmptyTensor(src.sizes(), src.scalar_type(), dest.device()));
    TT_RETURN_IF_ERROR(CopyCpuToTpu(src, src_tpu, transfer_sync_mode));
    return CopyTpuToTpu(src_tpu, dest);
  }

  // Ensure the source tensor is contiguous.
  at::Tensor effective_src = src.contiguous();

  // Ensure that we're copying from pinned memory if the copy is non-blocking.
  // This is reference-counted to prevent use-after-free if the user drops the
  // source tensor before the copy completes.
  // If the copy is blocking, then the copy will complete before we drop
  // effective_src, so we don't need to pin.
  if (transfer_sync_mode == TransferSyncMode::kEagerNonBlocking &&
      !IsTpuPinnedPtr(effective_src.storage().data_ptr().get())) {
    effective_src = effective_src.pin_memory(dest.device());
  }

  // The copy gives `dest` a new buffer. Unless a LayoutContext is active,
  // create it in `dest`'s current layout, so that e.g. load_state_dict() keeps
  // custom layouts.
  bool dest_layout_pushed = false;
  if (!GetContextState<LayoutContextState>().has_value()) {
    std::optional<xla::Layout> non_default_layout = GetNonDefaultLayout(dest);
    if (non_default_layout.has_value()) {
      PushContextState<LayoutContextState>(
          std::make_shared<const xla::Layout>(std::move(*non_default_layout)));
      dest_layout_pushed = true;
    }
  }
  absl::Cleanup exit_layout_context = [dest_layout_pushed] {
    if (dest_layout_pushed) PopContextState<LayoutContextState>();
  };

  TT_ASSIGN_OR_RETURN(TpuLayout tpu_layout, ResolveTpuLayout(effective_src));

  // NOTE: we are intentionally copying from Tensor::data_ptr() here,
  // and **not** Tensor::storage().data_ptr().get().
  // On CPU, Tensor::data_ptr() will account for the storage offset, while
  // Tensor::storage().data_ptr().get() will not.
  // However, we also need to pass in the storage pointer to indicate that the
  // host buffer should be kept alive during the transfer if it's pinned.
  auto transfer = HostToDeviceTransfer::FromTensor(
      effective_src.data_ptr(), effective_src.storage().data_ptr().get(),
      tpu_layout.element_type, tpu_layout.sizes);
  if (transfer_sync_mode == TransferSyncMode::kIsolated) {
    // Do the transfer immediately and block on completion; don't touch any
    // streams.
    TT_ASSIGN_OR_RETURN(DeviceBufferRef tpu_buf, transfer.Run());
    TT_RETURN_IF_ERROR(AdaptXlaError(
        tpu_buf.Synchronize(),
        /* context= */ "failed to copy buffer from host to device"));
    TT_RETURN_IF_ERROR(AssignBufferToAtTensor(std::move(tpu_buf), dest));
    return absl::OkStatus();
  }

  // Schedule the transfer on the stream in either kEagerBlocking or
  // kEagerNonBlocking mode, but only block if kEagerBlocking.
  TT_ASSIGN_OR_RETURN(DeviceBufferRef tpu_buf,
                      ScheduleHostToDeviceTransfer(std::move(transfer)));
  TT_RETURN_IF_ERROR(AssignBufferToAtTensor(std::move(tpu_buf), dest));

  if (transfer_sync_mode == TransferSyncMode::kEagerBlocking) {
    TT_ASSIGN_OR_RETURN(
        std::shared_ptr<EventSnapshot> snapshot,
        MaterializeCurrentStream(MaterializationReason::kCpuTransfer).Await());
    TT_RETURN_IF_ERROR(snapshot->Synchronize());
    TT_RETURN_IF_ERROR(SyncAndReturnStickyError());
  }
  return absl::OkStatus();
}

}  // namespace torch_tpu
