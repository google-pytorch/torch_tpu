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

#include "csrc/ops/copy_from/tpu_to_cpu.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "c10/core/DeviceGuard.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/reroute_warnings.h"
#include "csrc/common/status_builder.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/current_stream.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/materialize.h"
#include "csrc/eager/split_traversal.h"
#include "csrc/eager/structured_log_buffer.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/copy_from/copy_common.h"
#include "csrc/ops/copy_from/tpu_to_tpu.h"
#include "csrc/pjrt/pjrt_transfer.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/future.h"

namespace torch_tpu {

namespace {

// Translates a low-level XLA OOM error to an error that is easier to understand
// by a pytorch user. If the status is not an XLA OOM error, returns it as is.
//
// `elem_type` is the element type of the tensor that caused the XLA OOM error.
// `dims` is the dimensions of the tensor that caused the XLA OOM error.
static absl::Status TranslateXlaTensorOomError(absl::Status status,
                                               at::ScalarType elem_type,
                                               absl::Span<const int64_t> dims) {
  if (status.ok()) {
    return status;
  }

  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=All PyTorch dtypes are supported
                        // by mlir::ElementType.
      const auto dtype, ConvertTo<mlir::ElementType>(elem_type));
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=IsXlaOomError doesn't match format of
                 // compile-time OOM errors
      !IsXlaOomError(status), error::kResourceExhausted)
      << "the TPU ran out of memory while awaiting the materialization of "
         "value "
      << ToString(dtype) << "[" << absl::StrJoin(dims, ", ") << "]:\n"
      << status.message();

  // From this point, we know that the given status is not an XLA OOM error.
  return status;
}

// Ideal case: both src and dest are contiguous, tensors of the same shape and
// dtype, and src's storage buffer has exactly the right amount of data to copy
// into dest.
bool CanCopyStorageDirect(const at::Tensor& src, const at::Tensor& dest,
                          mlir::ElementType src_base_element_type) {
  if (src.scalar_type() != dest.scalar_type()) {
    // If the dtypes are different, we need to do a type conversion (not a
    // bitcast).
    return false;
  }

  if (src.sizes() != dest.sizes()) {
    // src needs to be broadcasted (either on TPU or CPU) to supply enough data
    // for dest.
    return false;
  }

  if (src.storage_offset() != 0) {
    // A storage copy will always start from an offset of 0 on the TPU side;
    // if src's storage offset is non-zero, a storage copy would move
    // out-of-scope data incorrectly, so we need a slice view.
    // Note that we don't care about dest's storage offset; CPU tensors will
    // automatically update Tensor::data_ptr() to reflect the offset.
    return false;
  }
  if (!src.is_contiguous() || !dest.is_contiguous()) {
    // dest must be contiguous, so that we can copy contiguous bytes into it
    // and ensure they are read properly.
    return false;
  }

  if (src_base_element_type == mlir::ElementType::PRED) {
    // Boolean data data is held in a 1-bit representation on TPU, but an 8-bit
    // representation on CPU. However, PJRT will copy booleans as bytes, in the
    // same representation as PyTorch.
    // So as long as we have 1 boolean per byte to write, we can do a direct
    // copy.
    return src.storage().nbytes() * 8 == dest.nbytes();
  }

  // For all element types of byte size or larger, the PjRt and PyTorch
  // representations are the same.
  // This means that we can do a direct storage copy if the number of bytes
  // sent matches the number of bytes needed.
  return src.storage().nbytes() == dest.nbytes();
}

// Materializes src_buffer_ref and copies it directly into dest_ptr.
absl::Status CopyDeviceBufferRefToCpuPtrImpl(
    const DeviceBufferRef& src_buffer_ref, void* dest_ptr,
    TransferSyncMode transfer_sync_mode) {
  if (src_buffer_ref.size_bytes() == 0) {
    // Copying nothing is a no-op.
    return absl::OkStatus();
  }
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=unreachable from Python
      dest_ptr != nullptr, error::kInvalidArgument)
      << "cannot copy a nonzero amount of data to a null pointer";
  TT_RET_CHECK(!src_buffer_ref.is_placeholder(), error::kInternal)
      << "cannot Materialize() a placeholder tensor or a tensor that "
         "depends on a placeholder. \nPlaceholder tensors should only "
         "appear in compiled mode, which should never try to "
         "materialize tensors";

  if (transfer_sync_mode == TransferSyncMode::kIsolated) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=not reachable from Python
        src_buffer_ref.is_materializing(), error::kFailedPrecondition)
        << "cannot materialize a tensor during a stream-isolated copy, "
           "materialization requires an active stream";
    xla::Future<> future = TpuMemcpyDtoHDirect(src_buffer_ref, dest_ptr);
    return future.Await();
  }

  if (transfer_sync_mode == TransferSyncMode::kEagerNonBlocking) {
    // Start a background materialization of src_buffer_ref (unless it's already
    // started), then start a non-blocking copy and record it on the stream.
    TT_RETURN_IF_ERROR(TranslateXlaTensorOomError(
        Materialize(src_buffer_ref, MaterializationReason::kCpuTransfer,
                    GraphSplitMode::kSplitGraph,
                    MaterializationBlockMode::kNonBlocking),
        ConvertTo<at::ScalarType>(src_buffer_ref.element_type()),
        src_buffer_ref.dimensions()));
    xla::Future<> future = TpuMemcpyDtoHDirect(src_buffer_ref, dest_ptr);
    RecordAsyncDeviceToHost(std::move(future));
    return absl::OkStatus();
  }

  // In eager blocking mode, start a full stream materialization (ensuring that
  // the copied buffer will be materialized), and then block until the event
  // snapshot is complete. This will await both the materialization and the
  // copy.
  xla::Future<> future = TpuMemcpyDtoHDirect(src_buffer_ref, dest_ptr);
  RecordAsyncDeviceToHost(std::move(future));
  std::vector<SharedDeviceBufferList> must_materialize = {
      src_buffer_ref.device_buffer_list()};
  xla::Future<std::shared_ptr<EventSnapshot>> snapshot_future =
      MaterializeCurrentStream(MaterializationReason::kCpuTransfer,
                               GraphSplitMode::kSplitGraph, must_materialize);
  TT_ASSIGN_OR_RETURN(std::shared_ptr<EventSnapshot> snapshot,
                      snapshot_future.Await());
  TT_RETURN_IF_ERROR(snapshot->Synchronize());
  return SyncAndReturnStickyError();
}

// Materializes the base buffer of src, and copies it byte-for-byte into dest's
// storage. This can be done asynchronously and does not materialize any
// ephemeral buffers.
absl::Status CopyTpuToCpuStorageDirect(const at::Tensor& src,
                                       const at::Tensor& dest,
                                       TransferSyncMode transfer_sync_mode) {
  TT_ASSIGN_OR_RETURN(DeviceBufferRef base_buf, GetBaseBuffer(src));
  ABSL_DCHECK(CanCopyStorageDirect(src, dest, base_buf.element_type()));
  return CopyDeviceBufferRefToCpuPtrImpl(base_buf, dest.data_ptr(),
                                         transfer_sync_mode);
}

// Converts a TPU source tensor into a contiguous DeviceBufferRef with the same
// size, shape, and dtype as a CPU destination tensor, and then copies it to the
// destination's storage.
absl::Status CopyTpuToCpuWithTpuConversion(
    const at::Tensor& src, const at::Tensor& dest,
    mlir::ElementType src_base_element_type,
    TransferSyncMode transfer_sync_mode) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=checked in "to" or "copy_" first
      src.dim() <= dest.dim(), error::kInvalidArgument)
      << "source tensor is not broadcastable to destination tensor; source "
         "tensor has "
      << src.dim() << " dimensions, destination tensor has " << dest.dim()
      << " dimensions";
  // Ensure the base buffer will materialize before we do any further work on
  // the current stream that depends on it.
  // If the base buffer is on this stream, this is guaranteed; no explicit
  // materialization is needed.
  // But if it's on another stream, we need to force that stream to sync and
  // execute the base before any views or copies we do here.
  TT_ASSIGN_OR_RETURN(DeviceBufferRef src_base_buf, GetBaseBuffer(src));
  const auto [curr_device_index, curr_stream_id] = GetCurrentDeviceStreamId();
  if (!src_base_buf.is_materializing() &&
      (src_base_buf.device_index() != curr_device_index ||
       src_base_buf.stream_id() != curr_stream_id)) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=not reachable from Python
        transfer_sync_mode != TransferSyncMode::kIsolated,
        error::kFailedPrecondition)
        << "cannot materialize a tensor during a stream-isolated copy, "
           "materialization requires an active stream";
    tsl::profiler::TraceMe trace_mat("CopyTpuToCpu::MaterializeBase");
    TT_RETURN_IF_ERROR(TranslateXlaTensorOomError(
        Materialize(src_base_buf, MaterializationReason::kCpuTransfer,
                    GraphSplitMode::kSplitGraph,
                    MaterializationBlockMode::kBlockUntilStarted),
        ConvertTo<at::ScalarType>(src_base_buf.element_type()),
        src_base_buf.dimensions()));
  }

  // Broadcast src to dest, if needed.
  // Transpose src and dest so that dest is contiguous, if needed.
  at::Tensor viewed_src = src;
  at::Tensor viewed_dest = dest;
  TT_RETURN_IF_ERROR(BroadcastAndTranspose(viewed_src, viewed_dest));
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=defensive check of logic in
                 // aten::is_non_overlapping_and_dense
      viewed_dest.is_contiguous(), error::kInternal)
      << "expected a non-overlapping, dense tensor to be contiguous after "
         "row-major transpose, got sizes: "
      << ToString(viewed_dest.sizes())
      << " and strides: " << ToString(viewed_dest.strides())
      << " which are not contiguous";

  if (CanCopyStorageDirect(viewed_src, viewed_dest, src_base_element_type)) {
    // In the case where src and dest are identical transposes of a contiguous
    // storage (no broadcast or type conversion) then after the transpose, we
    // can copy directly from src to dest.
    return CopyTpuToCpuStorageDirect(viewed_src, viewed_dest,
                                     transfer_sync_mode);
  }

  // Convert the source tensor to the destination tensor's dtype, if needed.
  at::Tensor converted_src = viewed_src;
  if (viewed_src.scalar_type() != dest.scalar_type()) {
    TT_ASSIGN_OR_RETURN(
        converted_src,
        MakeEmptyMemoryFormat(viewed_dest.sizes(), dest.scalar_type(),
                              /*layout_opt=*/std::nullopt, viewed_src.device(),
                              /*pin_memory_opt=*/false,
                              at::MemoryFormat::Contiguous));
    TT_RETURN_IF_ERROR(CopyTpuToTpu(viewed_src, converted_src));
  }

  // Start or schedule a copy from the TPU buffer into the destination CPU
  // storage. Because we only applied as_strided to dest, the storage data_ptr
  // is unchanged.
  TT_ASSIGN_OR_RETURN(DeviceBufferRef converted_src_buf,
                      GetBuffer(converted_src));
  return CopyDeviceBufferRefToCpuPtrImpl(converted_src_buf, dest.data_ptr(),
                                         transfer_sync_mode);
}

// PJRT can only asynchronously copy from a TPU tensor into contiguous CPU
// storage. Therefore, if the destination tensor is not contiguous, we must do
// a two step blocking copy:
//  1. Copy from the source TPU tensor into a temporary contiguous CPU buffer
//  2. Copy from the contiguous CPU buffer into the destination CPU buffer.
//
// We also use this case for blocking copies that require broadcasting or
// type conversion; doing the broadcast after the copy minimizes the amount of
// data transferred, and does conversions (both bitcasting from the base buffer
// to the src dtype, and non-bitcast conversions from src to dest dtypes) on
// host where they tend to be faster/more efficient.
absl::Status CopyTpuToCpuWithCpuConversion(
    const at::Tensor& src, const at::Tensor& dest,
    TransferSyncMode transfer_sync_mode) {
  // Materialize the contiguous base buffer of the source tensor and copy it to
  // a temporary CPU tensor. This must be blocking to avoid freeing cpu_temp
  // while it is still being used.
  TT_ASSIGN_OR_RETURN(DeviceBufferRef base_buf, GetBaseBuffer(src));
  if (transfer_sync_mode == TransferSyncMode::kEagerNonBlocking) {
    transfer_sync_mode = TransferSyncMode::kEagerBlocking;
  }
  TT_ASSIGN_OR_RETURN(at::Tensor cpu_temp, CopyDeviceBufferRefToCpuTensor(
                                               base_buf, transfer_sync_mode));

  // View the source Storage on host in the same way as on TPU.
  auto* cpu_temp_impl = cpu_temp.unsafeGetTensorImpl();
  cpu_temp_impl->set_storage_and_dtype(cpu_temp.storage(), src.dtype());
  cpu_temp_impl->set_sizes_and_strides(src.sizes(), src.strides(),
                                       src.storage_offset());

  // Then do a CPU-to-CPU copy from our temp tensor to the destination.
  // This will do three things:
  //   1. If necessary, broadcast from src.sizes() to dest.sizes().
  //   2. If necessary, convert (not a bitcast) from src.scalar_type() to
  //      dest.scalar_type().
  //   3. If dest is not contiguous and dense (e.g. it is a slice or has a
  //      non-row-major layout like ChannelsLast3D), this will copy it into
  //      the correct positions in dest's existing Storage.
  // This must be a blocking copy because as soon as this function returns,
  // cpu_temp will be deallocated.
  dest.copy_(cpu_temp, /*non_blocking=*/false);
  return absl::OkStatus();
}

}  // namespace

absl::Status CopyTpuToCpu(const at::Tensor& src, const at::Tensor& dest,
                          TransferSyncMode transfer_sync_mode) {
  tsl::profiler::TraceMe trace("CopyTpuToCpu");
  ABSL_VLOG(1) << "[AtenCopyFrom] TPU -> CPU copy path for "
               << ToString(src, "src");
  // Surface warnings rerouted from background threads that the copy waited on,
  // also on errors.
  absl::Cleanup cleanup = [] { SurfaceReroutedWarnings(); };
  // Blocking copies synchronize the current stream, which must be the stream
  // of the source tensor's device rather than the caller's current device.
  c10::DeviceGuard device_guard(src.device());

  if (src.numel() == 0) {
    // Shortcut: no data to move. Create a new storage for dest if it doesn't
    // have one, otherwise leave it as is.
    if (!dest.has_storage()) {
      at::Tensor empty_dest = at::empty(dest.sizes(), dest.options());
      dest.unsafeGetTensorImpl()->set_storage_keep_dtype(empty_dest.storage());
    }
    return absl::OkStatus();
  }
  TT_ASSIGN_OR_RETURN(DeviceBufferRef src_buf, GetBaseBuffer(src));

  if (CanCopyStorageDirect(src, dest, src_buf.element_type())) {
    // Ideal case: copy directly from TPU to CPU storage with no materialized
    // view, with async capability.
    return CopyTpuToCpuStorageDirect(src, dest, transfer_sync_mode);
  }

  if (!dest.is_non_overlapping_and_dense()) {
    // If we need to write non-consecutive bytes, we must do a blocking copy
    // with CPU conversion, even if the user requested a non-blocking copy,
    // because we cannot write consecutive bytes into the dest directly.
    return CopyTpuToCpuWithCpuConversion(src, dest, transfer_sync_mode);
  }

  if (transfer_sync_mode == TransferSyncMode::kEagerNonBlocking) {
    // If we can do a non-blocking copy and the user requested it, we do so.
    // This will materialize a view tensor with the size, shape, and dtype of
    // dest, and copy into its storage.
    return CopyTpuToCpuWithTpuConversion(src, dest, src_buf.element_type(),
                                         transfer_sync_mode);
  }

  // This is a blocking copy which does not require a CPU conversion, but we
  // could still do one if it would be more efficient.
  // If we materialize the view on TPU, then the transfer cost is the byte size
  // of the destination tensor we write into.
  // Note: src.nbytes() uses PyTorch's element size.
  int64_t tpu_conversion_transfer_cost = dest.nbytes();
  if (dest.scalar_type() == at::ScalarType::Float4_e2m1fn_x2) {
    // Even though both PyTorch and XLA use 4 bits per FP4 element, during
    // transfer they are moved as 1 byte per element, doubling the cost.
    tpu_conversion_transfer_cost *= 2;
  }

  // If we copy the entire storage buffer to host, then the transfer cost is the
  // byte size of the underlying storage. This is more efficient for broadcasts
  // of small tensors.
  // Note: src_buf.size_bytes() uses XLA's element size.
  int64_t cpu_conversion_transfer_cost = src_buf.size_bytes();
  if (src_buf.element_type() == mlir::ElementType::F4E2M1FN) {
    // Even though both PyTorch and XLA use 4 bits per FP4 element, during
    // transfer they are moved as 1 byte per element, doubling the cost.
    cpu_conversion_transfer_cost *= 2;
  } else if (src_buf.element_type() == mlir::ElementType::PRED) {
    // XLA uses 1 bit per PRED element, but transfers them as 1 byte per
    // element.
    cpu_conversion_transfer_cost *= 8;
  }

  // In case of a tie, we prefer to use a CPU conversion, as this requires less
  // work on the TPU; view operations are usually free on CPU.
  if (tpu_conversion_transfer_cost < cpu_conversion_transfer_cost) {
    return CopyTpuToCpuWithTpuConversion(src, dest, src_buf.element_type(),
                                         transfer_sync_mode);
  }
  return CopyTpuToCpuWithCpuConversion(src, dest, transfer_sync_mode);
}

absl::Status CopyDeviceBufferRefToCpuPtr(const DeviceBufferRef& src_buffer_ref,
                                         void* dest_ptr,
                                         TransferSyncMode transfer_sync_mode) {
  return CopyDeviceBufferRefToCpuPtrImpl(src_buffer_ref, dest_ptr,
                                         transfer_sync_mode);
}

absl::StatusOr<at::Tensor> CopyDeviceBufferRefToCpuTensor(
    const DeviceBufferRef& src_buffer_ref,
    TransferSyncMode transfer_sync_mode) {
  const at::ScalarType src_scalar_type =
      ConvertTo<at::ScalarType>(src_buffer_ref.element_type());
  at::IntArrayRef contiguous_dims = src_buffer_ref.dimensions();

  // Explanation:
  // One logical element of mlir::ElementType::F4E2M1FN is a single,
  // four-bit value. This is stored in one byte on TPU.
  // But one logical element of at::ScalarType::Float4_e2m1fn_x2; this is also
  // stored as one byte, but represents two 4-bit values.
  // Using at::empty(src_buffer_ref.dimensions()) would allocate twice as much
  // memory as needed.
  Dimensions fp4_adjusted_dims;
  if (src_scalar_type == at::ScalarType::Float4_e2m1fn_x2) {
    ABSL_CHECK(  // CRASH_OK=required representation of Float4_e2m1fn_x2 value
        !contiguous_dims.empty() && contiguous_dims.back() % 2 == 0);
    fp4_adjusted_dims = CopyIntVector(contiguous_dims);
    fp4_adjusted_dims.back() /= 2;
    contiguous_dims = fp4_adjusted_dims;
  }

  at::Tensor dest = at::empty(contiguous_dims,
                              at::TensorOptions()
                                  .dtype(src_scalar_type)
                                  .device(at::DeviceType::CPU)
                                  .memory_format(at::MemoryFormat::Contiguous));
  TT_RETURN_IF_ERROR(CopyDeviceBufferRefToCpuPtrImpl(
      src_buffer_ref, dest.data_ptr(), transfer_sync_mode));
  return dest;
}

}  // namespace torch_tpu
