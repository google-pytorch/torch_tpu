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

#ifndef TORCH_TPU_CSRC_PJRT_PJRT_TRANSFER_H_
#define TORCH_TPU_CSRC_PJRT_PJRT_TRANSFER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/context_manager.h"
#include "csrc/common/context_states.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

// A host-to-device transfer.
class HostToDeviceTransfer {
 public:
  // Defines a host-to-device transfer from a zero-filled buffer.
  // Constructs a bytes vector of the required size and calls FromBytesVector.
  static HostToDeviceTransfer FromZeroes(mlir::ElementType element_type,
                                         absl::Span<const int64_t> dimensions);

  // Defines a host-to-device transfer from a tensor's data pointer and storage
  // pointer.
  // If the element type is FP4E2M1FN, the tensor data will be unpacked and will
  // delegate to FromBytesVector.
  // If tensor_storage_ptr is a pinned pointer, it will be used to keep the
  // host data alive until the transfer completes.
  static HostToDeviceTransfer FromTensor(
      const void* absl_nonnull tensor_data_ptr,
      const void* absl_nonnull tensor_storage_ptr,
      mlir::ElementType element_type, absl::Span<const int64_t> dimensions);

  // Defines a host-to-device transfer from a bytes vector. The backing bytes
  // are kept alive until the transfer completes.
  static HostToDeviceTransfer FromBytesVector(
      std::vector<uint8_t> backing_data, mlir::ElementType element_type,
      absl::Span<const int64_t> dimensions);

  // Starts the transfer, and returns the DeviceBufferRef that will hold the
  // transferred data.
  // Idempotent; repeated calls will return the same DeviceBufferRef.
  absl::StatusOr<DeviceBufferRef> Run();

  // Waits until all of the futures in wait_for are ready, then starts the
  // transfer and returns the DeviceBufferRef that will hold the transferred
  // data.
  // Subsequent calls to RunAfter will fail if wait_for is not empty, as the
  // wait condition cannot be modified once set. Subsequent calls to Run (
  // or RunAfter with an empty span) are idempotent and return the same
  // DeviceBufferRef.
  absl::StatusOr<DeviceBufferRef> RunAfter(
      absl::Span<const xla::Future<void>> wait_for);

 private:
  // Private constructor. Use static methods to create HostToDevice objects.
  explicit HostToDeviceTransfer(const void* host_data,
                                const void* host_pinned_ptr,
                                mlir::ElementType element_type,
                                absl::Span<const int64_t> dimensions,
                                std::vector<uint8_t> backing_data)
      : impl_(std::make_unique<HostToDeviceTransferImpl>(
            host_data, host_pinned_ptr, element_type, CopyIntVector(dimensions),
            std::move(backing_data), GetContextState<LayoutContextState>())) {}

  // Helper function, common across multiple factories; unpacks FP4E2M1FN data
  // before construction.
  static std::vector<uint8_t> MakeUnpackedFp4(
      const void* host_data, absl::Span<const int64_t> dimensions);

  struct HostToDeviceTransferImpl {
    // The host data to be transferred. This will be nullptr if and only if
    // backing_data is non-empty.
    const void* host_data;
    // The pinned memory pointer, if one is provided. This is used to keep the
    // host data alive until the transfer completes.
    const void* absl_nullable host_pinned_ptr = nullptr;
    // The element type of the data to be transferred.
    mlir::ElementType element_type;
    // The dimensions of the data to be transferred.
    Dimensions dimensions;
    // The backing data for the host data, if one is provided.
    std::vector<uint8_t> backing_data;
    // The layout pulled from the context state. This is determined when the
    // transfer is defined, **not** when it is run.
    std::optional<std::shared_ptr<const xla::Layout>> layout = std::nullopt;
  };

  // Starts the transfer, and returns the PjRtBuffer that will contain the
  // transferred data.
  // This is the core of the HostToDeviceTransfer, and is used by both Run and
  // RunAfter.
  static absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> StartTransfer(
      absl_nonnull std::unique_ptr<HostToDeviceTransferImpl> impl);

  // The data for the transfer. Passed to a callback if RunAfter is used with
  // non-ready futures.
  absl_nullable std::unique_ptr<HostToDeviceTransferImpl> impl_;

  // The DeviceBufferRef that holds the transferred data.
  std::optional<DeviceBufferRef> device_buffer_ref_ = std::nullopt;
};

// A device-to-host transfer.
class DeviceToHostTransfer {
 public:
  // Defines a device-to-host transfer into a tensor. The bytes will be written
  // into tensor_data_ptr (which may include an offset); storage_data_ptr is
  // used to identify pinned memory and keep it alive until the transfer
  // completes. If the tensor is not in pinned memory, the caller is responsible
  // for keeping the storage alive until the transfer completes.
  static DeviceToHostTransfer IntoTensor(
      DeviceBufferRef buffer_ref, void* absl_nonnull tensor_data_ptr,
      const void* absl_nonnull storage_data_ptr);

  // Defines a device-to-host transfer into a byte buffer. The caller is
  // responsible for allocating a buffer of sufficient size to hold the data,
  // and for keeping it alive until the transfer completes.
  static DeviceToHostTransfer IntoBytes(DeviceBufferRef buffer_ref,
                                        void* absl_nonnull host_bytes_ptr);

  // Starts the transfer, or schedules it to run immediately after the buffer
  // materialization completes. Returns a future that will be set when the
  // transfer completes.
  // Idempotent; repeated calls will return the same future.
  xla::Future<> Run();

  // Starts the transfer, or schedules it to run immediately after all of the
  // futures in wait_for are ready and the buffer materialization completes.
  // Returns a future that will be set when the transfer completes.
  // Subsequent calls to RunAfter will fail if wait_for is not empty, as the
  // wait condition cannot be modified once set. Subsequent calls to Run (or
  // RunAfter with an empty span) are idempotent and return the same future.
  xla::Future<> RunAfter(absl::Span<const xla::Future<void>> wait_for);

 private:
  // Private constructor. Use static methods to create DeviceToHostTransfer
  // objects. dst_ptr is the final write destination; dst_pinned_ptr is the
  // storage pointer if one is provided through IntoTensor.
  explicit DeviceToHostTransfer(DeviceBufferRef buffer_ref,
                                void* absl_nonnull dst_ptr,
                                const void* absl_nullable dst_pinned_ptr);

  // Heap-allocated memory to hold state for the transfer. This is passed to
  // callback functions as the transfer progresses.
  struct DeviceToHostTransferImpl {
    // The device-side data being transferred.
    DeviceBufferRef buffer_ref;
    // If the transfer data type is FP4E2M1FN, this buffer will be used to
    // store the unpacked data on host, after which it will be packed and copied
    // to the destination pointer.
    std::vector<uint8_t> fp4_receiver;
    // The final destination pointer for the transfer.
    void* dst_ptr = nullptr;
    // The pinned memory pointer if one is provided through IntoTensor.
    const void* dst_pinned_ptr = nullptr;
    // The promise to be set when the transfer completes (or fails).
    xla::Promise<void> promise;
  };

  // Verifies that the materialization of the buffer_ref succeeded.
  // If ready_status is not ok, or if a valid, non-deleted PjRtBuffer cannot be
  // obtained, returns the error. Otherwise, returns a non-null, non-deleted
  // PjRtBuffer*.
  static absl::StatusOr<xla::PjRtBuffer* absl_nonnull> VerifyMaterialization(
      absl::Status ready_status, const DeviceToHostTransferImpl& impl);

  // A callback invoked when materialization has provided a PjRtBuffer to copy
  // from. This will start the transfer using ToLiteral and schedule the
  // callback OnToLiteralReady.
  static void OnTransferReady(
      absl::Status ready_status,
      absl_nonnull std::unique_ptr<DeviceToHostTransferImpl> impl);

  // A callback invoked when the PjRtBuffer has been copied to host memory.
  // This will then:
  //   1. Pack FP4 data if necessary.
  //   2. Format the transfer error using AdaptXlaError.
  //   3. Set the chained promise.
  //   4. Free the impl object, dropping the reference to the device buffer.
  static void OnToLiteralReady(
      absl::Status to_literal_status,
      absl_nonnull std::unique_ptr<DeviceToHostTransferImpl> impl);

  // The implementation state for the transfer. If this is null, the transfer
  // has already started and future_ can be used.
  absl_nullable std::unique_ptr<DeviceToHostTransferImpl> impl_;

  // The future that will be set when the transfer completes.
  xla::Future<void> future_;
};

// Owns a prepared PJRT raw buffer alias and its underlying DeviceBufferRef for
// zero-copy asynchronous DMA transfers.
//
// Each CopyDtoH / CopyHtoD call returns a self-retaining xla::Future<> whose
// completion callback keeps the underlying PJRT_RawBuffer* and DeviceBufferRef
// alive until the hardware DMA transfer finishes.
class PreparedRawDmaBuffer {
 public:
  // Awaits computation readiness on buffer_ref and creates a PJRT_RawBuffer
  // alias for direct DMA transfers.
  static absl::StatusOr<PreparedRawDmaBuffer> Prepare(
      DeviceBufferRef buffer_ref);

  // Initiates an asynchronous Device-to-Host raw DMA copy of transfer_size
  // bytes starting at offset in the device buffer into dst.
  absl::StatusOr<xla::Future<>> CopyDtoH(void* dst, int64_t offset,
                                         int64_t transfer_size) const;

  // Initiates an asynchronous Host-to-Device raw DMA copy of transfer_size
  // bytes from src into the device buffer starting at offset.
  absl::StatusOr<xla::Future<>> CopyHtoD(const void* src, int64_t offset,
                                         int64_t transfer_size) const;

 private:
  struct Holder;
  explicit PreparedRawDmaBuffer(std::shared_ptr<Holder> holder)
      : holder_(std::move(holder)) {}

  std::shared_ptr<Holder> holder_;
};

// Performs asynchronous non-blocking Device-to-Host (D2H) DMA transfer
// from a TPU device buffer to a host memory pointer using the PjRt RawBuffer C
// API. Returns a self-retaining xla::Future<> that preserves device buffer
// lifetime until the DMA transfer completes.
absl::StatusOr<xla::Future<>> TpuAsyncDmaCopyDtoH(
    const DeviceBufferRef& src_buffer, void* dst_host_ptr, int64_t copy_bytes);

// Performs asynchronous non-blocking Host-to-Device (H2D) DMA transfer
// from a host memory pointer into a TPU device buffer at a byte offset using
// the PjRt RawBuffer C API. Returns a self-retaining xla::Future<> that
// preserves device buffer lifetime until the DMA transfer completes.
absl::StatusOr<xla::Future<>> TpuAsyncDmaCopyHtoD(
    const void* src_host_ptr, const DeviceBufferRef& dst_buffer,
    int64_t dst_byte_offset, int64_t copy_bytes);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_PJRT_PJRT_TRANSFER_H_
