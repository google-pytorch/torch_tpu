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

#include "csrc/pjrt/pjrt_transfer.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/absl_vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/eager/current_stream.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/host_allocator.h"
#include "csrc/pjrt/pjrt_utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {
// Packs unpacked float4_e2m1fn_x2 data (1 element per byte) from unpacked_data
// into packed_data where each byte contains 2 elements (lower 4 bits of first
// element, upper 4 bits of second element).
// Contract:
// - packed_data must not be null.
// - packed_data must point to a buffer containing at least
//   unpacked_data.size() / 2 bytes.
void PackFp4Vector(const std::vector<uint8_t>& unpacked_data,
                   uint8_t* absl_nonnull packed_data) {
  ABSL_CHECK(packed_data != nullptr);          // CRASH_OK
  ABSL_CHECK_EQ(unpacked_data.size() % 2, 0);  // CRASH_OK
  size_t num_packed_bytes = unpacked_data.size() / 2;

  for (int64_t i = 0; i < num_packed_bytes; ++i) {
    uint8_t val0 = unpacked_data[2 * i] & 0x0F;
    uint8_t val1 = unpacked_data[2 * i + 1] & 0x0F;
    packed_data[i] = (val1 << 4) | val0;
  }
}

// Unpacks packed float4_e2m1fn_x2 data from host_data into a byte vector
// where each element occupies one byte (lower 4 bits). Contract:
// - host_data must not be null.
// - num_elements must be non-negative and even.
// - host_data must point to a buffer containing at least num_elements / 2
// bytes.
std::vector<uint8_t> UnpackFp4(const uint8_t* absl_nonnull host_data,
                               int64_t num_elements) {
  ABSL_CHECK(host_data != nullptr);    // CRASH_OK
  ABSL_CHECK_GE(num_elements, 0);      // CRASH_OK
  ABSL_CHECK_EQ(num_elements % 2, 0);  // CRASH_OK

  std::vector<uint8_t> temp_unpacked_data(num_elements);
  for (int64_t i = 0; i < num_elements / 2; ++i) {
    uint8_t byte = host_data[i];
    temp_unpacked_data[2 * i] = byte & 0x0F;
    temp_unpacked_data[2 * i + 1] = (byte >> 4) & 0x0F;
  }
  return temp_unpacked_data;
}

void ValidateFp4Dimensions(absl::Span<const int64_t> dimensions) {
  ABSL_CHECK(!dimensions.empty())  // CRASH_OK
      << "expected float4_e2m1fn_x2 tensors to be at least 1-dimensional, got "
         "0-dimensional";
  ABSL_CHECK_EQ(dimensions.back() % 2, 0)  // CRASH_OK
      << "expected even size in the last dimension for float4_e2m1fn_x2 "
         "tensors, got "
      << dimensions.back();
}

}  // namespace

HostToDeviceTransfer HostToDeviceTransfer::FromZeroes(
    mlir::ElementType element_type, absl::Span<const int64_t> dimensions) {
  const xla::PrimitiveType type = ConvertTo<xla::PrimitiveType>(element_type);
  size_t buffer_size_bytes =
      xla::ShapeUtil::ByteSizeOf(xla::ShapeUtil::MakeShape(type, dimensions));
  std::vector<uint8_t> backing_data(buffer_size_bytes, 0);
  return FromBytesVector(std::move(backing_data), element_type, dimensions);
}

std::vector<uint8_t> HostToDeviceTransfer::MakeUnpackedFp4(
    const void* host_data, absl::Span<const int64_t> dimensions) {
  ValidateFp4Dimensions(dimensions);
  int64_t num_elements = 1;
  for (const int64_t dim : dimensions) {
    num_elements *= dim;
  }
  return UnpackFp4(static_cast<const uint8_t*>(host_data), num_elements);
}

HostToDeviceTransfer HostToDeviceTransfer::FromTensor(
    const void* absl_nonnull tensor_data_ptr,
    const void* absl_nonnull tensor_storage_ptr, mlir::ElementType element_type,
    absl::Span<const int64_t> dimensions) {
  if (element_type == mlir::ElementType::F4E2M1FN) {
    std::vector<uint8_t> backing_data =
        MakeUnpackedFp4(tensor_data_ptr, dimensions);
    return HostToDeviceTransfer::FromBytesVector(std::move(backing_data),
                                                 element_type, dimensions);
  }
  return HostToDeviceTransfer(tensor_data_ptr, tensor_storage_ptr, element_type,
                              dimensions,
                              /*backing_data=*/{});
}

HostToDeviceTransfer HostToDeviceTransfer::FromBytesVector(
    std::vector<uint8_t> backing_data, mlir::ElementType element_type,
    absl::Span<const int64_t> dimensions) {
  return HostToDeviceTransfer(nullptr, nullptr, element_type, dimensions,
                              std::move(backing_data));
}

absl::StatusOr<DeviceBufferRef> HostToDeviceTransfer::Run() {
  return RunAfter({});
}

absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>>
HostToDeviceTransfer::StartTransfer(
    absl_nonnull std::unique_ptr<HostToDeviceTransferImpl> impl) {
  TT_ASSIGN_OR_RETURN(xla::PjRtClient* const client, GetPjRtClient());
  TT_ASSIGN_OR_RETURN(xla::PjRtDevice* const device, GetPjRtDevice());

  // Get the parameters for the PjRtBuffer to be created.
  const xla::PrimitiveType type =
      ConvertTo<xla::PrimitiveType>(impl->element_type);
  xla::PjRtClient::HostBufferSemantics semantics =
      xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes;
  TT_ASSIGN_OR_RETURN(xla::PjRtMemorySpace* const memory_space,
                      GetDefaultMemorySpace(device));
  const xla::Layout* const layout_ptr =
      impl->layout.has_value() ? impl->layout->get() : nullptr;

  // Start the transfer; this produces a PjRtBuffer that will contain the
  // transferred data.
  const void* host_data =
      (impl->host_data == nullptr ? impl->backing_data.data()
                                  : impl->host_data);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=checked in cpu_to_tpu.cc first
      host_data != nullptr, error::kInvalidArgument)
      << "either a nonnull host_data or a non-empty backing_data is required";
  TT_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      AdaptXlaError(
          client->BufferFromHostBuffer(
              host_data, type, impl->dimensions, std::nullopt, semantics,
              [backing_data = std::move(impl->backing_data)]() mutable {
                // This lambda ensures that the backing bytes vector is kept
                // alive until the transfer completes.
                backing_data = std::vector<uint8_t>();
              },
              memory_space, layout_ptr),
          /* context= */ "failed to start copy of buffer from host to device"));

  // If a tensor storage pointer was provided, keep the host data alive until
  // the transfer completes.
  if (impl->host_pinned_ptr != nullptr) {
    bool is_host_pinned =
        HoldPinnedPointerUntil(impl->host_pinned_ptr, buffer->GetReadyFuture());
    if (ABSL_VLOG_IS_ON(1)) {
      if (is_host_pinned) {
        ABSL_VLOG(1) << "[HostToDevice::Run] Host buffer is "
                        "pinned memory; keeping it alive until the "
                        "host-to-device transfer completes.";
      } else {
        ABSL_VLOG(1) << "[HostToDevice::Run] Host buffer is not "
                        "pinned, continuing";
      }
    }
  }
  return buffer;
}

absl::StatusOr<DeviceBufferRef> HostToDeviceTransfer::RunAfter(
    absl::Span<const xla::Future<void>> wait_for) {
  if (device_buffer_ref_.has_value()) {
    // Transfer has already been scheduled with some wait condition.
    ABSL_CHECK(wait_for.empty())  // CRASH_OK=invariant
        << "cannot modify wait condition after transfer has been started";
    return device_buffer_ref_.value();
  }
  ABSL_CHECK(impl_ != nullptr);  // CRASH_OK=invariant

  xla::Future<void> joined_wait_for = absl::OkStatus();
  if (wait_for.size() > 1) {
    joined_wait_for = xla::JoinFutures(wait_for);
  } else if (wait_for.size() == 1) {
    joined_wait_for = wait_for[0];
  }

  if (joined_wait_for.IsReady()) {
    // Nothing to wait for, start the transfer immediately.
    TT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtBuffer> buffer,
                        StartTransfer(std::move(impl_)));
    TT_ASSIGN_OR_RETURN(
        device_buffer_ref_,
        DeviceBufferList::CreateMaterialized(std::move(buffer)));
    return device_buffer_ref_.value();
  } else {
    // Create a pending DeviceBufferRef that will be updated when the transfer
    // completes.
    Shape expected_shape =
        Shape(impl_->dimensions, impl_->element_type);  // copy dimensions
    const auto [device_index, stream_id] = GetCurrentDeviceStreamId();
    TT_ASSIGN_OR_RETURN(device_buffer_ref_,
                        DeviceBufferList::CreatePendingTransfer(
                            expected_shape, device_index, stream_id));

    // Create a callback that will start the transfer and populate the
    // DeviceBufferList when the wait condition is met.
    joined_wait_for.OnReady(
        [device_buffer_list = device_buffer_ref_->device_buffer_list(),
         impl = std::move(impl_)](absl::Status status) mutable {
          if (!status.ok()) {
            ABSL_LOG(WARNING)
                << "futures that a pending transfer was waiting on "
                   "failed; continuing with transfer anyway";
          }
          auto buffer_or = StartTransfer(std::move(impl));
          if (!buffer_or.ok()) {
            device_buffer_list->SetAsError(buffer_or.status());
            return;
          }
          std::vector<std::unique_ptr<xla::PjRtBuffer>> buffers;
          buffers.push_back(std::move(buffer_or.value()));
          absl::Status set_materialized_status =
              device_buffer_list->SetAsMaterialized(std::move(buffers));
          if (!set_materialized_status.ok()) {
            ABSL_LOG(ERROR) << "host-to-device transfer failed to set "
                               "DeviceBufferList as materialized: "
                            << set_materialized_status;
          }
          device_buffer_list.reset();
        });
  }
  return device_buffer_ref_.value();
}

DeviceToHostTransfer::DeviceToHostTransfer(
    DeviceBufferRef buffer_ref, void* absl_nonnull dst_ptr,
    const void* absl_nullable dst_pinned_ptr) {
  // Create a promise and future to track the transfer's completion.
  auto [promise, future] = xla::MakePromise<void>();
  future_ = std::move(future);

  if (buffer_ref.size_bytes() == 0) {
    // Transferring nothing is a no-op and is effectively already complete.
    promise.Set(absl::OkStatus());
    return;
  }

  // If the element type is FP4E2M1FN, we will need to create a temporary
  // buffer to store the unpacked data.
  std::vector<uint8_t> fp4_receiver;
  if (buffer_ref.element_type() == mlir::ElementType::F4E2M1FN) {
    // buffer_ref.size_bytes() reflects the unpacked size of the data;
    // it uses xla::ShapeUtil::ByteSizeOfPrimitiveType which cannot report less
    // than 1 byte per element.
    fp4_receiver.resize(buffer_ref.size_bytes());
  }
  impl_ = std::make_unique<DeviceToHostTransferImpl>(
      std::move(buffer_ref), std::move(fp4_receiver), dst_ptr, dst_pinned_ptr,
      std::move(promise));
}

DeviceToHostTransfer DeviceToHostTransfer::IntoTensor(
    DeviceBufferRef buffer_ref, void* absl_nonnull tensor_data_ptr,
    const void* absl_nonnull storage_data_ptr) {
  return DeviceToHostTransfer(std::move(buffer_ref), tensor_data_ptr,
                              storage_data_ptr);
}

DeviceToHostTransfer DeviceToHostTransfer::IntoBytes(
    DeviceBufferRef buffer_ref, void* absl_nonnull host_bytes_ptr) {
  return DeviceToHostTransfer(std::move(buffer_ref), host_bytes_ptr,
                              /*dst_pinned_ptr=*/nullptr);
}

void DeviceToHostTransfer::OnToLiteralReady(
    absl::Status to_literal_status,
    absl_nonnull std::unique_ptr<DeviceToHostTransferImpl> impl) {
  if (to_literal_status.ok() && !impl->fp4_receiver.empty()) {
    PackFp4Vector(impl->fp4_receiver, static_cast<uint8_t*>(impl->dst_ptr));
  }
  impl->promise.Set(
      AdaptXlaError(to_literal_status,
                    /*context=*/"failed to copy buffer from device to host"));
  // impl is destructed here
}

absl::StatusOr<xla::PjRtBuffer* absl_nonnull>
DeviceToHostTransfer::VerifyMaterialization(
    absl::Status ready_status, const DeviceToHostTransferImpl& impl) {
  if (!ready_status.ok()) {
    return ready_status;
  }
  TT_ASSIGN_OR_RETURN(xla::PjRtBuffer* absl_nonnull buffer,
                      impl.buffer_ref.AwaitBuffer());
  if (buffer->IsDeleted()) {
    return TT_ERROR(error::kUnavailable)
           << "cannot start a device to host copy from a donated or "
              "deleted buffer";
  }
  return buffer;
}

void DeviceToHostTransfer::OnTransferReady(
    absl::Status ready_status,
    absl_nonnull std::unique_ptr<DeviceToHostTransferImpl> impl) {
  // Verify that the materialization succeeded by retrieving the PjRtBuffer.
  // If we can't, forward the error to the promise.
  absl::StatusOr<xla::PjRtBuffer*> buffer_or =
      VerifyMaterialization(ready_status, *impl);
  if (!buffer_or.ok()) {
    impl->promise.Set(buffer_or.status());
    return;  // impl is destructed here
  }

  // Materialization succeeded.
  // If we are copying FP4 data, we will copy to the temporary fp4_receiver
  // buffer; otherwise, we will copy directly to the destination pointer.
  void* effective_dst_ptr =
      (impl->fp4_receiver.empty() ? impl->dst_ptr : impl->fp4_receiver.data());
  xla::Shape xla_shape = xla::ShapeUtil::MakeShapeWithDescendingLayout(
      ConvertTo<xla::PrimitiveType>(impl->buffer_ref.element_type()),
      impl->buffer_ref.dimensions());
  auto literal = std::make_unique<xla::MutableBorrowingLiteral>(
      static_cast<char*>(effective_dst_ptr), xla_shape);

  // Start the transfer and queue the callback for when it completes.
  xla::Future<> to_literal_future = (*buffer_or)->ToLiteral(literal.get());
  to_literal_future.OnReady([literal = std::move(literal),
                             impl = std::move(impl)](absl::Status s) mutable {
    OnToLiteralReady(s, std::move(impl));
  });
  // impl is destructed here
}

xla::Future<> DeviceToHostTransfer::Run() { return RunAfter({}); }

xla::Future<> DeviceToHostTransfer::RunAfter(
    absl::Span<const xla::Future<void>> wait_for) {
  if (impl_ == nullptr) {
    ABSL_CHECK(wait_for.empty())  // CRASH_OK=invariant
        << "cannot modify wait condition after transfer has been started";
    return future_;
  }

  // If the destination pointer is into pinned memory, we will need to keep the
  // underlying buffer alive until the transfer completes.
  if (impl_->dst_pinned_ptr != nullptr) {
    bool dst_held = HoldPinnedPointerUntil(impl_->dst_pinned_ptr, future_);
    if (ABSL_VLOG_IS_ON(1)) {
      if (dst_held) {
        ABSL_VLOG(1)
            << "[DeviceToHostTransfer::Run] Destination pointer is pinned "
               "memory; keeping it alive until the "
               "device-to-host transfer completes.";
      } else {
        ABSL_VLOG(1)
            << "[DeviceToHostTransfer::Run] Destination pointer is not pinned "
               "memory, continuing";
      }
    }
  }

  xla::Future<> materialization_future =
      impl_->buffer_ref.GetMaterializationFuture();
  bool materialization_is_ready = materialization_future.IsReady();
  size_t num_futures = wait_for.size() + (materialization_is_ready ? 0 : 1);

  if (num_futures == 0) {
    // Nothing to wait for, start the transfer immediately.
    OnTransferReady(materialization_future.Await(), std::move(impl_));
  } else {
    // Create a joined future that will be ready when all of the wait_for
    // futures and the materialization future are ready.
    xla::Future<> joined_wait_for;
    if (materialization_future.IsReady()) {
      joined_wait_for = xla::JoinFutures(wait_for);
    } else {
      std::vector<xla::Future<void>> futures;
      futures.reserve(num_futures);
      for (const auto& future : wait_for) {
        futures.push_back(future);
      }
      futures.push_back(std::move(materialization_future));
      joined_wait_for = xla::JoinFutures(futures);
    }
    joined_wait_for.OnReady([impl = std::move(impl_)](absl::Status s) mutable {
      OnTransferReady(s, std::move(impl));
    });
  }
  return future_;
}

// RAII management holder for a PjRt RawBuffer alias and its underlying
// DeviceBufferRef.
//
// During asynchronous TPU DMA transfers (Device-to-Host or Host-to-Device),
// the DMA engine accesses physical device memory asynchronously. This RAII
// struct guarantees that:
// 1. The underlying DeviceBufferRef and its PjRtBuffer remain alive and
//    referenced until the DMA completes.
// 2. When the last reference to this holder is dropped, the PJRT_RawBuffer
//    alias is cleanly destroyed via the PjRt C API extension BEFORE
//    device_buffer_ref is destroyed.
struct PreparedRawDmaBuffer::Holder {
  // Reference-counted handle holding the underlying PyTorch TPU DeviceBuffer
  // alive. Declared first so it is destroyed last after ~Holder() releases the
  // PJRT_RawBuffer alias.
  DeviceBufferRef device_buffer_ref;
  // Global PjRt C API function table pointer.
  const PJRT_Api* c_api{nullptr};
  // PjRt RawBuffer extension function table pointer.
  const PJRT_RawBuffer_Extension* extension{nullptr};
  // Handle to the active raw buffer alias on the TPU device.
  PJRT_RawBuffer* buffer{nullptr};

  Holder(DeviceBufferRef ref, const PJRT_Api* api,
         const PJRT_RawBuffer_Extension* ext, PJRT_RawBuffer* buf)
      : device_buffer_ref(std::move(ref)),
        c_api(api),
        extension(ext),
        buffer(buf) {}

  Holder(const Holder&) = delete;
  Holder& operator=(const Holder&) = delete;

  ~Holder() {
    // Release the PjRt raw buffer alias before device_buffer_ref is destroyed.
    if (buffer != nullptr && c_api != nullptr && extension != nullptr &&
        extension->PJRT_RawBuffer_Destroy != nullptr) {
      PJRT_RawBuffer_Destroy_Args args;
      args.struct_size = PJRT_RawBuffer_Destroy_Args_STRUCT_SIZE;
      args.extension_start = nullptr;
      args.buffer = buffer;
      pjrt::LogFatalIfPjrtError(extension->PJRT_RawBuffer_Destroy(&args),
                                c_api);
    }
  }
};

absl::StatusOr<PreparedRawDmaBuffer> PreparedRawDmaBuffer::Prepare(
    DeviceBufferRef buffer_ref) {
  // Await the PjRtBuffer from the reference and verify computation readiness.
  TT_ASSIGN_OR_RETURN(xla::PjRtBuffer * buffer, buffer_ref.AwaitBuffer());
  TT_RETURN_IF_ERROR(
      AdaptXlaError(buffer->GetReadyFuture().Await(),
                    "TPU buffer computation failed or not ready"));

  // Ensure the buffer is backed by the PjRt C API client.
  auto* capi_buffer = dynamic_cast<xla::PjRtCApiBuffer*>(buffer);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=initialization condition
      capi_buffer != nullptr, error::kInvalidArgument)
      << "Expected PjRtCApiBuffer.";

  const PJRT_Api* c_api = capi_buffer->pjrt_c_api();
  auto* capi_client = dynamic_cast<xla::PjRtCApiClient*>(capi_buffer->client());
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=initialization condition
      capi_client != nullptr, error::kInvalidArgument)
      << "Expected PjRtCApiClient.";

  // Locate the RawBuffer extension required for direct DMA copy.
  const auto* extension = capi_client->FindExtension<PJRT_RawBuffer_Extension>(
      PJRT_Extension_Type::PJRT_Extension_Type_RawBuffer);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=initialization condition
      extension != nullptr, error::kUnavailable)
      << "RawBuffer extension not available.";

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=initialization condition
      extension->PJRT_RawBuffer_CreateRawAliasOfBuffer != nullptr,
      error::kUnavailable)
      << "PJRT_RawBuffer_CreateRawAliasOfBuffer not implemented.";

  // Create a raw alias of the PjRt buffer for direct memory access.
  PJRT_RawBuffer_CreateRawAliasOfBuffer_Args create_args;
  create_args.struct_size =
      PJRT_RawBuffer_CreateRawAliasOfBuffer_Args_STRUCT_SIZE;
  create_args.extension_start = nullptr;
  create_args.buffer = capi_buffer->c_buffer();
  create_args.raw_buffer = nullptr;

  PJRT_Error* create_error =
      extension->PJRT_RawBuffer_CreateRawAliasOfBuffer(&create_args);
  if (create_error != nullptr) {
    std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> error_cleanup(
        create_error, pjrt::MakeErrorDeleter(c_api));
    return AdaptXlaError(pjrt::PjrtErrorToStatus(create_error, c_api),
                         "Failed to create raw alias of buffer");
  }

  return PreparedRawDmaBuffer(std::make_shared<Holder>(
      std::move(buffer_ref), c_api, extension, create_args.raw_buffer));
}

absl::StatusOr<xla::Future<>> PreparedRawDmaBuffer::CopyDtoH(
    void* dst, int64_t offset, int64_t transfer_size) const {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=initialization condition
      holder_ != nullptr &&
          holder_->extension->PJRT_RawBuffer_CopyRawDeviceToHost != nullptr,
      error::kUnavailable)
      << "PJRT_RawBuffer_CopyRawDeviceToHost not implemented.";

  PJRT_RawBuffer_CopyRawDeviceToHost_Args args;
  args.struct_size = PJRT_RawBuffer_CopyRawDeviceToHost_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = holder_->buffer;
  args.dst = dst;
  args.offset = offset;
  args.transfer_size = transfer_size;
  args.event = nullptr;

  PJRT_Error* error =
      holder_->extension->PJRT_RawBuffer_CopyRawDeviceToHost(&args);
  if (error != nullptr) {
    std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> error_cleanup(
        error, pjrt::MakeErrorDeleter(holder_->c_api));
    return AdaptXlaError(pjrt::PjrtErrorToStatus(error, holder_->c_api),
                         "PJRT CopyRawDeviceToHost invocation failed");
  }

  auto [promise, future] = xla::MakePromise<void>();
  pjrt::ConvertCEventToCppFuture(args.event, holder_->c_api)
      .OnReady([holder = holder_,
                promise = std::move(promise)](absl::Status status) mutable {
        holder.reset();
        promise.Set(std::move(status));
      });
  return future;
}

absl::StatusOr<xla::Future<>> PreparedRawDmaBuffer::CopyHtoD(
    const void* src, int64_t offset, int64_t transfer_size) const {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=initialization condition
      holder_ != nullptr &&
          holder_->extension->PJRT_RawBuffer_CopyRawHostToDevice != nullptr,
      error::kUnavailable)
      << "PJRT_RawBuffer_CopyRawHostToDevice not implemented.";

  PJRT_RawBuffer_CopyRawHostToDevice_Args args;
  args.struct_size = PJRT_RawBuffer_CopyRawHostToDevice_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = holder_->buffer;
  args.src = src;
  args.offset = offset;
  args.transfer_size = transfer_size;
  args.event = nullptr;

  PJRT_Error* error =
      holder_->extension->PJRT_RawBuffer_CopyRawHostToDevice(&args);
  if (error != nullptr) {
    std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> error_cleanup(
        error, pjrt::MakeErrorDeleter(holder_->c_api));
    return AdaptXlaError(pjrt::PjrtErrorToStatus(error, holder_->c_api),
                         "PJRT CopyRawHostToDevice invocation failed");
  }

  auto [promise, future] = xla::MakePromise<void>();
  pjrt::ConvertCEventToCppFuture(args.event, holder_->c_api)
      .OnReady([holder = holder_,
                promise = std::move(promise)](absl::Status status) mutable {
        holder.reset();
        promise.Set(std::move(status));
      });
  return future;
}

absl::StatusOr<xla::Future<>> TpuAsyncDmaCopyDtoH(
    const DeviceBufferRef& src_buffer_ref, void* dst_host_ptr,
    int64_t copy_bytes) {
  TT_ASSIGN_OR_RETURN(PreparedRawDmaBuffer raw_buffer,
                      PreparedRawDmaBuffer::Prepare(src_buffer_ref));
  return raw_buffer.CopyDtoH(dst_host_ptr, /*offset=*/0, copy_bytes);
}

absl::StatusOr<xla::Future<>> TpuAsyncDmaCopyHtoD(
    const void* src_host_ptr, const DeviceBufferRef& dst_buffer_ref,
    int64_t dst_byte_offset, int64_t copy_bytes) {
  TT_ASSIGN_OR_RETURN(PreparedRawDmaBuffer raw_buffer,
                      PreparedRawDmaBuffer::Prepare(dst_buffer_ref));
  return raw_buffer.CopyHtoD(src_host_ptr, dst_byte_offset, copy_bytes);
}

}  // namespace torch_tpu
