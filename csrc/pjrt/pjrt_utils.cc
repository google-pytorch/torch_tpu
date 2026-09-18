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

#include "csrc/pjrt/pjrt_utils.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/ops/empty.h"
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/absl_vlog_is_on.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "c10/core/TensorImpl.h"
#include "c10/util/Exception.h"
#include "csrc/common/compilation.h"
#include "csrc/common/context_manager.h"
#include "csrc/common/context_states.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/env_vars.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/events_queue.h"
#include "csrc/pjrt/pjrt_state.h"
#include "llvm/ADT/STLExtras.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

// Unpacks packed float4_e2m1fn_x2 data from host_data into a byte vector where
// each element occupies one byte (lower 4 bits).
// Contract:
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

// Packs unpacked float4_e2m1fn_x2 data (1 element per byte) from unpacked_data
// into packed_data where each byte contains 2 elements (lower 4 bits of first
// element, upper 4 bits of second element).
// Contract:
// - unpacked_data must not be null.
// - packed_data must not be null.
// - num_packed_elements must be non-negative.
// - unpacked_data must point to a buffer containing at least
//   2 * num_packed_elements elements.
// - packed_data must point to a buffer containing at least num_packed_elements
//   bytes.
void PackFp4(const uint8_t* absl_nonnull unpacked_data,
             uint8_t* absl_nonnull packed_data, int64_t num_packed_elements) {
  ABSL_CHECK(unpacked_data != nullptr);   // CRASH_OK
  ABSL_CHECK(packed_data != nullptr);     // CRASH_OK
  ABSL_CHECK_GE(num_packed_elements, 0);  // CRASH_OK

  for (int64_t i = 0; i < num_packed_elements; ++i) {
    uint8_t val0 = unpacked_data[2 * i] & 0x0F;
    uint8_t val1 = unpacked_data[2 * i + 1] & 0x0F;
    packed_data[i] = (val1 << 4) | val0;
  }
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

/**
 * Unpacks float4_e2m1fn_x2 data from the device PjRtBuffer via a temporary
 * 1-byte-per-element buffer, then packs it back into the receiver CPU tensor.
 */
absl::Status TpuMemcpyDtoHFP4(xla::PjRtBuffer* absl_nonnull buffer,
                              absl::Span<const int64_t> buffer_expected_dims,
                              at::Tensor& cpu_tensor_receiver) {
  const at::ScalarType kScalarType =
      ConvertTo<at::ScalarType>(mlir::ElementType::F4E2M1FN);

  ABSL_CHECK(  // CRASH_OK
      buffer->element_type() == xla::PrimitiveType::F4E2M1FN)
      << "TpuMemcpyDtoHFP4() buffer dtype must be " << ToString(kScalarType)
      << ", got " << ToString(buffer->element_type());
  ABSL_CHECK_EQ(  // CRASH_OK
      cpu_tensor_receiver.scalar_type(), kScalarType)
      << "TpuMemcpyDtoHFP4() tensor receiver dtype must be "
      << ToString(kScalarType) << ", got " << ToString(buffer->element_type());

  // XLA's default layout for sub-byte types (like FP4) in host memory
  // literals is unpacked (1 byte per element). We allocate a
  // 1-byte-per-element temporary buffer to receive the unpacked elements from
  // PjRt/XLA.
  const int64_t num_packed_elements = cpu_tensor_receiver.numel();
  std::vector<uint8_t> unpacked_receiver(2 * num_packed_elements);

  xla::Shape xla_shape = xla::ShapeUtil::MakeShapeWithDescendingLayout(
      xla::PrimitiveType::F4E2M1FN, buffer_expected_dims);

  auto literal = std::make_unique<xla::MutableBorrowingLiteral>(
      reinterpret_cast<char*>(unpacked_receiver.data()), xla_shape);
  auto future = buffer->ToLiteral(literal.get());
  {
    tsl::profiler::TraceMe trace_await("TpuMemcpyDtoHFP4::Await");
    TT_RETURN_IF_ERROR(AdaptXlaError(
        future.Await(),
        /* context= */ absl::StrCat("failed to copy ", ToString(kScalarType),
                                    " buffer from device to host")));
  }

  // Pack the data back into cpu_tensor_receiver
  PackFp4(unpacked_receiver.data(),
          static_cast<uint8_t*>(cpu_tensor_receiver.data_ptr()),
          num_packed_elements);
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<DeviceBufferRef> TpuMallocAndMemcpyHtoD(
    const void* host_data, mlir::ElementType element_type,
    absl::Span<const int64_t> dimensions,
    std::optional<at::Tensor> backing_tensor) {
  tsl::profiler::TraceMe trace("TpuMallocAndMemcpyHtoD");
  if (backing_tensor.has_value() && backing_tensor->data_ptr() != host_data) {
    return TT_ERROR(error::kInvalidArgument)
           << "expected backing tensor data pointer to be " << host_data
           << ", got " << backing_tensor->data_ptr();
  }

  if (element_type == mlir::ElementType::F4E2M1FN) {
    ValidateFp4Dimensions(dimensions);
  }

  const xla::PrimitiveType type = ConvertTo<xla::PrimitiveType>(element_type);
  ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL ENTRY] host_data_is_null: "
               << (host_data == nullptr) << ", type: "
               << xla::primitive_util::LowercasePrimitiveTypeName(type)
               << ", dimensions: [" << absl::StrJoin(dimensions, ",") << "]";

  xla::PjRtClient* const client = PjrtBackend::GetInstance().GetClient();
  xla::PjRtDevice* const device = PjrtBackend::GetInstance().GetDevice();

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=XLA PjRt client invariants ensure
                 // PjRtClient is always initialized during runtime.
      client != nullptr, error::kFailedPrecondition)
      << "pjrt client not initialized in TpuMallocAndMemcpyHtoD";
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=XLA PjRt device invariants ensure
                 // PjRtDevice is always initialized during runtime.
      device != nullptr, error::kFailedPrecondition)
      << "pjrt device not initialized in TpuMallocAndMemcpyHtoD";

  int64_t num_elements = 1;
  if (dimensions.empty() && type != xla::TUPLE) {
    // Do nothing.
  } else {
    TT_ASSIGN_OR_RETURN(num_elements, NumElements(dimensions));
  }

  ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] Getting default memory "
                  "space for device: "
               << device->DebugString();
  TT_ASSIGN_OR_RETURN(xla::PjRtMemorySpace* const memory_space,
                      device->default_memory_space());
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=XLA PjRt memory space invariants ensure
                 // default memory space is always non-null.
      memory_space != nullptr, error::kInternal)
      << "default memory space is null";
  ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] Got memory space: "
               << memory_space->DebugString();

  xla::PjRtClient::HostBufferSemantics semantics =
      xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes;
  const void* effective_host_data = host_data;
  std::vector<char> zeroed_host_data_for_alloc;
  std::vector<uint8_t> temp_unpacked_data;

  if (host_data != nullptr && element_type == mlir::ElementType::F4E2M1FN) {
    temp_unpacked_data =
        UnpackFp4(static_cast<const uint8_t*>(host_data), num_elements);
    effective_host_data = temp_unpacked_data.data();
  } else if (host_data == nullptr && num_elements > 0) {
    size_t buffer_size_bytes =
        xla::ShapeUtil::ByteSizeOf(xla::ShapeUtil::MakeShape(type, dimensions));
    if (buffer_size_bytes > 0) {
      ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] host_data is nullptr, "
                      "num_elements > 0. Creating temporary zeroed host buffer "
                      "of size "
                   << buffer_size_bytes << " bytes.";
      zeroed_host_data_for_alloc.resize(buffer_size_bytes, 0);
      effective_host_data = zeroed_host_data_for_alloc.data();

    } else {
      ABSL_VLOG(1)
          << "[TpuMallocAndMemcpyHtoD INTERNAL] host_data is nullptr, "
             "num_elements > 0, but calculated byte size is 0. Passing "
             "nullptr as effective_host_data.";
    }
  }

  ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] Calling "
                  "client->BufferFromHostBuffer with"
               << " effective_host_data_is_null: "
               << (effective_host_data == nullptr) << ", type: "
               << xla::primitive_util::LowercasePrimitiveTypeName(type)
               << ", dimensions: [" << absl::StrJoin(dimensions, ",") << "]"
               << ", semantics: " << static_cast<int>(semantics);

  const xla::Layout* const layout_ptr =
      GetContextState<LayoutContextState>().value_or(nullptr).get();
  bool keep_host_data_alive = backing_tensor.has_value();

  TT_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      AdaptXlaError(
          client->BufferFromHostBuffer(
              effective_host_data, type, dimensions, std::nullopt, semantics,
              [backing_tensor = std::move(backing_tensor),
               temp_unpacked = std::move(temp_unpacked_data)]() mutable {
                // This lambda ensures that the backing tensor and the temporary
                // unpacked FP4 buffer are kept alive until the transfer
                // completes.
                backing_tensor.reset();
                temp_unpacked = std::vector<uint8_t>();
              },
              memory_space, layout_ptr),
          /* context= */ "failed to start copy of buffer from host to device"));

  ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] "
                  "client->BufferFromHostBuffer SUCCEEDED.";

  // Get necessary information from the buffer before moving it.
  auto future = buffer->GetReadyFuture();
  TT_ASSIGN_OR_RETURN(auto buffer_ref,
                      DeviceBufferList::CreateMaterialized(std::move(buffer)));
  if (!keep_host_data_alive) {
    ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] No backing tensor, "
                    "blocking on future.";
    TT_RETURN_IF_ERROR(AdaptXlaError(
        future.Await(),
        /* context= */ "failed to copy buffer from host to device"));
  } else {
    ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] Backing tensor present, "
                    "creating DeviceBufferRef and recording async transfer.";
    RecordAsyncHostToDevice(buffer_ref);
  }

  ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL EXIT] Created "
                  "DeviceBufferRef. Dims: ["
               << absl::StrJoin(buffer_ref.dimensions(), ",") << "]";
  return buffer_ref;
}

absl::StatusOr<at::Tensor> TpuMemcpyDtoH(const DeviceBufferRef& buffer_ref,
                                         bool non_blocking) {
  tsl::profiler::TraceMe trace("TpuMemcpyDtoH");
  ABSL_VLOG(1) << "[TpuMemcpyDtoH ENTRY] buffer_ref: "
               << buffer_ref.DebugString();

  if (non_blocking) {
    TORCH_WARN_ONCE(
        "non_blocking=True in .cpu() or .to('cpu') only works if the "
        "destination is already a pinned tensor. This will block until all "
        "pending d2h copies on the device complete.");
  }

  size_t buffer_expected_bytes = buffer_ref.size_bytes();
  const auto buffer_tensor_type =
      ConvertTo<at::ScalarType>(buffer_ref.element_type());
  absl::Span<const int64_t> buffer_expected_dims = buffer_ref.dimensions();
  Dimensions cpu_dims(buffer_expected_dims.begin(), buffer_expected_dims.end());
  if (buffer_ref.element_type() == mlir::ElementType::F4E2M1FN) {
    ValidateFp4Dimensions(cpu_dims);
    cpu_dims.back() /= 2;
  }

  at::Tensor cpu_tensor_receiver = at::empty(
      cpu_dims, at::TensorOptions().dtype(buffer_tensor_type).device(at::kCPU));

  if (buffer_expected_bytes == 0) {
    ABSL_VLOG(1) << "[TpuMemcpyDtoH] DeviceBufferRef size_bytes is 0. "
                    "Returning empty vector.";
    return cpu_tensor_receiver;
  }

  TT_ASSIGN_OR_RETURN(auto* buffer, buffer_ref.AwaitBuffer(),
                      _ << "device buffer ref has nonzero size, "
                           "but does not have a PjRtBuffer to copy from");

  ABSL_VLOG(1) << "[TpuMemcpyDtoH] PjRtBuffer Details - OnDeviceSizeInBytes: "
               << buffer->GetOnDeviceSizeInBytes()
               << ", IsDeleted: " << buffer->IsDeleted()
               << ", IsOnCpu: " << buffer->IsOnCpu()
               << ", Shape: " << buffer->on_device_shape().ToString(true);

  ABSL_VLOG(1) << "[TpuMemcpyDtoH] Calling PjRtBuffer::ToLiteral()...";

  if (buffer_ref.element_type() == mlir::ElementType::F4E2M1FN) {
    TT_RETURN_IF_ERROR(
        TpuMemcpyDtoHFP4(buffer, buffer_expected_dims, cpu_tensor_receiver));
    return cpu_tensor_receiver;
  }

  xla::Shape xla_shape = xla::ShapeUtil::MakeShapeWithDescendingLayout(
      ConvertTo<xla::PrimitiveType>(buffer_ref.element_type()),
      buffer_expected_dims);

  auto literal = std::make_unique<xla::MutableBorrowingLiteral>(
      static_cast<char*>(cpu_tensor_receiver.data_ptr()), xla_shape);
  auto future = buffer->ToLiteral(literal.get());
  {
    tsl::profiler::TraceMe trace_await("TpuMemcpyDtoH::Await");
    TT_RETURN_IF_ERROR(AdaptXlaError(
        future.Await(),
        /* context= */ "failed to copy buffer from device to host"));
  }
  return cpu_tensor_receiver;
}

absl::Status TpuMemcpyDtoHDirect(const DeviceBufferRef& buffer_ref,
                                 void* dst_ptr, bool non_blocking) {
  ABSL_CHECK_NE(buffer_ref.element_type(),  // CRASH_OK
                mlir::ElementType::F4E2M1FN)
      << "direct copy to host is not supported for float4_e2m1fn_x2";
  size_t buffer_expected_bytes = buffer_ref.size_bytes();
  if (buffer_expected_bytes == 0) {
    return absl::OkStatus();
  }

  TT_ASSIGN_OR_RETURN(auto* buffer, buffer_ref.AwaitBuffer(),
                      _ << "device buffer ref has nonzero size, "
                           "but does not have a PjRtBuffer to copy from");

  absl::Span<const int64_t> buffer_expected_dims = buffer_ref.dimensions();
  xla::Shape xla_shape = xla::ShapeUtil::MakeShapeWithDescendingLayout(
      ConvertTo<xla::PrimitiveType>(buffer_ref.element_type()),
      buffer_expected_dims);

  auto literal = std::make_unique<xla::MutableBorrowingLiteral>(
      static_cast<char*>(dst_ptr), xla_shape);
  xla::Future<> future = buffer->ToLiteral(literal.get());

  if (non_blocking) {
    future.OnReady([literal = std::move(literal)](absl::Status s) {
      if (!s.ok()) {
        ABSL_LOG(ERROR) << "Async D2H ToLiteral transfer failed: " << s;
      }
    });
    RecordAsyncDeviceToHost(std::move(future));
    return absl::OkStatus();
  }
  return AdaptXlaError(
      future.Await(),
      /* context= */ "failed to copy buffer from device into host buffer");
}

bool HloModuleContainsCollective(const xla::HloModule* hlo_module) {
  for (const xla::HloComputation* computation : hlo_module->computations()) {
    for (const xla::HloInstruction* instruction : computation->instructions()) {
      if (xla::IsCollective(instruction)) {
        return true;
      }
    }
  }
  return false;
}

absl::StatusOr<PjRtBufferPointers> Execute(
    const SharedLoadedExecutableWithMetadata& executable,
    std::vector<xla::PjRtBuffer* absl_nullable> argument_buffers) {
  tsl::profiler::TraceMe trace("Execute");
  xla::ExecuteOptions execute_options{.strict_shape_checking = true};

  std::vector<std::vector<xla::PjRtBuffer*>> execution_arguments;
  execution_arguments.reserve(1);
  execution_arguments.push_back(std::move(argument_buffers));

  if (ABSL_VLOG_IS_ON(8)) {
    TT_ASSIGN_OR_RETURN(const auto hlo_modules,
                        executable->GetLoadedExecutable()->GetHloModules());
    ABSL_CHECK_EQ(hlo_modules.size(), 1);  // CRASH_OK
    const auto& hlo_module = hlo_modules[0];

    if (HloModuleContainsCollective(hlo_module.get())) {
      TT_ASSIGN_OR_RETURN(
          const std::string fingerprint,
          executable->GetLoadedExecutable()->FingerprintExecutable());
      const std::optional<int> rank = GetIntegerEnvOnce<int, kRankEnvVar>();
      ABSL_VLOG(8) << "Executable with collectives on rank "
                   << (rank.has_value() ? absl::StrCat(*rank)
                                        : "<unknown_rank>")
                   << " has PjRT executable fingerprint: "
                   << absl::BytesToHexString(fingerprint)
                   << " and HLO module fingerprint: "
                   << hlo_module->GetFingerprint128();
    }
  }

  TT_ASSIGN_OR_RETURN(std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
                          results_per_device,
                      AdaptXlaError(executable->GetLoadedExecutable()->Execute(
                                        execution_arguments, execute_options),
                                    /* context= */ "failed to run executable"));

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=XLA execution contracts guarantee
                 // non-empty results on successful execution.
      !results_per_device.empty(), error::kInternal)
      << "xla execution did not return any results";

  PjRtBufferPointers result_pointers;
  result_pointers.reserve(results_per_device[0].size());
  for (auto&& [i, result] : llvm::enumerate(results_per_device[0])) {
    result_pointers.push_back(std::move(result));
  }

  return result_pointers;
}

namespace {

// RAII management holder for a PjRt RawBuffer alias and its underlying
// DeviceBufferRef.
//
// During asynchronous TPU DMA transfers (Device-to-Host or Host-to-Device), the
// DMA engine accesses physical device memory asynchronously. This RAII struct
// guarantees that:
// 1. The underlying DeviceBufferRef and its PjRtBuffer remain alive and
// referenced until the DMA completes.
// 2. When the last reference to this holder is dropped, the PJRT_RawBuffer
// alias is cleanly destroyed via
//    the PjRt C API extension.
struct RawBufferHolder {
  // Global PjRt C API function table pointer.
  const PJRT_Api* c_api{nullptr};
  // PjRt RawBuffer extension function table pointer.
  const PJRT_RawBuffer_Extension* extension{nullptr};
  // Handle to the active raw buffer alias on the TPU device.
  PJRT_RawBuffer* buffer{nullptr};
  // Reference-counted handle holding the underlying PyTorch TPU DeviceBuffer
  // alive.
  DeviceBufferRef device_buffer_ref;

  RawBufferHolder(const PJRT_Api* api, const PJRT_RawBuffer_Extension* ext,
                  PJRT_RawBuffer* buf, DeviceBufferRef ref)
      : c_api(api),
        extension(ext),
        buffer(buf),
        device_buffer_ref(std::move(ref)) {}

  RawBufferHolder(const RawBufferHolder&) = delete;
  RawBufferHolder& operator=(const RawBufferHolder&) = delete;

  ~RawBufferHolder() {
    // Release the PjRt raw buffer alias upon completion / destruction.
    if (buffer && c_api && extension && extension->PJRT_RawBuffer_Destroy) {
      PJRT_RawBuffer_Destroy_Args args;
      args.struct_size = PJRT_RawBuffer_Destroy_Args_STRUCT_SIZE;
      args.extension_start = nullptr;
      args.buffer = buffer;
      pjrt::LogFatalIfPjrtError(extension->PJRT_RawBuffer_Destroy(&args),
                                c_api);
    }
  }
};

// Context structure bundling all components required to initiate and track a
// raw DMA transfer.
struct RawBufferContext {
  // PjRt C API function table.
  const PJRT_Api* c_api{nullptr};
  // PjRt RawBuffer extension function table for DMA copy calls.
  const PJRT_RawBuffer_Extension* extension{nullptr};
  // Raw buffer handle passed to CopyRawDeviceToHost or CopyRawHostToDevice.
  PJRT_RawBuffer* raw_buffer{nullptr};
  // RAII hold object to be retained alongside the returned xla::Future<>.
  std::shared_ptr<RawBufferHolder> hold;
};

// Prepares a DeviceBufferRef for zero-copy DMA by:
// 1. Awaiting the PjRtBuffer to ensure pending device operations have finished.
// 2. Extracting the PjRt C API client and querying the
// PJRT_RawBuffer_Extension.
// 3. Creating a raw buffer alias of the underlying TPU device buffer.
// 4. Wrapping the alias in a RawBufferHolder to guarantee lifetime safety
// during async DMA.
absl::StatusOr<RawBufferContext> PrepareRawBuffer(
    const DeviceBufferRef& buffer_ref) {
  // Await the PjRtBuffer from the reference and verify computation readiness.
  TT_ASSIGN_OR_RETURN(xla::PjRtBuffer * buffer, buffer_ref.AwaitBuffer());
  TT_RETURN_IF_ERROR(
      AdaptXlaError(buffer->GetReadyFuture().Await(),
                    "TPU buffer computation failed or not ready"));

  // Ensure the buffer is backed by the PjRt C API client.
  auto* capi_buffer = dynamic_cast<xla::PjRtCApiBuffer*>(buffer);
  TT_RET_CHECK(capi_buffer != nullptr, error::kInvalidArgument)
      << "Expected PjRtCApiBuffer.";

  const PJRT_Api* c_api = capi_buffer->pjrt_c_api();
  auto* capi_client = dynamic_cast<xla::PjRtCApiClient*>(capi_buffer->client());
  TT_RET_CHECK(capi_client != nullptr, error::kInvalidArgument)
      << "Expected PjRtCApiClient.";

  // Locate the RawBuffer extension required for direct DMA copy.
  const auto* extension = capi_client->FindExtension<PJRT_RawBuffer_Extension>(
      PJRT_Extension_Type::PJRT_Extension_Type_RawBuffer);
  TT_RET_CHECK(extension != nullptr, error::kUnavailable)
      << "RawBuffer extension not available.";

  TT_RET_CHECK(extension->PJRT_RawBuffer_CreateRawAliasOfBuffer != nullptr,
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
  if (create_error) {
    return AdaptXlaError(pjrt::PjrtErrorToStatus(create_error, c_api),
                         "Failed to create raw alias of buffer");
  }

  PJRT_RawBuffer* c_raw_buffer = create_args.raw_buffer;

  // Package the raw buffer alias into an RAII holder to prevent premature
  // deallocation.
  auto hold = std::make_shared<RawBufferHolder>(c_api, extension, c_raw_buffer,
                                                buffer_ref);
  return RawBufferContext{c_api, extension, c_raw_buffer, std::move(hold)};
}

}  // namespace

absl::StatusOr<AsyncDmaResult> TpuAsyncDmaCopyDtoH(
    const DeviceBufferRef& src_buffer_ref, void* dst_host_ptr,
    int64_t copy_bytes) {
  TT_ASSIGN_OR_RETURN(auto ctx, PrepareRawBuffer(src_buffer_ref));
  TT_RET_CHECK(ctx.extension->PJRT_RawBuffer_CopyRawDeviceToHost != nullptr,
               error::kUnavailable)
      << "PJRT_RawBuffer_CopyRawDeviceToHost not implemented.";

  PJRT_RawBuffer_CopyRawDeviceToHost_Args args;
  args.struct_size = PJRT_RawBuffer_CopyRawDeviceToHost_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = ctx.raw_buffer;
  args.dst = dst_host_ptr;
  args.offset = 0;
  args.transfer_size = copy_bytes;
  args.event = nullptr;

  PJRT_Error* error = ctx.extension->PJRT_RawBuffer_CopyRawDeviceToHost(&args);
  if (error) {
    return AdaptXlaError(pjrt::PjrtErrorToStatus(error, ctx.c_api),
                         "PJRT CopyRawDeviceToHost invocation failed");
  }

  xla::Future<> future = pjrt::ConvertCEventToCppFuture(args.event, ctx.c_api);
  return AsyncDmaResult{std::move(future), std::move(ctx.hold)};
}

absl::StatusOr<AsyncDmaResult> TpuAsyncDmaCopyHtoD(
    const void* src_host_ptr, const DeviceBufferRef& dst_buffer_ref,
    int64_t dst_byte_offset, int64_t copy_bytes) {
  TT_ASSIGN_OR_RETURN(auto ctx, PrepareRawBuffer(dst_buffer_ref));
  TT_RET_CHECK(ctx.extension->PJRT_RawBuffer_CopyRawHostToDevice != nullptr,
               error::kUnavailable)
      << "PJRT_RawBuffer_CopyRawHostToDevice not implemented.";

  PJRT_RawBuffer_CopyRawHostToDevice_Args args;
  args.struct_size = PJRT_RawBuffer_CopyRawHostToDevice_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = ctx.raw_buffer;
  args.src = src_host_ptr;
  args.offset = dst_byte_offset;
  args.transfer_size = copy_bytes;
  args.event = nullptr;

  PJRT_Error* error = ctx.extension->PJRT_RawBuffer_CopyRawHostToDevice(&args);
  if (error) {
    return AdaptXlaError(pjrt::PjrtErrorToStatus(error, ctx.c_api),
                         "PJRT CopyRawHostToDevice invocation failed");
  }

  xla::Future<> future = pjrt::ConvertCEventToCppFuture(args.event, ctx.c_api);
  return AsyncDmaResult{std::move(future), std::move(ctx.hold)};
}

std::string ToString(const xla::PjRtBuffer& buffer) {
  std::ostringstream os;
  os << "PjRtBuffer[shape=" << buffer.on_device_shape().ToString() << "]";
  return os.str();
}

bool TpuDeviceSupportsSparseCore() {
  xla::PjRtDevice* const device = PjrtBackend::GetInstance().GetDevice();
  if (device == nullptr) {
    return false;
  }
  std::string_view kind = device->device_kind();
  if (kind == "TPU v5 lite" || kind == "TPU v5e" || kind == "TPU v4" ||
      kind == "TPU v4 lite" || kind == "TPU v3" || kind == "TPU v2" ||
      kind == "cpu" || kind.empty()) {
    return false;
  }
  return true;
}

}  // namespace torch_tpu
