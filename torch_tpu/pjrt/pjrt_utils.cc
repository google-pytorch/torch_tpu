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

#include "torch_tpu/pjrt/pjrt_utils.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/ops/empty.h"
#include "c10/core/Device.h"
#include "c10/core/TensorImpl.h"
#include "c10/core/impl/DeviceGuardImplInterface.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/context_manager.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/traceme.h"

namespace torch_tpu {

absl::StatusOr<DeviceBufferRef> TpuMallocAndMemcpyHtoD(
    const void* host_data, mlir::ElementType element_type,
    absl::Span<const int64_t> dimensions,
    std::optional<at::Tensor> backing_tensor) {
  tsl::profiler::TraceMe trace("TpuMallocAndMemcpyHtoD");
  if (backing_tensor.has_value() && backing_tensor->data_ptr() != host_data) {
    return TT_ERROR(error::kInvalidArgument)
           << "Backing tensor that was given is not matching the received "
              "host_data "
              "pointer.";
  }

  const xla::PrimitiveType type = ConvertTo<xla::PrimitiveType>(element_type);
  ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL ENTRY] host_data_is_null: "
               << (host_data == nullptr) << ", type: "
               << xla::primitive_util::LowercasePrimitiveTypeName(type)
               << ", dimensions: [" << absl::StrJoin(dimensions, ",") << "]";

  xla::PjRtClient* const client = PjrtBackend::GetInstance().GetClient();
  xla::PjRtDevice* const device = PjrtBackend::GetInstance().GetDevice();

  TT_RET_CHECK(client != nullptr, error::kFailedPrecondition)
      << "PjRt client not initialized in TpuMallocAndMemcpyHtoD.";
  TT_RET_CHECK(device != nullptr, error::kFailedPrecondition)
      << "PjRt device not initialized in TpuMallocAndMemcpyHtoD.";

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
  TT_RET_CHECK(memory_space != nullptr, error::kInternal)
      << "Default memory space is null.";
  ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] Got memory space: "
               << memory_space->DebugString();

  xla::PjRtClient::HostBufferSemantics semantics =
      xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes;
  const void* effective_host_data = host_data;
  std::vector<char> zeroed_host_data_for_alloc;

  if (host_data == nullptr && num_elements > 0) {
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

  auto annotations_layout = GetContextState<LayoutContextState>(nullptr);
  const xla::Layout* layout_ptr =
      annotations_layout ? annotations_layout.get() : nullptr;

  bool keep_host_data_alive = backing_tensor.has_value();

  TT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtBuffer> buffer,
                      client->BufferFromHostBuffer(
                          effective_host_data,

                          type, dimensions, std::nullopt, semantics,
                          [annotations_layout, backing_tensor = std::move(
                                                   backing_tensor)]() mutable {
                            // This lambda isn't really doing anything, but we
                            // need to ensure the C++ compiler doesn't optimize
                            // by dropping the backing tensor before the
                            // transfer completes.
                            if (backing_tensor.has_value()) {
                              backing_tensor.reset();
                            }
                          },
                          memory_space, layout_ptr));

  ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] "
                  "client->BufferFromHostBuffer SUCCEEDED.";

  // Get necessary information from the buffer before moving it.
  auto future = buffer->GetReadyFuture();
  TT_ASSIGN_OR_RETURN(auto buffer_ref,
                      DeviceBufferList::CreateMaterialized(std::move(buffer)));
  if (!keep_host_data_alive) {
    ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] No backing tensor, "
                    "blocking on future.";
    TT_RETURN_IF_ERROR(future.Await());
  } else {
    ABSL_VLOG(1) << "[TpuMallocAndMemcpyHtoD INTERNAL] Backing tensor present, "
                    "creating DeviceBufferRef and marking stream active.";
    const auto* impl =
        c10::impl::getDeviceGuardImpl(GetPrivateUse1DeviceType());
    TT_RET_CHECK(impl != nullptr, error::kInternal)
        << "TPU DeviceGuardImpl not found";
    const c10::Stream current_stream = impl->getStream(impl->getDevice());
    PjrtBackend::GetInstance().MarkStreamActive(impl->getDevice().index(),
                                                current_stream.id(), future);
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

  at::Tensor cpu_tensor_receiver =
      at::empty(buffer_expected_dims,
                at::TensorOptions().dtype(buffer_tensor_type).device(at::kCPU));

  if (buffer_expected_bytes == 0) {
    ABSL_VLOG(1) << "[TpuMemcpyDtoH] DeviceBufferRef size_bytes is 0. "
                    "Returning empty vector.";
    return cpu_tensor_receiver;
  }

  TT_ASSIGN_OR_RETURN(
      auto* buffer, buffer_ref.AwaitBuffer(),
      _ << " - TpuMemcpyDtoH: DeviceBufferRef has nonzero size, "
           "but does not have a PjRtBuffer to copy from.");

  ABSL_VLOG(1) << "[TpuMemcpyDtoH] PjRtBuffer Details - OnDeviceSizeInBytes: "
               << buffer->GetOnDeviceSizeInBytes()
               << ", IsDeleted: " << buffer->IsDeleted()
               << ", IsOnCpu: " << buffer->IsOnCpu()
               << ", Shape: " << buffer->on_device_shape().ToString(true);

  ABSL_VLOG(1) << "[TpuMemcpyDtoH] Calling PjRtBuffer::ToLiteral()...";

  xla::Shape xla_shape = xla::ShapeUtil::MakeShapeWithDescendingLayout(
      ConvertTo<xla::PrimitiveType>(buffer_ref.element_type()),
      buffer_expected_dims);

  auto literal = std::make_unique<xla::MutableBorrowingLiteral>(
      static_cast<char*>(cpu_tensor_receiver.data_ptr()), xla_shape);
  auto future = buffer->ToLiteral(literal.get());
  {
    tsl::profiler::TraceMe trace_await("TpuMemcpyDtoH::Await");
    TT_RETURN_IF_ERROR(future.Await());
  }
  return cpu_tensor_receiver;
}

absl::Status TpuMemcpyDtoHDirect(const DeviceBufferRef& buffer_ref,
                                 void* dst_ptr, bool non_blocking) {
  size_t buffer_expected_bytes = buffer_ref.size_bytes();
  if (buffer_expected_bytes == 0) {
    return absl::OkStatus();
  }

  TT_ASSIGN_OR_RETURN(
      auto* buffer, buffer_ref.AwaitBuffer(),
      _ << " - TpuMemcpyDtoHDirect: DeviceBufferRef has nonzero size, "
           "but does not have a PjRtBuffer to copy from.");

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
    const auto* impl =
        c10::impl::getDeviceGuardImpl(GetPrivateUse1DeviceType());
    TT_RET_CHECK(impl != nullptr, error::kInternal)
        << "TPU DeviceGuardImpl not found";
    const c10::Stream current_stream = impl->getStream(impl->getDevice());
    PjrtBackend::GetInstance().MarkStreamActive(impl->getDevice().index(),
                                                current_stream.id(), future);
    return absl::OkStatus();
  } else {
    return future.Await();
  }
}

absl::StatusOr<PjRtBufferPointers> Execute(
    const SharedLoadedExecutableWithMetadata& executable,
    std::vector<xla::PjRtBuffer* absl_nullable> argument_buffers) {
  tsl::profiler::TraceMe trace("Execute");
  xla::ExecuteOptions execute_options{.strict_shape_checking = true};

  std::vector<std::vector<xla::PjRtBuffer*>> execution_arguments;
  execution_arguments.reserve(1);
  execution_arguments.push_back(std::move(argument_buffers));

  TT_ASSIGN_OR_RETURN(std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
                          results_per_device,
                      executable->GetLoadedExecutable()->Execute(
                          execution_arguments, execute_options));

  TT_RET_CHECK(!results_per_device.empty(), error::kInternal)
      << "XLA execution did not return any results.";

  PjRtBufferPointers result_pointers;
  result_pointers.reserve(results_per_device[0].size());
  for (auto&& [i, result] : llvm::enumerate(results_per_device[0])) {
    result_pointers.push_back(std::move(result));
  }

  return result_pointers;
}

std::string ToString(const xla::PjRtBuffer& buffer) {
  std::ostringstream os;
  os << "PjRtBuffer[shape=" << buffer.on_device_shape().ToString() << "]";
  return os.str();
}

}  // namespace torch_tpu
