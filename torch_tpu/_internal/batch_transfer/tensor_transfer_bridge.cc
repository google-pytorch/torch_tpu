// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/_internal/batch_transfer/tensor_transfer_bridge.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "c10/core/DeviceType.h"
#include "c10/core/TensorOptions.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/pjrt/pjrt_utils.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/device_buffer.h"
#include "torch_tpu/runtime/eager_runtime_adapter.h"
#include "xla/tsl/concurrency/future.h"

namespace torch_tpu {

absl::StatusOr<at::Tensor> TensorTransferBridge::TransferHostToDevice(
    const at::Tensor& host_tensor, Device* target_device) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Tensor defined check
      host_tensor.defined(), error::kInvalidArgument)
      << "cannot transfer undefined tensor";

  std::vector<int64_t> shape;  // INT_VEC_OK
  for (int64_t s : host_tensor.sizes()) {
    shape.push_back(s);
  }

  const void* data_ptr = host_tensor.data_ptr();
  auto buffer_or = EagerRuntimeAdapter::CreateBufferFromHost(
      data_ptr, host_tensor.scalar_type(), shape, target_device);
  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=Buffer creation check
      std::unique_ptr<DeviceBuffer> buffer, std::move(buffer_or));

  return EagerRuntimeAdapter::CreateTensorFromBuffer(std::move(buffer));
}

absl::StatusOr<at::Tensor> TensorTransferBridge::TransferDeviceToHost(
    const at::Tensor& device_tensor) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Tensor defined check
      device_tensor.defined(), error::kInvalidArgument)
      << "cannot transfer undefined tensor";

  DeviceBuffer* buffer =
      EagerRuntimeAdapter::ExtractBufferFromTensor(device_tensor);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Buffer extraction check
      buffer != nullptr, error::kInvalidArgument)
      << "tensor is not backed by a TTR DeviceBuffer";

  std::vector<int64_t> shape(buffer->shape().begin(),  // INT_VEC_OK
                             buffer->shape().end());
  at::Tensor host_tensor = at::empty(
      shape, at::TensorOptions().dtype(buffer->dtype()).device(c10::kCPU));

  if (buffer->GetOnDeviceSizeInBytes() > 0 &&
      host_tensor.data_ptr() != nullptr) {
    auto future = buffer->CopyToHostAsync(host_tensor.data_ptr(),
                                          buffer->GetOnDeviceSizeInBytes());
    TT_RETURN_IF_ERROR(  // ERROR_COV_INFEASIBLE=Buffer ToHost check
        future.Await());
  }

  return host_tensor;
}

absl::StatusOr<std::vector<at::Tensor>>
TensorTransferBridge::BatchTransferHostToDevice(
    absl::Span<const at::Tensor> host_tensors, Device* target_device) {
  std::vector<at::Tensor> result;
  result.reserve(host_tensors.size());
  for (const auto& t : host_tensors) {
    auto device_t_or = TransferHostToDevice(t, target_device);
    TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=Batch transfer check
        at::Tensor device_t, std::move(device_t_or));
    result.push_back(std::move(device_t));
  }
  return result;
}

absl::StatusOr<DeviceBufferRef> TensorTransferBridge::MallocAndMemcpyHtoD(
    const void* host_data, mlir::ElementType element_type,
    absl::Span<const int64_t> dimensions,
    std::optional<at::Tensor> backing_tensor) {
  return TpuMallocAndMemcpyHtoD(host_data, element_type, dimensions,
                                std::move(backing_tensor));
}

absl::Status TensorTransferBridge::MemcpyDtoHDirect(
    const DeviceBufferRef& src_buf, void* host_dst, bool non_blocking) {
  return TpuMemcpyDtoHDirect(src_buf, host_dst, non_blocking);
}

absl::StatusOr<at::Tensor> TensorTransferBridge::MemcpyDtoH(
    const DeviceBufferRef& src_buf, bool non_blocking) {
  return TpuMemcpyDtoH(src_buf, non_blocking);
}

}  // namespace torch_tpu
