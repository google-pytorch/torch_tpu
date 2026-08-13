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

#ifndef TORCH_TPU_INTERNAL_BATCH_TRANSFER_TENSOR_TRANSFER_BRIDGE_H_
#define TORCH_TPU_INTERNAL_BATCH_TRANSFER_TENSOR_TRANSFER_BRIDGE_H_

#include <optional>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/Types.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/runtime/backend.h"

namespace torch_tpu {

// Provides high-throughput tensor data transfers between Host and Device
// using the active TTR backend.
class TensorTransferBridge {
 public:
  // Transfers a host tensor to a device-backed tensor.
  static absl::StatusOr<at::Tensor> TransferHostToDevice(
      const at::Tensor& host_tensor, Device* target_device = nullptr);

  // Transfers a device-backed tensor to a host CPU tensor.
  static absl::StatusOr<at::Tensor> TransferDeviceToHost(
      const at::Tensor& device_tensor);

  // Batch transfers multiple host tensors to device-backed tensors.
  static absl::StatusOr<std::vector<at::Tensor>> BatchTransferHostToDevice(
      absl::Span<const at::Tensor> host_tensors,
      Device* target_device = nullptr);

  // Allocates and copies host data to a device-backed DeviceBufferRef.
  static absl::StatusOr<DeviceBufferRef> MallocAndMemcpyHtoD(
      const void* host_data, mlir::ElementType element_type,
      absl::Span<const int64_t> dimensions,
      std::optional<at::Tensor> backing_tensor = std::nullopt);

  // Directly copies device buffer memory to a host destination pointer.
  static absl::Status MemcpyDtoHDirect(const DeviceBufferRef& src_buf,
                                       void* host_dst, bool non_blocking);

  // Copies device buffer memory to a new CPU tensor.
  static absl::StatusOr<at::Tensor> MemcpyDtoH(const DeviceBufferRef& src_buf,
                                               bool non_blocking);
};

}  // namespace torch_tpu

#endif  // THIRD_PARTY_PY_TORCH_TPU__INTERNAL_BATCH_TRANSFER_TENSOR_TRANSFER_BRIDGE_H_
