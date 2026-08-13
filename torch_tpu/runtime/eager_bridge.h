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

#ifndef TORCH_TPU_RUNTIME_EAGER_BRIDGE_H_
#define TORCH_TPU_RUNTIME_EAGER_BRIDGE_H_

#include <memory>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "torch_tpu/runtime/device_buffer.h"

namespace torch_tpu {

// Wraps a polymorphic DeviceBuffer into a PyTorch at::Tensor.
at::Tensor CreateAtTensorFromTtrBuffer(std::unique_ptr<DeviceBuffer> buffer);

// In-place updates an existing at::Tensor's Storage with a new DeviceBuffer.
absl::Status AssignTtrBufferToAtTensor(at::Tensor& tensor,
                                       std::unique_ptr<DeviceBuffer> buffer);

// Unwraps and extracts the underlying DeviceBuffer from a PyTorch at::Tensor
// context.
DeviceBuffer* GetTtrBufferFromAtTensor(const at::Tensor& tensor);

// Safely extracts ownership of DeviceBuffer from an at::Tensor without
// double-free.
std::unique_ptr<DeviceBuffer> ExtractTtrBufferFromAtTensor(at::Tensor& tensor);

}  // namespace torch_tpu

#endif  // TORCH_TPU_RUNTIME_EAGER_BRIDGE_H_
