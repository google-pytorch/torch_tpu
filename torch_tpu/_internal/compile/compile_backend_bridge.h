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

#ifndef TORCH_TPU_INTERNAL_COMPILE_COMPILE_BACKEND_BRIDGE_H_
#define TORCH_TPU_INTERNAL_COMPILE_COMPILE_BACKEND_BRIDGE_H_

#include <memory>
#include <string_view>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/device_buffer.h"
#include "torch_tpu/runtime/executable.h"

namespace torch_tpu {

// Provides MLIR/StableHLO compilation and executable dispatch using the active
// TorchTPU Runtime (TTR) backend.
class CompileBackendBridge {
 public:
  // Compiles a StableHLO MLIR module into a LoadedExecutable.
  static absl::StatusOr<std::unique_ptr<LoadedExecutable>> Compile(
      std::string_view mlir_module_str,
      const CompileOptions& options = CompileOptions());

  // Executes a LoadedExecutable on DeviceBuffers.
  static absl::StatusOr<std::vector<std::unique_ptr<DeviceBuffer>>> Execute(
      LoadedExecutable* executable,
      absl::Span<DeviceBuffer* const> argument_buffers,
      const ExecuteOptions& options = ExecuteOptions());

  // Executes a LoadedExecutable directly on at::Tensor inputs and returns
  // result at::Tensors.
  static absl::StatusOr<std::vector<at::Tensor>> ExecuteWithTensors(
      LoadedExecutable* executable, absl::Span<const at::Tensor> input_tensors,
      const ExecuteOptions& options = ExecuteOptions());
};

}  // namespace torch_tpu

#endif  // THIRD_PARTY_PY_TORCH_TPU__INTERNAL_COMPILE_COMPILE_BACKEND_BRIDGE_H_
