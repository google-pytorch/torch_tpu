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

#include "torch_tpu/_internal/compile/compile_backend_bridge.h"

#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/runtime/backend.h"
#include "torch_tpu/runtime/device_buffer.h"
#include "torch_tpu/runtime/eager_runtime_adapter.h"
#include "torch_tpu/runtime/executable.h"

namespace torch_tpu {

absl::StatusOr<std::unique_ptr<LoadedExecutable>> CompileBackendBridge::Compile(
    std::string_view mlir_module_str, const CompileOptions& options) {
  Backend* backend = Backend::GetInstance();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Backend pointer check
      backend != nullptr, error::kInternal)
      << "global backend instance is null";

  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=Compiler retrieval check
      Compiler * compiler, backend->GetCompiler());
  return compiler->Compile(mlir_module_str, options);
}

absl::StatusOr<std::vector<std::unique_ptr<DeviceBuffer>>>
CompileBackendBridge::Execute(LoadedExecutable* executable,
                              absl::Span<DeviceBuffer* const> argument_buffers,
                              const ExecuteOptions& options) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Executable pointer check
      executable != nullptr, error::kInvalidArgument)
      << "executable cannot be null";

  return executable->Execute(argument_buffers, options);
}

absl::StatusOr<std::vector<at::Tensor>>
CompileBackendBridge::ExecuteWithTensors(
    LoadedExecutable* executable, absl::Span<const at::Tensor> input_tensors,
    const ExecuteOptions& options) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Executable pointer check
      executable != nullptr, error::kInvalidArgument)
      << "executable cannot be null";

  std::vector<DeviceBuffer*> input_buffers;
  input_buffers.reserve(input_tensors.size());
  for (const auto& tensor : input_tensors) {
    DeviceBuffer* buf = EagerRuntimeAdapter::ExtractBufferFromTensor(tensor);
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Tensor buffer extraction check
        buf != nullptr, error::kInvalidArgument)
        << "input tensor is not backed by a TTR DeviceBuffer";
    input_buffers.push_back(buf);
  }

  auto out_buffers_or = executable->Execute(input_buffers, options);
  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=Execution result check
      std::vector<std::unique_ptr<DeviceBuffer>> out_buffers,
      std::move(out_buffers_or));

  std::vector<at::Tensor> out_tensors;
  out_tensors.reserve(out_buffers.size());
  for (auto& buf : out_buffers) {
    at::Tensor tensor =
        EagerRuntimeAdapter::CreateTensorFromBuffer(std::move(buf));
    out_tensors.push_back(std::move(tensor));
  }

  return out_tensors;
}

}  // namespace torch_tpu
