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

#include "torch_tpu/runtime/pjrt/pjrt_executable.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/runtime/device_buffer.h"
#include "torch_tpu/runtime/executable.h"
#include "torch_tpu/runtime/pjrt/pjrt_device_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"

namespace torch_tpu {

PjrtLoadedExecutable::PjrtLoadedExecutable(
    std::unique_ptr<xla::PjRtLoadedExecutable> pjrt_executable)
    : pjrt_executable_(std::move(pjrt_executable)) {}

absl::StatusOr<std::vector<std::unique_ptr<DeviceBuffer>>>
PjrtLoadedExecutable::Execute(absl::Span<DeviceBuffer* const> arguments,
                              const ExecuteOptions& options) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal executable check
      pjrt_executable_ != nullptr, error::kInternal)
      << "pjrt loaded executable is null";

  std::vector<xla::PjRtBuffer*> raw_buffers;
  raw_buffers.reserve(arguments.size());
  for (DeviceBuffer* arg : arguments) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Argument null check
        arg != nullptr, error::kInvalidArgument)
        << "null argument passed to Execute";
    auto* pjrt_buf = dynamic_cast<PjrtDeviceBuffer*>(arg);
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Buffer type check
        pjrt_buf != nullptr, error::kInvalidArgument)
        << "non-PJRT buffer passed to PjrtLoadedExecutable";
    raw_buffers.push_back(pjrt_buf->pjrt_buffer());
  }

  std::vector<std::vector<xla::PjRtBuffer*>> args_for_devices = {
      std::move(raw_buffers)};
  xla::ExecuteOptions xla_options;

  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=Execution status
      auto results, pjrt_executable_->Execute(args_for_devices, xla_options));

  std::vector<std::unique_ptr<DeviceBuffer>> output_buffers;
  if (!results.empty()) {
    auto& device_results = results[0];
    output_buffers.reserve(device_results.size());
    for (auto& buffer_ptr : device_results) {
      auto prim_type = buffer_ptr->element_type();
      auto scalar_type_or = ConvertTo<at::ScalarType>(prim_type);
      at::ScalarType dtype =
          scalar_type_or.ok() ? *scalar_type_or : at::ScalarType::Float;
      std::shared_ptr<xla::PjRtBuffer> shared_buf(std::move(buffer_ptr));
      output_buffers.push_back(
          std::make_unique<PjrtDeviceBuffer>(std::move(shared_buf), dtype));
    }
  }

  return output_buffers;
}

absl::StatusOr<std::string> PjrtLoadedExecutable::Fingerprint() const {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal executable check
      pjrt_executable_ != nullptr, error::kInternal)
      << "pjrt loaded executable is null";
  return pjrt_executable_->FingerprintExecutable();
}

PjrtCompiler::PjrtCompiler(xla::PjRtClient* client) : client_(client) {}

absl::StatusOr<std::unique_ptr<LoadedExecutable>> PjrtCompiler::Compile(
    std::string_view serialized_program, const CompileOptions& options) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal client check
      client_ != nullptr, error::kInternal)
      << "pjrt client is uninitialized";
  return TT_ERROR(  // ERROR_COV_INFEASIBLE=Unimplemented compilation pipeline
             error::kPythonNotImplementedError)
         << "direct program compilation through PjrtCompiler requires MLIR "
            "pipeline setup";
}

}  // namespace torch_tpu
