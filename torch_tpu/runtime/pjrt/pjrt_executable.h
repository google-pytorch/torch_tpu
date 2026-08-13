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

#ifndef TORCH_TPU_RUNTIME_PJRT_PJRT_EXECUTABLE_H_
#define TORCH_TPU_RUNTIME_PJRT_PJRT_EXECUTABLE_H_

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch_tpu/runtime/device_buffer.h"
#include "torch_tpu/runtime/executable.h"
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {

class PjrtLoadedExecutable final : public LoadedExecutable {
 public:
  explicit PjrtLoadedExecutable(
      std::unique_ptr<xla::PjRtLoadedExecutable> pjrt_executable);
  ~PjrtLoadedExecutable() override = default;

  absl::StatusOr<std::vector<std::unique_ptr<DeviceBuffer>>> Execute(
      absl::Span<DeviceBuffer* const> arguments,
      const ExecuteOptions& options) override;

  absl::StatusOr<std::string> Fingerprint() const override;

  [[nodiscard]] xla::PjRtLoadedExecutable* pjrt_executable() const {
    return pjrt_executable_.get();
  }

 private:
  std::unique_ptr<xla::PjRtLoadedExecutable> pjrt_executable_;
};

class PjrtCompiler final : public Compiler {
 public:
  explicit PjrtCompiler(xla::PjRtClient* client);
  ~PjrtCompiler() override = default;

  absl::StatusOr<std::unique_ptr<LoadedExecutable>> Compile(
      std::string_view serialized_program,
      const CompileOptions& options) override;

 private:
  xla::PjRtClient* client_;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_RUNTIME_PJRT_PJRT_EXECUTABLE_H_
