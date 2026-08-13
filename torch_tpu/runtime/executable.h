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

#ifndef TORCH_TPU_RUNTIME_EXECUTABLE_H_
#define TORCH_TPU_RUNTIME_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch_tpu/runtime/device_buffer.h"

namespace torch_tpu {

// Options controlling program execution.
struct ExecuteOptions {
  bool non_blocking = true;
  std::optional<int64_t> launch_stream_id = std::nullopt;
};

// Options controlling program compilation.
struct CompileOptions {
  std::string target_platform = "TPU";
  bool enable_fast_math = true;
  std::optional<int64_t> num_replicas = std::nullopt;
  std::optional<int64_t> num_partitions = std::nullopt;
};

// LoadedExecutable represents a compiled program resident on the backend.
class LoadedExecutable {
 public:
  virtual ~LoadedExecutable() = default;

  // Executes the compiled program with the given input buffers.
  virtual absl::StatusOr<std::vector<std::unique_ptr<DeviceBuffer>>> Execute(
      absl::Span<DeviceBuffer* const> arguments,
      const ExecuteOptions& options) = 0;

  // Returns a unique fingerprint identifying the compiled binary artifact.
  virtual absl::StatusOr<std::string> Fingerprint() const = 0;

 protected:
  LoadedExecutable() = default;
};

// Interface for backend graph compilers.
class Compiler {
 public:
  virtual ~Compiler() = default;

  virtual absl::StatusOr<std::unique_ptr<LoadedExecutable>> Compile(
      std::string_view serialized_program, const CompileOptions& options) = 0;

 protected:
  Compiler() = default;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_RUNTIME_EXECUTABLE_H_
