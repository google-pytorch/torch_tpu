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

#ifndef TORCH_TPU_EAGER_MATERIALIZE_COMMON_H_
#define TORCH_TPU_EAGER_MATERIALIZE_COMMON_H_

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/MLIRContext.h"
#include "torch_tpu/csrc/common/compilation.h"
#include "torch_tpu/csrc/common/compilation_spec.h"
#include "torch_tpu/csrc/common/context_states.h"
#include "torch_tpu/csrc/eager/device_buffer.h"
#include "torch_tpu/csrc/eager/structured_log_buffer.h"
#include "torch_tpu/csrc/eager/traversal.h"
#include "torch_tpu/pjrt/pjrt_utils.h"
#include "xla/pjrt/pjrt_client.h"

namespace torch_tpu {

// To materialize a tensor's data, we need to execute it on-device.
// This requires identifying the arguments to the executable, the MLIR for the
// executable, both of which may only be available asynchronously due to async
// execution and compilation.
// Once these are available, we can start the execution and write the output
// back to the output tensors.
class ExecutionTask {
 public:
  // Creates an execution task from a traversal.
  // This will:
  //   * Propagate bounded dynamism annotations if needed. This requires a
  //     non-null mlir_context for a bounded dynamic traversal.
  //   * Mark all nodes in the traversal as having been executed
  //   * Begin compiling the traversal into a kernel (or kernels)
  //   * Set all outputs to the "pending materialization" state
  //   * Log the resulting MLIR to support TORCH_TRACE, if enabled.
  static absl::StatusOr<ExecutionTask> FromTraversalWithLogging(
      absl_nonnull std::unique_ptr<Traversal> traversal,
      mlir::MLIRContext& mlir_context, CompilationSpec compilation_spec,
      MaterializationReason reason = MaterializationReason::kUnknown);

  // Creates an execution task from a traversal.
  // This will:
  //   * Propagate bounded dynamism annotations if needed. This requires a
  //     non-null mlir_context for a bounded dynamic traversal.
  //   * Mark all nodes in the traversal as having been executed
  //   * Begin compiling the traversal into a kernel (or kernels)
  //   * Set all outputs to the "pending materialization" state
  static absl::StatusOr<ExecutionTask> FromTraversal(
      absl_nonnull std::unique_ptr<Traversal> traversal,
      mlir::MLIRContext& mlir_context, CompilationSpec compilation_spec,
      std::string* absl_nullable out_mlir_text = nullptr);

  // Creates an execution task from a pre-compiled executable.
  // This will set all outputs to the "pending materialization" state.
  static absl::StatusOr<ExecutionTask> FromExecutable(
      SharedLoadedExecutableWithMetadata executable,
      std::vector<DeviceBufferRef> arguments,
      std::vector<DeviceBufferRef> outputs, std::string_view task_name = "");

  // Runs the execution task. This will:
  //   * Wait for all arguments to be materialized
  //   * Wait for the compiled kernel(s) to be ready
  //   * Execute the compiled kernel(s) to get new PjRtBuffers
  //   * Set all outputs to the "materialized" state with these buffers
  // Any failures will be set as errors on the output buffers.
  absl::Status Run();

  // Sets all output nodes as having a specified error.
  void SetOutputNodesAsError(absl::Status status);

 private:
  explicit ExecutionTask(std::string name,
                         std::vector<DeviceBufferRef> arguments,
                         std::vector<DeviceBufferRef> outputs,
                         CompiledKernel compiled_kernel)
      : name_(std::move(name)),
        arguments_(std::move(arguments)),
        outputs_(std::move(outputs)),
        compiled_kernel_(std::move(compiled_kernel)) {}

  // Runs the execution task and early-returns if there is any failure.
  absl::Status RunInternal();

  // Gets cached executables from the compiled kernel.
  // This will always include a static, fixed-shape kernel, and may also include
  // a preamble and/or postamble for dynamic shape padding and unpadding.
  absl::StatusOr<std::vector<SharedLoadedExecutableWithMetadata>>
  GetCachedExecutables(std::string_view task_name);

  // Gets the inner PjRtBuffers for the argument DeviceBufferRefs.
  absl::StatusOr<std::vector<xla::PjRtBuffer* absl_nullable>>
  GetArgumentBuffers();

  // Finishes a materialization by setting the output nodes as materialized with
  // the given results.
  absl::Status SetOutputNodesAsMaterialized(PjRtBufferPointers results);

  // A human-readable name for the task. Debug use only.
  std::string name_;
  // The arguments to the compiled kernel.
  std::vector<DeviceBufferRef> arguments_;
  // The outputs of the compiled kernel.
  std::vector<DeviceBufferRef> outputs_;
  // The compiled kernel; or, more precisely, the futures for the compiled
  // kernel and any dynamic adapters.
  CompiledKernel compiled_kernel_;
};

CompilationMode GetCompilationMode(EagerMode eager_mode);

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_MATERIALIZE_COMMON_H_
