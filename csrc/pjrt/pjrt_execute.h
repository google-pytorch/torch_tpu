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

#ifndef TORCH_TPU_CSRC_PJRT_PJRT_EXECUTE_H_
#define TORCH_TPU_CSRC_PJRT_PJRT_EXECUTE_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/compilation.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

// The result of executing a PjRtExecutable, which may produce multiple non-null
// PjRtBuffers.
using PjRtBufferPointers =
    std::vector<absl_nonnull std::unique_ptr<xla::PjRtBuffer>>;

// Controls whether Execute() validates that each runtime argument buffer's
// on-device layout matches the layout the executable was compiled for.
enum class ArgumentLayoutValidation {
  // Validate runtime argument layouts against compiled parameter layouts.
  kValidate,
  // Skip runtime argument layout validation. Safe when the executable was just
  // compiled or looked up in the compilation cache using the exact layouts of
  // the provided argument buffers.
  kSkip,
};

// Launches the executable. This is a synchronous call, and will block
// until the execution is fully enqueued (but not completed). Calls for D2H
// on the returned buffers will block until the execution and transfer from
// TPU to CPU are complete.
// If `executable->unbounded_dynamic_execution_metadata()` is present, routes
// logical `argument_buffers`, pre-allocated donated dynamic output buffers, and
// disjoint S32[rank] runtime shape metadata buffers to their physical PjRt
// parameter positions.
// When `layout_validation` is `ArgumentLayoutValidation::kValidate`, checks
// that each static argument buffer's on-device layout matches the layout the
// executable was compiled for.
absl::StatusOr<PjRtBufferPointers> Execute(
    const SharedLoadedExecutableWithMetadata& executable,
    std::vector<xla::PjRtBuffer* absl_nullable> argument_buffers,
    absl::Span<const Shape> logical_output_shapes,
    ArgumentLayoutValidation layout_validation =
        ArgumentLayoutValidation::kValidate);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_PJRT_PJRT_EXECUTE_H_
