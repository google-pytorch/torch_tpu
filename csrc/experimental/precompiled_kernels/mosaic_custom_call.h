/*
 * Copyright 2026 Google LLC
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

#ifndef TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_MOSAIC_CUSTOM_CALL_H_
#define TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_MOSAIC_CUSTOM_CALL_H_

#include <cstdint>
#include <memory>
#include <string_view>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/shape.h"

namespace torch_tpu {

// Optional hook run after `BuildMosaicCustomCall` creates the entry parameters
// (and any host-to-device copies) and before it creates `tpu_custom_call`. It
// may emit HLO instructions (e.g. a layout-preserving `bitcast`) into `builder`
// and replace entries of `operands` in place; `operands[aliased_parameter]`
// must remain unchanged.
using OperandsTransform =
    absl::AnyInvocable<void(xla::HloComputation::Builder& builder,
                            absl::Span<xla::HloInstruction*> operands)>;

// Wraps a serialized Mosaic kernel in one scheduled custom call. The kernel
// must already have been through the Mosaic serde pass (e.g.
// kiln::Kernel::Serialize()); MLIR text and bytecode are both accepted.
// The caller supplies shapes/layouts and metadata routing (-1 for no metadata).
// A parameter whose layout is in host memory (`S(5)`) takes a pinned host
// buffer, which the module copies to the device before the call.
// A nonnegative aliased_parameter requires the single result to alias that
// operand, which must be in device memory; the caller must permit its donation
// at execution. If non-null, `operands_transform` runs on the operands (after
// host-memory copies) before creating `tpu_custom_call` (without replacing
// `operands[aliased_parameter]`). Compile the returned module with HLO passes
// disabled to preserve this contract.
absl::StatusOr<std::unique_ptr<xla::HloModule>> BuildMosaicCustomCall(
    std::string_view serialized_kernel, absl::Span<const xla::Shape> parameters,
    const xla::Shape& result, absl::Span<const int64_t> operand_to_metadata,
    int aliased_parameter = -1, OperandsTransform operands_transform = nullptr);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_MOSAIC_CUSTOM_CALL_H_
