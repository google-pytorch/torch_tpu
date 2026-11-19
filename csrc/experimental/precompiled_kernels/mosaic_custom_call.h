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

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/shape.h"

namespace torch_tpu {

// Wraps a serialized Mosaic kernel in one scheduled custom call. The kernel
// must already have been through the Mosaic serde pass (e.g.
// kiln::Kernel::Serialize()); MLIR text and bytecode are both accepted.
// The caller supplies shapes/layouts and metadata routing (-1 for no metadata).
// A nonnegative aliased_parameter requires the single result to alias that
// operand; the caller must permit its donation at execution. Compile the
// returned module with HLO passes disabled to preserve this contract.
absl::StatusOr<std::unique_ptr<xla::HloModule>> BuildMosaicCustomCall(
    std::string_view serialized_kernel, absl::Span<const xla::Shape> parameters,
    const xla::Shape& result, absl::Span<const int64_t> operand_to_metadata,
    int aliased_parameter = -1);

// As `BuildMosaicCustomCall`, for a tuple `result`: element i must alias
// operand `aliased_parameters[i]`, unless that is negative.
absl::StatusOr<std::unique_ptr<xla::HloModule>> BuildMosaicTupleCustomCall(
    std::string_view serialized_kernel, absl::Span<const xla::Shape> parameters,
    const xla::Shape& result, absl::Span<const int64_t> operand_to_metadata,
    absl::Span<const int64_t> aliased_parameters);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_MOSAIC_CUSTOM_CALL_H_
