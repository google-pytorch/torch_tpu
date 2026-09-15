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

#ifndef TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_COMPILER_H_
#define TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_COMPILER_H_

#include <array>
#include <memory>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/shape.h"

namespace torch_tpu {

// Verify and convert Mosaic to versioned bytecode for a TPU custom call.
// Mutates the module into transport form; the caller owns its context.
absl::StatusOr<std::string> SerializeMosaic(mlir::ModuleOp module);

// Wrap serialized Mosaic in a dynamic custom call. Parameters, in order:
// input, private output, S32[input rank] extents, S32[output rank] extents.
// The caller supplies layouts. Payload extents become unbounded; the result
// must alias the private output. The Mosaic body sees both payloads and result.
// The body is opaque here, so this stays independent of how it was produced.
absl::StatusOr<std::unique_ptr<xla::HloModule>> BuildDynamicKernelHlo(
    std::string_view body, std::array<xla::Shape, 4> parameters);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_COMPILER_H_
