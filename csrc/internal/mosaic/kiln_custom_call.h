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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CUSTOM_CALL_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CUSTOM_CALL_H_

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {

// Wraps a Kiln kernel in a StableHLO `tpu_custom_call`.
//
// `inputs` must follow the kernel's operand order: scalar-prefetch operands
// first (in `ScalarPrefetchSpec` order), then the `InSpec` operands. The
// results correspond to the kernel's `OutSpec`s in order.
//
// `vmem_limit_bytes` optionally raises Mosaic's scoped VMEM limit for the
// kernel; when omitted the platform default applies.
//
// `needs_hlo_passes` must be set for a kernel whose body calls
// `SpliceStablehlo`: it makes Mosaic legalize the StableHLO before compiling.
absl::StatusOr<mlir::stablehlo::CustomCallOp> CreateKilnCustomCallOp(
    mlir::OpBuilder& builder, mlir::Location loc, kiln::Kernel kernel,
    mlir::ValueRange inputs, mlir::TypeRange output_types,
    std::optional<int64_t> vmem_limit_bytes = std::nullopt,
    bool needs_hlo_passes = false);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CUSTOM_CALL_H_
