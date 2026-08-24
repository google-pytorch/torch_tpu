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

#ifndef TORCH_TPU_OPS_REDUCTIONS_REDUCTIONS_H_
#define TORCH_TPU_OPS_REDUCTIONS_REDUCTIONS_H_

#include <cstdint>

#include "absl/functional/any_invocable.h"
#include "absl/types/span.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

enum class ReductionMode { kKeepDims, kDropDims };

mlir::MlirOp BuildReductionShlo(
    mlir::MlirOp input_op, absl::Span<const int64_t> dims, mlir::Type mlir_type,
    mlir::MlirOp init_val,
    absl::AnyInvocable<void(mlir::RegionBuilder&) const> reduce_fn,
    ReductionMode reduction_mode = ReductionMode::kDropDims);

// Builds the output unsqueeze for reduction ops that keep dimensions.
//
// The reduction_input is the input to the reduction op with original shape, and
// the reduction_output is the output of the reduction op with missing dims
// that need to be unsqueezed with dims of size 1.
mlir::MlirOp BuildKeepDimsShlo(mlir::MlirOp reduction_input,
                               mlir::MlirOp reduction_output,
                               absl::Span<const int64_t> dims);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_REDUCTIONS_REDUCTIONS_H_
