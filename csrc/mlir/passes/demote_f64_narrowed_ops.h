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

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_DEMOTE_F64_NARROWED_OPS_H_
#define TORCH_TPU_CSRC_MLIR_PASSES_DEMOTE_F64_NARROWED_OPS_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Creates an MLIR pass that demotes redundant f64 elementwise arithmetic chains
// back to f32 when all inputs originate from <=32-bit types or finite f32-range
// constants and all outputs are immediately converted to f32/bf16/f16, and
// rewrites PowOp(x, 0.5) to SqrtOp(x) and PowOp(x, -0.5) to RsqrtOp(x).
std::unique_ptr<mlir::Pass> createDemoteF64NarrowedOpsPass();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_MLIR_PASSES_DEMOTE_F64_NARROWED_OPS_H_
