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

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_UNFLATTEN_BATCHED_DOT_GENERAL_H_
#define TORCH_TPU_CSRC_MLIR_PASSES_UNFLATTEN_BATCHED_DOT_GENERAL_H_

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Creates an MLIR pass that unflattens 3D collapsed batch DotGeneralOp
// contractions into 4D DotGeneralOp contractions when the collapsed batch
// dimension factors into a 2D batch prefix [B, H] and at least one operand is
// produced by a 4D transpose that can be absorbed into DotGeneral dimension
// numbers.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createUnflattenBatchedDotGeneralPass();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_MLIR_PASSES_UNFLATTEN_BATCHED_DOT_GENERAL_H_
