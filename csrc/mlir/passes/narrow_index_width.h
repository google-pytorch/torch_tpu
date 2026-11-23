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

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_NARROW_INDEX_WIDTH_H_
#define TORCH_TPU_CSRC_MLIR_PASSES_NARROW_INDEX_WIDTH_H_

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Creates an MLIR pass that eliminates redundant `i32 -> i64` index widenings
// across StableHLO shape/view ops, `iota`, `compare`, `sort`, `gather`,
// `scatter`, `dynamic_slice`, and `dynamic_update_slice` when every indexed
// dimension is statically or bounded-dynamically proven to fit in signed
// 32-bit integers (`<= INT32_MAX`). Function argument and return types are
// preserved so escaping `i64` outputs retain ATen ABI parity.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createNarrowIndexWidthPass();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_MLIR_PASSES_NARROW_INDEX_WIDTH_H_
