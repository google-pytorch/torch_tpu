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

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_FOLD_DOT_TRANSPOSES_FOLD_DOT_TRANSPOSES_H_
#define TORCH_TPU_CSRC_MLIR_PASSES_FOLD_DOT_TRANSPOSES_FOLD_DOT_TRANSPOSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Creates an MLIR pass that folds `TransposeOp` and `ReshapeOp` chains directly
// into `DotOp` and `DotGeneralOp` contractions, and unflattens collapsed 3D
// batched dot generals to 4D to absorb 4D head transposes into dot dimension
// numbers.
std::unique_ptr<mlir::Pass> createFoldDotTransposesPass();

}  // namespace torch_tpu

// clang-format off
#endif  // TORCH_TPU_CSRC_MLIR_PASSES_FOLD_DOT_TRANSPOSES_FOLD_DOT_TRANSPOSES_H_
// clang-format on
