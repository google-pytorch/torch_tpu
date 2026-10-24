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

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_FACTOR_HEAD_NORM_AND_COMPLEX_ROPE_SUBLANES_H_
#define TORCH_TPU_CSRC_MLIR_PASSES_FACTOR_HEAD_NORM_AND_COMPLEX_ROPE_SUBLANES_H_

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Creates an MLIR pass that folds stride-2 complex rotary embeddings into
// real contractions and factors the token sublane dimension in head-split
// normalization subgraphs so XLA preserves the 2D projection sublane tiling.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createFactorHeadNormAndComplexRopeSublanesPass();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_MLIR_PASSES_FACTOR_HEAD_NORM_AND_COMPLEX_ROPE_SUBLANES_H_
