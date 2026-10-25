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

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_UNMASKED_SDPA_AND_BACKWARD_SOFTMAX_H_
#define TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_UNMASKED_SDPA_AND_BACKWARD_SOFTMAX_H_

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Creates a pass that folds provably non-zero unmasked SDPA backward sum_exp
// zero-guards, decouples backward SDPA softmax from forward online-softmax
// patterns, and canonicalizes unaligned/cross-attention SDPA softmax and
// backward rowsum subtractions.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createCanonicalizeUnmaskedSdpaAndBackwardSoftmaxPass();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_UNMASKED_SDPA_AND_BACKWARD_SOFTMAX_H_
