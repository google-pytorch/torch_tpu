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

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_OPTIMIZE_SDPA_AND_SOFTMAX_REDUCTIONS_H_
#define TORCH_TPU_CSRC_MLIR_PASSES_OPTIMIZE_SDPA_AND_SOFTMAX_REDUCTIONS_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Creates an MLIR pass that eliminates standalone reduce-window output fusions
// in unaligned and pre-normalized attention softmax reductions, hoists 3D
// reciprocals when the 4D attention matrix fits in VMEM or has a transposed
// batch dimension, and normalizes standalone forward 5D GQA and row-masked SDPA
// for XLA FlashAttention online-softmax fusion.
std::unique_ptr<mlir::Pass> createOptimizeSdpaAndSoftmaxReductionsPass();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_MLIR_PASSES_OPTIMIZE_SDPA_AND_SOFTMAX_REDUCTIONS_H_
