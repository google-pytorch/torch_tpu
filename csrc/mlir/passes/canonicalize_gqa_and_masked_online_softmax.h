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

#ifndef THIRD_PARTY_PY_TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_GQA_AND_MASKED_ONLINE_SOFTMAX_H_
#define THIRD_PARTY_PY_TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_GQA_AND_MASKED_ONLINE_SOFTMAX_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Returns a pass that normalizes forward 5D GQA Q@K^T and probs@V contractions
// to 4D, folds diagonal-unmasked causal/sliding-window row-mask reductions, and
// folds constant-step affine IotaOp packed-sequence scan masks so XLA's
// FlashAttention pass fuses GQA and masked attention into online-softmax.
std::unique_ptr<mlir::Pass> createCanonicalizeGqaAndMaskedOnlineSoftmaxPass();

}  // namespace torch_tpu

#endif  // THIRD_PARTY_PY_TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_GQA_AND_MASKED_ONLINE_SOFTMAX_H_
