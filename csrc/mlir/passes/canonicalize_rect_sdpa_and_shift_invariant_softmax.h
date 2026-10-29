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

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_RECT_SDPA_AND_SHIFT_INVARIANT_SOFTMAX_H_
#define TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_RECT_SDPA_AND_SHIFT_INVARIANT_SOFTMAX_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Returns a pass that canonicalizes rectangular lane-aligned-key SDPA softmax
// reductions, eliminates shift-invariant pre-softmax max-subtraction and
// zero-sum backward argmax scatters, and folds standalone diagonal-reflexive
// row_all_masked reductions.
std::unique_ptr<mlir::Pass>
createCanonicalizeRectangularSdpaAndShiftInvariantSoftmaxPass();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_RECT_SDPA_AND_SHIFT_INVARIANT_SOFTMAX_H_
