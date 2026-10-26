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

#ifndef THIRD_PARTY_PY_TORCH_TPU_CSRC_MLIR_PASSES_LOWER_PATCHIFY_CONV_BACKWARD_TO_DOT_H_
#define THIRD_PARTY_PY_TORCH_TPU_CSRC_MLIR_PASSES_LOWER_PATCHIFY_CONV_BACKWARD_TO_DOT_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Creates a pass that lowers non-overlapping 2D/3D patchify backward
// weight-gradient convolutions (along with their paired forward patchify
// convolutions sharing the same input tensor) and padded non-overlapping
// forward patchify convolutions into shared patch-extraction
// reshapes/transposes and 2D stablehlo.dot_general contractions.
std::unique_ptr<mlir::Pass> createLowerPatchifyConvBackwardToDotPass();

}  // namespace torch_tpu

#endif  // THIRD_PARTY_PY_TORCH_TPU_CSRC_MLIR_PASSES_LOWER_PATCHIFY_CONV_BACKWARD_TO_DOT_H_
