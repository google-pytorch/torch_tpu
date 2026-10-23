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

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_BROADCAST_CONTRACTIONS_AND_UNIT_REDUCTIONS_H_
#define TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_BROADCAST_CONTRACTIONS_AND_UNIT_REDUCTIONS_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Creates an MLIR pass that canonicalizes outer-product broadcast-multiply-add
// reductions into stablehlo.dot_general (with group-repeat broadcast
// deduplication and batch-transpose absorption), folds unit-dimension
// reductions and single-key softmaxes, and squeezes interior unit batch
// dimensions in dual-output LayerNorm reductions.
std::unique_ptr<mlir::Pass>
createCanonicalizeBroadcastContractionsAndUnitReductionsPass();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_BROADCAST_CONTRACTIONS_AND_UNIT_REDUCTIONS_H_
