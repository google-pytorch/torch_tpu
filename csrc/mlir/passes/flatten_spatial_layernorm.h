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

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_FLATTEN_SPATIAL_LAYERNORM_H_
#define TORCH_TPU_CSRC_MLIR_PASSES_FLATTEN_SPATIAL_LAYERNORM_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Creates a pass that hoists a 2D spatial-flattening ReshapeOp
// ([d_0, ..., d_{R-2}, C] -> [N, C]) upstream across a channel-last LayerNorm
// subgraph and folds an upstream channel-to-last transpose into a 2D
// Reshape([1, C, ...] -> [C, N]) + Transpose([C, N] -> [N, C]).
std::unique_ptr<mlir::Pass> createFlattenSpatialLayerNormPass();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_MLIR_PASSES_FLATTEN_SPATIAL_LAYERNORM_H_
