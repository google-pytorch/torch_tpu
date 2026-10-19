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

#ifndef THIRD_PARTY_PY_TORCH_TPU_CSRC_MLIR_PASSES_SINK_3D_HEAD_SPLIT_AND_FLATTEN_LAYERNORM_H_
#define THIRD_PARTY_PY_TORCH_TPU_CSRC_MLIR_PASSES_SINK_3D_HEAD_SPLIT_AND_FLATTEN_LAYERNORM_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Returns a pass that reorders post dot elementwise operations with attention
// head split transposes and flattens unit axis LayerNorm subgraphs to 2D.
std::unique_ptr<mlir::Pass> createSink3dHeadSplitAndFlattenLayerNormPass();

}  // namespace torch_tpu

#endif  // THIRD_PARTY_PY_TORCH_TPU_CSRC_MLIR_PASSES_SINK_3D_HEAD_SPLIT_AND_FLATTEN_LAYERNORM_H_
