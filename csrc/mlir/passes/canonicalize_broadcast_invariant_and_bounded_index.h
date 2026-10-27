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

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_BROADCAST_INVARIANT_AND_BOUNDED_INDEX_H_
#define TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_BROADCAST_INVARIANT_AND_BOUNDED_INDEX_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Creates an MLIR pass that:
// 1. Demotes broadcast-invariant batch dimensions in DotGeneralOp, normalizes
//    SliceOp offsets along broadcast-invariant axes, coalesces contiguous
//    SliceOp chains feeding ConcatenateOp, and hoists unit leading slices
//    through elementwise/view/batched-dot chains to fold with BroadcastInDimOp.
// 2. Narrows closed bounded i64 IotaOp-rooted index and comparison subgraphs
//    to i32 when static interval bounds fit in signed 32-bit integers.
// 3. Collapses singleton-expanded [C, 1, ..., 1] elementwise normalization
//    chains between 1D [C] reshapes and strips unit axes from BroadcastInDimOp.
std::unique_ptr<mlir::Pass>
createCanonicalizeBroadcastInvariantAndBoundedIndexPass();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_BROADCAST_INVARIANT_AND_BOUNDED_INDEX_H_
