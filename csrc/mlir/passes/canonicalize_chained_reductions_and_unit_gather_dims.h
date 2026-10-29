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

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_CHAINED_REDUCTIONS_AND_UNIT_GATHER_DIMS_H_
#define TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_CHAINED_REDUCTIONS_AND_UNIT_GATHER_DIMS_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Creates an MLIR pass that:
// 1. Fuses sequential additive/mean ReduceOp chains separated by optional
//    unit-dimension ReshapeOp, scalar/splat constant DivOp/MulOp, and
//    narrowing-widening float ConvertOp roundtrips into a single multi-axis
//    ReduceOp(Add) with a combined divisor/multiplier scale.
// 2. Squeezes redundant size-1 index_vector_dim from 1D-indexed GatherOp
//    start_indices and sets index_vector_dim = squeezed_rank so XLA's
//    GatherOptimizer (which runs before TpuScatterGatherNormalizer) can rewrite
//    eligible gathers, and narrows entry-parameter 64-bit negative-index wraps
//    SelectOp(CompareOp(idx, 0, LT), AddOp(idx, V), idx) feeding GatherOp to
//    i32 when V fits in signed 32-bit integer arithmetic.
std::unique_ptr<mlir::Pass>
createCanonicalizeChainedReductionsAndUnitGatherDimsPass();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_CHAINED_REDUCTIONS_AND_UNIT_GATHER_DIMS_H_
