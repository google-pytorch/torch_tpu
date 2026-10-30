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

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_POINTWISE_CONV_AND_FINITE_SDPA_BIASES_H_
#define TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_POINTWISE_CONV_AND_FINITE_SDPA_BIASES_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Creates an MLIR pass that:
// 1. Lowers unit-batch 1D pointwise ConvolutionOp (window={size=1}, stride=1,
//    pad=0, dilation=1, groups=1) to DotGeneralOp when the spatial token
//    dimension aligns with TPU sublane/lane tile boundaries, commuting
//    channel-first and channel-last 3D TransposeOp views across GLU and
//    cancelling SiLU pairs.
// 2. Folds redundant SDPA row_all_masked CompareOp(mask, -inf, EQ) and
//    clamped_mask MaxOp(mask, -FLT_MAX) when the attention bias subgraph is
//    provably finite (>= -FLT_MAX), folds uniform affine-iota CompareOp, and
//    hoists shared scalar query scaling across dual DotGeneralOp views.
std::unique_ptr<mlir::Pass>
createCanonicalizePointwiseConvAndFiniteSdpaBiasesPass();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_MLIR_PASSES_CANONICALIZE_POINTWISE_CONV_AND_FINITE_SDPA_BIASES_H_
