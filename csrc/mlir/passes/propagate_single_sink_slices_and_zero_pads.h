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

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_PROPAGATE_SINGLE_SINK_SLICES_AND_ZERO_PADS_H_
#define TORCH_TPU_CSRC_MLIR_PASSES_PROPAGATE_SINGLE_SINK_SLICES_AND_ZERO_PADS_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Creates a pass that hoists unit-stride SliceOps across closed single-sink
// forward subgraphs (including LayerNorm, RMSNorm, Softmax, MLP, Attention,
// GatherOp, ConcatenateOp, and causal cumulative-sum ReduceWindowOp) and
// propagates zero-padded slices across homogeneous backward subgraphs.
std::unique_ptr<mlir::Pass> createPropagateSingleSinkSlicesAndZeroPadsPass();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_MLIR_PASSES_PROPAGATE_SINGLE_SINK_SLICES_AND_ZERO_PADS_H_
