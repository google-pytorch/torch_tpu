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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_LIB_PIPELINE_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_LIB_PIPELINE_H_

#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "csrc/internal/mosaic/kiln/core/index_arg.h"
#include "csrc/internal/mosaic/kiln/core/ref.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
#include "csrc/internal/mosaic/kiln/core/shape.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace torch_tpu::kiln {

// The part of an HBM operand that one pipeline block covers: an offset and an
// extent per dimension. Extents may be smaller than the VMEM block shape (e.g.
// at array edges); the rest of the VMEM block is then left unspecified.
// Offsets and dynamic extents along the two minor dimensions must be provably
// tile-aligned (see `MultipleOf`), as for any DMA.
struct BlockRegion {
  llvm::SmallVector<IndexArg, 4> offsets;
  llvm::SmallVector<IndexArg, 4> sizes;
};

// An HBM operand streamed by `EmitPipeline` through VMEM blocks of
// `block_shape`. Step `s` uses block `block_index(s)` (default: `s`), which
// covers `region(block)` of `hbm`. Inputs are copied in before their step.
// Outputs are copied out after the last of consecutive steps with the same
// block (e.g. the K steps of a matmul output block). Block indices can come in
// any order; returning to a block starts it afresh (its content is not copied
// back in) and copies it out again. Consecutive distinct output blocks must
// cover disjoint parts of `hbm`, as their copies out may overlap.
struct PipelineOperand {
  Ref hbm;
  Dimensions block_shape;
  // Outputs only for now.
  absl::AnyInvocable<Scalar(Scalar step) const> block_index = nullptr;
  // Called with the step for inputs, and with the block index for outputs.
  absl::AnyInvocable<BlockRegion(Scalar block) const> region;
};

// One operand's block at one step of `EmitPipeline`.
struct PipelineBlock {
  // Its VMEM buffer.
  Ref ref;
  // Its block index (i32): `block_index(step)`, or `step` by default.
  Scalar index;
  // Whether this step is the first / last of the block (bool `Scalar`s, i.e.
  // runtime values). Always true for inputs, whose blocks span a single step
  // for now.
  Scalar is_first_step;
  Scalar is_last_step;
};

// What the body of `EmitPipeline` sees at one step.
struct PipelineStep {
  Scalar step;
  // The blocks of the inputs, filled for this step. The body may modify them
  // (e.g. to mask padding).
  llvm::SmallVector<PipelineBlock, 4> inputs;
  // The blocks of the outputs. Their content carries over between the steps of
  // one output block. After the last of them, the leading `region(index).sizes`
  // part of each is copied to HBM, so `body` must write all of it.
  llvm::SmallVector<PipelineBlock, 2> outputs;
};

// Software-pipelines a loop over steps [begin, end) that streams HBM operands
// through double-buffered VMEM blocks: while `body` runs step s, the input
// copies of step s + 1 and the copy of the previous output block may be in
// flight (an output slot is only waited for before its reuse, i.e. on entering
// the block after next). Allocates the VMEM blocks and DMA semaphores, and
// waits for every copy before returning.
void EmitPipeline(IndexArg begin, IndexArg end,
                  llvm::ArrayRef<PipelineOperand> inputs,
                  llvm::ArrayRef<PipelineOperand> outputs,
                  absl::FunctionRef<void(const PipelineStep&)> body);

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_LIB_PIPELINE_H_
