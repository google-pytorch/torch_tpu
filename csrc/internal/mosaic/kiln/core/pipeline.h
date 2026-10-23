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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_PIPELINE_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_PIPELINE_H_

#include <functional>

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

// An HBM operand read by `EmitPipeline`: step `s` copies `region(s)` of `hbm`
// into a VMEM block of `block_shape`.
struct PipelineInput {
  Ref hbm;
  Dimensions block_shape;
  std::function<BlockRegion(Scalar step)> region;  // STD_FUNCTION_OK
};

// What the body of `EmitPipeline` sees at one step.
struct PipelineStep {
  Scalar step;
  // The VMEM blocks of the inputs, filled for this step. The body may modify
  // them (e.g. to mask padding).
  llvm::SmallVector<Ref, 4> inputs;
};

// Software-pipelines a loop over steps [begin, end) (with 0 <= begin) that
// streams HBM inputs through double-buffered VMEM blocks: the input copies
// of step s + 1 are in flight while `body` runs step s. Allocates the VMEM
// blocks and DMA semaphores, and waits for every copy before returning.
void EmitPipeline(IndexArg begin, IndexArg end,
                  llvm::ArrayRef<PipelineInput> inputs,
                  absl::FunctionRef<void(const PipelineStep&)> body);

}  // namespace torch_tpu::kiln

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_PIPELINE_H_
