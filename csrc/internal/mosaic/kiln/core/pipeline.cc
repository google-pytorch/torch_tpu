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

#include "csrc/internal/mosaic/kiln/core/pipeline.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#include "absl/functional/function_ref.h"
#include "absl/log/absl_check.h"
#include "csrc/internal/mosaic/kiln/core/control_flow.h"
#include "csrc/internal/mosaic/kiln/core/dma.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/index_arg.h"
#include "csrc/internal/mosaic/kiln/core/memory.h"
#include "csrc/internal/mosaic/kiln/core/ref.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
#include "csrc/internal/mosaic/kiln/core/semaphore.h"
#include "csrc/internal/mosaic/kiln/core/shape.h"
#include "csrc/internal/mosaic/kiln/core/specs.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace torch_tpu::kiln {
namespace {

// Pipelined buffers have two slots: one being computed on, one being copied.
// TODO(b/576608541): Make the number of slots configurable (`pl.Buffered`).
constexpr int64_t kSlots = 2;

// The VMEM slots of a pipelined HBM operand, with one DMA semaphore per slot.
struct Slots {
  Ref vmem;
  SemaphoreArray sems;
};

// Allocates the VMEM slots for blocks of `hbm` (a `{2, block_shape...}`
// buffer), plus one DMA semaphore per slot.
Slots AllocaSlots(const Ref& hbm, const Dimensions& block_shape) {
  ABSL_CHECK(hbm.memory_space() == MemorySpace::kHbm)  // CRASH_OK
      << "pipelined operands must live in HBM";
  ABSL_CHECK_GE(hbm.rank(), 1)  // CRASH_OK
      << "pipelined HBM operands must be at least 1D";
  ABSL_CHECK_EQ(  // CRASH_OK
      hbm.rank(), static_cast<int64_t>(block_shape.size()))
      << "the block shape must have the rank of its HBM operand";
  Dimensions dims = {kSlots};
  dims.append(block_shape.begin(), block_shape.end());
  if (block_shape.size() == 1) {
    // A `{2, B}` buffer would be tiled two rows at a time, so a single slot
    // (one row) can't be sliced out. Put the slots side by side in a flat
    // `{2 * B}` buffer instead, as Pallas `BufferedRef` does.
    ABSL_CHECK_LE(  // CRASH_OK
        kSlots * block_shape[0], std::numeric_limits<int32_t>::max())
        << "1D pipelined buffer exceeds int32_t max";
    dims = {kSlots * block_shape[0]};
  }
  return Slots{.vmem = AllocaVmem(Shape(dims, hbm.dtype())),
               .sems = AllocaDmaSemaphore({kSlots})};
}

// TODO(b/576608541): Also accept block indices (Pallas-style `index_map`)
// and derive offsets and tile-rounded edge sizes from the HBM shape.
// The `region` of `hbm`.
Ref HbmWindow(const Ref& hbm, const BlockRegion& region) {
  ABSL_CHECK(region.offsets.size() == region.sizes.size() &&  // CRASH_OK
             static_cast<int64_t>(region.sizes.size()) == hbm.rank())
      << "a block region needs one offset and one extent per dimension";
  llvm::SmallVector<internal::DimDesc, 4> dims;
  for (size_t d = 0; d < region.sizes.size(); ++d) {
    dims.push_back(
        internal::DimDesc::From(Slice(region.offsets[d], region.sizes[d])));
  }
  return internal::SliceRefImpl(hbm, dims);
}

// The leading `sizes` part of slot `slot` of `vmem` (the whole slot when
// `sizes` is the block shape).
Ref SlotWindow(const Ref& vmem, Scalar slot, const Dimensions& block_shape,
               llvm::ArrayRef<IndexArg> sizes) {
  if (block_shape.size() == 1) {
    const int32_t extent = static_cast<int32_t>(block_shape[0]);
    return vmem(Slice(MultipleOf(slot * extent, extent), sizes[0]));
  }
  llvm::SmallVector<internal::DimDesc, 4> dims = {
      internal::DimDesc::From(IndexArg(slot))};
  for (const IndexArg& size : sizes) {
    dims.push_back(internal::DimDesc::From(Slice(0, size)));
  }
  return internal::SliceRefImpl(vmem, dims);
}

Ref Slot(const Ref& vmem, Scalar slot, const Dimensions& block_shape) {
  llvm::SmallVector<IndexArg, 4> sizes(block_shape.begin(), block_shape.end());
  return SlotWindow(vmem, slot, block_shape, sizes);
}

}  // namespace

void EmitPipeline(IndexArg begin_arg, IndexArg end_arg,
                  llvm::ArrayRef<PipelineInput> inputs,
                  absl::FunctionRef<void(const PipelineStep&)> body) {
  const Scalar begin = begin_arg.ToScalar();
  const Scalar end = end_arg.ToScalar();
  ABSL_CHECK(begin.dtype() == i32 && end.dtype() == i32)  // CRASH_OK
      << "pipeline bounds must be i32 Scalars";

  RunScoped([&] {
    llvm::SmallVector<Slots, 4> input_slots;
    for (const PipelineInput& input : inputs) {
      ABSL_CHECK(input.region != nullptr)  // CRASH_OK
          << "PipelineInput::region must be set";
      input_slots.push_back(AllocaSlots(input.hbm, input.block_shape));
    }
    // TODO(b/576608541): Add outputs, copied back to HBM while later steps
    // run.
    // TODO(b/576608541): Let consecutive steps accumulate into one output
    // block (e.g. the K steps of a matmul).

    // TODO(b/576608541): Skip the copy when a block is the same as the
    // previous step's (e.g. a matmul operand reused across output blocks).
    auto input_copy = [&](size_t i, Scalar step) {
      const BlockRegion region = inputs[i].region(step);
      const Scalar slot = step % kSlots;
      return MakeAsyncCopy(HbmWindow(inputs[i].hbm, region),
                           SlotWindow(input_slots[i].vmem, slot,
                                      inputs[i].block_shape, region.sizes),
                           input_slots[i].sems(slot));
    };
    auto start_inputs = [&](Scalar step) {
      for (size_t i = 0; i < inputs.size(); ++i) input_copy(i, step).Start();
    };

    If(begin < end, [&] {
      start_inputs(begin);
      For(begin, end, [&](Scalar step) {
        If(step + 1 < end, [&] { start_inputs(step + 1); });
        for (size_t i = 0; i < inputs.size(); ++i) input_copy(i, step).Wait();

        PipelineStep pipeline_step{.step = step};
        for (size_t i = 0; i < inputs.size(); ++i) {
          pipeline_step.inputs.push_back(
              Slot(input_slots[i].vmem, step % kSlots, inputs[i].block_shape));
        }
        body(pipeline_step);
      });
    });
  });
}

}  // namespace torch_tpu::kiln
