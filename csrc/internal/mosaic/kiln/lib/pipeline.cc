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

#include "csrc/internal/mosaic/kiln/lib/pipeline.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <variant>

#include "absl/functional/function_ref.h"
#include "absl/log/absl_check.h"
#include "csrc/internal/mosaic/kiln/core/control_flow.h"
#include "csrc/internal/mosaic/kiln/core/dma.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/elementwise.h"
#include "csrc/internal/mosaic/kiln/core/index_arg.h"
#include "csrc/internal/mosaic/kiln/core/memory.h"
#include "csrc/internal/mosaic/kiln/core/ref.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
#include "csrc/internal/mosaic/kiln/core/semaphore.h"
#include "csrc/internal/mosaic/kiln/core/shape.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace torch_tpu::kiln {
namespace {

// Default number of slots for double-buffered operands.
constexpr int64_t kDefaultSlots = 2;

// The VMEM slots of a pipelined HBM operand, with one DMA semaphore per slot.
struct Slots {
  VmemRef vmem;
  SemaphoreArray sems;
};

// Whether `dim` is squeezed out of the VMEM blocks.
bool IsSqueezed(const BlockDim& dim) {
  return std::holds_alternative<Squeezed>(dim);
}

// Allocates the VMEM slots for blocks of `hbm` (a `{num_slots, block_shape...}`
// buffer without the squeezed dimensions), plus one DMA semaphore per slot.
Slots AllocaSlots(const HbmRef& hbm, const BlockShape& block_shape,
                  const int64_t num_slots) {
  ABSL_CHECK_GE(num_slots, 2)  // CRASH_OK
      << "pipeline operands need at least 2 slots";
  ABSL_CHECK_LE(num_slots, std::numeric_limits<int32_t>::max())  // CRASH_OK
      << "pipeline num_slots exceeds int32_t max";
  ABSL_CHECK_GE(hbm.rank(), 1)  // CRASH_OK
      << "pipelined HBM operands must be at least 1D";
  ABSL_CHECK_EQ(  // CRASH_OK
      hbm.rank(), static_cast<int64_t>(block_shape.size()))
      << "the block shape must have the rank of its HBM operand";
  for (const BlockDim& dim :
       llvm::ArrayRef<BlockDim>(block_shape).take_back(2)) {
    ABSL_CHECK(!IsSqueezed(dim))  // CRASH_OK
        << "the two minor dimensions can't be squeezed";
  }
  Dimensions dims = {num_slots};
  for (const BlockDim& dim : block_shape) {
    if (!IsSqueezed(dim)) {
      dims.push_back(std::get<int64_t>(dim));
    }
  }
  if (block_shape.size() == 1) {
    // A `{num_slots, B}` buffer would be tiled multiple rows at a time, so a
    // single slot (one row) can't be sliced out. Put the slots side by side in
    // a flat `{num_slots * B}` buffer instead, as Pallas `BufferedRef` does.
    ABSL_CHECK_GT(dims[1], 0)  // CRASH_OK
        << "1D pipelined block size must be positive";
    ABSL_CHECK_LE(  // CRASH_OK
        dims[1], std::numeric_limits<int32_t>::max() / num_slots)
        << "1D pipelined buffer exceeds int32_t max";
    dims = {num_slots * dims[1]};
  }
  return Slots{.vmem = AllocaVmem(Shape(dims, hbm.dtype())),
               .sems = AllocaDmaSemaphore({num_slots})};
}

// The `region` of `hbm`, with the squeezed dimensions of `block_shape` indexed
// away.
HbmRef HbmWindow(const HbmRef& hbm, const BlockShape& block_shape,
                 const BlockRegion& region) {
  ABSL_CHECK(region.offsets.size() == region.sizes.size() &&  // CRASH_OK
             static_cast<int64_t>(region.sizes.size()) == hbm.rank())
      << "a block region needs one offset and one extent per dimension";
  llvm::SmallVector<Indexer, 4> indexers;
  for (size_t d = 0; d < region.sizes.size(); ++d) {
    if (IsSqueezed(block_shape[d])) {
      indexers.push_back(region.offsets[d]);
    } else {
      indexers.push_back(Slice(region.offsets[d], region.sizes[d]));
    }
  }
  return hbm.At(indexers);
}

// The leading `sizes` part of slot `slot` of `vmem` (the whole slot when
// `sizes` is the block shape).
VmemRef SlotWindow(const VmemRef& vmem, Scalar slot,
                   const BlockShape& block_shape,
                   llvm::ArrayRef<IndexArg> sizes) {
  if (block_shape.size() == 1) {
    const int32_t extent =
        static_cast<int32_t>(std::get<int64_t>(block_shape[0]));
    return vmem(Slice(slot * extent, sizes[0]));
  }
  llvm::SmallVector<Indexer, 4> indexers = {slot};
  for (size_t d = 0; d < sizes.size(); ++d) {
    if (!IsSqueezed(block_shape[d])) {
      indexers.push_back(Slice(0, sizes[d]));
    }
  }
  return vmem.At(indexers);
}

VmemRef Slot(const VmemRef& vmem, Scalar slot, const BlockShape& block_shape) {
  llvm::SmallVector<IndexArg, 4> sizes;
  for (const BlockDim& dim : block_shape) {
    sizes.push_back(IsSqueezed(dim) ? 1 : std::get<int64_t>(dim));
  }
  return SlotWindow(vmem, slot, block_shape, sizes);
}

// Whether a pipelined operand is copied into VMEM (an input) or out to HBM.
enum class Direction { kToVmem, kToHbm };

// Allocates the slots of each of `operands`.
llvm::SmallVector<Slots, 4> AllocaSlots(
    llvm::ArrayRef<PipelineOperand> operands) {
  llvm::SmallVector<Slots, 4> slots;
  for (const PipelineOperand& operand : operands) {
    ABSL_CHECK(operand.region != nullptr)  // CRASH_OK
        << "PipelineOperand::region must be set";
    slots.push_back(
        AllocaSlots(operand.hbm, operand.block_shape, operand.num_slots));
  }
  return slots;
}

// The copy of block `block` of `operand` between HBM and slot `slot`.
AsyncCopyDescriptor Copy(const PipelineOperand& operand, const Slots& slots,
                         Scalar block, Scalar slot, Direction direction) {
  const BlockRegion region = operand.region(block);
  HbmRef hbm = HbmWindow(operand.hbm, operand.block_shape, region);
  VmemRef vmem =
      SlotWindow(slots.vmem, slot, operand.block_shape, region.sizes);
  return direction == Direction::kToVmem
             ? MakeAsyncCopy(hbm, vmem, slots.sems(slot))
             : MakeAsyncCopy(vmem, hbm, slots.sems(slot));
}

// The slot of input block `step` of a pipeline starting at `begin`: cycling
// from slot 0 at `begin`, and non-negative for any `begin`.
Scalar InputSlot(const Scalar step, const Scalar begin,
                 const int64_t num_slots) {
  return (step - begin) % num_slots;
}

// Starts the prologue copies (up to `num_slots - 1` steps ahead) of each of
// `inputs`. Inputs sharing a prefetch offset share one guard `If`.
void StartInitialInputCopies(llvm::ArrayRef<PipelineOperand> inputs,
                             llvm::ArrayRef<Slots> slots, const Scalar begin,
                             const Scalar end) {
  int64_t max_slots = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    Copy(inputs[i], slots[i], begin, I32(0), Direction::kToVmem).Start();
    max_slots = std::max(max_slots, inputs[i].num_slots);
  }
  for (int64_t p = 1; p < max_slots - 1; ++p) {
    // Compare against `end - begin` so `begin + p` cannot wrap i32.
    If(end - begin > p, [&] {  // REF_CAPTURE_OK=called inline
      const Scalar step = begin + p;
      const Scalar slot = I32(static_cast<int32_t>(p));
      for (size_t i = 0; i < inputs.size(); ++i) {
        if (p < inputs[i].num_slots - 1) {
          Copy(inputs[i], slots[i], step, slot, Direction::kToVmem).Start();
        }
      }
    });
  }
}

// Starts the prefetch copies of `inputs` at `step` (each input at
// `step + num_slots - 1`), sharing one `If` across inputs with the same
// `num_slots`.
void StartPrefetchInputCopies(llvm::ArrayRef<PipelineOperand> inputs,
                              llvm::ArrayRef<Slots> slots, const Scalar step,
                              const Scalar begin, const Scalar end) {
  llvm::SmallVector<int64_t, 4> seen_slots;
  for (const PipelineOperand& input : inputs) {
    if (!llvm::is_contained(seen_slots, input.num_slots)) {
      seen_slots.push_back(input.num_slots);
    }
  }
  for (const int64_t num_slots : seen_slots) {
    // Compare against `end - step` so `step + (num_slots - 1)` cannot wrap i32.
    If(end - step > num_slots - 1, [&] {  // REF_CAPTURE_OK=called inline
      const Scalar prefetch_step = step + (num_slots - 1);
      const Scalar prefetch_slot = InputSlot(prefetch_step, begin, num_slots);
      for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].num_slots == num_slots) {
          Copy(inputs[i], slots[i], prefetch_step, prefetch_slot,
               Direction::kToVmem)
              .Start();
        }
      }
    });
  }
}

// Waits for the copies of the blocks of step `step` of each of `inputs`.
void WaitInputCopies(llvm::ArrayRef<PipelineOperand> inputs,
                     llvm::ArrayRef<Slots> slots, const Scalar step,
                     const Scalar begin) {
  for (size_t i = 0; i < inputs.size(); ++i) {
    Copy(inputs[i], slots[i], step, InputSlot(step, begin, inputs[i].num_slots),
         Direction::kToVmem)
        .Wait();
  }
}

// The blocks of step `step` of each of `inputs`, each spanning only that step.
llvm::SmallVector<PipelineBlock, 4> InputBlocks(
    llvm::ArrayRef<PipelineOperand> inputs, llvm::ArrayRef<Slots> slots,
    const Scalar step, const Scalar begin) {
  llvm::SmallVector<PipelineBlock, 4> blocks;
  if (inputs.empty()) return blocks;
  const Scalar true_scalar = I32(1).ToBool();
  for (size_t i = 0; i < inputs.size(); ++i) {
    blocks.push_back(PipelineBlock{
        .ref = Slot(slots[i].vmem, InputSlot(step, begin, inputs[i].num_slots),
                    inputs[i].block_shape),
        .index = step,
        .is_first_step = true_scalar,
        .is_last_step = true_scalar,
    });
  }
  return blocks;
}

// The block of `step` of each of `outputs`: `block_index(step)`, or `step` by
// default.
llvm::SmallVector<Scalar, 2> BlockIndices(
    llvm::ArrayRef<PipelineOperand> outputs, Scalar step) {
  llvm::SmallVector<Scalar, 2> blocks;
  for (const PipelineOperand& output : outputs) {
    if (output.block_index == nullptr) {
      blocks.push_back(step);
      continue;
    }
    const Scalar block = output.block_index(step);
    ABSL_CHECK(block.dtype() == i32)  // CRASH_OK
        << "PipelineOperand::block_index must return an i32 Scalar";
    blocks.push_back(block);
  }
  return blocks;
}

// The slot bookkeeping of one output, carried from step to step. Blocks
// alternate between the two slots, switching whenever the block index changes,
// so any sequence of block indices works.
struct OutputState {
  // The block of the previous step, and its slot.
  Scalar block;
  Scalar slot;
  // The block of the step about to run, computed by the previous step (or
  // before the loop), so that each block index is computed once and the
  // previous step's `is_last_step` and this step's slot change agree.
  Scalar next_block;
  // The block before `block`. Its copy out of slot `1 - slot` may still be in
  // flight. Only meaningful if `has_prev_block` (an i32 0 / 1) is 1.
  Scalar prev_block;
  Scalar has_prev_block;
};

// The loop variables of `states`, which must not be resized while in use.
llvm::SmallVector<LoopVar> LoopVars(llvm::MutableArrayRef<OutputState> states) {
  llvm::SmallVector<LoopVar> vars;
  for (OutputState& s : states) {
    vars.append(
        {&s.block, &s.slot, &s.next_block, &s.prev_block, &s.has_prev_block});
  }
  return vars;
}

// The state before the first step, whose block takes slot 0.
llvm::SmallVector<OutputState, 2> InitialStates(
    llvm::ArrayRef<PipelineOperand> outputs, Scalar begin) {
  llvm::SmallVector<OutputState, 2> states;
  for (const Scalar& block : BlockIndices(outputs, begin)) {
    states.push_back(OutputState{.block = block,
                                 .slot = I32(0),
                                 .next_block = block,
                                 .prev_block = I32(0),
                                 .has_prev_block = I32(0)});
  }
  return states;
}

// Adds the output blocks of `s.step` to `s`. On entering a new block, first
// waits for the copy out of the block before the previous one, whose slot it
// reuses. Advances `states` to this step.
void BeginOutputBlocks(llvm::ArrayRef<PipelineOperand> outputs,
                       llvm::ArrayRef<Slots> slots,
                       llvm::MutableArrayRef<OutputState> states, Scalar begin,
                       Scalar end, PipelineStep& s) {
  const Scalar step = s.step;
  const llvm::SmallVector<Scalar, 2> next_blocks =
      BlockIndices(outputs, Min(step + 1, end - 1));
  for (size_t o = 0; o < outputs.size(); ++o) {
    const OutputState prev = states[o];
    const Scalar block = prev.next_block;
    // False on the first step, whose block is the initial `prev.block`.
    const Scalar changed = block != prev.block;
    const Scalar slot = Select(changed, 1 - prev.slot, prev.slot);
    If(changed & (prev.has_prev_block == 1),
       [&] {  // REF_CAPTURE_OK=called inline
         Copy(outputs[o], slots[o], prev.prev_block, slot, Direction::kToHbm)
             .Wait();
       });
    s.outputs.push_back(PipelineBlock{
        .ref = Slot(slots[o].vmem, slot, outputs[o].block_shape),
        .index = block,
        .is_first_step = (step == begin) | changed,
        .is_last_step = (step + 1 == end) | (next_blocks[o] != block),
    });
    states[o] = OutputState{
        .block = block,
        .slot = slot,
        .next_block = next_blocks[o],
        .prev_block = Select(changed, prev.block, prev.prev_block),
        .has_prev_block = Select(changed, I32(1), prev.has_prev_block),
    };
  }
}

// Starts copying out the output blocks that end at `s.step`.
void EndOutputBlocks(llvm::ArrayRef<PipelineOperand> outputs,
                     llvm::ArrayRef<Slots> slots,
                     llvm::ArrayRef<OutputState> states,
                     const PipelineStep& s) {
  for (size_t o = 0; o < outputs.size(); ++o) {
    If(s.outputs[o].is_last_step, [&] {  // REF_CAPTURE_OK=called inline
      Copy(outputs[o], slots[o], states[o].block, states[o].slot,
           Direction::kToHbm)
          .Start();
    });
  }
}

}  // namespace

// TODO(b/576608541): Follow-ups:
// - Pallas-style block indices (`index_map`), with offsets and tile-rounded
//   edge sizes derived from the HBM shape;
// - configurable number of slots for outputs;
// - skipping the copy of an input block unchanged from the previous step.
void EmitPipeline(IndexArg begin_arg, IndexArg end_arg,
                  llvm::ArrayRef<PipelineOperand> inputs,
                  llvm::ArrayRef<PipelineOperand> outputs,
                  absl::FunctionRef<void(const PipelineStep&)> body) {
  const Scalar begin = begin_arg.ToScalar();
  const Scalar end = end_arg.ToScalar();
  ABSL_CHECK(begin.dtype() == i32 && end.dtype() == i32)  // CRASH_OK
      << "pipeline bounds must be i32 Scalars";
  for (const PipelineOperand& input : inputs) {
    ABSL_CHECK(input.block_index == nullptr)  // CRASH_OK
        << "PipelineOperand::block_index is not supported for inputs yet";
  }
  for (const PipelineOperand& output : outputs) {
    ABSL_CHECK_EQ(output.num_slots, kDefaultSlots)  // CRASH_OK
        << "PipelineOperand::num_slots != 2 is not supported for outputs yet";
  }

  RunScoped([&] {  // REF_CAPTURE_OK=called inline
    const llvm::SmallVector<Slots, 4> input_slots = AllocaSlots(inputs);
    const llvm::SmallVector<Slots, 4> output_slots = AllocaSlots(outputs);
    If(begin < end, [&] {  // REF_CAPTURE_OK=called inline
      llvm::SmallVector<OutputState, 2> states = InitialStates(outputs, begin);
      StartInitialInputCopies(inputs, input_slots, begin, end);
      For(begin, end, 1, LoopVars(states),
          [&](Scalar step) {  // REF_CAPTURE_OK=called inline
            StartPrefetchInputCopies(inputs, input_slots, step, begin, end);
            WaitInputCopies(inputs, input_slots, step, begin);
            PipelineStep pipeline_step{
                .step = step,
                .inputs = InputBlocks(inputs, input_slots, step, begin)};
            BeginOutputBlocks(outputs, output_slots, states, begin, end,
                              pipeline_step);
            body(pipeline_step);
            EndOutputBlocks(outputs, output_slots, states, pipeline_step);
          });

      // `states` now holds the final states. The loop waited for every output
      // copy but those of the last two blocks.
      for (size_t o = 0; o < outputs.size(); ++o) {
        const OutputState& s = states[o];
        If(s.has_prev_block == 1, [&] {  // REF_CAPTURE_OK=called inline
          Copy(outputs[o], output_slots[o], s.prev_block, 1 - s.slot,
               Direction::kToHbm)
              .Wait();
        });
        Copy(outputs[o], output_slots[o], s.block, s.slot, Direction::kToHbm)
            .Wait();
      }
    });
  });
}

}  // namespace torch_tpu::kiln
