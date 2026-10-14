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

#include "csrc/eager/materialization_heuristics.h"

#include <cstddef>
#include <cstdint>

#include "absl/types/span.h"
#include "csrc/common/shape.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/ops/op_names.h"

namespace torch_tpu {

// Returns true if the deferred op is a 2D spatial convolution that
// significantly benefits from FastRuntime (O2) compilation over default
// FastCompile (O1).
//
// Hardware & Compiler Performance Regimes:
//
// 1. When FastCompile (O1) wins and why:
//    - Applicable conditions:
//      a. Small batch sizes (batch <= 1).
//      b. Vector-aligned spatial widths (width % 128 == 0, e.g. 256x256).
//      c. Channels-last layouts (NHWC / minor dimension == channels).
//      d. Low compute volume (< 100M MACs).
//    - Architectural rationale:
//      O1 emits compact, straightforward vector loops (~200-500 instruction
//      bundles). The entire program binary (~10-30 KB) fits comfortably into
//      physical on-chip Instruction Memory (IMEM, ~32-64 KB), requiring zero
//      instruction memory overlays, minimal VMEM reservations, and negligible
//      software pipeline prologue/epilogue overhead. When spatial dimensions
//      or channel dimensions natively align to 128-element vector registers,
//      O1 achieves peak hardware vector lane utilization without transposition.
//      Execution completes in sub-millisecond latencies (0.18 ms - 0.44 ms fwd,
//      1.8 ms - 2.0 ms train).
//
// 2. When FastRuntime (O2) wins and why:
//    - Applicable conditions:
//      Heavy unaligned NCHW convolutions (batch > 1, width % 128 != 0 such as
//      width=40, and compute volume >= 100M MACs).
//    - Architectural rationale:
//      In PyTorch's default NCHW layout, the contiguous minor dimension is
//      spatial width (W). When W is unaligned (e.g. W=40), 88 out of 128 vector
//      sublanes (68.75%) must be padded or masked per row. Because O1 cannot
//      vectorize across row boundaries, execution serializes across sublane
//      padding, catastrophically inflating execution time to 37-64 ms (forward)
//      and 74-182 ms (training).
//      O2 rewrites tensor layouts into hardware-native tiles, unrolls spatial
//      loops, and schedules multi-stage software pipelines across systolic
//      Matrix Units (MXUs). This completely eliminates sublane serialization,
//      reducing execution time to ~0.60 ms (forward) and ~2.00 ms (training)
//      (37.7x to 106.6x speedups).
//
// 3. Why O2 incurs an execution latency floor (~0.6 ms fwd / ~2.0 ms train):
//    Aggressive loop unrolling and software pipelining bloat O2 binaries into
//    thousands of instruction bundles (2,287 fwd / 8,234 bwd; 194 KB / 566 KB)
//    and allocate ~33-35 MB of VMEM. Because the code exceeds physical IMEM,
//    the compiler splits it into 6 overlays stored in HBM. During execution,
//    each overlay must be paged into IMEM via DMA with synchronization
//    barriers. For heavy unaligned workloads, the 35x-106x math speedup dwarfs
//    the ~2.0 ms floor. For small/aligned workloads, this fixed floor causes
//    severe latency regressions.
//
// The heuristic below promotes a convolution to O2 if and only if the
// arithmetic savings vastly exceed the ~2.0 ms overlay paging penalty:
bool IsFastRuntimeConvolutionCandidate(const DeferredOp& op) {
  const OpName name = op.op_name();
  if (name != OpName::kConvolution && name != OpName::kConvolutionOut &&
      name != OpName::kConvolutionBackward) {
    return false;
  }

  const bool is_bwd = (name == OpName::kConvolutionBackward);
  const size_t min_inputs = is_bwd ? 3 : 2;
  if (op.inputs().size() < min_inputs) {
    return false;
  }

  const DeviceBufferRef& input_ref = is_bwd ? op.inputs()[1] : op.inputs()[0];
  const DeviceBufferRef& weight_ref = is_bwd ? op.inputs()[2] : op.inputs()[1];

  const absl::Span<const int64_t> input_dims = input_ref.dimensions();
  const absl::Span<const int64_t> weight_dims = weight_ref.dimensions();

  // Guard applies to 4D spatial convolutions (Conv2D: N, C, H, W).
  if (input_dims.size() != 4 || weight_dims.size() != 4) {
    return false;
  }

  const int64_t batch = input_dims[0];
  const int64_t in_c = input_dims[1];
  const int64_t h = input_dims[2];
  const int64_t w = input_dims[3];
  const int64_t out_c = weight_dims[0];

  // RULE 1: Small batch size (B <= 1).
  // - Why O1 is faster: Compute is tiny (< 100K FLOPs, < 10 us on MXUs). O1
  //   executes with 0 overlay page swaps in ~0.18 ms fwd / ~0.53 ms train.
  // - Why O2 is slower: Incurs the fixed ~0.6 ms / ~2.0 ms overlay paging
  //   floor and pipeline setup cost, causing a 2x-4x latency regression.
  if (batch <= 1) {
    return false;
  }

  // RULE 2: channels_last (NHWC layout).
  // - Why O1 is faster: The minor dimension is channels (dim 1). TPU vector
  //   registers are 128 elements wide, so contiguous channels align natively
  //   to vector lanes without transposition. O1 achieves peak vectorization
  //   with zero sublane serialization (~0.36 ms fwd / ~0.90 ms train).
  // - Why O2 is slower: O2 cannot improve vector lane utilization further, but
  //   still incurs the 6-overlay paging cost and 35 MB VMEM allocation,
  //   regressing latency to ~0.55 ms fwd / ~2.17 ms train (~2.4x slowdown).
  if (input_ref.has_custom_layout()) {
    const Shape input_shape = input_ref.shape();
    if (input_shape.layout().has_value()) {
      const auto& m2m = input_shape.layout()->minor_to_major;
      if (!m2m.empty() && m2m.front() == 1) {
        return false;
      }
    }
  }

  // RULE 3: Vector-aligned width (W % 128 == 0, e.g. W=256).
  // - Why O1 is faster / on par: In NCHW format, width is the minor dimension.
  //   When W is a multiple of 128, each row evenly fills 128-element vector
  //   registers without partial sublane padding. O1 matches O2 performance
  //   (e.g., 3.65 ms vs 3.70 ms eval at W=256) while avoiding O2 compilation
  //   latency and overlay thrashing.
  // - Why O2 is not needed: Layout transformation yields no benefit because the
  //   spatial row is already vector-aligned.
  if (w % 128 == 0) {
    return false;
  }

  // RULE 4: Compute volume gate (B * C_in * C_out * H * W >= 100M MACs) with
  // unaligned width (W % 128 != 0).
  // - Why O1 is disastrously slow: In NCHW with unaligned width (e.g. W=40),
  //   88 of the 128 vector sublanes are masked or padded per row. Without
  //   layout transformation, XLA serializes execution across row boundaries,
  //   inflating execution time to 37 ms - 182 ms.
  // - Why O2 is vastly faster (35x-98x speedup): O2 transforms the memory
  //   layout into hardware-optimal tiles, fully saturating vector and matrix
  //   units and dropping execution time to ~0.6 ms eval / ~2.0 ms train.
  //   Workload is large enough that the 35x-98x compute reduction easily
  //   amortizes the ~2.0 ms overlay paging floor.
  constexpr int64_t kMinMacsThreshold = 100'000'000;
  const int64_t approx_macs = batch * in_c * out_c * h * w;
  return approx_macs >= kMinMacsThreshold;
}

}  // namespace torch_tpu
