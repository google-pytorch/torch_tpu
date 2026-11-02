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

// Parts of the precompiled nonzero indices kernels shared by all input ranks:
// the Kiln compaction core and the glue that compiles and registers a kernel.
// Each rank defines how its input is scanned and its buffer contract in its own
// file, e.g. nonzero_indices.h for rank two.

#ifndef TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_NONZERO_INDICES_COMMON_H_
#define TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_NONZERO_INDICES_COMMON_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/shape.h"
#include "csrc/common/unbounded_dynamic_kernel_registry.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"

namespace torch_tpu::internal {

// Identifies a precompiled nonzero indices kernel. The value is used as the
// `id_key` of the kernel's `CompilationCacheKey`, which is persisted in the
// names of the tier-2 cache entries, so values must be explicit, non-negative,
// and must never be reused or renumbered. -1 is reserved for keys that don't
// identify a dynamic kernel.
enum class NonzeroKernelId : int {
  // The nonzero indices kernel for 2D inputs.
  kNonzeroKGreaterThan2 = 1,
};

constexpr int64_t kSublanes = 8;
constexpr int64_t kLanes = 128;
constexpr int64_t kTile = kSublanes * kLanes;  // Elements per compaction step.
constexpr int64_t kWindow = 16384;  // Scratch capacity, not an input limit.
// Shifts for the power-of-two geometry above, used to divide device values.
constexpr int64_t kSublaneShift = 3;  // kSublanes == 1 << kSublaneShift
constexpr int64_t kWindowShift = 14;  // kWindow == 1 << kWindowShift

// Geometry of one scratch window of an input row.
struct Window {
  kiln::Scalar base;   // Logical column origin
  kiln::Scalar n_pad;  // Width padded to 128-lane tiles
  kiln::Scalar tiles;  // 1024-column compaction steps
};

// Returns the window of `width` logical columns at `base`, padded to the
// physical transfer width; the compaction core masks the logical tail.
Window MakeWindow(kiln::Scalar base, kiln::Scalar width);

// Index phase of nonzero for an F32 input viewed as M rows of N columns: write
// the coordinates of every nonzero element, in input row-major order, into the
// caller's native S64[K,rank] allocation. The caller already knows K, so
// nothing is returned to the host. As in torch.nonzero, only +0.0 and -0.0 are
// zero: NaNs, infinities and subnormals are nonzero. K must count exactly these
// elements, since every one found is written.
//
// This is a stream compaction: each nonzero's rank is computed directly and the
// element is routed to it, at a fixed number of vector operations per input
// element.
//
// Work proceeds in tiles of kTile = 1024 elements of one row, held as an
// (8, 128) vector of sublanes by lanes and read in flat row-major order. The
// rank-specific subclass reads the tiles in logical row-major order and hands
// each to `AppendTile`, which then:
//
//   1. Classify. NonzeroFlags marks the elements that are both nonzero and
//      inside the logical input, so padding introduced by the physical tiling
//      is masked here rather than trimmed later.
//
//   2. Rank. Prefix counts the zeros strictly before each element, which is
//      exactly how far left a live element must travel. Counting within a
//      128-lane row is one MXU matmul against an upper-triangular matrix;
//      carrying the row totals down the eight sublanes is three
//      rotate-and-add steps.
//
//   3. Route. CompactPositions packs each live element into one i32 word,
//
//          bit 20 | bits 19..10 | bits 9..0
//          alive  | distance    | position
//
//      then runs a ten-round butterfly: round s moves every word whose
//      distance still has bit s set exactly 2^s slots to the left and clears
//      that bit. A distance is below 1024, so ten rounds drive all of them to
//      zero, leaving each live word in the slot its rank names. A round is one
//      lane rotate, plus a sublane rotate for the lanes that cross a row
//      boundary.
//
//      Take the tile [0, a, 0, b]: a sits at position 1 behind one zero and b
//      at position 3 behind two, so round 0 moves a to slot 0 and round 1
//      moves b to slot 1.
//
//   4. Emit. Append copies the compacted coordinates into one staging plane
//      per output column (row and column for a rank-two input, only column for
//      a rank-one input) at the current output offset and flushes to HBM once
//      1024 coordinates have accumulated. Only this step touches HBM.
//
// Subclasses stream input blocks with kiln::EmitPipeline, which prefetches the
// next block while the current one is compacted, and two output slots let a
// packed tile drain to HBM while the next stage fills.
//
// The output operand is an S32 view of the int64 allocation, shaped as
// 128-word storage rows; see the buffer contract in the rank's header.
// Coordinates are nonnegative and fit in a low word, so ZeroHighWords clears
// the high plane once and nothing writes it again. Every 128 coordinate tuples
// occupy `rank` consecutive storage rows: 128 values of the first coordinate,
// then 128 of the next. Every write lands inside the allocation's ordinary tile
// padding, so the caller need not over-allocate.
//
// Written in Kiln, so a `kiln::Scalar` or `kiln::Tile` is a device value and
// `kiln::For` / `kiln::If` emit device control flow; the C++ runs once at
// build time, and ordinary C++ loops unroll fixed instruction patterns.
class NonzeroCommon {
 public:
  virtual ~NonzeroCommon() = default;

  // Emits the kernel body.
  void Build();

 protected:
  // Allocate scratch and prebuild constant tiles inside `CreateKernel`.
  // `output` is the S32 view of the S64[K,rank] destination, and `m` and `n`
  // are the runtime row count and row length of the input (M = 1 for a
  // rank-one input).
  NonzeroCommon(kiln::Ref output, int64_t rank, kiln::Scalar m, kiln::Scalar n);

  // Scans the input, passing every row's tiles to `AppendTile` in logical
  // row-major order.
  virtual void ProcessInput() = 0;

  // Compacts `values`, the (8, 128) tile of row `row` starting at column
  // `col0`, and appends its nonzero coordinates to the output.
  void AppendTile(kiln::Tile values, kiln::Scalar row, kiln::Scalar col0);

  // Window `window` of a row scanned window by window.
  Window WideWindow(kiln::Scalar window) const;

  kiln::Scalar m() const { return m_; }
  kiln::Scalar n() const { return n_; }

 private:
  // Compacted 1024-column segment: packed positions, live count, and origin.
  struct Stream {
    kiln::Tile positions;
    kiln::Scalar total, row, col0;
  };

  // Zeros strictly before each live element, and the tile's live count.
  struct Ranks {
    kiln::Tile distance;
    kiln::Scalar total;
  };

  kiln::Ref OutputRows(kiln::Scalar q, int64_t rows);
  void ZeroHighWords();
  kiln::Tile NonzeroFlags(kiln::Tile values, kiln::Scalar row,
                          kiln::Scalar col0);
  Ranks Prefix(kiln::Tile z, kiln::Tile live);
  kiln::Tile CompactPositions(kiln::Tile flags, kiln::Tile distance);
  void Append(const Stream& stream);
  void Stage(kiln::Scalar first, kiln::Tile words, const Stream& stream);
  kiln::AsyncCopyDescriptor FlushCopy(kiln::IndexArg slot, kiln::Scalar q);
  void Flush(kiln::Scalar q);
  void FinishOutput();

  kiln::Ref output_;                       // HBM S32 view of native S64 storage
  std::vector<kiln::Ref> planes_;          // VMEM (16, 128) coordinate planes
  kiln::Ref packed_;                       // VMEM (kSlots, 16, 128) flush slots
  kiln::Ref fill_;                         // SMEM total appended coordinates
  kiln::Ref flush_count_;                  // SMEM flushes issued
  kiln::SemaphoreArray flush_sems_;        // Flush DMA completion per slot
  kiln::Tile lane_, sublane_, pos_, tri_;  // Constant vectors
  kiln::Scalar m_, n_;                     // Runtime input extents
};

// Emits the body of a nonzero indices kernel that reads the F32 `input` and
// writes the S32 view `output` of its S64 destination.
using NonzeroKernelBody = void (*)(kiln::Ref input, kiln::Ref output);

// Builds and compiles the Mosaic nonzero indices kernel `body` for an F32
// input of rank `rank` in layout `input_layout`.
absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
CompileNonzeroIndices(int64_t rank, const xla::Layout& input_layout,
                      NonzeroKernelBody body, xla::PjRtClient& client,
                      UniqueCompileOptions compile_options);

// Returns XLA's default layout for S64[K,rank] with K > rank, the on-device
// output layout of the buffer contract of the rank-`rank` kernel.
xla::Layout NonzeroOutputLayout(int64_t rank);

// Returns the UnboundedDynamicExecutionMetadata for the nonzero indices kernel
// for F32 inputs of rank `rank` (1 or 2) in layout `input_layout`, whose
// compatibility predicate is `is_compatible`.
UnboundedDynamicExecutionMetadata MakeNonzeroDynamicMetadata(
    int64_t rank, const xla::Layout& input_layout,
    UnboundedDynamicExecutionMetadata::CompatibilityFunc is_compatible);

// Returns the ShapelessKey a nonzero graph with a rank-`rank` F32 input would
// be lowered to.
ShapelessKey NonzeroCacheKey(int64_t rank);

}  // namespace torch_tpu::internal

#endif  // TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_NONZERO_INDICES_COMMON_H_
