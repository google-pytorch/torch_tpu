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

#include "csrc/experimental/precompiled_kernels/nonzero_indices.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/layout_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/unbounded_dynamic_kernel_registry.h"
#include "csrc/experimental/precompiled_kernels/kernel_utils.h"
#include "csrc/experimental/precompiled_kernels/mosaic_custom_call.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

// Identifies a precompiled nonzero indices kernel. The value is used as the
// `id_key` of the kernel's `CompilationCacheKey`, which is persisted in the
// names of the tier-2 cache entries, so values must be explicit, non-negative,
// and must never be reused or renumbered. -1 is reserved for keys that don't
// identify a dynamic kernel.
enum class NonzeroKernelId : int {
  // The nonzero indices kernel for 2D inputs.
  kNonzeroKGreaterThan2 = 1,
  // The nonzero indices kernel for 1D inputs tiled T(1024).
  kNonzeroInputTile1024 = 2,
};

using kiln::Scalar;
using kiln::Tile;

constexpr int64_t kSublanes = 8;
constexpr int64_t kLanes = 128;
constexpr int64_t kTile = kSublanes * kLanes;  // Elements per compaction step.
constexpr int64_t kDistanceShift = 10;  // Distance field of a packed word.
constexpr int64_t kAlive = 1 << 20;     // Alive bit of a packed word.
constexpr int64_t kWindow = 16384;      // Scratch capacity, not an input limit.
constexpr int64_t kZeroRows = 16;  // Storage rows per high-word zeroing DMA.
constexpr int64_t kSlots = 2;      // Double-buffered flush slots.
// Shifts for the power-of-two geometry above, used to divide device values.
constexpr int64_t kSublaneShift = 3;  // kSublanes == 1 << kSublaneShift
constexpr int64_t kLaneShift = 7;     // kLanes == 1 << kLaneShift
constexpr int64_t kTileShift = 10;    // kTile == 1 << kTileShift
constexpr int64_t kWindowShift = 14;  // kWindow == 1 << kWindowShift

// Index phase of nonzero for a rank-two F32 input: write the (row, column)
// coordinates of every nonzero element, in input row-major order, into the
// caller's native S64[K,2] allocation. The caller already knows K, so nothing
// is returned to the host. As in torch.nonzero, only +0.0 and -0.0 are zero:
// NaNs, infinities and subnormals are nonzero. K must count exactly these
// elements, since every one found is written.
//
// A rank-one F32 input is handled as the single row of a rank-two input, and
// only its column coordinates are written, into an S64[K,1] allocation.
//
// This is a stream compaction: each nonzero's rank is computed directly and the
// element is routed to it, at a fixed number of vector operations per input
// element.
//
// Work proceeds in tiles of kTile = 1024 elements, held as an (8, 128) vector
// of sublanes by lanes and read in flat row-major order. Per tile:
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
//   4. Emit. Append copies the compacted row and column coordinates into
//      staging planes at the current output offset and flushes to HBM once
//      1024 coordinates have accumulated (only columns for a rank-one input).
//      Only this step touches HBM.
//
// Input arrives by DMA eight physical rows at a time, but compaction always
// visits logical rows in order, so the output never needs reordering. A row
// too wide for the kWindow scratch buffer, and the single row of a rank-one
// input, is processed one window at a time. In all cases kiln::EmitPipeline
// prefetches the next input block while the current one is compacted, and two
// output slots let a packed tile drain to HBM while the next stage fills.
//
// The output operand is an S32 view of the int64 allocation, shaped as
// 128-word storage rows; see the buffer contract in the header. Coordinates are
// nonnegative and fit in a low word, so ZeroHighWords clears the high plane
// once and nothing writes it again. Every 128 pairs occupy two consecutive
// storage rows: 128 row indices, then their 128 column indices. For a rank-one
// input, every 128 indices occupy one storage row. Every write lands inside
// the allocation's ordinary tile padding, so the caller need not over-allocate.
//
// Written in Kiln, so a `kiln::Scalar` or `kiln::Tile` is a device value and
// `kiln::For` / `kiln::If` emit device control flow; the C++ runs once at
// build time, and ordinary C++ loops unroll fixed instruction patterns.

// Geometry of one scratch window (a whole row block unless n_ > kWindow or the
// input is rank one).
struct Window {
  Scalar base;   // Logical column origin
  Scalar n_pad;  // Width padded to 128-lane tiles
  Scalar tiles;  // 1024-column compaction steps
};

// Pad physical transfer width; NonzeroFlags masks the logical tail.
Window MakeWindow(Scalar base, Scalar width) {
  // TODO(b/562066667): Drop MultipleOf once the OSS jaxlib proves `& -kLanes`.
  const Scalar n_pad =
      kiln::MultipleOf((width + (kLanes - 1)) & -kLanes, kLanes);
  return {base, n_pad, (n_pad + (kTile - 1)) >> kTileShift};
}

// Compacted 1024-column segment: packed positions, live count, and origin.
struct Stream {
  Tile positions;
  Scalar total, row, col0;
};

// Zeros strictly before each live element, and the tile's live count.
struct Ranks {
  Tile distance;
  Scalar total;
};

// T[i,j] = (i <= j), so z*T gives inclusive zero counts along each 128-lane
// row. Inputs are 0/1 BF16 and accumulation is F32, so counts are exact.
Tile Triangle() {
  const Tile i = kiln::Iota(kiln::i32(kLanes, kLanes), 0);
  const Tile j = kiln::Iota(kiln::i32(kLanes, kLanes), 1);
  const Tile one = kiln::Full(kiln::bf16(kLanes, kLanes), 1.0);
  return Select(i <= j, one, 0.0);
}

class NonzeroKernel {
 public:
  // Allocate scratch and prebuild constant tiles inside `CreateKernel`.
  NonzeroKernel(kiln::HbmRef x, kiln::HbmRef output)
      : x_(x),
        output_(output),
        stage_rows_(kiln::AllocaVmem(kiln::i32(2 * kSublanes, kLanes))),
        stage_cols_(kiln::AllocaVmem(kiln::i32(2 * kSublanes, kLanes))),
        packed_(kiln::AllocaVmem(kiln::i32(kSlots, 2 * kSublanes, kLanes))),
        fill_(kiln::AllocaSmem(kiln::i32())),
        flush_count_(kiln::AllocaSmem(kiln::i32())),
        flush_sems_(kiln::AllocaDmaSemaphore({kSlots})),
        lane_(kiln::Iota(kiln::i32(kSublanes, kLanes), 1)),
        sublane_(kiln::Iota(kiln::i32(kSublanes, kLanes), 0)),
        pos_(sublane_ * kLanes + lane_),
        tri_(Triangle()),
        // A rank-one input is a single row.
        m_(x.rank() == 2 ? x.Dim(0) : kiln::I32(1)),
        n_(x.Dim(x.rank() - 1)) {}

  void Build() {
    kiln::Store(fill_, kiln::I32(0));
    kiln::Store(flush_count_, kiln::I32(0));
    kiln::Store(stage_rows_, kiln::Zeros(kiln::i32(2 * kSublanes, kLanes)));
    kiln::Store(stage_cols_, kiln::Zeros(kiln::i32(2 * kSublanes, kLanes)));
    ZeroHighWords();
    ProcessInput();
    FinishOutput();
  }

 private:
  // View `rows` 128-word storage rows of the S32 output at `q`.
  kiln::HbmRef OutputRows(Scalar q, int64_t rows) {
    return output_(kiln::Slice(q, rows), kiln::Slice(0, kLanes));
  }

  // Zero the high-word half of S64 storage in 16-row DMAs plus 1-row tails.
  void ZeroHighWords() {
    const Scalar rows = output_.Dim(0);
    const Scalar half = rows >> 1;
    const Scalar end = rows - (half & (kZeroRows - 1));
    // Borrow a flush buffer as a zero source before any flush runs.
    kiln::VmemRef zeros = packed_(0);
    kiln::Store(zeros, kiln::Zeros(kiln::i32(2 * kSublanes, kLanes)));
    kiln::For(half, end, kZeroRows, [this, &zeros](Scalar q) {
      kiln::SyncCopy(zeros, OutputRows(q, kZeroRows));
    });
    kiln::For(end, rows, [this, &zeros](Scalar q) {
      kiln::SyncCopy(zeros(kiln::Slice(0, 1)), OutputRows(q, 1));
    });
  }

  // Scan row windows when a row exceeds scratch or the input is rank one;
  // otherwise scan row blocks.
  void ProcessInput() {
    if (x_.rank() == 1) {
      ProcessWideRows();
      return;
    }
    kiln::If(
        n_ > kWindow, [this] { ProcessWideRows(); },
        [this] {
          const Window w = MakeWindow(kiln::I32(0), n_);
          const Scalar row_blocks = (m_ + (kSublanes - 1)) >> kSublaneShift;
          kiln::EmitPipeline(
              0, row_blocks,
              {InputBlocks([this, w](Scalar rb) { return BlockRows(w, rb); })},
              /*outputs=*/{}, [this, &w](const kiln::PipelineStep& s) {
                ProcessBlock(w, s.inputs[0].ref, s.step << kSublaneShift);
              });
        });
  }

  // Above kWindow columns, scan each row window by window to preserve order.
  void ProcessWideRows() {
    // Count windows so `base + kWindow` cannot overflow S32 near INT32_MAX.
    const Scalar windows = ((n_ - 1) >> kWindowShift) + 1;
    kiln::For(m_, [this, windows](Scalar row) {
      const Scalar rb = row >> kSublaneShift;
      kiln::EmitPipeline(
          0, windows, {InputBlocks([this, rb](Scalar window) {
            return BlockRows(WideWindow(window), rb);
          })},
          /*outputs=*/{}, [this, row](const kiln::PipelineStep& s) {
            const Window w = WideWindow(s.step);
            const kiln::VmemRef block = s.inputs[0].ref;
            kiln::For(w.tiles, [this, &w, &block, row](Scalar tile_idx) {
              Append(
                  CompactTile(w, block, tile_idx, row & (kSublanes - 1), row));
            });
          });
    });
  }

  // Window `window` of a row wider than kWindow.
  Window WideWindow(Scalar window) {
    const Scalar base = window * kWindow;
    return MakeWindow(base, Min(n_ - base, kWindow));
  }

  // Input stream whose step `s` holds `region(s)` of the input, prefetched
  // while step `s - 1` is compacted.
  kiln::PipelineOperand InputBlocks(
      absl::AnyInvocable<kiln::BlockRegion(Scalar) const> region) {
    const kiln::Dimensions block_shape =
        x_.rank() == 2 ? kiln::Dimensions{kSublanes, kWindow}
                       : kiln::Dimensions{kWindow};
    return {.hbm = x_, .block_shape = block_shape, .region = std::move(region)};
  }

  // Eight-row block `rb` of the input, over the columns of `w` (or the window
  // alone when the input is rank one).
  kiln::BlockRegion BlockRows(const Window& w, Scalar rb) const {
    if (x_.rank() == 2) {
      return {.offsets = {rb * kSublanes, w.base},
              .sizes = {kSublanes, w.n_pad}};
    }
    return {.offsets = {w.base}, .sizes = {w.n_pad}};
  }

  // Consume rows in logical order, independent of physical input tiling.
  void ProcessBlock(const Window& w, kiln::VmemRef block, Scalar r0) {
    kiln::For(kSublanes, [this, &w, &block, r0](Scalar r) {
      kiln::For(w.tiles, [this, &w, &block, r0, r](Scalar tile_idx) {
        Append(CompactTile(w, block, tile_idx, r, r0 + r));
      });
    });
  }

  // Compact one 1024-column segment of row `row_in_block` of an input block.
  Stream CompactTile(const Window& w, kiln::VmemRef block, Scalar tile_idx,
                     Scalar row_in_block, Scalar row) {
    const Scalar offset = tile_idx * kTile;
    const kiln::VmemRef segment =
        x_.rank() == 2
            ? block(kiln::Slice(row_in_block, 1), kiln::Slice(offset, kTile))
            : block(kiln::Slice(offset, kTile));
    const Tile values = kiln::Load(segment).Reshape({kSublanes, kLanes});
    const Scalar col0 = w.base + offset;
    const Tile flags = NonzeroFlags(values, row, col0);
    const auto [distance, total] = Prefix(1 - flags, flags);
    return {CompactPositions(flags, distance), total, row, col0};
  }

  // Mask in-bounds nonzeros; bitcast avoids flushing subnormals to zero.
  Tile NonzeroFlags(Tile values, Scalar row, Scalar col0) {
    const Tile nonzero = (values.Bitcast(kiln::i32) & 0x7fffffff) > 0;
    // Subtract `col0` rather than adding it to `pos_` to avoid S32 overflow.
    const Tile in_bounds = pos_ < Select(row < m_, n_ - col0, 0);
    return Select(nonzero & in_bounds, pos_.Full(1), 0);
  }

  // From zero flags `z` and live flags `m`, compute each live element's
  // left-shift distance (prior zeros) and the tile's live count.
  Ranks Prefix(Tile z, Tile m) {
    // Intra-row zero counts via MXU; prior-row totals via 3 sublane rotations.
    const Tile acc = kiln::Dot(z.ToBF16(), tri_).ToI32();
    const Tile row_zeros =
        acc.Slice({0, kLanes - 1}, {kSublanes, 1}).Lanes(kLanes);
    Tile above = row_zeros;
    for (int s = 0; s < kSublaneShift; ++s) {
      const int64_t st = int64_t{1} << s;
      above += Select(sublane_ >= st, above.Rotate(st, 0), 0);
    }
    return {(acc - z + (above - row_zeros)) * m,
            kTile - above.Extract({kSublanes - 1, 0})};
  }

  // Pack each live position with its shift distance and route it left bit by
  // bit, preserving relative order.
  Tile CompactPositions(Tile flags, Tile distance) {
    Tile word =
        Select(flags == 1, pos_ | (distance << kDistanceShift) | kAlive, 0);
    for (int s = 0; s < kTileShift; ++s) {
      const int64_t bit = int64_t{1} << (kDistanceShift + s);
      Tile src = word;
      if (s < kLaneShift) {
        // Shift left by `st` lanes, pulling wrapped lanes from the next row.
        const int64_t st = int64_t{1} << s;
        const Tile same = word.Rotate(kLanes - st, 1);
        src = Select(lane_ < kLanes - st, same, same.Rotate(kSublanes - 1, 0));
      } else {
        const int64_t st = int64_t{1} << (s - kLaneShift);
        src = Select(sublane_ < kSublanes - st, word.Rotate(kSublanes - st, 0),
                     0);
      }
      const Tile arriving = (src & bit) != 0;
      const Tile leaving = (word & bit) != 0;
      word = Select(arriving, src & ~bit, Select(leaving, 0, word));
    }
    return word;
  }

  // Stage up to 1024 coordinates at `fill_ % 1024`; rows 8..15 hold overflow
  // until rows 0..7 flush.
  void Append(const Stream& stream) {
    const Scalar fill = kiln::Load(fill_);
    const Scalar lane_offset = fill & (kLanes - 1);
    const Scalar stage_row = (fill >> kLaneShift) & (kSublanes - 1);
    const Tile rotated = stream.positions.Rotate(lane_offset, 1);
    // Rows stage_row..stage_row+7 take wrapped lanes from the row above.
    const Tile body = Select(lane_ >= lane_offset, rotated,
                             Select(sublane_ > 0, rotated.Rotate(1, 0), 0));
    // Row stage_row+8 takes lanes that wrapped past the last row.
    const Tile tail = Select(kiln::Iota(kiln::i32(1, kLanes), 1) < lane_offset,
                             rotated.Slice({kSublanes - 1, 0}, {1, kLanes}), 0);
    Stage(stage_row, body, stream);
    Stage(stage_row + kSublanes, tail, stream);
    const Scalar new_fill = fill + stream.total;
    kiln::Store(fill_, new_fill);
    kiln::If((new_fill >> kTileShift) > (fill >> kTileShift),
             [this, fill] { Flush((fill >> kTileShift) << kSublaneShift); });
  }

  // Masked store of live coordinates into staging rows starting at `first`.
  void Stage(Scalar first, Tile words, const Stream& stream) {
    const Tile live = (words & kAlive) != 0;
    const int64_t rows = words.dims()[0];
    kiln::Store(stage_rows_(kiln::Slice(first, rows)), words.Splat(stream.row),
                {.mask = live});
    kiln::Store(stage_cols_(kiln::Slice(first, rows)),
                stream.col0 + (words & (kTile - 1)), {.mask = live});
  }

  // DMA descriptor from flush `slot` to the `rank * 8` output rows at
  // `rank * q`. `Wait()` only depends on the semaphore and transfer size, so
  // any `q` suffices.
  kiln::AsyncCopyDescriptor FlushCopy(kiln::IndexArg slot, Scalar q) {
    const int64_t rows = x_.rank() * kSublanes;
    return kiln::MakeAsyncCopy(packed_(slot, kiln::Slice(0, rows)),
                               OutputRows(x_.rank() * q, rows),
                               flush_sems_(slot));
  }

  // Interleave 1024 staged (row, col) pairs, or copy 1024 indices for a
  // rank-one input, to HBM and shift overflow down.
  void Flush(Scalar q) {
    const Scalar flushes = kiln::Load(flush_count_);
    const Scalar slot = flushes & (kSlots - 1);
    const kiln::AsyncCopyDescriptor copy = FlushCopy(slot, q);
    kiln::If(flushes >= kSlots, [&copy] { copy.Wait(); });
    kiln::VmemRef packed = packed_(slot);
    if (x_.rank() == 2) {
      for (int r = 0; r < kSublanes; ++r) {
        kiln::Store(packed(kiln::Slice(2 * r, 1)),
                    kiln::Load(stage_rows_(kiln::Slice(r, 1))));
        kiln::Store(packed(kiln::Slice(2 * r + 1, 1)),
                    kiln::Load(stage_cols_(kiln::Slice(r, 1))));
      }
    } else {
      kiln::Store(packed(kiln::Slice(0, kSublanes)),
                  kiln::Load(stage_cols_(kiln::Slice(0, kSublanes))));
    }
    copy.Start();
    kiln::Store(flush_count_, flushes + 1);
    for (kiln::VmemRef stage : {stage_rows_, stage_cols_}) {
      kiln::Store(stage(kiln::Slice(0, kSublanes)),
                  kiln::Load(stage(kiln::Slice(kSublanes, kSublanes))));
      kiln::Store(stage(kiln::Slice(kSublanes, kSublanes)),
                  kiln::Zeros(kiln::i32(kSublanes, kLanes)));
    }
  }

  // Drain in-flight flushes and write any remaining partial stage rows.
  void FinishOutput() {
    const Scalar flushes = kiln::Load(flush_count_);
    for (int s = 0; s < kSlots; ++s) {
      kiln::If(flushes > s, [this, s] { FlushCopy(s, kiln::I32(0)).Wait(); });
    }
    const Scalar count = kiln::Load(fill_);
    const Scalar q = (count >> kTileShift) << kSublaneShift;
    const Scalar rows = ((count & (kTile - 1)) + (kLanes - 1)) >> kLaneShift;
    kiln::For(rows, [this, q](Scalar r) {
      const Scalar out = x_.rank() * (q + r);
      if (x_.rank() == 2) {
        kiln::SyncCopy(stage_rows_(kiln::Slice(r, 1)), OutputRows(out, 1));
      }
      kiln::SyncCopy(stage_cols_(kiln::Slice(r, 1)),
                     OutputRows(out + (x_.rank() - 1), 1));
    });
  }

  kiln::HbmRef x_;       // F32[M,N] or F32[N] input
  kiln::HbmRef output_;  // S32 view of native S64 storage
  // (16, 128) coordinate planes; rows are staged but unused for rank one.
  kiln::VmemRef stage_rows_, stage_cols_;
  kiln::VmemRef packed_;             // (kSlots, 16, 128) flush slots
  kiln::SmemRef fill_;               // Total appended coordinates
  kiln::SmemRef flush_count_;        // Flushes issued
  kiln::SemaphoreArray flush_sems_;  // Flush DMA completion per slot
  Tile lane_, sublane_, pos_, tri_;  // Constant vectors
  Scalar m_, n_;                     // Runtime input extents
};

// Define the Mosaic memrefs: input, private destination and aliased result.
// The HLO wrapper provides runtime dimensions and the result/destination
// alias.
absl::StatusOr<kiln::Kernel> BuildNonzeroIndices(int64_t rank) {
  const kiln::Dimensions dynamic_input(rank, kiln::kDynamic);
  const kiln::Dimensions dynamic_2d = {kiln::kDynamic, kiln::kDynamic};
  return kiln::CreateKernel(kiln::GridSpec{.name = "main"},
                            kiln::InSpec::Hbm(kiln::f32(dynamic_input)),
                            kiln::InSpec::Hbm(kiln::i32(dynamic_2d)),
                            kiln::OutSpec::Hbm(kiln::i32(dynamic_2d)),
                            [](kiln::HbmRef input, kiln::HbmRef /*destination*/,
                               kiln::HbmRef output) {
                              // The result aliases the private destination;
                              // only the result is written.
                              NonzeroKernel(input, output).Build();
                            });
}

// Returns an unbounded-dynamic shape of the tile's rank with a row-major
// tiled layout.
xla::Shape MakeUnbounded(const xla::PrimitiveType type, const xla::Tile& tile) {
  const int64_t rank = tile.dimensions().size();
  xla::Shape shape = xla::ShapeUtil::MakeShape(
      type, Dimensions(rank, xla::Shape::kUnboundedSize),
      std::vector<bool>(rank, true));
  *shape.mutable_layout()->add_tiles() = tile;
  return shape;
}

// Maps the logical S64[K,D] destination onto the S32[R,128] view the Mosaic
// kernel was compiled against; see the buffer contract in the header. Both
// occupy 1024 * D * ceil(K/128) bytes.
absl::StatusOr<xla::Shape> DestinationPhysicalShape(const Shape& logical,
                                                    const xla::Layout& layout) {
  const int64_t rows =
      2 * logical.dimensions()[1] * ((logical.dimensions()[0] + 127) / 128);
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::S32, {rows, 128});
  *shape.mutable_layout() = layout;
  return shape;
}

// Bitcasts the kernel's only result, the S32[R,128] destination, back to the
// logical S64[K,D]{0,1:T(D,128)} output; see the buffer contract in the
// header.
absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
BitcastToLogicalIndices(std::vector<std::unique_ptr<xla::PjRtBuffer>> results,
                        const absl::Span<const Shape> logical_output_shapes) {
  const Shape& logical = logical_output_shapes[0];
  // The kernel's documented destination layout; see the buffer contract in
  // the header. K > D keeps XLA's default S64[K,D] layout at {0,1:T(D,128)}.
  const xla::Layout layout({0, 1}, {xla::Tile({logical.dimensions()[1], 128})});
  const Dimensions dims = logical.dimensions();
  TT_ASSIGN_OR_RETURN(
      results[0],
      AdaptXlaError(
          results[0]->Bitcast(xla::S64, dims, &layout),
          /*context=*/"failed to bitcast the nonzero indices to S64"));
  return results;
}

// Returns the input layout of the buffer contract for rank-`kRank` inputs; see
// the header. A rank-one F32 array is stored in logical order whatever its
// tile, and XLA picks T(1024) for all but small arrays.
template <int kRank>
xla::Layout NonzeroInputLayout() {
  return kRank == 2 ? xla::Layout({1, 0}, {xla::Tile({8, 128})})
                    : xla::Layout({0}, {xla::Tile({1024})});
}

// Returns the parameters for the nonzero indices kernel.
template <int kRank>
std::vector<xla::Shape> NonzeroParameters() {
  const xla::Shape input =
      MakeUnbounded(xla::F32, NonzeroInputLayout<kRank>().tiles()[0]);
  const xla::Shape output_view = MakeUnbounded(xla::S32, xla::Tile({1, 128}));
  xla::Shape output_dims = xla::ShapeUtil::MakeShape(xla::S32, {2});
  *output_dims.mutable_layout() = xla::Layout({0}, {xla::Tile({128})});
  xla::Shape input_dims = xla::ShapeUtil::MakeShape(xla::S32, {kRank});
  *input_dims.mutable_layout() = output_dims.layout();
  return {input, output_view, input_dims, output_dims};
}

// Returns the UnboundedDynamicExecutionMetadata for the nonzero indices
// kernel.
template <int kRank>
UnboundedDynamicExecutionMetadata MakeNonzeroDynamicMetadata() {
  // One logical input, the F32[M,N] or F32[N] payload. The S64[K,kRank]
  // destination is a donated output: `Execute` allocates it directly in the
  // S32[R,128] view the kernel was compiled against, so callers never see the
  // physical view.
  UnboundedDynamicExecutionMetadata dynamic_meta{
      .logical_input_to_pjrt_param = {0},
      .donated_output_params =
          {
              UnboundedDynamicExecutionMetadata::DonatedOutputParam{
                  .logical_output_idx = 0,
                  .pjrt_param_idx = 1,
                  .physical_shape_fn = DestinationPhysicalShape,
              },
          },
      .metadata_params =
          {
              UnboundedDynamicExecutionMetadata::MetadataParamBinding{
                  .buffer_pjrt_idx = 0,  // F32 input -> its dimensions
                  .shape_pjrt_idx = 2,
              },
              UnboundedDynamicExecutionMetadata::MetadataParamBinding{
                  .buffer_pjrt_idx = 1,  // S32[R,128] destination -> [R,128]
                  .shape_pjrt_idx = 3,
              },
          },
      .pjrt_param_shapes = NonzeroParameters<kRank>(),
      .output_transform = BitcastToLogicalIndices,
      .is_compatible =
          [](absl::Span<const Shape> input_shapes,
             absl::Span<const Shape> output_shapes) {
            // The on-device layouts of the kernel's buffer contract.
            const std::vector<xla::Layout> expected_input_layouts = {
                NonzeroInputLayout<kRank>()};
            const std::vector<xla::Layout> expected_output_layouts = {
                xla::Layout({0, 1}, {xla::Tile({kRank, 128})})};
            // Resolved from the device defaults and any custom layouts on the
            // shapes.
            const absl::StatusOr<std::vector<xla::Layout>> input_layouts =
                ResolveXlaLayouts(input_shapes);
            const absl::StatusOr<std::vector<xla::Layout>> output_layouts =
                ResolveXlaLayouts(output_shapes);

            // The shapeless key already fixes the number, ranks and dtypes of
            // the operands: one rank-kRank F32 input and one rank-2 S64
            // output.
            return output_shapes[0].dimensions()[1] == kRank &&
                   input_layouts.ok() && output_layouts.ok() &&
                   *input_layouts == expected_input_layouts &&
                   *output_layouts == expected_output_layouts;
          },
  };
  return dynamic_meta;
}

// Builds and compiles the Mosaic nonzero kernel for rank-`kRank` inputs. See
// `GetNonzeroUnboundedDynamicKernels` in the header for their contracts.
template <int kRank>
absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
BuildNonzeroIndicesExecutable(xla::PjRtClient& client,
                              UniqueCompileOptions compile_options) {
  const xla::Shape output_view = MakeUnbounded(xla::S32, xla::Tile({1, 128}));
  const std::vector<xla::Shape> parameters = NonzeroParameters<kRank>();
  TT_ASSIGN_OR_RETURN(kiln::Kernel kernel, BuildNonzeroIndices(kRank));
  TT_ASSIGN_OR_RETURN(const std::string serialized_kernel,
                      std::move(kernel).Serialize());
  TT_ASSIGN_OR_RETURN(
      const std::unique_ptr<xla::HloModule> hlo,
      BuildMosaicCustomCall(serialized_kernel, parameters, output_view,
                            {2, 3, -1, -1}, /*aliased_parameter=*/1));
  xla::ExecutableBuildOptions& build =
      compile_options->executable_build_options;
  build.set_run_backend_only(true);
  build.set_result_layout(hlo->entry_computation_layout().result_shape());
  compile_options->argument_layouts = parameters;
  const xla::XlaComputation computation(hlo->ToProto());
  return AdaptXlaError(
      client.CompileAndLoad(computation, std::move(*compile_options)),
      /*context=*/"failed to compile the nonzero indices kernel");
}

// Returns the ShapelessKey a nonzero graph would be lowered to.
ShapelessKey NonZeroCacheKey(int64_t rank) {
  // Only ranks and dtypes reach the shapeless key, so these extents are
  // placeholders.
  const Dimensions input_dims(rank, -1);
  const Dimensions out_dims = {-1, static_cast<int64_t>(input_dims.size())};

  GraphSignature graph;
  graph.SetCorePinningMode(CorePinningMode::kUnpinned);
  const int in = graph.AddInput(input_dims, mlir::ElementType::F32);
  int out = -1;
  graph.AddOp(OpName::kNonzero, OpParamCacheKeys::Empty(),
              /*donated_inputs=*/{},
              [in, &out, &out_dims](GraphSignature::OpSignatureBuilder& op) {
                op.AddInput(in);
                out = op.AddOutput(out_dims, mlir::ElementType::I64);
              });
  graph.AddGraphOutput(out);
  return graph.GetKey().shapeless_key();
}

UnboundedDynamicKernel GetNonzero2DUnboundedDynamicKernel() {
  return UnboundedDynamicKernel{
      .build = BuildNonzeroIndicesExecutable<2>,
      .metadata = MakeNonzeroDynamicMetadata<2>(),
      .key = MakeUnboundedDynamicKernelCacheKey(
          NonZeroCacheKey(2),
          static_cast<int>(NonzeroKernelId::kNonzeroKGreaterThan2)),
      .enabled_by_default = true,
  };
}

UnboundedDynamicKernel GetNonzero1DUnboundedDynamicKernel() {
  return UnboundedDynamicKernel{
      .build = BuildNonzeroIndicesExecutable<1>,
      .metadata = MakeNonzeroDynamicMetadata<1>(),
      .key = MakeUnboundedDynamicKernelCacheKey(
          NonZeroCacheKey(1),
          static_cast<int>(NonzeroKernelId::kNonzeroInputTile1024)),
      .enabled_by_default = true,
  };
}

}  // namespace

std::vector<UnboundedDynamicKernel> GetNonzeroUnboundedDynamicKernels() {
  return {GetNonzero2DUnboundedDynamicKernel(),
          GetNonzero1DUnboundedDynamicKernel()};
}

}  // namespace torch_tpu
