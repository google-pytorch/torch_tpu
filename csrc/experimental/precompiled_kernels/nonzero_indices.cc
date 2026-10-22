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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
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
};

using kiln::Ref;
using kiln::Scalar;
using kiln::Tile;

constexpr int64_t kSublanes = 8;
constexpr int64_t kLanes = 128;
constexpr int64_t kTile = kSublanes * kLanes;  // Elements per compaction step.
constexpr int64_t kDistanceShift = 10;  // Distance field of a packed word.
constexpr int64_t kAlive = 1 << 20;     // Alive bit of a packed word.
constexpr int64_t kWindow = 16384;      // Scratch capacity, not an input limit.
constexpr int64_t kZeroRows = 16;  // Storage rows per high-word zeroing DMA.
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
//   4. Emit. Append copies the compacted coordinates into a staging plane pair
//      at the current output offset and flushes to HBM once 1024 pairs have
//      accumulated. Only this step touches HBM.
//
// Input arrives by DMA eight physical rows at a time, but compaction always
// visits logical rows in order, so the output never needs reordering. A row
// too wide for the kWindow scratch buffer is processed one window at a time.
// Input and output DMA are both synchronous at this stage.
//
// The output operand is an S32 view of the int64 allocation, shaped as
// 128-word storage rows; see the buffer contract in the header. Coordinates are
// nonnegative and fit in a low word, so ZeroHighWords clears the high plane
// once and nothing writes it again. Every 128 pairs occupy two consecutive
// storage rows: 128 row indices, then their 128 column indices. Every write
// lands inside the allocation's ordinary tile padding, so the caller need not
// over-allocate.
//
// Written in Kiln, so a `kiln::Scalar` or `kiln::Tile` is a device value and
// `kiln::For` / `kiln::If` emit device control flow; the C++ runs once at
// build time, and ordinary C++ loops unroll fixed instruction patterns.

// Geometry of one scratch window (a whole row block unless n_ > kWindow).
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
  NonzeroKernel(Ref x, Ref output)
      : x_(x),
        output_(output),
        buf_(kiln::AllocaVmem(kiln::f32(1, kSublanes, kWindow))),
        stage_rows_(kiln::AllocaVmem(kiln::i32(2 * kSublanes, kLanes))),
        stage_cols_(kiln::AllocaVmem(kiln::i32(2 * kSublanes, kLanes))),
        packed_(kiln::AllocaVmem(kiln::i32(2 * kSublanes, kLanes))),
        fill_(kiln::AllocaSmem(kiln::i32())),
        in_sem_(kiln::AllocaDmaSemaphore()),
        lane_(kiln::Iota(kiln::i32(kSublanes, kLanes), 1)),
        sublane_(kiln::Iota(kiln::i32(kSublanes, kLanes), 0)),
        pos_(sublane_ * kLanes + lane_),
        tri_(Triangle()),
        m_(x.Dim(0)),
        n_(x.Dim(1)) {}

  void Build() {
    fill_ = kiln::I32(0);
    stage_rows_ = kiln::Zeros(kiln::i32(2 * kSublanes, kLanes));
    stage_cols_ = kiln::Zeros(kiln::i32(2 * kSublanes, kLanes));
    ZeroHighWords();
    ProcessInput();
    FinishOutput();
  }

 private:
  // View `rows` 128-word storage rows of the S32 output at `q`.
  Ref OutputRows(Scalar q, int64_t rows) {
    return output_(kiln::Slice(q, rows), kiln::Slice(0, kLanes));
  }

  // Zero the high-word half of S64 storage in 16-row DMAs plus 1-row tails.
  void ZeroHighWords() {
    const Scalar rows = output_.Dim(0);
    const Scalar half = rows >> 1;
    const Scalar end = rows - (half & (kZeroRows - 1));
    // Borrow the flush buffer as a zero source before any flush runs.
    packed_ = kiln::Zeros(kiln::i32(2 * kSublanes, kLanes));
    kiln::For(half, end, kZeroRows, [this](Scalar q) {
      kiln::SyncCopy(packed_, OutputRows(q, kZeroRows));
    });
    kiln::For(end, rows, [this](Scalar q) {
      kiln::SyncCopy(packed_(kiln::Slice(0, 1)), OutputRows(q, 1));
    });
  }

  // Scan row windows when a row exceeds scratch; otherwise scan row blocks.
  void ProcessInput() {
    kiln::If(
        n_ > kWindow, [this] { ProcessWideRows(); },
        [this] {
          const Window w = MakeWindow(kiln::I32(0), n_);
          const Scalar row_blocks = (m_ + (kSublanes - 1)) >> kSublaneShift;
          kiln::For(row_blocks, [this, &w](Scalar rb) {
            Fetch(w, rb, 0);
            WaitInput(w, rb, 0);
            ProcessBlock(w, 0, rb << kSublaneShift);
          });
        });
  }

  // Above kWindow columns, scan each row window by window to preserve order.
  void ProcessWideRows() {
    // Count windows so `base + kWindow` cannot overflow S32 near INT32_MAX.
    const Scalar windows = ((n_ - 1) >> kWindowShift) + 1;
    kiln::For(m_, [this, windows](Scalar row) {
      kiln::For(windows, [this, row](Scalar window) {
        const Scalar base = window * kWindow;
        const Window w = MakeWindow(base, Min(n_ - base, kWindow));
        Fetch(w, row >> kSublaneShift, 0);
        WaitInput(w, row >> kSublaneShift, 0);
        kiln::For(w.tiles, [this, &w, row](Scalar tile_idx) {
          Append(CompactTile(w, 0, tile_idx, row & (kSublanes - 1), row));
        });
      });
    });
  }

  // DMA descriptor for eight-row block `rb` of `w` into scratch `slot`.
  kiln::AsyncCopyDescriptor BlockCopy(const Window& w, Scalar rb,
                                      kiln::IndexArg slot) {
    return kiln::MakeAsyncCopy(x_(kiln::Slice(rb * kSublanes, kSublanes),
                                  kiln::Slice(w.base, w.n_pad)),
                               buf_(slot, kiln::all, kiln::Slice(0, w.n_pad)),
                               in_sem_);
  }

  void Fetch(const Window& w, Scalar rb, kiln::IndexArg slot) {
    BlockCopy(w, rb, slot).Start();
  }

  void WaitInput(const Window& w, Scalar rb, kiln::IndexArg slot) {
    BlockCopy(w, rb, slot).Wait();
  }

  // Consume rows in logical order, independent of physical input tiling.
  void ProcessBlock(const Window& w, kiln::IndexArg slot, Scalar r0) {
    kiln::For(kSublanes, [this, &w, slot, r0](Scalar r) {
      kiln::For(w.tiles, [this, &w, slot, r0, r](Scalar tile_idx) {
        Append(CompactTile(w, slot, tile_idx, r, r0 + r));
      });
    });
  }

  // Compact one 1024-column segment of a fetched row into live positions.
  Stream CompactTile(const Window& w, kiln::IndexArg slot, Scalar tile_idx,
                     Scalar row_in_block, Scalar row) {
    const Scalar offset = tile_idx * kTile;
    const Tile values = Tile(buf_(slot, kiln::Slice(row_in_block, 1),
                                  kiln::Slice(offset, kTile)))
                            .Reshape({kSublanes, kLanes});
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

  // Stage up to 1024 pairs at `fill_ % 1024`; rows 8..15 hold overflow until
  // rows 0..7 flush.
  void Append(const Stream& stream) {
    const Scalar fill = Scalar(fill_);
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
    fill_ = new_fill;
    kiln::If((new_fill >> kTileShift) > (fill >> kTileShift),
             [this, fill] { Flush((fill >> kTileShift) << kSublaneShift); });
  }

  // Masked store of live coordinates into staging rows starting at `first`.
  void Stage(Scalar first, Tile words, const Stream& stream) {
    const Tile live = (words & kAlive) != 0;
    const int64_t rows = words.dims()[0];
    stage_rows_(kiln::Slice(first, rows)).Store(words.Splat(stream.row), live);
    stage_cols_(kiln::Slice(first, rows))
        .Store(stream.col0 + (words & (kTile - 1)), live);
  }

  // Interleave 1024 staged (row, col) pairs to HBM and shift overflow down.
  void Flush(Scalar q) {
    for (int r = 0; r < kSublanes; ++r) {
      packed_(kiln::Slice(2 * r, 1)) = Tile(stage_rows_(kiln::Slice(r, 1)));
      packed_(kiln::Slice(2 * r + 1, 1)) = Tile(stage_cols_(kiln::Slice(r, 1)));
    }
    kiln::SyncCopy(packed_, OutputRows(2 * q, 2 * kSublanes));
    for (Ref stage : {stage_rows_, stage_cols_}) {
      stage(kiln::Slice(0, kSublanes)) =
          Tile(stage(kiln::Slice(kSublanes, kSublanes)));
      stage(kiln::Slice(kSublanes, kSublanes)) =
          kiln::Zeros(kiln::i32(kSublanes, kLanes));
    }
  }

  // Flush any remaining partial stage rows within normal tile padding.
  void FinishOutput() {
    const Scalar count = Scalar(fill_);
    const Scalar q = (count >> kTileShift) << kSublaneShift;
    const Scalar rows = ((count & (kTile - 1)) + (kLanes - 1)) >> kLaneShift;
    kiln::For(rows, [this, q](Scalar r) {
      kiln::SyncCopy(stage_rows_(kiln::Slice(r, 1)),
                     OutputRows(2 * (q + r), 1));
      kiln::SyncCopy(stage_cols_(kiln::Slice(r, 1)),
                     OutputRows(2 * (q + r) + 1, 1));
    });
  }

  Ref x_;                            // HBM F32[M,N] input
  Ref output_;                       // HBM S32 view of native S64 storage
  Ref buf_;                          // VMEM (1, 8, kWindow) input block
  Ref stage_rows_, stage_cols_;      // VMEM (16, 128) coordinate planes
  Ref packed_;                       // VMEM interleaved coordinate tiles
  Ref fill_;                         // SMEM total appended coordinates
  kiln::Semaphore in_sem_;           // Input DMA completion
  Tile lane_, sublane_, pos_, tri_;  // Constant vectors
  Scalar m_, n_;                     // Runtime input extents
};

// Define the Mosaic memrefs: input, private destination and aliased result.
// The HLO wrapper provides runtime dimensions and the result/destination
// alias.
absl::StatusOr<kiln::Kernel> BuildNonzeroIndices() {
  const kiln::Dimensions dynamic_2d = {kiln::kDynamic, kiln::kDynamic};
  return kiln::CreateKernel(kiln::GridSpec{.name = "main"},
                            kiln::InSpec::Hbm(kiln::f32(dynamic_2d)),
                            kiln::InSpec::Hbm(kiln::i32(dynamic_2d)),
                            kiln::OutSpec::Hbm(kiln::i32(dynamic_2d)),
                            [](Ref input, Ref /*destination*/, Ref output) {
                              // The result aliases the private destination;
                              // only the result is written.
                              NonzeroKernel(input, output).Build();
                            });
}

// Returns a rank-2 unbounded-dynamic shape with a row-major tiled layout.
xla::Shape MakeUnbounded(const xla::PrimitiveType type, const xla::Tile& tile) {
  xla::Shape shape = xla::ShapeUtil::MakeShape(
      type, {xla::Shape::kUnboundedSize, xla::Shape::kUnboundedSize},
      {true, true});
  *shape.mutable_layout() = xla::Layout({1, 0}, {tile});
  return shape;
}

// Maps the logical S64[K,2] destination onto the S32[R,128] view the Mosaic
// kernel was compiled against; see the buffer contract in the header. Both
// occupy 2048 * ceil(K/128) bytes.
absl::StatusOr<xla::Shape> DestinationPhysicalShape(const Shape& logical,
                                                    const xla::Layout& layout) {
  const int64_t rows = 4 * ((logical.dimensions()[0] + 127) / 128);
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::S32, {rows, 128});
  *shape.mutable_layout() = layout;
  return shape;
}

// Bitcasts the kernel's only result, the S32[R,128] destination, back to the
// logical S64[K,2]{0,1:T(2,128)} output; see the buffer contract in the
// header.
absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
BitcastToLogicalIndices(std::vector<std::unique_ptr<xla::PjRtBuffer>> results,
                        const absl::Span<const Shape> logical_output_shapes) {
  const Shape& logical = logical_output_shapes[0];
  // The kernel's documented destination layout; see the buffer contract in
  // the header. K > 2 keeps XLA's default S64[K,2] layout at {0,1:T(2,128)}.
  const xla::Layout layout({0, 1}, {xla::Tile({2, 128})});
  const Dimensions dims = logical.dimensions();
  TT_ASSIGN_OR_RETURN(
      results[0],
      WrapXlaError(results[0]->Bitcast(xla::S64, dims, &layout),
                   /*context=*/"failed to bitcast the nonzero indices to S64"));
  return results;
}

// Returns the parameters for the nonzero indices kernel.
std::vector<xla::Shape> NonzeroParameters() {
  const xla::Shape input = MakeUnbounded(xla::F32, xla::Tile({8, 128}));
  const xla::Shape output_view = MakeUnbounded(xla::S32, xla::Tile({1, 128}));
  xla::Shape dims = xla::ShapeUtil::MakeShape(xla::S32, {2});
  *dims.mutable_layout() = xla::Layout({0}, {xla::Tile({128})});
  return {input, output_view, dims, dims};
}

// Returns the UnboundedDynamicExecutionMetadata for the nonzero indices
// kernel.
UnboundedDynamicExecutionMetadata MakeNonzeroDynamicMetadata() {
  // One logical input, the F32[M,N] payload. The S64[K,2] destination is a
  // donated output: `Execute` allocates it directly in the S32[R,128] view the
  // kernel was compiled against, so callers never see the physical view.
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
                  .buffer_pjrt_idx = 0,  // F32[M,N] input -> [M,N]
                  .shape_pjrt_idx = 2,
              },
              UnboundedDynamicExecutionMetadata::MetadataParamBinding{
                  .buffer_pjrt_idx = 1,  // S32[R,128] destination -> [R,128]
                  .shape_pjrt_idx = 3,
              },
          },
      .pjrt_param_shapes = NonzeroParameters(),
      .output_transform = BitcastToLogicalIndices,
      .is_compatible =
          [](absl::Span<const Shape> input_shapes,
             absl::Span<const Shape> output_shapes) {
            return input_shapes.size() == 1 &&
                   input_shapes[0].dimensions().size() == 2 &&
                   input_shapes[0].dtype() == mlir::ElementType::F32 &&
                   output_shapes.size() == 1 &&
                   output_shapes[0].dimensions().size() == 2 &&
                   output_shapes[0].dimensions()[0] > 2 &&
                   output_shapes[0].dimensions()[1] == 2 &&
                   output_shapes[0].dtype() == mlir::ElementType::I64;
          },
  };
  return dynamic_meta;
}

// Builds and compiles the Mosaic nonzero kernel. See
// `GetNonzeroUnboundedDynamicKernel` for its contract.
absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
BuildNonzeroIndicesExecutable(xla::PjRtClient& client,
                              UniqueCompileOptions compile_options) {
  const xla::Shape output_view = MakeUnbounded(xla::S32, xla::Tile({1, 128}));
  const std::vector<xla::Shape> parameters = NonzeroParameters();
  TT_ASSIGN_OR_RETURN(kiln::Kernel kernel, BuildNonzeroIndices());
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
  return WrapXlaError(
      client.CompileAndLoad(computation, std::move(*compile_options)),
      /*context=*/"failed to compile the nonzero indices kernel");
}

// Returns the ShapelessKey a nonzero graph would be lowered to.
ShapelessKey NonZeroCacheKey() {
  // Only ranks and dtypes reach the shapeless key, so these extents are
  // placeholders. The rank must be 2: the kernel only accepts F32[M,N].
  const Dimensions input_dims = {-1, -1};
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

}  // namespace

UnboundedDynamicKernel GetNonzeroUnboundedDynamicKernel() {
  return UnboundedDynamicKernel{
      .build = BuildNonzeroIndicesExecutable,
      .metadata = MakeNonzeroDynamicMetadata(),
      .key = MakeUnboundedDynamicKernelCacheKey(
          NonZeroCacheKey(),
          static_cast<int>(NonzeroKernelId::kNonzeroKGreaterThan2)),
  };
}

}  // namespace torch_tpu
