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

#include <bit>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/unbounded_dynamic_kernel_registry.h"
#include "csrc/experimental/precompiled_kernels/kernel_utils.h"
#include "csrc/experimental/precompiled_kernels/stream_compaction.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "csrc/internal/mosaic/kiln_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

// Identifies a precompiled nonzero indices kernel. Kernels of different input
// ranks or dtypes are told apart by their extentless keys, so they may share
// an id. The value is used as the `id_key` of the kernel's
// `CompilationCacheKey`, which is persisted in the names of the tier-2 cache
// entries, so values must be explicit, non-negative, and must never be reused
// or renumbered. -1 is reserved for keys that don't identify a dynamic kernel.
enum class NonzeroKernelId : int {
  // The nonzero indices kernel for 2D inputs.
  kNonzeroKGreaterThan2 = 1,
  // The nonzero indices kernels named by the minor dimension of their input's
  // first tile: 1D inputs, and 3D to 8D inputs with kNonzeroInputMinorTile128.
  kNonzeroInputMinorTile1024 = 2,
  kNonzeroInputMinorTile512 = 3,
  kNonzeroInputMinorTile256 = 4,
  kNonzeroInputMinorTile128 = 5,
};

using kiln::Scalar;
using kiln::Tile;

// Distance field of a position word.
constexpr int64_t kDistanceShift = kTileShift;
constexpr int64_t kWindow = 16384;    // Scratch capacity, not an input limit.
constexpr int64_t kZeroRows = 512;    // Storage rows per high-word zero DMA.
constexpr int64_t kWindowShift = 14;  // kWindow == 1 << kWindowShift

// Storage rows per 128 coordinate tuples in each word plane of the S64[K,rank]
// output: the second-minor tile of its layout {0,1:T(P,128)}, which rounds the
// rank up to a power of two.
constexpr int64_t TileRows(int64_t rank) {
  return static_cast<int64_t>(std::bit_ceil(static_cast<uint64_t>(rank)));
}

// Index phase of nonzero for a rank-two F32 input: write the (row, column)
// coordinates of every nonzero element, in input row-major order, into the
// caller's native S64[K,2] allocation. The caller already knows K, so nothing
// is returned to the host. As in torch.nonzero, only +0.0 and -0.0 are zero:
// NaNs, infinities and subnormals are nonzero. K must count exactly these
// elements, since every one found is written.
//
// A rank-one F32 input is handled as the single row of a rank-two input, and
// only its column coordinates are written, into an S64[K,1] allocation. An
// input of rank 3 to 8 is handled as its [M,N] slices in row-major order, in
// one loop whose index is split into the leading coordinates. A U8 input is
// handled like an F32 one, but rows of a U8 block are packed four to a 32-bit
// word, so RowSegment reads them through an I32 view of the block and
// NonzeroFlags checks the byte of the row in each word.
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
//   2. Rank. ComputePrefixRanks counts the zeros strictly before each element,
//      which is exactly how far left a live element must travel. Counting
//      within a 128-lane row is one MXU matmul against an upper-triangular
//      matrix; carrying the row totals down the eight sublanes is three
//      rotate-and-add steps.
//
//   3. Route. CompactSegment packs each live element into one i32 word,
//
//          bits 19..10 | bits 9..0
//          distance    | position
//
//      then runs CompactTile's ten-round butterfly: round s moves every word
//      whose distance still has bit s set exactly 2^s slots to the left and
//      clears that bit. A distance is below 1024, so ten rounds drive all of
//      them to zero, leaving each live position in the slot its rank names. A
//      round is one lane rotate, plus a sublane rotate for the lanes that cross
//      a row boundary.
//
//      Take the tile [0, a, 0, b]: a sits at position 1 behind one zero and b
//      at position 3 behind two, so round 0 moves a to slot 0 and round 1
//      moves b to slot 1.
//
//   4. Emit. Append copies the compacted coordinates into staging planes at
//      the current output offset and flushes to HBM once 1024 nonzeros have
//      accumulated. Only this step touches HBM.
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
// input, every 128 indices occupy one storage row. Above rank two, every 128
// tuples occupy TileRows(rank) storage rows, one per coordinate and then tile
// padding. Every write lands inside the allocation's ordinary tile padding, so
// the caller need not over-allocate.
//
// Written in Kiln, so a `kiln::Scalar` or `kiln::Tile` is a device value and
// `kiln::For` / `kiln::If` emit device control flow; the C++ runs once at
// build time, and ordinary C++ loops unroll fixed instruction patterns.

// Compacted 1024-column segment: positions in [0, total), live count, origin.
struct Stream {
  Tile positions;
  Scalar total, row, col0;
};

class NonzeroKernel {
 public:
  // Allocate scratch and prebuild constant tiles inside `CreateKernel`.
  NonzeroKernel(kiln::HbmRef x, kiln::HbmRef output)
      : x_(x),
        output_(output),
        lane_(kiln::Iota(kiln::i32(kSublanes, kLanes), 1)),
        sublane_(kiln::Iota(kiln::i32(kSublanes, kLanes), 0)),
        pos_(sublane_ * kLanes + lane_),
        tri_(PrefixTriangle()),
        staged_(output, x.rank(), TileRows(x.rank()), output.Dim(0) >> 1),
        zeros_(kiln::AllocaVmem(kiln::i32(kZeroRows, kLanes))),
        // A rank-one input is a single row.
        m_(x.rank() == 1 ? kiln::I32(1) : x.Dim(x.rank() - 2)),
        n_(x.Dim(x.rank() - 1)) {}

  void Build() {
    ZeroHighWords();
    ProcessInput();
    staged_.Finish();
  }

 private:
  // View `rows` 128-word storage rows of the S32 output at `q`.
  kiln::HbmRef OutputRows(Scalar q, kiln::IndexArg rows) {
    return output_(kiln::Slice(q, rows), kiln::Slice(0, kLanes));
  }

  // Zero the high-word half of S64 storage in kZeroRows-row DMAs, starting all
  // of them before waiting for any so that they overlap.
  void ZeroHighWords() {
    kiln::Store(zeros_, kiln::Zeros(kiln::i32(kZeroRows, kLanes)));
    const kiln::Semaphore sem = kiln::AllocaDmaSemaphore();
    const Scalar rows = output_.Dim(0);
    const Scalar half = rows >> 1;
    const auto copy = [this, &sem, &rows](Scalar q) {
      const Scalar n = Min(rows - q, kZeroRows);
      return kiln::MakeAsyncCopy(zeros_(kiln::Slice(0, n)), OutputRows(q, n),
                                 sem);
    };
    kiln::For(half, rows, kZeroRows, [&copy](Scalar q) { copy(q).Start(); });
    kiln::For(half, rows, kZeroRows, [&copy](Scalar q) { copy(q).Wait(); });
  }

  // Scan a rank-one input by row windows. Otherwise scan each [M,N] slice in
  // row-major order, with its leading indices in `lead_`.
  void ProcessInput() {
    if (x_.rank() == 1) {
      ProcessWideRows();
      return;
    }
    // Each slice fills at least one 4 KiB tile of HBM, so this fits in S32.
    Scalar slices = kiln::I32(1);
    for (int64_t axis = 0; axis < x_.rank() - 2; ++axis) {
      slices = slices * x_.Dim(axis);
    }
    kiln::For(slices, [this](Scalar slice) {
      // Split the slice index into leading indices, innermost first.
      lead_.clear();
      for (int64_t axis = x_.rank() - 3; axis >= 0; --axis) {
        lead_.insert(lead_.begin(), slice % x_.Dim(axis));
        slice = slice / x_.Dim(axis);
      }
      kiln::If(
          n_ > kWindow, [this] { ProcessWideRows(); },
          [this] {
            const TileWindow w = MakeTileWindow(kiln::I32(0), n_);
            const Scalar row_blocks = (m_ + (kSublanes - 1)) >> kSublaneShift;
            kiln::EmitPipeline(
                0, row_blocks, {InputBlocks([this, w](Scalar rb) {
                  return BlockRows(w, rb);
                })},
                /*outputs=*/{}, [this, &w](const kiln::PipelineStep& s) {
                  ProcessBlock(w, s.inputs[0].ref, s.step << kSublaneShift);
                });
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
            const TileWindow w = WideWindow(s.step);
            const kiln::VmemRef block = s.inputs[0].ref;
            kiln::For(w.tiles, [this, &w, &block, row](Scalar tile_idx) {
              Append(CompactSegment(w, block, tile_idx, row & (kSublanes - 1),
                                    row));
            });
          });
    });
  }

  // Window `window` of a row wider than kWindow.
  TileWindow WideWindow(Scalar window) {
    const Scalar base = window * kWindow;
    // DMAs of a rank-one U8 input move whole T(512)(128)(4,1) tiles, four
    // 128-element rows packed into 128 words.
    const int64_t granule =
        x_.rank() == 1 && x_.dtype() == kiln::u8 ? 4 * kLanes : kLanes;
    return MakeTileWindow(base, Min(n_ - base, kWindow), granule);
  }

  // Input stream whose step `s` holds `region(s)` of the input, prefetched
  // while step `s - 1` is compacted, with the leading dimensions squeezed out.
  kiln::PipelineOperand InputBlocks(
      absl::AnyInvocable<kiln::BlockRegion(Scalar) const> region) {
    kiln::BlockShape block_shape = x_.rank() == 1
                                       ? kiln::BlockShape{kWindow}
                                       : kiln::BlockShape{kSublanes, kWindow};
    block_shape.insert(block_shape.begin(), lead_.size(), kiln::squeezed);
    return {.hbm = x_, .block_shape = block_shape, .region = std::move(region)};
  }

  // Eight-row block `rb` of the [M,N] slice at `lead_`, over the columns of
  // `w` (or the window alone when the input is rank one).
  kiln::BlockRegion BlockRows(const TileWindow& w, Scalar rb) const {
    if (x_.rank() == 1) {
      return {.offsets = {w.base}, .sizes = {w.n_pad}};
    }
    kiln::BlockRegion region = {.offsets = {rb * kSublanes, w.base},
                                .sizes = {kSublanes, w.n_pad}};
    region.offsets.insert(region.offsets.begin(), lead_.begin(), lead_.end());
    region.sizes.insert(region.sizes.begin(), lead_.size(), 1);
    return region;
  }

  // Consume rows in logical order, independent of physical input tiling.
  void ProcessBlock(const TileWindow& w, kiln::VmemRef block, Scalar r0) {
    kiln::For(kSublanes, [this, &w, &block, r0](Scalar r) {
      kiln::For(w.tiles, [this, &w, &block, r0, r](Scalar tile_idx) {
        Append(CompactSegment(w, block, tile_idx, r, r0 + r));
      });
    });
  }

  // Compact one 1024-column segment of row `row_in_block` of an input block.
  Stream CompactSegment(const TileWindow& w, kiln::VmemRef block,
                        Scalar tile_idx, Scalar row_in_block, Scalar row) {
    const Scalar offset = tile_idx * kTile;
    const bool is_u8 = x_.dtype() == kiln::u8;
    const Tile loaded =
        kiln::Load(RowSegment(block, row_in_block, kiln::Slice(offset, kTile)));
    // A rank-one U8 segment is loaded whole, so it's widened here.
    const Tile values = (is_u8 && x_.rank() == 1 ? loaded.ToI32() : loaded)
                            .Reshape({kSublanes, kLanes});
    // Above rank one, a U8 row is byte `row_in_block % 4` of its words.
    const Scalar shift =
        is_u8 && x_.rank() != 1 ? (row_in_block & 3) << 3 : kiln::I32(0);
    const Scalar col0 = w.base + offset;
    const Tile flags = NonzeroFlags(values.Bitcast(kiln::i32), shift,
                                    is_u8 ? 8 : 31, row, col0);
    const auto [distance, total] = ComputePrefixRanks(flags, sublane_, tri_);
    const Tile positions =
        CompactTile(Select(flags == 1, pos_ | (distance << kDistanceShift), 0),
                    kDistanceShift, lane_, sublane_);
    return {positions, total, row, col0};
  }

  // Returns `columns` of row `row_in_block` of `block`. A U8 block packs rows
  // 4g to 4g+3 into the bytes of row g of its I32 view, the lowest row in the
  // lowest byte, and Mosaic can't load those rows alone, so above rank one a
  // U8 row comes with the three rows packed alongside it.
  kiln::VmemRef RowSegment(kiln::VmemRef block, Scalar row_in_block,
                           kiln::SliceDim columns) const {
    if (x_.rank() == 1) {
      return block(columns);
    }
    if (x_.dtype() == kiln::u8) {
      return block.Bitcast(kiln::i32)(kiln::Slice(row_in_block >> 2, 1),
                                      columns);
    }
    return block(kiln::Slice(row_in_block, 1), columns);
  }

  // Mask in-bounds elements with any of the `bits` bits from `shift` of their
  // I32 words set: those of a U8, or all but the sign of an F32, so that -0.0
  // is zero (the bitcast to I32 avoids flushing subnormals to zero).
  Tile NonzeroFlags(Tile words, Scalar shift, int64_t bits, Scalar row,
                    Scalar col0) {
    const Tile nonzero = ((words >> shift) & ((int64_t{1} << bits) - 1)) > 0;
    // Subtract `col0` rather than adding it to `pos_` to avoid S32 overflow.
    const Tile in_bounds = pos_ < Select(row < m_, n_ - col0, 0);
    return Select(nonzero & in_bounds, pos_.Full(1), 0);
  }

  // Stage up to 1024 coordinates: leading indices, row (above rank one), then
  // column.
  void Append(const Stream& stream) {
    std::vector<Scalar> scalar_prefix = lead_;
    if (x_.rank() != 1) {
      scalar_prefix.push_back(stream.row);
    }
    staged_.Append({stream.col0 + stream.positions}, stream.total,
                   scalar_prefix);
  }

  kiln::HbmRef x_;                   // F32 or U8 input of rank 1 to 8
  kiln::HbmRef output_;              // S32 view of native S64 storage
  Tile lane_, sublane_, pos_, tri_;  // Constant vectors
  StagedOutput staged_;              // Double-buffered HBM output writer
  kiln::VmemRef zeros_;              // Zero source for ZeroHighWords
  Scalar m_, n_;                     // Runtime extents of an [M,N] slice
  // Leading indices of the [M,N] slice being emitted.
  std::vector<Scalar> lead_;
};

// Define the Mosaic memrefs: input, private destination and aliased result.
// The HLO wrapper provides runtime dimensions and the result/destination
// alias.
absl::StatusOr<kiln::Kernel> BuildNonzeroIndices(int64_t rank,
                                                 kiln::DType dtype) {
  const kiln::Dimensions dynamic_input(rank, kiln::kDynamic);
  const kiln::Dimensions dynamic_2d = {kiln::kDynamic, kiln::kDynamic};
  return kiln::CreateKernel(kiln::GridSpec{.name = "main"},
                            kiln::InSpec::Hbm(dtype(dynamic_input)),
                            kiln::InSpec::Hbm(kiln::i32(dynamic_2d)),
                            kiln::OutSpec::Hbm(kiln::i32(dynamic_2d)),
                            [](kiln::HbmRef input, kiln::HbmRef /*destination*/,
                               kiln::HbmRef output) {
                              // The result aliases the private destination;
                              // only the result is written.
                              NonzeroKernel(input, output).Build();
                            });
}

// Maps the logical S64[K,D] destination onto the S32[R,128] view the Mosaic
// kernel was compiled against; see the buffer contract in the header. Both
// occupy 1024 * TileRows(D) * ceil(K/128) bytes.
absl::StatusOr<xla::Shape> DestinationPhysicalShape(const Shape& logical,
                                                    const xla::Layout& layout) {
  const int64_t rows = 2 * TileRows(logical.dimensions()[1]) *
                       ((logical.dimensions()[0] + 127) / 128);
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::S32, {rows, 128});
  *shape.mutable_layout() = layout;
  return shape;
}

// Bitcasts the kernel's only result, the S32[R,128] destination, back to the
// logical S64[K,D]{0,1:T(P,128)} output, P = TileRows(D); see the buffer
// contract in the header.
absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
BitcastToLogicalIndices(std::vector<std::unique_ptr<xla::PjRtBuffer>> results,
                        const absl::Span<const Shape> logical_output_shapes) {
  const Shape& logical = logical_output_shapes[0];
  // The kernel's documented destination layout; see the buffer contract in
  // the header. K > P keeps XLA's default S64[K,D] layout at {0,1:T(P,128)}.
  const xla::Layout layout(
      {0, 1}, {xla::Tile({TileRows(logical.dimensions()[1]), 128})});
  const Dimensions dims = logical.dimensions();
  TT_ASSIGN_OR_RETURN(
      results[0],
      WrapXlaError(results[0]->Bitcast(xla::S64, dims, &layout),
                   /*context=*/"failed to bitcast the nonzero indices to S64"));
  return results;
}

// Returns the input layout of the buffer contract for rank-`kRank` `kDtype`
// inputs whose first tile is T(8,kTileMinorDim) or T(kTileMinorDim); see the
// header. A rank-one F32 array is stored in logical order whatever its tile,
// and the kernel reads no further than the T(128) padding, so the 1-D kernels
// differ only in this layout. U8 layouts add inner tiles that pack four rows
// (of 128 elements for rank one) to a word, which loads unpack.
template <mlir::ElementType kDtype, int kRank, int64_t kTileMinorDim>
xla::Layout NonzeroInputLayout() {
  Indices minor_to_major(kRank);
  std::iota(minor_to_major.rbegin(), minor_to_major.rend(), 0);
  std::vector<xla::Tile> tiles = {kRank == 1 ? xla::Tile({kTileMinorDim})
                                             : xla::Tile({8, kTileMinorDim})};
  if constexpr (kDtype == mlir::ElementType::UI8 && kRank == 1) {
    tiles.push_back(xla::Tile({kLanes}));
  }
  if constexpr (kDtype == mlir::ElementType::UI8) {
    tiles.push_back(xla::Tile({4, 1}));
  }
  return xla::Layout(minor_to_major, tiles);
}

// Returns the parameters for the nonzero indices kernel.
template <mlir::ElementType kDtype, int kRank, int64_t kTileMinorDim>
std::vector<xla::Shape> NonzeroParameters() {
  const xla::Shape input = MakeUnboundedShape(
      ConvertTo<xla::PrimitiveType>(kDtype), kRank,
      NonzeroInputLayout<kDtype, kRank, kTileMinorDim>().tiles());
  return {input, MakeS32RowViewShape(), MakeRuntimeDimsShape(kRank),
          MakeRuntimeDimsShape(2)};
}

// Returns the UnboundedDynamicExecutionMetadata for the nonzero indices
// kernel.
template <mlir::ElementType kDtype, int kRank, int64_t kTileMinorDim>
UnboundedDynamicExecutionMetadata MakeNonzeroDynamicMetadata() {
  // One logical input, the [...,M,N] or [N] payload. The
  // S64[K,kRank] destination is a donated output: `Execute` allocates it
  // directly in the S32[R,128] view the kernel was compiled against, so callers
  // never see the physical view.
  return MakeUnaryDonatedOutputMetadata(
      NonzeroParameters<kDtype, kRank, kTileMinorDim>(),
      DestinationPhysicalShape, BitcastToLogicalIndices,
      [](absl::Span<const Shape> input_shapes,
         absl::Span<const Shape> output_shapes) {
        // The extentless key already fixes the number, ranks and dtypes of
        // the operands: one rank-kRank kDtype input and one rank-2 S64
        // output. The layouts are those of the kernel's buffer contract.
        return output_shapes[0].dimensions()[1] == kRank &&
               HasDeviceLayouts(
                   input_shapes,
                   {NonzeroInputLayout<kDtype, kRank, kTileMinorDim>()}) &&
               HasDeviceLayouts(
                   output_shapes,
                   {xla::Layout({0, 1}, {xla::Tile({TileRows(kRank), 128})})});
      });
}

// Builds and compiles the Mosaic nonzero kernel for rank-`kRank` inputs. See
// `GetNonzeroUnboundedDynamicKernels` in the header for their contracts.
template <mlir::ElementType kDtype, int kRank, int64_t kTileMinorDim>
absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
BuildNonzeroIndicesExecutable(xla::PjRtClient& client,
                              UniqueCompileOptions compile_options) {
  TT_ASSIGN_OR_RETURN(const kiln::DType kiln_dtype, ToKilnDType(kDtype));
  TT_ASSIGN_OR_RETURN(kiln::Kernel kernel,
                      BuildNonzeroIndices(kRank, kiln_dtype));
  return CompileMosaicKernel(client, std::move(compile_options),
                             std::move(kernel),
                             NonzeroParameters<kDtype, kRank, kTileMinorDim>(),
                             MakeS32RowViewShape(), {2, 3, -1, -1},
                             /*aliased_parameter=*/1);
}

// Returns the id of the kernel for rank-`rank` inputs whose first tile has
// minor dimension `tile_minor_dim`; see NonzeroKernelId.
constexpr NonzeroKernelId GetNonzeroKernelId(const int rank,
                                             const int64_t tile_minor_dim) {
  if (rank == 2) {
    return NonzeroKernelId::kNonzeroKGreaterThan2;
  }
  switch (tile_minor_dim) {
    case 1024:
      return NonzeroKernelId::kNonzeroInputMinorTile1024;
    case 512:
      return NonzeroKernelId::kNonzeroInputMinorTile512;
    case 256:
      return NonzeroKernelId::kNonzeroInputMinorTile256;
    default:
      return NonzeroKernelId::kNonzeroInputMinorTile128;
  }
}

template <mlir::ElementType kDtype, int kRank, int64_t kTileMinorDim = 128>
UnboundedDynamicKernelRegistration GetNonzeroUnboundedDynamicKernel() {
  return {
      .kernel =
          UnboundedDynamicKernel{
              .build =
                  BuildNonzeroIndicesExecutable<kDtype, kRank, kTileMinorDim>,
              .metadata =
                  MakeNonzeroDynamicMetadata<kDtype, kRank, kTileMinorDim>(),
              .key = MakeUnboundedDynamicKernelCacheKey(
                  UnaryOpExtentlessKey(OpName::kNonzero, kRank, kDtype,
                                       /*out_rank=*/2, mlir::ElementType::I64),
                  static_cast<int>(GetNonzeroKernelId(kRank, kTileMinorDim))),
              .enabled_by_default = true,
          },
  };
}

// Returns the `kDtype` kernels: one 1-D kernel for each of XLA's default 1-D
// layouts for `kDtype`, whose first tile is T(k1dTileMinorDim), then the 2-D
// to 8-D kernels. NonzeroInputLayout adds the rest of each layout, e.g. the
// (128)(4,1) of U8[N]{0:T(1024)(128)(4,1)}.
template <mlir::ElementType kDtype, int64_t... k1dTileMinorDim>
std::vector<UnboundedDynamicKernelRegistration> GetNonzeroKernelsOfDtype() {
  return {GetNonzeroUnboundedDynamicKernel<kDtype, 1, k1dTileMinorDim>()...,
          GetNonzeroUnboundedDynamicKernel<kDtype, 2>(),
          GetNonzeroUnboundedDynamicKernel<kDtype, 3>(),
          GetNonzeroUnboundedDynamicKernel<kDtype, 4>(),
          GetNonzeroUnboundedDynamicKernel<kDtype, 5>(),
          GetNonzeroUnboundedDynamicKernel<kDtype, 6>(),
          GetNonzeroUnboundedDynamicKernel<kDtype, 7>(),
          GetNonzeroUnboundedDynamicKernel<kDtype, 8>()};
}

}  // namespace

std::vector<UnboundedDynamicKernelRegistration>
GetNonzeroUnboundedDynamicKernels() {
  std::vector<UnboundedDynamicKernelRegistration> kernels =
      GetNonzeroKernelsOfDtype<mlir::ElementType::F32, 1024, 512, 256, 128>();
  absl::c_move(GetNonzeroKernelsOfDtype<mlir::ElementType::UI8, 1024, 512>(),
               std::back_inserter(kernels));
  return kernels;
}

}  // namespace torch_tpu
