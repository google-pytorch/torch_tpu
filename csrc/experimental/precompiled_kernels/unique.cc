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

#include "csrc/experimental/precompiled_kernels/unique.h"

#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/unbounded_dynamic_kernel_registry.h"
#include "csrc/experimental/precompiled_kernels/kernel_utils.h"
#include "csrc/experimental/precompiled_kernels/stream_compaction.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
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

using kiln::Scalar;
using kiln::Tile;

// Identifies each registered kernel. These IDs are part of the compilation
// cache key and must never be renumbered or reused. The values and counts
// kernels are told apart by their extentless keys, so they share IDs.
enum class UniqueKernelId : int {
  kInputMinorTile1024 = 1,
  kInputMinorTile512 = 2,
  kInputMinorTile256 = 3,
  kInputMinorTile128 = 4,
};

// The graph a kernel serves: what it writes for each run of equal values.
enum class UniqueOutput {
  kValues,  // `kUniqueValuesFromSorted`: the run's value, as S32[U].
  kCounts,  // `kUniqueCountsFromSorted`: the run's length, as S64[U].
};

// Input elements per pipelined window: 64 KiB, double-buffered in VMEM.
constexpr int64_t kWindow = 16 * kTile;
constexpr int64_t kWindowShift = 14;  // kWindow == 1 << kWindowShift

// Keeps `base + n_pad` and the tile-padded physical element count
// `ceil(N / 1024) * 1024` within int32 arithmetic.
constexpr int64_t kMaxSupportedN = std::numeric_limits<int32_t>::max() - 1024;

// Writes the unique values, or the counts, of the sorted int32 input `x` to
// `out`, an S32[R,128] view of the output.
//
// The input is streamed in 16384-element windows of 1024-element tiles. In
// each tile, the elements that differ from their predecessor (the starts of
// runs) are moved to the front, and appended to a two-tile staging buffer.
// Each full tile is copied to `out` asynchronously; the last partial tile is
// copied at the end.
//
// For values, each start appends its value. For counts, each start except the
// first appends the length of the run it closes, its index minus the previous
// start's; the last run's length, N minus the last start, is appended at the
// end. The S64 counts are below 2^31, so only their low words, the first half
// of `out`'s rows, are staged; the high words are zeroed.
class UniqueKernel {
 public:
  UniqueKernel(kiln::HbmRef x, kiln::HbmRef out, const UniqueOutput output)
      : x_(x),
        out_(out),
        output_(output),
        prev_key_(kiln::AllocaSmem(kiln::i32())),
        last_start_(kiln::AllocaSmem(kiln::i32())),
        lane_(kiln::Iota(kiln::i32(kSublanes, kLanes), 1)),
        sublane_(kiln::Iota(kiln::i32(kSublanes, kLanes), 0)),
        pos_(sublane_ * kLanes + lane_),
        tri_(PrefixTriangle()),
        staged_(out, /*planes=*/1, /*tile_rows=*/1,
                output == UniqueOutput::kCounts ? out.Dim(0) >> 1 : out.Dim(0)),
        n_(x.Dim(0)) {}

  void Build() {
    kiln::Store(prev_key_, kiln::I32(0));
    if (output_ == UniqueOutput::kCounts) {
      kiln::Store(last_start_, kiln::I32(0));
      ZeroHighWords(out_);
    }
    ProcessInput();
    if (output_ == UniqueOutput::kCounts) {
      staged_.Append({pos_.Splat(n_ - kiln::Load(last_start_))}, kiln::I32(1));
    }
    staged_.Finish();
  }

 private:
  void ProcessInput() {
    const Scalar windows = ((n_ - 1) >> kWindowShift) + 1;
    kiln::PipelineOperand in_op = {
        .hbm = x_,
        .block_shape = {kWindow},
        .region =
            [this](Scalar window) {
              const TileWindow w = WideWindow(window);
              return kiln::BlockRegion{.offsets = {w.base}, .sizes = {w.n_pad}};
            },
    };
    kiln::EmitPipeline(0, windows, {std::move(in_op)}, /*outputs=*/{},
                       [this](const kiln::PipelineStep& step) {
                         const TileWindow w = WideWindow(step.step);
                         const kiln::VmemRef block = step.inputs[0].ref;
                         kiln::For(w.tiles,
                                   [this, &w, &block](Scalar tile_idx) {
                                     ProcessTile(w, block, tile_idx);
                                   });
                       });
  }

  TileWindow WideWindow(const Scalar window) {
    const Scalar base = window * kWindow;
    return MakeTileWindow(base, Min(n_ - base, kWindow));
  }

  void ProcessTile(const TileWindow& w, const kiln::VmemRef block,
                   const Scalar tile_idx) {
    const Scalar offset = tile_idx * kTile;
    const Tile values = kiln::Load(block(kiln::Slice(offset, kTile)))
                            .Reshape({kSublanes, kLanes});
    const Scalar col0 = w.base + offset;
    const Tile flags = StartFlags(values, col0);
    kiln::Store(prev_key_, values.Extract({kSublanes - 1, kLanes - 1}));
    const CompactionRanks ranks = ComputePrefixRanks(flags, sublane_, tri_);
    kiln::If(ranks.total > 0, [this, &ranks, &values, col0] {
      if (output_ == UniqueOutput::kValues) {
        Tile compacted = values;
        CompactTile(ranks.distance, /*distance_shift=*/0, lane_, sublane_,
                    {&compacted});
        staged_.Append({compacted}, ranks.total);
        return;
      }
      Tile starts = pos_ + col0;
      CompactTile(ranks.distance, /*distance_shift=*/0, lane_, sublane_,
                  {&starts});
      // Each start closes the previous run; `Append` drops slots past `total`.
      staged_.Append({starts - ShiftRight(starts, kiln::Load(last_start_))},
                     ranks.total);
      // Starts are increasing, so the last live one is the largest.
      const Tile live_starts = Select(pos_ < ranks.total, starts, 0);
      kiln::Store(
          last_start_,
          kiln::MaxReduce(kiln::MaxReduce(live_starts, 1), 0).Extract({0, 0}));
    });
  }

  // Returns `tile` shifted one element later in flat row-major order, with
  // `first` at element 0.
  Tile ShiftRight(const Tile tile, const Scalar first) const {
    const Tile right_lanes = tile.Rotate(1, 1);
    const Tile from_row_above =
        Select(sublane_ > 0, right_lanes.Rotate(1, 0), first);
    return Select(lane_ > 0, right_lanes, from_row_above);
  }

  // Returns 1 where an in-bounds element starts a new run of equal values:
  // every element that differs from its predecessor across lane, sublane and
  // tile boundaries, and element 0 for values (for counts, the first run is
  // closed by the next start, so element 0 isn't flagged).
  Tile StartFlags(const Tile values, const Scalar col0) const {
    const Tile differs = values != ShiftRight(values, kiln::Load(prev_key_));
    const Tile is_start = output_ == UniqueOutput::kValues
                              ? differs | ((pos_ == 0) & (col0 == 0))
                              : differs & ((pos_ != 0) | (col0 != 0));
    const Tile in_bounds = pos_ < (n_ - col0);
    return Select(is_start & in_bounds, pos_.Full(1), 0);
  }

  kiln::HbmRef x_;
  kiln::HbmRef out_;
  const UniqueOutput output_;
  kiln::SmemRef prev_key_;    // Last element of the previous tile.
  kiln::SmemRef last_start_;  // Counts: index of the last start so far.
  Tile lane_, sublane_, pos_, tri_;
  StagedOutput staged_;
  const Scalar n_;
};

absl::StatusOr<kiln::Kernel> BuildUniqueKernel(const UniqueOutput output) {
  const kiln::Shape rows = kiln::i32(kiln::kDynamic, kiln::kDynamic);
  return kiln::CreateKernel(
      kiln::GridSpec{.name = "main"},
      kiln::InSpec::Hbm(kiln::i32(kiln::kDynamic)), kiln::InSpec::Hbm(rows),
      kiln::OutSpec::Hbm(rows),
      [output](kiln::HbmRef x, kiln::HbmRef /*destination*/, kiln::HbmRef out) {
        // The output aliases the private destination; only the output is
        // written.
        UniqueKernel(x, out, output).Build();
      });
}

template <int64_t kInputTileMinor>
std::vector<xla::Shape> UniqueParameters() {
  const xla::Shape input =
      MakeUnboundedShape(xla::S32, /*rank=*/1, {xla::Tile({kInputTileMinor})});
  return {input, MakeS32RowViewShape(), MakeRuntimeDimsShape(1),
          MakeRuntimeDimsShape(2)};
}

// Maps the logical S32[U] or S64[U] output onto the S32[R,128] view the kernel
// was compiled against; see the buffer contract in the header.
template <UniqueOutput kOutput>
absl::StatusOr<xla::Shape> DestinationPhysicalShape(const Shape& logical,
                                                    const xla::Layout& layout) {
  const int64_t u = logical.dimensions()[0];
  const int64_t tile_minor = Default1DTileMinor(u);
  const int64_t word_rows =
      ((u + tile_minor - 1) / tile_minor) * (tile_minor / kLanes);
  const int64_t rows =
      kOutput == UniqueOutput::kCounts ? 2 * word_rows : word_rows;
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::S32, {rows, kLanes});
  *shape.mutable_layout() = layout;
  return shape;
}

// Bitcasts the kernel's only result, the S32[R,128] view, back to the logical
// S32[U] or S64[U] output with XLA's default layout.
template <UniqueOutput kOutput>
absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>> BitcastToLogical(
    std::vector<std::unique_ptr<xla::PjRtBuffer>> results,
    const absl::Span<const Shape> logical_output_shapes) {
  const Dimensions& dims = logical_output_shapes[0].dimensions();
  const xla::Layout layout = Default1DLayout(Default1DTileMinor(dims[0]));
  TT_ASSIGN_OR_RETURN(
      results[0],
      WrapXlaError(results[0]->Bitcast(
                       kOutput == UniqueOutput::kCounts ? xla::S64 : xla::S32,
                       dims, &layout),
                   /*context=*/"failed to bitcast the unique output"));
  return results;
}

template <UniqueOutput kOutput, int64_t kInputTileMinor>
absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
BuildUniqueExecutable(xla::PjRtClient& client,
                      UniqueCompileOptions compile_options) {
  TT_ASSIGN_OR_RETURN(kiln::Kernel kernel, BuildUniqueKernel(kOutput));
  return CompileMosaicKernel(
      client, std::move(compile_options), std::move(kernel),
      UniqueParameters<kInputTileMinor>(), MakeS32RowViewShape(),
      /*operand_to_metadata=*/{2, 3, -1, -1},
      /*aliased_parameter=*/1);
}

// Whether the kernel for input tile `input_tile_minor` serves these shapes.
bool IsCompatible(const int64_t input_tile_minor,
                  const absl::Span<const Shape> input_shapes,
                  const absl::Span<const Shape> output_shapes) {
  // The extentless key already fixes one rank-1 int32 input and one rank-1
  // output of the graph's dtype.
  const int64_t n = input_shapes[0].dimensions()[0];
  const int64_t u = output_shapes[0].dimensions()[0];
  return 1 <= n && n <= kMaxSupportedN && 1 <= u &&
         HasDeviceLayouts(input_shapes, {Default1DLayout(input_tile_minor)}) &&
         HasDeviceLayouts(output_shapes,
                          {Default1DLayout(Default1DTileMinor(u))});
}

template <UniqueOutput kOutput, int64_t kInputTileMinor>
UnboundedDynamicExecutionMetadata MakeUniqueMetadata() {
  // One logical input, the sorted S32[N]. The output is a donated output:
  // `Execute` allocates it directly in the S32[R,128] view the kernel was
  // compiled against.
  return MakeUnaryDonatedOutputMetadata(
      UniqueParameters<kInputTileMinor>(), DestinationPhysicalShape<kOutput>,
      BitcastToLogical<kOutput>,
      [](absl::Span<const Shape> input_shapes,
         absl::Span<const Shape> output_shapes) {
        return IsCompatible(kInputTileMinor, input_shapes, output_shapes);
      });
}

template <UniqueOutput kOutput, int64_t kInputTileMinor>
UnboundedDynamicKernelRegistration GetUniqueKernel(const UniqueKernelId id) {
  const bool counts = kOutput == UniqueOutput::kCounts;
  return {
      .kernel =
          UnboundedDynamicKernel{
              .build = BuildUniqueExecutable<kOutput, kInputTileMinor>,
              .metadata = MakeUniqueMetadata<kOutput, kInputTileMinor>(),
              .key = MakeUnboundedDynamicKernelCacheKey(
                  UnaryOpExtentlessKey(
                      counts ? OpName::kUniqueCountsFromSorted
                             : OpName::kUniqueValuesFromSorted,
                      /*in_rank=*/1, mlir::ElementType::I32,
                      /*out_rank=*/1,
                      counts ? mlir::ElementType::I64 : mlir::ElementType::I32),
                  static_cast<int>(id)),
              .enabled_by_default = true,
          },
  };
}

template <UniqueOutput kOutput>
std::vector<UnboundedDynamicKernelRegistration> GetUniqueKernels() {
  return {GetUniqueKernel<kOutput, 1024>(UniqueKernelId::kInputMinorTile1024),
          GetUniqueKernel<kOutput, 512>(UniqueKernelId::kInputMinorTile512),
          GetUniqueKernel<kOutput, 256>(UniqueKernelId::kInputMinorTile256),
          GetUniqueKernel<kOutput, 128>(UniqueKernelId::kInputMinorTile128)};
}

}  // namespace

std::vector<UnboundedDynamicKernelRegistration>
GetUniqueUnboundedDynamicKernels() {
  std::vector<UnboundedDynamicKernelRegistration> kernels =
      GetUniqueKernels<UniqueOutput::kValues>();
  absl::c_move(GetUniqueKernels<UniqueOutput::kCounts>(),
               std::back_inserter(kernels));
  return kernels;
}

}  // namespace torch_tpu
