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
#include <limits>
#include <memory>
#include <utility>
#include <vector>

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
// cache key and must never be renumbered or reused.
enum class UniqueKernelId : int {
  kInputMinorTile1024 = 1,
  kInputMinorTile512 = 2,
  kInputMinorTile256 = 3,
  kInputMinorTile128 = 4,
};

// Input elements per pipelined window: 64 KiB, double-buffered in VMEM.
constexpr int64_t kWindow = 16 * kTile;
constexpr int64_t kWindowShift = 14;  // kWindow == 1 << kWindowShift

// Keeps `base + n_pad` and the tile-padded physical element count
// `ceil(N / 1024) * 1024` within int32 arithmetic.
constexpr int64_t kMaxSupportedN = std::numeric_limits<int32_t>::max() - 1024;

// Writes the unique values of the sorted int32 input `x` to `out`, an
// S32[R,128] view of the output.
//
// The input is streamed in 16384-element windows of 1024-element tiles. In
// each tile, the elements that differ from their predecessor are moved to the
// front, and appended to a two-tile staging buffer. Each full tile of unique
// values is copied to `out` asynchronously; the last partial tile is copied at
// the end.
class UniqueValuesKernel {
 public:
  UniqueValuesKernel(kiln::HbmRef x, kiln::HbmRef out)
      : x_(x),
        prev_key_(kiln::AllocaSmem(kiln::i32())),
        lane_(kiln::Iota(kiln::i32(kSublanes, kLanes), 1)),
        sublane_(kiln::Iota(kiln::i32(kSublanes, kLanes), 0)),
        pos_(sublane_ * kLanes + lane_),
        tri_(PrefixTriangle()),
        staged_(out, /*planes=*/1, /*tile_rows=*/1, out.Dim(0)),
        n_(x.Dim(0)) {}

  void Build() {
    kiln::Store(prev_key_, kiln::I32(0));
    ProcessInput();
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
    kiln::If(ranks.total > 0, [this, &ranks, &values] {
      Tile compacted = values;
      CompactTile(ranks.distance, /*distance_shift=*/0, lane_, sublane_,
                  {&compacted});
      staged_.Append({compacted}, ranks.total);
    });
  }

  // Returns 1 where an in-bounds element starts a new run of equal values:
  // element 0, and every element that differs from its predecessor across
  // lane, sublane and tile boundaries.
  Tile StartFlags(const Tile values, const Scalar col0) {
    const Tile right_lanes = values.Rotate(1, 1);
    const Tile prev_sublane_wrap =
        Select(sublane_ > 0, right_lanes.Rotate(1, 0), kiln::Load(prev_key_));
    const Tile prev = Select(lane_ > 0, right_lanes, prev_sublane_wrap);
    const Tile is_global_zero = (pos_ == 0) & (col0 == 0);
    const Tile in_bounds = pos_ < (n_ - col0);
    const Tile is_start = (values != prev) | is_global_zero;
    return Select(is_start & in_bounds, pos_.Full(1), 0);
  }

  kiln::HbmRef x_;
  kiln::SmemRef prev_key_;  // Last element of the previous tile.
  Tile lane_, sublane_, pos_, tri_;
  StagedOutput staged_;
  const Scalar n_;
};

absl::StatusOr<kiln::Kernel> BuildUniqueValuesKernel() {
  const kiln::Shape rows = kiln::i32(kiln::kDynamic, kiln::kDynamic);
  return kiln::CreateKernel(
      kiln::GridSpec{.name = "main"},
      kiln::InSpec::Hbm(kiln::i32(kiln::kDynamic)), kiln::InSpec::Hbm(rows),
      kiln::OutSpec::Hbm(rows),
      [](kiln::HbmRef x, kiln::HbmRef /*destination*/, kiln::HbmRef output) {
        // The output aliases the private destination; only the output is
        // written.
        UniqueValuesKernel(x, output).Build();
      });
}

template <int64_t kInputTileMinor>
std::vector<xla::Shape> UniqueValuesParameters() {
  const xla::Shape input =
      MakeUnboundedShape(xla::S32, /*rank=*/1, {xla::Tile({kInputTileMinor})});
  return {input, MakeS32RowViewShape(), MakeRuntimeDimsShape(1),
          MakeRuntimeDimsShape(2)};
}

// Maps the logical S32[U] output onto the S32[R,128] view the kernel was
// compiled against; see the buffer contract in the header.
absl::StatusOr<xla::Shape> DestinationPhysicalShape(const Shape& logical,
                                                    const xla::Layout& layout) {
  const int64_t u = logical.dimensions()[0];
  const int64_t tile_minor = Default1DTileMinor(u);
  const int64_t rows =
      ((u + tile_minor - 1) / tile_minor) * (tile_minor / kLanes);
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::S32, {rows, kLanes});
  *shape.mutable_layout() = layout;
  return shape;
}

// Bitcasts the kernel's only result, the S32[R,128] view, back to the logical
// S32[U] output with XLA's default layout.
absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
BitcastToLogicalValues(std::vector<std::unique_ptr<xla::PjRtBuffer>> results,
                       const absl::Span<const Shape> logical_output_shapes) {
  const Dimensions& dims = logical_output_shapes[0].dimensions();
  const xla::Layout layout = Default1DLayout(Default1DTileMinor(dims[0]));
  TT_ASSIGN_OR_RETURN(
      results[0],
      WrapXlaError(results[0]->Bitcast(xla::S32, dims, &layout),
                   /*context=*/"failed to bitcast the unique values"));
  return results;
}

template <int64_t kInputTileMinor>
absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
BuildUniqueValuesExecutable(xla::PjRtClient& client,
                            UniqueCompileOptions compile_options) {
  TT_ASSIGN_OR_RETURN(kiln::Kernel kernel, BuildUniqueValuesKernel());
  return CompileMosaicKernel(
      client, std::move(compile_options), std::move(kernel),
      UniqueValuesParameters<kInputTileMinor>(), MakeS32RowViewShape(),
      /*operand_to_metadata=*/{2, 3, -1, -1},
      /*aliased_parameter=*/1);
}

// Whether the kernel for input tile `input_tile_minor` serves these shapes.
bool IsCompatible(const int64_t input_tile_minor,
                  const absl::Span<const Shape> input_shapes,
                  const absl::Span<const Shape> output_shapes) {
  // The extentless key already fixes one rank-1 int32 input and one rank-1
  // int32 output.
  // `Unique2` returns early for empty inputs, so N >= 1 and hence U >= 1.
  const int64_t n = input_shapes[0].dimensions()[0];
  const int64_t u = output_shapes[0].dimensions()[0];
  return n <= kMaxSupportedN &&
         HasDeviceLayouts(input_shapes, {Default1DLayout(input_tile_minor)}) &&
         HasDeviceLayouts(output_shapes,
                          {Default1DLayout(Default1DTileMinor(u))});
}

template <int64_t kInputTileMinor>
UnboundedDynamicExecutionMetadata MakeUniqueValuesMetadata() {
  // One logical input, the sorted S32[N]. The S32[U] output is a donated
  // output: `Execute` allocates it directly in the S32[R,128] view the kernel
  // was compiled against.
  return MakeUnaryDonatedOutputMetadata(
      UniqueValuesParameters<kInputTileMinor>(), DestinationPhysicalShape,
      BitcastToLogicalValues,
      [](absl::Span<const Shape> input_shapes,
         absl::Span<const Shape> output_shapes) {
        return IsCompatible(kInputTileMinor, input_shapes, output_shapes);
      });
}

template <int64_t kInputTileMinor>
UnboundedDynamicKernelRegistration GetUniqueValuesKernel(
    const UniqueKernelId id) {
  return {
      .kernel =
          UnboundedDynamicKernel{
              .build = BuildUniqueValuesExecutable<kInputTileMinor>,
              .metadata = MakeUniqueValuesMetadata<kInputTileMinor>(),
              .key = MakeUnboundedDynamicKernelCacheKey(
                  UnaryOpExtentlessKey(OpName::kUniqueValuesFromSorted,
                                       /*in_rank=*/1, mlir::ElementType::I32,
                                       /*out_rank=*/1, mlir::ElementType::I32),
                  static_cast<int>(id)),
              .enabled_by_default = true,
          },
  };
}

}  // namespace

std::vector<UnboundedDynamicKernelRegistration>
GetUniqueUnboundedDynamicKernels() {
  return {GetUniqueValuesKernel<1024>(UniqueKernelId::kInputMinorTile1024),
          GetUniqueValuesKernel<512>(UniqueKernelId::kInputMinorTile512),
          GetUniqueValuesKernel<256>(UniqueKernelId::kInputMinorTile256),
          GetUniqueValuesKernel<128>(UniqueKernelId::kInputMinorTile128)};
}

}  // namespace torch_tpu
