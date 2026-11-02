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

#include "csrc/experimental/precompiled_kernels/nonzero_indices_common.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
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

namespace torch_tpu::internal {
namespace {

using kiln::Ref;
using kiln::Scalar;
using kiln::Tile;

constexpr int64_t kLaneShift = 7;       // kLanes == 1 << kLaneShift
constexpr int64_t kTileShift = 10;      // kTile == 1 << kTileShift
constexpr int64_t kDistanceShift = 10;  // Distance field of a packed word.
constexpr int64_t kAlive = 1 << 20;     // Alive bit of a packed word.
constexpr int64_t kZeroRows = 16;  // Storage rows per high-word zeroing DMA.
constexpr int64_t kSlots = 2;      // Double-buffered flush slots.

// T[i,j] = (i <= j), so z*T gives inclusive zero counts along each 128-lane
// row. Inputs are 0/1 BF16 and accumulation is F32, so counts are exact.
Tile Triangle() {
  const Tile i = kiln::Iota(kiln::i32(kLanes, kLanes), 0);
  const Tile j = kiln::Iota(kiln::i32(kLanes, kLanes), 1);
  const Tile one = kiln::Full(kiln::bf16(kLanes, kLanes), 1.0);
  return Select(i <= j, one, 0.0);
}

// One VMEM (16, 128) staging plane per output column.
std::vector<Ref> AllocaPlanes(const int64_t rank) {
  std::vector<Ref> planes;
  planes.reserve(rank);
  for (int64_t p = 0; p < rank; ++p) {
    planes.push_back(kiln::AllocaVmem(kiln::i32(2 * kSublanes, kLanes)));
  }
  return planes;
}

}  // namespace

Window MakeWindow(Scalar base, Scalar width) {
  // TODO(b/562066667): Drop MultipleOf once the OSS jaxlib proves `& -kLanes`.
  const Scalar n_pad =
      kiln::MultipleOf((width + (kLanes - 1)) & -kLanes, kLanes);
  return {base, n_pad, (n_pad + (kTile - 1)) >> kTileShift};
}

NonzeroCommon::NonzeroCommon(Ref output, int64_t rank, Scalar m, Scalar n)
    : output_(output),
      planes_(AllocaPlanes(rank)),
      packed_(kiln::AllocaVmem(kiln::i32(kSlots, 2 * kSublanes, kLanes))),
      fill_(kiln::AllocaSmem(kiln::i32())),
      flush_count_(kiln::AllocaSmem(kiln::i32())),
      flush_sems_(kiln::AllocaDmaSemaphore({kSlots})),
      lane_(kiln::Iota(kiln::i32(kSublanes, kLanes), 1)),
      sublane_(kiln::Iota(kiln::i32(kSublanes, kLanes), 0)),
      pos_(sublane_ * kLanes + lane_),
      tri_(Triangle()),
      m_(m),
      n_(n) {}

void NonzeroCommon::Build() {
  kiln::Store(fill_, kiln::I32(0));
  kiln::Store(flush_count_, kiln::I32(0));
  for (Ref plane : planes_) {
    kiln::Store(plane, kiln::Zeros(kiln::i32(2 * kSublanes, kLanes)));
  }
  ZeroHighWords();
  ProcessInput();
  FinishOutput();
}

Window NonzeroCommon::WideWindow(Scalar window) const {
  const Scalar base = window * kWindow;
  return MakeWindow(base, Min(n_ - base, kWindow));
}

void NonzeroCommon::AppendTile(Tile values, Scalar row, Scalar col0) {
  const Tile flags = NonzeroFlags(values, row, col0);
  const auto [distance, total] = Prefix(1 - flags, flags);
  Append({CompactPositions(flags, distance), total, row, col0});
}

// View `rows` 128-word storage rows of the S32 output at `q`.
Ref NonzeroCommon::OutputRows(Scalar q, int64_t rows) {
  return output_(kiln::Slice(q, rows), kiln::Slice(0, kLanes));
}

// Zero the high-word half of S64 storage in 16-row DMAs plus 1-row tails.
void NonzeroCommon::ZeroHighWords() {
  const Scalar rows = output_.Dim(0);
  const Scalar half = rows >> 1;
  const Scalar end = rows - (half & (kZeroRows - 1));
  // Borrow a flush buffer as a zero source before any flush runs.
  Ref zeros = packed_(0);
  kiln::Store(zeros, kiln::Zeros(kiln::i32(2 * kSublanes, kLanes)));
  kiln::For(half, end, kZeroRows, [this, &zeros](Scalar q) {
    kiln::SyncCopy(zeros, OutputRows(q, kZeroRows));
  });
  kiln::For(end, rows, [this, &zeros](Scalar q) {
    kiln::SyncCopy(zeros(kiln::Slice(0, 1)), OutputRows(q, 1));
  });
}

// Mask in-bounds nonzeros; bitcast avoids flushing subnormals to zero.
Tile NonzeroCommon::NonzeroFlags(Tile values, Scalar row, Scalar col0) {
  const Tile nonzero = (values.Bitcast(kiln::i32) & 0x7fffffff) > 0;
  // Subtract `col0` rather than adding it to `pos_` to avoid S32 overflow.
  const Tile in_bounds = pos_ < Select(row < m_, n_ - col0, 0);
  return Select(nonzero & in_bounds, pos_.Full(1), 0);
}

// From zero flags `z` and live flags `live`, compute each live element's
// left-shift distance (prior zeros) and the tile's live count.
NonzeroCommon::Ranks NonzeroCommon::Prefix(Tile z, Tile live) {
  // Intra-row zero counts via MXU; prior-row totals via 3 sublane rotations.
  const Tile acc = kiln::Dot(z.ToBF16(), tri_).ToI32();
  const Tile row_zeros =
      acc.Slice({0, kLanes - 1}, {kSublanes, 1}).Lanes(kLanes);
  Tile above = row_zeros;
  for (int s = 0; s < kSublaneShift; ++s) {
    const int64_t st = int64_t{1} << s;
    above += Select(sublane_ >= st, above.Rotate(st, 0), 0);
  }
  return {(acc - z + (above - row_zeros)) * live,
          kTile - above.Extract({kSublanes - 1, 0})};
}

// Pack each live position with its shift distance and route it left bit by
// bit, preserving relative order.
Tile NonzeroCommon::CompactPositions(Tile flags, Tile distance) {
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
      src =
          Select(sublane_ < kSublanes - st, word.Rotate(kSublanes - st, 0), 0);
    }
    const Tile arriving = (src & bit) != 0;
    const Tile leaving = (word & bit) != 0;
    word = Select(arriving, src & ~bit, Select(leaving, 0, word));
  }
  return word;
}

// Stage up to 1024 coordinates at `fill_ % 1024`; rows 8..15 hold overflow
// until rows 0..7 flush.
void NonzeroCommon::Append(const Stream& stream) {
  const Scalar fill = kiln::LoadScalar(fill_);
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

// Masked store of live coordinates into staging rows starting at `first`: the
// row index into the row plane of a rank-two input, and the column index into
// the last plane.
void NonzeroCommon::Stage(Scalar first, Tile words, const Stream& stream) {
  const Tile live = (words & kAlive) != 0;
  const int64_t rows = words.dims()[0];
  if (planes_.size() == 2) {
    kiln::Store(planes_.front()(kiln::Slice(first, rows)),
                words.Splat(stream.row), {.mask = live});
  }
  kiln::Store(planes_.back()(kiln::Slice(first, rows)),
              stream.col0 + (words & (kTile - 1)), {.mask = live});
}

// DMA descriptor from flush `slot` to the output rows of the 1024
// coordinates at storage row group `q`: rows `rank*q .. rank*(q+8)-1`.
// `Wait()` only depends on the semaphore and transfer size, so any `q`
// suffices.
kiln::AsyncCopyDescriptor NonzeroCommon::FlushCopy(kiln::IndexArg slot,
                                                   Scalar q) {
  const int64_t num_planes = static_cast<int64_t>(planes_.size());
  const int64_t rows = num_planes * kSublanes;
  return kiln::MakeAsyncCopy(packed_(slot, kiln::Slice(0, rows)),
                             OutputRows(num_planes * q, rows),
                             flush_sems_(slot));
}

// Write 1024 staged coordinates to HBM, interleaving the planes' storage rows,
// and shift overflow down.
void NonzeroCommon::Flush(Scalar q) {
  const Scalar flushes = kiln::LoadScalar(flush_count_);
  const Scalar slot = flushes & (kSlots - 1);
  const kiln::AsyncCopyDescriptor copy = FlushCopy(slot, q);
  kiln::If(flushes >= kSlots, [&copy] { copy.Wait(); });
  Ref packed = packed_(slot);
  const int64_t num_planes = static_cast<int64_t>(planes_.size());
  for (int r = 0; r < kSublanes; ++r) {
    for (int64_t p = 0; p < num_planes; ++p) {
      kiln::Store(packed(kiln::Slice(num_planes * r + p, 1)),
                  kiln::Load(planes_[p](kiln::Slice(r, 1))));
    }
  }
  copy.Start();
  kiln::Store(flush_count_, flushes + 1);
  for (Ref plane : planes_) {
    kiln::Store(plane(kiln::Slice(0, kSublanes)),
                kiln::Load(plane(kiln::Slice(kSublanes, kSublanes))));
    kiln::Store(plane(kiln::Slice(kSublanes, kSublanes)),
                kiln::Zeros(kiln::i32(kSublanes, kLanes)));
  }
}

// Drain in-flight flushes and write any remaining partial stage rows.
void NonzeroCommon::FinishOutput() {
  const Scalar flushes = kiln::LoadScalar(flush_count_);
  for (int s = 0; s < kSlots; ++s) {
    kiln::If(flushes > s, [this, s] { FlushCopy(s, kiln::I32(0)).Wait(); });
  }
  const Scalar count = kiln::LoadScalar(fill_);
  const Scalar q = (count >> kTileShift) << kSublaneShift;
  const Scalar rows = ((count & (kTile - 1)) + (kLanes - 1)) >> kLaneShift;
  const int64_t num_planes = static_cast<int64_t>(planes_.size());
  kiln::For(rows, [this, num_planes, q](Scalar r) {
    for (int64_t p = 0; p < num_planes; ++p) {
      kiln::SyncCopy(planes_[p](kiln::Slice(r, 1)),
                     OutputRows(num_planes * (q + r) + p, 1));
    }
  });
}

namespace {

// Define the Mosaic memrefs: input, private destination and aliased result.
// The HLO wrapper provides runtime dimensions and the result/destination
// alias.
absl::StatusOr<kiln::Kernel> BuildNonzeroIndices(const int64_t rank,
                                                 NonzeroKernelBody body) {
  const kiln::Dimensions dynamic_input(rank, kiln::kDynamic);
  const kiln::Dimensions dynamic_2d = {kiln::kDynamic, kiln::kDynamic};
  return kiln::CreateKernel(kiln::GridSpec{.name = "main"},
                            kiln::InSpec::Hbm(kiln::f32(dynamic_input)),
                            kiln::InSpec::Hbm(kiln::i32(dynamic_2d)),
                            kiln::OutSpec::Hbm(kiln::i32(dynamic_2d)),
                            [body](Ref input, Ref /*destination*/, Ref output) {
                              // The result aliases the private destination;
                              // only the result is written.
                              body(input, output);
                            });
}

// Returns an unbounded-dynamic shape of `rank` with the tiled layout `layout`.
xla::Shape MakeUnbounded(const xla::PrimitiveType type, const int64_t rank,
                         const xla::Layout& layout) {
  xla::Shape shape = xla::ShapeUtil::MakeShape(
      type, Dimensions(rank, xla::Shape::kUnboundedSize),
      std::vector<bool>(rank, true));
  *shape.mutable_layout() = layout;
  return shape;
}

// Returns the S32[R,128] view of the S64[K,rank] destination that the Mosaic
// kernel writes.
xla::Shape OutputView() {
  return MakeUnbounded(xla::S32, 2, xla::Layout({1, 0}, {xla::Tile({1, 128})}));
}

// Maps the logical S64[K,kRank] destination onto the S32[R,128] view the
// Mosaic kernel was compiled against; see the buffer contract in the rank's
// header. Both occupy 1024 * kRank * ceil(K/128) bytes.
template <int kRank>
absl::StatusOr<xla::Shape> DestinationPhysicalShape(const Shape& logical,
                                                    const xla::Layout& layout) {
  const int64_t rows = 2 * kRank * ((logical.dimensions()[0] + 127) / 128);
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::S32, {rows, 128});
  *shape.mutable_layout() = layout;
  return shape;
}

// Bitcasts the kernel's only result, the S32[R,128] destination, back to the
// logical S64[K,kRank] output; see the buffer contract in the rank's header.
template <int kRank>
absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
BitcastToLogicalIndices(std::vector<std::unique_ptr<xla::PjRtBuffer>> results,
                        const absl::Span<const Shape> logical_output_shapes) {
  const xla::Layout layout = NonzeroOutputLayout(kRank);
  const Dimensions dims = logical_output_shapes[0].dimensions();
  TT_ASSIGN_OR_RETURN(
      results[0],
      AdaptXlaError(
          results[0]->Bitcast(xla::S64, dims, &layout),
          /*context=*/"failed to bitcast the nonzero indices to S64"));
  return results;
}

// Returns the parameters for the nonzero indices kernel for rank-`rank` inputs
// in layout `input_layout`.
std::vector<xla::Shape> NonzeroParameters(const int64_t rank,
                                          const xla::Layout& input_layout) {
  const xla::Shape input = MakeUnbounded(xla::F32, rank, input_layout);
  const auto metadata = [](int64_t rank) {
    xla::Shape dims = xla::ShapeUtil::MakeShape(xla::S32, {rank});
    *dims.mutable_layout() = xla::Layout({0}, {xla::Tile({128})});
    return dims;
  };
  return {input, OutputView(), metadata(rank), metadata(2)};
}

// `MakeNonzeroDynamicMetadata` for rank `kRank`, which selects the rank's
// destination view and result bitcast.
template <int kRank>
UnboundedDynamicExecutionMetadata MakeNonzeroDynamicMetadataForRank(
    const xla::Layout& input_layout,
    UnboundedDynamicExecutionMetadata::CompatibilityFunc is_compatible) {
  // One logical input, the F32 payload. The S64[K,kRank] destination is a
  // donated output: `Execute` allocates it directly in the S32[R,128] view the
  // kernel was compiled against, so callers never see the physical view.
  UnboundedDynamicExecutionMetadata dynamic_meta{
      .logical_input_to_pjrt_param = {0},
      .donated_output_params =
          {
              UnboundedDynamicExecutionMetadata::DonatedOutputParam{
                  .logical_output_idx = 0,
                  .pjrt_param_idx = 1,
                  .physical_shape_fn = DestinationPhysicalShape<kRank>,
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
      .pjrt_param_shapes = NonzeroParameters(kRank, input_layout),
      .output_transform = BitcastToLogicalIndices<kRank>,
      .is_compatible = is_compatible,
  };
  return dynamic_meta;
}

}  // namespace

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
CompileNonzeroIndices(const int64_t rank, const xla::Layout& input_layout,
                      NonzeroKernelBody body, xla::PjRtClient& client,
                      UniqueCompileOptions compile_options) {
  const std::vector<xla::Shape> parameters =
      NonzeroParameters(rank, input_layout);
  TT_ASSIGN_OR_RETURN(kiln::Kernel kernel, BuildNonzeroIndices(rank, body));
  TT_ASSIGN_OR_RETURN(const std::string serialized_kernel,
                      std::move(kernel).Serialize());
  TT_ASSIGN_OR_RETURN(
      const std::unique_ptr<xla::HloModule> hlo,
      BuildMosaicCustomCall(serialized_kernel, parameters, OutputView(),
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

xla::Layout NonzeroOutputLayout(const int64_t rank) {
  return xla::Layout({0, 1}, {xla::Tile({rank, 128})});
}

UnboundedDynamicExecutionMetadata MakeNonzeroDynamicMetadata(
    const int64_t rank, const xla::Layout& input_layout,
    UnboundedDynamicExecutionMetadata::CompatibilityFunc is_compatible) {
  return rank == 1
             ? MakeNonzeroDynamicMetadataForRank<1>(input_layout, is_compatible)
             : MakeNonzeroDynamicMetadataForRank<2>(input_layout,
                                                    is_compatible);
}

ShapelessKey NonzeroCacheKey(const int64_t rank) {
  // Only ranks and dtypes reach the shapeless key, so these extents are
  // placeholders.
  const Dimensions input_dims(rank, -1);
  const Dimensions out_dims = {-1, rank};

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

}  // namespace torch_tpu::internal
