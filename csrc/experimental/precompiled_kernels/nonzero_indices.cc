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
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/layout_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/unbounded_dynamic_kernel_registry.h"
#include "csrc/experimental/precompiled_kernels/kernel_utils.h"
#include "csrc/experimental/precompiled_kernels/nonzero_indices_common.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"

namespace torch_tpu {
namespace {

using internal::kLanes;
using internal::kSublanes;
using internal::kSublaneShift;
using internal::kTile;
using internal::kWindow;
using internal::kWindowShift;
using internal::MakeWindow;
using internal::Window;
using kiln::Ref;
using kiln::Scalar;
using kiln::Tile;

constexpr int64_t kRank = 2;  // Input rank

// The nonzero compaction core (see `NonzeroCommon`) fed an F32[M,N] input
// in logical row-major order, writing (row, column) pairs into S64[K,2].
//
// Input arrives by DMA eight physical rows at a time, but compaction always
// visits logical rows in order, so the output never needs reordering. A row
// too wide for the kWindow scratch buffer is processed one window at a time.
class Nonzero2DKernel : public internal::NonzeroCommon {
 public:
  Nonzero2DKernel(Ref x, Ref output)
      : NonzeroCommon(output, kRank, x.Dim(0), x.Dim(1)), x_(x) {}

 private:
  // Scan rows that exceed scratch window by window; otherwise scan row
  // blocks.
  void ProcessInput() override {
    kiln::If(
        n() > kWindow, [this] { ProcessWideRows(); },
        [this] {
          const Window w = MakeWindow(kiln::I32(0), n());
          const Scalar row_blocks = (m() + (kSublanes - 1)) >> kSublaneShift;
          kiln::EmitPipeline(
              0, row_blocks,
              {InputBlocks([this, w](Scalar rb) { return BlockRows(w, rb); })},
              /*outputs=*/{}, [this, &w](const kiln::PipelineStep& s) {
                ProcessBlock(w, s.inputs[0], s.step << kSublaneShift);
              });
        });
  }

  // Above kWindow columns, scan each row window by window to preserve order.
  void ProcessWideRows() {
    // Count windows so `base + kWindow` cannot overflow S32 near INT32_MAX.
    const Scalar windows = ((n() - 1) >> kWindowShift) + 1;
    kiln::For(m(), [this, windows](Scalar row) {
      const Scalar rb = row >> kSublaneShift;
      kiln::EmitPipeline(
          0, windows, {InputBlocks([this, rb](Scalar window) {
            return BlockRows(WideWindow(window), rb);
          })},
          /*outputs=*/{}, [this, row](const kiln::PipelineStep& s) {
            const Window w = WideWindow(s.step);
            const Ref block = s.inputs[0];
            kiln::For(w.tiles, [this, &w, &block, row](Scalar tile_idx) {
              CompactTile(w, block, tile_idx, row & (kSublanes - 1), row);
            });
          });
    });
  }

  // Input stream whose step `s` holds `region(s)` of the input, prefetched
  // while step `s - 1` is compacted.
  kiln::PipelineOperand InputBlocks(
      absl::AnyInvocable<kiln::BlockRegion(Scalar) const> region) {
    return {.hbm = x_,
            .block_shape = {kSublanes, kWindow},
            .region = std::move(region)};
  }

  // Eight-row block `rb` of the input, over the columns of `w`.
  kiln::BlockRegion BlockRows(const Window& w, Scalar rb) const {
    return {.offsets = {rb * kSublanes, w.base}, .sizes = {kSublanes, w.n_pad}};
  }

  // Consume rows in logical order, independent of physical input tiling.
  void ProcessBlock(const Window& w, Ref block, Scalar r0) {
    kiln::For(kSublanes, [this, &w, &block, r0](Scalar r) {
      kiln::For(w.tiles, [this, &w, &block, r0, r](Scalar tile_idx) {
        CompactTile(w, block, tile_idx, r, r0 + r);
      });
    });
  }

  // Compact one 1024-column segment of row `row_in_block` of an input block.
  void CompactTile(const Window& w, Ref block, Scalar tile_idx,
                   Scalar row_in_block, Scalar row) {
    const Scalar offset = tile_idx * kTile;
    const Tile values = kiln::Load(block(kiln::Slice(row_in_block, 1),
                                         kiln::Slice(offset, kTile)))
                            .Reshape({kSublanes, kLanes});
    AppendTile(values, row, w.base + offset);
  }

  Ref x_;  // HBM F32[M,N] input
};

// The on-device input layout of the buffer contract; see the header.
xla::Layout InputLayout() { return xla::Layout({1, 0}, {xla::Tile({8, 128})}); }

// Builds and compiles the Mosaic kernel. See `GetNonzeroUnboundedDynamicKernel`
// for its contract.
absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
BuildNonzeroIndicesExecutable(xla::PjRtClient& client,
                              UniqueCompileOptions compile_options) {
  return internal::CompileNonzeroIndices(
      kRank, InputLayout(),
      [](Ref input, Ref output) { Nonzero2DKernel(input, output).Build(); },
      client, std::move(compile_options));
}

// Returns whether the kernel serves a nonzero call with these logical shapes:
// one F32[M,N] input and one S64[K,2] output, in the layouts of the
// buffer contract.
bool IsCompatible(absl::Span<const Shape> input_shapes,
                  absl::Span<const Shape> output_shapes) {
  // Resolved from the device defaults and any custom layouts on the shapes.
  const absl::StatusOr<std::vector<xla::Layout>> input_layouts =
      ResolveXlaLayouts(input_shapes);
  const absl::StatusOr<std::vector<xla::Layout>> output_layouts =
      ResolveXlaLayouts(output_shapes);

  return input_shapes.size() == 1 &&
         input_shapes[0].dimensions().size() == kRank &&
         input_shapes[0].dtype() == mlir::ElementType::F32 &&
         output_shapes.size() == 1 &&
         output_shapes[0].dimensions().size() == 2 &&
         output_shapes[0].dimensions()[1] == kRank &&
         output_shapes[0].dtype() == mlir::ElementType::I64 &&
         input_layouts.ok() && output_layouts.ok() &&
         *input_layouts == std::vector<xla::Layout>{InputLayout()} &&
         *output_layouts ==
             std::vector<xla::Layout>{internal::NonzeroOutputLayout(kRank)};
}

}  // namespace

UnboundedDynamicKernel GetNonzeroUnboundedDynamicKernel() {
  return UnboundedDynamicKernel{
      .build = BuildNonzeroIndicesExecutable,
      .metadata = internal::MakeNonzeroDynamicMetadata(kRank, InputLayout(),
                                                       IsCompatible),
      .key = MakeUnboundedDynamicKernelCacheKey(
          internal::NonzeroCacheKey(kRank),
          static_cast<int>(internal::NonzeroKernelId::kNonzeroKGreaterThan2)),
  };
}

}  // namespace torch_tpu
