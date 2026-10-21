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
#include "csrc/common/compile_options_key.h"
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

namespace torch_tpu {
namespace {

constexpr int64_t kSublanes = 8;
constexpr int64_t kLanes = 128;

// Scan rows in order. For each 128-column block, load the containing 8x128
// input tile, extract the current row, and inspect its valid lanes in order.
// This baseline reloads the input tile for each row.
//
// Each selected element appends (row,column) to two 128-element staging
// vectors at slot count % 128. Increment count and flush both vectors whenever
// the stage fills. Staging persists across input tiles and rows. After the
// scan, flush a partial stage into the output's normal tile padding.
//
// Conceptually:
//   count = 0
//   for row in [0,M):
//     for col in [0,N):
//       if IsNonzero(input[row,col]): output[count++] = (row,col)
// This preserves lexicographic ordering without a sort or prefix sum.
//
// Written in Kiln, so a `kiln::Scalar` or `kiln::Tile` is a device value and
// `kiln::For` / `kiln::If` emit device control flow; the C++ runs once at
// build time.

// Synchronous DMA: source and destination are safe to reuse after this
// returns.
void Copy(kiln::Ref src, kiln::Ref dst, kiln::Semaphore sem) {
  kiln::AsyncCopy(src, dst, sem).Wait();
}

// Copy one 128-word VMEM tile to a storage-row index in the native output.
void WriteTile(kiln::Ref tile, kiln::Scalar tile_index, kiln::Ref output,
               kiln::Semaphore sem) {
  Copy(tile, output(kiln::Slice(tile_index, 1), kiln::Slice(0, kLanes)), sem);
}

// Each 128-pair block occupies two low-word tiles: rows, then columns.
// first is a multiple of 128, so its tile index is 2 * first / 128.
void Flush(kiln::Scalar first, kiln::Ref stage_rows, kiln::Ref stage_cols,
           kiln::Ref output, kiln::Semaphore sem) {
  kiln::Scalar tile = first >> 6;
  WriteTile(stage_rows, tile, output, sem);
  WriteTile(stage_cols, tile + 1, output, sem);
}

// Flush a partial final stage; an exactly full stage was already written.
void FlushTail(kiln::Ref count, kiln::Ref stage_rows, kiln::Ref stage_cols,
               kiln::Ref output, kiln::Semaphore sem) {
  kiln::Scalar n = kiln::Scalar(count);
  kiln::If((n & 127) != 0,
           [&] { Flush(n & -128, stage_rows, stage_cols, output, sem); });
}

// Place one pair at count % 128 and advance the device counter.
// Flush only when all 128 staged pairs have been filled.
void Append(kiln::Scalar row, kiln::Scalar col, kiln::Ref count,
            kiln::Tile position, kiln::Ref stage_rows, kiln::Ref stage_cols,
            kiln::Ref output, kiln::Semaphore sem) {
  kiln::Scalar n = kiln::Scalar(count);
  kiln::Tile mask = position == (n & 127);
  stage_rows.Store(kiln::Splat(row, stage_rows.dims()), mask);
  stage_cols.Store(kiln::Splat(col, stage_cols.dims()), mask);
  kiln::Scalar next = n + 1;
  count = next;
  kiln::If((next & 127) == 0,
           [&] { Flush(n & -128, stage_rows, stage_cols, output, sem); });
}

// Gather a runtime-selected lane, then extract its replicated value as a
// scalar.
kiln::Scalar ExtractLane(kiln::Tile values, kiln::Scalar lane) {
  kiln::Tile gathered = kiln::Gather(values, kiln::Splat(lane, values.dims()),
                                     /*axis=*/1);
  return kiln::Extract(gathered, {0, 0});
}

// Classify one logical lane using the predicate specified in the header.
kiln::Scalar IsNonzero(kiln::Tile values, kiln::Scalar lane) {
  // TPU scalar arithmetic is integer-only. Compare float magnitude bits:
  // any set bit is nonzero, which excludes only +0.0 and -0.0.
  kiln::Scalar bits = Bitcast(ExtractLane(values, lane), kiln::i32);
  return (bits & 0x7fffffff) > 0;
}

// DMA requires whole physical tiles. Select the requested logical row after
// loading its containing 8x128 tile; the scalar scan masks the column tail.
kiln::Tile LoadRow(kiln::Scalar row, kiln::Scalar col, kiln::Ref input,
                   kiln::Ref tile, kiln::Semaphore sem) {
  kiln::Ref src =
      input(kiln::Slice(kiln::MultipleOf(row & -8, kSublanes), kSublanes),
            kiln::Slice(kiln::MultipleOf(col, kLanes), kLanes));
  Copy(src, tile, sem);
  // Rotate the wanted row into sublane 0, then keep only that sublane.
  kiln::Tile values =
      kiln::Rotate(kiln::Tile(tile), kSublanes - (row & 7), /*axis=*/0);
  return values.Slice({0, 0}, {1, kLanes});
}

// Define the Mosaic memrefs: input, private destination and aliased result.
// The HLO wrapper provides runtime dimensions and the result/destination
// alias.
absl::StatusOr<kiln::Kernel> BuildNonzeroIndices() {
  const kiln::Dimensions dynamic_2d = {kiln::kDynamic, kiln::kDynamic};
  return kiln::CreateKernel(
      kiln::GridSpec{.name = "main"}, kiln::InSpec::Hbm(kiln::f32(dynamic_2d)),
      kiln::InSpec::Hbm(kiln::i32(dynamic_2d)),
      kiln::OutSpec::Hbm(kiln::i32(dynamic_2d)),
      [](kiln::Ref input, kiln::Ref /*destination*/, kiln::Ref output) {
        // The result aliases the private destination; only the result is
        // written. Allocate one input tile, two coordinate stages and a
        // scalar count.
        kiln::Ref tile = kiln::AllocaVmem(kiln::f32(kSublanes, kLanes));
        kiln::Ref stage_rows = kiln::AllocaVmem(kiln::i32(1, kLanes));
        kiln::Ref stage_cols = kiln::AllocaVmem(kiln::i32(1, kLanes));
        kiln::Ref count = kiln::AllocaSmem(kiln::i32());
        kiln::Semaphore sem = kiln::AllocaDmaSemaphore();
        kiln::Tile position = kiln::Iota(kiln::i32(1, kLanes), /*axis=*/1);
        stage_rows = kiln::Zeros(kiln::i32(1, kLanes));
        stage_cols = kiln::Zeros(kiln::i32(1, kLanes));
        count = kiln::I32(0);
        // Coordinates fit S32: zero the native int64 high words once; the scan
        // and flushes only write low words.
        kiln::Scalar num_tiles = output.Dim(0);
        kiln::For(num_tiles >> 1, num_tiles, [&](kiln::Scalar tile_index) {
          WriteTile(stage_rows, tile_index, output, sem);
        });

        // Emit the complete scalar scan, then flush the remaining output
        // coordinates.
        kiln::Scalar num_rows = input.Dim(0), num_cols = input.Dim(1);
        kiln::For(0, num_rows, [&](kiln::Scalar row) {
          kiln::For(0, num_cols, kLanes, [&](kiln::Scalar col) {
            kiln::Tile values = LoadRow(row, col, input, tile, sem);
            kiln::Scalar limit = Min(num_cols - col, kLanes);
            kiln::For(0, limit, [&](kiln::Scalar lane) {
              kiln::If(IsNonzero(values, lane), [&] {
                Append(row, col + lane, count, position, stage_rows, stage_cols,
                       output, sem);
              });
            });
          });
        });
        FlushTail(count, stage_rows, stage_cols, output, sem);
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
      AdaptXlaError(
          results[0]->Bitcast(xla::S64, dims, &layout),
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

}  // namespace

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
             absl::Span<const xla::Layout> input_layouts,
             absl::Span<const Shape> output_shapes,
             absl::Span<const xla::Layout> output_layouts) {
            return input_shapes.size() == 1 && input_layouts.size() == 1 &&
                   input_shapes[0].dimensions().size() == 2 &&
                   input_shapes[0].dtype() == mlir::ElementType::F32 &&
                   input_layouts[0] ==
                       xla::Layout({1, 0}, {xla::Tile({8, 128})}) &&
                   output_shapes.size() == 1 && output_layouts.size() == 1 &&
                   output_shapes[0].dimensions().size() == 2 &&
                   output_shapes[0].dimensions()[0] > 2 &&
                   output_shapes[0].dimensions()[1] == 2 &&
                   output_shapes[0].dtype() == mlir::ElementType::I64 &&
                   output_layouts[0] ==
                       xla::Layout({0, 1}, {xla::Tile({2, 128})});
          },
  };
  return dynamic_meta;
}

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
  return AdaptXlaError(
      client.CompileAndLoad(computation, std::move(*compile_options)),
      /*context=*/"failed to compile the nonzero indices kernel");
}

CompilationCacheKey NonZeroCacheKey() {
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
              [&](GraphSignature::OpSignatureBuilder& op) {
                op.AddInput(in);
                out = op.AddOutput(out_dims, mlir::ElementType::I64);
              });
  graph.AddGraphOutput(out);
  return CompilationCacheKey(graph.GetKey(), CompileOptionsKey({}),
                             "nonzero_2d");
}

}  // namespace torch_tpu
