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

#include "csrc/experimental/precompiled_kernels/sort/sort_kernels.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
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
#include "csrc/experimental/precompiled_kernels/sort/bitonic.h"
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

// Identifies a precompiled sort kernel. The value is the `id_key` of the
// kernel's `CompilationCacheKey`, which is persisted in the names of tier-2
// cache entries, so values must never be reused or renumbered.
enum class SortKernelId : int {
  kSort1024Unstable = 1,
  kSort1024Stable = 2,
};

constexpr int64_t kRows = 8;
constexpr int64_t kLanes = 128;
constexpr int64_t kTile = kRows * kLanes;

// The default TPU layout of an S32[1024] or S64[1024] array.
xla::Layout VectorLayout() { return xla::Layout({0}, {xla::Tile({kTile})}); }

// An (8, 128) tile is stored row by row, so its bytes are those of the
// S32[1024] vector. An S64[1024] vector is stored as its 1024 low words
// followed by its 1024 high words, i.e. as an S32[16, 128] array.
xla::Shape TileShape(int64_t rows) {
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::S32, {rows, kLanes});
  *shape.mutable_layout() = xla::Layout({1, 0}, {xla::Tile({kRows, kLanes})});
  return shape;
}

xla::Shape InputShape() {
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::S32, {kTile});
  *shape.mutable_layout() = VectorLayout();
  return shape;
}

// Sorts the input keys and writes them to `values`, and their original
// positions to the low words of `indices`, zeroing the high words.
absl::StatusOr<kiln::Kernel> BuildSortKernel() {
  return kiln::CreateKernel(
      kiln::GridSpec{.name = "main"}, kiln::InSpec::Hbm(kiln::i32(kTile)),
      kiln::OutSpec::Hbm(kiln::i32(kRows, kLanes)),
      kiln::OutSpec::Hbm(kiln::i32(2 * kRows, kLanes)),
      [](kiln::HbmRef input, kiln::HbmRef values, kiln::HbmRef indices) {
        const kiln::VmemRef input_vmem = kiln::AllocaVmem(kiln::i32(kTile));
        kiln::SyncCopy(input, input_vmem);
        kiln::Tile keys = kiln::Load(input_vmem).Reshape({kRows, kLanes});
        kiln::Tile positions =
            kiln::Iota(kiln::i32(kRows, kLanes), /*axis=*/0) * kLanes +
            kiln::Iota(kiln::i32(kRows, kLanes), /*axis=*/1);
        BitonicSortTile(keys, positions);

        const kiln::VmemRef values_vmem =
            kiln::AllocaVmem(kiln::i32(kRows, kLanes));
        kiln::Store(values_vmem, keys);
        kiln::SyncCopy(values_vmem, values);
        const kiln::VmemRef indices_vmem =
            kiln::AllocaVmem(kiln::i32(2 * kRows, kLanes));
        kiln::Store(indices_vmem(kiln::Slice(0, kRows), kiln::Slice(0, kLanes)),
                    positions);
        kiln::Store(
            indices_vmem(kiln::Slice(kRows, kRows), kiln::Slice(0, kLanes)),
            kiln::Zeros(kiln::i32(kRows, kLanes)));
        kiln::SyncCopy(indices_vmem, indices);
      });
}

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> BuildSortExecutable(
    xla::PjRtClient& client, UniqueCompileOptions compile_options) {
  const std::vector<xla::Shape> parameters = {InputShape()};
  TT_ASSIGN_OR_RETURN(kiln::Kernel kernel, BuildSortKernel());
  TT_ASSIGN_OR_RETURN(const std::string serialized_kernel,
                      std::move(kernel).Serialize());
  TT_ASSIGN_OR_RETURN(
      const std::unique_ptr<xla::HloModule> hlo,
      BuildMosaicCustomCall(serialized_kernel, parameters,
                            xla::ShapeUtil::MakeTupleShape(
                                {TileShape(kRows), TileShape(2 * kRows)}),
                            /*operand_to_metadata=*/{-1}));
  xla::ExecutableBuildOptions& build =
      compile_options->executable_build_options;
  build.set_run_backend_only(true);
  build.set_result_layout(hlo->entry_computation_layout().result_shape());
  compile_options->argument_layouts = parameters;
  const xla::XlaComputation computation(hlo->ToProto());
  return AdaptXlaError(
      client.CompileAndLoad(computation, std::move(*compile_options)),
      /*context=*/"failed to compile the sort kernel");
}

// Views the kernel's S32 tiles as the logical S32[1024] values and S64[1024]
// indices.
absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>> ToLogicalOutputs(
    std::vector<std::unique_ptr<xla::PjRtBuffer>> results,
    absl::Span<const Shape> /*logical_output_shapes*/) {
  const xla::Layout layout = VectorLayout();
  TT_ASSIGN_OR_RETURN(
      results[0], AdaptXlaError(results[0]->Bitcast(xla::S32, {kTile}, &layout),
                                /*context=*/"failed to bitcast sorted values"));
  TT_ASSIGN_OR_RETURN(
      results[1], AdaptXlaError(results[1]->Bitcast(xla::S64, {kTile}, &layout),
                                /*context=*/"failed to bitcast sort indices"));
  return results;
}

bool IsCompatible(absl::Span<const Shape> input_shapes,
                  absl::Span<const Shape> output_shapes) {
  // The shapeless key already fixes one rank-1 S32 input, and S32 and S64
  // rank-1 outputs.
  const absl::StatusOr<std::vector<xla::Layout>> input_layouts =
      ResolveXlaLayouts(input_shapes);
  const absl::StatusOr<std::vector<xla::Layout>> output_layouts =
      ResolveXlaLayouts(output_shapes);
  return input_shapes[0].dimensions()[0] == kTile && input_layouts.ok() &&
         output_layouts.ok() &&
         *input_layouts == std::vector<xla::Layout>{VectorLayout()} &&
         *output_layouts ==
             std::vector<xla::Layout>{VectorLayout(), VectorLayout()};
}

UnboundedDynamicExecutionMetadata MakeSortMetadata() {
  return UnboundedDynamicExecutionMetadata{
      .logical_input_to_pjrt_param = {0},
      .pjrt_param_shapes = {InputShape()},
      .output_transform = ToLogicalOutputs,
      .is_compatible = IsCompatible,
  };
}

// Returns the ShapelessKey of the graph that `torch.sort` traces for a 1-D
// int32 input, ascending along dim -1.
ShapelessKey SortCacheKey(std::optional<bool> stable_opt) {
  absl::StatusOr<OpParamCacheKeys> param_keys =
      *OpParamCacheKeysBuilder()
           .SetParam("stable_opt", stable_opt)
           .SetParam("dim", int64_t{-1})
           .SetParam("descending", false);
  ABSL_CHECK_OK(param_keys);  // CRASH_OK=static-init registration bug
  // Only ranks and dtypes reach the shapeless key, so the extents are
  // placeholders.
  const Dimensions dims = {-1};
  GraphSignature graph;
  graph.SetCorePinningMode(CorePinningMode::kUnpinned);
  const int in = graph.AddInput(dims, mlir::ElementType::I32);
  int values = -1;
  int indices = -1;
  graph.AddOp(
      OpName::kSortValuesStable, *std::move(param_keys),
      /*donated_inputs=*/{},
      [in, &dims, &values, &indices](GraphSignature::OpSignatureBuilder& op) {
        op.AddInput(in);
        values = op.AddOutput(dims, mlir::ElementType::I32);
        indices = op.AddOutput(dims, mlir::ElementType::I64);
      });
  graph.AddGraphOutput(values);
  graph.AddGraphOutput(indices);
  return graph.GetKey().shapeless_key();
}

UnboundedDynamicKernel GetSortKernel(std::optional<bool> stable_opt,
                                     SortKernelId id) {
  return UnboundedDynamicKernel{
      .build = BuildSortExecutable,
      .metadata = MakeSortMetadata(),
      .key = MakeUnboundedDynamicKernelCacheKey(SortCacheKey(stable_opt),
                                                static_cast<int>(id)),
      .enabled_by_default = false,
  };
}

}  // namespace

std::vector<UnboundedDynamicKernel> GetSortUnboundedDynamicKernels() {
  return {GetSortKernel(false, SortKernelId::kSort1024Unstable),
          GetSortKernel(true, SortKernelId::kSort1024Stable)};
}

}  // namespace torch_tpu
