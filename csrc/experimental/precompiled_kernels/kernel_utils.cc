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

#include "csrc/experimental/precompiled_kernels/kernel_utils.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/compile_options_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/layout_utils.h"
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

CompilationCacheKey MakeUnboundedDynamicKernelCacheKey(ExtentlessKey key,
                                                       int id_key) {
  return CompilationCacheKey(GraphKey(key, DimensionsKey({})),
                             CompileOptionsKey({}), id_key);
}

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> CompileMosaicKernel(
    xla::PjRtClient& client, UniqueCompileOptions compile_options,
    kiln::Kernel kernel, absl::Span<const xla::Shape> inputs,
    const xla::Shape& output, absl::Span<const int64_t> operand_to_metadata,
    int aliased_parameter, OperandsTransform operands_transform) {
  TT_ASSIGN_OR_RETURN(const std::string serialized_kernel,
                      std::move(kernel).Serialize());
  TT_ASSIGN_OR_RETURN(
      const std::unique_ptr<xla::HloModule> hlo,
      BuildMosaicCustomCall(serialized_kernel, inputs, output,
                            operand_to_metadata, aliased_parameter,
                            std::move(operands_transform)));
  xla::ExecutableBuildOptions& build =
      compile_options->executable_build_options;
  build.set_run_backend_only(true);
  build.set_result_layout(hlo->entry_computation_layout().result_shape());
  compile_options->argument_layouts =
      std::vector<xla::Shape>(inputs.begin(), inputs.end());
  const xla::XlaComputation computation(hlo->ToProto());
  return WrapXlaError(
      client.CompileAndLoad(computation, std::move(*compile_options)),
      /*context=*/"failed to compile the Mosaic kernel");
}

xla::Layout Default1DLayout(const int64_t tile_minor) {
  return xla::Layout({0}, {xla::Tile({tile_minor})});
}

xla::Shape MakeUnboundedShape(const xla::PrimitiveType type, const int64_t rank,
                              absl::Span<const xla::Tile> tiles) {
  xla::Shape shape = xla::ShapeUtil::MakeShape(
      type, Dimensions(rank, xla::Shape::kUnboundedSize),
      std::vector<bool>(rank, true));
  for (const xla::Tile& tile : tiles) {
    *shape.mutable_layout()->add_tiles() = tile;
  }
  return shape;
}

xla::Shape MakeS32RowViewShape() {
  return MakeUnboundedShape(xla::S32, /*rank=*/2, {xla::Tile({1, kLanes})});
}

xla::Shape MakeRuntimeDimsShape(const int64_t rank) {
  xla::Shape dims = xla::ShapeUtil::MakeShape(xla::S32, {rank});
  *dims.mutable_layout() = Default1DLayout(kLanes);
  dims.mutable_layout()->set_memory_space(xla::Layout::kHostMemorySpace);
  return dims;
}

UnboundedDynamicExecutionMetadata MakeUnaryDonatedOutputMetadata(
    std::vector<xla::Shape> pjrt_param_shapes,
    UnboundedDynamicExecutionMetadata::PhysicalOutputShapeFunc
        physical_shape_fn,
    UnboundedDynamicExecutionMetadata::OutputTransform output_transform,
    UnboundedDynamicExecutionMetadata::CompatibilityFunc is_compatible) {
  ABSL_CHECK_EQ(  // CRASH_OK=static-init registration bug
      pjrt_param_shapes.size(), 4)
      << "A unary donated-output kernel must have 4 PJRT parameters. This is a "
         "TorchTPU bug.";
  return UnboundedDynamicExecutionMetadata{
      .logical_input_to_pjrt_param = {0},
      .donated_output_params =
          {
              UnboundedDynamicExecutionMetadata::DonatedOutputParam{
                  .logical_output_idx = 0,
                  .pjrt_param_idx = 1,
                  .physical_shape_fn = std::move(physical_shape_fn),
              },
          },
      .metadata_params =
          {
              UnboundedDynamicExecutionMetadata::MetadataParamBinding{
                  .buffer_pjrt_idx = 0,  // Input -> its dimensions
                  .shape_pjrt_idx = 2,
              },
              UnboundedDynamicExecutionMetadata::MetadataParamBinding{
                  .buffer_pjrt_idx = 1,  // Output view -> its dimensions
                  .shape_pjrt_idx = 3,
              },
          },
      .pjrt_param_shapes = std::move(pjrt_param_shapes),
      .output_transform = std::move(output_transform),
      .is_compatible = std::move(is_compatible),
  };
}

ExtentlessKey UnaryOpExtentlessKey(const OpName op_name, const int64_t in_rank,
                                   const mlir::ElementType in_dtype,
                                   const int64_t out_rank,
                                   const mlir::ElementType out_dtype) {
  // Only ranks and dtypes reach the extentless key, so extents are
  // placeholders.
  const Dimensions in_dims(in_rank, -1);
  const Dimensions out_dims(out_rank, -1);
  GraphSignature graph;
  graph.SetCorePinningMode(CorePinningMode::kUnpinned);
  const int in = graph.AddInput(in_dims, in_dtype);
  int out = -1;
  graph.AddOp(
      op_name, OpParamCacheKeys::Empty(), /*donated_inputs=*/{},
      [in, &out, &out_dims, out_dtype](GraphSignature::OpSignatureBuilder& op) {
        op.AddInput(in);
        out = op.AddOutput(out_dims, out_dtype);
      });
  graph.AddGraphOutput(out);
  return graph.GetKey().extentless_key();
}

bool HasDeviceLayouts(absl::Span<const Shape> shapes,
                      absl::Span<const xla::Layout> layouts) {
  const absl::StatusOr<std::vector<xla::Layout>> resolved =
      ResolveXlaLayouts(shapes);
  return resolved.ok() && absl::Span<const xla::Layout>(*resolved) == layouts;
}

}  // namespace torch_tpu
