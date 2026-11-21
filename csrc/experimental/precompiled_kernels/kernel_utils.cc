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

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/compile_options_key.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/layout_utils.h"
#include "csrc/common/shape.h"
#include "csrc/experimental/precompiled_kernels/mosaic_custom_call.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/shape.h"

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
    int aliased_parameter) {
  TT_ASSIGN_OR_RETURN(const std::string serialized_kernel,
                      std::move(kernel).Serialize());
  TT_ASSIGN_OR_RETURN(
      const std::unique_ptr<xla::HloModule> hlo,
      BuildMosaicCustomCall(serialized_kernel, inputs, output,
                            operand_to_metadata, aliased_parameter));
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

bool HasDeviceLayouts(absl::Span<const Shape> shapes,
                      absl::Span<const xla::Layout> layouts) {
  const absl::StatusOr<std::vector<xla::Layout>> resolved =
      ResolveXlaLayouts(shapes);
  return resolved.ok() && absl::Span<const xla::Layout>(*resolved) == layouts;
}

}  // namespace torch_tpu
