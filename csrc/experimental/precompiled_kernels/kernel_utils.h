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

#ifndef TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_KERNEL_UTILS_H_
#define TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_KERNEL_UTILS_H_

#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/shape.h"
#include "csrc/experimental/precompiled_kernels/mosaic_custom_call.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

// Returns the compilation cache key that a precompiled unbounded dynamic kernel
// serving graphs with extentless key `key` is registered under: `key` with
// empty dimensions and compile options keys, and `id_key`, which tells apart
// kernels that share an extentless key.
CompilationCacheKey MakeUnboundedDynamicKernelCacheKey(ExtentlessKey key,
                                                       int id_key);

// Compiles a Mosaic kernel, wrapped by `BuildMosaicCustomCall` with the other
// arguments, for exactly the given input and output shapes and layouts: only
// the backend runs, so no HLO pass changes them.
absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> CompileMosaicKernel(
    xla::PjRtClient& client, UniqueCompileOptions compile_options,
    kiln::Kernel kernel, absl::Span<const xla::Shape> inputs,
    const xla::Shape& output, absl::Span<const int64_t> operand_to_metadata,
    int aliased_parameter = -1, OperandsTransform operands_transform = nullptr);

// Returns an unbounded dynamic shape of rank `rank` with a row-major layout
// tiled by `tiles`.
xla::Shape MakeUnboundedShape(xla::PrimitiveType type, int64_t rank,
                              absl::Span<const xla::Tile> tiles);

// Whether `shapes` resolve to exactly the on-device `layouts`, e.g. those a
// kernel was compiled for.
bool HasDeviceLayouts(absl::Span<const Shape> shapes,
                      absl::Span<const xla::Layout> layouts);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_KERNEL_UTILS_H_
