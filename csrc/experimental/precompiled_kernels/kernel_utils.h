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
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/shape.h"
#include "csrc/experimental/precompiled_kernels/mosaic_custom_call.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

// TPU vector register geometry (8 sublanes x 128 lanes = 1024 32-bit elements)
// and corresponding bit shifts for power-of-two divisions/multiplications.
inline constexpr int64_t kSublanes = 8;
inline constexpr int64_t kSublaneShift = 3;
inline constexpr int64_t kLanes = 128;
inline constexpr int64_t kLaneShift = 7;
inline constexpr int64_t kTile = kSublanes * kLanes;
inline constexpr int64_t kTileShift = kSublaneShift + kLaneShift;

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

// Returns XLA's default 1-D tile minor dimension for a 32-bit or 64-bit tensor
// of length `size`: `size` rounded up to a power of two in [128, 1024].
[[nodiscard]] int64_t Default1DTileMinor(int64_t size);

// Returns a 1-D layout `{0:T(tile_minor)}`.
[[nodiscard]] xla::Layout Default1DLayout(int64_t tile_minor);

// Returns an unbounded dynamic shape of rank `rank` with a row-major layout
// tiled by `tiles`.
xla::Shape MakeUnboundedShape(xla::PrimitiveType type, int64_t rank,
                              absl::Span<const xla::Tile> tiles);

// Returns the S32[?,?]{1,0:T(1,128)} row-view shape used by kernels that write
// 128-element storage rows into a donated output.
[[nodiscard]] xla::Shape MakeS32RowViewShape();

// Returns the static S32[rank]{0:T(128)} shape for a disjoint runtime dimension
// metadata parameter.
[[nodiscard]] xla::Shape MakeRuntimeDimsShape(int64_t rank);

// Returns execution metadata for a kernel with one logical input (param 0),
// one donated dynamic output (param 1), and their runtime dimension metadata
// parameters (params 2 and 3).
[[nodiscard]] UnboundedDynamicExecutionMetadata MakeUnaryDonatedOutputMetadata(
    std::vector<xla::Shape> pjrt_param_shapes,
    UnboundedDynamicExecutionMetadata::PhysicalOutputShapeFunc
        physical_shape_fn,
    UnboundedDynamicExecutionMetadata::OutputTransform output_transform,
    UnboundedDynamicExecutionMetadata::CompatibilityFunc is_compatible);

// Returns the ExtentlessKey of an unpinned single-op graph with empty op param
// cache keys, one input of (`in_rank`, `in_dtype`), and one output of
// (`out_rank`, `out_dtype`).
[[nodiscard]] ExtentlessKey UnaryOpExtentlessKey(OpName op_name,
                                                 int64_t in_rank,
                                                 mlir::ElementType in_dtype,
                                                 int64_t out_rank,
                                                 mlir::ElementType out_dtype);

// Whether `shapes` resolve to exactly the on-device `layouts`, e.g. those a
// kernel was compiled for.
bool HasDeviceLayouts(absl::Span<const Shape> shapes,
                      absl::Span<const xla::Layout> layouts);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EXPERIMENTAL_PRECOMPILED_KERNELS_KERNEL_UTILS_H_
