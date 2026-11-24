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

#include "csrc/experimental/precompiled_kernels/matmul.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
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
#include "csrc/common/shape.h"
#include "csrc/common/unbounded_dynamic_kernel_registry.h"
#include "csrc/experimental/precompiled_kernels/kernel_utils.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "csrc/internal/mosaic/kiln_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

using ::mlir::ElementType;
using ::mlir::stablehlo::Precision;

// TODO(b/576608541): Follow-ups:
// - Plumb `vmem_limit_bytes` through `CompileMosaicKernel`, add a few tile-size
//   buckets routed by shape, and gate larger tiles by chip VMEM capacity.
// - Cache or avoid per-call H2D shape-metadata transfers in
//   `PrepareDynamicExecutionArguments` (dominant cost on small shapes).
// - Partition along K (split-K) when `num_blocks` is small or odd so megacore
//   chips don't leave a TensorCore idle.
// - Support column-major {0,1}, small-row T(1..4,128), and transposed RHS
//   layouts, plus float16, `out_dtype`, `addmm`, `bmm`, and batched `matmul`.
// - Trim VMEM/HBM traffic: store `Dot(...)` directly on `is_first_step` instead
//   of zero-store + reload + add; avoid the K-tail VMEM round-trip once Mosaic
//   supports non-32-bit masked stores (or in-register masking); skip unchanged
//   input DMAs once `EmitPipeline` supports `block_index` on inputs; evaluate
//   >2 pipeline slots and block swizzling.

// Identifies a precompiled matmul kernel. The value is the `id_key` of the
// kernel's `CompilationCacheKey`, which is persisted in the names of tier-2
// cache entries, so values must never be reused or renumbered.
enum class MatmulKernelId : int {
  kBlock512 = 1,
};

// Output blocks are kBlock x kBlock, and contraction steps kBlock deep.
constexpr int64_t kBlock = 512;
// HBM tiles are (8, 128) for both f32 and bf16. bf16 VMEM tiles have 16 rows,
// but a copy rounded up to them would run past the end of the HBM buffer.
constexpr int64_t kHbmRows = 8;
constexpr int64_t kLanes = 128;
// Mosaic splits a static parallel grid axis across the TensorCores of a
// megacore chip, and runs its programs one after the other elsewhere.
constexpr int64_t kNumPrograms = 2;

// Returns the MXU precision with the numerics of XLA's dot at `precision`.
kiln::ContractPrecision ToContractPrecision(const Precision precision) {
  switch (precision) {
    case Precision::DEFAULT:  // EXPLICIT_PRECISION_OK=mapping
      return kiln::ContractPrecision::kBF16;
    case Precision::HIGH:  // EXPLICIT_PRECISION_OK=mapping
      return kiln::ContractPrecision::kBF16x3;
    case Precision::HIGHEST:  // EXPLICIT_PRECISION_OK=mapping
      return kiln::ContractPrecision::kFP32;
  }
}

kiln::Scalar CeilDiv(const kiln::Scalar value, const int64_t divisor) {
  return (value + (divisor - 1)) / divisor;
}

// Rounds `value` up to a multiple of `tile`, a power of two.
kiln::Scalar RoundUp(const kiln::Scalar value, const int64_t tile) {
  return kiln::MultipleOf((value + (tile - 1)) & -tile, tile);
}

// Builds the kernel computing `out[M,N] = a[M,K] @ b[K,N]` for any
// M, K, N >= 1.
//
// Each output block accumulates in VMEM over its contraction steps. All
// (output block, contraction step) pairs form a single pipelined loop, so that
// the copies of a step overlap the MXU work of the previous one, across output
// blocks too.
absl::StatusOr<kiln::Kernel> BuildMatmulKernel(const ElementType element_type,
                                               const Precision precision) {
  TT_ASSIGN_OR_RETURN(const kiln::DType dtype, ToKilnDType(element_type));
  const kiln::Shape matrix = dtype(kiln::kDynamic, kiln::kDynamic);
  return kiln::CreateKernel(
      kiln::GridSpec{.name = "matmul", .grid = {kNumPrograms}},
      kiln::InSpec::Hbm(matrix), kiln::InSpec::Hbm(matrix),
      // The donated output buffer, which the result aliases.
      kiln::InSpec::Hbm(matrix), kiln::OutSpec::Hbm(matrix),
      [dtype, precision](kiln::HbmRef a, kiln::HbmRef b,
                         kiln::HbmRef /*donated*/, kiln::HbmRef out) {
        const kiln::Scalar m = a.Dim(0);
        const kiln::Scalar k = a.Dim(1);
        const kiln::Scalar n = b.Dim(1);
        const kiln::Scalar blocks_per_row = CeilDiv(n, kBlock);
        const kiln::Scalar num_blocks = CeilDiv(m, kBlock) * blocks_per_row;
        const kiln::Scalar steps_per_block = CeilDiv(k, kBlock);

        // This program's share of the output blocks.
        const kiln::Scalar blocks_per_program =
            CeilDiv(num_blocks, kNumPrograms);
        const kiln::Scalar first_block =
            Min(kiln::ProgramId(0) * blocks_per_program, num_blocks);
        const kiln::Scalar end_block =
            Min(first_block + blocks_per_program, num_blocks);

        // The rows, columns and depth that a step covers: an offset and an
        // extent. Extents are rounded up to whole HBM tiles where they are
        // copied, the excess being tile padding.
        const auto row = [blocks_per_row](const kiln::Scalar block) {
          return kiln::MultipleOf((block / blocks_per_row) * kBlock, kHbmRows);
        };
        const auto rows = [m, &row](const kiln::Scalar block) {
          return RoundUp(Min(m - row(block), kBlock), kHbmRows);
        };
        const auto col = [blocks_per_row](const kiln::Scalar block) {
          return kiln::MultipleOf((block % blocks_per_row) * kBlock, kLanes);
        };
        const auto cols = [n, &col](const kiln::Scalar block) {
          return RoundUp(Min(n - col(block), kBlock), kLanes);
        };
        const auto depth_offset = [steps_per_block](const kiln::Scalar step) {
          return kiln::MultipleOf((step % steps_per_block) * kBlock, kLanes);
        };
        const auto depth = [k, &depth_offset](const kiln::Scalar step) {
          return Min(k - depth_offset(step), kBlock);
        };

        // K is the lane dimension of `a`, but the row dimension of `b`.
        kiln::PipelineOperand a_blocks{
            .hbm = a,
            .block_shape = {kBlock, kBlock},
            .region = [steps_per_block, &row, &rows, &depth_offset,
                       &depth](const kiln::Scalar step) {
              const kiln::Scalar block = step / steps_per_block;
              return kiln::BlockRegion{
                  .offsets = {row(block), depth_offset(step)},
                  .sizes = {rows(block), RoundUp(depth(step), kLanes)}};
            }};
        kiln::PipelineOperand b_blocks{
            .hbm = b,
            .block_shape = {kBlock, kBlock},
            .region = [steps_per_block, &col, &cols, &depth_offset,
                       &depth](const kiln::Scalar step) {
              const kiln::Scalar block = step / steps_per_block;
              return kiln::BlockRegion{
                  .offsets = {depth_offset(step), col(block)},
                  .sizes = {RoundUp(depth(step), kHbmRows), cols(block)}};
            }};
        kiln::PipelineOperand out_blocks{
            .hbm = out,
            .block_shape = {kBlock, kBlock},
            .block_index =
                [steps_per_block](const kiln::Scalar step) {
                  return step / steps_per_block;
                },
            .region =
                [&row, &rows, &col, &cols](const kiln::Scalar block) {
                  return kiln::BlockRegion{.offsets = {row(block), col(block)},
                                           .sizes = {rows(block), cols(block)}};
                }};

        // An f32 result accumulates in its output block. A bf16 one accumulates
        // in f32 scratch, converted into the output block after the last step.
        const std::optional<kiln::VmemRef> f32_accumulator =
            dtype == kiln::f32
                ? std::nullopt
                : std::optional(kiln::AllocaVmem(kiln::f32(kBlock, kBlock)));

        kiln::EmitPipeline(
            first_block * steps_per_block, end_block * steps_per_block,
            {std::move(a_blocks), std::move(b_blocks)}, {std::move(out_blocks)},
            [precision, &depth, &f32_accumulator](const kiln::PipelineStep& s) {
              const kiln::VmemRef a_block = s.inputs[0].ref;
              const kiln::VmemRef b_block = s.inputs[1].ref;
              const kiln::PipelineBlock& out_block = s.outputs[0];
              // Past the contraction edge, both blocks hold tile padding or
              // stale data, and garbage * 0 may be NaN: zero both.
              const kiln::Scalar step_depth = depth(s.step);
              kiln::If(step_depth < kBlock, [step_depth, a_block, b_block] {
                const kiln::Shape indices = kiln::i32(kBlock, kBlock);
                kiln::Store(a_block,
                            Select(kiln::Iota(indices, /*axis=*/1) < step_depth,
                                   kiln::Load(a_block), 0));
                kiln::Store(b_block,
                            Select(kiln::Iota(indices, /*axis=*/0) < step_depth,
                                   kiln::Load(b_block), 0));
              });

              const kiln::VmemRef accumulator =
                  f32_accumulator.value_or(out_block.ref);
              kiln::If(out_block.is_first_step, [accumulator] {
                kiln::Store(accumulator, kiln::Zeros(accumulator.shape()));
              });
              kiln::Store(
                  accumulator,
                  kiln::Load(accumulator) +
                      kiln::Dot(kiln::Load(a_block), kiln::Load(b_block),
                                {.precision = ToContractPrecision(precision)}));
              if (f32_accumulator.has_value()) {
                kiln::If(out_block.is_last_step, [out_block, accumulator] {
                  kiln::Store(out_block.ref, kiln::Load(accumulator));
                });
              }
            });
      });
}

// Returns the row-major T(8,128) layout of a matrix of `dtype`.
xla::Layout MatrixLayout(const ElementType dtype) {
  if (dtype == ElementType::BF16) {
    return xla::Layout({1, 0}, {xla::Tile({8, 128}), xla::Tile({2, 1})});
  }
  return xla::Layout({1, 0}, {xla::Tile({8, 128})});
}

// Returns the parameters of the matmul executable: `a`, `b` and the donated
// output, then the S32[2] dimensions of each.
std::vector<xla::Shape> MatmulParameters(const ElementType dtype) {
  xla::Shape matrix = xla::ShapeUtil::MakeShape(
      dtype == ElementType::BF16 ? xla::BF16 : xla::F32,
      {xla::Shape::kUnboundedSize, xla::Shape::kUnboundedSize}, {true, true});
  *matrix.mutable_layout() = MatrixLayout(dtype);
  xla::Shape dimensions = xla::ShapeUtil::MakeShape(xla::S32, {2});
  *dimensions.mutable_layout() = xla::Layout({0}, {xla::Tile({128})});
  return {matrix, matrix, matrix, dimensions, dimensions, dimensions};
}

template <ElementType kDtype, Precision kPrecision>
absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
BuildMatmulExecutable(xla::PjRtClient& client,
                      UniqueCompileOptions compile_options) {
  TT_ASSIGN_OR_RETURN(kiln::Kernel kernel,
                      BuildMatmulKernel(kDtype, kPrecision));
  const std::vector<xla::Shape> parameters = MatmulParameters(kDtype);
  return CompileMosaicKernel(client, std::move(compile_options),
                             std::move(kernel), parameters,
                             /*output=*/parameters[2],
                             /*operand_to_metadata=*/{3, 4, 5, -1, -1, -1},
                             /*aliased_parameter=*/2);
}

template <ElementType kDtype>
UnboundedDynamicExecutionMetadata MakeMatmulMetadata() {
  using Metadata = UnboundedDynamicExecutionMetadata;
  return Metadata{
      .logical_input_to_pjrt_param = {0, 1},
      .donated_output_params = {Metadata::DonatedOutputParam{
          .logical_output_idx = 0, .pjrt_param_idx = 2}},
      .metadata_params =
          {
              Metadata::MetadataParamBinding{.buffer_pjrt_idx = 0,
                                             .shape_pjrt_idx = 3},
              Metadata::MetadataParamBinding{.buffer_pjrt_idx = 1,
                                             .shape_pjrt_idx = 4},
              Metadata::MetadataParamBinding{.buffer_pjrt_idx = 2,
                                             .shape_pjrt_idx = 5},
          },
      .pjrt_param_shapes = MatmulParameters(kDtype),
      .is_compatible =
          [](absl::Span<const Shape> input_shapes,
             absl::Span<const Shape> output_shapes) {
            // The extentless key already fixes two rank-2 inputs and a rank-2
            // output of `kDtype`.
            const xla::Layout layout = MatrixLayout(kDtype);
            // Empty products (K = 0 wouldn't write zeros) and dimensions that
            // overflow S32 runtime metadata or `CeilDiv(..., kBlock)` are left
            // to XLA.
            constexpr int64_t kMaxDim =
                std::numeric_limits<int32_t>::max() - (kBlock - 1);
            const auto valid_dim = [](const int64_t dim) {
              return dim > 0 && dim <= kMaxDim;
            };
            return valid_dim(input_shapes[0].dimensions()[0]) &&
                   valid_dim(input_shapes[0].dimensions()[1]) &&
                   valid_dim(input_shapes[1].dimensions()[1]) &&
                   HasDeviceLayouts(input_shapes, {layout, layout}) &&
                   HasDeviceLayouts(output_shapes, {layout});
          },
  };
}

// Returns the ExtentlessKey of the graph that `torch.mm` traces for two
// matrices of `dtype` at matmul precision `precision`.
ExtentlessKey MmCacheKey(const ElementType dtype, const Precision precision) {
  absl::StatusOr<OpParamCacheKeys> param_keys =
      *OpParamCacheKeysBuilder().SetParam("precision", precision);
  ABSL_CHECK_OK(param_keys);  // CRASH_OK=static-init registration bug
  // Only ranks and dtypes reach the extentless key, so the extents are
  // placeholders.
  const Dimensions dims = {-1, -1};
  GraphSignature graph;
  graph.SetCorePinningMode(CorePinningMode::kUnpinned);
  const int lhs = graph.AddInput(dims, dtype);
  const int rhs = graph.AddInput(dims, dtype);
  int out = -1;
  graph.AddOp(
      OpName::kMmOut, *std::move(param_keys), /*donated_inputs=*/{},
      [lhs, rhs, &dims, dtype, &out](GraphSignature::OpSignatureBuilder& op) {
        op.AddInput(lhs);
        op.AddInput(rhs);
        out = op.AddOutput(dims, dtype);
      });
  graph.AddGraphOutput(out);
  return graph.GetKey().extentless_key();
}

template <ElementType kDtype, Precision kPrecision>
UnboundedDynamicKernel GetMatmulKernel() {
  return UnboundedDynamicKernel{
      .build = BuildMatmulExecutable<kDtype, kPrecision>,
      .metadata = MakeMatmulMetadata<kDtype>(),
      .key = MakeUnboundedDynamicKernelCacheKey(
          MmCacheKey(kDtype, kPrecision),
          static_cast<int>(MatmulKernelId::kBlock512)),
      .enabled_by_default = false,
  };
}

}  // namespace

std::vector<UnboundedDynamicKernelRegistration>
GetMatmulUnboundedDynamicKernels() {
  constexpr Precision kDefault =
      Precision::DEFAULT;                       // EXPLICIT_PRECISION_OK=key
  constexpr Precision kHigh = Precision::HIGH;  // EXPLICIT_PRECISION_OK=key
  constexpr Precision kHighest =
      Precision::HIGHEST;  // EXPLICIT_PRECISION_OK=key
  return {
      {.kernel = GetMatmulKernel<ElementType::F32, kDefault>()},
      {.kernel = GetMatmulKernel<ElementType::F32, kHigh>()},
      {.kernel = GetMatmulKernel<ElementType::F32, kHighest>()},
      // BF16 operands are exact in a single bf16 pass, so one kernel serves
      // every precision.
      {.kernel = GetMatmulKernel<ElementType::BF16, kDefault>(),
       .additional_search_keys = {MmCacheKey(ElementType::BF16, kHigh),
                                  MmCacheKey(ElementType::BF16, kHighest)}},
  };
}

}  // namespace torch_tpu
