// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "csrc/ops/repeat_interleave/repeat_interleave_aten_kernels.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/repeat_interleave.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/GradMode.h"
#include "c10/core/ScalarType.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/core/WrapDimMinimal.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/const_tensor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fake_tensor_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/copy_from/copy_common.h"
#include "csrc/ops/copy_from/tpu_to_cpu.h"
#include "csrc/ops/cumsum/cumsum.h"
#include "csrc/ops/index_select/index_select.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/sum.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/csrc/autograd/custom_function.h"
#include "torch/headeronly/core/DeviceType.h"

namespace torch_tpu {

// =============================================================================
// Architecture & Dispatch Overview: Uniform vs. Variable `repeat_interleave`
// =============================================================================
//
// This file implements the TPU ATen eager and autograd kernels for PyTorch's
// `torch.repeat_interleave` operator family:
//   1. `aten::repeat_interleave.self_int(Tensor self, SymInt repeats,
//       int? dim=None, SymInt? output_size=None)`
//   2. `aten::repeat_interleave.self_Tensor(Tensor self, Tensor repeats,
//       int? dim=None, int? output_size=None)`
//
// -----------------------------------------------------------------------------
// Why a Dedicated Uniform Lowering Matters on TPU
// -----------------------------------------------------------------------------
// Without a dedicated kernel, uniform `repeat_interleave` (`repeats` is an int
// or a 1-element tensor) suffers from two major bottlenecks:
//
//   1. PyTorch Composite Fallback Overhead (Eager Mode):
//      PyTorch's default composite implementation decomposes uniform
//      `repeat_interleave` into a sequence of separate ATen operations:
//        `self.unsqueeze(dim + 1).expand(...).clone(Contiguous).reshape(...)`
//      In eager mode on TPU:
//        - Each intermediate op incurs host-side C++/Python dispatch overhead.
//        - In training backward, PyTorch's autograd engine differentiates
//          through `clone(Contiguous)` and `expand`, allocating a redundant
//          contiguous copy of the full expanded gradient tensor (`[B, T, H*R,
//          D]`) before summing along the repeat axis. For large LLM/SSM tensors
//          (e.g., Mamba-3 or GQA KV heads), this doubles peak backward memory.
//
//   2. Gather vs. Broadcast Lowering (Device Execution):
//      Lowering uniform `repeat_interleave` via variable-length index selection
//      (`BuildRepeatInterleaveShlo`: `CumSum` + $O(M \times N)$ comparison
//      matrix + `Gather`) allocates large index buffers and executes indirect
//      memory gathers on TPU VPU/HBM.
//      By contrast, uniform repeats can be expressed purely as metadata
//      transformations + native broadcast/reduction:
//        - Forward:  `stablehlo.broadcast_in_dim` + `stablehlo.reshape`
//        - Backward: `stablehlo.reshape` + `stablehlo.reduce` (`add`)
//      XLA fuses `broadcast_in_dim` + `reshape` into direct MXU/VPU register
//      broadcasts and lowers the backward pass to a single contiguous reduction
//      without any intermediate copies.
//
// -----------------------------------------------------------------------------
// Dispatch Regimes Handled in This File
// -----------------------------------------------------------------------------
//   - Eager Inference (`PrivateUse1`):
//     Executes `AtenRepeatInterleaveSelfInt` / `AtenRepeatInterleave`, building
//     and caching single-op StableHLO graphs via `TT_KERNEL` and `DispatchOp`.
//   - Eager Training (`AutogradPrivateUse1`):
//     Executes `AtenRepeatInterleaveSelfIntAutogradNode`, a custom
//     `torch::autograd::Function` that retains ZERO forward activation tensors
//     in memory (saving only lightweight `int64_t` shape metadata on
//     `AutogradContext`) and dispatches backward directly to
//     `BuildRepeatInterleaveBackwardShlo`.
//   - Compilation / Tracing (`torch.compile`, `FakeTensor`, `functorch`):
//     Native `TT_KERNEL` dispatches require physical PJRT device buffers and
//     cannot run on storage-less `FakeTensor`s or functorch wrappers. When
//     `IsFakeOrFunctorchTensor` detects tracing mode, execution delegates to
//     `DecomposeRepeatInterleaveSelfInt` (`unsqueeze -> expand -> reshape`),
//     which traces without Dynamo graph breaks and lowers cleanly to the same
//     `BroadcastInDim + Reshape` StableHLO ops in the compiler backend.

namespace {

// Lowers variable-length `repeat_interleave` (`repeats` is a 1D tensor of
// per-element repeat counts) to StableHLO using cumulative sums, broadcasting
// comparisons, and index gather.
//
// Unlike uniform repeats, variable-length repeats cannot use `BroadcastInDim`
// because each element $i$ along `dim` is repeated a different number of times
// $R_i$, which would require a ragged/jagged intermediate dimension. Instead,
// we construct the flat source-index mapping vector $I \in \mathbb{Z}^M$ of
// length $M = \sum_i R_i$ on device and gather from `self`:
//
// Algorithm Walkthrough:
//   Given `self = [A, B, C]`, `repeats = [2, 1, 3]`, `output_size = 6`:
//   1. Inclusive cumulative sum along axis 0:
//        `cumsum = cumsum(repeats) = [2, 3, 6]`
//   2. Exclusive cumulative sum (start offset of each element's output run):
//        `cumsum_exclusive = cumsum - repeats = [0, 2, 3]`
//   3. Compare output position `iota(M)` against start offsets via 2D broadcast
//      greater-than-or-equal (`iota[M, 1] >= cumsum_exclusive[1, N]`):
//        `iota_bcst`      = `[[0], [1], [2], [3], [4], [5]]` (shape `[6, 1]`)
//        `cumsum_ex_bcst` = `[[0, 2, 3]]`                    (shape `[1, 3]`)
//        `comp` (`[6, 3]`) =
//          row 0 (pos 0 >= [0,2,3]): `[T, F, F]` -> sum = 1
//          row 1 (pos 1 >= [0,2,3]): `[T, F, F]` -> sum = 1
//          row 2 (pos 2 >= [0,2,3]): `[T, T, F]` -> sum = 2
//          row 3 (pos 3 >= [0,2,3]): `[T, T, T]` -> sum = 3
//          row 4 (pos 4 >= [0,2,3]): `[T, T, T]` -> sum = 3
//          row 5 (pos 5 >= [0,2,3]): `[T, T, T]` -> sum = 3
//   4. Convert boolean mask to `i64` and reduce-sum along axis 1 (`sum_indices`
//      gives 1-based bucket indices `[1, 1, 2, 3, 3, 3]`), then subtract 1 for
//      0-based gather indices:
//        `indices = sum_indices - 1 = [0, 0, 1, 2, 2, 2]`
//   5. Gather along `dim_val` using `BuildIndexSelectShlo`:
//        `output = self.index_select(dim_val, indices) = [A, A, B, C, C, C]`
//
// Steps 1-4 live in `BuildRepeatInterleaveIndicesShlo`, which also lowers
// `aten::repeat_interleave.Tensor` (the index vector alone, with no gather).
absl::StatusOr<mlir::MlirOp> BuildRepeatInterleaveIndicesShlo(
    mlir::MlirOp repeats, const int64_t output_size) {
  auto& builder = repeats.getBuilder();
  auto& op_builder = builder.getOpBuilder();
  const auto i64_type = op_builder.getI64Type();

  const mlir::RankedTensorType repeats_type = GetTensorTypeOrDie(repeats);
  const int64_t r_size = repeats_type.getDimSize(0);

  // Ensure `repeats` is i64 so cumulative sums and index arithmetic do not
  // overflow int32 on large tensors.
  mlir::MlirOp repeats_op =
      mlir::stablehlo::ConvertElementType(repeats, i64_type);

  // Step 1: Compute inclusive cumulative sum of `repeats` along axis 0.
  TT_ASSIGN_OR_RETURN(auto cumsum,
                      BuildCumsumShlo(0, std::nullopt, repeats_op));

  // Step 2: Compute exclusive cumulative sum (`cumsum_exclusive = cumsum -
  // repeats`), representing the starting output index for each input element.
  auto cumsum_exclusive = mlir::stablehlo::Subtract(cumsum, repeats_op);

  // Step 3: Generate flat output positions `iota(M)` of shape `[output_size,
  // 1]` and compare against `cumsum_exclusive` of shape `[1, r_size]`. Formula:
  // `indices = (iota(M)[:, None] >= cumsum_exclusive[None, :]).sum(-1) - 1`
  const auto iota_type =
      mlir::makeTensorType(builder.getContext(), {output_size}, i64_type);
  auto iota = mlir::stablehlo::Iota(builder, iota_type, 0);

  auto iota_reshaped = mlir::stablehlo::Reshape(iota, {output_size, 1});
  auto cumsum_ex_reshaped =
      mlir::stablehlo::Reshape(cumsum_exclusive, {1, r_size});

  // Note: Broadcasting to `[output_size, r_size]` materializes an O(M * N)
  // intermediate comparison matrix. For 1-element `repeats`, callers should
  // always route to `BuildUniformRepeatInterleaveShlo` to avoid this cost.
  TT_ASSIGN_OR_RETURN(auto iota_bcst,
                      BroadcastIfNeeded(iota_reshaped, {output_size, r_size}));
  TT_ASSIGN_OR_RETURN(
      auto cumsum_bcst,
      BroadcastIfNeeded(cumsum_ex_reshaped, {output_size, r_size}));

  auto comp = mlir::stablehlo::Compare(
      iota_bcst, cumsum_bcst, mlir::stablehlo::ComparisonDirection::GE);
  auto comp_i64 = mlir::stablehlo::ConvertElementType(comp, i64_type);

  // Step 4: Sum across axis 1 (`r_size`) to count how many start offsets are
  // less than or equal to each output position `k`, yielding 1-based indices.
  auto zero_init = MakeScalarConstant(builder, 0, i64_type);
  auto sum_indices = mlir::stablehlo::Reduce(
      builder, {comp_i64}, {zero_init},
      [](mlir::RegionBuilder& rb) {
        auto& body_builder = rb.getOpBuilder();
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
            body_builder.getI64Type(), rb.getRegion(), body_builder);
      },
      {1})[0];

  // Convert 1-based counts to 0-based gather indices.
  auto one_cst = MakeConstantLike(sum_indices, 1LL);
  return mlir::stablehlo::Subtract(sum_indices, one_cst);
}

absl::StatusOr<mlir::MlirOp> BuildRepeatInterleaveShlo(
    mlir::MlirOp self, mlir::MlirOp repeats, const std::optional<int64_t> dim,
    const int64_t output_size) {
  mlir::MlirOp self_op = self;
  mlir::MlirOp repeats_op = repeats;
  const int64_t dim_val = dim.value_or(0);

  // PyTorch semantics: when `dim` is omitted (`None`), both `self` and
  // `repeats` are flattened to 1D prior to repeating along dimension 0.
  if (!dim.has_value()) {
    self_op = Flatten(self);
    repeats_op = Flatten(repeats);
  }

  // Steps 1-4: Build the 0-based source index for every output position.
  TT_ASSIGN_OR_RETURN(
      auto indices, BuildRepeatInterleaveIndicesShlo(repeats_op, output_size));

  // Step 5: Select elements from `self_op` along `dim_val` using `indices`.
  return BuildIndexSelectShlo(self_op, dim_val, indices);
}

// Lowers `repeat_interleave` with uniform integer repeats (`repeats` is a
// scalar integer) to StableHLO using `BroadcastInDim` and `Reshape`.
//
// Unlike variable-length `BuildRepeatInterleaveShlo` (which requires $O(M
// \times N)$ comparison matrices and indirect index gathers), uniform repeats
// can be expressed purely as a rank-(N+1) broadcast followed by a rank-N
// reshape:
//
// Mathematical & Shape Transformation Walkthrough:
//   Suppose `self` has shape `[B, T, H, D]` (rank 4), `dim = 2`, `repeats = R`.
//   We want each slice `self[:, :, h, :]` to be replicated contiguously `R`
//   times along axis 2 so that output index `(b, t, h * R + r, d)` equals
//   `self[b, t, h, d]` for all $r \in [0, R)$.
//
//   1. Insert a new broadcast axis of size `R` immediately after `dim` (at
//      index `dim + 1 = 3`):
//        `bcast_shape` = `[B, T, H, R, D]` (rank 5)
//        `bcast_dims`  = `[0, 1, 2, 4]`
//      Here `bcast_dims[i]` specifies which axis of `bcast_shape` corresponds
//      to axis `i` of `self`. Axes `0..dim` map to `0..dim`, while axes
//      `dim+1..rank-1` shift right by 1 to `dim+2..rank`. Axis `dim+1` has no
//      source axis in `self`, so `BroadcastInDim` replicates `self` `R` times
//      along that axis.
//
//   2. Emit `stablehlo.broadcast_in_dim(self, bcast_dims) -> [B, T, H, R, D]`.
//
//   3. Collapse adjacent axes `dim` (size `H`) and `dim + 1` (size `R`) into a
//      single axis of size `H * R` via `stablehlo.reshape`:
//        `out_shape` = `[B, T, H * R, D]` (rank 4)
//      Because row-major layout places axis `dim + 1` (`R`) as the fast-varying
//      inner dimension relative to axis `dim` (`H`), collapsing `(H, R) -> H*R`
//      interleaves the `R` copies of each element `h` contiguously:
//        `[h_0, h_0, ..., h_1, h_1, ..., h_{H-1}, h_{H-1}]`.
//
// When `dim` is `std::nullopt`, `self` is first flattened to 1D (`dim_val = 0`)
// before broadcasting to `[N, R]` and reshaping to `[N * R]`.
absl::StatusOr<mlir::MlirOp> BuildUniformRepeatInterleaveShlo(
    mlir::MlirOp self, std::optional<int64_t> dim, int64_t repeats) {
  auto& builder = self.getBuilder();
  mlir::MlirOp self_op = self;
  const int64_t dim_val = dim.value_or(0);

  // If no dimension was specified, PyTorch semantics require flattening the
  // input tensor to 1D and repeating along axis 0.
  if (!dim.has_value()) {
    self_op = Flatten(self);
  }

  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
  const auto shape = self_type.getShape();
  const int64_t rank = self_type.getRank();

  // Edge case: if `repeats == 0` or the target dimension is already empty
  // (`shape[dim_val] == 0`), the output extent along `dim_val` is 0. Return a
  // rank-preserving zero-length slice along `dim_val` rather than constructing
  // a degenerate zero-extent `BroadcastInDim`.
  if (repeats == 0 || (rank > 0 && shape[dim_val] == 0)) {
    const llvm::SmallVector<int64_t> start_indices(rank, 0);
    llvm::SmallVector<int64_t> limit_indices(shape.begin(), shape.end());
    if (rank > 0) {
      limit_indices[dim_val] = 0;
    }
    const llvm::SmallVector<int64_t> strides(rank, 1);
    return mlir::stablehlo::Slice(self_op, start_indices, limit_indices,
                                  strides);
  }

  // Identity fast path: repeating each element once (`repeats == 1`) leaves
  // the tensor unchanged (or returns the flattened 1D tensor when `dim` was
  // `std::nullopt`). Returning `self_op` directly avoids emitting redundant
  // MLIR broadcast and reshape operations.
  if (repeats == 1) {
    return self_op;
  }

  // Step 1: Construct the rank-(N+1) broadcast shape by inserting `repeats` at
  // axis `dim_val + 1`, immediately following the target dimension `dim_val`.
  llvm::SmallVector<int64_t> bcast_shape;
  bcast_shape.reserve(rank + 1);
  for (int64_t i = 0; i <= dim_val; ++i) {
    bcast_shape.push_back(shape[i]);
  }
  bcast_shape.push_back(repeats);
  for (int64_t i = dim_val + 1; i < rank; ++i) {
    bcast_shape.push_back(shape[i]);
  }

  // Step 2: Construct the dimension mapping vector `bcast_dims` for
  // `stablehlo.broadcast_in_dim`. Each input axis `i` in `self_op` maps to:
  //   - `i`     for `i <= dim_val`
  //   - `i + 1` for `i > dim_val` (shifted right by the newly inserted axis).
  llvm::SmallVector<int64_t> bcast_dims;
  bcast_dims.reserve(rank);
  for (int64_t i = 0; i <= dim_val; ++i) {
    bcast_dims.push_back(i);
  }
  for (int64_t i = dim_val + 1; i < rank; ++i) {
    bcast_dims.push_back(i + 1);
  }

  const auto bcast_type = mlir::makeTensorType(
      builder.getContext(), bcast_shape, self_type.getElementType());
  const auto bcast =
      mlir::stablehlo::BroadcastInDim(bcast_type, self_op, bcast_dims);

  // Step 3: Collapse adjacent axes `dim_val` (size D) and `dim_val + 1`
  // (size `repeats`) into a single axis of size `D * repeats`.
  llvm::SmallVector<int64_t> out_shape(shape.begin(), shape.end());
  out_shape[dim_val] *= repeats;
  return mlir::stablehlo::Reshape(bcast, out_shape);
}

// Holds validated and canonicalized parameters for uniform `repeat_interleave`.
struct RepeatInterleaveSelfIntValidation {
  // Non-negative integer repeat count.
  int64_t repeats_val;
  // Canonical non-negative dimension index in `[0, self.dim() - 1]`, or
  // `std::nullopt` if `dim` was omitted and `self` is flattened to 1D.
  std::optional<int64_t> resolved_dim;
  // Total output extent along `resolved_dim` (`dim_size * repeats_val`).
  int64_t resolved_output_size;
};

// Validates and canonicalizes arguments for `repeat_interleave.self_int`
// according to upstream PyTorch ATen semantics:
//   1. Materializes `repeats` via `guard_int` (which records a symbolic guard
//      when tracing with dynamic shapes or extracts the concrete `int64_t` in
//      eager mode) and verifies `repeats >= 0`.
//   2. Wraps negative `dim` indices against `self.dim()` via `SafeWrapDim`,
//      throwing `error::kInvalidArgument` if out of range.
//   3. Computes `resolved_output_size = dim_size * repeats_val` and verifies
//      that any user-supplied `output_size` matches exactly.
RepeatInterleaveSelfIntValidation ValidateRepeatInterleaveSelfInt(
    const at::Tensor& self, c10::SymInt repeats, std::optional<int64_t> dim,
    std::optional<c10::SymInt> output_size) {
  const int64_t repeats_val = repeats.guard_int(__FILE__, __LINE__);
  TT_CHECK_THROW(repeats_val >= 0, error::kInvalidArgument)
      << "expected repeats to be greater than or equal to zero, got "
      << repeats_val;

  std::optional<int64_t> resolved_dim = std::nullopt;
  int64_t dim_size;
  if (dim.has_value()) {
    TT_ASSIGN_OR_THROW(const int64_t wrapped_dim,
                       SafeWrapDim(*dim, self.dim()));
    resolved_dim = wrapped_dim;
    dim_size = self.size(wrapped_dim);
  } else {
    dim_size = self.numel();
  }

  const int64_t resolved_output_size = dim_size * repeats_val;
  if (output_size.has_value()) {
    const int64_t output_size_val = output_size->guard_int(__FILE__, __LINE__);
    TT_CHECK_THROW(output_size_val == resolved_output_size,
                   error::kInvalidArgument)
        << "expected output_size to equal " << resolved_output_size << ", got "
        << output_size_val;
  }
  return {repeats_val, resolved_dim, resolved_output_size};
}

// Decomposes uniform `repeat_interleave` into standard ATen view operations
// (`unsqueeze` -> `expand` -> `reshape`) when tracing under `torch.compile`
// (`FakeTensor` / `ProxyTorchDispatchMode`) or `functorch` transforms.
//
// Why this decomposition is used during tracing:
//   - Native `TT_KERNEL` dispatches operate on physical TPU device buffers and
//     cannot execute on storage-less `FakeTensor`s or functorch wrappers.
//   - Decomposing into primitive ATen view operations allows TorchDynamo /
//   AOTAutograd
//     to trace through `repeat_interleave` without graph breaks and derive
//     exact symbolic output shapes.
//   - Note on `clone(Contiguous)` in FX graphs: calling `.reshape()` on the
//     non-contiguous expanded FakeTensor causes ATen's view logic to emit
//     `unsqueeze -> expand -> clone(Contiguous) -> _unsafe_view` in the traced
//     FX graph. Unlike eager execution (where `clone` allocates and copies a
//     physical buffer), the TorchTPU FX-to-StableHLO exporter recognizes
//     `clone(expand) -> _unsafe_view` and lowers it directly to
//     `stablehlo.broadcast_in_dim` + `stablehlo.reshape` with zero copies.
at::Tensor DecomposeRepeatInterleaveSelfInt(
    const at::Tensor& self, c10::SymInt repeats, std::optional<int64_t> dim,
    std::optional<c10::SymInt> output_size) {
  int64_t resolved_dim = 0;
  if (dim.has_value()) {
    resolved_dim = c10::maybe_wrap_dim(*dim, self.dim());
  }
  at::Tensor input = dim.has_value() ? self : self.flatten();

  const c10::SymIntArrayRef in_sym_sizes = input.sym_sizes();
  std::vector<c10::SymInt> expand_shape(in_sym_sizes.begin(),
                                        in_sym_sizes.end());
  expand_shape.insert(expand_shape.begin() + resolved_dim + 1, repeats);

  const c10::SymInt resolved_output_size =
      output_size.has_value() ? *output_size
                              : (in_sym_sizes[resolved_dim] * repeats);
  std::vector<c10::SymInt> out_shape(in_sym_sizes.begin(), in_sym_sizes.end());
  out_shape[resolved_dim] = resolved_output_size;

  return input.unsqueeze(resolved_dim + 1)
      .expand_symint(expand_shape)
      .reshape_symint(out_shape);
}

struct RepeatsStats {
  int64_t sum = 0;
  int64_t min = 0;
};

// Synchronizes `repeats` to host memory (if on TPU) via direct device-to-host
// DMA (`CopyTpuToCpu`) without dispatching any TPU reduction kernel or ATen
// operator, and computes both `sum(repeats)` and `min(repeats)` in a single C++
// pass.
absl::StatusOr<RepeatsStats> ComputeRepeatsSumAndMinOnHost(
    const at::Tensor& repeats) {
  const int64_t numel = repeats.numel();
  if (numel == 0) {
    return RepeatsStats{0, 0};
  }
  at::Tensor host_repeats =
      repeats.is_cpu()
          ? repeats.contiguous()
          : at::empty(repeats.sizes(), at::TensorOptions()
                                           .dtype(repeats.scalar_type())
                                           .device(at::kCPU));
  if (!repeats.is_cpu()) {
    TT_RETURN_IF_ERROR(CopyTpuToCpu(
        repeats, host_repeats, GetTransferSyncMode(/*non_blocking=*/false)));
  }
  RepeatsStats stats{0, std::numeric_limits<int64_t>::max()};
  if (host_repeats.scalar_type() == at::kLong) {
    const int64_t* const data = host_repeats.data_ptr<int64_t>();
    for (int64_t i = 0; i < numel; ++i) {
      const int64_t v = data[i];
      stats.sum += v;
      stats.min = std::min(stats.min, v);
    }
  } else {
    const int32_t* const data = host_repeats.data_ptr<int32_t>();
    for (int64_t i = 0; i < numel; ++i) {
      const int64_t v = static_cast<int64_t>(data[i]);
      stats.sum += v;
      stats.min = std::min(stats.min, v);
    }
  }
  return stats;
}

}  // namespace

// Eager `PrivateUse1` kernel for `aten::repeat_interleave.self_int`.
//
// Dispatches uniform scalar repeats directly to
// `BuildUniformRepeatInterleaveShlo`
// (`BroadcastInDim` + `Reshape`), avoiding index tensor construction,
// device-to-host synchronization, or indirect memory gathers. When invoked
// under `torch.compile` or functorch (`IsFakeOrFunctorchTensor`), delegates to
// `DecomposeRepeatInterleaveSelfInt`.
at::Tensor AtenRepeatInterleaveSelfInt(const at::Tensor& self,
                                       c10::SymInt repeats,
                                       std::optional<int64_t> dim,
                                       std::optional<c10::SymInt> output_size) {
  if (IsFakeOrFunctorchTensor(self)) {
    return DecomposeRepeatInterleaveSelfInt(self, repeats, dim, output_size);
  }
  // Normalize optional `output_size` to `-1` when omitted so `TT_KERNEL`
  // includes it in the compilation cache key (`param_keys`).
  const int64_t output_size_val =
      output_size.has_value() ? output_size->guard_int(__FILE__, __LINE__) : -1;
  TT_KERNEL(
      OpName::kRepeatInterleaveSelfInt, param_keys,
      (self, repeats, dim, output_size_val), {
        const auto [repeats_val, resolved_dim, resolved_output_size] =
            ValidateRepeatInterleaveSelfInt(self, repeats, dim, output_size);

        TT_ASSIGN_OR_THROW(const mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        // Determine output tensor dimensions: either replace axis
        // `*resolved_dim` with `resolved_output_size` or produce a 1D tensor
        // `{resolved_output_size}` if `dim` was omitted (`std::nullopt`).
        Dimensions out_dims;
        if (resolved_dim.has_value()) {
          out_dims = CopyIntVector(self.sizes());
          out_dims[*resolved_dim] = resolved_output_size;
        } else {
          out_dims = Dimensions({resolved_output_size});
        }

        auto shlo_builder =
            [resolved_dim,
             repeats_val](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildUniformRepeatInterleaveShlo(input, resolved_dim,
                                                  repeats_val);
        };

        TT_ASSIGN_OR_THROW(
            auto result,
            (DispatchOp<1>(std::move(shlo_builder), self,
                           {.out_dtype = out_dtype,
                            .out_dims = out_dims,
                            .op_param_cache_keys = std::move(param_keys)})));
        return MakeTensor(result);
      });
}

// Eager `PrivateUse1` kernel for `aten::repeat_interleave.self_Tensor`.
//
// Optimization for 1-Element Tensor Repeats (`repeats.numel() == 1`):
//   Passing a 1-element tensor (e.g., `repeats = torch.tensor([8])`) represents
//   uniform repetition across all elements of `self`. Routing this through the
//   general variable-length kernel (`BuildRepeatInterleaveShlo`) would perform
//   an unnecessary `CumSum`, materialize an $O(M \times N)$ comparison matrix,
//   and execute an indirect gather.
//   Instead, we intercept `repeats.numel() == 1` inside `TT_KERNEL` and lower
//   directly via `BuildUniformRepeatInterleaveShlo` (`BroadcastInDim` +
//   `Reshape`):
//     - If `output_size` is explicitly provided, we compute the scalar repeat
//       factor purely on the host (`*output_size / dim_size`) with zero device
//       synchronization.
//     - If `output_size` is omitted, we copy `repeats` directly to host memory
//       via `ComputeRepeatsSumAndMinOnHost` (`CopyTpuToCpu`) without
//       dispatching any intermediate ATen or TPU reduction ops.
at::Tensor AtenRepeatInterleave(const at::Tensor& self,
                                const at::Tensor& repeats,
                                std::optional<int64_t> dim,
                                std::optional<int64_t> output_size) {
  TT_KERNEL(
      OpName::kRepeatInterleaveSelfTensor, param_keys,
      (self, repeats, dim, output_size), {
        TT_ASSIGN_OR_THROW(const mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));

        if (repeats.numel() == 1) {
          int64_t scalar_repeats;
          if (output_size.has_value()) {
            int64_t dim_size = self.numel();
            if (dim.has_value()) {
              TT_ASSIGN_OR_THROW(const int64_t wrapped_dim,
                                 SafeWrapDim(*dim, self.dim()));
              dim_size = self.size(wrapped_dim);
            }
            scalar_repeats = dim_size > 0 ? (*output_size / dim_size) : 0;
          } else {
            TT_ASSIGN_OR_THROW(const RepeatsStats stats,
                               ComputeRepeatsSumAndMinOnHost(repeats));
            scalar_repeats = stats.sum;
          }
          const auto [repeats_val, resolved_dim, resolved_output_size] =
              ValidateRepeatInterleaveSelfInt(
                  self, scalar_repeats, dim,
                  output_size.has_value()
                      ? std::optional<c10::SymInt>(*output_size)
                      : std::nullopt);

          Dimensions out_dims;
          if (resolved_dim.has_value()) {
            out_dims = CopyIntVector(self.sizes());
            out_dims[*resolved_dim] = resolved_output_size;
          } else {
            out_dims = Dimensions({resolved_output_size});
          }

          auto shlo_builder =
              [resolved_dim, repeats_val](
                  mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
            return BuildUniformRepeatInterleaveShlo(input, resolved_dim,
                                                    repeats_val);
          };

          TT_ASSIGN_OR_THROW(
              auto result,
              (DispatchOp<1>(std::move(shlo_builder), self,
                             {.out_dtype = out_dtype,
                              .out_dims = out_dims,
                              .op_param_cache_keys = std::move(param_keys)})));
          return MakeTensor(result);
        }

        int64_t resolved_output_size;
        if (output_size.has_value()) {
          resolved_output_size = *output_size;
        } else {
          // Host sync to get the resolved output size when it is not provided
          // explicitly.
          //
          // StableHLO compilation requires knowing the output shape statically
          // (or its bounds) at compile time. Since the output size of
          // repeat_interleave depends on the *values* of the `repeats` tensor,
          // and those values are on the device, we copy `repeats` to the host
          // via `CopyTpuToCpu` and sum on CPU without dispatching an extra TPU
          // reduction kernel.
          TT_ASSIGN_OR_THROW(const RepeatsStats stats,
                             ComputeRepeatsSumAndMinOnHost(repeats));
          resolved_output_size = stats.sum;
        }

        std::optional<int64_t> resolved_dim = std::nullopt;
        if (dim.has_value()) {
          TT_ASSIGN_OR_THROW(const int64_t wrapped_dim,
                             SafeWrapDim(*dim, self.dim()));
          resolved_dim = wrapped_dim;
        }

        Dimensions out_dims;
        if (resolved_dim.has_value()) {
          out_dims = CopyIntVector(self.sizes());
          out_dims[*resolved_dim] = resolved_output_size;
        } else {
          out_dims = Dimensions({resolved_output_size});
        }

        auto shlo_builder = [resolved_dim, resolved_output_size](
                                FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          return BuildRepeatInterleaveShlo(inputs[0], inputs[1], resolved_dim,
                                           resolved_output_size);
        };

        TT_ASSIGN_OR_THROW(
            auto result,
            (DispatchOp<2>(std::move(shlo_builder), {self, repeats},
                           {.out_dtype = out_dtype,
                            .out_dims = out_dims,
                            .op_param_cache_keys = std::move(param_keys)})));
        return MakeTensor(result);
      });
}

namespace {

// Validates `repeats` (1D, `int64` or `int32`, non-negative elements) and
// resolves the total output size for `repeat_interleave.Tensor` and
// `repeat_interleave.Tensor_out`.
int64_t ValidateAndResolveRepeatInterleaveTensor(
    const at::Tensor& repeats, const std::optional<c10::SymInt>& output_size,
    const int64_t output_size_val) {
  TT_CHECK_THROW(repeats.dim() == 1, error::kInvalidArgument)
      << "expected repeats to be a 1D tensor, got " << repeats.dim() << "D";
  TT_CHECK_THROW(
      repeats.scalar_type() == at::kLong || repeats.scalar_type() == at::kInt,
      error::kInvalidArgument)
      << "expected repeats to have int64 or int32 dtype, got "
      << ToString(repeats.scalar_type());

  int64_t total = 0;
  if (repeats.size(0) > 0) {
    if (output_size.has_value()) {
      total = output_size_val;
      TT_CHECK_THROW(total >= 0, error::kInvalidArgument)
          << "expected output_size to be greater than or equal to zero, got "
          << total;
    } else {
      TT_ASSIGN_OR_THROW(const RepeatsStats stats,
                         ComputeRepeatsSumAndMinOnHost(repeats));
      TT_CHECK_THROW(stats.min >= 0, error::kInvalidArgument)
          << "expected repeats to be greater than or equal to zero, got "
          << stats.min;
      total = stats.sum;
    }
  }
  return total;
}

absl::StatusOr<mlir::MlirOp> BuildRepeatInterleaveTensorShlo(
    mlir::MlirOp input, const int64_t total) {
  TT_ASSIGN_OR_RETURN(auto indices,
                      BuildRepeatInterleaveIndicesShlo(input, total));
  return mlir::stablehlo::ConvertElementType(
      indices, GetTensorTypeOrDie(input).getElementType());
}

}  // namespace

// Eager `PrivateUse1` kernel for `aten::repeat_interleave.Tensor`.
//
// Returns the 1D index vector `[0] * repeats[0] + [1] * repeats[1] + ...` in
// the dtype of `repeats`. This is the index tensor that the variable-length
// `self_Tensor` path gathers with, so it reuses
// `BuildRepeatInterleaveIndicesShlo` and skips the gather.
//
// When `output_size` is omitted, `ComputeRepeatsSumAndMinOnHost` copies
// `repeats` directly to host memory (`CopyTpuToCpu`) and computes both the
// total output size and the minimum element (for the non-negative check) on CPU
// in a single pass, avoiding a separate TPU reduction dispatch.
at::Tensor AtenRepeatInterleaveTensor(
    const at::Tensor& repeats, const std::optional<c10::SymInt> output_size) {
  const int64_t output_size_val =
      output_size.has_value() ? output_size->guard_int(__FILE__, __LINE__) : -1;
  TT_KERNEL(
      OpName::kRepeatInterleaveTensor, param_keys, (repeats, output_size_val), {
        const int64_t total = ValidateAndResolveRepeatInterleaveTensor(
            repeats, output_size, output_size_val);
        if (total == 0) {
          TT_ASSIGN_OR_THROW(
              at::Tensor empty,
              MakeEmptyTensor({0}, repeats.scalar_type(), repeats.device()));
          return empty;
        }

        TT_ASSIGN_OR_THROW(const mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(repeats.scalar_type()));
        auto shlo_builder =
            [total](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildRepeatInterleaveTensorShlo(input, total);
        };
        TT_ASSIGN_OR_THROW(
            auto result,
            (DispatchOp<1>(std::move(shlo_builder), repeats,
                           {.out_dtype = out_dtype,
                            .out_dims = Dimensions({total}),
                            .op_param_cache_keys = std::move(param_keys)})));
        return MakeTensor(result);
      });
}

// Eager `PrivateUse1` kernel for `aten::repeat_interleave.Tensor_out`.
//
// Resizes `out` to `{total}` and dispatches `BuildRepeatInterleaveTensorShlo`
// via `DispatchOpOut<1>` to guarantee buffer donation when `out` is
// pre-allocated.
at::Tensor& AtenRepeatInterleaveTensorOut(
    const at::Tensor& repeats, const std::optional<c10::SymInt> output_size,
    at::Tensor& out) {
  const int64_t output_size_val =
      output_size.has_value() ? output_size->guard_int(__FILE__, __LINE__) : -1;
  TT_KERNEL(
      OpName::kRepeatInterleaveTensorOut, param_keys,
      (repeats, output_size_val, out), {
        TT_THROW_IF_ERROR(ValidateOutDtype(out, repeats.scalar_type(),
                                           /*allow_out_dtype_cast=*/false));
        const int64_t total = ValidateAndResolveRepeatInterleaveTensor(
            repeats, output_size, output_size_val);
        if (total == 0) {
          TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, {0}));
          return out;
        }

        TT_ASSIGN_OR_THROW(const mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(repeats.scalar_type()));
        auto shlo_builder =
            [total](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildRepeatInterleaveTensorShlo(input, total);
        };
        TT_THROW_IF_ERROR(
            (DispatchOpOut<1>(std::move(shlo_builder), repeats, out,
                              {.out_dtype = out_dtype,
                               .out_dims = Dimensions({total}),
                               .op_param_cache_keys = std::move(param_keys)})));
        return out;
      });
}

namespace {

// Lowers the backward pass (Vector-Jacobian Product / VJP) of uniform
// `repeat_interleave` to StableHLO using `Reshape` and `ReduceSum`.
//
// Mathematical Derivation of the Adjoint:
//   In the forward pass, uniform `repeat_interleave` along axis $d$ with repeat
//   factor $R$ is a linear operator $\mathcal{B}: \mathbb{R}^{\dots \times D
//   \times \dots}
//   \to \mathbb{R}^{\dots \times (D \cdot R) \times \dots}$ defined by:
//     $y_{\dots, i \cdot R + r, \dots} = x_{\dots, i, \dots}$  for $r \in \{0,
//     \dots, R-1\}$.
//   By the chain rule, given upstream gradient $\nabla_y$ of shape
//   `[... , D * R, ...]`, the gradient with respect to input $x$ is the adjoint
//   operator $\mathcal{B}^T(\nabla_y)$, which sums the $R$ replicated gradient
//   slices corresponding to each source element $i$:
//     $(\nabla_x)_{\dots, i, \dots} = \sum_{r=0}^{R-1} (\nabla_y)_{\dots, i
//     \cdot R + r, \dots}$.
//
// StableHLO Lowering Steps:
//   Case A: `resolved_dim` has a value $d \in [0, \text{rank}-1]$
//     Given `self_sizes = [B, T, H, D]`, `d = 2`, `repeats = R`, and upstream
//     gradient `grad_op` of shape `[B, T, H * R, D]`:
//       1. Split axis $d$ (size $H \cdot R$) into two adjacent axes `(H, R)` at
//          positions `(d, d + 1)` via `stablehlo.reshape`:
//            `inter_shape` = `[B, T, H, R, D]` (rank 5)
//       2. Sum across the inserted repeat axis (`reduce_dim = d + 1 = 3`) via
//          `BuildSumShlo`:
//            `reduced`     = `sum(reshaped, dim=3) -> [B, T, H, D]` (rank 4)
//
//   Case B: `resolved_dim` is `std::nullopt` (flattened 1D forward pass)
//     When `dim` was omitted in forward, `self` (of arbitrary shape
//     `self_sizes` with $N = \prod \text{self\_sizes}$ elements) was flattened
//     to 1D and repeated to produce a 1D output of length $N \cdot R$. Thus
//     `grad_op` has 1D shape `[N * R]`:
//       1. Reshape `grad_op` to 2D `inter_shape = [N, R]`.
//       2. Sum across axis `reduce_dim = 1` to obtain a 1D tensor of shape
//       `[N]`.
//       3. If `self` was not originally 1D (`self_sizes.size() != 1`), reshape
//          the 1D reduced tensor `[N]` back to `self_sizes`.
//
// Memory Savings vs. PyTorch Default Autograd:
//   PyTorch's composite backward for `unsqueeze -> expand -> clone -> reshape`
//   executes `view_backward -> clone_backward (allocates full [B, T, H*R, D]
//   contiguous copy) -> expand_backward (sum)`. By lowering directly to
//   `Reshape + ReduceSum` in StableHLO, XLA reduces `grad_op` in-place from
//   HBM/VMEM without allocating any intermediate `[B, T, H*R, D]` buffer copy,
//   cutting eager training backward peak memory by up to 50%.
absl::StatusOr<mlir::MlirOp> BuildRepeatInterleaveBackwardShlo(
    mlir::MlirOp grad_op, absl::Span<const int64_t> self_sizes,
    std::optional<int64_t> resolved_dim, int64_t repeats) {
  // Fast path: if `repeats == 0`, the gradient with respect to `self` is zero.
  if (repeats == 0) {
    return MakeConstant(grad_op.getBuilder(), 0,
                        GetTensorTypeOrDie(grad_op).getElementType(),
                        self_sizes);
  }
  // Fast path: if `repeats == 1`, forward was an identity map (or a 1D flatten
  // when `resolved_dim` was `std::nullopt`), so backward simply reshapes
  // `grad_op` back to `self_sizes`.
  if (repeats == 1) {
    return mlir::stablehlo::Reshape(grad_op, self_sizes);
  }
  llvm::SmallVector<int64_t> inter_shape;
  int64_t reduce_dim = 1;
  if (resolved_dim.has_value()) {
    // Case A: Split `resolved_dim` into `(self_sizes[*resolved_dim], repeats)`.
    inter_shape.reserve(self_sizes.size() + 1);
    for (int64_t i = 0; i < static_cast<int64_t>(self_sizes.size()); ++i) {
      inter_shape.push_back(self_sizes[i]);
      if (i == *resolved_dim) {
        inter_shape.push_back(repeats);
      }
    }
    reduce_dim = *resolved_dim + 1;
  } else {
    // Case B: Input was flattened to 1D of length `total_elems` before
    // repeating.
    int64_t total_elems = 1;
    for (int64_t s : self_sizes) {
      total_elems *= s;
    }
    inter_shape = {total_elems, repeats};
    reduce_dim = 1;
  }
  mlir::MlirOp reshaped = mlir::stablehlo::Reshape(grad_op, inter_shape);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp reduced,
                      BuildSumShlo(reshaped, {reduce_dim}));
  // If forward flattened a multi-dimensional input
  // (`!resolved_dim.has_value()`), restore the original tensor rank and shape
  // `self_sizes`.
  if (!resolved_dim.has_value() && self_sizes.size() != 1) {
    return mlir::stablehlo::Reshape(reduced, self_sizes);
  }
  return reduced;
}

// Dispatches `BuildRepeatInterleaveBackwardShlo` as a native TPU eager kernel.
//
// Caches compiled backward executables keyed by `grad_output` shape/dtype,
// target `self_sizes_ref`, `repeats_val`, and `resolved_dim`.
at::Tensor RepeatInterleaveBackwardKernel(const at::Tensor& grad_output,
                                          at::IntArrayRef self_sizes_ref,
                                          int64_t repeats_val,
                                          std::optional<int64_t> resolved_dim) {
  TT_KERNEL(
      OpName::kRepeatInterleaveBackward, param_keys,
      (grad_output, self_sizes_ref, repeats_val, resolved_dim), {
        TT_ASSIGN_OR_THROW(
            const mlir::ElementType out_dtype,
            ConvertTo<mlir::ElementType>(grad_output.scalar_type()));
        Dimensions out_dims(self_sizes_ref.begin(), self_sizes_ref.end());
        llvm::SmallVector<int64_t> self_sizes_vec(self_sizes_ref.begin(),
                                                  self_sizes_ref.end());
        auto shlo_builder =
            [self_sizes_vec, resolved_dim,
             repeats_val](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          return BuildRepeatInterleaveBackwardShlo(
              input, absl::MakeConstSpan(self_sizes_vec), resolved_dim,
              repeats_val);
        };
        TT_ASSIGN_OR_THROW(
            auto result,
            (DispatchOp<1>(std::move(shlo_builder), grad_output,
                           {.out_dtype = out_dtype,
                            .out_dims = out_dims,
                            .op_param_cache_keys = std::move(param_keys)})));
        return MakeTensor(result);
      });
}

// Custom autograd node (`torch::autograd::Function`) for uniform
// `repeat_interleave.self_int`.
//
// Zero-Activation-Memory Design:
//   Because uniform `repeat_interleave` is a linear operator whose Jacobian
//   depends only on tensor shapes (`self_sizes`), the scalar `repeats` factor,
//   and the target dimension `dim`, we do NOT call
//   `ctx->save_for_backward({self})`. Instead, we store only lightweight
//   `IValue` shape/scalar metadata in `ctx->saved_data`. This ensures zero
//   forward activation tensors are pinned in TPU HBM between the forward and
//   backward passes.
struct AtenRepeatInterleaveSelfIntAutogradNode
    : public torch::autograd::Function<
          AtenRepeatInterleaveSelfIntAutogradNode> {
  static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                            const at::Tensor& self, c10::SymInt repeats,
                            std::optional<int64_t> dim,
                            std::optional<c10::SymInt> output_size) {
    const int64_t repeats_val = repeats.guard_int(__FILE__, __LINE__);
    at::Tensor result =
        AtenRepeatInterleaveSelfInt(self, repeats, dim, output_size);
    // Save only lightweight shape and scalar metadata on the context.
    ctx->saved_data["self_sizes"] = self.sizes().vec();  // VEC_OK
    ctx->saved_data["repeats"] = repeats_val;
    ctx->saved_data["has_dim"] = dim.has_value();
    if (dim.has_value()) {
      ctx->saved_data["dim"] = c10::maybe_wrap_dim(*dim, self.dim());
    }
    return result;
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      std::vector<at::Tensor> grad_outputs) {
    const at::Tensor& grad_output = grad_outputs[0];
    const int64_t repeats_val = ctx->saved_data["repeats"].toInt();
    const std::vector<int64_t> self_sizes =  // INT_VEC_OK
        ctx->saved_data["self_sizes"].toIntVector();
    const bool has_dim = ctx->saved_data["has_dim"].toBool();
    std::optional<int64_t> resolved_dim = std::nullopt;
    if (has_dim) {
      resolved_dim = ctx->saved_data["dim"].toInt();
    }
    at::Tensor grad_self = RepeatInterleaveBackwardKernel(
        grad_output, self_sizes, repeats_val, resolved_dim);
    // Return gradients corresponding to the 4 inputs of `forward`:
    // `(self, repeats, dim, output_size)`. Only `self` requires a gradient;
    // non-differentiable scalar/metadata arguments receive undefined
    // `at::Tensor()`.
    return {grad_self, at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

}  // namespace

// Entry point for `AutogradPrivateUse1` registration of
// `aten::repeat_interleave.self_int`.
//
// Dispatch Decision Tree:
//   1. Tracing / Compilation (`IsFakeOrFunctorchTensor(self)`):
//      Delegates to `DecomposeRepeatInterleaveSelfInt` (`unsqueeze + expand +
//      reshape`) so TorchDynamo / AOTAutograd / functorch can trace symbolic
//      shapes and differentiate using standard view rules.
//   2. Inference / No-Grad (`!GradMode::is_enabled() ||
//   !self.requires_grad()`):
//      Bypasses `torch::autograd::Function::apply` overhead and invokes the
//      `PrivateUse1` forward kernel `AtenRepeatInterleaveSelfInt` directly.
//   3. Eager Training (`GradMode::is_enabled() && self.requires_grad()`):
//      Applies `AtenRepeatInterleaveSelfIntAutogradNode` to record the custom
//      backward node (`RepeatInterleaveBackwardKernel`) on the autograd tape.
at::Tensor AtenRepeatInterleaveSelfIntAutograd(
    const at::Tensor& self, c10::SymInt repeats, std::optional<int64_t> dim,
    std::optional<c10::SymInt> output_size) {
  if (IsFakeOrFunctorchTensor(self)) {
    return DecomposeRepeatInterleaveSelfInt(self, repeats, dim, output_size);
  }
  if (!c10::GradMode::is_enabled() || !self.requires_grad()) {
    return AtenRepeatInterleaveSelfInt(self, repeats, dim, output_size);
  }
  return AtenRepeatInterleaveSelfIntAutogradNode::apply(self, repeats, dim,
                                                        output_size);
}

namespace {

// Attempts to extract a scalar repeat count when `repeats` is a 1-element
// tensor. Returns `false` if `repeats` is multi-element or if its value cannot
// be determined without device synchronization during symbolic tracing.
// Returns an error if `dim` is out of range for `self` (any `dim` is for a
// 0-dim `self`).
// Takes `ConstTensor`s so that symbolic shapes are read without being
// specialized.
absl::StatusOr<bool> TryExtractScalarRepeatCount(
    const ConstTensor& self, const ConstTensor& repeats,
    std::optional<int64_t> dim, const std::optional<c10::SymInt>& output_size,
    c10::SymInt& repeats_val) {
  if (repeats.sym_numel() != 1) {
    return false;
  }
  if (output_size.has_value()) {
    c10::SymInt dim_size;
    if (dim.has_value()) {
      TT_ASSIGN_OR_RETURN(dim_size, self.sym_size(*dim));
    } else {
      dim_size = self.sym_numel();
    }
    repeats_val = dim_size > 0 ? (*output_size / dim_size) : c10::SymInt(0);
    return true;
  }
  if (!IsFakeOrFunctorchTensor(repeats)) {
    // Reading the values of `repeats` needs the underlying tensor. This only
    // happens for real (non-fake) data, whose shapes are concrete.
    const auto stats = ComputeRepeatsSumAndMinOnHost(repeats.UnsafeToTensor());
    if (!stats.ok()) {
      return false;
    }
    repeats_val = stats->sum;
    return true;
  }
  return false;
}

}  // namespace

// Entry point for `AutogradPrivateUse1` registration of
// `aten::repeat_interleave.self_Tensor`.
//
// Dispatch Decision Tree:
//   1. 1-Element Tensor Repeats (`repeats.numel() == 1`):
//      Extracts the scalar repeat count (using `output_size / dim_size` if
//      available to avoid device sync, or reading the scalar item if on eager
//      CPU/TPU) and delegates to `AtenRepeatInterleaveSelfIntAutograd`. This
//      upgrades both forward and backward passes from gather/scatter-add to
//      `BroadcastInDim` (forward) and `Reshape + ReduceSum` (backward).
//   2. Variable Multi-Element Repeats with Autograd or Tracing:
//      When `self.requires_grad()` is true in grad mode (or when tracing under
//      `FakeTensor` / `functorch`), decomposes into:
//        `repeat_indices = at::repeat_interleave(repeats, output_size)`
//        `output = input.index_select(d, repeat_indices)`
//      This allows PyTorch's built-in autograd engine to differentiate through
//      `index_select` (using `index_select_backward` / `index_add_`) without
//      requiring a custom scatter-add backward kernel.
//   3. Variable Multi-Element Repeats in Inference Mode:
//      Invokes the native `PrivateUse1` kernel `AtenRepeatInterleave`.
at::Tensor AtenRepeatInterleaveAutograd(
    const at::Tensor& self, const at::Tensor& repeats,
    std::optional<int64_t> dim, std::optional<c10::SymInt> output_size) {
  if ((c10::GradMode::is_enabled() && self.requires_grad()) ||
      IsFakeOrFunctorchTensor(self)) {
    c10::SymInt repeats_val = 0;
    TT_ASSIGN_OR_THROW(const bool has_scalar_repeat_count,
                       TryExtractScalarRepeatCount(self, repeats, dim,
                                                   output_size, repeats_val));
    if (has_scalar_repeat_count) {
      return AtenRepeatInterleaveSelfIntAutograd(self, repeats_val, dim,
                                                 output_size);
    }
    at::Tensor input = dim.has_value() ? self : self.flatten();
    const int64_t d = dim.value_or(0);
    at::Tensor repeat_indices =
        at::repeat_interleave_symint(repeats, output_size);
    return input.index_select(d, repeat_indices);
  }
  // `self` holds real data here, so `output_size` is a concrete value.
  return AtenRepeatInterleave(
      self, repeats, dim,
      output_size.has_value()
          ? std::optional<int64_t>(output_size->guard_int(__FILE__, __LINE__))
          : std::nullopt);
}

}  // namespace torch_tpu
