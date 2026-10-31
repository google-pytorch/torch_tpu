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

#include "csrc/ops/grouped_mm/grouped_mm_aten_kernels.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/BlasBackend.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/macro_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/index_select/index_select.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/precision_context.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

bool IsScalar(const at::Tensor& t) {
  return t.dim() == 0 || (t.dim() == 1 && t.size(0) == 1);
}

bool HasShape1D(const at::Tensor& t, int64_t d0) {
  return t.dim() == 1 && t.size(0) == d0;
}

bool HasShape2D(const at::Tensor& t, int64_t d0, int64_t d1) {
  return t.dim() == 2 && t.size(0) == d0 && t.size(1) == d1;
}

bool IsRowwiseScales2d2d(const at::Tensor& self, const at::Tensor& mat2,
                         const at::Tensor& scale_a, const at::Tensor& scale_b,
                         const std::optional<at::Tensor>& offs) {
  if (!offs.has_value() || !offs->defined() || offs->dim() != 1) {
    return false;
  }
  const int64_t g = offs->size(0);
  return HasShape1D(scale_a, g * self.size(0)) &&
         HasShape1D(scale_b, g * mat2.size(1));
}

// Returns true if scale_a and scale_b match the rowwise/per-group scale shapes
// for torch._scaled_grouped_mm:
//   2D x 3D: scale_a [M], scale_b [G, N]
//   3D x 2D: scale_a [G, M], scale_b [N]
//   2D x 2D: scale_a [G * M], scale_b [G * N]
//   3D x 3D: scale_a [G, M], scale_b [G, N]
bool IsRowwiseScales(const at::Tensor& self, const at::Tensor& mat2,
                     const at::Tensor& scale_a, const at::Tensor& scale_b,
                     const std::optional<at::Tensor>& offs) {
  if ((self.dim() != 2 && self.dim() != 3) ||
      (mat2.dim() != 2 && mat2.dim() != 3)) {
    return false;
  }
  const bool a_is_2d = self.dim() == 2;
  const bool b_is_2d = mat2.dim() == 2;
  if (a_is_2d && !b_is_2d) {
    // Case 1: 2D x 3D
    return HasShape1D(scale_a, self.size(0)) &&
           HasShape2D(scale_b, mat2.size(0), mat2.size(2));
  } else if (!a_is_2d && b_is_2d) {
    // Case 2: 3D x 2D
    return HasShape2D(scale_a, self.size(0), self.size(1)) &&
           HasShape1D(scale_b, mat2.size(1));
  } else if (a_is_2d && b_is_2d) {
    // Case 3: 2D x 2D
    return IsRowwiseScales2d2d(self, mat2, scale_a, scale_b, offs);
  } else {
    // Case 4: 3D x 3D
    return HasShape2D(scale_a, self.size(0), self.size(1)) &&
           HasShape2D(scale_b, mat2.size(0), mat2.size(2));
  }
}

// Maps token indices along a ragged dimension of size `ragged_size` to their
// corresponding group id in [0, G-1] based on cumulative offsets `offs_op` [G].
// Tokens beyond offs[G-1] clamp to G-1.
mlir::MlirOp MapRaggedTokensToGroup(mlir::MlirOp offs_op, int64_t ragged_size) {
  mlir::MlirBuilder& builder = offs_op.getBuilder();
  const auto offs_type = GetTensorTypeOrDie(offs_op);
  const int64_t g = offs_type.getShape()[0];
  const mlir::Type idx_type = offs_type.getElementType();

  auto iota = mlir::stablehlo::Iota(
      builder,
      mlir::makeTensorType(builder.getContext(), {ragged_size, g}, idx_type),
      0);
  auto offs_row = mlir::stablehlo::Reshape(offs_op, {1, g});
  auto offs_broadcasted = mlir::stablehlo::BroadcastInDim(
      mlir::makeTensorType(builder.getContext(), {ragged_size, g}, idx_type),
      offs_row, {0, 1});
  auto ge = mlir::stablehlo::Compare(iota, offs_broadcasted,
                                     mlir::stablehlo::ComparisonDirection::GE);
  auto ge_i = mlir::stablehlo::ConvertElementType(ge, idx_type);
  auto zero = MakeScalarConstant(builder, 0, idx_type);
  auto gid = mlir::stablehlo::Reduce(
      builder, {ge_i}, {zero},
      [idx_type](mlir::RegionBuilder& rb) {
        auto& body_builder = rb.getOpBuilder();
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
            idx_type, rb.getRegion(), body_builder);
      },
      {1})[0];
  auto gmax = MakeConstantLike(gid, g - 1);
  return mlir::stablehlo::Min(gid, gmax);
}

// Zeroes out rows (`ragged_dim = 0`) or columns (`ragged_dim = 1`) of 2D
// `scaled` at indices >= `offs_op[g - 1]`. CHLO RaggedDot leaves tokens past
// `offs[-1]` unmasked.
mlir::MlirOp MaskRaggedTailTokens(mlir::MlirOp scaled, mlir::MlirOp offs_op,
                                  int64_t ragged_dim) {
  mlir::MlirBuilder& builder = scaled.getBuilder();
  const auto scaled_type = GetTensorTypeOrDie(scaled);
  const auto offs_type = GetTensorTypeOrDie(offs_op);
  const int64_t g = offs_type.getShape()[0];
  mlir::MlirOp zero =
      MakeScalarConstant(builder, 0, scaled_type.getElementType());
  mlir::MlirOp zero_2d = mlir::stablehlo::BroadcastInDim(scaled_type, zero, {});
  if (g == 0) {
    return zero_2d;
  }
  const mlir::Type idx_type = offs_type.getElementType();
  const mlir::Type idx_2d_type = mlir::makeTensorType(
      builder.getContext(), scaled_type.getShape(), idx_type);
  mlir::MlirOp last_off =
      mlir::stablehlo::Slice(offs_op, {g - 1}, {g}, {1});  // [1]
  mlir::MlirOp last_off_scalar = mlir::stablehlo::Reshape(last_off, {});
  mlir::MlirOp last_off_2d =
      mlir::stablehlo::BroadcastInDim(idx_2d_type, last_off_scalar, {});
  mlir::MlirOp iota = mlir::stablehlo::Iota(builder, idx_2d_type, ragged_dim);
  mlir::MlirOp valid_mask = mlir::stablehlo::Compare(
      iota, last_off_2d, mlir::stablehlo::ComparisonDirection::LT);
  return mlir::stablehlo::Select(valid_mask, scaled, zero_2d);
}

// Lowers rowwise/per-group scale multiplication for grouped_mm result.
mlir::MlirOp ApplyRowwiseGroupScales(mlir::MlirOp mm_result,
                                     mlir::MlirOp scale_a, mlir::MlirOp scale_b,
                                     std::optional<mlir::MlirOp> offs_op,
                                     bool a_is_2d, bool b_is_2d) {
  mlir::MlirBuilder& builder = mm_result.getBuilder();
  const auto mm_type = GetTensorTypeOrDie(mm_result);

  const mlir::Type elem_a = GetTensorTypeOrDie(scale_a).getElementType();
  const mlir::Type elem_b = GetTensorTypeOrDie(scale_b).getElementType();

  if (a_is_2d && !b_is_2d) {
    // Case 1 (2D x 3D): mm_result [M, N], scale_a [M], scale_b [G, N]
    ABSL_CHECK(offs_op.has_value());  // CRASH_OK
    if (GetTensorTypeOrDie(*offs_op).getShape()[0] == 0) {
      return MaskRaggedTailTokens(mm_result, *offs_op, /*ragged_dim=*/0);
    }
    const int64_t m = mm_type.getShape()[0];
    const int64_t n = mm_type.getShape()[1];
    mlir::MlirOp gid = MapRaggedTokensToGroup(*offs_op, m);
    mlir::MlirOp sb_rows = BuildIndexSelectShlo(scale_b, 0, gid);  // [M, N]
    mlir::MlirOp sa_reshaped = mlir::stablehlo::Reshape(scale_a, {m, 1});
    mlir::MlirOp sa_bcast = mlir::stablehlo::BroadcastInDim(
        mlir::makeTensorType(builder.getContext(), {m, n}, elem_a), sa_reshaped,
        {0, 1});
    mlir::MlirOp combined = mlir::stablehlo::Mul(sa_bcast, sb_rows);
    mlir::MlirOp scaled = mlir::stablehlo::Mul(mm_result, combined);
    return MaskRaggedTailTokens(scaled, *offs_op, /*ragged_dim=*/0);
  } else if (!a_is_2d && b_is_2d) {
    // Case 2 (3D x 2D): mm_result [M, N], scale_a [G, M], scale_b [N]
    ABSL_CHECK(offs_op.has_value());  // CRASH_OK
    if (GetTensorTypeOrDie(*offs_op).getShape()[0] == 0) {
      return MaskRaggedTailTokens(mm_result, *offs_op, /*ragged_dim=*/1);
    }
    const int64_t m = mm_type.getShape()[0];
    const int64_t n = mm_type.getShape()[1];
    mlir::MlirOp gid = MapRaggedTokensToGroup(*offs_op, n);
    mlir::MlirOp sa_cols = BuildIndexSelectShlo(scale_a, 0, gid);  // [N, M]
    mlir::MlirOp sa_rows =
        mlir::stablehlo::Transpose(sa_cols, {1, 0});  // [M, N]
    mlir::MlirOp sb_reshaped = mlir::stablehlo::Reshape(scale_b, {1, n});
    mlir::MlirOp sb_bcast = mlir::stablehlo::BroadcastInDim(
        mlir::makeTensorType(builder.getContext(), {m, n}, elem_b), sb_reshaped,
        {0, 1});
    mlir::MlirOp combined = mlir::stablehlo::Mul(sa_rows, sb_bcast);
    mlir::MlirOp scaled = mlir::stablehlo::Mul(mm_result, combined);
    return MaskRaggedTailTokens(scaled, *offs_op, /*ragged_dim=*/1);
  } else {
    // Cases 3 & 4 (2D x 2D and 3D x 3D): mm_result [G, M, N],
    // scale_a [G*M] or [G, M], scale_b [G*N] or [G, N].
    const int64_t g = mm_type.getShape()[0];
    const int64_t m = mm_type.getShape()[1];
    const int64_t n = mm_type.getShape()[2];
    mlir::MlirOp sa_3d = mlir::stablehlo::Reshape(scale_a, {g, m, 1});
    mlir::MlirOp sb_3d = mlir::stablehlo::Reshape(scale_b, {g, 1, n});
    mlir::MlirOp sa_bcast = mlir::stablehlo::BroadcastInDim(
        mlir::makeTensorType(builder.getContext(), {g, m, n}, elem_a), sa_3d,
        {0, 1, 2});
    mlir::MlirOp sb_bcast = mlir::stablehlo::BroadcastInDim(
        mlir::makeTensorType(builder.getContext(), {g, m, n}, elem_b), sb_3d,
        {0, 1, 2});
    mlir::MlirOp combined = mlir::stablehlo::Mul(sa_bcast, sb_bcast);
    return mlir::stablehlo::Mul(mm_result, combined);
  }
}

bool UnwrapTransposed3dMat2(mlir::MlirOp& mat2_op) {
  if (GetTensorTypeOrDie(mat2_op).getRank() != 3) {
    return false;
  }
  auto transpose_op =
      mat2_op.getValue().getDefiningOp<mlir::stablehlo::TransposeOp>();
  if (!transpose_op) {
    return false;
  }
  const absl::Span<const int64_t> perm(transpose_op.getPermutation().data(),
                                       transpose_op.getPermutation().size());
  if (perm.size() == 3 && perm[0] == 0 && perm[1] == 2 && perm[2] == 1) {
    mat2_op = mlir::MlirOp(mat2_op.getBuilder(), transpose_op.getOperand());
    return true;
  }
  return false;
}

mlir::MlirOp ComputeGroupSizesFromOffsets(mlir::MlirOp offs_op) {
  mlir::MlirBuilder& builder = offs_op.getBuilder();
  if (!GetTensorTypeOrDie(offs_op).getElementType().isInteger(32)) {
    offs_op = mlir::stablehlo::ConvertElementType(
        offs_op, builder.getOpBuilder().getI32Type());
  }
  const mlir::RankedTensorType offs_type = GetTensorTypeOrDie(offs_op);
  const int64_t num_groups = offs_type.getShape()[0];
  mlir::MlirOp zero_op =
      MakeConstant(builder, 0, offs_type.getElementType(), {1});
  const Strides strides(1, 1);
  const Indices start_indices(1, 0);
  const Indices limit_indices(1, std::max<int64_t>(0, num_groups - 1));
  mlir::MlirOp offs_slice =
      mlir::stablehlo::Slice(offs_op, start_indices, limit_indices, strides);
  mlir::MlirOp concat_op =
      mlir::stablehlo::Concatenate(builder, {zero_op, offs_slice}, 0);
  return mlir::stablehlo::Subtract(offs_op, concat_op);
}

mlir::MlirOp BuildRaggedDot2d3d(mlir::MlirOp self_op, mlir::MlirOp mat2_op,
                                mlir::MlirOp offs_op, bool transpose_mat2,
                                mlir::ElementType comp_dtype,
                                mlir::ArrayAttr precision_config) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
  const mlir::RankedTensorType mat2_type = GetTensorTypeOrDie(mat2_op);
  mlir::MlirOp group_sizes = ComputeGroupSizesFromOffsets(offs_op);
  const int64_t rhs_contracting_dim = transpose_mat2 ? 2 : 1;
  const int64_t rhs_out_dim = transpose_mat2 ? 1 : 2;
  const mlir::chlo::RaggedDotDimensionNumbersAttr dimension_numbers =
      mlir::chlo::RaggedDotDimensionNumbersAttr::get(
          &self_op.getContext(), /*lhsBatchingDimensions=*/{},
          /*rhsBatchingDimensions=*/{},
          /*lhsContractingDimensions=*/{1},
          /*rhsContractingDimensions=*/{rhs_contracting_dim},
          /*lhsRaggedDimensions=*/{0},
          /*rhsGroupDimensions=*/{0});
  const mlir::Type out_type = mlir::makeTensorType(
      self_op.getContext(),
      {self_type.getShape()[0], mat2_type.getShape()[rhs_out_dim]}, comp_dtype);
  return mlir::chlo::RaggedDot(out_type, self_op, mat2_op, group_sizes,
                               dimension_numbers, precision_config);
}

mlir::MlirOp BuildBatchedDot3d3d(mlir::MlirOp self_op, mlir::MlirOp mat2_op,
                                 bool transpose_mat2,
                                 mlir::stablehlo::Precision precision) {
  auto precision_config = mlir::stablehlo::PrecisionConfigAttr::get(
      &self_op.getContext(), {precision, precision});
  const int64_t rhs_contracting_dim = transpose_mat2 ? 2 : 1;
  const mlir::stablehlo::DotDimensionNumbersAttr dot_dimension_numbers =
      mlir::stablehlo::DotDimensionNumbersAttr::get(
          &self_op.getContext(),
          /*lhs_batch_dimensions=*/{0},
          /*rhs_batch_dimensions=*/{0},
          /*lhs_contracting_dimensions=*/{2},
          /*rhs_contracting_dimensions=*/{rhs_contracting_dim});
  return mlir::stablehlo::DotGeneral(self_op, mat2_op, dot_dimension_numbers,
                                     precision_config);
}

// XLA:TPU's grouped matmul (Gmm) kernel tiles the ragged token dimension by
// gcd(T, 256), so Case 3 pads T to a multiple of this to get full tiles.
constexpr int64_t kRaggedTokenTile = 256;

// Returns rank-2 `op` with `num_rows` zero rows appended to dimension 0.
mlir::MlirOp AppendZeroRows(mlir::MlirOp op, int64_t num_rows) {
  mlir::MlirOp zero = MakeScalarConstant(
      op.getBuilder(), 0, GetTensorTypeOrDie(op).getElementType());
  return mlir::stablehlo::Pad(op, zero, /*edge_padding_low=*/Dimensions{0, 0},
                              /*edge_padding_high=*/Dimensions{num_rows, 0},
                              /*interior_padding=*/Dimensions{0, 0});
}

mlir::MlirOp BuildRaggedDot2d2d(mlir::MlirOp self_op, mlir::MlirOp mat2_op,
                                mlir::MlirOp offs_op,
                                mlir::ElementType comp_dtype,
                                mlir::ArrayAttr precision_config) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
  const mlir::RankedTensorType mat2_type = GetTensorTypeOrDie(mat2_op);
  const int64_t num_groups = GetTensorTypeOrDie(offs_op).getShape()[0];
  mlir::MlirOp group_sizes = ComputeGroupSizesFromOffsets(offs_op);

  // Put the ragged dimension first in both operands ([T, m] x [T, n]). This
  // is the grad_y form that XLA:TPU lowers to its Gmm kernel. With
  // lhs_ragged_dimensions={1}, XLA falls back to RaggedDotExpander, which
  // returns wrong values for groups inside the first contracting tile and
  // can fault on some T (github.com/google-pytorch/torch_tpu/issues/2668).
  mlir::MlirOp lhs = mlir::stablehlo::Transpose(self_op, {1, 0});
  mlir::MlirOp rhs = mat2_op;
  // Gmm also needs T to be a multiple of the sublane count, so zero-pad T to
  // a multiple of kRaggedTokenTile. The pad rows follow the last group, so
  // no group covers them. Bounded-dynamic T is not padded; if its bound is
  // not a multiple of the sublane count, XLA falls back to RaggedDotExpander
  // (#2668).
  // TODO(amainz): Pad bounded-dynamic T to a multiple of kRaggedTokenTile.
  if (!self_type.isDynamicDim(1) && !mat2_type.isDynamicDim(0)) {
    const int64_t num_tokens = self_type.getDimSize(1);
    const int64_t padded_tokens = std::max<int64_t>(
        kRaggedTokenTile, (num_tokens + kRaggedTokenTile - 1) /
                              kRaggedTokenTile * kRaggedTokenTile);
    const int64_t num_pad_rows = padded_tokens - num_tokens;
    if (num_pad_rows > 0) {
      lhs = AppendZeroRows(lhs, num_pad_rows);
      rhs = AppendZeroRows(rhs, num_pad_rows);
    }
  }

  const mlir::chlo::RaggedDotDimensionNumbersAttr dimension_numbers =
      mlir::chlo::RaggedDotDimensionNumbersAttr::get(
          &self_op.getContext(), /*lhsBatchingDimensions=*/{},
          /*rhsBatchingDimensions=*/{},
          /*lhsContractingDimensions=*/{0},
          /*rhsContractingDimensions=*/{0},
          /*lhsRaggedDimensions=*/{0},
          /*rhsGroupDimensions=*/{});

  const auto out_type = mlir::makeTensorType(
      self_op.getContext(),
      {num_groups, self_type.getShape()[0], mat2_type.getShape()[1]},
      comp_dtype);
  return mlir::chlo::RaggedDot(out_type, lhs, rhs, group_sizes,
                               dimension_numbers, precision_config);
}

// Lowers grouped matmul to MLIR StableHLO/CHLO.
//
// Dimension Notation (mirrors CHLO_RaggedDotOp):
//   b: batch dimension (expert batch size in MoE)
//   m: lhs non-contracting dimension (tokens per expert)
//   k: contracting dimension (input channels / hidden size)
//   n: rhs non-contracting dimension (output channels / projection size)
//   g: group dimension (expert count / number of partitions along ragged dim)
//
// Lowering Cases:
// - Case 1 (2D x 3D with offs):
//   Input:  self [m_total, k], mat2 [g, k, n], offs [g]
//   Output: [m_total, n]
//   Maps to RaggedDot Mode 1 (lhs non-contracting dim 'm' is ragged).
//   The leading dimension 'm_total' is partitioned into g groups [m_i, ...].
//
// - Case 2 (3D x 2D with offs):
//   Input:  self [g, b, k], mat2 [k, Sum(n_i)], offs [g]
//   Output: [b, Sum(n_i)]
//   Each expert i has output channel size n_i.
//   We transpose to map to RaggedDot Mode 1, where mat2 is the lhs and self is
//   the rhs in RaggedDot:
//     mat2 -> [Sum(n_i), k] (partitioned along dim 0 into g groups of [n_i, k])
//     self -> [g, k, b]
//   RaggedDot produces [Sum(n_i), b], which is transposed back to [b, Sum(n_i)]
//
// - Case 3 (2D x 2D with offs):
//   Input:  self [m, Sum(k_i)], mat2 [Sum(k_i), n], offs [g]
//   Output: [g, m, n]
//   Maps to RaggedDot Mode 2 (Contracting dim 'k' is ragged across g groups).
//   Self's trailing dim (dim 1) and mat2's leading dim (dim 0) share the
//   same ragged partitioning along Sum(k_i). Self is transposed to
//   [Sum(k_i), m] so the ragged dim is dim 0 of both operands, and Sum(k_i) is
//   zero-padded to a multiple of kRaggedTokenTile.
//
// - Case 4 (3D x 3D without offs):
//   Input:  self [b, m, k], mat2 [b, k, n]
//   Output: [b, m, n]
//   Degrades to standard batched matrix multiplication (DotGeneral).
absl::StatusOr<mlir::MlirOp> BuildGroupedMmShlo(
    mlir::MlirOp self_op, mlir::MlirOp mat2_op,
    std::optional<mlir::MlirOp> offs_op, mlir::ElementType out_dtype,
    mlir::stablehlo::Precision precision) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
  const bool transpose_mat2 = UnwrapTransposed3dMat2(mat2_op);
  const mlir::RankedTensorType mat2_type = GetTensorTypeOrDie(mat2_op);

  const bool a_is_2d = self_type.getRank() == 2;
  const bool b_is_2d = mat2_type.getRank() == 2;

  // Upcast input tensors to the computation precision (e.g., F32 or BF16)
  // prior to RaggedDot/DotGeneral because CHLO RaggedDot does not natively
  // handle sub-byte types like FP4/MXFP8 or mixed precision without explicit
  // casts, and TPU Mosaic RaggedDot fails to legalize F16 pack_subelements.
  // Cases 1-3 keep FP8 x FP8 operands: Gmm consumes f8 natively on the MXU.
  const mlir::ElementType comp_dtype = (out_dtype == mlir::ElementType::F16)
                                           ? mlir::ElementType::F32
                                           : out_dtype;
  const auto is_f8 = [](mlir::Type t) {
    return mlir::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(t);
  };
  const bool is_ragged_dot = a_is_2d || b_is_2d;
  const bool skip_upcast = is_f8(self_type.getElementType()) &&
                           is_f8(mat2_type.getElementType()) && is_ragged_dot;
  if (!skip_upcast) {
    self_op = mlir::stablehlo::ConvertElementType(self_op, comp_dtype);
    mat2_op = mlir::stablehlo::ConvertElementType(mat2_op, comp_dtype);
  }

  // Gmm kernel requires DEFAULT precision when consuming native FP8 operands.
  const mlir::stablehlo::Precision effective_precision =
      skip_upcast
          ? mlir::stablehlo::Precision::DEFAULT  // EXPLICIT_PRECISION_OK
          : precision;
  const mlir::chlo::Precision chlo_precision =
      static_cast<mlir::chlo::Precision>(effective_precision);
  auto precision_attr =
      mlir::chlo::PrecisionAttr::get(&self_op.getContext(), chlo_precision);
  auto precision_config = mlir::ArrayAttr::get(
      &self_op.getContext(), {precision_attr, precision_attr});

  mlir::MlirOp result;
  if (a_is_2d && !b_is_2d) {
    // Case 1 (2D x 3D with offs): lhs first dimension is ragged (Mode 1).
    ABSL_CHECK(offs_op.has_value());  // CRASH_OK
    result = BuildRaggedDot2d3d(self_op, mat2_op, *offs_op, transpose_mat2,
                                comp_dtype, precision_config);
  } else if (!a_is_2d && b_is_2d) {
    // Case 2 (3D x 2D with offs): rhs tailing dimension is ragged (Mode 1).
    ABSL_CHECK(offs_op.has_value());  // CRASH_OK
    mlir::MlirOp group_sizes = ComputeGroupSizesFromOffsets(*offs_op);

    mlir::MlirOp mat2_transposed = mlir::stablehlo::Transpose(mat2_op, {1, 0});
    mlir::MlirOp self_transposed =
        mlir::stablehlo::Transpose(self_op, {0, 2, 1});

    const mlir::chlo::RaggedDotDimensionNumbersAttr dimension_numbers =
        mlir::chlo::RaggedDotDimensionNumbersAttr::get(
            &self_op.getContext(), /*lhsBatchingDimensions=*/{},
            /*rhsBatchingDimensions=*/{},
            /*lhsContractingDimensions=*/{1},
            /*rhsContractingDimensions=*/{1},
            /*lhsRaggedDimensions=*/{0},
            /*rhsGroupDimensions=*/{0});

    const mlir::Type ragged_out_type = mlir::makeTensorType(
        self_op.getContext(),
        {mat2_type.getShape()[1], self_type.getShape()[1]}, comp_dtype);
    mlir::MlirOp ragged_dot =
        mlir::chlo::RaggedDot(ragged_out_type, mat2_transposed, self_transposed,
                              group_sizes, dimension_numbers, precision_config);
    result = mlir::stablehlo::Transpose(ragged_dot, {1, 0});
  } else if (a_is_2d && b_is_2d) {
    // Case 3 (2D x 2D with offs): contracting dimension is ragged (Mode 2).
    // The trailing dimension of lhs and the leading dimension of rhs are
    // partitioned according to the same offs tensor.
    ABSL_CHECK(offs_op.has_value());  // CRASH_OK
    result = BuildRaggedDot2d2d(self_op, mat2_op, *offs_op, comp_dtype,
                                precision_config);
  } else {
    // Case 4 (3D x 3D without offs): fallback to standard batched matrix
    // multiplication (DotGeneral).
    result = BuildBatchedDot3d3d(self_op, mat2_op, transpose_mat2, precision);
  }
  if (comp_dtype != out_dtype) {
    result = mlir::stablehlo::ConvertElementType(result, out_dtype);
  }
  return result;
}

bool IsScalarScaleOp(mlir::MlirOp scale_op) {
  const auto scale_type = GetTensorTypeOrDie(scale_op);
  return scale_type.getRank() == 0 ||
         (scale_type.getRank() == 1 && scale_type.getShape()[0] == 1);
}

absl::StatusOr<mlir::MlirOp> ApplyScalarGroupScales(mlir::MlirOp mm_result,
                                                    mlir::MlirOp scale_a_f32,
                                                    mlir::MlirOp scale_b_f32) {
  TT_ASSIGN_OR_RETURN(auto broadcasted_scales,
                      ApplyBroadcastIfNeeded<2>({scale_a_f32, scale_b_f32}));
  mlir::MlirOp combined_scale =
      mlir::stablehlo::Mul(broadcasted_scales[0], broadcasted_scales[1]);
  TT_ASSIGN_OR_RETURN(auto broadcasted_mm_scale,
                      ApplyBroadcastIfNeeded<2>({mm_result, combined_scale}));
  return mlir::stablehlo::Mul(broadcasted_mm_scale[0], broadcasted_mm_scale[1]);
}

absl::StatusOr<mlir::MlirOp> BuildScaledGroupedMmShlo(
    mlir::MlirOp self_op, mlir::MlirOp mat2_op, mlir::MlirOp scale_a,
    mlir::MlirOp scale_b, std::optional<mlir::MlirOp> offs_op,
    std::optional<mlir::MlirOp> bias, std::optional<mlir::MlirOp> scale_result,
    mlir::ElementType out_dtype, mlir::stablehlo::Precision precision) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
  const mlir::RankedTensorType mat2_type = GetTensorTypeOrDie(mat2_op);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType self_elem_type,
                      ConvertTo<mlir::ElementType>(self_type.getElementType()));

  // For sub-byte types like FP4 or MXFP8, computation precision is typically
  // elevated to F32 to preserve accuracy and because XLA does not support
  // sub-byte math directly. Standard floating types (BF16, F32) usually map to
  // themselves.
  TT_ASSIGN_OR_RETURN(const mlir::ElementType comp_dtype,
                      InferComputationDtype(self_elem_type));

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp mm_result,
      BuildGroupedMmShlo(self_op, mat2_op, offs_op, comp_dtype, precision));
  TT_ASSIGN_OR_RETURN(mm_result,
                      CastIfNeeded(mm_result, mlir::ElementType::F32));

  TT_ASSIGN_OR_RETURN(auto scale_a_f32,
                      CastIfNeeded(scale_a, mlir::ElementType::F32));
  TT_ASSIGN_OR_RETURN(auto scale_b_f32,
                      CastIfNeeded(scale_b, mlir::ElementType::F32));

  const bool a_is_2d = self_type.getRank() == 2;
  const bool b_is_2d = mat2_type.getRank() == 2;
  const bool is_scalar_scale =
      IsScalarScaleOp(scale_a) && IsScalarScaleOp(scale_b);

  mlir::MlirOp scaled_result;
  if (!is_scalar_scale) {
    scaled_result = ApplyRowwiseGroupScales(mm_result, scale_a_f32, scale_b_f32,
                                            offs_op, a_is_2d, b_is_2d);
  } else {
    TT_ASSIGN_OR_RETURN(
        scaled_result,
        ApplyScalarGroupScales(mm_result, scale_a_f32, scale_b_f32));
  }

  if (bias.has_value()) {
    TT_ASSIGN_OR_RETURN(auto bias_f32,
                        CastIfNeeded(*bias, mlir::ElementType::F32));
    TT_ASSIGN_OR_RETURN(auto broadcasted_bias,
                        ApplyBroadcastIfNeeded<2>({scaled_result, bias_f32}));
    scaled_result =
        mlir::stablehlo::Add(broadcasted_bias[0], broadcasted_bias[1]);
  }

  if (scale_result.has_value()) {
    TT_ASSIGN_OR_RETURN(auto scale_result_f32,
                        CastIfNeeded(*scale_result, mlir::ElementType::F32));
    TT_ASSIGN_OR_RETURN(
        auto broadcasted_scale_result,
        ApplyBroadcastIfNeeded<2>({scaled_result, scale_result_f32}));
    scaled_result = mlir::stablehlo::Mul(broadcasted_scale_result[0],
                                         broadcasted_scale_result[1]);
  }

  scaled_result = mlir::stablehlo::ConvertElementType(scaled_result, out_dtype);
  return scaled_result;
}

// Validates the `_grouped_mm` inputs that can be checked without looking at
// the tensor data. Keep in sync with the TPU Meta kernel in
// _internal/compile/grouped_mm_meta.py so eager and compile agree.
absl::Status ValidateGroupedMmInputs(const at::Tensor& self,
                                     const at::Tensor& mat2,
                                     const std::optional<at::Tensor>& offs,
                                     const std::optional<at::Tensor>& bias,
                                     std::optional<at::ScalarType> out_dtype) {
  // In ragged grouped GEMM, adding bias inside the kernel requires
  // expensive dynamic indexing because each expert group has an irregular
  // token count. To optimize performance, PyTorch CUDA/CPU kernels
  // explicitly reject bias.
  // Eager models also perform out.add_(bias) externally.
  TT_RET_CHECK(!bias.has_value() || !bias->defined(),
               error::kPythonNotImplementedError)
      << "expected bias to be undefined, got defined tensor";

  TT_RET_CHECK(self.scalar_type() == mat2.scalar_type(),
               error::kInvalidArgument)
      << "expected self and mat2 to have the same dtype, got "
      << ToString(self.scalar_type()) << " and "
      << ToString(mat2.scalar_type());

  TT_RET_CHECK(self.scalar_type() == at::ScalarType::Float ||
                   self.scalar_type() == at::ScalarType::Half ||
                   self.scalar_type() == at::ScalarType::BFloat16,
               error::kInvalidArgument)
      << "expected self to be float32, float16, or bfloat16, got "
      << ToString(self.scalar_type());

  // Matches CUDA: output dtype must match `self`.
  const at::ScalarType target_dtype = out_dtype.value_or(self.scalar_type());
  TT_RET_CHECK(target_dtype == self.scalar_type(), error::kInvalidArgument)
      << "expected out_dtype to match self dtype "
      << ToString(self.scalar_type()) << ", got " << ToString(target_dtype);

  if (offs.has_value() && offs->defined()) {
    TT_RET_CHECK(offs->dim() == 1, error::kInvalidArgument)
        << "expected offs to be 1D, got " << offs->dim() << "D";
    // Unlike upstream (int32 only), int64 is accepted since `cumsum` of int
    // counts defaults to int64; the lowering narrows it to int32.
    TT_RET_CHECK(offs->scalar_type() == at::ScalarType::Int ||
                     offs->scalar_type() == at::ScalarType::Long,
                 error::kInvalidArgument)
        << "expected offs to be int32 or int64, got "
        << ToString(offs->scalar_type());
    TT_RET_CHECK(offs->stride(0) == 1, error::kInvalidArgument)
        << "expected offs to have stride 1 along dim 0, got "
        << offs->stride(0);
  }
  return absl::OkStatus();
}

absl::StatusOr<Dimensions> GetGroupedMmOutputSize(
    const at::Tensor& self, const at::Tensor& mat2,
    const std::optional<at::Tensor>& offs) {
  const bool a_is_2d = self.dim() == 2;
  const bool b_is_2d = mat2.dim() == 2;
  const bool has_offs = offs.has_value() && offs->defined();

  TT_RET_CHECK(self.dim() == 2 || self.dim() == 3, error::kInvalidArgument)
      << "expected self to be 2D or 3D, got " << self.dim() << "D";
  TT_RET_CHECK(mat2.dim() == 2 || mat2.dim() == 3, error::kInvalidArgument)
      << "expected mat2 to be 2D or 3D, got " << mat2.dim() << "D";

  if (a_is_2d || b_is_2d) {
    TT_RET_CHECK(has_offs, error::kInvalidArgument)
        << "offsets must be provided when at least one of self or mat2 is a 2D "
           "tensor";
  } else {
    TT_RET_CHECK(!has_offs, error::kInvalidArgument)
        << "offsets must not be provided when both self and mat2 are 3D "
           "tensors";
  }

  Dimensions out_size;
  if (a_is_2d) {
    if (b_is_2d) {
      out_size = {offs->size(0), self.size(0), mat2.size(1)};
    } else {
      TT_RET_CHECK(offs->size(0) == mat2.size(0), error::kInvalidArgument)
          << "expected offs batch size to match mat2 batch size, got "
          << offs->size(0) << " and " << mat2.size(0);
      out_size = {self.size(0), mat2.size(-1)};
    }
  } else {
    if (b_is_2d) {
      TT_RET_CHECK(offs->size(0) == self.size(0), error::kInvalidArgument)
          << "expected offs batch size to match self batch size, got "
          << offs->size(0) << " and " << self.size(0);
      out_size = {self.size(1), mat2.size(1)};
    } else {
      TT_RET_CHECK(self.size(0) == mat2.size(0), error::kInvalidArgument)
          << "expected self batch size to match mat2 batch size, got "
          << self.size(0) << " and " << mat2.size(0);
      out_size = {self.size(0), self.size(1), mat2.size(-1)};
    }
  }

  // Checked after batch sizes (upstream checks it first) so existing batch
  // errors are unchanged. Without it, a bad K crashes the lowering. Unlike
  // upstream, 2D x 2D (ragged K) is checked too.
  TT_RET_CHECK(self.size(-1) == mat2.size(-2), error::kInvalidArgument)
      << "expected the contraction dimension of self and mat2 to match, got "
      << self.size(-1) << " and " << mat2.size(-2);
  return out_size;
}

absl::StatusOr<DeviceBufferRef> GroupedMm(
    const at::Tensor& self, const at::Tensor& mat2,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    std::optional<at::ScalarType> out_dtype, OpParamCacheKeys& param_keys) {
  TT_RETURN_IF_ERROR(
      ValidateGroupedMmInputs(self, mat2, offs, bias, out_dtype));

  const bool has_offs = offs.has_value() && offs->defined();

  TT_ASSIGN_OR_RETURN(Dimensions out_dims,
                      GetGroupedMmOutputSize(self, mat2, offs));

  at::ScalarType target_dtype = out_dtype.value_or(self.scalar_type());
  TT_ASSIGN_OR_RETURN(mlir::ElementType output_elem_type,
                      ConvertTo<mlir::ElementType>(target_dtype));

  const auto current_precision = GetAndAddPrecisionTo(param_keys);

  std::vector<at::Tensor> inputs = {self, mat2};
  if (has_offs) {
    inputs.push_back(*offs);
  }

  auto op_builder =
      [output_elem_type, current_precision](
          absl::Span<mlir::MlirOp> inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
    auto self_op = inputs[0];
    auto mat2_op = inputs[1];
    std::optional<mlir::MlirOp> offs_op =
        inputs.size() > 2 ? std::make_optional(inputs[2]) : std::nullopt;
    TT_ASSIGN_OR_RETURN(
        auto result, BuildGroupedMmShlo(self_op, mat2_op, offs_op,
                                        output_elem_type, current_precision));
    return result;
  };

  TT_ASSIGN_OR_RETURN(
      auto result_buffer,
      DispatchOp<kDynamicSize>(std::move(op_builder), inputs,
                               {.out_dtype = output_elem_type,
                                .out_dims = out_dims,
                                .op_param_cache_keys = std::move(param_keys)}));
  return result_buffer;
}

absl::StatusOr<DeviceBufferRef> ScaledGroupedMm(
    const at::Tensor& self, const at::Tensor& mat2, const at::Tensor& scale_a,
    const at::Tensor& scale_b, const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& scale_result,
    std::optional<at::ScalarType> out_dtype, OpParamCacheKeys& param_keys) {
  const bool has_offs = offs.has_value() && offs->defined();
  const bool has_bias = bias.has_value() && bias->defined();
  const bool has_scale_result =
      scale_result.has_value() && scale_result->defined();

  TT_ASSIGN_OR_RETURN(Dimensions out_dims,
                      GetGroupedMmOutputSize(self, mat2, offs));

  const bool tensorwise = IsScalar(scale_a) && IsScalar(scale_b);
  const bool rowwise = IsRowwiseScales(self, mat2, scale_a, scale_b, offs);
  TT_RET_CHECK(tensorwise || rowwise, error::kPythonNotImplementedError)
      << "expected scale_a and scale_b to both be scalar/size-1 or match "
         "rowwise shapes for self "
      << ToString(self.sizes()) << " and mat2 " << ToString(mat2.sizes())
      << ", got scale_a " << ToString(scale_a.sizes()) << " and scale_b "
      << ToString(scale_b.sizes());
  if (has_bias) {
    TT_RET_CHECK(IsScalar(*bias), error::kPythonNotImplementedError)
        << "expected bias to be 1D or scalar, got " << bias->dim() << "D";
  }
  if (has_scale_result) {
    TT_RET_CHECK(IsScalar(*scale_result), error::kPythonNotImplementedError)
        << "expected scale_result to be 1D or scalar, got "
        << scale_result->dim() << "D";
  }

  const at::ScalarType target_dtype = out_dtype.value_or(self.scalar_type());
  TT_ASSIGN_OR_RETURN(mlir::ElementType output_elem_type,
                      ConvertTo<mlir::ElementType>(target_dtype));

  const auto current_precision = GetAndAddPrecisionTo(param_keys);

  std::vector<at::Tensor> inputs = {self, mat2, scale_a, scale_b};
  if (has_offs) inputs.push_back(*offs);
  if (has_bias) inputs.push_back(*bias);
  if (has_scale_result) inputs.push_back(*scale_result);

  auto op_builder =
      [output_elem_type, current_precision, has_offs, has_bias,
       has_scale_result](
          absl::Span<mlir::MlirOp> inputs,
          mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
    auto self_op = inputs[0];
    auto mat2_op = inputs[1];
    auto scale_a_op = inputs[2];
    auto scale_b_op = inputs[3];

    const size_t kOptionalArgStartInd = 4;
    size_t idx = kOptionalArgStartInd;
    auto offs_op = has_offs ? std::make_optional(inputs[idx++]) : std::nullopt;
    auto bias_op = has_bias ? std::make_optional(inputs[idx++]) : std::nullopt;
    auto scale_result_op =
        has_scale_result ? std::make_optional(inputs[idx++]) : std::nullopt;

    TT_ASSIGN_OR_RETURN(
        auto result,
        BuildScaledGroupedMmShlo(self_op, mat2_op, scale_a_op, scale_b_op,
                                 offs_op, bias_op, scale_result_op,
                                 output_elem_type, current_precision));
    return result;
  };

  TT_ASSIGN_OR_RETURN(
      auto result_buffer,
      DispatchOp<kDynamicSize>(std::move(op_builder), inputs,
                               {.out_dtype = output_elem_type,
                                .out_dims = out_dims,
                                .op_param_cache_keys = std::move(param_keys)}));
  return result_buffer;
}

#if TT_TORCH_VERSION_GE(2, 14)
absl::Status ValidateScaledGroupedMmV2Inputs(
    const at::Tensor& self, const at::Tensor& mat2,
    const at::ITensorListRef& scale_a, at::IntArrayRef recipe_a,
    at::IntArrayRef swizzle_a, const at::ITensorListRef& scale_b,
    at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b,
    const std::optional<at::Tensor>& offs, at::IntArrayRef contraction_dim) {
#else
absl::Status ValidateScaledGroupedMmV2Inputs(
    const at::Tensor& self, const at::Tensor& mat2, at::TensorList scale_a,
    at::IntArrayRef recipe_a, at::IntArrayRef swizzle_a, at::TensorList scale_b,
    at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b,
    const std::optional<at::Tensor>& offs, at::IntArrayRef contraction_dim) {
#endif
  if (!contraction_dim.empty()) {
    TT_RET_CHECK(contraction_dim.size() == 2, error::kInvalidArgument)
        << "expected contraction_dim list to contain exactly 2 elements, got "
        << contraction_dim.size();
    // Like upstream, only (-1, -2) is supported: the last dim of self and the
    // second-to-last dim of mat2.
    const int64_t dim_a = contraction_dim[0] < 0
                              ? contraction_dim[0] + self.dim()
                              : contraction_dim[0];
    const int64_t dim_b = contraction_dim[1] < 0
                              ? contraction_dim[1] + mat2.dim()
                              : contraction_dim[1];
    TT_RET_CHECK(dim_a == self.dim() - 1 && dim_b == mat2.dim() - 2,
                 error::kPythonNotImplementedError)
        << "expected contraction_dim to be [-1, -2], got ["
        << contraction_dim[0] << ", " << contraction_dim[1] << "]";
  }

  TT_RET_CHECK(scale_a.size() == 1, error::kInvalidArgument)
      << "expected scale_a list to contain exactly 1 tensor, got "
      << scale_a.size();
  TT_RET_CHECK(scale_b.size() == 1, error::kInvalidArgument)
      << "expected scale_b list to contain exactly 1 tensor, got "
      << scale_b.size();

  TT_RET_CHECK(recipe_a.size() == 1, error::kInvalidArgument)
      << "expected recipe_a list to contain exactly 1 recipe, got "
      << recipe_a.size();
  TT_RET_CHECK(recipe_b.size() == 1, error::kInvalidArgument)
      << "expected recipe_b list to contain exactly 1 recipe, got "
      << recipe_b.size();

  constexpr auto kTensorWise =
      static_cast<int64_t>(at::blas::ScalingType::TensorWise);
  constexpr auto kRowWise =
      static_cast<int64_t>(at::blas::ScalingType::RowWise);
  TT_RET_CHECK(recipe_a[0] == kTensorWise || recipe_a[0] == kRowWise,
               error::kPythonNotImplementedError)
      << "expected scaling recipe for scale_a to be TensorWise or RowWise, got "
      << recipe_a[0];
  TT_RET_CHECK(recipe_b[0] == kTensorWise || recipe_b[0] == kRowWise,
               error::kPythonNotImplementedError)
      << "expected scaling recipe for scale_b to be TensorWise or RowWise, got "
      << recipe_b[0];
  TT_RET_CHECK(recipe_a[0] == recipe_b[0], error::kPythonNotImplementedError)
      << "expected recipe_a and recipe_b to match, got " << recipe_a[0]
      << " and " << recipe_b[0];

  const at::Tensor& s_a = scale_a.front();
  const at::Tensor& s_b = scale_b.front();
  const bool scale_matches_recipe =
      recipe_a[0] == kTensorWise ? (IsScalar(s_a) && IsScalar(s_b))
                                 : IsRowwiseScales(self, mat2, s_a, s_b, offs);
  TT_RET_CHECK(scale_matches_recipe, error::kInvalidArgument)
      << "expected scale_a and scale_b shapes to match scaling recipe "
      << recipe_a[0] << ", got scale_a " << ToString(s_a.sizes())
      << " and scale_b " << ToString(s_b.sizes());

  // An empty swizzle list means NO_SWIZZLE (the
  // torch.nn.functional.scaled_grouped_mm default).
  TT_RET_CHECK(swizzle_a.size() <= 1, error::kInvalidArgument)
      << "expected swizzle_a list to contain at most 1 swizzle mode, got "
      << swizzle_a.size();
  TT_RET_CHECK(swizzle_b.size() <= 1, error::kInvalidArgument)
      << "expected swizzle_b list to contain at most 1 swizzle mode, got "
      << swizzle_b.size();
  constexpr auto kNoSwizzle =
      static_cast<int64_t>(at::blas::SwizzleType::NO_SWIZZLE);
  TT_RET_CHECK(swizzle_a.empty() || swizzle_a[0] == kNoSwizzle,
               error::kPythonNotImplementedError)
      << "expected swizzle_a to be NO_SWIZZLE, got " << swizzle_a[0];
  TT_RET_CHECK(swizzle_b.empty() || swizzle_b[0] == kNoSwizzle,
               error::kPythonNotImplementedError)
      << "expected swizzle_b to be NO_SWIZZLE, got " << swizzle_b[0];

  return absl::OkStatus();
}

// Unpacks v2 args and delegates to the existing v1 ScaledGroupedMm path.
#if TT_TORCH_VERSION_GE(2, 14)
absl::StatusOr<DeviceBufferRef> ScaledGroupedMmV2(
    const at::Tensor& self, const at::Tensor& mat2,
    const at::ITensorListRef& scale_a, at::IntArrayRef recipe_a,
    at::IntArrayRef swizzle_a, const at::ITensorListRef& scale_b,
    at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    std::optional<at::ScalarType> out_dtype, at::IntArrayRef contraction_dim,
    OpParamCacheKeys param_keys) {
#else
absl::StatusOr<DeviceBufferRef> ScaledGroupedMmV2(
    const at::Tensor& self, const at::Tensor& mat2, at::TensorList scale_a,
    at::IntArrayRef recipe_a, at::IntArrayRef swizzle_a, at::TensorList scale_b,
    at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    std::optional<at::ScalarType> out_dtype, at::IntArrayRef contraction_dim,
    OpParamCacheKeys param_keys) {
#endif
  TT_RETURN_IF_ERROR(ValidateScaledGroupedMmV2Inputs(
      self, mat2, scale_a, recipe_a, swizzle_a, scale_b, recipe_b, swizzle_b,
      offs, contraction_dim));

  // Extract the single scale tensor from each list.
  const at::Tensor& s_a = scale_a.front();
  const at::Tensor& s_b = scale_b.front();

  // Delegate to the v1 path. scale_result is not part of v2.
  return ScaledGroupedMm(self, mat2, s_a, s_b, offs, bias,
                         /*scale_result=*/std::nullopt, out_dtype, param_keys);
}

}  // namespace

at::Tensor AtenGroupedMm(const at::Tensor& self, const at::Tensor& mat2,
                         const std::optional<at::Tensor>& offs,
                         const std::optional<at::Tensor>& bias,
                         std::optional<at::ScalarType> out_dtype) {
  TT_KERNEL(OpName::kGroupedMm, param_keys, (self, mat2, offs, bias, out_dtype),
            {
              TT_ASSIGN_OR_THROW(
                  auto result_buffer,
                  GroupedMm(self, mat2, offs, bias, out_dtype, param_keys));
              return MakeTensor(result_buffer);
            });
}

at::Tensor AtenScaledGroupedMm(const at::Tensor& self, const at::Tensor& mat2,
                               const at::Tensor& scale_a,
                               const at::Tensor& scale_b,
                               const std::optional<at::Tensor>& offs,
                               const std::optional<at::Tensor>& bias,
                               const std::optional<at::Tensor>& scale_result,
                               std::optional<at::ScalarType> out_dtype,
                               bool use_fast_accum) {
  TT_KERNEL(OpName::kScaledGroupedMm, param_keys,
            (self, mat2, scale_a, scale_b, offs, bias, scale_result, out_dtype,
             IgnoreInCacheKey(use_fast_accum, "Doesn't affect SHLO")),
            {
              // use_fast_accum is accepted and ignored: the MXU always
              // accumulates in F32.
              TT_ASSIGN_OR_THROW(
                  auto result_buffer,
                  ScaledGroupedMm(self, mat2, scale_a, scale_b, offs, bias,
                                  scale_result, out_dtype, param_keys));
              return MakeTensor(result_buffer);
            });
}

#if TT_TORCH_VERSION_GE(2, 14)  // See grouped_mm_aten_kernels.h.
at::Tensor AtenScaledGroupedMmV2(
    const at::Tensor& self, const at::Tensor& mat2,
    const at::ITensorListRef& scale_a, at::IntArrayRef recipe_a,
    at::IntArrayRef swizzle_a, const at::ITensorListRef& scale_b,
    at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    std::optional<at::ScalarType> out_dtype, at::IntArrayRef contraction_dim,
    bool use_fast_accum) {
#else
at::Tensor AtenScaledGroupedMmV2(
    const at::Tensor& self, const at::Tensor& mat2, at::TensorList scale_a,
    at::IntArrayRef recipe_a, at::IntArrayRef swizzle_a, at::TensorList scale_b,
    at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    std::optional<at::ScalarType> out_dtype, at::IntArrayRef contraction_dim,
    bool use_fast_accum) {
#endif
  TT_KERNEL(
      OpName::kScaledGroupedMmV2, param_keys,
      (self, mat2, scale_a,
       IgnoreInCacheKey(recipe_a, "Validated upfront; doesn't affect SHLO"),
       IgnoreInCacheKey(swizzle_a, "Validated upfront; doesn't affect SHLO"),
       scale_b,
       IgnoreInCacheKey(recipe_b, "Validated upfront; doesn't affect SHLO"),
       IgnoreInCacheKey(swizzle_b, "Validated upfront; doesn't affect SHLO"),
       offs, bias, out_dtype,
       IgnoreInCacheKey(contraction_dim,
                        "Validated upfront; doesn't affect SHLO"),
       IgnoreInCacheKey(use_fast_accum, "Doesn't affect SHLO")),
      {
        // use_fast_accum is accepted and ignored: the MXU always
        // accumulates in F32.
        TT_ASSIGN_OR_THROW(
            auto result_buffer,
            ScaledGroupedMmV2(self, mat2, scale_a, recipe_a, swizzle_a, scale_b,
                              recipe_b, swizzle_b, offs, bias, out_dtype,
                              contraction_dim, std::move(param_keys)));
        return MakeTensor(result_buffer);
      });
}

#if TT_TORCH_VERSION_GE(2, 14)  // See grouped_mm_aten_kernels.h.
at::Tensor& AtenScaledGroupedMmV2Out(
    const at::Tensor& self, const at::Tensor& mat2,
    const at::ITensorListRef& scale_a, at::IntArrayRef recipe_a,
    at::IntArrayRef swizzle_a, const at::ITensorListRef& scale_b,
    at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    std::optional<at::ScalarType> out_dtype, at::IntArrayRef contraction_dim,
    bool use_fast_accum, at::Tensor& out) {
#else
at::Tensor& AtenScaledGroupedMmV2Out(
    const at::Tensor& self, const at::Tensor& mat2, at::TensorList scale_a,
    at::IntArrayRef recipe_a, at::IntArrayRef swizzle_a, at::TensorList scale_b,
    at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    std::optional<at::ScalarType> out_dtype, at::IntArrayRef contraction_dim,
    bool use_fast_accum, at::Tensor& out) {
#endif
  TT_KERNEL(
      OpName::kScaledGroupedMmV2Out, param_keys,
      (self, mat2, scale_a,
       IgnoreInCacheKey(recipe_a, "Validated upfront; doesn't affect SHLO"),
       IgnoreInCacheKey(swizzle_a, "Validated upfront; doesn't affect SHLO"),
       scale_b,
       IgnoreInCacheKey(recipe_b, "Validated upfront; doesn't affect SHLO"),
       IgnoreInCacheKey(swizzle_b, "Validated upfront; doesn't affect SHLO"),
       offs, bias, out_dtype,
       IgnoreInCacheKey(contraction_dim,
                        "Validated upfront; doesn't affect SHLO"),
       IgnoreInCacheKey(use_fast_accum, "Doesn't affect SHLO"), out),
      {
        // use_fast_accum is accepted and ignored: the MXU always
        // accumulates in F32.
        TT_ASSIGN_OR_THROW(
            auto result_buffer,
            ScaledGroupedMmV2(self, mat2, scale_a, recipe_a, swizzle_a, scale_b,
                              recipe_b, swizzle_b, offs, bias, out_dtype,
                              contraction_dim, std::move(param_keys)));
        TT_THROW_IF_ERROR(
            ValidateOutDtype(out, out_dtype.value_or(self.scalar_type()),
                             /*allow_out_dtype_cast=*/false));
        TT_THROW_IF_ERROR(internal::AssignBufferToOutput(
            out, std::move(result_buffer), /*enable_autodonation=*/false));
        return out;
      });
}

}  // namespace torch_tpu
