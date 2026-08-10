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

#include "torch_tpu/ops/grouped_mm/grouped_mm_aten_kernels.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/precision_context.h"

namespace torch_tpu {
namespace {

bool IsScalar(const at::Tensor& t) {
  return t.dim() == 0 || (t.dim() == 1 && t.size(0) == 1);
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
//   same ragged partitioning along Sum(k_i).
//
// - Case 4 (3D x 3D without offs):
//   Input:  self [b, m, k], mat2 [b, k, n]
//   Output: [b, m, n]
//   Degrades to standard batched matrix multiplication (DotGeneral).
absl::StatusOr<mlir::MlirOp> BuildGroupedMmShlo(
    mlir::MlirOp self_op, mlir::MlirOp mat2_op,
    std::optional<mlir::MlirOp> offs_op, mlir::ElementType out_dtype,
    mlir::stablehlo::Precision precision) {
  mlir::MlirBuilder& builder = self_op.getBuilder();
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self_op);
  const mlir::RankedTensorType mat2_type = GetTensorTypeOrDie(mat2_op);

  // Upcast input tensors to the computation precision (e.g., F32 or BF16)
  // prior to RaggedDot/DotGeneral because CHLO RaggedDot does not natively
  // handle sub-byte types like FP4/MXFP8 or mixed precision without explicit
  // casts. For standard types (e.g., BF16→BF16), this is a no-op that the
  // MLIR canonicalizer eliminates.
  self_op = mlir::stablehlo::ConvertElementType(self_op, out_dtype);
  mat2_op = mlir::stablehlo::ConvertElementType(mat2_op, out_dtype);

  const bool a_is_2d = self_type.getRank() == 2;
  const bool b_is_2d = mat2_type.getRank() == 2;

  mlir::chlo::Precision chlo_precision =
      static_cast<mlir::chlo::Precision>(precision);
  auto precision_attr =
      mlir::chlo::PrecisionAttr::get(&self_op.getContext(), chlo_precision);
  auto precision_config = mlir::ArrayAttr::get(
      &self_op.getContext(), {precision_attr, precision_attr});

  if (a_is_2d && !b_is_2d) {
    // Case 1 (2D x 3D with offs): lhs first dimension is ragged (Mode 1).
    ABSL_CHECK(offs_op.has_value());  // CRASH_OK
    const mlir::RankedTensorType offs_type = GetTensorTypeOrDie(*offs_op);
    const int64_t num_groups = offs_type.getShape()[0];

    mlir::MlirOp zero_op =
        MakeConstant(builder, 0, offs_type.getElementType(), {1});
    const Strides strides(1, 1);
    const Indices start_indices(1, 0);
    const Indices limit_indices(1, std::max<int64_t>(0, num_groups - 1));
    mlir::MlirOp offs_slice =
        mlir::stablehlo::Slice(*offs_op, start_indices, limit_indices, strides);
    mlir::MlirOp concat_op =
        mlir::stablehlo::Concatenate(builder, {zero_op, offs_slice}, 0);
    mlir::MlirOp group_sizes = mlir::stablehlo::Subtract(*offs_op, concat_op);

    const mlir::chlo::RaggedDotDimensionNumbersAttr dimension_numbers =
        mlir::chlo::RaggedDotDimensionNumbersAttr::get(
            &self_op.getContext(), /*lhsBatchingDimensions=*/{},
            /*rhsBatchingDimensions=*/{},
            /*lhsContractingDimensions=*/{1},
            /*rhsContractingDimensions=*/{1},
            /*lhsRaggedDimensions=*/{0},
            /*rhsGroupDimensions=*/{0});

    const mlir::Type out_type = mlir::makeTensorType(
        self_op.getContext(),
        {self_type.getShape()[0], mat2_type.getShape()[2]}, out_dtype);
    return mlir::chlo::RaggedDot(out_type, self_op, mat2_op, group_sizes,
                                 dimension_numbers, precision_config);
  } else if (!a_is_2d && b_is_2d) {
    // Case 2 (3D x 2D with offs): rhs tailing dimension is ragged (Mode 1).
    ABSL_CHECK(offs_op.has_value());  // CRASH_OK
    const mlir::RankedTensorType offs_type = GetTensorTypeOrDie(*offs_op);
    const int64_t num_groups = offs_type.getShape()[0];

    mlir::MlirOp zero_op =
        MakeConstant(builder, 0, offs_type.getElementType(), {1});
    const Strides strides(1, 1);
    const Indices start_indices(1, 0);
    const Indices limit_indices(1, std::max<int64_t>(0, num_groups - 1));
    mlir::MlirOp offs_slice =
        mlir::stablehlo::Slice(*offs_op, start_indices, limit_indices, strides);
    mlir::MlirOp concat_op =
        mlir::stablehlo::Concatenate(builder, {zero_op, offs_slice}, 0);
    mlir::MlirOp group_sizes = mlir::stablehlo::Subtract(*offs_op, concat_op);

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
        {mat2_type.getShape()[1], self_type.getShape()[1]}, out_dtype);
    mlir::MlirOp ragged_dot =
        mlir::chlo::RaggedDot(ragged_out_type, mat2_transposed, self_transposed,
                              group_sizes, dimension_numbers, precision_config);
    return mlir::stablehlo::Transpose(ragged_dot, {1, 0});
  } else if (a_is_2d && b_is_2d) {
    // Case 3 (2D x 2D with offs): contracting dimension is ragged (Mode 2).
    // The trailing dimension of lhs and the leading dimension of rhs are
    // partitioned according to the same offs tensor.
    ABSL_CHECK(offs_op.has_value());  // CRASH_OK
    const mlir::RankedTensorType offs_type = GetTensorTypeOrDie(*offs_op);
    const int64_t num_groups = offs_type.getShape()[0];

    mlir::MlirOp zero_op =
        MakeConstant(builder, 0, offs_type.getElementType(), {1});
    const Strides strides(1, 1);
    const Indices start_indices(1, 0);
    const Indices limit_indices(1, std::max<int64_t>(0, num_groups - 1));
    mlir::MlirOp offs_slice =
        mlir::stablehlo::Slice(*offs_op, start_indices, limit_indices, strides);
    mlir::MlirOp concat_op =
        mlir::stablehlo::Concatenate(builder, {zero_op, offs_slice}, 0);
    mlir::MlirOp group_sizes = mlir::stablehlo::Subtract(*offs_op, concat_op);

    const mlir::chlo::RaggedDotDimensionNumbersAttr dimension_numbers =
        mlir::chlo::RaggedDotDimensionNumbersAttr::get(
            &self_op.getContext(), /*lhsBatchingDimensions=*/{},
            /*rhsBatchingDimensions=*/{},
            /*lhsContractingDimensions=*/{1},
            /*rhsContractingDimensions=*/{0},
            /*lhsRaggedDimensions=*/{1},
            /*rhsGroupDimensions=*/{});

    const auto out_type = mlir::makeTensorType(
        self_op.getContext(),
        {num_groups, self_type.getShape()[0], mat2_type.getShape()[1]},
        out_dtype);
    return mlir::chlo::RaggedDot(out_type, self_op, mat2_op, group_sizes,
                                 dimension_numbers, precision_config);
  } else {
    // Case 4 (3D x 3D without offs): fallback to standard batched matrix
    // multiplication (DotGeneral).
    auto precision_config = mlir::stablehlo::PrecisionConfigAttr::get(
        &self_op.getContext(), {precision, precision});
    const mlir::stablehlo::DotDimensionNumbersAttr dot_dimension_numbers =
        mlir::stablehlo::DotDimensionNumbersAttr::get(
            &self_op.getContext(),
            /*lhs_batch_dimensions=*/{0},
            /*rhs_batch_dimensions=*/{0},
            /*lhs_contracting_dimensions=*/{2},
            /*rhs_contracting_dimensions=*/{1});
    return mlir::stablehlo::DotGeneral(self_op, mat2_op, dot_dimension_numbers,
                                       precision_config);
  }
}

// Expands a 1D per-group scale [g] to a ragged dimension [ragged_size] using
// cumulative offsets. Each element in the output gets the scale value for its
// group, determined by building a one-hot indicator matrix from offsets and
// then multiplying by the scale vector via Dot.
//
// TODO: For Cases 1 & 2, pre-scaling the 3D input before RaggedDot avoids
// this expansion. Deferring to a follow-up CL after benchmarking.
absl::StatusOr<mlir::MlirOp> ExpandScaleToRaggedDim(mlir::MlirOp scale,
                                                    mlir::MlirOp offs,
                                                    int64_t ragged_size) {
  mlir::MlirBuilder& builder = scale.getBuilder();
  const auto scale_type = GetTensorTypeOrDie(scale);
  const int64_t num_groups = scale_type.getShape()[0];

  // Cast offs to F32 for comparison arithmetic with the iota.
  TT_ASSIGN_OR_RETURN(mlir::MlirOp offs_f32,
                      CastIfNeeded(offs, mlir::ElementType::F32));

  // Build iota [ragged_size] containing indices 0..ragged_size-1 as F32.
  const auto iota_type = mlir::makeTensorType(
      builder.getContext(), {ragged_size}, mlir::ElementType::F32);
  mlir::MlirOp iota = mlir::stablehlo::Iota(builder, iota_type, 0);

  // Build indicator matrix [ragged_size, num_groups] where each row has a 1.0
  // in exactly the column corresponding to its group.
  //
  // For cumulative offsets [o0, o1, ..., o_{g-1}]:
  //   ge_mask[i, g] = (i >= offs[g])
  //   lt_mask[i, g] = (i < offs[g])
  //
  // The shifted masks give us:
  //   ge_mask_shifted[i, g] = (i >= offs[g-1]) for g>0, true for g=0
  //
  // indicator[i, g] = ge_mask_shifted[i, g] AND lt_mask[i, g]
  //
  // Simpler approach: indicator = ge_mask_for_left_boundary AND NOT ge_mask.
  // But the simplest is: indicator[i, g] = 1 iff g = argmax over g' of
  // (offs[g'] <= i). We compute this as:
  //   group_count[i] = sum_g (i >= offs[g])  (count of passed boundaries)
  //   group_id[i] = group_count[i]  (since offs are cumulative and sorted)
  //
  // Then the expanded scale is just: scale[group_id[i]].
  // We compute group_count via Dot: ones[1, g] summed by indicator.
  // But actually, the simplest approach: just compute indicator and Dot with
  // scale.

  // Broadcasting: iota [ragged_size] -> [ragged_size, num_groups]
  //               offs [num_groups]  -> [ragged_size, num_groups]
  auto bcast_type = mlir::makeTensorType(
      builder.getContext(), {ragged_size, num_groups}, mlir::ElementType::F32);
  std::vector<int64_t> iota_bcast_dims = {0};  // INT_VEC_OK
  mlir::MlirOp iota_2d =
      mlir::stablehlo::BroadcastInDim(bcast_type, iota, iota_bcast_dims);

  std::vector<int64_t> offs_bcast_dims = {1};  // INT_VEC_OK
  mlir::MlirOp offs_2d =
      mlir::stablehlo::BroadcastInDim(bcast_type, offs_f32, offs_bcast_dims);

  // Build shifted offsets for left boundary: offs_shifted = [0, offs[0], ...,
  // offs[g-2]]. This gives the left boundary for each group.
  mlir::MlirOp zero_op =
      MakeConstant(builder, 0.0f, mlir::ElementType::F32, {1});
  const Strides strides(1, 1);
  const Indices start_indices(1, 0);
  const Indices limit_indices(1, std::max<int64_t>(0, num_groups - 1));
  mlir::MlirOp offs_prefix =
      mlir::stablehlo::Slice(offs_f32, start_indices, limit_indices, strides);
  mlir::MlirOp offs_shifted =
      mlir::stablehlo::Concatenate(builder, {zero_op, offs_prefix}, 0);

  mlir::MlirOp offs_shifted_2d = mlir::stablehlo::BroadcastInDim(
      bcast_type, offs_shifted, offs_bcast_dims);

  // ge_shifted[i, g] = (i >= offs_shifted[g]): 1 if position i is at or past
  // the left boundary of group g.
  mlir::MlirOp ge_shifted = mlir::stablehlo::Compare(
      iota_2d, offs_shifted_2d, mlir::stablehlo::ComparisonDirection::GE);

  // lt_mask[i, g] = (i < offs[g]): 1 if position i has NOT passed boundary g.
  mlir::MlirOp lt_mask = mlir::stablehlo::Compare(
      iota_2d, offs_2d, mlir::stablehlo::ComparisonDirection::LT);

  // indicator[i, g] = ge_shifted[i, g] AND lt_mask[i, g]
  // This is 1 iff position i belongs to group g.
  mlir::MlirOp indicator = mlir::stablehlo::And(ge_shifted, lt_mask);

  // Convert indicator to F32 and multiply with scale via Dot:
  // result[i] = sum_g indicator[i, g] * scale[g]
  // This is a matrix-vector product: indicator [ragged_size, g] @ scale [g].
  TT_ASSIGN_OR_RETURN(mlir::MlirOp indicator_f32,
                      CastIfNeeded(indicator, mlir::ElementType::F32));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp scale_f32,
                      CastIfNeeded(scale, mlir::ElementType::F32));

  // Dot: indicator [ragged_size, num_groups] × scale [num_groups] →
  //      result [ragged_size]
  return mlir::stablehlo::Dot(indicator_f32, scale_f32);
}

absl::StatusOr<mlir::MlirOp> BuildScaledGroupedMmShlo(
    mlir::MlirOp self_op, mlir::MlirOp mat2_op, mlir::MlirOp scale_a,
    mlir::MlirOp scale_b, std::optional<mlir::MlirOp> offs_op,
    std::optional<mlir::MlirOp> bias, std::optional<mlir::MlirOp> scale_result,
    mlir::ElementType out_dtype, mlir::stablehlo::Precision precision) {
  const auto self_type = GetTensorTypeOrDie(self_op);
  TT_ASSIGN_OR_RETURN(mlir::ElementType self_elem_type,
                      ConvertTo<mlir::ElementType>(self_type.getElementType()));

  // For sub-byte types like FP4 or MXFP8, computation precision is typically
  // elevated to F32 to preserve accuracy and because XLA does not support
  // sub-byte math directly. Standard floating types (BF16, F32) usually map to
  // themselves.
  TT_ASSIGN_OR_RETURN(auto comp_dtype, InferComputationDtype(self_elem_type));

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp mm_result,
      BuildGroupedMmShlo(self_op, mat2_op, offs_op, comp_dtype, precision));
  TT_ASSIGN_OR_RETURN(mm_result,
                      CastIfNeeded(mm_result, mlir::ElementType::F32));

  TT_ASSIGN_OR_RETURN(auto scale_a_f32,
                      CastIfNeeded(scale_a, mlir::ElementType::F32));
  TT_ASSIGN_OR_RETURN(auto scale_b_f32,
                      CastIfNeeded(scale_b, mlir::ElementType::F32));
  TT_ASSIGN_OR_RETURN(auto broadcasted_scales,
                      ApplyBroadcastIfNeeded<2>({scale_a_f32, scale_b_f32}));
  auto combined_scale =
      mlir::stablehlo::Mul(broadcasted_scales[0], broadcasted_scales[1]);

  const auto result_type = GetTensorTypeOrDie(mm_result);
  const auto combined_scale_type = GetTensorTypeOrDie(combined_scale);
  const bool is_scalar_scale = combined_scale_type.getRank() == 0 ||
                               (combined_scale_type.getRank() == 1 &&
                                combined_scale_type.getShape()[0] == 1);

  // Broadcast combined scale to match the mm result shape.
  // - Scalar scales: broadcast via ApplyBroadcastIfNeeded (all cases).
  // - 1D per-group scales with rank-3 result (Cases 3 & 4): BroadcastInDim
  //   along dim 0.
  // - 1D per-group scales with rank-2 ragged result (Cases 1 & 2): expand
  //   using gather-based indexing from offsets.
  auto broadcast_scale_to_result =
      [&](mlir::MlirOp scale) -> absl::StatusOr<mlir::MlirOp> {
    const auto scale_type = GetTensorTypeOrDie(scale);
    if (scale_type.getRank() == 0 ||
        (scale_type.getRank() == 1 && scale_type.getShape()[0] == 1)) {
      return scale;
    }
    if (result_type.getRank() == 3) {
      auto bcast_type =
          mlir::makeTensorType(scale.getContext(), result_type.getShape(),
                               scale_type.getElementType());
      std::vector<int64_t> bcast_dims = {0};  // INT_VEC_OK
      return mlir::stablehlo::BroadcastInDim(bcast_type, scale, bcast_dims);
    }
    // Rank-2 ragged result: expand per-group scale using offsets.
    ABSL_CHECK(offs_op.has_value())  // CRASH_OK
        << "Per-group scales require offsets for 2D ragged results. "
        << "This is a TorchTPU bug.";
    const bool a_is_2d = self_type.getRank() == 2;
    const int64_t ragged_size =
        a_is_2d ? result_type.getShape()[0] : result_type.getShape()[1];
    return ExpandScaleToRaggedDim(scale, *offs_op, ragged_size);
  };

  TT_ASSIGN_OR_RETURN(auto rank_expanded_combined_scale,
                      broadcast_scale_to_result(combined_scale));

  // For 2D ragged results with per-group scales, the expanded scale is 1D
  // [ragged_size]. We need to broadcast it to [ragged_size, n] or [b,
  // ragged_size] before element-wise multiplication with the 2D result.
  if (!is_scalar_scale && result_type.getRank() == 2) {
    const bool a_is_2d = self_type.getRank() == 2;
    auto bcast_type = mlir::makeTensorType(
        rank_expanded_combined_scale.getContext(), result_type.getShape(),
        combined_scale_type.getElementType());
    // Case 1 (2D×3D): scale is along dim 0 (ragged rows).
    // Case 2 (3D×2D): scale is along dim 1 (ragged cols).
    std::vector<int64_t> bcast_dims = {a_is_2d ? 0 : 1};  // INT_VEC_OK
    rank_expanded_combined_scale = mlir::stablehlo::BroadcastInDim(
        bcast_type, rank_expanded_combined_scale, bcast_dims);
  }

  TT_ASSIGN_OR_RETURN(
      auto broadcasted_mm_scale,
      ApplyBroadcastIfNeeded<2>({mm_result, rank_expanded_combined_scale}));
  auto scaled_result =
      mlir::stablehlo::Mul(broadcasted_mm_scale[0], broadcasted_mm_scale[1]);

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

absl::StatusOr<c10::SmallVector<int64_t, 3>> GetGroupedMmOutputSize(
    const at::Tensor& self, const at::Tensor& mat2,
    const std::optional<at::Tensor>& offs) {
  const bool a_is_2d = self.dim() == 2;
  const bool b_is_2d = mat2.dim() == 2;
  const bool has_offs = offs.has_value() && offs->defined();

  TT_RET_CHECK(self.dim() == 2 || self.dim() == 3, error::kInvalidArgument)
      << "expected self to be 2D or 3D, got " << self.dim() << "D";
  TT_RET_CHECK(mat2.dim() == 2 || mat2.dim() == 3, error::kInvalidArgument)
      << "expected mat2 to be 2D or 3D, got " << mat2.dim() << "D";
  TT_RET_CHECK(has_offs == (a_is_2d || b_is_2d), error::kInvalidArgument)
      << "expected offs to be provided if and only if either self or mat2 is "
         "2D";

  c10::SmallVector<int64_t, 3> out_size;
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
  return out_size;
}

absl::StatusOr<DeviceBufferRef> GroupedMm(
    const at::Tensor& self, const at::Tensor& mat2,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    std::optional<at::ScalarType> out_dtype, OpParamCacheKeys& param_keys) {
  // In ragged grouped GEMM, adding bias inside the kernel requires
  // expensive dynamic indexing because each expert group has an irregular
  // token count. To optimize performance, PyTorch CUDA/CPU kernels
  // explicitly reject bias.
  // Eager models also perform out.add_(bias) externally.
  TT_RET_CHECK(!bias.has_value() || !bias->defined(),
               error::kPythonNotImplementedError)
      << "expected bias to be undefined, got defined tensor";

  const bool has_offs = offs.has_value() && offs->defined();

  TT_ASSIGN_OR_RETURN(auto out_size, GetGroupedMmOutputSize(self, mat2, offs));

  at::ScalarType target_dtype = out_dtype.value_or(self.scalar_type());
  TT_ASSIGN_OR_RETURN(mlir::ElementType output_elem_type,
                      ConvertTo<mlir::ElementType>(target_dtype));

  Dimensions out_dims_vec(out_size.begin(), out_size.end());

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
                                .out_dims = out_dims_vec,
                                .op_param_cache_keys = std::move(param_keys)}));
  return result_buffer;
}

absl::StatusOr<DeviceBufferRef> ScaledGroupedMm(
    const at::Tensor& self, const at::Tensor& mat2, const at::Tensor& scale_a,
    const at::Tensor& scale_b, const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& scale_result,
    std::optional<at::ScalarType> out_dtype, bool use_fast_accum,
    OpParamCacheKeys& param_keys) {
  TT_RET_CHECK(!use_fast_accum, error::kPythonNotImplementedError)
      << "use_fast_accum=true is not supported yet";

  const bool has_offs = offs.has_value() && offs->defined();
  const bool has_bias = bias.has_value() && bias->defined();
  const bool has_scale_result =
      scale_result.has_value() && scale_result->defined();

  // Scales must be scalar (per-tensor) or 1D (per-group). 2D+ scales are not
  // supported.
  TT_RET_CHECK(scale_a.dim() <= 1, error::kPythonNotImplementedError)
      << "expected scale_a to be 1D or scalar, got " << scale_a.dim() << "D";
  TT_RET_CHECK(scale_b.dim() <= 1, error::kPythonNotImplementedError)
      << "expected scale_b to be 1D or scalar, got " << scale_b.dim() << "D";
  if (!IsScalar(scale_a) || !IsScalar(scale_b)) {
    const int64_t num_groups = has_offs ? offs->size(0) : self.size(0);
    if (!IsScalar(scale_a)) {
      TT_RET_CHECK(scale_a.numel() == num_groups, error::kInvalidArgument)
          << "expected per-group scale_a to have " << num_groups
          << " elements, got " << scale_a.numel();
    }
    if (!IsScalar(scale_b)) {
      TT_RET_CHECK(scale_b.numel() == num_groups, error::kInvalidArgument)
          << "expected per-group scale_b to have " << num_groups
          << " elements, got " << scale_b.numel();
    }
  }
  if (has_bias) {
    TT_RET_CHECK(IsScalar(*bias), error::kPythonNotImplementedError)
        << "expected bias to be scalar, got " << bias->dim() << "D";
  }
  if (has_scale_result) {
    TT_RET_CHECK(IsScalar(*scale_result), error::kPythonNotImplementedError)
        << "expected scale_result to be scalar, got " << scale_result->dim()
        << "D";
  }

  TT_ASSIGN_OR_RETURN(auto out_size, GetGroupedMmOutputSize(self, mat2, offs));

  const at::ScalarType target_dtype = out_dtype.value_or(self.scalar_type());
  TT_ASSIGN_OR_RETURN(mlir::ElementType output_elem_type,
                      ConvertTo<mlir::ElementType>(target_dtype));

  Dimensions out_dims_vec(out_size.begin(), out_size.end());
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
                                .out_dims = out_dims_vec,
                                .op_param_cache_keys = std::move(param_keys)}));
  return result_buffer;
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
             use_fast_accum),
            {
              TT_ASSIGN_OR_THROW(
                  auto result_buffer,
                  ScaledGroupedMm(self, mat2, scale_a, scale_b, offs, bias,
                                  scale_result, out_dtype, use_fast_accum,
                                  param_keys));
              return MakeTensor(result_buffer);
            });
}

}  // namespace torch_tpu
