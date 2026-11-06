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

#include "csrc/ops/sparse_coo/sparse_coo.h"

#include <array>
#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/Scalar.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/utils.h"
#include "csrc/ops/cumsum/cumsum.h"
#include "csrc/ops/index_select/index_select.h"
#include "csrc/ops/op_builder_utils.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

mlir::MlirOp ScatterAddSparseToDense(mlir::MlirBuilder& builder,
                                     mlir::MlirOp operand,
                                     mlir::MlirOp scatter_indices_2d,
                                     mlir::MlirOp updates) {
  const int64_t operand_rank = GetTensorTypeOrDie(operand).getRank();
  const int64_t k = GetTensorTypeOrDie(scatter_indices_2d).getDimSize(1);
  const int64_t m = operand_rank - k;
  llvm::SmallVector<int64_t, 4> update_window_dims;
  for (int64_t i = 1; i <= m; ++i) {
    update_window_dims.push_back(i);
  }
  llvm::SmallVector<int64_t, 4> inserted_window_dims;
  llvm::SmallVector<int64_t, 4> scatter_dims_to_operand_dims;
  for (int64_t i = 0; i < k; ++i) {
    inserted_window_dims.push_back(i);
    scatter_dims_to_operand_dims.push_back(i);
  }
  const auto scatter_dims = mlir::stablehlo::ScatterDimensionNumbersAttr::get(
      &builder.getContext(), update_window_dims, inserted_window_dims,
      /*input_batching_dims=*/{}, /*scatter_indices_batching_dims=*/{},
      scatter_dims_to_operand_dims, /*index_vector_dim=*/1);
  const auto body = [elem_type = GetTensorTypeOrDie(updates).getElementType()](
                        mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        elem_type, rb.getRegion(), rb.getOpBuilder());
  };
  return mlir::stablehlo::Scatter({operand}, scatter_indices_2d, {updates},
                                  body, scatter_dims)[0];
}

void BuildLexicographicalI64Comparator(const int64_t num_keys,
                                       mlir::RegionBuilder& rb) {
  auto& b = rb.getOpBuilder();
  const mlir::Location loc = rb.getRegion().getLoc();
  mlir::Block* block = rb.getRegion().empty() ? b.createBlock(&rb.getRegion())
                                              : &rb.getRegion().front();
  const mlir::RankedTensorType scalar_i64 =
      mlir::RankedTensorType::get({}, b.getI64Type());
  for (int64_t i = 0; i < num_keys; ++i) {
    block->addArguments({scalar_i64, scalar_i64}, {loc, loc});
  }
  // Last key (perm_iota) has unique elements, so LT is strictly total.
  mlir::Value res = mlir::stablehlo::CompareOp::create(
      b, loc, block->getArgument(2 * (num_keys - 1)),
      block->getArgument(2 * (num_keys - 1) + 1),
      mlir::stablehlo::ComparisonDirection::LT,
      mlir::stablehlo::ComparisonType::SIGNED);
  for (int64_t i = num_keys - 2; i >= 0; --i) {
    const mlir::Value lhs = block->getArgument(2 * i);
    const mlir::Value rhs = block->getArgument(2 * i + 1);
    const mlir::Value lt = mlir::stablehlo::CompareOp::create(
        b, loc, lhs, rhs, mlir::stablehlo::ComparisonDirection::LT,
        mlir::stablehlo::ComparisonType::SIGNED);
    const mlir::Value eq = mlir::stablehlo::CompareOp::create(
        b, loc, lhs, rhs, mlir::stablehlo::ComparisonDirection::EQ,
        mlir::stablehlo::ComparisonType::SIGNED);
    res = mlir::stablehlo::SelectOp::create(b, loc, eq, res, lt);
  }
  mlir::stablehlo::ReturnOp::create(b, loc, res);
}

void BuildMaskDescIotaAscComparator(mlir::RegionBuilder& rb) {
  auto& b = rb.getOpBuilder();
  const mlir::Location loc = rb.getRegion().getLoc();
  mlir::Block* block = rb.getRegion().empty() ? b.createBlock(&rb.getRegion())
                                              : &rb.getRegion().front();
  const mlir::RankedTensorType scalar_i1 =
      mlir::RankedTensorType::get({}, b.getI1Type());
  const mlir::RankedTensorType scalar_i64 =
      mlir::RankedTensorType::get({}, b.getI64Type());
  block->addArguments({scalar_i1, scalar_i1, scalar_i64, scalar_i64},
                      {loc, loc, loc, loc});
  const mlir::Value lhs_mask = block->getArgument(0);
  const mlir::Value rhs_mask = block->getArgument(1);
  const mlir::Value lhs_iota = block->getArgument(2);
  const mlir::Value rhs_iota = block->getArgument(3);

  const mlir::Value gt_mask = mlir::stablehlo::CompareOp::create(
      b, loc, lhs_mask, rhs_mask, mlir::stablehlo::ComparisonDirection::GT,
      mlir::stablehlo::ComparisonType::UNSIGNED);
  const mlir::Value eq_mask = mlir::stablehlo::CompareOp::create(
      b, loc, lhs_mask, rhs_mask, mlir::stablehlo::ComparisonDirection::EQ,
      mlir::stablehlo::ComparisonType::UNSIGNED);
  const mlir::Value lt_iota = mlir::stablehlo::CompareOp::create(
      b, loc, lhs_iota, rhs_iota, mlir::stablehlo::ComparisonDirection::LT,
      mlir::stablehlo::ComparisonType::SIGNED);
  const mlir::Value res =
      mlir::stablehlo::SelectOp::create(b, loc, eq_mask, lt_iota, gt_mask);
  mlir::stablehlo::ReturnOp::create(b, loc, res);
}

struct SortedSparseCoo {
  mlir::MlirOp indices;
  mlir::MlirOp values;
};

SortedSparseCoo SortSparseCoordinatesAndValues(mlir::MlirBuilder& builder,
                                               mlir::MlirOp indices_op,
                                               mlir::MlirOp values_op) {
  const mlir::RankedTensorType indices_type = GetTensorTypeOrDie(indices_op);
  const int64_t k = indices_type.getDimSize(0);
  const int64_t nnz = indices_type.getDimSize(1);
  const auto i64_type = builder.getOpBuilder().getI64Type();

  llvm::SmallVector<mlir::MlirOp, 8> sort_inputs;
  sort_inputs.reserve(k + 1);
  for (int64_t d = 0; d < k; ++d) {
    mlir::MlirOp row_2d =
        mlir::stablehlo::Slice(indices_op, {d, 0}, {d + 1, nnz}, {1, 1});
    sort_inputs.push_back(mlir::stablehlo::Reshape(row_2d, {nnz}));
  }
  const auto iota_type =
      mlir::makeTensorType(builder.getContext(), {nnz}, i64_type);
  sort_inputs.push_back(mlir::stablehlo::Iota(builder, iota_type, 0));

  const int64_t num_keys = k + 1;
  const auto comparator = [num_keys](mlir::RegionBuilder& rb) {
    BuildLexicographicalI64Comparator(num_keys, rb);
  };
  const auto sorted = mlir::stablehlo::Sort(
      builder, sort_inputs, comparator, /*dimension=*/0, /*is_stable=*/false);

  llvm::SmallVector<mlir::MlirOp, 8> sorted_rows;
  sorted_rows.reserve(k);
  for (int64_t d = 0; d < k; ++d) {
    sorted_rows.push_back(mlir::stablehlo::Reshape(sorted[d], {1, nnz}));
  }
  mlir::MlirOp sorted_indices =
      (k == 1) ? sorted_rows[0]
               : mlir::stablehlo::Concatenate(builder, sorted_rows,
                                              /*dimension=*/0);
  mlir::MlirOp sorted_values =
      BuildIndexSelectShlo(values_op, /*dim=*/0, sorted[k]);
  return {sorted_indices, sorted_values};
}

mlir::MlirOp ComputeUniqueCoordinateMask(mlir::MlirBuilder& builder,
                                         mlir::MlirOp sorted_indices) {
  const mlir::RankedTensorType indices_type =
      GetTensorTypeOrDie(sorted_indices);
  const int64_t k = indices_type.getDimSize(0);
  const int64_t nnz = indices_type.getDimSize(1);
  mlir::MlirOp head =
      mlir::stablehlo::Slice(sorted_indices, {0, 0}, {k, nnz - 1}, {1, 1});
  mlir::MlirOp tail =
      mlir::stablehlo::Slice(sorted_indices, {0, 1}, {k, nnz}, {1, 1});
  mlir::MlirOp diff_2d = mlir::stablehlo::Compare(
      tail, head, mlir::stablehlo::ComparisonDirection::NE);
  mlir::MlirOp init_or =
      MakeScalarConstant(builder, 0, builder.getOpBuilder().getI1Type());
  mlir::MlirOp diff_1d = mlir::stablehlo::Reduce(
      builder, {diff_2d}, {init_or},
      [](mlir::RegionBuilder& rb) {
        auto& b = rb.getOpBuilder();
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::OrOp>(
            b.getI1Type(), rb.getRegion(), b);
      },
      {0})[0];
  mlir::MlirOp first_true =
      MakeConstant(builder, 1, builder.getOpBuilder().getI1Type(), {1});
  return mlir::stablehlo::Concatenate(builder, {first_true, diff_1d},
                                      /*dimension=*/0);
}

mlir::MlirOp CompactUniqueIndices(mlir::MlirBuilder& builder,
                                  mlir::MlirOp sorted_indices,
                                  mlir::MlirOp is_unique_mask) {
  const int64_t nnz = GetTensorTypeOrDie(sorted_indices).getDimSize(1);
  const auto i64_type = builder.getOpBuilder().getI64Type();
  const auto iota_type =
      mlir::makeTensorType(builder.getContext(), {nnz}, i64_type);
  mlir::MlirOp iota = mlir::stablehlo::Iota(builder, iota_type, 0);
  const auto sorted_mask_iota = mlir::stablehlo::Sort(
      builder, {is_unique_mask, iota}, BuildMaskDescIotaAscComparator,
      /*dimension=*/0, /*is_stable=*/false);
  return BuildIndexSelectShlo(sorted_indices, /*dim=*/1, sorted_mask_iota[1]);
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildSparseToDenseShlo(
    mlir::MlirOp indices_op, mlir::MlirOp values_op,
    absl::Span<const int64_t> out_sizes, mlir::ElementType out_dtype) {
  auto& builder = indices_op.getBuilder();
  TT_ASSIGN_OR_RETURN(mlir::MlirOp zero_dense,
                      MakeConstant(builder, c10::Scalar(0), out_dtype,
                                   mlir::ArrayRef<int64_t>(out_sizes.data(),
                                                           out_sizes.size())));
  if (GetTensorTypeOrDie(values_op).getDimSize(0) == 0) {
    return zero_dense;
  }
  TT_ASSIGN_OR_RETURN(mlir::MlirOp converted_values,
                      CastIfNeeded(values_op, out_dtype));
  mlir::MlirOp scatter_indices = mlir::stablehlo::Transpose(indices_op, {1, 0});
  return ScatterAddSparseToDense(builder, zero_dense, scatter_indices,
                                 converted_values);
}

absl::StatusOr<mlir::MlirOp> BuildAddDenseSparseShlo(
    mlir::MlirOp dense_self_op, mlir::MlirOp sparse_indices_op,
    mlir::MlirOp sparse_values_op, std::optional<mlir::MlirOp> alpha_op,
    mlir::ElementType out_dtype) {
  auto& builder = dense_self_op.getBuilder();
  TT_ASSIGN_OR_RETURN(mlir::MlirOp converted_dense,
                      CastIfNeeded(dense_self_op, out_dtype));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp scaled_values,
                      CastIfNeeded(sparse_values_op, out_dtype));
  if (alpha_op.has_value()) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp converted_alpha,
                        CastIfNeeded(*alpha_op, out_dtype));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp bcast_alpha,
                        BroadcastIfNeeded(converted_alpha, scaled_values));
    scaled_values = mlir::stablehlo::Mul(scaled_values, bcast_alpha);
  }
  mlir::MlirOp scatter_indices =
      mlir::stablehlo::Transpose(sparse_indices_op, {1, 0});
  return ScatterAddSparseToDense(builder, converted_dense, scatter_indices,
                                 scaled_values);
}

absl::StatusOr<std::array<mlir::MlirOp, 2>> BuildAddSparseSparseShlo(
    mlir::MlirOp indices1_op, mlir::MlirOp values1_op, mlir::MlirOp indices2_op,
    mlir::MlirOp values2_op, std::optional<mlir::MlirOp> alpha_op,
    mlir::ElementType out_dtype) {
  auto& builder = indices1_op.getBuilder();
  mlir::MlirOp new_indices = mlir::stablehlo::Concatenate(
      builder, {indices1_op, indices2_op}, /*dimension=*/1);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp converted_values1,
                      CastIfNeeded(values1_op, out_dtype));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp scaled_values2,
                      CastIfNeeded(values2_op, out_dtype));
  if (alpha_op.has_value()) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp converted_alpha,
                        CastIfNeeded(*alpha_op, out_dtype));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp bcast_alpha,
                        BroadcastIfNeeded(converted_alpha, scaled_values2));
    scaled_values2 = mlir::stablehlo::Mul(scaled_values2, bcast_alpha);
  }
  mlir::MlirOp new_values = mlir::stablehlo::Concatenate(
      builder, {converted_values1, scaled_values2}, /*dimension=*/0);
  return std::array<mlir::MlirOp, 2>{new_indices, new_values};
}

absl::StatusOr<mlir::MlirOp> BuildMulSparseScalarShlo(
    mlir::MlirOp values_op, mlir::MlirOp scalar_op,
    mlir::ElementType out_dtype) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp converted_values,
                      CastIfNeeded(values_op, out_dtype));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp converted_scalar,
                      CastIfNeeded(scalar_op, out_dtype));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp bcast_scalar,
                      BroadcastIfNeeded(converted_scalar, converted_values));
  return mlir::stablehlo::Mul(converted_values, bcast_scalar);
}

absl::StatusOr<mlir::MlirOp> BuildNegSparseShlo(mlir::MlirOp values_op,
                                                mlir::ElementType out_dtype) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp converted_values,
                      CastIfNeeded(values_op, out_dtype));
  return mlir::stablehlo::Neg(converted_values);
}

absl::StatusOr<std::array<mlir::MlirOp, 3>> BuildCoalesceAndCountShlo(
    mlir::MlirOp indices_op, mlir::MlirOp values_op) {
  auto& builder = indices_op.getBuilder();
  const auto [sorted_indices, sorted_values] =
      SortSparseCoordinatesAndValues(builder, indices_op, values_op);
  mlir::MlirOp is_unique_mask =
      ComputeUniqueCoordinateMask(builder, sorted_indices);

  const auto i64_type = builder.getOpBuilder().getI64Type();
  mlir::MlirOp mask_i64 =
      mlir::stablehlo::ConvertElementType(is_unique_mask, i64_type);
  mlir::MlirOp zero_i64 = MakeScalarConstant(builder, 0, i64_type);
  mlir::MlirOp count_tensor = mlir::stablehlo::Reduce(
      builder, {mask_i64}, {zero_i64},
      [](mlir::RegionBuilder& rb) {
        auto& b = rb.getOpBuilder();
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
            b.getI64Type(), rb.getRegion(), b);
      },
      {0})[0];

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp cumsum_mask,
      BuildCumsumShlo(/*normalized_dim=*/0, mlir::ElementType::I64, mask_i64));
  mlir::MlirOp one_like = MakeConstantLike(cumsum_mask, 1LL);
  mlir::MlirOp dest_idx_1d = mlir::stablehlo::Subtract(cumsum_mask, one_like);
  const int64_t nnz = GetTensorTypeOrDie(indices_op).getDimSize(1);
  mlir::MlirOp dest_idx_2d = mlir::stablehlo::Reshape(dest_idx_1d, {nnz, 1});

  const mlir::RankedTensorType val_type = GetTensorTypeOrDie(values_op);
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp zero_values,
      MakeConstant(builder, c10::Scalar(0), GetElementTypeOrDie(values_op),
                   val_type.getShape()));
  mlir::MlirOp compacted_values =
      ScatterAddSparseToDense(builder, zero_values, dest_idx_2d, sorted_values);
  mlir::MlirOp compacted_indices =
      CompactUniqueIndices(builder, sorted_indices, is_unique_mask);
  return std::array<mlir::MlirOp, 3>{compacted_indices, compacted_values,
                                     count_tensor};
}

absl::StatusOr<std::array<mlir::MlirOp, 2>> BuildSliceCoalescedShlo(
    mlir::MlirOp compacted_indices_op, mlir::MlirOp compacted_values_op,
    const int64_t output_size) {
  const int64_t k = GetTensorTypeOrDie(compacted_indices_op).getDimSize(0);
  mlir::MlirOp sliced_indices = mlir::stablehlo::Slice(
      compacted_indices_op, {0, 0}, {k, output_size}, {1, 1});

  const mlir::ArrayRef<int64_t> val_shape =
      GetTensorTypeOrDie(compacted_values_op).getShape();
  Dimensions start(val_shape.size(), 0);
  Dimensions limit = CopyIntVector(val_shape);
  limit[0] = output_size;
  Dimensions strides(val_shape.size(), 1);
  mlir::MlirOp sliced_values =
      mlir::stablehlo::Slice(compacted_values_op, start, limit, strides);
  return std::array<mlir::MlirOp, 2>{sliced_indices, sliced_values};
}

}  // namespace torch_tpu
