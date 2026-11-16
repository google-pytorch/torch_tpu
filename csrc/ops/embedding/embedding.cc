/*
 * Copyright 2025 Google LLC
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

#include "csrc/ops/embedding/embedding.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/utils.h"
#include "csrc/ops/cumsum/cumsum.h"
#include "csrc/ops/index_select/index_select.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/reductions/sum.h"
#include "csrc/ops/unary.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

struct MaxAggregationResult {
  mlir::MlirOp output;
  mlir::MlirOp max_indices;
};

// Gathers embedding rows from `weight` based on `indices`.
absl::StatusOr<mlir::MlirOp> BuildEmbeddingGather(mlir::MlirBuilder& builder,
                                                  mlir::MlirOp weight,
                                                  mlir::MlirOp indices) {
  const mlir::RankedTensorType weight_type = GetTensorTypeOrDie(weight);
  const int64_t indices_rank = GetTensorTypeOrDie(indices).getRank();
  llvm::SmallVector<int64_t, 1> offset_dims;
  for (int64_t i = 1; i < weight_type.getRank(); ++i) {
    offset_dims.push_back(indices_rank + i - 1);
  }
  auto dimension_numbers = mlir::stablehlo::GatherDimensionNumbersAttr::get(
      &builder.getContext(), offset_dims, /*collapsed_slice_dims=*/{0},
      /*operand_batching_dims=*/{}, /*start_index_batching_dims=*/{},
      /*start_index_map=*/{0}, /*index_vector_dim=*/indices_rank);
  Dimensions slice_sizes = CopyIntVector(weight_type.getShape());
  slice_sizes[0] = 1;
  return mlir::stablehlo::Gather(weight, indices, dimension_numbers,
                                 slice_sizes);
}

// Builds a simple scatter operation.
enum class HasWindow {
  kNo,
  kYes,
};

mlir::MlirOp BuildSimpleScatter(mlir::MlirBuilder& builder,
                                mlir::MlirOp operand, mlir::MlirOp indices,
                                mlir::MlirOp updates, HasWindow has_window) {
  llvm::SmallVector<int64_t, 1> update_window_dims;
  if (has_window == HasWindow::kYes) {
    for (int64_t i = 1; i < GetTensorTypeOrDie(operand).getRank(); ++i) {
      update_window_dims.push_back(i);
    }
  }
  auto scatter_dims = mlir::stablehlo::ScatterDimensionNumbersAttr::get(
      &builder.getContext(), update_window_dims, /*inserted_window_dims=*/{0},
      /*input_batching_dims=*/{}, /*scatter_indices_batching_dims=*/{},
      /*scatter_dims_to_operand_dims=*/{0}, /*index_vector_dim=*/1);
  auto body = [elem_type = GetTensorTypeOrDie(updates).getElementType()](
                  mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        elem_type, rb.getRegion(), rb.getOpBuilder());
  };
  return mlir::stablehlo::Scatter({operand}, indices, {updates}, body,
                                  scatter_dims)[0];
}

// Builds a tensor of shape [num_indices] where result[i] is the bag ID for
// index i, which is used to map each index to its corresponding bag.
absl::StatusOr<mlir::MlirOp> BuildOffset2Bag(mlir::MlirBuilder& builder,
                                             mlir::MlirOp indices,
                                             mlir::MlirOp offsets,
                                             mlir::Type i64_type,
                                             int64_t batch_size) {
  const int64_t num_indices = GetTensorTypeOrDie(indices).getDimSize(0);
  auto init = mlir::stablehlo::Constant(
      builder, mlir::DenseElementsAttr::get(
                   mlir::RankedTensorType::get({num_indices}, i64_type),
                   builder.getOpBuilder().getZeroAttr(i64_type)));
  auto ones_updates = mlir::stablehlo::Constant(
      builder, mlir::DenseElementsAttr::get(
                   mlir::RankedTensorType::get({batch_size}, i64_type),
                   builder.getOpBuilder().getIntegerAttr(i64_type, 1)));
  mlir::MlirOp active_offsets = offsets;
  if (GetTensorTypeOrDie(offsets).getDimSize(0) > batch_size) {
    auto iota_batch = mlir::stablehlo::Iota(
        builder, mlir::RankedTensorType::get({batch_size}, i64_type), 0);
    TT_ASSIGN_OR_RETURN(active_offsets,
                        BuildEmbeddingGather(builder, offsets, iota_batch));
  }
  TT_ASSIGN_OR_RETURN(auto scatter_indices, Unsqueeze(active_offsets, 1));
  auto bag_starts = BuildSimpleScatter(builder, init, scatter_indices,
                                       ones_updates, HasWindow::kNo);
  TT_ASSIGN_OR_RETURN(auto cumsum,
                      BuildCumsumShlo(0, std::nullopt, bag_starts));
  auto one_scalar = MakeScalarConstant(builder, 1, i64_type);
  TT_ASSIGN_OR_RETURN(auto one_bcst, BroadcastIfNeeded(one_scalar, cumsum));
  return mlir::stablehlo::Subtract(cumsum, one_bcst);
}

// Builds the scatter operation for MAX mode, which aggregates both the
// embedding values and indices. This implementation avoids a multi-operand
// scatter to prevent TPU backend crashes, using a two-pass approach.
absl::StatusOr<MaxAggregationResult> BuildMaxAggregation(
    mlir::MlirBuilder& builder, mlir::MlirOp updates, mlir::MlirOp indices,
    mlir::MlirOp val_init, mlir::MlirOp idx_init, mlir::Type i64_type,
    mlir::Type val_type, mlir::MlirOp offset2bag) {
  llvm::SmallVector<int64_t, 1> update_window_dims;
  for (int64_t i = 1; i < GetTensorTypeOrDie(val_init).getRank(); ++i) {
    update_window_dims.push_back(i);
  }
  auto scatter_dims = mlir::stablehlo::ScatterDimensionNumbersAttr::get(
      &builder.getContext(), update_window_dims, /*inserted_window_dims=*/{0},
      /*input_batching_dims=*/{}, /*scatter_indices_batching_dims=*/{},
      /*scatter_dims_to_operand_dims=*/{0}, /*index_vector_dim=*/1);

  // Scatter 1: MAX values
  auto val_body = [val_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::MaxOp>(
        val_type, rb.getRegion(), rb.getOpBuilder());
  };
  auto max_values = mlir::stablehlo::Scatter({val_init}, indices, {updates},
                                             val_body, scatter_dims)[0];

  // Gather max_values back to updates shape to find which indices achieved the
  // max
  TT_ASSIGN_OR_RETURN(auto max_values_gathered,
                      BuildEmbeddingGather(builder, max_values, offset2bag));

  // Compare to find valid indices
  auto is_max = mlir::stablehlo::Compare(
      updates, max_values_gathered, mlir::stablehlo::ComparisonDirection::EQ);

  // Construct iota representing the original linear indices
  const int64_t num_updates = GetTensorTypeOrDie(updates).getDimSize(0);
  auto iota = mlir::stablehlo::Iota(
      builder, mlir::RankedTensorType::get({num_updates}, i64_type), 0);
  mlir::MlirOp iota_extended = iota;
  for (int64_t i = 1; i < GetTensorTypeOrDie(val_init).getRank(); ++i) {
    TT_ASSIGN_OR_RETURN(iota_extended, Unsqueeze(iota_extended, i));
  }
  TT_ASSIGN_OR_RETURN(auto iota_bcst,
                      BroadcastIfNeeded(iota_extended, updates));

  // Select valid indices (use max_i64 for non-max elements)
  auto max_i64_scalar = MakeScalarConstant(
      builder, std::numeric_limits<int64_t>::max(), i64_type);
  TT_ASSIGN_OR_RETURN(auto max_i64_bcst,
                      BroadcastIfNeeded(max_i64_scalar, iota_bcst));
  auto valid_indices = mlir::stablehlo::Select(is_max, iota_bcst, max_i64_bcst);

  // Scatter 2: MIN index
  auto idx_body = [i64_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::MinOp>(
        i64_type, rb.getRegion(), rb.getOpBuilder());
  };

  // Initialize intermediate tensor to max_i64 for the MIN scatter
  int64_t batch_size = GetTensorTypeOrDie(val_init).getDimSize(0);
  int64_t emb_dim = GetTensorTypeOrDie(val_init).getDimSize(1);
  auto max_idx_init_min = mlir::stablehlo::Constant(
      builder, mlir::DenseElementsAttr::get(
                   mlir::RankedTensorType::get({batch_size, emb_dim}, i64_type),
                   builder.getOpBuilder().getI64IntegerAttr(
                       std::numeric_limits<int64_t>::max())));

  auto max_indices_min = mlir::stablehlo::Scatter(
      {max_idx_init_min}, indices, {valid_indices}, idx_body, scatter_dims)[0];

  // Handle empty bags: replace max_i64 with 0 (original idx_init value)
  TT_ASSIGN_OR_RETURN(auto max_i64_bcst_batch,
                      BroadcastIfNeeded(max_i64_scalar, max_indices_min));
  auto is_empty_idx =
      mlir::stablehlo::Compare(max_indices_min, max_i64_bcst_batch,
                               mlir::stablehlo::ComparisonDirection::EQ);
  auto zero_i64 = MakeScalarConstant(builder, 0, i64_type);
  TT_ASSIGN_OR_RETURN(auto zero_i64_bcst,
                      BroadcastIfNeeded(zero_i64, max_indices_min));
  auto final_max_indices =
      mlir::stablehlo::Select(is_empty_idx, zero_i64_bcst, max_indices_min);

  return MaxAggregationResult{max_values, final_max_indices};
}

absl::StatusOr<mlir::MlirOp> BuildRenormRow(mlir::MlirBuilder& builder,
                                            mlir::MlirOp row, double max_norm,
                                            double norm_type) {
  const auto row_type = GetTensorTypeOrDie(row);
  const auto orig_elem_type = row_type.getElementType();
  bool needs_upcast = orig_elem_type.isF16() || orig_elem_type.isBF16();
  mlir::Type elem_type =
      needs_upcast ? builder.getOpBuilder().getF32Type() : orig_elem_type;

  TT_ASSIGN_OR_RETURN(mlir::ElementType elem_dtype,
                      ConvertTo<mlir::ElementType>(elem_type));
  mlir::ElementType real_elem_type = RealComponentOf(elem_dtype);

  mlir::MlirOp acc_row = row;
  if (needs_upcast) {
    TT_ASSIGN_OR_RETURN(acc_row, PromoteFloatDtype(acc_row));
  }

  mlir::MlirOp norm;
  if (norm_type == 1.0) {
    mlir::MlirOp abs_row = mlir::stablehlo::Abs(acc_row);
    TT_ASSIGN_OR_RETURN(norm, BuildSumShlo(abs_row, {row_type.getRank() - 1},
                                           ReductionMode::kKeepDims));
  } else if (norm_type == 2.0) {
    mlir::MlirOp mul_row;
    if (llvm::isa<mlir::ComplexType>(elem_type)) {
      TT_ASSIGN_OR_RETURN(mlir::MlirOp conj_row,
                          BuildConjPhysicalShlo(acc_row));
      mlir::MlirOp complex_mul = mlir::stablehlo::Mul(acc_row, conj_row);
      // complex_mul is guaranteed to be a real number
      mul_row = mlir::stablehlo::Real(complex_mul);
    } else {
      mul_row = mlir::stablehlo::Mul(acc_row, acc_row);
    }
    TT_ASSIGN_OR_RETURN(norm, BuildSumShlo(mul_row, {row_type.getRank() - 1},
                                           ReductionMode::kKeepDims));
    norm = mlir::stablehlo::Sqrt(norm);
  } else {
    auto p_scalar = MakeScalarConstant(builder, norm_type, real_elem_type);
    auto p_inv_scalar =
        MakeScalarConstant(builder, 1.0 / norm_type, real_elem_type);
    mlir::MlirOp abs_row = mlir::stablehlo::Abs(acc_row);
    TT_ASSIGN_OR_RETURN(auto p_bcst, BroadcastIfNeeded(p_scalar, abs_row));
    mlir::MlirOp pow_row = mlir::stablehlo::Pow(abs_row, p_bcst);
    TT_ASSIGN_OR_RETURN(norm, BuildSumShlo(pow_row, {row_type.getRank() - 1},
                                           ReductionMode::kKeepDims));
    TT_ASSIGN_OR_RETURN(auto p_inv_bcst, BroadcastIfNeeded(p_inv_scalar, norm));
    norm = mlir::stablehlo::Pow(norm, p_inv_bcst);
  }
  auto max_norm_scalar = MakeScalarConstant(builder, max_norm, real_elem_type);
  TT_ASSIGN_OR_RETURN(auto max_norm_bcst,
                      BroadcastIfNeeded(max_norm_scalar, norm));
  auto too_large = mlir::stablehlo::Compare(
      norm, max_norm_bcst, mlir::stablehlo::ComparisonDirection::GT);
  auto eps_scalar = MakeScalarConstant(builder, 1e-7, real_elem_type);
  TT_ASSIGN_OR_RETURN(auto eps_bcst, BroadcastIfNeeded(eps_scalar, norm));
  auto norm_plus_eps = mlir::stablehlo::Add(norm, eps_bcst);
  auto multiplier = mlir::stablehlo::Div(max_norm_bcst, norm_plus_eps);
  auto ones_scalar = MakeScalarConstant(builder, 1.0, real_elem_type);
  TT_ASSIGN_OR_RETURN(auto ones_bcst,
                      BroadcastIfNeeded(ones_scalar, multiplier));
  auto safe_multiplier =
      mlir::stablehlo::Select(too_large, multiplier, ones_bcst);

  mlir::MlirOp mult_bcst = safe_multiplier;
  if (llvm::isa<mlir::ComplexType>(elem_type)) {
    mult_bcst = mlir::stablehlo::ConvertElementType(safe_multiplier, elem_type);
  }

  TT_ASSIGN_OR_RETURN(mult_bcst, BroadcastIfNeeded(mult_bcst, acc_row));
  auto result = mlir::stablehlo::Mul(acc_row, mult_bcst);
  if (needs_upcast) {
    result = mlir::stablehlo::ConvertElementType(result, orig_elem_type);
  }
  return result;
}

absl::StatusOr<mlir::MlirOp> ApplyPerSampleWeights(
    mlir::MlirOp tensor, const std::optional<mlir::MlirOp>& per_sample_weights,
    bool needs_upcast) {
  if (!per_sample_weights.has_value()) {
    return tensor;
  }
  mlir::MlirOp psw = *per_sample_weights;
  if (GetTensorTypeOrDie(psw).getRank() != 1) {
    psw = Flatten(psw);
  }
  if (needs_upcast) {
    TT_ASSIGN_OR_RETURN(psw, PromoteFloatDtype(psw));
  }
  TT_ASSIGN_OR_RETURN(auto psw_unsqueezed, Unsqueeze(psw, 1));
  TT_ASSIGN_OR_RETURN(auto psw_bcst, BroadcastIfNeeded(psw_unsqueezed, tensor));
  return mlir::stablehlo::Mul(tensor, psw_bcst);
}

struct MaskedPaddingResult {
  mlir::MlirOp gathered;
  mlir::MlirOp is_padding;
};

absl::StatusOr<MaskedPaddingResult> MaskPaddingIndices(
    mlir::MlirBuilder& builder, mlir::MlirOp gathered,
    mlir::MlirOp flattened_indices, int64_t padding_idx, int64_t mode,
    mlir::Type acc_elem_type) {
  if (padding_idx < 0) {
    return MaskedPaddingResult{gathered, mlir::MlirOp()};
  }
  auto pad_idx_scalar = MakeScalarConstant(
      builder, padding_idx,
      GetTensorTypeOrDie(flattened_indices).getElementType());
  TT_ASSIGN_OR_RETURN(auto pad_idx_bcst,
                      BroadcastIfNeeded(pad_idx_scalar, flattened_indices));
  auto is_padding =
      mlir::stablehlo::Compare(flattened_indices, pad_idx_bcst,
                               mlir::stablehlo::ComparisonDirection::EQ);
  TT_ASSIGN_OR_RETURN(auto is_pad_u, Unsqueeze(is_padding, 1));
  TT_ASSIGN_OR_RETURN(auto mask_bcst, BroadcastIfNeeded(is_pad_u, gathered));
  auto fill_scalar =
      MakeScalarConstant(builder, mode == 2 ? -INFINITY : 0.0, acc_elem_type);
  TT_ASSIGN_OR_RETURN(auto fill_bcst, BroadcastIfNeeded(fill_scalar, gathered));
  return MaskedPaddingResult{
      mlir::stablehlo::Select(mask_bcst, fill_bcst, gathered), is_padding};
}

absl::StatusOr<mlir::MlirOp> BuildBagSize(
    mlir::MlirBuilder& builder, mlir::MlirOp bag_size_init,
    mlir::MlirOp scatter_indices, int64_t num_indices, int64_t padding_idx,
    mlir::MlirOp is_padding, mlir::Type i64_type) {
  mlir::MlirOp ones_indices = mlir::stablehlo::Constant(
      builder, mlir::DenseElementsAttr::get(
                   mlir::RankedTensorType::get({num_indices}, i64_type),
                   builder.getOpBuilder().getIntegerAttr(i64_type, 1)));
  if (padding_idx >= 0) {
    auto zero_i64 = MakeScalarConstant(builder, 0, i64_type);
    TT_ASSIGN_OR_RETURN(auto zero_i64_bcst,
                        BroadcastIfNeeded(zero_i64, ones_indices));
    ones_indices =
        mlir::stablehlo::Select(is_padding, zero_i64_bcst, ones_indices);
  }
  return BuildSimpleScatter(builder, bag_size_init, scatter_indices,
                            ones_indices, HasWindow::kNo);
}

absl::StatusOr<mlir::MlirOp> ScaleByBagSize(mlir::MlirBuilder& builder,
                                            mlir::MlirOp output,
                                            mlir::MlirOp bag_size,
                                            mlir::Type acc_elem_type) {
  auto bag_size_float =
      mlir::stablehlo::ConvertElementType(bag_size, acc_elem_type);
  TT_ASSIGN_OR_RETURN(auto bs_unsqueezed, Unsqueeze(bag_size_float, 1));
  TT_ASSIGN_OR_RETURN(auto bs_bcst, BroadcastIfNeeded(bs_unsqueezed, output));
  auto zeros_float = MakeScalarConstant(builder, 0.0, acc_elem_type);
  TT_ASSIGN_OR_RETURN(auto zeros_bcst, BroadcastIfNeeded(zeros_float, bs_bcst));
  auto is_empty = mlir::stablehlo::Compare(
      bs_bcst, zeros_bcst, mlir::stablehlo::ComparisonDirection::EQ);
  auto ones_float = MakeScalarConstant(builder, 1.0, acc_elem_type);
  TT_ASSIGN_OR_RETURN(auto ones_bcst, BroadcastIfNeeded(ones_float, bs_bcst));
  auto safe_divisor = mlir::stablehlo::Select(is_empty, ones_bcst, bs_bcst);
  return mlir::stablehlo::Div(output, safe_divisor);
}

absl::StatusOr<mlir::MlirOp> BuildMeanBagBackward(mlir::MlirBuilder& builder,
                                                  mlir::MlirOp grad_indices,
                                                  mlir::MlirOp bag_size,
                                                  mlir::MlirOp offset2bag,
                                                  mlir::MlirOp zeros_float,
                                                  mlir::Type acc_elem_type) {
  auto bag_size_float =
      mlir::stablehlo::ConvertElementType(bag_size, acc_elem_type);
  TT_ASSIGN_OR_RETURN(
      auto bs_g, BuildEmbeddingGather(builder, bag_size_float, offset2bag));
  TT_ASSIGN_OR_RETURN(auto bs_u, Unsqueeze(bs_g, 1));
  TT_ASSIGN_OR_RETURN(auto bs_b, BroadcastIfNeeded(bs_u, grad_indices));
  TT_ASSIGN_OR_RETURN(auto zeros_bcst_bs, BroadcastIfNeeded(zeros_float, bs_b));
  auto is_zero = mlir::stablehlo::Compare(
      bs_b, zeros_bcst_bs, mlir::stablehlo::ComparisonDirection::EQ);
  auto ones_float = MakeScalarConstant(builder, 1.0, acc_elem_type);
  TT_ASSIGN_OR_RETURN(auto ones_bcst, BroadcastIfNeeded(ones_float, bs_b));
  auto safe_bs = mlir::stablehlo::Select(is_zero, ones_bcst, bs_b);
  return mlir::stablehlo::Div(grad_indices, safe_bs);
}

absl::StatusOr<mlir::MlirOp> BuildMaxBagBackward(
    mlir::MlirBuilder& builder, mlir::MlirOp acc_grad, mlir::MlirOp max_indices,
    int64_t num_indices, int64_t emb_dim, mlir::MlirOp zeros_float,
    mlir::Type acc_elem_type, mlir::Type i64_type) {
  auto gi_init = mlir::stablehlo::Constant(
      builder,
      mlir::DenseElementsAttr::get(
          mlir::RankedTensorType::get({num_indices, emb_dim}, acc_elem_type),
          builder.getOpBuilder().getZeroAttr(acc_elem_type)));
  auto d1i = mlir::stablehlo::Iota(
      builder, mlir::RankedTensorType::get({emb_dim}, i64_type), 0);
  TT_ASSIGN_OR_RETURN(auto d1b, BroadcastIfNeeded(d1i, max_indices));
  TT_ASSIGN_OR_RETURN(auto mu, Unsqueeze(max_indices, 2));
  TT_ASSIGN_OR_RETURN(auto d1u, Unsqueeze(d1b, 2));
  auto coords = mlir::stablehlo::Concatenate(builder, {mu, d1u}, 2);
  auto neg1_scalar = MakeScalarConstant(builder, -1, i64_type);
  TT_ASSIGN_OR_RETURN(auto neg1_bcst,
                      BroadcastIfNeeded(neg1_scalar, max_indices));
  auto is_v = mlir::stablehlo::Compare(
      max_indices, neg1_bcst, mlir::stablehlo::ComparisonDirection::GT);
  TT_ASSIGN_OR_RETURN(auto isv_u, Unsqueeze(is_v, 2));
  TT_ASSIGN_OR_RETURN(auto isv_b, BroadcastIfNeeded(isv_u, coords));
  auto zero_c = mlir::stablehlo::Constant(
      builder, mlir::DenseElementsAttr::get(
                   GetTensorTypeOrDie(coords),
                   builder.getOpBuilder().getI64IntegerAttr(0)));
  auto safe_coords = mlir::stablehlo::Select(isv_b, coords, zero_c);
  auto body = [acc_elem_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        acc_elem_type, rb.getRegion(), rb.getOpBuilder());
  };
  TT_ASSIGN_OR_RETURN(auto ag_zero_bcst,
                      BroadcastIfNeeded(zeros_float, acc_grad));
  auto safe_updates = mlir::stablehlo::Select(is_v, acc_grad, ag_zero_bcst);
  auto scatter_dims = mlir::stablehlo::ScatterDimensionNumbersAttr::get(
      &builder.getContext(), {}, {0, 1}, {}, {}, {0, 1}, 2);
  return mlir::stablehlo::Scatter({gi_init}, safe_coords, {safe_updates}, body,
                                  scatter_dims)[0];
}

absl::StatusOr<mlir::MlirOp> ZeroOutPaddingGradIndices(
    mlir::MlirBuilder& builder, mlir::MlirOp grad_indices,
    mlir::MlirOp flattened_indices, int64_t padding_idx,
    mlir::MlirOp zeros_float) {
  if (padding_idx < 0) {
    return grad_indices;
  }
  auto pad_scalar = MakeScalarConstant(
      builder, padding_idx,
      GetTensorTypeOrDie(flattened_indices).getElementType());
  TT_ASSIGN_OR_RETURN(auto pad_bcst,
                      BroadcastIfNeeded(pad_scalar, flattened_indices));
  auto is_p = mlir::stablehlo::Compare(
      flattened_indices, pad_bcst, mlir::stablehlo::ComparisonDirection::EQ);
  TT_ASSIGN_OR_RETURN(auto isp_u, Unsqueeze(is_p, 1));
  TT_ASSIGN_OR_RETURN(auto mask_bcst, BroadcastIfNeeded(isp_u, grad_indices));
  TT_ASSIGN_OR_RETURN(auto fill_bcst,
                      BroadcastIfNeeded(zeros_float, grad_indices));
  return mlir::stablehlo::Select(mask_bcst, fill_bcst, grad_indices);
}

absl::StatusOr<mlir::MlirOp> ScaleGradByFrequency(
    mlir::MlirBuilder& builder, mlir::MlirOp grad_indices,
    mlir::MlirOp flattened_indices, int64_t num_indices, int64_t nw,
    mlir::MlirOp zeros_float, mlir::Type acc_elem_type, mlir::Type i64_type) {
  auto counts_init = mlir::stablehlo::Constant(
      builder, mlir::DenseElementsAttr::get(
                   mlir::RankedTensorType::get({nw}, i64_type),
                   builder.getOpBuilder().getZeroAttr(i64_type)));
  TT_ASSIGN_OR_RETURN(auto fi_u, Unsqueeze(flattened_indices, 1));
  auto ones_indices_v = mlir::stablehlo::Constant(
      builder, mlir::DenseElementsAttr::get(
                   mlir::RankedTensorType::get({num_indices}, i64_type),
                   builder.getOpBuilder().getIntegerAttr(i64_type, 1)));
  auto counts = BuildSimpleScatter(builder, counts_init, fi_u, ones_indices_v,
                                   HasWindow::kNo);
  auto counts_acc = mlir::stablehlo::ConvertElementType(counts, acc_elem_type);
  TT_ASSIGN_OR_RETURN(
      auto ci_g, BuildEmbeddingGather(builder, counts_acc, flattened_indices));
  TT_ASSIGN_OR_RETURN(auto ci_u, Unsqueeze(ci_g, 1));
  TT_ASSIGN_OR_RETURN(auto ci_b, BroadcastIfNeeded(ci_u, grad_indices));
  TT_ASSIGN_OR_RETURN(auto zeros_bcst_c, BroadcastIfNeeded(zeros_float, ci_b));
  auto is_z = mlir::stablehlo::Compare(
      ci_b, zeros_bcst_c, mlir::stablehlo::ComparisonDirection::EQ);
  auto ones_float = MakeScalarConstant(builder, 1.0, acc_elem_type);
  TT_ASSIGN_OR_RETURN(auto ones_bcst, BroadcastIfNeeded(ones_float, ci_b));
  auto safe_c = mlir::stablehlo::Select(is_z, ones_bcst, ci_b);
  return mlir::stablehlo::Div(grad_indices, safe_c);
}

absl::StatusOr<mlir::MlirOp> ResolveMeanBagSize(
    mlir::MlirBuilder& builder, mlir::MlirOp bag_size,
    mlir::MlirOp active_offset2bag, mlir::MlirOp flattened_indices,
    int64_t batch_size, int64_t num_indices, int64_t padding_idx,
    mlir::Type i64_type) {
  if (GetTensorTypeOrDie(bag_size).getDimSize(0) > 0) {
    return bag_size;
  }
  auto bag_size_init = mlir::stablehlo::Constant(
      builder, mlir::DenseElementsAttr::get(
                   mlir::RankedTensorType::get({batch_size}, i64_type),
                   builder.getOpBuilder().getZeroAttr(i64_type)));
  TT_ASSIGN_OR_RETURN(auto scatter_indices, Unsqueeze(active_offset2bag, 1));
  mlir::MlirOp is_padding;
  if (padding_idx >= 0) {
    auto pad_scalar = MakeScalarConstant(
        builder, padding_idx,
        GetTensorTypeOrDie(flattened_indices).getElementType());
    TT_ASSIGN_OR_RETURN(auto pad_bcst,
                        BroadcastIfNeeded(pad_scalar, flattened_indices));
    is_padding = mlir::stablehlo::Compare(
        flattened_indices, pad_bcst, mlir::stablehlo::ComparisonDirection::EQ);
  }
  return BuildBagSize(builder, bag_size_init, scatter_indices, num_indices,
                      padding_idx, is_padding, i64_type);
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildEmbeddingShlo(mlir::MlirOp weight,
                                                mlir::MlirOp indices) {
  mlir::MlirBuilder& builder = weight.getBuilder();
  const auto weight_type = GetTensorTypeOrDie(weight);
  const auto indices_type = GetTensorTypeOrDie(indices);
  const int64_t num_embeddings = weight_type.getDimSize(0);
  const int64_t embedding_dim = weight_type.getDimSize(1);
  const int64_t indices_rank = indices_type.getRank();
  if (num_embeddings == 0) {
    Dimensions out_shape;
    if (indices_rank == 0) {
      out_shape = {embedding_dim};
    } else {
      out_shape = CopyIntVector(indices_type.getShape());
      out_shape.push_back(embedding_dim);
    }
    const auto out_type =
        mlir::RankedTensorType::get(out_shape, weight_type.getElementType());
    return mlir::stablehlo::Constant(
        builder, mlir::DenseElementsAttr::get(
                     out_type, builder.getOpBuilder().getZeroAttr(
                                   weight_type.getElementType())));
  }
  if (indices_rank == 0) {
    TT_ASSIGN_OR_RETURN(const mlir::MlirOp unsqueezed, Unsqueeze(indices, 0));
    TT_ASSIGN_OR_RETURN(const mlir::MlirOp gathered,
                        BuildEmbeddingGather(builder, weight, unsqueezed));
    return Squeeze(gathered, {0});
  }
  return BuildEmbeddingGather(builder, weight, indices);
}

absl::StatusOr<MlirOpResults<4>> BuildEmbeddingBagShlo(
    mlir::MlirOp weight, mlir::MlirOp indices, mlir::MlirOp offsets,
    bool scale_grad_by_freq, int64_t mode, bool sparse,
    std::optional<mlir::MlirOp> per_sample_weights, bool include_last_offset,
    int64_t padding_idx) {
  mlir::MlirBuilder& builder = weight.getBuilder();
  const auto weight_type = GetTensorTypeOrDie(weight);
  const auto weight_elem_type = weight_type.getElementType();
  const int64_t emb_dim = weight_type.getDimSize(1);
  int64_t batch_size = GetTensorTypeOrDie(offsets).getDimSize(0);
  if (include_last_offset) batch_size -= 1;

  mlir::Type i64_type = builder.getOpBuilder().getI64Type();
  bool needs_upcast = weight_elem_type.isF16() || weight_elem_type.isBF16();
  mlir::Type acc_elem_type =
      needs_upcast ? builder.getOpBuilder().getF32Type() : weight_elem_type;
  mlir::MlirOp acc_weight = weight;
  if (needs_upcast) {
    TT_ASSIGN_OR_RETURN(acc_weight, PromoteFloatDtype(acc_weight));
  }
  mlir::MlirOp flattened_indices = indices;
  if (GetTensorTypeOrDie(indices).getRank() != 1) {
    flattened_indices = Flatten(indices);
  }
  const int64_t num_indices =
      GetTensorTypeOrDie(flattened_indices).getDimSize(0);
  auto output_init = mlir::stablehlo::Constant(
      builder,
      mlir::DenseElementsAttr::get(
          mlir::RankedTensorType::get({batch_size, emb_dim}, acc_elem_type),
          builder.getOpBuilder().getZeroAttr(acc_elem_type)));
  auto bag_size_init = mlir::stablehlo::Constant(
      builder, mlir::DenseElementsAttr::get(
                   mlir::RankedTensorType::get({batch_size}, i64_type),
                   builder.getOpBuilder().getZeroAttr(i64_type)));
  mlir::RankedTensorType max_indices_type =
      (mode == 2) ? mlir::RankedTensorType::get({batch_size, emb_dim}, i64_type)
                  : mlir::RankedTensorType::get({batch_size}, i64_type);
  auto max_indices_init = mlir::stablehlo::Constant(
      builder,
      mlir::DenseElementsAttr::get(
          max_indices_type, builder.getOpBuilder().getZeroAttr(i64_type)));
  if (num_indices == 0) {
    auto empty_idx = mlir::stablehlo::Constant(
        builder, mlir::DenseElementsAttr::get(
                     mlir::RankedTensorType::get({0}, i64_type),
                     builder.getOpBuilder().getZeroAttr(i64_type)));
    return MlirOpResults<4>(
        {output_init, empty_idx, bag_size_init, max_indices_init});
  }
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp gathered,
      BuildEmbeddingGather(builder, acc_weight, flattened_indices));
  TT_ASSIGN_OR_RETURN(
      gathered,
      ApplyPerSampleWeights(gathered, per_sample_weights, needs_upcast));
  TT_ASSIGN_OR_RETURN((auto [masked_gathered, is_padding]),
                      MaskPaddingIndices(builder, gathered, flattened_indices,
                                         padding_idx, mode, acc_elem_type));
  gathered = masked_gathered;
  TT_ASSIGN_OR_RETURN(mlir::MlirOp offset2bag,
                      BuildOffset2Bag(builder, flattened_indices, offsets,
                                      i64_type, batch_size));
  TT_ASSIGN_OR_RETURN(auto scatter_indices, Unsqueeze(offset2bag, 1));
  mlir::MlirOp output;
  mlir::MlirOp final_bag_size;
  mlir::MlirOp max_indices;

  auto bag_mode = static_cast<EmbeddingBagMode>(mode);

  switch (bag_mode) {
    case EmbeddingBagMode::kSum: {
      output = BuildSimpleScatter(builder, output_init, scatter_indices,
                                  gathered, HasWindow::kYes);
      final_bag_size = bag_size_init;
      max_indices = bag_size_init;
      break;
    }
    case EmbeddingBagMode::kMean: {
      output = BuildSimpleScatter(builder, output_init, scatter_indices,
                                  gathered, HasWindow::kYes);
      TT_ASSIGN_OR_RETURN(
          final_bag_size,
          BuildBagSize(builder, bag_size_init, scatter_indices, num_indices,
                       padding_idx, is_padding, i64_type));
      max_indices = final_bag_size;
      TT_ASSIGN_OR_RETURN(
          output,
          ScaleByBagSize(builder, output, final_bag_size, acc_elem_type));
      break;
    }
    case EmbeddingBagMode::kMax: {
      auto neg_inf_scalar =
          MakeScalarConstant(builder, -INFINITY, acc_elem_type);
      TT_ASSIGN_OR_RETURN(auto neg_inf_bcst,
                          BroadcastIfNeeded(neg_inf_scalar, output_init));
      TT_ASSIGN_OR_RETURN(
          auto max_res,
          BuildMaxAggregation(builder, gathered, scatter_indices, neg_inf_bcst,
                              max_indices_init, i64_type, acc_elem_type,
                              offset2bag));
      output = max_res.output;
      max_indices = max_res.max_indices;

      auto is_neg_inf = mlir::stablehlo::Compare(
          output, neg_inf_bcst, mlir::stablehlo::ComparisonDirection::EQ);
      auto zero_scalar = MakeScalarConstant(builder, 0.0, acc_elem_type);
      TT_ASSIGN_OR_RETURN(auto zero_bcst,
                          BroadcastIfNeeded(zero_scalar, output));
      output = mlir::stablehlo::Select(is_neg_inf, zero_bcst, output);

      TT_ASSIGN_OR_RETURN(
          final_bag_size,
          BuildBagSize(builder, bag_size_init, scatter_indices, num_indices,
                       padding_idx, is_padding, i64_type));
      break;
    }
  }

  if (needs_upcast) {
    output = mlir::stablehlo::ConvertElementType(output, weight_elem_type);
  }
  return MlirOpResults<4>({output, offset2bag, final_bag_size, max_indices});
}

absl::StatusOr<mlir::MlirOp> BuildEmbeddingBagBackwardShlo(
    mlir::MlirOp grad, mlir::MlirOp indices, mlir::MlirOp offsets,
    mlir::MlirOp offset2bag, mlir::MlirOp bag_size, mlir::MlirOp max_indices,
    at::SymInt num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse,
    std::optional<mlir::MlirOp> per_sample_weights, int64_t padding_idx) {
  mlir::MlirBuilder& builder = grad.getBuilder();
  const auto grad_type = GetTensorTypeOrDie(grad);
  const auto grad_elem_type = grad_type.getElementType();
  const int64_t emb_dim = grad_type.getDimSize(1),
                nw = num_weights.expect_int();
  bool needs_upcast = grad_elem_type.isF16() || grad_elem_type.isBF16();
  mlir::Type acc_elem_type =
      needs_upcast ? builder.getOpBuilder().getF32Type() : grad_elem_type;
  mlir::Type i64_type = builder.getOpBuilder().getI64Type();
  mlir::MlirOp acc_grad = grad;
  if (needs_upcast) {
    TT_ASSIGN_OR_RETURN(acc_grad, PromoteFloatDtype(acc_grad));
  }
  auto grad_weight_init = mlir::stablehlo::Constant(
      builder, mlir::DenseElementsAttr::get(
                   mlir::RankedTensorType::get({nw, emb_dim}, acc_elem_type),
                   builder.getOpBuilder().getZeroAttr(acc_elem_type)));
  mlir::MlirOp flattened_indices = indices;
  if (GetTensorTypeOrDie(indices).getRank() != 1) {
    flattened_indices = Flatten(indices);
  }
  const int64_t num_indices =
      GetTensorTypeOrDie(flattened_indices).getDimSize(0);
  if (num_indices == 0) {
    return needs_upcast ? mlir::stablehlo::ConvertElementType(grad_weight_init,
                                                              grad_elem_type)
                        : grad_weight_init;
  }
  auto zeros_float = MakeScalarConstant(builder, 0.0, acc_elem_type);
  auto bag_mode = static_cast<EmbeddingBagMode>(mode);

  if (bag_mode == EmbeddingBagMode::kMax) {
    auto bag_size_acc = mlir::stablehlo::ConvertElementType(
        bag_size, GetTensorTypeOrDie(flattened_indices).getElementType());
    auto zero_idx_scalar = MakeScalarConstant(
        builder, 0, GetTensorTypeOrDie(flattened_indices).getElementType());
    TT_ASSIGN_OR_RETURN(auto zero_idx_bcst,
                        BroadcastIfNeeded(zero_idx_scalar, bag_size_acc));
    auto bag_is_empty = mlir::stablehlo::Compare(
        bag_size_acc, zero_idx_bcst, mlir::stablehlo::ComparisonDirection::EQ);
    TT_ASSIGN_OR_RETURN(auto bie_u, Unsqueeze(bag_is_empty, 1));
    TT_ASSIGN_OR_RETURN(auto bie_b, BroadcastIfNeeded(bie_u, acc_grad));
    TT_ASSIGN_OR_RETURN(auto zeros_bcst,
                        BroadcastIfNeeded(zeros_float, acc_grad));
    acc_grad = mlir::stablehlo::Select(bie_b, zeros_bcst, acc_grad);
  }

  mlir::MlirOp grad_indices;
  switch (bag_mode) {
    case EmbeddingBagMode::kSum:
    case EmbeddingBagMode::kMean: {
      TT_ASSIGN_OR_RETURN(grad_indices,
                          BuildEmbeddingGather(builder, acc_grad, offset2bag));
      if (bag_mode == EmbeddingBagMode::kMean) {
        TT_ASSIGN_OR_RETURN(
            grad_indices,
            BuildMeanBagBackward(builder, grad_indices, bag_size, offset2bag,
                                 zeros_float, acc_elem_type));
      }
      TT_ASSIGN_OR_RETURN(
          grad_indices, ApplyPerSampleWeights(grad_indices, per_sample_weights,
                                              needs_upcast));
      break;
    }
    case EmbeddingBagMode::kMax: {
      TT_ASSIGN_OR_RETURN(
          grad_indices,
          BuildMaxBagBackward(builder, acc_grad, max_indices, num_indices,
                              emb_dim, zeros_float, acc_elem_type, i64_type));
      break;
    }
  }
  TT_ASSIGN_OR_RETURN(
      grad_indices,
      ZeroOutPaddingGradIndices(builder, grad_indices, flattened_indices,
                                padding_idx, zeros_float));
  if (scale_grad_by_freq) {
    TT_ASSIGN_OR_RETURN(
        grad_indices, ScaleGradByFrequency(
                          builder, grad_indices, flattened_indices, num_indices,
                          nw, zeros_float, acc_elem_type, i64_type));
  }
  TT_ASSIGN_OR_RETURN(auto si_v, Unsqueeze(flattened_indices, 1));
  auto grad_weight = BuildSimpleScatter(builder, grad_weight_init, si_v,
                                        grad_indices, HasWindow::kYes);
  if (needs_upcast) {
    grad_weight =
        mlir::stablehlo::ConvertElementType(grad_weight, grad_elem_type);
  }
  return grad_weight;
}

absl::StatusOr<mlir::MlirOp> BuildEmbeddingBagSparseBackwardIndexGradShlo(
    mlir::MlirOp grad, mlir::MlirOp indices, mlir::MlirOp offsets,
    mlir::MlirOp offset2bag, mlir::MlirOp bag_size, int64_t mode,
    std::optional<mlir::MlirOp> per_sample_weights, int64_t padding_idx) {
  mlir::MlirBuilder& builder = grad.getBuilder();
  const auto grad_type = GetTensorTypeOrDie(grad);
  const auto grad_elem_type = grad_type.getElementType();
  const int64_t batch_size = grad_type.getDimSize(0);
  const int64_t emb_dim = grad_type.getDimSize(1);
  bool needs_upcast = grad_elem_type.isF16() || grad_elem_type.isBF16();
  mlir::Type acc_elem_type =
      needs_upcast ? builder.getOpBuilder().getF32Type() : grad_elem_type;
  mlir::Type i64_type = builder.getOpBuilder().getI64Type();
  mlir::MlirOp acc_grad = grad;
  if (needs_upcast) {
    TT_ASSIGN_OR_RETURN(acc_grad, PromoteFloatDtype(acc_grad));
  }
  mlir::MlirOp flattened_indices = indices;
  if (GetTensorTypeOrDie(indices).getRank() != 1) {
    flattened_indices = Flatten(indices);
  }
  const int64_t num_indices =
      GetTensorTypeOrDie(flattened_indices).getDimSize(0);
  if (num_indices == 0) {
    return mlir::stablehlo::Constant(
        builder, mlir::DenseElementsAttr::get(
                     mlir::RankedTensorType::get({0, emb_dim}, grad_elem_type),
                     builder.getOpBuilder().getZeroAttr(grad_elem_type)));
  }
  mlir::MlirOp active_offset2bag = offset2bag;
  if (GetTensorTypeOrDie(offset2bag).getDimSize(0) == 0) {
    TT_ASSIGN_OR_RETURN(active_offset2bag,
                        BuildOffset2Bag(builder, flattened_indices, offsets,
                                        i64_type, batch_size));
  }
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp index_grad,
      BuildEmbeddingGather(builder, acc_grad, active_offset2bag));
  if (static_cast<EmbeddingBagMode>(mode) == EmbeddingBagMode::kMean) {
    auto zeros_float = MakeScalarConstant(builder, 0.0, acc_elem_type);
    TT_ASSIGN_OR_RETURN(mlir::MlirOp active_bag_size,
                        ResolveMeanBagSize(builder, bag_size, active_offset2bag,
                                           flattened_indices, batch_size,
                                           num_indices, padding_idx, i64_type));
    TT_ASSIGN_OR_RETURN(
        index_grad,
        BuildMeanBagBackward(builder, index_grad, active_bag_size,
                             active_offset2bag, zeros_float, acc_elem_type));
  }
  TT_ASSIGN_OR_RETURN(
      index_grad,
      ApplyPerSampleWeights(index_grad, per_sample_weights, needs_upcast));
  if (needs_upcast) {
    index_grad =
        mlir::stablehlo::ConvertElementType(index_grad, grad_elem_type);
  }
  return index_grad;
}

absl::StatusOr<mlir::MlirOp> BuildEmbeddingDenseBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp indices, at::SymInt num_weights,
    at::SymInt padding_idx, bool scale_grad_by_freq) {
  mlir::MlirBuilder& builder = grad_output.getBuilder();
  const auto grad_tensor_type = GetTensorTypeOrDie(grad_output);
  const auto grad_element_type = grad_tensor_type.getElementType();
  const bool should_promote_to_float32 =
      scale_grad_by_freq &&
      (grad_element_type.isF16() || grad_element_type.isBF16());
  mlir::Type accumulation_element_type =
      should_promote_to_float32 ? builder.getOpBuilder().getF32Type()
                                : grad_element_type;
  mlir::Type int64_type = builder.getOpBuilder().getI64Type();
  mlir::MlirOp grad_output_converted = grad_output;
  if (should_promote_to_float32) {
    TT_ASSIGN_OR_RETURN(grad_output_converted,
                        PromoteFloatDtype(grad_output_converted));
  }
  mlir::MlirOp flattened_indices = indices;
  if (GetTensorTypeOrDie(indices).getRank() != 1) {
    flattened_indices = Flatten(indices);
  }
  const int64_t num_indices =
      GetTensorTypeOrDie(flattened_indices).getDimSize(0);
  const int64_t num_weights_value = num_weights.expect_int();
  const int64_t embedding_dim =
      grad_tensor_type.getDimSize(grad_tensor_type.getRank() - 1);

  auto initial_grad_weights = mlir::stablehlo::Constant(
      builder,
      mlir::DenseElementsAttr::get(
          mlir::RankedTensorType::get({num_weights_value, embedding_dim},
                                      accumulation_element_type),
          builder.getOpBuilder().getZeroAttr(accumulation_element_type)));
  if (num_indices == 0) {
    return should_promote_to_float32
               ? mlir::stablehlo::ConvertElementType(initial_grad_weights,
                                                     grad_element_type)
               : initial_grad_weights;
  }

  mlir::MlirOp formatted_grad_output = grad_output_converted;
  if (grad_tensor_type.getRank() == 1) {
    TT_ASSIGN_OR_RETURN(formatted_grad_output,
                        Unsqueeze(grad_output_converted, 0));
  } else if (grad_tensor_type.getRank() > 2) {
    TT_ASSIGN_OR_RETURN(
        formatted_grad_output,
        ReshapeFromStaticDimensions(grad_output_converted,
                                    CopyIntVector(grad_tensor_type.getShape()),
                                    {num_indices, embedding_dim}));
  }

  if (padding_idx.expect_int() >= 0) {
    auto padding_idx_scalar = MakeScalarConstant(
        builder, padding_idx.expect_int(),
        GetTensorTypeOrDie(flattened_indices).getElementType());
    TT_ASSIGN_OR_RETURN(
        auto padding_idx_broadcasted,
        BroadcastIfNeeded(padding_idx_scalar, flattened_indices));
    auto is_padding_mask =
        mlir::stablehlo::Compare(flattened_indices, padding_idx_broadcasted,
                                 mlir::stablehlo::ComparisonDirection::EQ);
    TT_ASSIGN_OR_RETURN(auto mask_unsqueezed, Unsqueeze(is_padding_mask, 1));
    TT_ASSIGN_OR_RETURN(
        auto mask_broadcasted,
        BroadcastIfNeeded(mask_unsqueezed, formatted_grad_output));
    auto zeros_constant =
        MakeScalarConstant(builder, 0.0, accumulation_element_type);
    TT_ASSIGN_OR_RETURN(
        auto zeros_broadcasted,
        BroadcastIfNeeded(zeros_constant, formatted_grad_output));
    formatted_grad_output = mlir::stablehlo::Select(
        mask_broadcasted, zeros_broadcasted, formatted_grad_output);
  }

  TT_ASSIGN_OR_RETURN(auto scatter_indices, Unsqueeze(flattened_indices, 1));
  auto grad_weights =
      BuildSimpleScatter(builder, initial_grad_weights, scatter_indices,
                         formatted_grad_output, HasWindow::kYes);

  if (scale_grad_by_freq) {
    auto initial_count_weights = mlir::stablehlo::Constant(
        builder,
        mlir::DenseElementsAttr::get(
            mlir::RankedTensorType::get({num_weights_value}, int64_type),
            builder.getOpBuilder().getZeroAttr(int64_type)));
    auto ones_indices_vector = mlir::stablehlo::Constant(
        builder, mlir::DenseElementsAttr::get(
                     mlir::RankedTensorType::get({num_indices}, int64_type),
                     builder.getOpBuilder().getIntegerAttr(int64_type, 1)));
    auto weight_occurrence_counts =
        BuildSimpleScatter(builder, initial_count_weights, scatter_indices,
                           ones_indices_vector, HasWindow::kNo);
    auto counts_converted = mlir::stablehlo::ConvertElementType(
        weight_occurrence_counts, accumulation_element_type);
    TT_ASSIGN_OR_RETURN(auto counts_unsqueezed, Unsqueeze(counts_converted, 1));
    TT_ASSIGN_OR_RETURN(auto counts_broadcasted,
                        BroadcastIfNeeded(counts_unsqueezed, grad_weights));
    auto zeros_constant =
        MakeScalarConstant(builder, 0.0, accumulation_element_type);
    TT_ASSIGN_OR_RETURN(auto zeros_counts_broadcasted,
                        BroadcastIfNeeded(zeros_constant, counts_broadcasted));
    auto is_zero_counts_mask =
        mlir::stablehlo::Compare(counts_broadcasted, zeros_counts_broadcasted,
                                 mlir::stablehlo::ComparisonDirection::EQ);
    auto ones_constant =
        MakeScalarConstant(builder, 1.0, accumulation_element_type);
    TT_ASSIGN_OR_RETURN(auto ones_broadcasted,
                        BroadcastIfNeeded(ones_constant, counts_broadcasted));
    auto safe_divisor_counts = mlir::stablehlo::Select(
        is_zero_counts_mask, ones_broadcasted, counts_broadcasted);
    grad_weights = mlir::stablehlo::Div(grad_weights, safe_divisor_counts);
  }

  return should_promote_to_float32 ? mlir::stablehlo::ConvertElementType(
                                         grad_weights, grad_element_type)
                                   : grad_weights;
}

absl::StatusOr<mlir::MlirOp> BuildEmbeddingRenormShlo(mlir::MlirOp weight,
                                                      mlir::MlirOp indices,
                                                      double max_norm,
                                                      double norm_type) {
  const auto wt = GetTensorTypeOrDie(weight);
  const int64_t ed = wt.getDimSize(1);
  const auto it = GetTensorTypeOrDie(indices);
  if (it.getRank() == 0) {
    mlir::MlirOp z =
        MakeScalarConstant(weight.getBuilder(), 0, mlir::ElementType::I64);
    mlir::MlirOp st[] = {indices, z};
    int64_t sz[] = {1, ed};
    TT_ASSIGN_OR_RETURN(
        auto row, BuildRenormRow(weight.getBuilder(),
                                 mlir::stablehlo::DynamicSlice(weight, st, sz),
                                 max_norm, norm_type));
    return mlir::stablehlo::Reshape(row, {ed});
  }
  TT_ASSIGN_OR_RETURN(
      auto rs, BuildEmbeddingGather(weight.getBuilder(), weight, indices));
  return BuildRenormRow(weight.getBuilder(), rs, max_norm, norm_type);
}

namespace {

void BuildSparseMaskDescIotaAscComparator(mlir::RegionBuilder& rb) {
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

}  // namespace

absl::StatusOr<std::array<mlir::MlirOp, 2>>
BuildEmbeddingSparseBackwardNoPadShlo(mlir::MlirOp grad_output,
                                      mlir::MlirOp indices) {
  const int64_t total_indices = GetTensorTypeOrDie(indices).getNumElements();
  const mlir::RankedTensorType grad_type = GetTensorTypeOrDie(grad_output);
  const int64_t num_features = grad_type.getDimSize(grad_type.getRank() - 1);
  mlir::MlirOp index_2d = mlir::stablehlo::Reshape(indices, {1, total_indices});
  TT_ASSIGN_OR_RETURN(index_2d, CastIfNeeded(index_2d, mlir::ElementType::I64));
  mlir::MlirOp values_2d =
      mlir::stablehlo::Reshape(grad_output, {total_indices, num_features});
  return std::array<mlir::MlirOp, 2>{index_2d, values_2d};
}

absl::StatusOr<std::array<mlir::MlirOp, 3>> BuildEmbeddingSparseBackwardPadShlo(
    mlir::MlirOp grad_output, mlir::MlirOp indices, const int64_t padding_idx) {
  auto& builder = grad_output.getBuilder();
  const int64_t total_indices = GetTensorTypeOrDie(indices).getNumElements();
  const mlir::RankedTensorType grad_type = GetTensorTypeOrDie(grad_output);
  const int64_t num_features = grad_type.getDimSize(grad_type.getRank() - 1);

  mlir::MlirOp flat_indices =
      mlir::stablehlo::Reshape(indices, {total_indices});
  TT_ASSIGN_OR_RETURN(flat_indices,
                      CastIfNeeded(flat_indices, mlir::ElementType::I64));
  mlir::MlirOp flat_grad =
      mlir::stablehlo::Reshape(grad_output, {total_indices, num_features});

  TT_ASSIGN_OR_RETURN(mlir::MlirOp pad_cst,
                      MakeConstant(builder, at::Scalar(padding_idx),
                                   mlir::ElementType::I64, {total_indices}));
  mlir::MlirOp mask = mlir::stablehlo::Compare(
      flat_indices, pad_cst, mlir::stablehlo::ComparisonDirection::NE);

  const auto i64_type = builder.getOpBuilder().getI64Type();
  mlir::MlirOp mask_i64 = mlir::stablehlo::ConvertElementType(mask, i64_type);
  mlir::MlirOp zero_i64 = MakeScalarConstant(builder, 0, i64_type);
  mlir::MlirOp count_op = mlir::stablehlo::Reduce(
      builder, {mask_i64}, {zero_i64},
      [](mlir::RegionBuilder& rb) {
        auto& b = rb.getOpBuilder();
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
            b.getI64Type(), rb.getRegion(), b);
      },
      {0})[0];

  const auto iota_type =
      mlir::makeTensorType(builder.getContext(), {total_indices}, i64_type);
  mlir::MlirOp iota = mlir::stablehlo::Iota(builder, iota_type, 0);
  const auto sorted = mlir::stablehlo::Sort(
      builder, {mask, iota}, BuildSparseMaskDescIotaAscComparator,
      /*dimension=*/0, /*is_stable=*/false);
  mlir::MlirOp compacted_indices_1d =
      BuildIndexSelectShlo(flat_indices, /*dim=*/0, sorted[1]);
  mlir::MlirOp compacted_indices_2d =
      mlir::stablehlo::Reshape(compacted_indices_1d, {1, total_indices});
  mlir::MlirOp compacted_values =
      BuildIndexSelectShlo(flat_grad, /*dim=*/0, sorted[1]);
  return std::array<mlir::MlirOp, 3>{compacted_indices_2d, compacted_values,
                                     count_op};
}

}  // namespace torch_tpu
