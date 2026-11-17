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

#include "csrc/ops/unique/unique.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/index_select/index_select.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/sort/sort.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

mlir::MlirOp FlattenIfNeeded(mlir::MlirOp input) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  if (input_type.getRank() == 1) {
    return input;
  }
  return mlir::stablehlo::Reshape(input, {input_type.getNumElements()});
}

absl::StatusOr<mlir::MlirOp> BuildUniqueMask(mlir::MlirBuilder& builder,
                                             mlir::MlirOp sorted_input) {
  const mlir::RankedTensorType sorted_type = GetTensorTypeOrDie(sorted_input);
  const int64_t n = sorted_type.getDimSize(0);
  if (n == 0) {
    return MakeZeroSizedTensor(builder, mlir::ElementType::PRED);
  }
  if (n == 1) {
    return MakeConstant(builder, 1, mlir::ElementType::PRED, {1});
  }

  mlir::MlirOp start = mlir::stablehlo::Slice(sorted_input, {1}, {n}, {1});
  mlir::MlirOp end = mlir::stablehlo::Slice(sorted_input, {0}, {n - 1}, {1});
  mlir::MlirOp diffs = mlir::stablehlo::Compare(
      start, end, mlir::stablehlo::ComparisonDirection::NE);

  mlir::MlirOp first_true =
      MakeConstant(builder, 1, mlir::ElementType::PRED, {1});
  return mlir::stablehlo::Concatenate(builder, {first_true, diffs}, 0);
}

absl::StatusOr<mlir::MlirOp> BuildCumsum1D(mlir::MlirBuilder& builder,
                                           mlir::MlirOp input) {
  const mlir::RankedTensorType type = GetTensorTypeOrDie(input);
  const int64_t n = type.getDimSize(0);
  auto window_dimensions =
      mlir::DenseI64ArrayAttr::get(&builder.getContext(), {n});
  auto window_strides =
      mlir::DenseI64ArrayAttr::get(&builder.getContext(), {1});

  mlir::SmallVector<int64_t, 2> padding_values = {n - 1, 0};
  auto padding = mlir::DenseIntElementsAttr::get(
      mlir::RankedTensorType::get({1, 2}, builder.getOpBuilder().getI64Type()),
      llvm::ArrayRef<int64_t>(padding_values));

  auto body_builder = [](mlir::RegionBuilder& rb) {
    auto& op_builder = rb.getOpBuilder();
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        op_builder.getI64Type(), rb.getRegion(), op_builder);
  };

  auto init_value =
      MakeScalarConstant(builder, 0, builder.getOpBuilder().getI64Type());
  return mlir::stablehlo::ReduceWindow(builder, {input}, {init_value},
                                       body_builder, window_dimensions,
                                       window_strides,
                                       /*base_dilations=*/{},
                                       /*window_dilations=*/{}, padding)[0];
}

// Holds a 2D view of a tensor with slices along `dim` moved to axis 0 and all
// remaining dimensions flattened into axis 1.
struct FlattenedSlices2D {
  mlir::MlirOp input_2d;
  int64_t num_slices = 0;
  int64_t slice_elements = 0;
};

FlattenedSlices2D TransposeAndFlattenSlices(mlir::MlirOp input,
                                            const int64_t dim) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const int64_t rank = input_type.getRank();
  const int64_t s = input_type.getDimSize(dim);

  mlir::MlirOp input_dim0 = input;
  if (dim != 0 && rank > 1) {
    Dimensions permutation;
    permutation.reserve(rank);
    permutation.push_back(dim);
    const Dimensions batch_dims = GetBatchDimensions(rank, dim);
    permutation.insert(permutation.end(), batch_dims.begin(), batch_dims.end());
    input_dim0 = mlir::stablehlo::Transpose(input, permutation);
  }

  int64_t slice_elements = 1;
  for (int64_t i = 0; i < rank; ++i) {
    if (i != dim) {
      slice_elements *= input_type.getDimSize(i);
    }
  }

  mlir::MlirOp input_2d =
      mlir::stablehlo::Reshape(input_dim0, {s, slice_elements});
  return FlattenedSlices2D{input_2d, s, slice_elements};
}

// GENERIC: Batch size limit for the StableHLO multi-key sort comparator
// builder (matching JAX's `jnp.lexsort` default `batch_size = 16`). Sorting
// up to 16 columns per radix sort step bounds comparator region size and
// compile-time unrolling while minimizing dynamic-slice loop iterations.
inline constexpr int64_t kLexSortBatchSize = 16;

struct ValCompareResult {
  mlir::Value lt;
  mlir::Value eq;
};

ValCompareResult BuildFloatStableColumnComparator(mlir::OpBuilder& b,
                                                  const mlir::Location loc,
                                                  const mlir::Value lhs_val,
                                                  const mlir::Value rhs_val) {
  // Use IEEE FLOAT comparison (not TOTALORDER) so that -0.0 == +0.0 is true
  // for lexicographical slice ordering.
  const mlir::Value float_lt = mlir::stablehlo::CompareOp::create(
      b, loc, lhs_val, rhs_val, mlir::stablehlo::ComparisonDirection::LT);
  const mlir::Value float_eq = mlir::stablehlo::CompareOp::create(
      b, loc, lhs_val, rhs_val, mlir::stablehlo::ComparisonDirection::EQ);
  const mlir::Value lhs_nan = mlir::stablehlo::CompareOp::create(
      b, loc, lhs_val, lhs_val, mlir::stablehlo::ComparisonDirection::NE);
  const mlir::Value rhs_nan = mlir::stablehlo::CompareOp::create(
      b, loc, rhs_val, rhs_val, mlir::stablehlo::ComparisonDirection::NE);
  const mlir::Value not_lhs_nan =
      mlir::stablehlo::NotOp::create(b, loc, lhs_nan);
  const mlir::Value lhs_not_nan_and_rhs_nan =
      mlir::stablehlo::AndOp::create(b, loc, not_lhs_nan, rhs_nan);
  const mlir::Value val_lt =
      mlir::stablehlo::OrOp::create(b, loc, float_lt, lhs_not_nan_and_rhs_nan);
  const mlir::Value both_nan =
      mlir::stablehlo::AndOp::create(b, loc, lhs_nan, rhs_nan);
  const mlir::Value val_eq =
      mlir::stablehlo::OrOp::create(b, loc, float_eq, both_nan);
  return ValCompareResult{val_lt, val_eq};
}

ValCompareResult BuildIntStableColumnComparator(mlir::OpBuilder& b,
                                                const mlir::Location loc,
                                                const mlir::Type elem_type,
                                                const mlir::Value lhs_val,
                                                const mlir::Value rhs_val) {
  const auto cmp_type =
      (elem_type.isUnsignedInteger() || elem_type.isSignlessInteger(1))
          ? mlir::stablehlo::ComparisonType::UNSIGNED
          : mlir::stablehlo::ComparisonType::SIGNED;
  const mlir::Value val_lt = mlir::stablehlo::CompareOp::create(
      b, loc, lhs_val, rhs_val, mlir::stablehlo::ComparisonDirection::LT,
      cmp_type);
  const mlir::Value val_eq = mlir::stablehlo::CompareOp::create(
      b, loc, lhs_val, rhs_val, mlir::stablehlo::ComparisonDirection::EQ,
      cmp_type);
  return ValCompareResult{val_lt, val_eq};
}

ValCompareResult CompareColumnElements(mlir::OpBuilder& b,
                                       const mlir::Location loc,
                                       const mlir::Type elem_type,
                                       const mlir::Value lhs_val,
                                       const mlir::Value rhs_val) {
  return mlir::isa<mlir::FloatType>(elem_type)
             ? BuildFloatStableColumnComparator(b, loc, lhs_val, rhs_val)
             : BuildIntStableColumnComparator(b, loc, elem_type, lhs_val,
                                              rhs_val);
}

void BuildMultiColumnComparator(const mlir::Type elem_type,
                                const int64_t num_cols,
                                mlir::RegionBuilder& rb) {
  auto& b = rb.getOpBuilder();
  const mlir::OpBuilder::InsertionGuard guard(b);
  const mlir::Location loc = rb.getRegion().getLoc();
  mlir::Block* const block = rb.getRegion().empty()
                                 ? b.createBlock(&rb.getRegion())
                                 : &rb.getRegion().front();
  const mlir::RankedTensorType scalar_elem =
      mlir::RankedTensorType::get({}, elem_type);
  const mlir::RankedTensorType scalar_i64 =
      mlir::RankedTensorType::get({}, b.getI64Type());

  std::vector<mlir::Type> arg_types;
  std::vector<mlir::Location> arg_locs(2 * (num_cols + 2), loc);
  arg_types.reserve(2 * (num_cols + 2));
  for (int64_t c = 0; c < num_cols; ++c) {
    arg_types.push_back(scalar_elem);
    arg_types.push_back(scalar_elem);
  }
  arg_types.push_back(scalar_i64);
  arg_types.push_back(scalar_i64);
  arg_types.push_back(scalar_i64);
  arg_types.push_back(scalar_i64);
  block->addArguments(arg_types, arg_locs);

  const mlir::Value lhs_pos = block->getArgument(2 * num_cols);
  const mlir::Value rhs_pos = block->getArgument(2 * num_cols + 1);
  mlir::Value res = mlir::stablehlo::CompareOp::create(
      b, loc, lhs_pos, rhs_pos, mlir::stablehlo::ComparisonDirection::LT,
      mlir::stablehlo::ComparisonType::SIGNED);

  for (int64_t c = num_cols - 1; c >= 0; --c) {
    const mlir::Value lhs_val = block->getArgument(2 * c);
    const mlir::Value rhs_val = block->getArgument(2 * c + 1);
    const ValCompareResult val_cmp =
        CompareColumnElements(b, loc, elem_type, lhs_val, rhs_val);
    res =
        mlir::stablehlo::SelectOp::create(b, loc, val_cmp.eq, res, val_cmp.lt);
  }
  mlir::stablehlo::ReturnOp::create(b, loc, res);
}

mlir::SmallVector<mlir::MlirOp> SortColumnsBatch(
    mlir::MlirBuilder& builder, mlir::MlirOp batch_2d, mlir::MlirOp perm,
    const int64_t s, const int64_t num_cols, const mlir::Type elem_type) {
  std::vector<mlir::MlirOp> sort_inputs;
  sort_inputs.reserve(num_cols + 2);
  if (num_cols == 1) {
    sort_inputs.push_back(mlir::stablehlo::Reshape(batch_2d, {s}));
  } else {
    for (int64_t c = 0; c < num_cols; ++c) {
      mlir::MlirOp col_2d =
          mlir::stablehlo::Slice(batch_2d, {0, c}, {s, c + 1}, {1, 1});
      sort_inputs.push_back(mlir::stablehlo::Reshape(col_2d, {s}));
    }
  }

  const auto i64_type = builder.getOpBuilder().getI64Type();
  const auto s_iota_type =
      mlir::makeTensorType(builder.getContext(), {s}, i64_type);
  mlir::MlirOp pos = mlir::stablehlo::Iota(builder, s_iota_type, 0);
  sort_inputs.push_back(pos);
  sort_inputs.push_back(perm);

  return mlir::stablehlo::Sort(
      builder, sort_inputs,
      [elem_type, num_cols](mlir::RegionBuilder& rb) {
        BuildMultiColumnComparator(elem_type, num_cols, rb);
      },
      0, /*is_stable=*/false);
}

mlir::MlirOp RunLexSortBatchStep(mlir::MlirBuilder& builder,
                                 mlir::MlirOp batch_2d, mlir::MlirOp perm,
                                 const int64_t s, const int64_t num_cols,
                                 const mlir::Type elem_type) {
  mlir::MlirOp batch_permuted = BuildIndexSelectShlo(batch_2d, 0, perm);
  return SortColumnsBatch(builder, batch_permuted, perm, s, num_cols, elem_type)
      .back();
}

std::vector<mlir::MlirOp> LexSortWhileBody(mlir::RegionBuilder& body,
                                           const int64_t s,
                                           const mlir::Type elem_type) {
  mlir::MlirBuilder body_builder(body.getOpBuilder(),
                                 body.getOpBuilder().getUnknownLoc());
  auto args =
      mlir::stablehlo::Arguments(body, body.getOp<mlir::stablehlo::WhileOp>());
  mlir::MlirOp j = args[0];
  mlir::MlirOp perm = args[1];
  mlir::MlirOp mat = args[2];

  const auto i64_type = body_builder.getOpBuilder().getI64Type();
  mlir::MlirOp zero_idx = MakeScalarConstant(body_builder, 0, i64_type);
  mlir::MlirOp batch_2d =
      mlir::stablehlo::DynamicSlice(mat, {zero_idx, j}, {s, kLexSortBatchSize});
  mlir::MlirOp next_perm = RunLexSortBatchStep(body_builder, batch_2d, perm, s,
                                               kLexSortBatchSize, elem_type);

  mlir::MlirOp step_idx =
      MakeScalarConstant(body_builder, kLexSortBatchSize, i64_type);
  mlir::MlirOp next_j = mlir::stablehlo::Subtract(j, step_idx);
  return {next_j, next_perm, mat};
}

struct LexSort2DOutputs {
  mlir::MlirOp sorted_2d;
  mlir::MlirOp p_indices;
};

LexSort2DOutputs BuildLexicographicalSort2DShlo(mlir::MlirBuilder& builder,
                                                mlir::MlirOp input_2d,
                                                const int64_t s,
                                                const int64_t slice_elements) {
  if (s == 1) {
    return LexSort2DOutputs{
        input_2d, MakeConstant(builder, 0, mlir::ElementType::I64, {1})};
  }

  const mlir::Type elem_type = GetTensorTypeOrDie(input_2d).getElementType();
  const auto i64_type = builder.getOpBuilder().getI64Type();
  const auto s_iota_type =
      mlir::makeTensorType(builder.getContext(), {s}, i64_type);
  mlir::MlirOp iota = mlir::stablehlo::Iota(builder, s_iota_type, 0);

  if (slice_elements == 1) {
    auto sorted = SortColumnsBatch(builder, input_2d, iota, s, 1, elem_type);
    return LexSort2DOutputs{mlir::stablehlo::Reshape(sorted[0], {s, 1}),
                            sorted.back()};
  }

  if (slice_elements <= kLexSortBatchSize) {
    mlir::MlirOp p_indices =
        SortColumnsBatch(builder, input_2d, iota, s, slice_elements, elem_type)
            .back();
    mlir::MlirOp sorted_2d = BuildIndexSelectShlo(input_2d, 0, p_indices);
    return LexSort2DOutputs{sorted_2d, p_indices};
  }

  const int64_t rem = slice_elements % kLexSortBatchSize;
  mlir::MlirOp init_j =
      MakeScalarConstant(builder, slice_elements - kLexSortBatchSize, i64_type);
  auto whl = mlir::stablehlo::While(
      builder, {init_j, iota, input_2d},
      [rem](mlir::RegionBuilder& cond) {
        mlir::MlirBuilder cond_builder(cond.getOpBuilder(),
                                       cond.getOpBuilder().getUnknownLoc());
        auto args = mlir::stablehlo::Arguments(
            cond, cond.getOp<mlir::stablehlo::WhileOp>());
        const auto cond_i64_type = cond_builder.getOpBuilder().getI64Type();
        mlir::MlirOp rem_cst =
            MakeScalarConstant(cond_builder, rem, cond_i64_type);
        mlir::stablehlo::Return(
            cond,
            mlir::stablehlo::Compare(args[0], rem_cst,
                                     mlir::stablehlo::ComparisonDirection::GE));
      },
      [s, elem_type](mlir::RegionBuilder& body) {
        mlir::stablehlo::Return(body, LexSortWhileBody(body, s, elem_type));
      });

  mlir::MlirOp p_indices = whl[1];
  if (rem > 0) {
    mlir::MlirOp rem_2d =
        mlir::stablehlo::Slice(input_2d, {0, 0}, {s, rem}, {1, 1});
    p_indices =
        RunLexSortBatchStep(builder, rem_2d, p_indices, s, rem, elem_type);
  }
  mlir::MlirOp sorted_2d = BuildIndexSelectShlo(input_2d, 0, p_indices);
  return LexSort2DOutputs{sorted_2d, p_indices};
}

absl::StatusOr<mlir::MlirOp> BuildUniqueDimMask(mlir::MlirBuilder& builder,
                                                mlir::MlirOp sorted_2d,
                                                const int64_t s,
                                                const int64_t slice_elements) {
  if (s == 1) {
    return MakeConstant(builder, 1, mlir::ElementType::PRED, {1});
  }

  mlir::MlirOp tail =
      mlir::stablehlo::Slice(sorted_2d, {1, 0}, {s, slice_elements}, {1, 1});
  mlir::MlirOp head = mlir::stablehlo::Slice(sorted_2d, {0, 0},
                                             {s - 1, slice_elements}, {1, 1});
  // Use IEEE NE (NaN != NaN) so slices containing NaN are not deduplicated.
  mlir::MlirOp diff_2d = mlir::stablehlo::Compare(
      tail, head, mlir::stablehlo::ComparisonDirection::NE);

  mlir::MlirOp init_or =
      MakeScalarConstant(builder, 0, builder.getOpBuilder().getI1Type());
  mlir::MlirOp diff_1d = mlir::stablehlo::Reduce(
      builder, {diff_2d}, {init_or},
      [](mlir::RegionBuilder& rb) {
        auto& body_builder = rb.getOpBuilder();
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::OrOp>(
            body_builder.getI1Type(), rb.getRegion(), body_builder);
      },
      {1})[0];

  mlir::MlirOp first_true =
      MakeConstant(builder, 1, mlir::ElementType::PRED, {1});
  return mlir::stablehlo::Concatenate(builder, {first_true, diff_1d}, 0);
}

mlir::MlirOp ExtractSortedUniqueIndices(mlir::MlirBuilder& builder,
                                        mlir::MlirOp mask,
                                        const int64_t total_elements,
                                        const int64_t output_size) {
  const auto i64_type = builder.getOpBuilder().getI64Type();
  const auto mask_sort_comparator = [i64_type](mlir::RegionBuilder& rb) {
    auto& body_builder = rb.getOpBuilder();
    mlir::stablehlo::buildSortComparisonBody(
        {body_builder.getI1Type(), i64_type},
        mlir::stablehlo::ComparisonDirection::GE, std::nullopt, &rb.getRegion(),
        &body_builder);
  };

  const auto s_iota_type =
      mlir::makeTensorType(builder.getContext(), {total_elements}, i64_type);
  mlir::MlirOp s_iota = mlir::stablehlo::Iota(builder, s_iota_type, 0);
  auto unique_indices_sort = mlir::stablehlo::Sort(
      builder, {mask, s_iota}, mask_sort_comparator, 0, /*is_stable=*/false);
  mlir::MlirOp s_unique_indices =
      mlir::stablehlo::Slice(unique_indices_sort[1], {0}, {output_size}, {1});

  const auto i64_lt_comparator = [i64_type](mlir::RegionBuilder& rb) {
    auto& body_builder = rb.getOpBuilder();
    mlir::stablehlo::buildSortComparisonBody(
        {i64_type}, mlir::stablehlo::ComparisonDirection::LT, std::nullopt,
        &rb.getRegion(), &body_builder);
  };
  return mlir::stablehlo::Sort(builder, {s_unique_indices}, i64_lt_comparator,
                               0, /*is_stable=*/false)[0];
}

absl::StatusOr<mlir::MlirOp> BuildUniqueDimInverseIndices(
    mlir::MlirBuilder& builder, mlir::MlirOp mask, mlir::MlirOp p_indices) {
  const auto i64_type = builder.getOpBuilder().getI64Type();
  mlir::MlirOp mask_i64 = mlir::stablehlo::ConvertElementType(mask, i64_type);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp cumsum_mask_full,
                      BuildCumsum1D(builder, mask_i64));
  mlir::MlirOp one_cst = MakeConstantLike(cumsum_mask_full, 1LL);
  mlir::MlirOp cumsum_mask =
      mlir::stablehlo::Subtract(cumsum_mask_full, one_cst);

  const auto i64_i64_lt_comparator = [i64_type](mlir::RegionBuilder& rb) {
    auto& body_builder = rb.getOpBuilder();
    mlir::stablehlo::buildSortComparisonBody(
        {i64_type, i64_type}, mlir::stablehlo::ComparisonDirection::LT,
        std::nullopt, &rb.getRegion(), &body_builder);
  };
  return mlir::stablehlo::Sort(builder, {p_indices, cumsum_mask},
                               i64_i64_lt_comparator, 0,
                               /*is_stable=*/false)[1];
}

mlir::MlirOp BuildUniqueDimCounts(mlir::MlirBuilder& builder,
                                  mlir::MlirOp s_unique_indices,
                                  const int64_t s, const int64_t output_size) {
  mlir::MlirOp s_cst = MakeConstant(builder, s, mlir::ElementType::I64, {1});
  mlir::MlirOp unique_indices_plus_s =
      mlir::stablehlo::Concatenate(builder, {s_unique_indices, s_cst}, 0);
  mlir::MlirOp c_start = mlir::stablehlo::Slice(unique_indices_plus_s, {1},
                                                {output_size + 1}, {1});
  mlir::MlirOp c_end =
      mlir::stablehlo::Slice(unique_indices_plus_s, {0}, {output_size}, {1});
  return mlir::stablehlo::Subtract(c_start, c_end);
}

absl::StatusOr<BuildUniqueShloOutputs> BuildAllUniqueFinalizeShlo(
    mlir::MlirBuilder& builder, mlir::MlirOp input, mlir::MlirOp p_indices,
    const int64_t dim, const int64_t num_slices,
    const ReturnInverse return_inverse, const ReturnCounts return_counts) {
  BuildUniqueShloOutputs outputs;
  outputs.unique_values = BuildIndexSelectShlo(input, dim, p_indices);

  if (return_inverse == ReturnInverse::kTrue) {
    const auto i64_type = builder.getOpBuilder().getI64Type();
    const auto s_iota_type =
        mlir::makeTensorType(builder.getContext(), {num_slices}, i64_type);
    mlir::MlirOp iota = mlir::stablehlo::Iota(builder, s_iota_type, 0);
    const auto i64_i64_lt_comparator = [i64_type](mlir::RegionBuilder& rb) {
      auto& body_builder = rb.getOpBuilder();
      mlir::stablehlo::buildSortComparisonBody(
          {i64_type, i64_type}, mlir::stablehlo::ComparisonDirection::LT,
          std::nullopt, &rb.getRegion(), &body_builder);
    };
    outputs.inverse_indices =
        mlir::stablehlo::Sort(builder, {p_indices, iota}, i64_i64_lt_comparator,
                              0, /*is_stable=*/false)[1];
  } else {
    TT_ASSIGN_OR_RETURN(outputs.inverse_indices,
                        MakeZeroSizedTensor(builder, mlir::ElementType::I64));
  }

  if (return_counts == ReturnCounts::kTrue) {
    outputs.counts =
        MakeConstant(builder, 1, mlir::ElementType::I64, {num_slices});
  } else {
    TT_ASSIGN_OR_RETURN(outputs.counts,
                        MakeZeroSizedTensor(builder, mlir::ElementType::I64));
  }

  return outputs;
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildUniqueGetOutputSizeShlo(
    mlir::ElementType element_type, mlir::MlirOp input, bool sorted) {
  auto& builder = input.getBuilder();
  mlir::MlirOp flattened_input = FlattenIfNeeded(input);
  mlir::MlirOp sorted_input =
      BuildSortShlo(flattened_input, /*stable=*/false, 0, /*descending=*/false)
          .values;

  TT_ASSIGN_OR_RETURN(mlir::MlirOp mask,
                      BuildUniqueMask(builder, sorted_input));
  mlir::MlirOp mask_i64 = mlir::stablehlo::ConvertElementType(
      mask, builder.getOpBuilder().getI64Type());
  auto init_value =
      MakeScalarConstant(builder, 0, builder.getOpBuilder().getI64Type());
  return mlir::stablehlo::Reduce(
      builder, {mask_i64}, {init_value},
      [](mlir::RegionBuilder& rb) {
        auto& body_builder = rb.getOpBuilder();
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
            body_builder.getI64Type(), rb.getRegion(), body_builder);
      },
      {0})[0];
}

absl::StatusOr<BuildUniqueShloOutputs> BuildUnique2Shlo(int64_t output_size,
                                                        mlir::MlirOp input,
                                                        bool sorted,
                                                        bool return_inverse,
                                                        bool return_counts) {
  BuildUniqueShloOutputs outputs;
  auto& builder = input.getBuilder();
  auto& op_builder = builder.getOpBuilder();
  mlir::MlirOp flattened_input = FlattenIfNeeded(input);
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(flattened_input);
  const int64_t n = input_type.getDimSize(0);
  const mlir::ElementType val_type = GetElementTypeOrDie(flattened_input);

  auto i64_type = op_builder.getI64Type();

  if (n == 0) {
    TT_ASSIGN_OR_RETURN(outputs.unique_values,
                        MakeZeroSizedTensor(builder, val_type));
    TT_ASSIGN_OR_RETURN(outputs.counts,
                        MakeZeroSizedTensor(builder, mlir::ElementType::I64));

    if (return_inverse) {
      // When return_inverse is true, PyTorch expects empty inverse_indices
      // shape to match the original input tensor's shape even when the input
      // size is zero (e.g. dynamic shape with a 0-sized dimension).
      TT_ASSIGN_OR_RETURN(
          outputs.inverse_indices,
          MakeZeroSizedTensor(builder, mlir::ElementType::I64,
                              GetTensorTypeOrDie(input).getShape()));
    } else {
      // When return_inverse is false, the inverse_indices shape is (0,).
      TT_ASSIGN_OR_RETURN(outputs.inverse_indices,
                          MakeZeroSizedTensor(builder, mlir::ElementType::I64));
    }
    return outputs;
  }

  // 1. Sort input and get permutation indices
  SortShloOutputs sorted_res =
      BuildSortShlo(flattened_input, /*stable=*/false, 0, /*descending=*/false);
  mlir::MlirOp sorted_input = sorted_res.values;
  mlir::MlirOp p_indices =
      mlir::stablehlo::ConvertElementType(sorted_res.indices, i64_type);

  TT_ASSIGN_OR_RETURN(mlir::MlirOp mask,
                      BuildUniqueMask(builder, sorted_input));

  // 2. Compute unique values
  mlir::MlirOp s_unique_indices =
      ExtractSortedUniqueIndices(builder, mask, n, output_size);

  auto gather_dims_attr = mlir::stablehlo::GatherDimensionNumbersAttr::get(
      &builder.getContext(),
      /*offset_dims=*/{},
      /*collapsed_slice_dims=*/{0},
      /*operand_batching_dims=*/{},
      /*start_indices_batching_dims=*/{},
      /*start_index_map=*/{0},
      /*index_vector_dim=*/1);
  mlir::MlirOp s_unique_indices_reshaped =
      mlir::stablehlo::Reshape(s_unique_indices, {output_size, 1});
  outputs.unique_values = mlir::stablehlo::Gather(
      sorted_input, s_unique_indices_reshaped, gather_dims_attr, {1});

  // 3. Inverse indices
  if (return_inverse) {
    TT_ASSIGN_OR_RETURN(outputs.inverse_indices,
                        BuildUniqueDimInverseIndices(builder, mask, p_indices));
    const mlir::RankedTensorType original_input_type =
        GetTensorTypeOrDie(input);
    if (original_input_type.getRank() != 1) {
      outputs.inverse_indices = mlir::stablehlo::Reshape(
          outputs.inverse_indices, original_input_type.getShape());
    }
  } else {
    TT_ASSIGN_OR_RETURN(outputs.inverse_indices,
                        MakeZeroSizedTensor(builder, i64_type));
  }

  // 4. Counts
  if (return_counts) {
    outputs.counts =
        BuildUniqueDimCounts(builder, s_unique_indices, n, output_size);
  } else {
    TT_ASSIGN_OR_RETURN(outputs.counts, MakeZeroSizedTensor(builder, i64_type));
  }

  return outputs;
}

absl::StatusOr<BuildUniqueDimPrepareOutputs> BuildUniqueDimPrepareShlo(
    mlir::MlirOp input, const int64_t dim) {
  auto& builder = input.getBuilder();
  const FlattenedSlices2D slices = TransposeAndFlattenSlices(input, dim);
  const LexSort2DOutputs sorted = BuildLexicographicalSort2DShlo(
      builder, slices.input_2d, slices.num_slices, slices.slice_elements);

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp mask,
      BuildUniqueDimMask(builder, sorted.sorted_2d, slices.num_slices,
                         slices.slice_elements));
  mlir::MlirOp mask_i64 = mlir::stablehlo::ConvertElementType(
      mask, builder.getOpBuilder().getI64Type());
  mlir::MlirOp init_value =
      MakeScalarConstant(builder, 0, builder.getOpBuilder().getI64Type());
  mlir::MlirOp count = mlir::stablehlo::Reduce(
      builder, {mask_i64}, {init_value},
      [](mlir::RegionBuilder& rb) {
        auto& body_builder = rb.getOpBuilder();
        mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
            body_builder.getI64Type(), rb.getRegion(), body_builder);
      },
      {0})[0];
  return BuildUniqueDimPrepareOutputs{count, sorted.p_indices, mask};
}

absl::StatusOr<BuildUniqueShloOutputs> BuildUniqueDimFinalizeShlo(
    const int64_t output_size, mlir::MlirOp input, mlir::MlirOp p_indices,
    mlir::MlirOp mask, const int64_t dim, const ReturnInverse return_inverse,
    const ReturnCounts return_counts) {
  auto& builder = input.getBuilder();
  const int64_t num_slices = GetTensorTypeOrDie(input).getDimSize(dim);
  if (output_size == num_slices) {
    return BuildAllUniqueFinalizeShlo(builder, input, p_indices, dim,
                                      num_slices, return_inverse,
                                      return_counts);
  }

  BuildUniqueShloOutputs outputs;
  mlir::MlirOp s_unique_indices =
      ExtractSortedUniqueIndices(builder, mask, num_slices, output_size);
  mlir::MlirOp orig_unique_indices =
      BuildIndexSelectShlo(p_indices, 0, s_unique_indices);
  outputs.unique_values = BuildIndexSelectShlo(input, dim, orig_unique_indices);

  if (return_inverse == ReturnInverse::kTrue) {
    TT_ASSIGN_OR_RETURN(outputs.inverse_indices,
                        BuildUniqueDimInverseIndices(builder, mask, p_indices));
  } else {
    TT_ASSIGN_OR_RETURN(outputs.inverse_indices,
                        MakeZeroSizedTensor(builder, mlir::ElementType::I64));
  }

  if (return_counts == ReturnCounts::kTrue) {
    outputs.counts = BuildUniqueDimCounts(builder, s_unique_indices, num_slices,
                                          output_size);
  } else {
    TT_ASSIGN_OR_RETURN(outputs.counts,
                        MakeZeroSizedTensor(builder, mlir::ElementType::I64));
  }

  return outputs;
}

}  // namespace torch_tpu
