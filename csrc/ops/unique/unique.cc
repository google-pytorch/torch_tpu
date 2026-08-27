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

#include "torch_tpu/csrc/ops/unique/unique.h"

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/sort/sort.h"

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
  auto mask_sort_comparator = [i64_type](mlir::RegionBuilder& rb) {
    auto& body_builder = rb.getOpBuilder();
    mlir::stablehlo::buildSortComparisonBody(
        {body_builder.getI1Type(), i64_type},
        mlir::stablehlo::ComparisonDirection::GE, std::nullopt, &rb.getRegion(),
        &body_builder);
  };

  auto s_iota_type = mlir::makeTensorType(builder.getContext(), {n}, i64_type);
  auto s_iota = mlir::stablehlo::Iota(builder, s_iota_type, 0);
  auto unique_indices_sort = mlir::stablehlo::Sort(
      builder, {mask, s_iota}, mask_sort_comparator, 0, /*is_stable=*/false);
  mlir::MlirOp s_unique_indices =
      mlir::stablehlo::Slice(unique_indices_sort[1], {0}, {output_size}, {1});

  auto i64_lt_comparator = [i64_type](mlir::RegionBuilder& rb) {
    auto& body_builder = rb.getOpBuilder();
    mlir::stablehlo::buildSortComparisonBody(
        {i64_type}, mlir::stablehlo::ComparisonDirection::LT, std::nullopt,
        &rb.getRegion(), &body_builder);
  };
  s_unique_indices =
      mlir::stablehlo::Sort(builder, {s_unique_indices}, i64_lt_comparator, 0,
                            /*is_stable=*/false)[0];

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
    mlir::MlirOp mask_i64 = mlir::stablehlo::ConvertElementType(mask, i64_type);
    TT_ASSIGN_OR_RETURN(auto cumsum_mask_full,
                        BuildCumsum1D(builder, mask_i64));
    mlir::MlirOp one_cst = MakeConstantLike(cumsum_mask_full, 1LL);
    mlir::MlirOp cumsum_mask =
        mlir::stablehlo::Subtract(cumsum_mask_full, one_cst);

    auto i64_i64_lt_comparator = [i64_type](mlir::RegionBuilder& rb) {
      auto& body_builder = rb.getOpBuilder();
      mlir::stablehlo::buildSortComparisonBody(
          {i64_type, i64_type}, mlir::stablehlo::ComparisonDirection::LT,
          std::nullopt, &rb.getRegion(), &body_builder);
    };
    auto unsorted =
        mlir::stablehlo::Sort(builder, {p_indices, cumsum_mask},
                              i64_i64_lt_comparator, 0, /*is_stable=*/false);
    outputs.inverse_indices = unsorted[1];
    const mlir::RankedTensorType original_input_type =
        GetTensorTypeOrDie(input);
    if (original_input_type.getRank() > 1) {
      outputs.inverse_indices = mlir::stablehlo::Reshape(
          outputs.inverse_indices, original_input_type.getShape());
    }
  } else {
    TT_ASSIGN_OR_RETURN(outputs.inverse_indices,
                        MakeZeroSizedTensor(builder, i64_type));
  }

  // 4. Counts
  if (return_counts) {
    mlir::MlirOp n_cst = MakeConstant(builder, n, mlir::ElementType::I64, {1});
    mlir::MlirOp unique_indices_plus_n =
        mlir::stablehlo::Concatenate(builder, {s_unique_indices, n_cst}, 0);
    mlir::MlirOp c_start = mlir::stablehlo::Slice(unique_indices_plus_n, {1},
                                                  {output_size + 1}, {1});
    mlir::MlirOp c_end =
        mlir::stablehlo::Slice(unique_indices_plus_n, {0}, {output_size}, {1});
    outputs.counts = mlir::stablehlo::Subtract(c_start, c_end);
  } else {
    TT_ASSIGN_OR_RETURN(outputs.counts, MakeZeroSizedTensor(builder, i64_type));
  }

  return outputs;
}

}  // namespace torch_tpu
