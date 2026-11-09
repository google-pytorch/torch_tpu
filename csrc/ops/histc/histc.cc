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

#include "csrc/ops/histc/histc.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {

// Builds a reduction over all dimensions of the input tensor, resulting in a
// rank 0 tensor with a single element.
template <typename ReduceOp>
mlir::MlirOp BuildReduce(mlir::MlirOp input_op, mlir::Attribute init_attr) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);

  mlir::DenseElementsAttr value_init_attr = mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get({}, input_type.getElementType()), init_attr);
  mlir::MlirOp value_init = stablehlo::Constant(builder, value_init_attr);

  auto reduce_builder =
      [&](mlir::RegionBuilder& rb) {  // REF_CAPTURE_OK=called synchronously
        stablehlo::buildReduceBody<ReduceOp>(input_type.getElementType(),
                                             rb.getRegion(), rb.getOpBuilder());
      };
  auto result = stablehlo::Reduce(builder, {input_op}, {value_init},
                                  reduce_builder, GetAllDimensions(input_op));
  // Reducing over all dimensions yields a single result.
  return result[0];
}

// Builds a reduction to compute the minimum value over all elements of
// input_op.
mlir::MlirOp BuildReduceMinShlo(mlir::MlirOp input_op) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  mlir::Attribute init_attr = GetMaxFiniteValueAttr(input_type.getElementType(),
                                                    builder.getOpBuilder());
  return BuildReduce<stablehlo::MinOp>(input_op, init_attr);
}

// Builds a reduction to compute the maximum value over all elements of
// input_op.
mlir::MlirOp BuildReduceMaxShlo(mlir::MlirOp input_op) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  mlir::Attribute init_attr = GetMinFiniteValueAttr(input_type.getElementType(),
                                                    builder.getOpBuilder());
  return BuildReduce<stablehlo::MaxOp>(input_op, init_attr);
}

// Adjusts the min and max to account for the case where min == max.
// If min == max, set min = min - 1 and max = max + 1.
void AdjustMinMaxForEq(mlir::MlirOp& min_op, mlir::MlirOp& max_op) {
  mlir::Type dtype = GetTensorTypeOrDie(min_op).getElementType();

  mlir::MlirBuilder& builder = min_op.getBuilder();
  mlir::MlirOp one = MakeScalarConstant(builder, 1, dtype);
  mlir::MlirOp min_op_minus_one = stablehlo::Subtract(min_op, one);
  mlir::MlirOp max_op_plus_one = stablehlo::Add(max_op, one);

  mlir::MlirOp cond_op =
      stablehlo::Compare(min_op, max_op, stablehlo::ComparisonDirection::EQ);
  min_op = stablehlo::Select(cond_op, min_op_minus_one, min_op);
  max_op = stablehlo::Select(cond_op, max_op_plus_one, max_op);
}

// Computes bin indices for each element in input_op.
mlir::MlirOp BuildIndices(mlir::MlirOp input_op, mlir::MlirOp min_op,
                          mlir::MlirOp max_op, int64_t bins) {
  // floor((input_op - min) * bins / (max - min))
  mlir::MlirOp numerator = stablehlo::Subtract(input_op, min_op);
  mlir::MlirOp denominator = stablehlo::Subtract(max_op, min_op);
  mlir::MlirOp bins_op = MakeConstantLike(input_op, bins);
  mlir::MlirOp mul_bins = stablehlo::Mul(numerator, bins_op);
  mlir::MlirOp div = stablehlo::Div(mul_bins, denominator);
  if (GetTensorTypeOrDie(input_op).getElementType().isInteger()) {
    return stablehlo::ConvertElementType(div, mlir::ElementType::I32);
  }
  mlir::MlirOp indices = stablehlo::Floor(div);
  return stablehlo::ConvertElementType(indices, mlir::ElementType::I32);
}

// Clamps the indices to the range [0, bins - 1].
mlir::MlirOp BuildClampedIndices(mlir::MlirOp indices, int64_t bins) {
  // clamped_indices = clamp(indices, 0, bins - 1)
  // Use Min and Max instead of Clamp because stablehlo::Clamp drops dynamic
  // bounds.
  mlir::MlirOp zero_indices = MakeConstantLike(indices, 0);
  mlir::MlirOp bins_minus_1_indices = MakeConstantLike(indices, bins - 1);
  mlir::MlirOp min_indices = stablehlo::Min(indices, bins_minus_1_indices);
  return stablehlo::Max(min_indices, zero_indices);
}

// Returns a boolean tensor that is true for elements in [min_op, max_op].
mlir::MlirOp BuildInRange(mlir::MlirOp input_op, mlir::MlirOp min_op,
                          mlir::MlirOp max_op) {
  // in_range = (input >= min) & (input <= max)
  mlir::MlirOp ge_min =
      stablehlo::Compare(input_op, min_op, stablehlo::ComparisonDirection::GE);
  mlir::MlirOp le_max =
      stablehlo::Compare(input_op, max_op, stablehlo::ComparisonDirection::LE);
  return stablehlo::And(ge_min, le_max);
}

// Builds a scatter-add operation to accumulate histogram counts.
// Counts are only incremented for elements where in_range is true.
// This implementation was adapted from:
// torch_tpu/ops/bincount/bincount_aten_kernels.cc
mlir::MlirOp BuildScatterAdd(mlir::MlirOp clamped_indices,
                             mlir::MlirOp in_range, int64_t bins,
                             mlir::Type out_dtype, mlir::MlirBuilder& builder) {
  // Flatten indices and in_range tensors to 1D to be used in scatter.
  mlir::MlirOp clamped_indices_flat = clamped_indices;
  mlir::MlirOp in_range_flat = in_range;
  if (GetTensorTypeOrDie(clamped_indices).getRank() != 1) {
    clamped_indices_flat = Flatten(clamped_indices);
    in_range_flat = Flatten(in_range);
  }

  // The output histogram tensor of size `bins` is initialized to zeros.
  Dimensions output_dims = {bins};
  mlir::MlirOp empty_output = MakeConstant(builder, 0, out_dtype, output_dims);
  // Scatter indices can be 1D when index_vector_dim = 1.
  mlir::MlirOp scatter_indices = clamped_indices_flat;

  // Contribution to histogram is 1 if element is in range, 0 otherwise.
  mlir::MlirOp scatter_update =
      stablehlo::ConvertElementType(in_range_flat, out_dtype);

  // Create a region builder callback for scatter-add.
  // The scatter operation will compute:
  //   output[scatter_indices[i]] += scatter_update[i]
  auto scatter_builder =
      [&](mlir::RegionBuilder& builder) {  // REF_CAPTURE_OK=called locally
        stablehlo::buildReduceBody<stablehlo::AddOp>(
            out_dtype, builder.getRegion(), builder.getOpBuilder());
      };

  auto scatter_dimension_numbers =
      mlir::stablehlo::ScatterDimensionNumbersAttr::get(
          &builder.getContext(),
          /*updateWindowDims=*/{},
          /*inserted_window_dims=*/{0},
          /*input_batching_dims=*/{},
          /*scatter_indices_batching_dims=*/{},
          /*scatter_dims_to_operand_dims=*/{0},
          /*index_vector_dim=*/1);

  mlir::MlirOp hist =
      mlir::stablehlo::Scatter(empty_output, scatter_indices, scatter_update,
                               scatter_builder, scatter_dimension_numbers)[0];
  return hist;
}

}  // namespace

absl::StatusOr<HistcBounds> BuildHistcBoundsShlo(mlir::MlirOp input_op) {
  // Below code mirrors the PyTorch CUDA implementation:

  // TODO(b/474571547): Replace with PyTorch min/max operation once nans are
  // propagated.
  mlir::MlirOp min_op = BuildReduceMinShlo(input_op);
  mlir::MlirOp max_op = BuildReduceMaxShlo(input_op);

  // Account for the case where min == max by adjusting the min and max.
  AdjustMinMaxForEq(min_op, max_op);

  return HistcBounds{.min = min_op, .max = max_op};
}

absl::StatusOr<mlir::MlirOp> BuildHistcShlo(mlir::MlirOp input_op, int64_t bins,
                                            mlir::MlirOp min_op,
                                            mlir::MlirOp max_op) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  mlir::Type input_dtype = GetTensorTypeOrDie(input_op).getElementType();
  mlir::MlirOp compute_input_op = input_op;
  if (input_dtype.isInteger()) {
    compute_input_op =
        stablehlo::ConvertElementType(input_op, mlir::ElementType::I64);
    min_op = stablehlo::ConvertElementType(min_op, mlir::ElementType::I64);
    max_op = stablehlo::ConvertElementType(max_op, mlir::ElementType::I64);
  }

  // Broadcast min and max to the shape of the input tensor.
  TT_ASSIGN_OR_RETURN(min_op, BroadcastIfNeeded(min_op, compute_input_op));
  TT_ASSIGN_OR_RETURN(max_op, BroadcastIfNeeded(max_op, compute_input_op));

  // Compute bin indices:
  //   indices = floor((input - min) * bins / (max - min))
  mlir::MlirOp indices = BuildIndices(compute_input_op, min_op, max_op, bins);
  // Then clamp indices to [0, bins - 1].
  mlir::MlirOp clamped_indices = BuildClampedIndices(indices, bins);

  // Values outside of [min_op, max_op] are ignored, which mirrors the PyTorch
  // implementation.
  mlir::MlirOp in_range = BuildInRange(compute_input_op, min_op, max_op);

  // COST_MODEL: For static bins <= 256 and bounded intermediate
  // (num_elements * bins <= 2^26, <= 256MB for 32-bit types), dense comparison
  // + reduction avoids TPU scatter serialization and contention.
  constexpr int64_t kMaxDenseBins = 256;
  constexpr int64_t kMaxDenseElements = 1LL << 26;
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  if (bins <= kMaxDenseBins && input_type.hasStaticShape() &&
      input_type.getNumElements() * bins <= kMaxDenseElements) {
    const int64_t num_elements = input_type.getNumElements();
    if (num_elements == 0) {
      return MakeConstant(builder, 0, input_dtype, {bins});
    }

    mlir::MlirOp clamped_indices_flat = clamped_indices;
    mlir::MlirOp in_range_flat = in_range;
    if (GetTensorTypeOrDie(clamped_indices).getRank() != 1) {
      clamped_indices_flat = Flatten(clamped_indices);
      in_range_flat = Flatten(in_range);
    }

    mlir::MlirOp bucket_idx_2d =
        stablehlo::Reshape(clamped_indices_flat, {num_elements, 1});
    mlir::MlirOp in_range_2d =
        stablehlo::Reshape(in_range_flat, {num_elements, 1});

    mlir::MlirOp iota_bins = stablehlo::Iota(
        builder,
        makeTensorType(builder.getContext(), {1, bins},
                       GetTensorTypeOrDie(bucket_idx_2d).getElementType()),
        /*iota_dimension=*/1);

    TT_ASSIGN_OR_RETURN(auto broadcasted_indices,
                        ApplyBroadcastIfNeeded(bucket_idx_2d, iota_bins));
    mlir::MlirOp is_bin_match =
        stablehlo::Compare(broadcasted_indices[0], broadcasted_indices[1],
                           stablehlo::ComparisonDirection::EQ);

    TT_ASSIGN_OR_RETURN(mlir::MlirOp in_range_broadcasted,
                        BroadcastIfNeeded(in_range_2d, is_bin_match));
    mlir::MlirOp valid_match =
        stablehlo::And(is_bin_match, in_range_broadcasted);

    mlir::MlirOp one =
        MakeConstant(builder, 1, input_dtype, {num_elements, bins});
    mlir::MlirOp zero =
        MakeConstant(builder, 0, input_dtype, {num_elements, bins});
    mlir::MlirOp counts_matrix = stablehlo::Select(valid_match, one, zero);

    mlir::DenseElementsAttr zero_attr = mlir::DenseElementsAttr::get(
        mlir::RankedTensorType::get({}, input_dtype),
        builder.getOpBuilder().getZeroAttr(input_dtype));
    mlir::MlirOp init_val = stablehlo::Constant(builder, zero_attr);
    auto reduce_builder = [input_dtype](mlir::RegionBuilder& rb) {
      stablehlo::buildReduceBody<stablehlo::AddOp>(input_dtype, rb.getRegion(),
                                                   rb.getOpBuilder());
    };
    auto result = stablehlo::Reduce(builder, {counts_matrix}, {init_val},
                                    reduce_builder, {0});
    return result[0];
  }

  // Scatter-add to accumulate counts.
  return BuildScatterAdd(clamped_indices, in_range, bins, input_dtype, builder);
}

}  // namespace torch_tpu
