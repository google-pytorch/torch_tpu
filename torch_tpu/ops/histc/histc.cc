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

#include "torch_tpu/ops/histc/histc.h"

#include <cstdint>
#include <numeric>

#include "absl/status/statusor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/ops/op_builder_utils.h"

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

  auto reduce_builder = [&](mlir::RegionBuilder& rb) {
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

// Broadcasts the input tensor to a given shape using BroadcastInDim.
mlir::MlirOp BuildBroadcastToShape(mlir::MlirOp input_op,
                                   llvm::ArrayRef<int64_t> shape) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  const auto output_type = input_type.clone(shape);
  const int64_t input_rank = input_type.getRank();
  const int64_t output_rank = shape.size();
  Dimensions broadcast_dims(input_rank);
  std::iota(broadcast_dims.begin(), broadcast_dims.end(),
            output_rank - input_rank);
  return mlir::stablehlo::BroadcastInDim(output_type, input_op, broadcast_dims);
}

// Computes bin indices for each element in input_op.
mlir::MlirOp BuildIndices(mlir::MlirOp input_op, mlir::MlirOp min_op,
                          mlir::MlirOp max_op, int64_t bins,
                          mlir::MlirBuilder& builder) {
  // floor((input_op - min) * bins / (max - min))
  mlir::MlirOp numerator = stablehlo::Subtract(input_op, min_op);
  mlir::MlirOp denominator = stablehlo::Subtract(max_op, min_op);
  mlir::RankedTensorType input_op_type = GetTensorTypeOrDie(input_op);
  mlir::MlirOp bins_op = MakeConstant(builder, bins, input_op_type);
  mlir::MlirOp mul_bins = stablehlo::Mul(numerator, bins_op);
  mlir::MlirOp div = stablehlo::Div(mul_bins, denominator);
  if (input_op_type.getElementType().isInteger()) {
    return div;
  }
  mlir::MlirOp indices = stablehlo::Floor(div);
  return stablehlo::ConvertElementType(indices, mlir::ElementType::I32);
}

// Clamps the indices to the range [0, bins - 1].
mlir::MlirOp BuildClampedIndices(mlir::MlirOp indices, int64_t bins,
                                 mlir::MlirBuilder& builder) {
  // clamped_indices = clamp(indices, 0, bins - 1)
  const mlir::RankedTensorType indices_type = GetTensorTypeOrDie(indices);
  mlir::MlirOp zero_indices = MakeConstant(builder, 0.0, indices_type);
  mlir::MlirOp bins_minus_1_indices =
      MakeConstant(builder, bins - 1, indices_type);
  return stablehlo::Clamp(zero_indices, indices, bins_minus_1_indices);
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
  const mlir::RankedTensorType clamped_indices_type =
      GetTensorTypeOrDie(clamped_indices);
  int64_t num_inputs = 1;
  for (auto dim_size : clamped_indices_type.getShape()) {
    num_inputs *= dim_size;
  }
  // Flatten indices and in_range tensors to 1D to be used in scatter.
  auto clamped_indices_flat = stablehlo::Reshape(clamped_indices, {num_inputs});
  auto in_range_flat = stablehlo::Reshape(in_range, {num_inputs});

  // The output histogram tensor of size `bins` is initialized to zeros.
  Dimensions output_dims = {bins};
  auto empty_output = MakeConstant(builder, 0, out_dtype, output_dims);
  // Scatter indices must be of shape {num_updates, 1} to update a 1D tensor.
  auto scatter_indices =
      stablehlo::Reshape(clamped_indices_flat, {num_inputs, 1});

  // Contribution to histogram is 1 if element is in range, 0 otherwise.
  mlir::MlirOp one = MakeConstant(builder, 1, out_dtype, {num_inputs});
  mlir::MlirOp zero = MakeConstant(builder, 0, out_dtype, {num_inputs});
  mlir::MlirOp scatter_update = stablehlo::Select(in_range_flat, one, zero);

  // Create a region builder callback for scatter-add.
  // The scatter operation will compute:
  //   output[scatter_indices[i]] += scatter_update[i]
  auto scatter_builder = [&](mlir::RegionBuilder& builder) {
    stablehlo::buildReduceBody<stablehlo::AddOp>(out_dtype, builder.getRegion(),
                                                 builder.getOpBuilder());
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

  // Broadcast min and max to the shape of the input tensor.
  min_op =
      BuildBroadcastToShape(min_op, GetTensorTypeOrDie(input_op).getShape());
  max_op =
      BuildBroadcastToShape(max_op, GetTensorTypeOrDie(input_op).getShape());

  // Compute bin indices:
  //   indices = floor((input - min) * bins / (max - min))
  mlir::MlirOp indices = BuildIndices(input_op, min_op, max_op, bins, builder);
  // Then clamp indices to [0, bins - 1].
  mlir::MlirOp clamped_indices = BuildClampedIndices(indices, bins, builder);

  // Values outside of [min_op, max_op] are ignored, which mirrors the PyTorch
  // implementation.
  mlir::MlirOp in_range = BuildInRange(input_op, min_op, max_op);
  // Scatter-add to accumulate counts.
  mlir::MlirOp hist =
      BuildScatterAdd(clamped_indices, in_range, bins, input_dtype, builder);

  return hist;
}

}  // namespace torch_tpu
